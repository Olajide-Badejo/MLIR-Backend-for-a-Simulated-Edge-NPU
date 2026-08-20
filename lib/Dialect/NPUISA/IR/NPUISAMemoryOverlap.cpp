//===- NPUISAMemoryOverlap.cpp - The overlap test ---------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Whether two memrefs touch the same bytes.
//
// The whole reason this file exists rather than a two line `lhs == rhs` in the
// verifier is Section 8's sentence about it: once buffers are views, a
// partially overlapping view is a real race that an identity check does not
// see. The allocator of Section 13.1 materialises every scratchpad offset as a
// `memref.view` over one flat `memref<Nxi8, #npu.scratchpad>`, so after
// allocation almost every pair of distinct buffers in a function is a pair of
// distinct SSA values over one shared allocation, and identity is exactly the
// wrong question to ask about them.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::npuisa;

namespace {

/// The byte width of an element type, or nothing when it has none this project
/// can rely on.
///
/// `getIntOrFloatBitWidth` is in bits and a memref of `i1` would be eight of
/// them to a byte, so the conversion is a division that has to be exact. This
/// dialect only ever carries `f32` and `i8`, both of which divide cleanly, and
/// anything else returns nothing rather than rounding: a byte range computed
/// from a rounded element size is a byte range that is wrong by less than a
/// byte, which is the most dangerous kind of wrong because it looks right.
std::optional<int64_t> byteWidth(Type elementType) {
  if (!elementType.isIntOrFloat())
    return std::nullopt;
  const unsigned bits = elementType.getIntOrFloatBitWidth();
  if (bits == 0 || bits % 8 != 0)
    return std::nullopt;
  return static_cast<int64_t>(bits / 8);
}

/// The byte span a memref denotes, measured from its own first byte.
///
/// **This is computed from the strides, not from the product of the extents,
/// and the difference is the whole of the stride 0 question.** A memref with an
/// identity layout spans the product of its extents, and that is the common
/// case and the fast path below. A memref with a strided layout spans one past
/// the largest linear index any of its indices can reach, which is
/// `1 + sum((extent - 1) * stride)` elements.
///
/// The two agree for a permutation layout, which is what an NHWC buffer gets:
/// the same contiguous block, read in a different order. They disagree for the
/// rank 1 channel broadcast of ADR 0005, `sizes [1, 8, 4, 4]` over
/// `strides [0, 1, 0, 0]`, where the extents multiply to 128 elements and the
/// strides reach 8. Eight is the true answer: that view addresses C floats and
/// spans exactly those, which is what the 512 byte tensor it is *shaped* like
/// never touches.
///
/// Getting this from the extents would not be unsafe in the overlap rule's
/// direction, since a range that is too large only ever reports more overlaps
/// than there are. It would be unsafe in the *usable* direction: every
/// broadcast operand would appear to collide with whatever the allocator packed
/// next to it, and a legal asynchronous program would be refused for a race it
/// does not have.
///
/// The span is a closed hull rather than an exact set. A view with a stride
/// larger than its inner extents addresses some of the bytes in that range and
/// not others, and this reports the whole interval. That direction is the safe
/// one, and it is the only approximation in this file.
///
/// The layout's own offset is deliberately not read here. Offsets are
/// accumulated by the walk in `computeBufferRange`, from the operations that
/// introduce them, and reading the type's offset as well would count a subview
/// or a cast twice.
std::optional<int64_t> byteSpan(MemRefType type) {
  if (!type.hasStaticShape())
    return std::nullopt;
  std::optional<int64_t> width = byteWidth(type.getElementType());
  if (!width)
    return std::nullopt;

  int64_t elements = 1;
  for (int64_t extent : type.getShape()) {
    if (extent < 0)
      return std::nullopt;
    elements *= extent;
  }
  if (elements == 0)
    return 0;
  if (type.getLayout().isIdentity())
    return elements * *width;

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  if (strides.size() != type.getShape().size())
    return std::nullopt;

  int64_t last = 0;
  for (auto [extent, stride] : llvm::zip_equal(type.getShape(), strides)) {
    if (ShapedType::isDynamic(stride) || stride < 0)
      return std::nullopt;
    last += (extent - 1) * stride;
  }
  return (last + 1) * *width;
}

/// The constant value an index-like SSA value holds, or nothing.
///
/// A non constant offset is the case Section 8 names explicitly: if the offsets
/// are not static, refuse the async form with a diagnostic rather than assuming
/// disjointness. This returning nothing is what produces that refusal.
std::optional<int64_t> constantIndex(Value value) {
  IntegerAttr attr;
  if (matchPattern(value, m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

/// Why a chain stopped being analysable. Kept as a string rather than an enum
/// because it is only ever printed, and every producer of one is right here.
StringRef notAnalysableReason(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return "it is a block argument, so its address is not known here";

  if (isa<memref::ViewOp>(def))
    return "its byte offset within the underlying buffer is not a constant";
  if (isa<memref::SubViewOp>(def))
    return "its subview offsets, sizes or strides are not all static, or its "
           "strides are not contiguous";
  if (isa<memref::ReinterpretCastOp>(def))
    return "its reinterpret_cast offset is not a constant, or it changes the "
           "element type";
  if (isa<memref::AllocOp, memref::AllocaOp>(def))
    return "its allocation has a dynamic extent or an element type of no fixed "
           "byte width";

  return "it is not derived from an allocation by views this analysis "
         "understands";
}

} // namespace

std::optional<BufferRange> mlir::npuisa::computeBufferRange(Value value) {
  BufferRange range;
  Value current = value;

  // The size is fixed by the type of the value we were asked about, before any
  // walking: a view's own type says how many bytes it denotes, and walking back
  // to the allocation changes only where those bytes start.
  auto type = dyn_cast<MemRefType>(current.getType());
  if (!type)
    return std::nullopt;
  std::optional<int64_t> size = byteSpan(type);
  if (!size)
    return std::nullopt;
  range.size = *size;

  // Walk back to the root allocation, accumulating byte offsets. The loop is
  // bounded by the definition chain, which is acyclic in valid IR; a malformed
  // module cannot reach here because the verifier that calls this runs after
  // the structural checks.
  while (true) {
    Operation *def = current.getDefiningOp();
    if (!def) {
      // A block argument is a root. A function argument memref is a real buffer
      // whose address this analysis cannot see, so two distinct block arguments
      // are treated as distinct bases below and one compared with itself
      // overlaps, which are both the right answers.
      range.base = current;
      return range;
    }

    if (isa<memref::AllocOp, memref::AllocaOp>(def)) {
      range.base = current;
      return range;
    }

    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      // memref.view takes a byte offset into a flat memref<Nxi8>, which is
      // exactly the form the allocator emits. A non constant offset stops the
      // walk with nothing, which the caller turns into a refusal.
      std::optional<int64_t> offset = constantIndex(view.getByteShift());
      if (!offset)
        return std::nullopt;
      range.offset += *offset;
      current = view.getSource();
      continue;
    }

    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      // A `memref.reinterpret_cast` resets the offset, the extents and the
      // strides against the *base pointer* of its source, which is why only its
      // own offset is added here and the source's own layout offset is not
      // consulted. Both of this project's producers of one write `offset: [0]`,
      // so the common answer is that this contributes nothing but the walk
      // continues; a non zero offset is handled anyway rather than asserted
      // away, because the operation permits it and a later pass may use it.
      //
      // The size was already fixed from the cast's own type at the top of this
      // function, by `byteSpan`, which is where the stride 0 decision is made
      // and explained. That decision is what makes this case worth adding
      // rather than leaving `Unknown`: a broadcast view spans C floats, not the
      // N by C by H by W its shape suggests, and the two answers differ by a
      // factor of the spatial extent.
      auto sourceType = dyn_cast<MemRefType>(cast.getSource().getType());
      auto resultType = dyn_cast<MemRefType>(cast.getType());
      if (!sourceType || !resultType ||
          sourceType.getElementType() != resultType.getElementType())
        return std::nullopt;
      std::optional<int64_t> width = byteWidth(resultType.getElementType());
      if (!width)
        return std::nullopt;

      SmallVector<OpFoldResult> mixedOffsets = cast.getMixedOffsets();
      if (mixedOffsets.size() != 1)
        return std::nullopt;
      std::optional<int64_t> elementOffset =
          getConstantIntValue(mixedOffsets.front());
      if (!elementOffset || *elementOffset < 0)
        return std::nullopt;

      range.offset += *elementOffset * *width;
      current = cast.getSource();
      continue;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      // A subview is understood only in the contiguous, fully static form the
      // passes in this project emit. Anything else returns nothing rather than
      // an approximation, because an approximate answer to "do these race" is
      // a wrong answer half the time and the wrong half is silent.
      auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
      if (!sourceType || !sourceType.hasStaticShape())
        return std::nullopt;
      std::optional<int64_t> width = byteWidth(sourceType.getElementType());
      if (!width)
        return std::nullopt;

      // Every stride is 1 and every offset is a constant, or this analysis
      // declines to answer. getMixedStrides and getMixedOffsets fold the
      // static and dynamic halves into one list, so a dynamic entry shows up
      // as an OpFoldResult with no constant value rather than as a separate
      // list to remember to check.
      SmallVector<OpFoldResult> mixedStrides = subview.getMixedStrides();
      for (OpFoldResult stride : mixedStrides) {
        std::optional<int64_t> value = getConstantIntValue(stride);
        if (!value || *value != 1)
          return std::nullopt;
      }

      SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();
      ArrayRef<int64_t> sourceShape = sourceType.getShape();
      if (mixedOffsets.size() != sourceShape.size())
        return std::nullopt;

      SmallVector<int64_t> offsets;
      offsets.reserve(mixedOffsets.size());
      for (OpFoldResult entry : mixedOffsets) {
        std::optional<int64_t> value = getConstantIntValue(entry);
        if (!value)
          return std::nullopt;
        offsets.push_back(*value);
      }

      // The element offset is the dot product of the static offsets with the
      // source's row major strides, which is the only stride order a memref
      // with the default identity layout has.
      int64_t elementOffset = 0;
      int64_t stride = 1;
      for (int64_t axis = static_cast<int64_t>(sourceShape.size()) - 1;
           axis >= 0; --axis) {
        elementOffset += offsets[axis] * stride;
        stride *= sourceShape[axis];
      }
      range.offset += elementOffset * *width;
      current = subview.getSource();
      continue;
    }

    // An operation this analysis does not understand. Stopping here with
    // nothing is what makes an unrecognised view a refusal rather than a
    // silently assumed disjointness.
    return std::nullopt;
  }
}

OverlapResult mlir::npuisa::overlaps(Value lhs, Value rhs) {
  std::optional<BufferRange> left = computeBufferRange(lhs);
  std::optional<BufferRange> right = computeBufferRange(rhs);
  if (!left || !right)
    return OverlapResult::Unknown;

  // Different roots are different memories. Two independent allocations do not
  // alias, and this is the one place where comparing values is right: it is a
  // comparison of *allocations*, not of the buffers under test, and the whole
  // point of walking back was to find them.
  if (left->base != right->base)
    return OverlapResult::Disjoint;

  // One base, so the byte ranges decide. Half open intervals intersect when
  // each starts before the other ends. Two identical values land here too and
  // report Overlaps, which is correct and is why identity never needs a special
  // case: a range always overlaps itself.
  const int64_t leftEnd = left->offset + left->size;
  const int64_t rightEnd = right->offset + right->size;
  if (left->offset < rightEnd && right->offset < leftEnd)
    return OverlapResult::Overlaps;
  return OverlapResult::Disjoint;
}

StringRef mlir::npuisa::describeWhyNotAnalysable(Value value) {
  if (computeBufferRange(value))
    return "";
  return notAnalysableReason(value);
}
