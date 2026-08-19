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

/// The total byte size of a statically shaped memref.
std::optional<int64_t> byteSize(MemRefType type) {
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
  return elements * *width;
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
  std::optional<int64_t> size = byteSize(type);
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
