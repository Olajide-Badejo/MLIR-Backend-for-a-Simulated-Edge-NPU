//===- NPUISAMemoryOverlap.h - The overlap test ----------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The byte range a memref denotes, and whether two of them overlap.
//
// This is a header rather than an implementation detail of the verifier because
// the unit tests call it directly. Section 8 asks for the overlap rule to be
// decided on view offsets and extents rather than on SSA identity, and a test
// that could only reach the rule through a verifier diagnostic would be testing
// the message rather than the arithmetic.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISAMEMORYOVERLAP_H
#define NPU_DIALECT_NPUISA_IR_NPUISAMEMORYOVERLAP_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace npuisa {

/// A half open byte range `[offset, offset + size)` within one underlying
/// buffer, plus the buffer it is measured from.
///
/// `base` is the root allocation the value was derived from, found by walking
/// back through the view like operations. Two ranges are only comparable when
/// their bases are the same value: two independent `memref.alloc` results are
/// two different memories and an offset of 0 in each means two different bytes.
struct BufferRange {
  /// The root allocation this range is measured from.
  Value base;
  /// The byte offset of the first byte, from the start of `base`.
  int64_t offset = 0;
  /// The number of bytes. Always positive for a well formed range.
  int64_t size = 0;
};

/// The result of asking whether two values touch the same bytes.
///
/// Three outcomes rather than two, and the third is the one that matters.
/// `Unknown` is what an unanalysable offset produces, and Section 8 requires it
/// to be treated as a refusal rather than as a disjointness: "I cannot prove
/// these overlap" and "these are disjoint" are different answers and only one
/// of them makes a program safe.
enum class OverlapResult {
  /// The two ranges provably share at least one byte.
  Overlaps,
  /// The two ranges are provably disjoint, either because their bases are
  /// distinct allocations or because their byte ranges do not intersect.
  Disjoint,
  /// Not decidable from the IR: a non static offset, a base that is not a
  /// recognised allocation, or an element type with no fixed byte width.
  Unknown,
};

/// Walks `value` back to its root allocation and computes the byte range it
/// denotes, or nothing when the chain is not analysable.
///
/// The chain this understands is the one the allocator of Section 13.1
/// produces: a `memref.view` at a constant byte offset over a flat
/// `memref<Nxi8>`, possibly nested, over a `memref.alloc`. A `memref.subview`
/// is understood in the same way when its offsets, sizes and strides are all
/// static and its strides are the identity, which is the only form the passes
/// in this project emit; anything else returns nothing rather than an
/// approximation, because an approximate answer to "do these race" is a wrong
/// answer half the time.
std::optional<BufferRange> computeBufferRange(Value value);

/// Whether the two values touch any byte in common.
///
/// **This never compares SSA values for identity as its decision procedure.**
/// Identical values do overlap and the arithmetic says so on its own, because
/// two identical values produce two identical ranges and a range always
/// overlaps itself. What the arithmetic additionally catches, and what an
/// identity check cannot, is two different `memref.view` results over one flat
/// scratchpad buffer whose byte ranges partially intersect: distinct SSA
/// values, one shared region of memory, a real race.
OverlapResult overlaps(Value lhs, Value rhs);

/// A human readable reason for an `Unknown` result, for use in a diagnostic.
/// Returns an empty string when the value is analysable.
llvm::StringRef describeWhyNotAnalysable(Value value);

} // namespace npuisa
} // namespace mlir

#endif // NPU_DIALECT_NPUISA_IR_NPUISAMEMORYOVERLAP_H
