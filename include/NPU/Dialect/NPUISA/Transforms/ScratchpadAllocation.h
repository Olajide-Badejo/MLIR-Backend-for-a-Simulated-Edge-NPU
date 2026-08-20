//===- ScratchpadAllocation.h - The allocator's arithmetic ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 13.1's arithmetic, with no IR in it.
//
// The split is the same one `NPUISAMemoryOverlap.h` makes and for the same
// reason: the unit tests call this directly. A property test that compared a
// sweep line against a brute force recomputation through a pass would be
// generating MLIR modules to describe interval sets, which is a slow way of
// writing down three integers and a way that cannot reach the interesting
// shapes. Section 17.2 asks for randomized *interval sets*, so intervals are
// what this header takes.
//
// Everything here is deterministic. Ground rule 16: every heuristic breaks ties
// by a documented rule, and each rule is written next to the function that
// applies it rather than in a document that can drift from it.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADALLOCATION_H
#define NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADALLOCATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace npuisa {

//===----------------------------------------------------------------------===//
// The live interval.
//===----------------------------------------------------------------------===//

/// One buffer's rectangle: a live range and a size.
///
/// Section 13.1 frames the problem as 2D strip packing rather than linear scan,
/// and this struct is that framing written down. The width is
/// `[definition, lastUse]`, the height is `bytes`, and the strip has a height
/// of the budget. Nothing here knows what a memref is.
///
/// `definition` and `lastUse` are operation indices in one straight line block,
/// and the range is closed at both ends: a buffer is live at the operation that
/// allocates it and at the last operation that reads or writes it. A buffer
/// nothing uses has `lastUse == definition`.
struct LiveInterval {
  /// The index of the operation that allocates the buffer.
  int64_t definition = 0;
  /// The index of the last operation that reads or writes it, inclusive.
  int64_t lastUse = 0;
  /// The buffer's size in bytes, from its type and element type.
  int64_t bytes = 0;

  /// The number of operations the buffer is live across, at least one.
  int64_t span() const { return lastUse - definition + 1; }
};

//===----------------------------------------------------------------------===//
// The sweep line.
//===----------------------------------------------------------------------===//

/// Where the pressure peaks, and how high.
struct PeakPressure {
  /// The largest number of bytes live at any one index.
  int64_t bytes = 0;
  /// The first index at which that many bytes are live.
  int64_t index = 0;
};

/// The peak simultaneous live bytes, by the sweep line of Section 13.1.
///
/// One event list of `(index, +size)` at each definition and `(lastUse + 1,
/// -size)` at each death, sorted once, walked once, maintaining a running
/// resident total. O(n log n) for the sort and O(n) for the walk. The naive
/// nested formulation is O(instructions times buffers) and gets recomputed
/// inside the spill loop, which makes the whole pass cubic in the worst case,
/// which is why this is the shape from the start rather than an optimization
/// somebody applies later.
///
/// **Tie breaking is specified, not incidental.** The first index with a
/// strictly greater sum wins the peak, and at equal indices deaths are ordered
/// before definitions. An off by one there changes which buffer is spilled and
/// therefore the published numbers, so `unittests/Dialect/NPUISA/
/// AllocatorTest.cpp` asserts the index as well as the value against a brute
/// force recomputation.
///
/// **This is a lower bound, not a placement test.** Peak simultaneous live
/// bytes is the minimum arena any placement could need. Fragmentation means a
/// program whose peak sits under the budget can still fail to place, so the
/// spill trigger is "offset assignment failed" and never "peak exceeded
/// budget".
///
/// An empty interval set peaks at zero bytes at index zero.
PeakPressure sweepLinePeak(llvm::ArrayRef<LiveInterval> intervals);

/// The same answer computed the slow way: for every index in range, the sum of
/// the sizes of the intervals covering it.
///
/// This exists so that the property test of Section 17.2 has something to
/// compare the sweep line against, and it lives here rather than in the test so
/// that both halves of the comparison are compiled with the same flags and read
/// the same struct. It is O(indices times intervals) and is never called by the
/// pass.
PeakPressure bruteForcePeak(llvm::ArrayRef<LiveInterval> intervals);

//===----------------------------------------------------------------------===//
// Offset assignment.
//===----------------------------------------------------------------------===//

/// Which of the two offset assignment strategies of Section 13.1 to use.
enum class Strategy {
  /// The greedy by size offset calculation algorithm [R28], the one shipped in
  /// TFLite Micro's arena planner: buffers are placed largest first, each at
  /// the lowest offset that does not collide with an already placed buffer it
  /// is live at the same time as.
  Pack,
  /// The named baseline: the same placement rule applied in definition order,
  /// which is the interval scheme this project started from.
  Interval,
};

/// Which of the two spill heuristics of Section 13.1 to use.
enum class SpillHeuristic {
  /// Spill the buffer with the longest live range crossing the pressure peak.
  LongestRange,
  /// A Belady style rule: `cost = bytes * (1 + reloads)`, smallest cost wins.
  Cost,
};

/// The result of a successful offset assignment.
struct Placement {
  /// The byte offset assigned to each interval, in the order the intervals were
  /// given, never in the order they were placed.
  llvm::SmallVector<int64_t> offsets;
  /// One past the last byte any buffer occupies. This is what the function's
  /// `npuisa.scratchpad_bytes` attribute records: the arena that was actually
  /// used, not the budget that was available.
  int64_t highWaterMark = 0;
};

/// Why an offset assignment failed.
struct PlacementFailure {
  /// The interval that could not be placed, as an index into the input.
  int64_t interval = 0;
  /// The lowest offset at which it would have fitted, which is what exceeded
  /// the budget. Reported so a diagnostic can quote a number rather than say
  /// that something did not fit.
  int64_t wantedOffset = 0;
};

/// Assigns every interval a byte offset in `[0, budget)`, or says which one it
/// could not place.
///
/// Both strategies share one placement rule and differ only in the order they
/// consider buffers in, which is the whole difference between greedy by size
/// and the interval scheme. The rule is first fit against the already placed
/// buffers whose live ranges overlap this one: walk their occupied byte ranges
/// in increasing offset order and take the first gap the buffer fits in, with
/// the candidate offset rounded up to `alignment` after every occupied block.
///
/// Ties in the ordering break deterministically, and the rules are these:
///
/// - `Pack`: larger bytes first, then longer span, then earlier definition
///   index. Size is the primary key because that is what makes the algorithm
///   greedy by size; the other two exist so that two buffers of equal size
///   never depend on the order a hash table happened to produce.
/// - `Interval`: earlier definition index, then larger bytes, then longer span.
///   Definition order is the primary key because that is what makes it the
///   interval scheme rather than a second copy of the packer.
///
/// `alignment` must be a positive power of two. Offsets are aligned; sizes are
/// not padded, so the high water mark is the last byte genuinely occupied and
/// the fragmentation ratio it feeds is not inflated by the tail of the last
/// buffer.
///
/// A zero sized interval is placed at the first aligned offset that its live
/// neighbours leave free and occupies nothing, which keeps its offset
/// meaningful without letting it consume the arena.
std::optional<Placement> assignOffsets(llvm::ArrayRef<LiveInterval> intervals,
                                       Strategy strategy, int64_t alignment,
                                       int64_t budget,
                                       PlacementFailure *failure = nullptr);

//===----------------------------------------------------------------------===//
// The spill heuristics.
//===----------------------------------------------------------------------===//

/// One buffer, as the spill heuristics see it.
struct SpillCandidate {
  /// The live interval, so that both heuristics can read bytes and span.
  LiveInterval interval;
  /// The number of uses strictly after the pressure peak index. This is the
  /// `reloads` of Section 13.1's cost rule: every one of them becomes a
  /// `dma_load` if this buffer is spilled.
  int64_t usesAfterPeak = 0;
  /// Whether this buffer can be spilled at all. A buffer that is written more
  /// than once, that has a view taken of it, that is itself a spill reload, or
  /// that has no use after its definition is not spillable, and the pass
  /// documents each of those.
  bool spillable = true;
};

/// Picks the buffer to spill, or nothing when no candidate is spillable.
///
/// The candidate set is the buffers live across the pressure peak, which the
/// caller filters, and the returned value is an index into `candidates`.
///
/// **`LongestRange`**: the longest live range crossing the peak. Section 13.1
/// specifies the primary key and leaves the ties to the implementation, so the
/// remaining keys are the same ones the cost rule uses, in the same order:
/// longest span, then larger bytes, then earlier definition index. Two rules
/// that break ties differently would make a heuristic ablation partly a
/// measurement of the tie breaker.
///
/// **`Cost`**: `cost = bytes * (1 + usesAfterPeak)`, and the smallest cost
/// wins, which is Section 13.1 word for word. Ties break by larger bytes first,
/// then longer span, then earlier definition index, which is also Section 13.1
/// word for word.
std::optional<int64_t>
chooseSpillVictim(llvm::ArrayRef<SpillCandidate> candidates,
                  SpillHeuristic heuristic);

//===----------------------------------------------------------------------===//
// The option strings.
//===----------------------------------------------------------------------===//

/// Parses a `strategy` option value, or nothing when it is not one of ours.
std::optional<Strategy> parseStrategy(llvm::StringRef text);

/// Parses a `spill-heuristic` option value, or nothing.
std::optional<SpillHeuristic> parseSpillHeuristic(llvm::StringRef text);

/// The accepted `strategy` values, in the order a diagnostic should list them.
llvm::StringRef strategyOptions();

/// The accepted `spill-heuristic` values, in the order a diagnostic lists them.
llvm::StringRef spillHeuristicOptions();

/// The name of a strategy, for a diagnostic or a remark.
llvm::StringRef strategyName(Strategy strategy);

/// The name of a spill heuristic.
llvm::StringRef spillHeuristicName(SpillHeuristic heuristic);

/// The default scratchpad budget in bytes, used when the function carries no
/// `npuisa.scratchpad_budget` attribute and the pass was given no `budget`
/// option.
///
/// One mebibyte. Section 13.1 names no constant, so this one is inherited
/// rather than invented: it is the budget the previous build of this project
/// called the default and reported every generous budget cell at, and Section
/// 15 requires the tight budgets to be measured as a fraction of the peak
/// observed at the default. Changing it silently would move every cell in the
/// project's history at once.
inline constexpr int64_t kDefaultScratchpadBudget = 1048576;

/// The default offset alignment in bytes.
///
/// Sixty four, because the array of Section 5.3 is 16 by 16 and consumes a row
/// of 16 `f32` lanes at a time, which is 64 bytes. An offset that is not a
/// multiple of that would put a row across two SRAM words on the machine this
/// models. It is a pass option so that a test can pin it, and it is a power of
/// two so that the rounding is a mask.
inline constexpr int64_t kDefaultAlignment = 64;

} // namespace npuisa
} // namespace mlir

#endif // NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADALLOCATION_H
