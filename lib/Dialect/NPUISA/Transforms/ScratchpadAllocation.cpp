//===- ScratchpadAllocation.cpp - The allocator's arithmetic ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The sweep line, the two offset assignment strategies, and the two spill
// heuristics of Section 13.1. No MLIR types appear in this file, which is what
// lets `unittests/Dialect/NPUISA/AllocatorTest.cpp` compare the sweep line
// against a brute force recomputation on randomized interval sets rather than
// on generated modules.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/ScratchpadAllocation.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>

using namespace mlir;
using namespace mlir::npuisa;

namespace {

/// One end of a live range, as the sweep line sees it.
///
/// `isDefinition` is the sort key that encodes Section 13.1's ordering rule at
/// equal indices: deaths come before definitions, so a buffer that dies at the
/// same index another is born at does not inflate the peak by being counted
/// twice. `false` sorts before `true`, so a death carrying `false` is exactly
/// the rule and needs no comparator of its own.
struct SweepEvent {
  int64_t index = 0;
  bool isDefinition = false;
  int64_t delta = 0;
};

/// Rounds `value` up to the next multiple of `alignment`.
///
/// `alignment` is a positive power of two, which the caller checks, so this is
/// the mask form rather than a division. A negative value cannot reach here:
/// offsets start at zero and only ever grow.
int64_t alignUp(int64_t value, int64_t alignment) {
  assert(value >= 0 && "an offset is never negative");
  assert(alignment > 0 && "the alignment is validated by the caller");
  return (value + alignment - 1) & ~(alignment - 1);
}

/// A buffer that already has an offset, during placement.
struct Occupied {
  int64_t offset = 0;
  int64_t size = 0;
  int64_t definition = 0;
  int64_t lastUse = 0;
};

/// Whether two closed index ranges share an index.
///
/// Closed at both ends on purpose: a buffer is live at the operation that
/// allocates it and at its last use, so two buffers whose ranges touch at one
/// index really are live together there and must not share a byte. Making this
/// half open is the off by one that produces the silent wrong answer of
/// Section 13.1's fourth allocator test, reuse while live.
bool rangesOverlap(int64_t firstDefinition, int64_t firstLastUse,
                   int64_t secondDefinition, int64_t secondLastUse) {
  return firstDefinition <= secondLastUse && secondDefinition <= firstLastUse;
}

/// The order buffers are considered in, per strategy. Returns the indices of
/// `intervals`, permuted.
llvm::SmallVector<int64_t> placementOrder(llvm::ArrayRef<LiveInterval> intervals,
                                          Strategy strategy) {
  llvm::SmallVector<int64_t> order;
  order.reserve(intervals.size());
  for (int64_t index = 0; index < static_cast<int64_t>(intervals.size());
       ++index)
    order.push_back(index);

  // std::stable_sort rather than std::sort, even though every comparator below
  // is a total order on distinct indices. The tie breakers end in the
  // definition index, which two buffers allocated by the same operation cannot
  // share, so the order is already total; stability costs nothing and means a
  // future key that is not total does not silently reintroduce an ordering that
  // depends on the sort implementation.
  if (strategy == Strategy::Pack) {
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
      const LiveInterval &left = intervals[a];
      const LiveInterval &right = intervals[b];
      if (left.bytes != right.bytes)
        return left.bytes > right.bytes;
      if (left.span() != right.span())
        return left.span() > right.span();
      return left.definition < right.definition;
    });
    return order;
  }

  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    const LiveInterval &left = intervals[a];
    const LiveInterval &right = intervals[b];
    if (left.definition != right.definition)
      return left.definition < right.definition;
    if (left.bytes != right.bytes)
      return left.bytes > right.bytes;
    return left.span() > right.span();
  });
  return order;
}

} // namespace

//===----------------------------------------------------------------------===//
// The sweep line.
//===----------------------------------------------------------------------===//

PeakPressure mlir::npuisa::sweepLinePeak(llvm::ArrayRef<LiveInterval> intervals) {
  PeakPressure peak;
  if (intervals.empty())
    return peak;

  llvm::SmallVector<SweepEvent> events;
  events.reserve(intervals.size() * 2);
  for (const LiveInterval &interval : intervals) {
    events.push_back({interval.definition, true, interval.bytes});
    events.push_back({interval.lastUse + 1, false, -interval.bytes});
  }

  std::stable_sort(events.begin(), events.end(),
                   [](const SweepEvent &left, const SweepEvent &right) {
                     if (left.index != right.index)
                       return left.index < right.index;
                     return left.isDefinition < right.isDefinition;
                   });

  // One walk. The peak is updated only on a strictly greater sum, which is what
  // makes the *first* index with that sum the reported one: a later index that
  // merely equals the running maximum never replaces it. Checking after every
  // event rather than after every index gives the same answer, because within
  // an index the deaths run first and the running total is therefore at its
  // largest after the last definition at that index.
  int64_t resident = 0;
  for (const SweepEvent &event : events) {
    resident += event.delta;
    if (resident > peak.bytes) {
      peak.bytes = resident;
      peak.index = event.index;
    }
  }
  return peak;
}

PeakPressure
mlir::npuisa::bruteForcePeak(llvm::ArrayRef<LiveInterval> intervals) {
  PeakPressure peak;
  if (intervals.empty())
    return peak;

  int64_t first = intervals.front().definition;
  int64_t last = intervals.front().lastUse;
  for (const LiveInterval &interval : intervals) {
    first = std::min(first, interval.definition);
    last = std::max(last, interval.lastUse);
  }

  for (int64_t index = first; index <= last; ++index) {
    int64_t resident = 0;
    for (const LiveInterval &interval : intervals)
      if (interval.definition <= index && index <= interval.lastUse)
        resident += interval.bytes;
    if (resident > peak.bytes) {
      peak.bytes = resident;
      peak.index = index;
    }
  }
  return peak;
}

//===----------------------------------------------------------------------===//
// Offset assignment.
//===----------------------------------------------------------------------===//

std::optional<Placement>
mlir::npuisa::assignOffsets(llvm::ArrayRef<LiveInterval> intervals,
                            Strategy strategy, int64_t alignment,
                            int64_t budget, PlacementFailure *failure) {
  assert(alignment > 0 && (alignment & (alignment - 1)) == 0 &&
         "the alignment is a positive power of two, checked by the pass");

  Placement placement;
  placement.offsets.assign(intervals.size(), 0);
  if (intervals.empty())
    return placement;

  llvm::SmallVector<Occupied> placed;
  placed.reserve(intervals.size());

  llvm::SmallVector<Occupied> neighbours;
  for (int64_t index : placementOrder(intervals, strategy)) {
    const LiveInterval &interval = intervals[index];

    // Only the buffers this one is live at the same time as constrain it. That
    // is the whole content of the allocator: two buffers whose live ranges are
    // disjoint may share a byte offset, and two whose ranges touch may not.
    neighbours.clear();
    for (const Occupied &other : placed)
      if (rangesOverlap(interval.definition, interval.lastUse, other.definition,
                        other.lastUse))
        neighbours.push_back(other);
    std::stable_sort(neighbours.begin(), neighbours.end(),
                     [](const Occupied &left, const Occupied &right) {
                       return left.offset < right.offset;
                     });

    // First fit: walk the occupied blocks in increasing offset order and take
    // the first gap this buffer fits in. The candidate starts aligned at zero
    // and is realigned after every block it has to skip, so every offset this
    // returns is a multiple of the alignment.
    int64_t candidate = 0;
    for (const Occupied &other : neighbours) {
      if (candidate + interval.bytes <= other.offset)
        break;
      candidate = std::max(candidate, alignUp(other.offset + other.size,
                                              alignment));
    }

    if (candidate + interval.bytes > budget) {
      if (failure)
        *failure = PlacementFailure{index, candidate};
      return std::nullopt;
    }

    placement.offsets[index] = candidate;
    placement.highWaterMark =
        std::max(placement.highWaterMark, candidate + interval.bytes);
    placed.push_back({candidate, interval.bytes, interval.definition,
                      interval.lastUse});
  }

  return placement;
}

//===----------------------------------------------------------------------===//
// The spill heuristics.
//===----------------------------------------------------------------------===//

std::optional<int64_t>
mlir::npuisa::chooseSpillVictim(llvm::ArrayRef<SpillCandidate> candidates,
                                SpillHeuristic heuristic) {
  std::optional<int64_t> best;
  for (int64_t index = 0; index < static_cast<int64_t>(candidates.size());
       ++index) {
    const SpillCandidate &candidate = candidates[index];
    if (!candidate.spillable)
      continue;
    if (!best) {
      best = index;
      continue;
    }

    const SpillCandidate &incumbent = candidates[*best];
    bool better = false;
    if (heuristic == SpillHeuristic::LongestRange) {
      // The longest live range crossing the peak, then the same three keys the
      // cost rule uses so that an ablation between the two heuristics is not
      // partly a measurement of two different tie breakers.
      if (candidate.interval.span() != incumbent.interval.span())
        better = candidate.interval.span() > incumbent.interval.span();
      else if (candidate.interval.bytes != incumbent.interval.bytes)
        better = candidate.interval.bytes > incumbent.interval.bytes;
      else
        better = candidate.interval.definition < incumbent.interval.definition;
    } else {
      // cost = bytes * (1 + reloads), smallest cost wins. Ties break by larger
      // bytes first, then longer span, then earlier definition index, which is
      // Section 13.1 word for word.
      const int64_t candidateCost =
          candidate.interval.bytes * (1 + candidate.usesAfterPeak);
      const int64_t incumbentCost =
          incumbent.interval.bytes * (1 + incumbent.usesAfterPeak);
      if (candidateCost != incumbentCost)
        better = candidateCost < incumbentCost;
      else if (candidate.interval.bytes != incumbent.interval.bytes)
        better = candidate.interval.bytes > incumbent.interval.bytes;
      else if (candidate.interval.span() != incumbent.interval.span())
        better = candidate.interval.span() > incumbent.interval.span();
      else
        better = candidate.interval.definition < incumbent.interval.definition;
    }

    if (better)
      best = index;
  }
  return best;
}

//===----------------------------------------------------------------------===//
// The option strings.
//===----------------------------------------------------------------------===//

std::optional<Strategy> mlir::npuisa::parseStrategy(llvm::StringRef text) {
  if (text == "pack")
    return Strategy::Pack;
  if (text == "interval")
    return Strategy::Interval;
  return std::nullopt;
}

std::optional<SpillHeuristic>
mlir::npuisa::parseSpillHeuristic(llvm::StringRef text) {
  if (text == "longest-range")
    return SpillHeuristic::LongestRange;
  if (text == "cost")
    return SpillHeuristic::Cost;
  return std::nullopt;
}

llvm::StringRef mlir::npuisa::strategyOptions() { return "pack, interval"; }

llvm::StringRef mlir::npuisa::spillHeuristicOptions() {
  return "longest-range, cost";
}

llvm::StringRef mlir::npuisa::strategyName(Strategy strategy) {
  return strategy == Strategy::Pack ? "pack" : "interval";
}

llvm::StringRef mlir::npuisa::spillHeuristicName(SpillHeuristic heuristic) {
  return heuristic == SpillHeuristic::LongestRange ? "longest-range" : "cost";
}
