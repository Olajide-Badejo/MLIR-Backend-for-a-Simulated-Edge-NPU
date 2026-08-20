//===- AllocatorTest.cpp - The allocator's arithmetic ----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 17.2's property test, and the hand written cases around it.
//
// **Why this is a unit test and not a lit test.** The lit files in
// `test/Dialect/NPUISA/` check what the allocator does to a program, which is
// the right level for the offsets and the spills. They cannot reach the two
// claims here. The first is the sweep line agreeing with a brute force
// recomputation on a thousand randomized interval sets: writing those as MLIR
// modules would be a slow way of writing down three integers, and the shapes
// that break an off by one are exactly the ones nobody writes by hand. The
// second is the tie breaking, which is a decision about two candidates that
// score equally and is therefore invisible in the output unless the test knows
// what the other candidate was.
//
// Every generator here takes a fixed seed, per ground rule 16. A failure that
// cannot be reproduced is a rumour.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/ScratchpadAllocation.h"

#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <random>

using namespace mlir;
using namespace mlir::npuisa;

namespace {

//===----------------------------------------------------------------------===//
// The generator.
//===----------------------------------------------------------------------===//

/// The seed. One constant, named, so that a failing case can be reproduced by
/// reading the case number out of the failure message and running the loop
/// again. Section 17.2 asks for a fixed seed and this is it.
constexpr unsigned kSeed = 0x5EEDu;

/// The number of randomized cases. Section 17.2 asks for at least 1000.
constexpr int kCases = 1000;

/// One randomized interval set.
///
/// The shape of the distribution matters more than its size. Live ranges are
/// drawn short relative to the index space, so that a typical set has many
/// disjoint intervals and the peak is a genuine maximum rather than the sum of
/// everything; and the index space is drawn small enough that ranges collide
/// often, because a set where nothing overlaps tests neither the sweep line's
/// ordering nor the placement's collision rule. Sizes are drawn from a range
/// that straddles the alignment, so that both the rounded and the exact cases
/// occur.
llvm::SmallVector<LiveInterval> randomIntervals(std::mt19937 &engine) {
  std::uniform_int_distribution<int> countOf(0, 24);
  const int count = countOf(engine);

  llvm::SmallVector<LiveInterval> intervals;
  intervals.reserve(count);
  std::uniform_int_distribution<int64_t> startOf(0, 60);
  std::uniform_int_distribution<int64_t> lengthOf(0, 12);
  std::uniform_int_distribution<int64_t> bytesOf(1, 300);
  for (int index = 0; index < count; ++index) {
    LiveInterval interval;
    interval.definition = startOf(engine);
    interval.lastUse = interval.definition + lengthOf(engine);
    interval.bytes = bytesOf(engine);
    intervals.push_back(interval);
  }
  return intervals;
}

/// Whether two closed index ranges share an index. Written out again here
/// rather than reused from the implementation, deliberately: a test that
/// borrowed the predicate under test would agree with it by construction.
bool live(const LiveInterval &left, const LiveInterval &right) {
  return left.definition <= right.lastUse && right.definition <= left.lastUse;
}

//===----------------------------------------------------------------------===//
// 1. The sweep line, on cases small enough to reason about by hand.
//===----------------------------------------------------------------------===//

TEST(SweepLine, AnEmptySetPeaksAtNothing) {
  PeakPressure peak = sweepLinePeak({});
  EXPECT_EQ(peak.bytes, 0);
  EXPECT_EQ(peak.index, 0);
}

TEST(SweepLine, OneIntervalPeaksAtItsDefinition) {
  llvm::SmallVector<LiveInterval> intervals{{4, 9, 512}};
  PeakPressure peak = sweepLinePeak(intervals);
  EXPECT_EQ(peak.bytes, 512);
  EXPECT_EQ(peak.index, 4);
}

// The tie break Section 13.1 specifies: at equal indices, deaths are ordered
// before definitions. The first interval dies at index 3 and the second is
// defined at 3, so at no point are both resident and the peak is 512, not 1024.
//
// This is the off by one the specification calls out by name, and getting it
// wrong changes which buffer is spilled and therefore the published numbers.
TEST(SweepLine, ADeathAtTheSameIndexAsADefinitionComesFirst) {
  llvm::SmallVector<LiveInterval> intervals{{0, 2, 512}, {3, 6, 512}};
  PeakPressure peak = sweepLinePeak(intervals);
  EXPECT_EQ(peak.bytes, 512)
      << "an interval whose last use is index 2 is dead at index 3";
  EXPECT_EQ(peak.index, 0);
}

// And the other side of it. A last use *at* index 3 keeps the interval resident
// at index 3, so both are live and the peak is 1024. The two tests differ by one
// in one field and by a factor of two in the answer.
TEST(SweepLine, ALastUseAtTheSameIndexIsStillLive) {
  llvm::SmallVector<LiveInterval> intervals{{0, 3, 512}, {3, 6, 512}};
  PeakPressure peak = sweepLinePeak(intervals);
  EXPECT_EQ(peak.bytes, 1024);
  EXPECT_EQ(peak.index, 3);
}

// The first index with a strictly greater sum wins. Both index 2 and index 8
// hold 768 bytes here, and the earlier one is the answer, because a later index
// that merely equals the running maximum never replaces it.
TEST(SweepLine, TheFirstIndexWithTheGreatestSumWins) {
  llvm::SmallVector<LiveInterval> intervals{
      {0, 4, 256}, {2, 4, 512}, {6, 12, 256}, {8, 12, 512}};
  PeakPressure peak = sweepLinePeak(intervals);
  EXPECT_EQ(peak.bytes, 768);
  EXPECT_EQ(peak.index, 2) << "the earlier of two equal peaks is the peak";
}

//===----------------------------------------------------------------------===//
// 2. The property test of Section 17.2.
//===----------------------------------------------------------------------===//

// The sweep line against a brute force recomputation, on randomized interval
// sets, at least 1000 cases, fixed seed.
//
// **The index is asserted as well as the value**, and that is the half that
// matters. Two implementations agreeing on how many bytes are live at the worst
// moment is worth having; agreeing on *which* moment is what the spill heuristic
// reads, because the candidate set is the buffers live across the peak index. A
// property test that checked only the total would pass against a sweep line
// whose ordering at equal indices was backwards.
TEST(SweepLineProperty, TheSweepLineAgreesWithBruteForce) {
  std::mt19937 engine(kSeed);
  int nonTrivial = 0;
  for (int caseNumber = 0; caseNumber < kCases; ++caseNumber) {
    llvm::SmallVector<LiveInterval> intervals = randomIntervals(engine);
    PeakPressure fast = sweepLinePeak(intervals);
    PeakPressure slow = bruteForcePeak(intervals);

    ASSERT_EQ(fast.bytes, slow.bytes)
        << "case " << caseNumber << " of " << kCases << " at seed " << kSeed
        << ": the sweep line and the brute force disagree on the peak";
    ASSERT_EQ(fast.index, slow.index)
        << "case " << caseNumber << " of " << kCases << " at seed " << kSeed
        << ": the sweep line and the brute force disagree on *where* the peak "
           "is, which is what the spill heuristic reads";
    if (intervals.size() > 1 && fast.bytes > 0)
      ++nonTrivial;
  }

  std::cout << "[ SWEEP    ] " << kCases << " randomized interval sets at seed "
            << kSeed << ", " << nonTrivial
            << " of them with more than one interval and a non zero peak\n";
  EXPECT_GT(nonTrivial, kCases / 2)
      << "most cases must be non trivial, or the property is being tested "
         "against empty sets";
}

// The placement invariant, on the same generator. This is the "no reuse while
// live" case of Section 13.1 asserted as a property rather than as one hand
// written program: **two buffers that are live at the same index never share a
// byte.** A reuse bug here would be silently wrong rather than loudly wrong,
// which is the worst kind, and one lit test can only rule it out for one shape.
//
// Three more invariants ride along, because they cost one loop each and each of
// them is a way the placement could be wrong while the overlap rule held: every
// offset is aligned, nothing crosses the budget, and the high water mark is at
// least the sweep line peak, which is what makes the peak a lower bound rather
// than an estimate.
TEST(PlacementProperty, NoTwoBuffersLiveAtOnceShareAByte) {
  const Strategy strategies[] = {Strategy::Pack, Strategy::Interval};
  constexpr int64_t kAlignment = 64;
  constexpr int64_t kBudget = 1 << 20;

  for (Strategy strategy : strategies) {
    std::mt19937 engine(kSeed);
    for (int caseNumber = 0; caseNumber < kCases; ++caseNumber) {
      llvm::SmallVector<LiveInterval> intervals = randomIntervals(engine);
      std::optional<Placement> placement =
          assignOffsets(intervals, strategy, kAlignment, kBudget);
      ASSERT_TRUE(placement)
          << "case " << caseNumber << " under " << strategyName(strategy).str()
          << ": a megabyte is more than these sets can need, so a failure here "
             "is the placement giving up rather than the budget binding";
      ASSERT_EQ(placement->offsets.size(), intervals.size());

      for (size_t i = 0; i < intervals.size(); ++i) {
        EXPECT_EQ(placement->offsets[i] % kAlignment, 0)
            << "case " << caseNumber << ", buffer " << i
            << ": every offset is a multiple of the alignment";
        EXPECT_LE(placement->offsets[i] + intervals[i].bytes,
                  placement->highWaterMark);

        for (size_t j = i + 1; j < intervals.size(); ++j) {
          if (!live(intervals[i], intervals[j]))
            continue;
          const int64_t firstEnd = placement->offsets[i] + intervals[i].bytes;
          const int64_t secondEnd = placement->offsets[j] + intervals[j].bytes;
          ASSERT_FALSE(placement->offsets[i] < secondEnd &&
                       placement->offsets[j] < firstEnd)
              << "case " << caseNumber << " under "
              << strategyName(strategy).str() << ": buffers " << i << " and "
              << j << " are live together and were given overlapping bytes";
        }
      }

      EXPECT_GE(placement->highWaterMark, sweepLinePeak(intervals).bytes)
          << "case " << caseNumber
          << ": the peak is a lower bound on any placement, so a high water "
             "mark below it means one of the two is wrong";
    }
  }

  std::cout << "[ PLACE    ] " << kCases << " sets at seed " << kSeed
            << ", under each of the two strategies\n";
}

//===----------------------------------------------------------------------===//
// 3. Offset assignment, on cases whose answer is written out.
//===----------------------------------------------------------------------===//

TEST(Placement, ADeadBuffersBytesComeBack) {
  // [0, 3] and [4, 7] never share an index, so the second takes the first's
  // offset and the arena is one buffer wide rather than two.
  llvm::SmallVector<LiveInterval> intervals{{0, 3, 512}, {4, 7, 512}};
  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 64, 4096);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->offsets[0], 0);
  EXPECT_EQ(packed->offsets[1], 0);
  EXPECT_EQ(packed->highWaterMark, 512);
}

TEST(Placement, TwoLiveBuffersDoNotShareAnOffset) {
  llvm::SmallVector<LiveInterval> intervals{{0, 4, 512}, {4, 7, 512}};
  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 64, 4096);
  ASSERT_TRUE(packed);
  EXPECT_NE(packed->offsets[0], packed->offsets[1]);
  EXPECT_EQ(packed->highWaterMark, 1024);
}

TEST(Placement, OffsetsRoundUpToTheAlignment) {
  // 68 bytes is 17 f32, and the next offset is 128 rather than 68 because 68
  // rounds up to the next multiple of 64. The high water mark is then 196, and
  // the 60 bytes between 68 and 128 are lost to alignment rather than to
  // fragmentation, which is worth separating: they are the price of the
  // hardware's word, not of the algorithm.
  llvm::SmallVector<LiveInterval> intervals{{0, 4, 68}, {1, 5, 68}};
  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 64, 4096);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->offsets[0], 0);
  EXPECT_EQ(packed->offsets[1], 128);
  EXPECT_EQ(packed->highWaterMark, 196);
}

TEST(Placement, AnAlignmentOfOnePacksTight) {
  llvm::SmallVector<LiveInterval> intervals{{0, 4, 68}, {1, 5, 68}};
  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 1, 4096);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->offsets[1], 68)
      << "the alignment is an option, and at 1 it costs nothing";
  EXPECT_EQ(packed->highWaterMark, 136);
}

// The two strategies on the fragmentation case of Section 13.1: a program whose
// peak is 768 bytes, which `pack` achieves and `interval` does not.
//
// This is the same program `test/Dialect/NPUISA/scratchpad-alloc.mlir` carries,
// as three integers rather than as MLIR, and having both is the point: the lit
// file proves the pass produces this placement and this file proves the
// placement is what the algorithm says it should be.
TEST(Placement, ThePackerBeatsTheIntervalSchemeOnFragmentation) {
  llvm::SmallVector<LiveInterval> intervals{
      {0, 3, 256}, {1, 5, 256}, {4, 6, 512}};
  EXPECT_EQ(sweepLinePeak(intervals).bytes, 768);

  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 64, 1 << 20);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->highWaterMark, 768) << "greedy by size finds the optimum";
  EXPECT_EQ(packed->offsets[2], 0) << "the largest buffer goes down first";

  std::optional<Placement> intervalScheme =
      assignOffsets(intervals, Strategy::Interval, 64, 1 << 20);
  ASSERT_TRUE(intervalScheme);
  EXPECT_EQ(intervalScheme->highWaterMark, 1024)
      << "definition order leaves a 256 byte hole nothing fits in";
  EXPECT_EQ(intervalScheme->offsets[2], 512);
}

// The failure report, which is what the budget too small diagnostic quotes.
TEST(Placement, AFailureNamesTheBufferAndTheOffsetItWanted) {
  llvm::SmallVector<LiveInterval> intervals{{0, 4, 512}, {1, 5, 512}};
  PlacementFailure failure;
  std::optional<Placement> packed =
      assignOffsets(intervals, Strategy::Pack, 64, 768, &failure);
  EXPECT_FALSE(packed);
  EXPECT_EQ(failure.interval, 1)
      << "the second buffer is the one that could not be placed";
  EXPECT_EQ(failure.wantedOffset, 512)
      << "and 512 plus 512 is what overran the budget of 768";
}

// Peak under the budget, placement over it. Section 13.1 turns on this being
// possible: it is why the spill trigger is offset assignment failing and never
// the peak exceeding the budget.
TEST(Placement, APeakUnderTheBudgetCanStillFailToPlace) {
  llvm::SmallVector<LiveInterval> intervals{
      {0, 3, 512}, {1, 5, 512}, {4, 6, 1024}};
  constexpr int64_t kBudget = 1536;
  EXPECT_EQ(sweepLinePeak(intervals).bytes, kBudget)
      << "the peak fits the budget exactly, so the peak test would pass";

  EXPECT_FALSE(assignOffsets(intervals, Strategy::Interval, 64, kBudget))
      << "and the interval placement still fails, which is the whole point";
  EXPECT_TRUE(assignOffsets(intervals, Strategy::Pack, 64, kBudget))
      << "while the packer places the same program inside the same budget";
}

TEST(Placement, AnEmptyProgramNeedsNoArena) {
  std::optional<Placement> packed =
      assignOffsets({}, Strategy::Pack, 64, 4096);
  ASSERT_TRUE(packed);
  EXPECT_EQ(packed->highWaterMark, 0);
  EXPECT_TRUE(packed->offsets.empty());
}

//===----------------------------------------------------------------------===//
// 4. The spill heuristics, and their tie breaks.
//===----------------------------------------------------------------------===//

TEST(SpillHeuristic, NothingSpillableIsNoVictim) {
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 10, 512}, 1, false}, {{2, 4, 256}, 1, false}};
  EXPECT_FALSE(chooseSpillVictim(candidates, SpillHeuristic::LongestRange));
  EXPECT_FALSE(chooseSpillVictim(candidates, SpillHeuristic::Cost));
}

TEST(SpillHeuristic, AnUnspillableCandidateIsSkippedRatherThanChosen) {
  // The first would win on both rules if it were spillable. It is not, so the
  // second is the answer, and an implementation that filtered after choosing
  // rather than before would return the first and spill nothing.
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 40, 64}, 0, false}, {{2, 6, 512}, 1, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::LongestRange), 1);
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::Cost), 1);
}

TEST(SpillHeuristic, LongestRangeTakesTheLongestSpan) {
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 4, 4096}, 3, true}, {{1, 30, 64}, 1, true}, {{2, 8, 512}, 2, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::LongestRange), 1)
      << "span 30 beats span 7 and span 5, whatever the sizes are";
}

// The tie break chain for `longest-range`, one key at a time. Equal spans fall
// through to bytes, and equal bytes fall through to the definition index.
TEST(SpillHeuristic, LongestRangeBreaksASpanTieByBytes) {
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 9, 256}, 1, true}, {{4, 13, 512}, 1, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::LongestRange), 1)
      << "equal spans of 10, so the larger buffer wins";
}

TEST(SpillHeuristic, LongestRangeBreaksAByteTieByDefinitionIndex) {
  llvm::SmallVector<SpillCandidate> candidates{
      {{7, 16, 512}, 1, true}, {{3, 12, 512}, 1, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::LongestRange), 1)
      << "equal spans and equal sizes, so the earlier definition wins";
}

// Section 13.1's cost rule, word for word: cost = bytes * (1 + reloads), and
// the **smallest** cost is spilled. The temptation is to spill the largest
// buffer, and this test is what stops somebody restoring that instinct.
TEST(SpillHeuristic, CostSpillsTheSmallestCost) {
  // 512 * (1 + 3) = 2048, 256 * (1 + 1) = 512, 1024 * (1 + 0) = 1024.
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 20, 512}, 3, true}, {{1, 9, 256}, 1, true}, {{2, 6, 1024}, 0, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::Cost), 1);
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::LongestRange), 0)
      << "and the other rule picks a different buffer on the same input, which "
         "is what makes the option worth having";
}

// A buffer with many reloads is expensive to spill even when it is small, which
// is the whole content of the Belady style rule.
TEST(SpillHeuristic, CostCountsTheReloads) {
  // 64 * (1 + 9) = 640 against 512 * (1 + 0) = 512.
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 30, 64}, 9, true}, {{1, 5, 512}, 0, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::Cost), 1)
      << "the small buffer read nine more times costs more than the large one "
         "read none";
}

TEST(SpillHeuristic, CostBreaksACostTieByLargerBytes) {
  // 512 * (1 + 1) = 1024 and 1024 * (1 + 0) = 1024.
  llvm::SmallVector<SpillCandidate> candidates{
      {{0, 9, 512}, 1, true}, {{1, 10, 1024}, 0, true}};
  EXPECT_EQ(chooseSpillVictim(candidates, SpillHeuristic::Cost), 1)
      << "equal cost, so the larger buffer wins, because it frees more room";
}

TEST(SpillHeuristic, CostBreaksAByteTieBySpanThenDefinition) {
  // Equal cost and equal bytes, so the longer span wins.
  llvm::SmallVector<SpillCandidate> spanTie{{{0, 4, 512}, 1, true},
                                            {{2, 20, 512}, 1, true}};
  EXPECT_EQ(chooseSpillVictim(spanTie, SpillHeuristic::Cost), 1);

  // Equal on all three, so the earlier definition wins.
  llvm::SmallVector<SpillCandidate> definitionTie{{{9, 19, 512}, 1, true},
                                                  {{2, 12, 512}, 1, true}};
  EXPECT_EQ(chooseSpillVictim(definitionTie, SpillHeuristic::Cost), 1);
}

// The choice does not depend on the order the candidates arrive in, which is
// what "deterministic" has to mean in practice: the caller builds the candidate
// list by walking the IR, and a rule that preferred whichever equal candidate
// came first would move every published number the day an unrelated pass
// reordered two allocations.
TEST(SpillHeuristic, TheChoiceDoesNotDependOnTheCandidateOrder) {
  llvm::SmallVector<SpillCandidate> forwards{
      {{0, 20, 512}, 3, true}, {{1, 9, 256}, 1, true}, {{2, 6, 1024}, 0, true}};
  llvm::SmallVector<SpillCandidate> backwards{
      {{2, 6, 1024}, 0, true}, {{1, 9, 256}, 1, true}, {{0, 20, 512}, 3, true}};

  for (SpillHeuristic heuristic :
       {SpillHeuristic::LongestRange, SpillHeuristic::Cost}) {
    std::optional<int64_t> first = chooseSpillVictim(forwards, heuristic);
    std::optional<int64_t> second = chooseSpillVictim(backwards, heuristic);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(forwards[*first].interval.definition,
              backwards[*second].interval.definition)
        << "under " << spillHeuristicName(heuristic).str()
        << " the same candidate must win from either direction";
  }
}

//===----------------------------------------------------------------------===//
// 5. The option strings.
//===----------------------------------------------------------------------===//

TEST(Options, TheAcceptedValuesParse) {
  EXPECT_EQ(parseStrategy("pack"), Strategy::Pack);
  EXPECT_EQ(parseStrategy("interval"), Strategy::Interval);
  EXPECT_EQ(parseSpillHeuristic("longest-range"), SpillHeuristic::LongestRange);
  EXPECT_EQ(parseSpillHeuristic("cost"), SpillHeuristic::Cost);
}

TEST(Options, AnythingElseIsRefused) {
  EXPECT_FALSE(parseStrategy("greedy"));
  EXPECT_FALSE(parseStrategy("Pack")) << "the values are case sensitive";
  EXPECT_FALSE(parseStrategy(""));
  EXPECT_FALSE(parseSpillHeuristic("belady"));
  EXPECT_FALSE(parseSpillHeuristic("longest_range"))
      << "the option is spelt with a hyphen and only with a hyphen";
}

// The listed values are the values that parse. A diagnostic that offered an
// option the parser rejects would send somebody round the loop twice.
TEST(Options, TheDiagnosticListsExactlyWhatParses) {
  EXPECT_TRUE(parseStrategy(strategyName(Strategy::Pack)));
  EXPECT_TRUE(parseStrategy(strategyName(Strategy::Interval)));
  EXPECT_TRUE(
      parseSpillHeuristic(spillHeuristicName(SpillHeuristic::LongestRange)));
  EXPECT_TRUE(parseSpillHeuristic(spillHeuristicName(SpillHeuristic::Cost)));

  EXPECT_NE(strategyOptions().find(strategyName(Strategy::Pack)),
            llvm::StringRef::npos);
  EXPECT_NE(strategyOptions().find(strategyName(Strategy::Interval)),
            llvm::StringRef::npos);
  EXPECT_NE(spillHeuristicOptions().find(
                spillHeuristicName(SpillHeuristic::LongestRange)),
            llvm::StringRef::npos);
  EXPECT_NE(
      spillHeuristicOptions().find(spillHeuristicName(SpillHeuristic::Cost)),
      llvm::StringRef::npos);
}

// The default budget is one mebibyte, and it is asserted here rather than left
// as a constant nobody reads. Section 15 measures every tight budget as a
// fraction of the peak observed at the default, so moving this number silently
// would move every tight budget cell in the project's history at once.
TEST(Options, TheDefaultBudgetIsOneMebibyte) {
  EXPECT_EQ(kDefaultScratchpadBudget, 1024 * 1024);
  EXPECT_EQ(kDefaultAlignment, 64);
}

} // namespace
