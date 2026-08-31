//===- CostModelTest.cpp - the two timelines and the charges --*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 5.5, asserted rather than described.
//
// Four claims are made here and each one is the kind that quietly stops being
// true: the constants are the ones the header says and the Python mirror
// repeats; a dependent chain serializes; two independent streams overlap and
// `overlap_fraction` reaches its endpoints rather than hovering near a half;
// and the single port flag reproduces the sum, which is what makes any number
// published under the simpler model still reproducible.
//
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "NPU/Simulator/CostModel.h"

#include "gtest/gtest.h"

#include <vector>

using namespace nbin;
using namespace npusim;

namespace {

//===----------------------------------------------------------------------===//
// The constants.
//===----------------------------------------------------------------------===//

TEST(FrozenConstants, TheCostModelsNumbers) {
  // Every number of Section 5.5, written out. This test exists for the same
  // reason the format's frozen constants test does: these are assumptions the
  // report quotes, and an assumption that moved without anybody noticing would
  // move every published cycle count with it.
  //
  // `test/Python/test_cost_model_mirror.py` asserts the Python mirror carries
  // the same names and the same values, so between the two there is no way to
  // change one copy and not the other.
  EXPECT_EQ(kArrayDim, 16);
  EXPECT_EQ(kPeakMacsPerCycleF32, 256);
  EXPECT_EQ(kPeakMacsPerCycleI8, 1024);
  EXPECT_DOUBLE_EQ(kDramBandwidthBytesPerCycle, 16.0);
  EXPECT_DOUBLE_EQ(kDmaDescriptorCycles, 64.0);
  EXPECT_DOUBLE_EQ(kDmaStridedElementCycles, 0.5);
  EXPECT_EQ(kElementwiseLaneWidth, 16);
  EXPECT_DOUBLE_EQ(kIssueOverheadCycles, 4.0);
  EXPECT_DOUBLE_EQ(kWeightPreloadCycles, 16.0);

  // The array is square and one MAC per processing element per cycle, so the
  // f32 peak is the array's area. The two are separate constants on purpose;
  // this asserts they still agree.
  EXPECT_EQ(kPeakMacsPerCycleF32, kArrayDim * kArrayDim);
  // Four int8 multiplies packed per f32 lane.
  EXPECT_EQ(kPeakMacsPerCycleI8, 4 * kPeakMacsPerCycleF32);
}

//===----------------------------------------------------------------------===//
// The charges.
//===----------------------------------------------------------------------===//

TEST(CostModel, TheDmaChargeHasThreeTerms) {
  // bytes over bandwidth, plus the descriptor, plus the stride penalty.
  //
  //   64 bytes at 16 bytes per cycle is 4 cycles, plus 64 for the descriptor.
  EXPECT_DOUBLE_EQ(dmaCycles(64, 16, 1), 4.0 + 64.0);
  //   the same transfer with a stride of two adds half a cycle per element.
  EXPECT_DOUBLE_EQ(dmaCycles(64, 16, 2), 4.0 + 64.0 + 16.0 * 0.5);
  // A stride of zero is not a unit stride either: a broadcast read still cannot
  // be coalesced into a burst.
  EXPECT_GT(dmaCycles(64, 16, 0), dmaCycles(64, 16, 1));
}

TEST(CostModel, MacsAreRawAndTheOccupancyIsBeside) {
  // The MAC count is the shape's, whatever the array did with it.
  EXPECT_EQ(gemmCharge(4, 16, 16, kPeakMacsPerCycleF32).macs, 4 * 16 * 16);
  EXPECT_EQ(gemmCharge(7, 3, 5, kPeakMacsPerCycleF32).macs, 7 * 3 * 5);
  // Folded across several tiles the count is still the product, which is what
  // makes it reconstructible from the shape rather than from the model.
  EXPECT_EQ(gemmCharge(4, 40, 33, kPeakMacsPerCycleF32).macs, 4 * 40 * 33);

  // A tile that fills the array occupies all of it.
  const ComputeCharge full = gemmCharge(1024, 16, 16, kPeakMacsPerCycleF32);
  EXPECT_DOUBLE_EQ(full.utilization, 1.0);
  // And a long stream of activation rows amortises the preload almost away.
  EXPECT_GT(full.delta, 0.98);

  // A single column tile occupies one sixteenth of one sixteenth.
  const ComputeCharge narrow = gemmCharge(1024, 1, 1, kPeakMacsPerCycleF32);
  EXPECT_DOUBLE_EQ(narrow.utilization, 1.0 / 256.0);
}

TEST(CostModel, ANarrowTileCostsMorePerMac) {
  // The direction of the correction, which Section 5.5 is emphatic about: a
  // tile occupying a quarter of the array takes four times as long as its MAC
  // count alone suggests. Written the other way round the formula would claim a
  // small tile does more work, which is the opposite of the effect.
  const ComputeCharge full = gemmCharge(256, 16, 16, kPeakMacsPerCycleF32);
  const ComputeCharge half = gemmCharge(256, 16, 8, kPeakMacsPerCycleF32);

  const double fullPerMac = full.cycles / static_cast<double>(full.macs);
  const double halfPerMac = half.cycles / static_cast<double>(half.macs);
  EXPECT_GT(halfPerMac, fullPerMac);
  // Half the columns is half the occupancy, so twice the cycles per MAC.
  EXPECT_NEAR(halfPerMac / fullPerMac, 2.0, 1e-9);

  // `effectiveMacs` is what the cycles imply at peak, and it is never below the
  // raw count. Nothing in the energy path sees it.
  EXPECT_GE(full.effectiveMacs, static_cast<double>(full.macs));
  EXPECT_GT(half.effectiveMacs, static_cast<double>(half.macs));
}

TEST(CostModel, OverlapFractionReachesItsEndpoints) {
  // Fully serialized: the total is the sum, so nothing was hidden.
  EXPECT_DOUBLE_EQ(overlapFraction(100.0, 40.0, 140.0), 0.0);
  // Perfect overlap: the total is the longer timeline, so the whole of the
  // shorter one was hidden underneath it.
  EXPECT_DOUBLE_EQ(overlapFraction(100.0, 40.0, 100.0), 1.0);
  // Halfway.
  EXPECT_DOUBLE_EQ(overlapFraction(100.0, 40.0, 120.0), 0.5);
  // A program with nothing on one port reports zero rather than claiming
  // perfect overlap of a timeline that does not exist.
  EXPECT_DOUBLE_EQ(overlapFraction(0.0, 40.0, 40.0), 0.0);

  // The naive form this one replaces, one minus total over the sum, would read
  // 0.286 for the perfectly overlapped case above and could never reach 1.
  // Asserting the difference is what stops somebody simplifying it back.
  EXPECT_NE(overlapFraction(100.0, 40.0, 100.0), 1.0 - 100.0 / 140.0);
}

//===----------------------------------------------------------------------===//
// The two timelines, over real programs.
//===----------------------------------------------------------------------===//

/// A chain in which every instruction waits for the one before it.
///
/// The alternation matters and it is the point of the shape: the program ends
/// on the compute port, so the `HALT` cannot hide underneath a transfer that is
/// still running. A chain that ended with a `DMA_STORE` would leave the four
/// cycle issue overhead of the `HALT` overlapping it, and the fraction would
/// read a little under a half for a program that is serialized in every way
/// that matters.
Program buildSerializedChain() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, -2, 3, -4});
  const int64_t first = builder.scratch(4);
  const int64_t second = builder.scratch(4);
  const int64_t third = builder.scratch(4);
  const int64_t sink = builder.output(shape);

  builder.add(dmaLoad(first, shape, at(MemSpace::Dram, source, shape)));
  builder.add(compute(Opcode::RELU, second, shape,
                      {at(MemSpace::Scratchpad, first, shape)}));
  builder.add(dmaStore(sink, shape, at(MemSpace::Scratchpad, second, shape)));
  // Reading the output region back is legal: the store above wrote it, so it is
  // defined. It is what makes the next compute instruction wait for the
  // transfer rather than start beside it.
  builder.add(dmaLoad(third, shape, at(MemSpace::Dram, sink, shape)));
  builder.add(compute(Opcode::RELU, first, shape,
                      {at(MemSpace::Scratchpad, third, shape)}));
  builder.add(halt());

  // Scratchpad: three buffers of 4 f32 elements. 12 * 4 = 48 bytes.
  return builder.finish(48);
}

TEST(Timelines, DependencySerializesAndOverlapReadsZero) {
  Harness harness(buildSerializedChain());
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_TRUE(result.reachedHalt);
  EXPECT_GT(result.stats.dmaCycles, 0.0);
  EXPECT_GT(result.stats.computeCycles, 0.0);

  // Nothing ran beside anything, so the total is the sum of the two timelines.
  EXPECT_DOUBLE_EQ(result.stats.cycles,
                   result.stats.dmaCycles + result.stats.computeCycles);
  EXPECT_DOUBLE_EQ(result.stats.overlapFraction, 0.0);
}

/// A program whose two ports have nothing to say to each other.
///
/// One transfer brings a buffer on chip. After that the compute instructions
/// all read that one buffer and write buffers nobody loads, and the remaining
/// transfers all read constants and write buffers nobody computes on. The DMA
/// timeline is much the longer of the two, so the whole compute timeline hides
/// underneath it and the fraction reads exactly one.
Program buildOverlappedProgram() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, -2, 3, -4});
  const int64_t loaded = builder.scratch(4);

  constexpr int kTransfers = 20;
  constexpr int kComputes = 5;

  std::vector<int64_t> spare;
  spare.reserve(kTransfers);
  for (int index = 0; index < kTransfers; ++index)
    spare.push_back(builder.scratch(4));
  std::vector<int64_t> results;
  results.reserve(kComputes);
  for (int index = 0; index < kComputes; ++index)
    results.push_back(builder.scratch(4));

  builder.add(dmaLoad(loaded, shape, at(MemSpace::Dram, source, shape)));
  for (int index = 0; index < kTransfers; ++index)
    builder.add(
        dmaLoad(spare[index], shape, at(MemSpace::Dram, source, shape)));
  for (int index = 0; index < kComputes; ++index)
    builder.add(compute(Opcode::RELU, results[index], shape,
                        {at(MemSpace::Scratchpad, loaded, shape)}));
  builder.add(halt());

  // Scratchpad: 1 loaded buffer, 20 spare transfer targets and 5 results, each
  // 4 f32 elements. 26 * 4 * 4 = 416 bytes.
  return builder.finish(416);
}

TEST(Timelines, IndependentStreamsOverlapCompletely) {
  Harness harness(buildOverlappedProgram());
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_GT(result.stats.dmaCycles, result.stats.computeCycles);

  // The DMA timeline is the longer one and it never waits, so the total is
  // exactly it and the whole compute timeline is hidden.
  EXPECT_DOUBLE_EQ(result.stats.cycles, result.stats.dmaCycles);
  EXPECT_NEAR(result.stats.overlapFraction, 1.0, 1e-9);
}

TEST(Timelines, SinglePortReproducesTheSum) {
  // Section 5.5 keeps the single port model behind a flag so that any number
  // published under it stays reproducible. Under the flag there is one clock,
  // so the total is the sum of what was placed on the two ports, and the
  // overlap fraction is zero because nothing overlapped anything.
  Harness harness(buildOverlappedProgram());
  SimOptions options;
  options.singlePort = true;
  const SimResult result = harness.run(options);

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_DOUBLE_EQ(result.stats.cycles,
                   result.stats.dmaCycles + result.stats.computeCycles);
  EXPECT_DOUBLE_EQ(result.stats.overlapFraction, 0.0);

  // And it is strictly worse than the two port total, which is the whole reason
  // the two port model exists.
  Harness twoPort(buildOverlappedProgram());
  const SimResult overlapped = twoPort.run();
  ASSERT_TRUE(overlapped.ok()) << overlapped.error.value_or("");
  EXPECT_GT(result.stats.cycles, overlapped.stats.cycles);

  // The work itself did not change: the two runs place the same charges on the
  // same ports and differ only in whether the ports are one clock or two.
  EXPECT_DOUBLE_EQ(result.stats.dmaCycles, overlapped.stats.dmaCycles);
  EXPECT_DOUBLE_EQ(result.stats.computeCycles, overlapped.stats.computeCycles);
  EXPECT_EQ(result.stats.instructions, overlapped.stats.instructions);
}

TEST(Timelines, EveryInstructionPaysTheIssueOverhead) {
  // The fixed per instruction overhead of Section 5.5, isolated: a program of
  // nothing but control instructions costs exactly the overhead times the
  // count. It is what makes a program of a thousand tiny instructions cost
  // something rather than nothing.
  Builder builder;
  for (int index = 0; index < 10; ++index)
    builder.add(nop());
  builder.add(halt());

  // Scratchpad: nothing at all, because nothing here touches memory.
  Harness harness(builder.finish(0));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_EQ(result.stats.instructions, 11u);
  EXPECT_DOUBLE_EQ(result.stats.cycles, 11.0 * kIssueOverheadCycles);
}

} // namespace
