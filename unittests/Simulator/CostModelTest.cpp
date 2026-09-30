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
  // *Added at P14 with the INT8 terms.* The packing factor the int8 peak has
  // always been written as a multiple of, now a constant of its own.
  EXPECT_EQ(kI8MacsPerLane, 4);

  // The array is square and one MAC per processing element per cycle, so the
  // f32 peak is the array's area. The two are separate constants on purpose;
  // this asserts they still agree.
  EXPECT_EQ(kPeakMacsPerCycleF32, kArrayDim * kArrayDim);
  // Four int8 multiplies packed per f32 lane.
  EXPECT_EQ(kPeakMacsPerCycleI8, kI8MacsPerLane * kPeakMacsPerCycleF32);
}

TEST(CostModel, TheInt8PackingIsExactlyTheRatioOfThePeaks) {
  // What `SimOptions::int8AtF32Peak` rests on: the charge at the int8 peak,
  // multiplied by the packing factor, is the charge at the f32 peak, **bit for
  // bit**. Every term of the charge is divided by the peak once and the two
  // peaks differ by a power of two, so no rounding separates them. Asserted
  // over narrow tiles, folded ones and a depthwise convolution, which are the
  // shapes whose charge has the most terms.
  const int64_t shapes[][3] = {
      {4, 16, 16}, {7, 3, 5}, {4, 40, 33}, {1024, 1, 1}, {25, 400, 120}};
  for (const auto &shape : shapes) {
    const ComputeCharge f32 =
        gemmCharge(shape[0], shape[1], shape[2], kPeakMacsPerCycleF32);
    const ComputeCharge i8 =
        gemmCharge(shape[0], shape[1], shape[2], kPeakMacsPerCycleI8);
    EXPECT_EQ(f32.cycles, i8.cycles * static_cast<double>(kI8MacsPerLane));
    EXPECT_EQ(f32.macs, i8.macs);
    EXPECT_EQ(f32.effectiveMacs, i8.effectiveMacs);
  }
  const ComputeCharge depthwiseF32 =
      conv2dCharge(1, 8, 8, 8, 8, 3, 3, 8, kPeakMacsPerCycleF32);
  const ComputeCharge depthwiseI8 =
      conv2dCharge(1, 8, 8, 8, 8, 3, 3, 8, kPeakMacsPerCycleI8);
  EXPECT_EQ(depthwiseF32.cycles,
            depthwiseI8.cycles * static_cast<double>(kI8MacsPerLane));
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

/// The number of weight tiles a `k` by `n` matrix folds into on this array.
///
/// Written out here rather than taken from the model, because a test that asked
/// the model how many folds it thought there were would agree with itself.
int64_t foldCount(int64_t k, int64_t n) {
  const int64_t dim = kArrayDim;
  return ((k + dim - 1) / dim) * ((n + dim - 1) / dim);
}

TEST(CostModel, TheWeightPreloadIsChargedOncePerFold) {
  // **This test exists because D-0045 said the opposite.** That entry recorded
  // that `gemmCharge` "computes delta once per instruction and applies it to
  // every tile, so the sixteen cycle pipeline fill is amortised across the whole
  // layer no matter how many times the array is actually refilled". The premise
  // is true and the conclusion does not follow from it, which is why the claim
  // survived a reading of the code.
  //
  // Applying the same *fraction* to every tile does not charge the fill once. At
  // the f32 peak the array's area and the peak are the same number, so
  // `utilization * peak` is exactly `rows * columns` and a tile's charge reduces
  // to
  //
  //     tileMacs / (utilization * delta * peak) = m / delta = m + kWeightPreloadCycles
  //
  // for every tile, whole or partial. With `T` folds the instruction is charged
  // `T * (m + kWeightPreloadCycles)`, which is the fill counted `T` times: once
  // per refill, which is what a weight stationary array does and what SCALE-Sim
  // charges.
  //
  // The two accountings are asserted apart rather than only the right one
  // asserted, because they agree whenever there is exactly one fold, and every
  // shape small enough to check by hand has exactly one fold.
  const int64_t peak = kPeakMacsPerCycleF32;

  struct Case {
    int64_t m, k, n;
  };
  // The last two are D-0045's own reproduction shape and the narrow tail of the
  // fully connected layer Section 5.5 names.
  const Case cases[] = {{64, 16, 16},  {64, 32, 16}, {196, 27, 6},
                        {1024, 9, 1},  {64, 72, 8},  {16, 256, 120}};

  for (const Case &shape : cases) {
    const ComputeCharge charge =
        gemmCharge(shape.m, shape.k, shape.n, peak);
    const int64_t folds = foldCount(shape.k, shape.n);

    const double perFold =
        static_cast<double>(folds) *
        (static_cast<double>(shape.m) + kWeightPreloadCycles);
    const double oncePerInstruction =
        static_cast<double>(folds) * static_cast<double>(shape.m) +
        kWeightPreloadCycles;

    EXPECT_NEAR(charge.cycles, perFold, 1e-9 * perFold)
        << "m=" << shape.m << " k=" << shape.k << " n=" << shape.n
        << ": the charge stopped being the per fold accounting";

    if (folds > 1) {
      // The discriminating half. Without it this test would pass against a
      // model that had been changed to charge the fill once.
      EXPECT_GT(charge.cycles, oncePerInstruction)
          << "m=" << shape.m << " k=" << shape.k << " n=" << shape.n
          << ": the charge became the once per instruction accounting";
      EXPECT_NEAR(charge.cycles - oncePerInstruction,
                  static_cast<double>(folds - 1) * kWeightPreloadCycles,
                  1e-9 * perFold);
    }
  }
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

/// An integer matrix multiplication of 16 rows against a 32 by 24 weight
/// matrix, loaded, computed and stored.
///
/// Its charge folds the weights into two by two tiles, one of them sixteen
/// columns wide and one eight, so the packing assumption applies to a charge
/// with more than one term in it.
Program buildIntegerMatmul() {
  std::vector<int8_t> lhs(16 * 32);
  for (size_t index = 0; index < lhs.size(); ++index)
    lhs[index] = static_cast<int8_t>(static_cast<int>(index % 17) - 8);
  std::vector<int8_t> rhs(32 * 24);
  for (size_t index = 0; index < rhs.size(); ++index)
    rhs[index] = static_cast<int8_t>(static_cast<int>(index % 13) - 6);

  Builder builder;
  const std::vector<int64_t> lhsShape = {16, 32};
  const std::vector<int64_t> rhsShape = {32, 24};
  const std::vector<int64_t> resultShape = {16, 24};

  const int64_t lhsRegion = builder.constantI8(lhsShape, lhs);
  const int64_t rhsRegion = builder.constantI8(rhsShape, rhs);
  const int64_t lhsBuffer = builder.scratch(16 * 32, ElemType::I8);
  const int64_t rhsBuffer = builder.scratch(32 * 24, ElemType::I8);
  const int64_t resultBuffer = builder.scratch(16 * 24, ElemType::I8);
  const int64_t sink = builder.output(resultShape, ElemType::I8);

  builder.add(dmaLoad(lhsBuffer, lhsShape,
                      at(MemSpace::Dram, lhsRegion, lhsShape, ElemType::I8)));
  builder.add(dmaLoad(rhsBuffer, rhsShape,
                      at(MemSpace::Dram, rhsRegion, rhsShape, ElemType::I8)));
  Instruction matmul =
      compute(Opcode::MATMUL, resultBuffer, resultShape,
              {at(MemSpace::Scratchpad, lhsBuffer, lhsShape, ElemType::I8),
               at(MemSpace::Scratchpad, rhsBuffer, rhsShape, ElemType::I8)},
              ElemType::I8);
  matmul.requantMultiplier = kHalfMultiplier;
  matmul.requantShift = 4;
  builder.add(std::move(matmul));
  builder.add(dmaStore(
      sink, resultShape,
      at(MemSpace::Scratchpad, resultBuffer, resultShape, ElemType::I8)));
  builder.add(halt());

  // Scratchpad: 512 + 768 + 384 one byte elements, 1664 bytes.
  return builder.finish(1664);
}

TEST(Timelines, TheInt8PackingIsSeparable) {
  // Section 14's gate asks for the int8 throughput assumption's share of a
  // cycle win separately from the DMA traffic reduction's, and this is the
  // mechanism: the same program run twice, once as the machine is modelled
  // and once with every int8 MAC charged at the f32 peak.
  Harness packed(buildIntegerMatmul());
  const SimResult withPacking = packed.run();
  SimOptions options;
  options.int8AtF32Peak = true;
  Harness unpacked(buildIntegerMatmul());
  const SimResult withoutPacking = unpacked.run(options);
  ASSERT_TRUE(withPacking.ok()) << withPacking.error.value_or("");
  ASSERT_TRUE(withoutPacking.ok()) << withoutPacking.error.value_or("");

  // Only the integer instruction's charge moves, and by exactly what the
  // packing bought: three quarters of the f32 charge, which is three times the
  // int8 one.
  const ComputeCharge charge = gemmCharge(16, 32, 24, kPeakMacsPerCycleI8);
  EXPECT_EQ(withPacking.stats.int8Macs, 16u * 32u * 24u);
  EXPECT_DOUBLE_EQ(withoutPacking.stats.computeCycles -
                       withPacking.stats.computeCycles,
                   charge.cycles * static_cast<double>(kI8MacsPerLane - 1));
  EXPECT_GT(withoutPacking.stats.cycles, withPacking.stats.cycles);

  // And nothing else: the work is the same work.
  EXPECT_DOUBLE_EQ(withoutPacking.stats.dmaCycles, withPacking.stats.dmaCycles);
  EXPECT_EQ(withoutPacking.stats.instructions, withPacking.stats.instructions);
  EXPECT_EQ(withoutPacking.stats.macs, withPacking.stats.macs);
  EXPECT_EQ(withoutPacking.stats.int8Macs, withPacking.stats.int8Macs);
  EXPECT_DOUBLE_EQ(withoutPacking.stats.effectiveMacs,
                   withPacking.stats.effectiveMacs);
  EXPECT_EQ(withoutPacking.stats.dramBytesRead,
            withPacking.stats.dramBytesRead);
  EXPECT_EQ(withoutPacking.stats.scratchpadElementsRead,
            withPacking.stats.scratchpadElementsRead);

  // An f32 program has no int8 MAC to charge differently, so the option leaves
  // every figure it produces exactly as it was.
  Harness plain(buildOverlappedProgram());
  const SimResult f32 = plain.run();
  Harness optioned(buildOverlappedProgram());
  const SimResult f32Optioned = optioned.run(options);
  ASSERT_TRUE(f32.ok() && f32Optioned.ok());
  EXPECT_DOUBLE_EQ(f32Optioned.stats.cycles, f32.stats.cycles);
  EXPECT_DOUBLE_EQ(f32Optioned.stats.computeCycles, f32.stats.computeCycles);
  EXPECT_DOUBLE_EQ(f32Optioned.stats.dmaCycles, f32.stats.dmaCycles);
}

TEST(CostModel, AStridedMoveCostsMoreThanThePermutationThatAvoidsIt) {
  // **The whole of `-npu-assign-layout`'s decision, as a relation between two
  // constants.** Section 5.5 charges layout in exactly one place, the non unit
  // innermost stride penalty, and it charges it to NHWC and never to NCHW: an
  // NHWC tensor is materialised as a buffer at NCHW extents with permuted
  // strides, so its innermost stride is the channel count. The alternative to
  // paying that penalty is to perform the permutation, which is one elementwise
  // pass. The layout question therefore reduces to a per element race between
  // `kDmaStridedElementCycles` and `1 / kElementwiseLaneWidth`.
  //
  // It is asserted here, in the file that owns both constants, rather than
  // beside the pass, because the conclusion the report publishes is about the
  // machine and not about the pass: recalibrating either constant has to fail a
  // test rather than quietly reverse a published answer.
  const double stridedPerElement = kDmaStridedElementCycles;
  const double permutePerElement =
      1.0 / static_cast<double>(kElementwiseLaneWidth);

  EXPECT_GT(stridedPerElement, permutePerElement);
  EXPECT_DOUBLE_EQ(stridedPerElement / permutePerElement, 8.0);

  // It is a ratio and not a threshold, so no extent reverses it. A transfer's
  // other two terms are the bytes and the fixed descriptor cost, and both are
  // charged whichever layout the buffer is in, so they cancel out of the
  // comparison rather than tipping it at some size.
  for (int64_t elements : {int64_t{1}, int64_t{16}, int64_t{90}, int64_t{1024},
                           int64_t{1} << 20}) {
    const double strided =
        dmaCycles(elements * 4, elements, /*innermostStride=*/3);
    const double contiguous =
        dmaCycles(elements * 4, elements, /*innermostStride=*/1);
    EXPECT_DOUBLE_EQ(strided - contiguous,
                     static_cast<double>(elements) * kDmaStridedElementCycles);
    EXPECT_GT(strided - contiguous, elementwiseCycles(elements));
  }
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
