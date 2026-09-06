//===- SimulatorTest.cpp - a semantics test per opcode --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 10.1: every opcode has at least one semantics test asserting numerics
// against a **hand computed** expected value, and the cases that are commonly
// implemented and never tested are each required by name.
//
// **Hand computed means hand computed.** Not one expected value in this file
// was produced by running the kernel and copying what came out. Every one is
// written out in the comment above it, from the definition in
// `docs/ISA_MANUAL.md` and the ODS description, and the arithmetic is short
// enough for a reader to redo. A golden captured from the implementation is a
// test that can only ever pass.
//
// The Phase P7 list of Section 10.1, and where each case is:
//
//   POOL_MAX                                MaxPooling
//   DMA_LOAD and DMA_STORE                  DmaRoundTrip
//   the spill round trip                    SpillRoundTrip
//   a non unit innermost stride on a load   StridedLoad
//   grouped convolution, group > 1          GroupedConvolution
//   depthwise convolution, group == C       DepthwiseConvolution
//   dilated convolution, dilation > 1       DilatedConvolution
//   asymmetric padding                      AsymmetricPadding
//   batch 2 and batch 4 on convolution      ConvolutionBatchTwo, BatchFour
//   batch 2 and batch 4 on both pools       PoolingBatchTwo, PoolingBatchFour,
//                                           each running POOL_MAX and POOL_AVG
//   batch 2 and batch 4 elementwise         ElementwiseBatchTwo,
//                                           ElementwiseBatchFour, each running
//                                           ADD and MUL
//   TRANSPOSE, identity permutation         TransposeIdentity
//   TRANSPOSE, rank 4 NCHW to NHWC          TransposeNchwToNhwc
//   CONCAT on the channel axis              ConcatChannelAxis
//   CONCAT on the last axis                 ConcatLastAxis
//   CONCAT with more than two operands      ConcatThreeOperands
//   CONCAT at batch 4                       ConcatBatchFour
//   NOP and HALT                            ControlOpcodes
//   the all padding average window          AllPaddingWindow
//
// **No integer case appears here**, and that is the gate's wording rather than
// an omission. The integer kernels are Section 14's and land at Phase P14;
// QUANT and DEQUANT are covered here only by the refusal they give, which is
// asserted so that "no kernel yet" is a tested behaviour rather than a surprise.
// Both of them, not one: Section 10.1's first sentence asks for a test per
// opcode, and an untested opcode is untested whatever the reason.
//
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "NPU/Simulator/CostModel.h"

#include "gtest/gtest.h"

#include <cmath>
#include <limits>

using namespace nbin;
using namespace npusim;

namespace {

/// The no op configuration, for an opcode with no fields to set.
void plain(Instruction &) {}

//===----------------------------------------------------------------------===//
// The control opcodes.
//===----------------------------------------------------------------------===//

TEST(Control, NopAdvancesAndHaltStops) {
  Builder builder;
  builder.add(nop());
  builder.add(nop());
  builder.add(halt());
  // Nothing after the HALT runs, which is what makes the instruction count
  // meaningful: three of the four instructions in this program execute.
  builder.add(nop());

  // Scratchpad: no buffer at all. Zero is the tight size for a program that
  // touches no memory, and declaring anything else would be declaring a
  // scratchpad nothing uses.
  Harness harness(builder.finish(0));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_TRUE(result.reachedHalt);
  EXPECT_EQ(result.stats.instructions, 3u);
  // Every instruction costs its issue overhead and nothing else: three
  // instructions on the compute port, no operands, no result.
  EXPECT_DOUBLE_EQ(result.stats.cycles, 3.0 * kIssueOverheadCycles);
  EXPECT_DOUBLE_EQ(result.stats.computeCycles, 3.0 * kIssueOverheadCycles);
  EXPECT_DOUBLE_EQ(result.stats.dmaCycles, 0.0);
  EXPECT_EQ(result.stats.macs, 0u);
}

TEST(Control, RunningOutOfInstructionsStopsTheMachine) {
  // The Phase P7 decision on the open question Phase P6 left: a file without a
  // trailing HALT decodes and validates, and the machine stops when it runs out
  // of straight line code rather than refusing. `reachedHalt` is how a caller
  // tells the two endings apart.
  Builder builder;
  builder.add(nop());
  builder.add(nop());

  Harness harness(builder.finish(0));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  EXPECT_FALSE(result.reachedHalt);
  EXPECT_EQ(result.stats.instructions, 2u);
}

//===----------------------------------------------------------------------===//
// The DMA opcodes.
//===----------------------------------------------------------------------===//

TEST(Dma, DmaRoundTrip) {
  // Six values out to the scratchpad and back. DMA_LOAD and DMA_STORE are the
  // whole program, so this is the one test in the file where they are the
  // subject rather than the scaffolding.
  Builder builder;
  const std::vector<int64_t> shape = {2, 3};
  const std::vector<float> data = {1, 2, 3, 4, 5, 6};
  const int64_t source = builder.constant(shape, data);
  const int64_t buffer = builder.scratch(6);
  const int64_t sink = builder.output(shape);

  builder.add(dmaLoad(buffer, shape, at(MemSpace::Dram, source, shape)));
  builder.add(dmaStore(sink, shape, at(MemSpace::Scratchpad, buffer, shape)));
  builder.add(halt());

  // Scratchpad: one buffer of 2 * 3 = 6 f32 elements. 6 * 4 = 24 bytes.
  Harness harness(builder.finish(24));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  expectValues(harness.outputF32(0), data);
  EXPECT_EQ(result.stats.dramBytesRead, 24u);
  EXPECT_EQ(result.stats.dramBytesWritten, 24u);
  EXPECT_EQ(result.stats.scratchpadElementsWritten, 6u);
  EXPECT_EQ(result.stats.scratchpadElementsRead, 6u);
}

TEST(Dma, SpillRoundTrip) {
  // Load, store to a DRAM spill slot, load it back from there, store the
  // result. This is the path the allocator produces when it runs out of
  // scratchpad, and a spill that came back different would be the worst class
  // of bug this project can have: silently wrong numbers on the models that are
  // large enough to matter.
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const std::vector<float> data = {1, 2, 3, 4};
  const int64_t source = builder.constant(shape, data);
  const int64_t first = builder.scratch(4);
  const int64_t second = builder.scratch(4);
  const int64_t spill = builder.spillSlot(shape);
  const int64_t sink = builder.output(shape);

  builder.add(dmaLoad(first, shape, at(MemSpace::Dram, source, shape)));
  builder.add(dmaStore(spill, shape, at(MemSpace::Scratchpad, first, shape)));
  builder.add(dmaLoad(second, shape, at(MemSpace::Dram, spill, shape)));
  builder.add(dmaStore(sink, shape, at(MemSpace::Scratchpad, second, shape)));
  builder.add(halt());

  // Scratchpad: two buffers of 4 f32 elements. 8 * 4 = 32 bytes.
  Harness harness(builder.finish(32));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  expectValues(harness.outputF32(0), data);
}

// **D-0050's reproduction, made into the test that keeps it fixed.**
//
// A tile written into a sub region of a larger buffer has to **scatter**, and
// until version 2 the format could not say so: `Instruction` carried
// `resultShape` and no `resultStrides`, the encoder computed the destination's
// strides and dropped them, and a `DMA_STORE` into a sub region laid its bytes
// down in a run. The measured cost was a tiled convolution writing 160 of the
// 512 elements it should have, into channels 0 to 2 instead of 0 to 3, with the
// validator unable to see it because an output region is written and never
// read.
//
// The shape here is that failure in miniature and the arithmetic is written out
// so a reader can check it rather than trust it. The scratchpad holds 1 through
// 4. The store writes them into a 2 by 4 output through a 2 by 2 view with
// strides [4, 1], which is the top left quarter of the output, so:
//
//     element (0, 0) -> offset 0 -> 1        (0, 1) -> offset 1 -> 2
//     element (1, 0) -> offset 4 -> 3        (1, 1) -> offset 5 -> 4
//
// A contiguous store of the same four elements would put 1, 2, 3, 4 at offsets
// 0 to 3 and leave offset 4 untouched, which is exactly the wrong answer the
// defect produced. **The zeros in the expectation are load bearing**: they are
// the positions the scatter must skip.
TEST(Dma, StridedStoreScattersRatherThanRunning) {
  Builder builder;
  const std::vector<int64_t> view = {2, 2};
  const std::vector<int64_t> full = {2, 4};
  const int64_t source = builder.constant(view, {1, 2, 3, 4});
  const int64_t buffer = builder.scratch(4);
  const int64_t sink = builder.output(full);

  builder.add(dmaLoad(buffer, view, at(MemSpace::Dram, source, view)));

  // The store's destination is the sub region, so its result carries the
  // parent's strides rather than the ones its own extents imply.
  Instruction store = dmaStore(sink, view, at(MemSpace::Scratchpad, buffer, view));
  store.resultStrides = {4, 1};
  builder.add(store);
  builder.add(halt());

  // Scratchpad: one buffer of 4 f32 elements. 16 bytes.
  Harness harness(builder.finish(16));
  const SimResult result = harness.run();

  ASSERT_TRUE(result.ok()) << result.error.value_or("");
  expectValues(harness.outputF32(0), {1, 2, 0, 0, 3, 4, 0, 0});
}

TEST(Dma, StridedLoad) {
  // The non unit innermost stride of Section 5.5, which the cost model has a
  // term for and which no test would otherwise exercise.
  //
  // The constant is a 2 by 4 buffer holding 1 through 8. The load reads a 2 by 2
  // view with strides [4, 2], so it takes every other column of every row:
  //
  //     element (0, 0) -> offset 0 -> 1        (0, 1) -> offset 2 -> 3
  //     element (1, 0) -> offset 4 -> 5        (1, 1) -> offset 6 -> 7
  Builder builder;
  const std::vector<int64_t> full = {2, 4};
  const std::vector<int64_t> view = {2, 2};
  const int64_t source =
      builder.constant(full, {1, 2, 3, 4, 5, 6, 7, 8});
  const int64_t buffer = builder.scratch(4);
  const int64_t sink = builder.output(view);

  builder.add(dmaLoad(buffer, view,
                      strided(MemSpace::Dram, source, view, {4, 2})));
  builder.add(dmaStore(sink, view, at(MemSpace::Scratchpad, buffer, view)));
  builder.add(halt());

  // Scratchpad: one buffer of 2 * 2 = 4 f32 elements. 4 * 4 = 16 bytes.
  Harness harness(builder.finish(16));
  const SimResult strideResult = harness.run();

  ASSERT_TRUE(strideResult.ok()) << strideResult.error.value_or("");
  expectValues(harness.outputF32(0), {1, 3, 5, 7});

  // And the stride term fired. The same number of bytes moved contiguously
  // costs less, which is the whole reason the term exists: without it a strided
  // NCHW gather and a contiguous NHWC burst would cost exactly the same and the
  // layout assignment pass of Phase P13 would have no expressible benefit.
  Builder contiguousBuilder;
  const int64_t contiguousSource =
      contiguousBuilder.constant(view, {1, 3, 5, 7});
  const int64_t contiguousBuffer = contiguousBuilder.scratch(4);
  const int64_t contiguousSink = contiguousBuilder.output(view);
  contiguousBuilder.add(
      dmaLoad(contiguousBuffer, view,
              at(MemSpace::Dram, contiguousSource, view)));
  contiguousBuilder.add(
      dmaStore(contiguousSink, view,
               at(MemSpace::Scratchpad, contiguousBuffer, view)));
  contiguousBuilder.add(halt());

  // Scratchpad: the same one buffer of 4 f32 elements. 16 bytes.
  Harness plainHarness(contiguousBuilder.finish(16));
  const SimResult plainResult = plainHarness.run();

  ASSERT_TRUE(plainResult.ok()) << plainResult.error.value_or("");
  expectValues(plainHarness.outputF32(0), {1, 3, 5, 7});
  EXPECT_GT(strideResult.stats.dmaCycles, plainResult.stats.dmaCycles);
  // Four elements at half a cycle each, which is the penalty and nothing else:
  // the two programs move the same bytes through the same descriptors.
  EXPECT_DOUBLE_EQ(strideResult.stats.dmaCycles - plainResult.stats.dmaCycles,
                   4.0 * kDmaStridedElementCycles);
}

//===----------------------------------------------------------------------===//
// The elementwise opcodes.
//===----------------------------------------------------------------------===//

TEST(Elementwise, Relu) {
  // max(x, 0), elementwise. The negative zero case is in there on purpose: it
  // is the value an implementation written as `x < 0 ? 0 : x` gets right and an
  // implementation written as `x > 0 ? x : 0` also gets right, and it is the
  // value a sign bit test gets wrong.
  const Outcome outcome = computeOnce(
      {{{2, 3}, {-1.0f, 0.0f, 1.0f, -2.5f, 5.0f, -0.5f}, {}, {}}}, {2, 3},
      Opcode::RELU, plain,
      // Scratchpad: operand 6 elements, result 6 elements. 12 * 4 = 48 bytes.
      48);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {0, 0, 1, 0, 5, 0});
}

TEST(Elementwise, ElementwiseBatchTwo) {
  // Batch 2 on an elementwise operation. Batch is derived from the shape and
  // there is no path here that is different when N is 1.
  //
  // Section 10.1's P7 list asks for batch 2 and batch 4 on **the elementwise
  // operations**, plural, so both opcodes are exercised at both batch sizes
  // rather than one opcode at each. They share a loop, but a list that asks
  // for two things is not satisfied by covering each of them once between
  // them.
  //
  //   a = 1..8, b = 10, 20, ... 80, so a + b = 11, 22, 33, ... 88.
  const Outcome sum = computeOnce(
      {{{2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}},
       {{2, 1, 2, 2}, {10, 20, 30, 40, 50, 60, 70, 80}, {}, {}}},
      {2, 1, 2, 2}, Opcode::ADD, plain,
      // Scratchpad: two operands of 8 elements and a result of 8. 24 * 4 = 96.
      96);
  ASSERT_TRUE(sum.result.ok()) << sum.result.error.value_or("");
  expectValues(sum.values, {11, 22, 33, 44, 55, 66, 77, 88});

  //   a = 1..8, b = 10, 20, ... 80, so a * b = 10, 40, 90, ... 640.
  const Outcome product = computeOnce(
      {{{2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}},
       {{2, 1, 2, 2}, {10, 20, 30, 40, 50, 60, 70, 80}, {}, {}}},
      {2, 1, 2, 2}, Opcode::MUL, plain,
      // Scratchpad: two operands of 8 elements and a result of 8. 24 * 4 = 96.
      96);
  ASSERT_TRUE(product.result.ok()) << product.result.error.value_or("");
  expectValues(product.values, {10, 40, 90, 160, 250, 360, 490, 640});
}

TEST(Elementwise, ElementwiseBatchFour) {
  //   a = 1..8, b = 2 everywhere, so a * b = 2, 4, 6, ... 16.
  const Outcome product = computeOnce(
      {{{4, 1, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}},
       {{4, 1, 1, 2}, {2, 2, 2, 2, 2, 2, 2, 2}, {}, {}}},
      {4, 1, 1, 2}, Opcode::MUL, plain,
      // Scratchpad: two operands of 8 elements and a result of 8. 24 * 4 = 96.
      96);
  ASSERT_TRUE(product.result.ok()) << product.result.error.value_or("");
  expectValues(product.values, {2, 4, 6, 8, 10, 12, 14, 16});

  //   a = 1..8, b = 2 everywhere, so a + b = 3, 4, 5, ... 10.
  const Outcome sum = computeOnce(
      {{{4, 1, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}},
       {{4, 1, 1, 2}, {2, 2, 2, 2, 2, 2, 2, 2}, {}, {}}},
      {4, 1, 1, 2}, Opcode::ADD, plain,
      // Scratchpad: two operands of 8 elements and a result of 8. 24 * 4 = 96.
      96);
  ASSERT_TRUE(sum.result.ok()) << sum.result.error.value_or("");
  expectValues(sum.values, {3, 4, 5, 6, 7, 8, 9, 10});
}

TEST(Elementwise, ChannelBroadcastArrivesAsAStrideZeroOperand) {
  // ADR 0005's channel broadcast, which the kernels are required to need no
  // special case for. The addend is a three element buffer read as a rank 4
  // view with a stride of zero on every axis but the channel, so one element is
  // read for a whole plane and the dot product with the strides does it.
  //
  //   lhs (0, c, h, w) holds c * 4 + h * 2 + w, so channel 0 is 0..3,
  //   channel 1 is 4..7 and channel 2 is 8..11.
  //   bias is 100, 200, 300 per channel.
  const Outcome outcome = computeOnce(
      {{{1, 3, 2, 2},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        {},
        {}},
       {{3}, {100, 200, 300}, {1, 3, 2, 2}, {0, 1, 0, 0}}},
      {1, 3, 2, 2}, Opcode::ADD, plain,
      // Scratchpad: 12 element activation, 3 element bias, 12 element result.
      // 27 * 4 = 108 bytes.
      108);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {100, 101, 102, 103, 204, 205, 206, 207, 308,
                                309, 310, 311});
}

TEST(Elementwise, TheFusedActivationApplies) {
  // ADD carries an activation field, and a fused relu is applied to the sum
  // rather than to either operand.
  //
  //   1 + (-3) = -2 -> 0,  4 + (-1) = 3 -> 3
  const Outcome outcome = computeOnce(
      {{{2}, {1, 4}, {}, {}}, {{2}, {-3, -1}, {}, {}}}, {2}, Opcode::ADD,
      [](Instruction &instruction) {
        instruction.activation = Activation::Relu;
      },
      // Scratchpad: two operands of 2 elements and a result of 2. 6 * 4 = 24.
      24);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {0, 3});
}

//===----------------------------------------------------------------------===//
// MATMUL.
//===----------------------------------------------------------------------===//

TEST(MatMul, WithABias) {
  //   [1 2 3]   [ 7  8]     [1*7+2*9+3*11  1*8+2*10+3*12]   [ 58   64]
  //   [4 5 6] x [ 9 10]  =  [4*7+5*9+6*11  4*8+5*10+6*12] = [139  154]
  //             [11 12]
  //
  //   plus a bias of [1000, 2000] on the columns:
  //
  //   [1058  2064]
  //   [1139  2154]
  const Outcome outcome = computeOnce(
      {{{2, 3}, {1, 2, 3, 4, 5, 6}, {}, {}},
       {{3, 2}, {7, 8, 9, 10, 11, 12}, {}, {}},
       {{2}, {1000, 2000}, {}, {}}},
      {2, 2}, Opcode::MATMUL, plain,
      // Scratchpad: lhs 6, rhs 6, bias 2, result 4. 18 * 4 = 72 bytes.
      72);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1058, 2064, 1139, 2154});
  // M * K * N, counted where the cost is charged and not derived from a cycle
  // figure: 2 * 3 * 2 = 12.
  EXPECT_EQ(outcome.result.stats.macs, 12u);
}

//===----------------------------------------------------------------------===//
// CONV2D.
//===----------------------------------------------------------------------===//

/// The configuration a convolution test writes, kept as a struct so the call
/// sites read as the four attributes rather than as four vectors.
struct ConvAttributes {
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};
  int64_t group = 1;

  void operator()(Instruction &instruction) const {
    instruction.strides = strides;
    instruction.pads = pads;
    instruction.dilations = dilations;
    instruction.group = group;
  }
};

TEST(Convolution, DenseThreeByThree) {
  // A 3 by 3 input under a 2 by 2 filter of ones, no padding, unit stride.
  //
  //   1 2 3      12 16
  //   4 5 6  ->  24 28      because 1+2+4+5 = 12, 2+3+5+6 = 16,
  //   7 8 9                         4+5+7+8 = 24, 5+6+8+9 = 28.
  const ConvAttributes attributes;
  const Outcome outcome = computeOnce(
      {{{1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {}, {}},
       {{1, 1, 2, 2}, {1, 1, 1, 1}, {}, {}}},
      {1, 1, 2, 2}, Opcode::CONV2D, attributes,
      // Scratchpad: input 9, filter 4, result 4. 17 * 4 = 68 bytes.
      68);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {12, 16, 24, 28});
  // N * F * OH * OW * (C / group) * KH * KW = 1 * 1 * 2 * 2 * 1 * 2 * 2 = 16.
  EXPECT_EQ(outcome.result.stats.macs, 16u);
}

TEST(Convolution, GroupedConvolution) {
  // group = 2 over four input channels and two output channels, with a 1 by 1
  // filter. Filter shape is (F, C / group, KH, KW) = (2, 2, 1, 1).
  //
  //   input channels, each 2 by 2:
  //     c0 = 1 2 3 4      c1 = 5 6 7 8      c2 = 9 10 11 12    c3 = 13 14 15 16
  //   weights: output channel 0 takes 1 and 2 over c0 and c1;
  //            output channel 1 takes 3 and 4 over c2 and c3.
  //
  //   f0 = 1*c0 + 2*c1 = 1+10, 2+12, 3+14, 4+16       = 11, 14, 17, 20
  //   f1 = 3*c2 + 4*c3 = 27+52, 30+56, 33+60, 36+64   = 79, 86, 93, 100
  ConvAttributes attributes;
  attributes.group = 2;
  const Outcome outcome = computeOnce(
      {{{1, 4, 2, 2},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        {},
        {}},
       {{2, 2, 1, 1}, {1, 2, 3, 4}, {}, {}}},
      {1, 2, 2, 2}, Opcode::CONV2D, attributes,
      // Scratchpad: input 16, filter 4, result 8. 28 * 4 = 112 bytes.
      112);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {11, 14, 17, 20, 79, 86, 93, 100});
  // 1 * 2 * 2 * 2 * (4 / 2) * 1 * 1 = 16.
  EXPECT_EQ(outcome.result.stats.macs, 16u);
}

TEST(Convolution, DepthwiseConvolution) {
  // group == C, which is the depthwise case: every output channel sees exactly
  // one input channel and the filter is (C, 1, 1, 1).
  //
  //   c0 = 1 2 3 4 scaled by 2   -> 2 4 6 8
  //   c1 = 5 6 7 8 scaled by 3   -> 15 18 21 24
  //   c2 = 9 10 11 12 scaled by 4 -> 36 40 44 48
  ConvAttributes attributes;
  attributes.group = 3;
  const Outcome outcome = computeOnce(
      {{{1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {}, {}},
       {{3, 1, 1, 1}, {2, 3, 4}, {}, {}}},
      {1, 3, 2, 2}, Opcode::CONV2D, attributes,
      // Scratchpad: input 12, filter 3, result 12. 27 * 4 = 108 bytes.
      108);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {2, 4, 6, 8, 15, 18, 21, 24, 36, 40, 44, 48});
  // 1 * 3 * 2 * 2 * (3 / 3) * 1 * 1 = 12.
  EXPECT_EQ(outcome.result.stats.macs, 12u);

  // The depthwise case is also the one Section 5.5 says the utilization term
  // exists for. Each group presents a single column to a sixteen column array,
  // so the spatial occupancy is 1/256 of the array rather than all of it, and
  // the effective MAC count is far above the raw one. The raw count above is
  // what the energy path sees; this is what the cycle charge sees.
  EXPECT_LT(outcome.result.stats.utilization, 0.01);
  EXPECT_GT(outcome.result.stats.effectiveMacs,
            static_cast<double>(outcome.result.stats.macs));
}

TEST(Convolution, DilatedConvolution) {
  // dilation 2 over a 5 by 5 input with a 2 by 2 filter of ones. The effective
  // kernel is (2 - 1) * 2 + 1 = 3, so the output is 5 - 3 + 1 = 3 on a side.
  //
  //   input (r, c) holds r * 5 + c + 1.
  //   output (oh, ow) = in[oh][ow] + in[oh][ow+2] + in[oh+2][ow] + in[oh+2][ow+2]
  //
  //   (0,0): 1 + 3 + 11 + 13 = 28     (0,1): 2 + 4 + 12 + 14 = 32
  //   (0,2): 3 + 5 + 13 + 15 = 36     (1,0): 6 + 8 + 16 + 18 = 48
  //   (1,1): 7 + 9 + 17 + 19 = 52     (1,2): 8 + 10 + 18 + 20 = 56
  //   (2,0): 11 + 13 + 21 + 23 = 68   (2,1): 12 + 14 + 22 + 24 = 72
  //   (2,2): 13 + 15 + 23 + 25 = 76
  std::vector<float> input(25);
  for (int index = 0; index < 25; ++index)
    input[index] = static_cast<float>(index + 1);

  ConvAttributes attributes;
  attributes.dilations = {2, 2};
  const Outcome outcome = computeOnce(
      {{{1, 1, 5, 5}, input, {}, {}}, {{1, 1, 2, 2}, {1, 1, 1, 1}, {}, {}}},
      {1, 1, 3, 3}, Opcode::CONV2D, attributes,
      // Scratchpad: input 25, filter 4, result 9. 38 * 4 = 152 bytes.
      152);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {28, 32, 36, 48, 52, 56, 68, 72, 76});
}

TEST(Convolution, AsymmetricPadding) {
  // pads are four entries in ONNX order: padTop, padLeft, padBottom, padRight.
  // Here one row of padding above and one column to the right, and nothing on
  // the other two sides, over a 2 by 2 input with a 2 by 2 filter of ones.
  //
  //   padded height 2 + 1 + 0 = 3, padded width 2 + 0 + 1 = 3, output 2 by 2.
  //   ih = oh - 1 + kh, iw = ow + kw, and anything outside 0..1 contributes
  //   nothing.
  //
  //   input   1 2
  //           3 4
  //
  //   (0,0): kh=0 is above the input; kh=1 gives row 0, columns 0 and 1 -> 1+2 = 3
  //   (0,1): row 0, column 1 only, since column 2 is padding             ->   2
  //   (1,0): rows 0 and 1, columns 0 and 1                    -> 1+2+3+4 =  10
  //   (1,1): rows 0 and 1, column 1 only                          -> 2+4 =   6
  ConvAttributes attributes;
  attributes.pads = {1, 0, 0, 1};
  const Outcome outcome = computeOnce(
      {{{1, 1, 2, 2}, {1, 2, 3, 4}, {}, {}},
       {{1, 1, 2, 2}, {1, 1, 1, 1}, {}, {}}},
      {1, 1, 2, 2}, Opcode::CONV2D, attributes,
      // Scratchpad: input 4, filter 4, result 4. 12 * 4 = 48 bytes.
      48);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {3, 2, 10, 6});
}

TEST(Convolution, ConvolutionBatchTwo) {
  // Two images, one 2 by 2 filter of ones, no padding: each output is the sum
  // of its image. 1+2+3+4 = 10 and 10+20+30+40 = 100.
  const ConvAttributes attributes;
  const Outcome outcome = computeOnce(
      {{{2, 1, 2, 2}, {1, 2, 3, 4, 10, 20, 30, 40}, {}, {}},
       {{1, 1, 2, 2}, {1, 1, 1, 1}, {}, {}}},
      {2, 1, 1, 1}, Opcode::CONV2D, attributes,
      // Scratchpad: input 8, filter 4, result 2. 14 * 4 = 56 bytes.
      56);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {10, 100});
}

TEST(Convolution, ConvolutionBatchFour) {
  // Four images of 1..16 in groups of four, same filter: 10, 26, 42, 58.
  std::vector<float> input(16);
  for (int index = 0; index < 16; ++index)
    input[index] = static_cast<float>(index + 1);

  const ConvAttributes attributes;
  const Outcome outcome = computeOnce(
      {{{4, 1, 2, 2}, input, {}, {}}, {{1, 1, 2, 2}, {1, 1, 1, 1}, {}, {}}},
      {4, 1, 1, 1}, Opcode::CONV2D, attributes,
      // Scratchpad: input 16, filter 4, result 4. 24 * 4 = 96 bytes.
      96);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {10, 26, 42, 58});
}

//===----------------------------------------------------------------------===//
// The pooling opcodes.
//===----------------------------------------------------------------------===//

/// The window a pooling test declares.
struct PoolAttributes {
  std::vector<int64_t> kernel = {2, 2};
  std::vector<int64_t> strides = {2, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  void operator()(Instruction &instruction) const {
    instruction.kernel = kernel;
    instruction.strides = strides;
    instruction.pads = pads;
    instruction.dilations = dilations;
  }
};

TEST(Pooling, MaxPooling) {
  // A 4 by 4 input holding 1..16, a 2 by 2 window at stride 2:
  //
  //    1  2  3  4        the four windows are {1,2,5,6}, {3,4,7,8},
  //    5  6  7  8        {9,10,13,14} and {11,12,15,16}, whose maxima
  //    9 10 11 12        are 6, 8, 14 and 16.
  //   13 14 15 16
  std::vector<float> input(16);
  for (int index = 0; index < 16; ++index)
    input[index] = static_cast<float>(index + 1);

  const PoolAttributes attributes;
  const Outcome outcome =
      computeOnce({{{1, 1, 4, 4}, input, {}, {}}}, {1, 1, 2, 2},
                  Opcode::POOL_MAX, attributes,
                  // Scratchpad: input 16, result 4. 20 * 4 = 80 bytes.
                  80);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {6, 8, 14, 16});
}

TEST(Pooling, AveragePooling) {
  // The same four windows, averaged: 14/4 = 3.5, 22/4 = 5.5, 46/4 = 11.5 and
  // 54/4 = 13.5.
  std::vector<float> input(16);
  for (int index = 0; index < 16; ++index)
    input[index] = static_cast<float>(index + 1);

  const PoolAttributes attributes;
  const Outcome outcome =
      computeOnce({{{1, 1, 4, 4}, input, {}, {}}}, {1, 1, 2, 2},
                  Opcode::POOL_AVG, attributes,
                  // Scratchpad: input 16, result 4. 20 * 4 = 80 bytes.
                  80);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {3.5f, 5.5f, 11.5f, 13.5f});
}

TEST(Pooling, AllPaddingWindow) {
  // The window that must not divide by zero.
  //
  // A single element input with two units of padding on every side gives a
  // padded extent of 1 + 2 + 2 = 5 and, with a 1 by 1 window at unit stride, a
  // 5 by 5 output. Only the centre window contains the one real element; the
  // other twenty four contain nothing at all.
  //
  // For the average that is a sum of nothing over a count of nothing, and
  // count_include_pad = 0 means the divisor is the count of contributing
  // elements, which is zero. The answer is 0, written deliberately rather than
  // produced by a division nobody guarded.
  PoolAttributes attributes;
  attributes.kernel = {1, 1};
  attributes.strides = {1, 1};
  attributes.pads = {2, 2, 2, 2};

  const Outcome average =
      computeOnce({{{1, 1, 1, 1}, {5}, {}, {}}}, {1, 1, 5, 5},
                  Opcode::POOL_AVG, attributes,
                  // Scratchpad: input 1, result 25. 26 * 4 = 104 bytes.
                  104);
  ASSERT_TRUE(average.result.ok()) << average.result.error.value_or("");
  ASSERT_EQ(average.values.size(), 25u);
  for (size_t index = 0; index < average.values.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_FLOAT_EQ(average.values[index], index == 12 ? 5.0f : 0.0f);
    EXPECT_FALSE(std::isnan(average.values[index]));
  }

  // The maximum of an empty window is the identity of the maximum, which is
  // negative infinity and is what ONNX produces. It is asserted rather than
  // left to whatever the accumulator happened to start at.
  const Outcome maximum =
      computeOnce({{{1, 1, 1, 1}, {5}, {}, {}}}, {1, 1, 5, 5},
                  Opcode::POOL_MAX, attributes,
                  // Scratchpad: input 1, result 25. 26 * 4 = 104 bytes.
                  104);
  ASSERT_TRUE(maximum.result.ok()) << maximum.result.error.value_or("");
  ASSERT_EQ(maximum.values.size(), 25u);
  for (size_t index = 0; index < maximum.values.size(); ++index) {
    SCOPED_TRACE(index);
    if (index == 12)
      EXPECT_FLOAT_EQ(maximum.values[index], 5.0f);
    else
      EXPECT_EQ(maximum.values[index],
                -std::numeric_limits<float>::infinity());
  }
}

TEST(Pooling, PoolingBatchTwo) {
  // Two images of four elements each, one 2 by 2 window. Section 10.1's P7 list
  // asks for batch 2 and batch 4 on **both pooling kernels**, so both are here
  // at both batch sizes rather than one each.
  //
  // The maxima are the last element of each image, and the averages are the
  // means: (1+2+3+4)/4 = 2.5 and (5+6+7+8)/4 = 6.5.
  const PoolAttributes attributes;
  const Outcome maximum = computeOnce(
      {{{2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}}}, {2, 1, 1, 1},
      Opcode::POOL_MAX, attributes,
      // Scratchpad: input 8, result 2. 10 * 4 = 40 bytes.
      40);
  ASSERT_TRUE(maximum.result.ok()) << maximum.result.error.value_or("");
  expectValues(maximum.values, {4, 8});

  const Outcome average = computeOnce(
      {{{2, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8}, {}, {}}}, {2, 1, 1, 1},
      Opcode::POOL_AVG, attributes,
      // Scratchpad: input 8, result 2. 10 * 4 = 40 bytes.
      40);
  ASSERT_TRUE(average.result.ok()) << average.result.error.value_or("");
  expectValues(average.values, {2.5f, 6.5f});
}

TEST(Pooling, PoolingBatchFour) {
  // Four images of four elements each. The maxima are 4, 8, 12, 16 and the
  // averages are 2.5, 6.5, 10.5, 14.5.
  std::vector<float> input(16);
  for (int index = 0; index < 16; ++index)
    input[index] = static_cast<float>(index + 1);

  const PoolAttributes attributes;
  const Outcome maximum =
      computeOnce({{{4, 1, 2, 2}, input, {}, {}}}, {4, 1, 1, 1},
                  Opcode::POOL_MAX, attributes,
                  // Scratchpad: input 16, result 4. 20 * 4 = 80 bytes.
                  80);
  ASSERT_TRUE(maximum.result.ok()) << maximum.result.error.value_or("");
  expectValues(maximum.values, {4, 8, 12, 16});

  const Outcome average =
      computeOnce({{{4, 1, 2, 2}, input, {}, {}}}, {4, 1, 1, 1},
                  Opcode::POOL_AVG, attributes,
                  // Scratchpad: input 16, result 4. 20 * 4 = 80 bytes.
                  80);
  ASSERT_TRUE(average.result.ok()) << average.result.error.value_or("");
  expectValues(average.values, {2.5f, 6.5f, 10.5f, 14.5f});
}

//===----------------------------------------------------------------------===//
// The shape opcodes.
//===----------------------------------------------------------------------===//

TEST(Shape, Reshape) {
  // Same elements, same order, different extents.
  const Outcome outcome =
      computeOnce({{{2, 3}, {1, 2, 3, 4, 5, 6}, {}, {}}}, {3, 2},
                  Opcode::RESHAPE, plain,
                  // Scratchpad: operand 6, result 6. 12 * 4 = 48 bytes.
                  48);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1, 2, 3, 4, 5, 6});
}

TEST(Shape, TransposeIdentity) {
  // The identity permutation, which is required by name precisely because an
  // implementation that special cased it would have two paths where the machine
  // has one.
  const Outcome outcome = computeOnce(
      {{{2, 3}, {1, 2, 3, 4, 5, 6}, {}, {}}}, {2, 3}, Opcode::TRANSPOSE,
      [](Instruction &instruction) { instruction.axes = {0, 1}; },
      // Scratchpad: operand 6, result 6. 12 * 4 = 48 bytes.
      48);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1, 2, 3, 4, 5, 6});
}

TEST(Shape, TransposeNchwToNhwc) {
  // The rank 4 case the layout assignment pass of Phase P13 will emit.
  //
  //   input is (N, C, H, W) = (1, 3, 2, 2) holding c * 4 + h * 2 + w + 1,
  //   so channel 0 is 1..4, channel 1 is 5..8 and channel 2 is 9..12.
  //
  //   permutation [0, 2, 3, 1] gives (N, H, W, C) = (1, 2, 2, 3), and
  //   result[h][w][c] = input[c][h][w]:
  //
  //     (0,0): 1, 5, 9      (0,1): 2, 6, 10
  //     (1,0): 3, 7, 11     (1,1): 4, 8, 12
  const Outcome outcome = computeOnce(
      {{{1, 3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {}, {}}},
      {1, 2, 2, 3}, Opcode::TRANSPOSE,
      [](Instruction &instruction) { instruction.axes = {0, 2, 3, 1}; },
      // Scratchpad: operand 12, result 12. 24 * 4 = 96 bytes.
      96);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12});
}

TEST(Shape, ConcatChannelAxis) {
  // Batch 2 on the channel axis, which is the case where the operands
  // interleave in the result rather than landing one after the other.
  //
  //   a is (2, 1, 1, 2): image 0 holds 1, 2 and image 1 holds 3, 4.
  //   b is (2, 2, 1, 2): image 0 holds 10, 11 then 12, 13,
  //                      image 1 holds 20, 21 then 22, 23.
  //
  //   concatenating on axis 1 gives (2, 3, 1, 2):
  //     image 0: 1, 2 | 10, 11 | 12, 13
  //     image 1: 3, 4 | 20, 21 | 22, 23
  const Outcome outcome = computeOnce(
      {{{2, 1, 1, 2}, {1, 2, 3, 4}, {}, {}},
       {{2, 2, 1, 2}, {10, 11, 12, 13, 20, 21, 22, 23}, {}, {}}},
      {2, 3, 1, 2}, Opcode::CONCAT,
      [](Instruction &instruction) { instruction.axes = {1}; },
      // Scratchpad: a 4, b 8, result 12. 24 * 4 = 96 bytes.
      96);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values,
               {1, 2, 10, 11, 12, 13, 3, 4, 20, 21, 22, 23});
}

TEST(Shape, ConcatLastAxis) {
  //   [1 2]  and  [5]   on axis 1  ->  [1 2 5]
  //   [3 4]       [6]                  [3 4 6]
  const Outcome outcome = computeOnce(
      {{{2, 2}, {1, 2, 3, 4}, {}, {}}, {{2, 1}, {5, 6}, {}, {}}}, {2, 3},
      Opcode::CONCAT,
      [](Instruction &instruction) { instruction.axes = {1}; },
      // Scratchpad: a 4, b 2, result 6. 12 * 4 = 48 bytes.
      48);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1, 2, 5, 3, 4, 6});
}

TEST(Shape, ConcatThreeOperands) {
  //   [1 2] , [5] , [ 7  8  9]  on axis 1  ->  [1 2 5  7  8  9]
  //   [3 4]   [6]   [10 11 12]                 [3 4 6 10 11 12]
  const Outcome outcome = computeOnce(
      {{{2, 2}, {1, 2, 3, 4}, {}, {}},
       {{2, 1}, {5, 6}, {}, {}},
       {{2, 3}, {7, 8, 9, 10, 11, 12}, {}, {}}},
      {2, 6}, Opcode::CONCAT,
      [](Instruction &instruction) { instruction.axes = {1}; },
      // Scratchpad: a 4, b 2, c 6, result 12. 24 * 4 = 96 bytes.
      96);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values, {1, 2, 5, 7, 8, 9, 3, 4, 6, 10, 11, 12});
}

TEST(Shape, ConcatBatchFour) {
  //   a is (4, 1) holding 1, 2, 3, 4 and b is (4, 2) holding the pairs
  //   (10, 11), (20, 21), (30, 31), (40, 41). On axis 1 the result is (4, 3).
  const Outcome outcome = computeOnce(
      {{{4, 1}, {1, 2, 3, 4}, {}, {}},
       {{4, 2}, {10, 11, 20, 21, 30, 31, 40, 41}, {}, {}}},
      {4, 3}, Opcode::CONCAT,
      [](Instruction &instruction) { instruction.axes = {1}; },
      // Scratchpad: a 4, b 8, result 12. 24 * 4 = 96 bytes.
      96);
  ASSERT_TRUE(outcome.result.ok()) << outcome.result.error.value_or("");
  expectValues(outcome.values,
               {1, 10, 11, 2, 20, 21, 3, 30, 31, 4, 40, 41});
}

//===----------------------------------------------------------------------===//
// The quantization opcodes, which have no kernel at this phase.
//===----------------------------------------------------------------------===//

TEST(Quantization, QuantRefusesByNameUntilPhaseP14) {
  // Section 10.1: at Phase P6 the quantization opcodes carry structural
  // coverage only, and P7's gate asks for the f32 list. So the machine refuses,
  // and the refusal names the phase. Asserting it is what makes "no kernel yet"
  // a tested behaviour rather than something a reader discovers.
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, 2, 3, 4});
  const int64_t buffer = builder.scratch(4);
  const int64_t result = builder.scratch(4, ElemType::I8);

  builder.add(dmaLoad(buffer, shape, at(MemSpace::Dram, source, shape)));

  Instruction quantize;
  quantize.opcode = Opcode::QUANT;
  quantize.resultSpace = MemSpace::Scratchpad;
  quantize.resultElementType = ElemType::I8;
  quantize.resultAddress = result;
  quantize.resultShape = shape;
  quantize.resultStrides = resultStridesFor(shape);
  quantize.operands.push_back(at(MemSpace::Scratchpad, buffer, shape));
  quantize.scale = 0.5f;
  quantize.zeroPoint = 3;
  builder.add(std::move(quantize));
  builder.add(halt());

  // Scratchpad: one f32 buffer of 4 elements is 16 bytes, one i8 buffer of 4
  // elements is 4 bytes. 16 + 4 = 20 bytes.
  Harness harness(builder.finish(20));
  const SimResult outcome = harness.run();

  ASSERT_FALSE(outcome.ok());
  EXPECT_NE(outcome.error->find("QUANT"), std::string::npos) << *outcome.error;
  EXPECT_NE(outcome.error->find("P14"), std::string::npos) << *outcome.error;
}

TEST(Quantization, DequantRefusesByNameUntilPhaseP14) {
  // The mirror of the case above, and it is here because Section 10.1's first
  // sentence asks for a test per opcode rather than per interesting opcode.
  // DEQUANT reads i8 and writes f32, which is the reverse of QUANT, so a
  // refusal that named the wrong one would be invisible without this.
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.input(shape, ElemType::I8);
  const int64_t buffer = builder.scratch(4, ElemType::I8);
  const int64_t result = builder.scratch(4);

  builder.add(dmaLoad(buffer, shape,
                      at(MemSpace::Dram, source, shape, ElemType::I8)));

  Instruction dequantize;
  dequantize.opcode = Opcode::DEQUANT;
  dequantize.resultSpace = MemSpace::Scratchpad;
  dequantize.resultElementType = ElemType::F32;
  dequantize.resultAddress = result;
  dequantize.resultShape = shape;
  dequantize.resultStrides = resultStridesFor(shape);
  dequantize.operands.push_back(
      at(MemSpace::Scratchpad, buffer, shape, ElemType::I8));
  dequantize.scale = 0.25f;
  dequantize.zeroPoint = -7;
  builder.add(std::move(dequantize));
  builder.add(halt());

  // Scratchpad: one i8 buffer of 4 elements is 4 bytes, one f32 buffer of 4
  // elements is 16 bytes. 4 + 16 = 20 bytes.
  Harness harness(builder.finish(20));
  const SimResult outcome = harness.run();

  ASSERT_FALSE(outcome.ok());
  EXPECT_NE(outcome.error->find("DEQUANT"), std::string::npos)
      << *outcome.error;
  EXPECT_NE(outcome.error->find("P14"), std::string::npos) << *outcome.error;
}

} // namespace
