//===- TilingInterfaceTest.cpp - TilingInterface unit tests -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the TilingInterface external models, covering the grouped and
// batched iteration domains, and for the destination passing partition rule on
// every compute operation.
//
// These are unit tests rather than lit tests because what is under test is a
// C++ interface query and not a piece of IR syntax. A lit test can only see
// what an operation prints, and an iteration domain is not printed.
//
// No pass consumes TilingInterface yet, deliberately. What these tests pin is
// the interface's answers, so that when the tiling pass arrives a disagreement
// between the two is attributable to one of them.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/Interfaces/NPUTilingInterfaceImpl.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FormatVariadic.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::npu;

namespace {

/// A context with the npu dialect loaded and the tiling models attached.
///
/// The registration call is the point of the promised interface mechanism: the
/// dialect promises TilingInterface on each compute operation and this call is
/// what fulfils the promise. A fixture that forgot the call would fail with a
/// named error saying the interface was promised and never provided, which is
/// the whole reason the promise exists.
class NPUTilingTest : public ::testing::Test {
protected:
  NPUTilingTest() {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, tensor::TensorDialect, NPUDialect>();
    registerNPUTilingInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  /// Parses a module and returns the first operation of the named kind inside
  /// it. Keeping the IR as text rather than building it with an OpBuilder means
  /// these tests exercise the same parser every lit test does, so a test that
  /// passes here describes IR a user could actually write.
  template <typename OpTy>
  OpTy parseFirst(StringRef moduleText) {
    module = parseSourceString<ModuleOp>(moduleText, &context);
    EXPECT_TRUE(module) << "failed to parse:\n" << moduleText.str();
    if (!module)
      return nullptr;
    OpTy found = nullptr;
    module->walk([&](OpTy op) {
      if (!found)
        found = op;
    });
    EXPECT_TRUE(found) << "no operation of the expected kind in:\n"
                       << moduleText.str();
    return found;
  }

  /// The static upper bounds of an iteration domain, which is all this dialect
  /// ever has: every tensor here is statically shaped, so every range is a
  /// constant.
  static SmallVector<int64_t> staticBounds(ArrayRef<Range> ranges) {
    SmallVector<int64_t> bounds;
    for (const Range &range : ranges) {
      std::optional<int64_t> size = getConstantIntValue(range.size);
      EXPECT_TRUE(size.has_value())
          << "an iteration domain of this dialect is always static";
      bounds.push_back(size.value_or(-1));
    }
    return bounds;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

//===----------------------------------------------------------------------===//
// The batched iteration domain.
//===----------------------------------------------------------------------===//

// Batch is a first class dimension everywhere in this dialect, and the
// iteration domain is where that promise either holds or quietly does not. A
// domain that opened with the channel loop, or that carried a batch bound of 1
// for a batch of 4, would be a domain a tiling pass could not split over the
// batch, and the shortcut would be invisible in the printed IR.
TEST_F(NPUTilingTest, Conv2DIterationDomainIsBatched) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<4x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                 %d: tensor<4x8x8x8xf32>) -> tensor<4x8x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<4x3x8x8xf32>, tensor<8x3x3x3xf32>)
                      outs(%d : tensor<4x8x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<4x8x8x8xf32>
      return %0 : tensor<4x8x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = dyn_cast<TilingInterface>(conv.getOperation());
  ASSERT_TRUE(tiling) << "conv2d must implement TilingInterface; if this fails, "
                         "registerNPUTilingInterfaceExternalModels was not "
                         "called";

  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  // (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW)
  ASSERT_EQ(bounds.size(), 8u);
  EXPECT_EQ(bounds[0], 4) << "the batch bound is the batch, not 1";
  EXPECT_EQ(bounds[1], 1) << "one group";
  EXPECT_EQ(bounds[2], 8) << "eight output channels in that one group";
  EXPECT_EQ(bounds[3], 8);
  EXPECT_EQ(bounds[4], 8);
  EXPECT_EQ(bounds[5], 3) << "three input channels per group";
  EXPECT_EQ(bounds[6], 3);
  EXPECT_EQ(bounds[7], 3);

  SmallVector<utils::IteratorType> types = tiling.getLoopIteratorTypes();
  ASSERT_EQ(types.size(), 8u);
  for (unsigned i = 0; i < 5; ++i)
    EXPECT_EQ(types[i], utils::IteratorType::parallel)
        << "dimension " << i << " of a convolution domain is parallel";
  for (unsigned i = 5; i < 8; ++i)
    EXPECT_EQ(types[i], utils::IteratorType::reduction)
        << "dimension " << i << " of a convolution domain reduces";
}

//===----------------------------------------------------------------------===//
// The grouped iteration domain.
//===----------------------------------------------------------------------===//

// A depthwise convolution: group equals the input channel count, so each output
// channel sees exactly one input channel.
//
// This is the shape that makes the split between the group dimension and the
// per group output channel dimension load bearing. Collapsed into a single
// output channel loop of 8, this domain would be identical to the dense
// convolution's, and a pass splitting that loop would produce tiles reading the
// wrong input channels. Split, the two are distinguishable: G = 8 and Cout/G =
// 1 here against G = 1 and Cout/G = 8 above.
TEST_F(NPUTilingTest, Conv2DIterationDomainIsGrouped) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<2x8x8x8xf32>, %w: tensor<8x1x3x3xf32>,
                 %d: tensor<2x8x8x8xf32>) -> tensor<2x8x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<2x8x8x8xf32>, tensor<8x1x3x3xf32>)
                      outs(%d : tensor<2x8x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 8 : i64}
           -> tensor<2x8x8x8xf32>
      return %0 : tensor<2x8x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = cast<TilingInterface>(conv.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  ASSERT_EQ(bounds.size(), 8u);
  EXPECT_EQ(bounds[0], 2) << "batch";
  EXPECT_EQ(bounds[1], 8) << "eight groups, one per channel";
  EXPECT_EQ(bounds[2], 1) << "one output channel per group";
  EXPECT_EQ(bounds[5], 1) << "one input channel per group";

  // The product of the group and per group channel bounds is the output channel
  // count, which is what a reader expects; the split is what makes the grouping
  // visible to a pass.
  EXPECT_EQ(bounds[1] * bounds[2], 8);
}

// A grouped convolution that is neither dense nor depthwise, so that the two
// dimensions are both greater than one and a collapse cannot be mistaken for
// correct on a degenerate case.
TEST_F(NPUTilingTest, Conv2DIterationDomainWithFourGroups) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<3x8x8x8xf32>, %w: tensor<12x2x3x3xf32>,
                 %d: tensor<3x12x8x8xf32>) -> tensor<3x12x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<3x8x8x8xf32>, tensor<12x2x3x3xf32>)
                      outs(%d : tensor<3x12x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 4 : i64}
           -> tensor<3x12x8x8xf32>
      return %0 : tensor<3x12x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = cast<TilingInterface>(conv.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  ASSERT_EQ(bounds.size(), 8u);
  EXPECT_EQ(bounds[0], 3) << "batch";
  EXPECT_EQ(bounds[1], 4) << "four groups";
  EXPECT_EQ(bounds[2], 3) << "twelve output channels over four groups";
  EXPECT_EQ(bounds[5], 2) << "eight input channels over four groups";
}

//===----------------------------------------------------------------------===//
// The result tile position of a grouped convolution.
//===----------------------------------------------------------------------===//

// A tile of one whole group maps to a contiguous stretch of output channels.
TEST_F(NPUTilingTest, Conv2DResultTilePositionFoldsGroupAndChannel) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<3x8x8x8xf32>, %w: tensor<12x2x3x3xf32>,
                 %d: tensor<3x12x8x8xf32>) -> tensor<3x12x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<3x8x8x8xf32>, tensor<12x2x3x3xf32>)
                      outs(%d : tensor<3x12x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 4 : i64}
           -> tensor<3x12x8x8xf32>
      return %0 : tensor<3x12x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = cast<TilingInterface>(conv.getOperation());
  OpBuilder builder(&context);

  // Group 2, all three of its output channels, the whole batch and the whole
  // spatial extent. In result coordinates that is channels 6 through 8.
  SmallVector<OpFoldResult> offsets = {
      builder.getIndexAttr(0), builder.getIndexAttr(2), builder.getIndexAttr(0),
      builder.getIndexAttr(0), builder.getIndexAttr(0), builder.getIndexAttr(0),
      builder.getIndexAttr(0), builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {
      builder.getIndexAttr(3), builder.getIndexAttr(1), builder.getIndexAttr(3),
      builder.getIndexAttr(8), builder.getIndexAttr(8), builder.getIndexAttr(2),
      builder.getIndexAttr(3), builder.getIndexAttr(3)};

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  ASSERT_TRUE(succeeded(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)));

  ASSERT_EQ(resultOffsets.size(), 4u);
  EXPECT_EQ(getConstantIntValue(resultOffsets[0]), 0) << "batch offset";
  EXPECT_EQ(getConstantIntValue(resultOffsets[1]), 6)
      << "group 2 of 3 channels each starts at output channel 6";
  EXPECT_EQ(getConstantIntValue(resultSizes[1]), 3);
  EXPECT_EQ(getConstantIntValue(resultSizes[0]), 3) << "the whole batch";
}

// A tile spanning several groups but only part of each one names a set of
// output channels that is not contiguous, so it is not expressible as an offset
// and a size. The model declines rather than returning a plausible wrong
// answer, which is the behaviour a consuming pass has to be able to rely on.
TEST_F(NPUTilingTest, Conv2DResultTilePositionDeclinesNonContiguousChannels) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<1x8x8x8xf32>, %w: tensor<12x2x3x3xf32>,
                 %d: tensor<1x12x8x8xf32>) -> tensor<1x12x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<12x2x3x3xf32>)
                      outs(%d : tensor<1x12x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 4 : i64}
           -> tensor<1x12x8x8xf32>
      return %0 : tensor<1x12x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = cast<TilingInterface>(conv.getOperation());
  OpBuilder builder(&context);

  // Two groups, but only the first output channel of each. Those are channels 0
  // and 3, which no single offset and size describes.
  SmallVector<OpFoldResult> offsets(8, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = {
      builder.getIndexAttr(1), builder.getIndexAttr(2), builder.getIndexAttr(1),
      builder.getIndexAttr(8), builder.getIndexAttr(8), builder.getIndexAttr(2),
      builder.getIndexAttr(3), builder.getIndexAttr(3)};

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  EXPECT_TRUE(failed(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)))
      << "a tile naming a non contiguous set of output channels must be "
         "declined, not approximated";
}

//===----------------------------------------------------------------------===//
// The other iteration domains.
//===----------------------------------------------------------------------===//

TEST_F(NPUTilingTest, MatMulIterationDomainIsBatchedOverM) {
  auto matmul = parseFirst<MatMulOp>(R"mlir(
    func.func @f(%a: tensor<4x16xf32>, %b: tensor<16x10xf32>,
                 %d: tensor<4x10xf32>) -> tensor<4x10xf32> {
      %0 = npu.matmul ins(%a, %b : tensor<4x16xf32>, tensor<16x10xf32>)
                      outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
      return %0 : tensor<4x10xf32>
    }
  )mlir");
  ASSERT_TRUE(matmul);

  auto tiling = cast<TilingInterface>(matmul.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  ASSERT_EQ(bounds.size(), 3u);
  EXPECT_EQ(bounds[0], 4) << "M is the batch of a fully connected layer";
  EXPECT_EQ(bounds[1], 10);
  EXPECT_EQ(bounds[2], 16);

  SmallVector<utils::IteratorType> types = tiling.getLoopIteratorTypes();
  ASSERT_EQ(types.size(), 3u);
  EXPECT_EQ(types[0], utils::IteratorType::parallel);
  EXPECT_EQ(types[1], utils::IteratorType::parallel);
  EXPECT_EQ(types[2], utils::IteratorType::reduction);
}

TEST_F(NPUTilingTest, MaxPoolIterationDomainIsBatchedAndReducesOverTheWindow) {
  auto pool = parseFirst<MaxPool2DOp>(R"mlir(
    func.func @f(%x: tensor<5x8x8x8xf32>, %d: tensor<5x8x4x4xf32>)
        -> tensor<5x8x4x4xf32> {
      %0 = npu.max_pool2d ins(%x : tensor<5x8x8x8xf32>)
                          outs(%d : tensor<5x8x4x4xf32>)
                          {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                           pads = array<i64: 0, 0, 0, 0>,
                           dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
           -> tensor<5x8x4x4xf32>
      return %0 : tensor<5x8x4x4xf32>
    }
  )mlir");
  ASSERT_TRUE(pool);

  auto tiling = cast<TilingInterface>(pool.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  // (N, C, Hout, Wout, KH, KW)
  ASSERT_EQ(bounds.size(), 6u);
  EXPECT_EQ(bounds[0], 5) << "batch";
  EXPECT_EQ(bounds[1], 8) << "channels, unchanged by pooling";
  EXPECT_EQ(bounds[2], 4);
  EXPECT_EQ(bounds[3], 4);
  EXPECT_EQ(bounds[4], 2) << "the window reduces";
  EXPECT_EQ(bounds[5], 2);

  SmallVector<utils::IteratorType> types = tiling.getLoopIteratorTypes();
  ASSERT_EQ(types.size(), 6u);
  EXPECT_EQ(types[3], utils::IteratorType::parallel);
  EXPECT_EQ(types[4], utils::IteratorType::reduction);
  EXPECT_EQ(types[5], utils::IteratorType::reduction);
}

// The NHWC form of the same pool. The domain is written in NCHW order whatever
// the tensor's layout, because the domain describes the computation and not the
// memory, and it is getResultTilePosition that maps back into the layout.
TEST_F(NPUTilingTest, PoolIterationDomainIsLayoutIndependent) {
  auto pool = parseFirst<MaxPool2DOp>(R"mlir(
    func.func @f(%x: tensor<5x8x8x8xf32, #npu.layout<nhwc>>,
                 %d: tensor<5x4x4x8xf32, #npu.layout<nhwc>>)
        -> tensor<5x4x4x8xf32, #npu.layout<nhwc>> {
      %0 = npu.max_pool2d ins(%x : tensor<5x8x8x8xf32, #npu.layout<nhwc>>)
                          outs(%d : tensor<5x4x4x8xf32, #npu.layout<nhwc>>)
                          {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                           pads = array<i64: 0, 0, 0, 0>,
                           dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
           -> tensor<5x4x4x8xf32, #npu.layout<nhwc>>
      return %0 : tensor<5x4x4x8xf32, #npu.layout<nhwc>>
    }
  )mlir");
  ASSERT_TRUE(pool);

  auto tiling = cast<TilingInterface>(pool.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  ASSERT_EQ(bounds.size(), 6u);
  EXPECT_EQ(bounds[0], 5) << "batch";
  EXPECT_EQ(bounds[1], 8) << "the channel bound is the channel count, which "
                             "under NHWC is the last extent";
  EXPECT_EQ(bounds[2], 4);
  EXPECT_EQ(bounds[3], 4);

  // The result tile position maps the NCHW ordered domain back into the NHWC
  // result, so a channel tile lands on the last axis rather than the second.
  SmallVector<OpFoldResult> offsets(6, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = {
      builder.getIndexAttr(5), builder.getIndexAttr(2), builder.getIndexAttr(4),
      builder.getIndexAttr(4), builder.getIndexAttr(2), builder.getIndexAttr(2)};
  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  ASSERT_TRUE(succeeded(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)));
  ASSERT_EQ(resultSizes.size(), 4u);
  EXPECT_EQ(getConstantIntValue(resultSizes[0]), 5) << "batch first";
  EXPECT_EQ(getConstantIntValue(resultSizes[3]), 2)
      << "the channel tile is on the last axis under NHWC";
}

TEST_F(NPUTilingTest, ElementwiseIterationDomainIsTheResultShape) {
  auto add = parseFirst<AddOp>(R"mlir(
    func.func @f(%a: tensor<7x8x4x4xf32>, %b: tensor<7x8x4x4xf32>,
                 %d: tensor<7x8x4x4xf32>) -> tensor<7x8x4x4xf32> {
      %0 = npu.add ins(%a, %b : tensor<7x8x4x4xf32>, tensor<7x8x4x4xf32>)
                   outs(%d : tensor<7x8x4x4xf32>) -> tensor<7x8x4x4xf32>
      return %0 : tensor<7x8x4x4xf32>
    }
  )mlir");
  ASSERT_TRUE(add);

  auto tiling = cast<TilingInterface>(add.getOperation());
  OpBuilder builder(&context);
  SmallVector<int64_t> bounds = staticBounds(tiling.getIterationDomain(builder));

  ASSERT_EQ(bounds.size(), 4u);
  EXPECT_EQ(bounds[0], 7) << "batch";
  EXPECT_EQ(bounds[1], 8);
  EXPECT_EQ(bounds[2], 4);
  EXPECT_EQ(bounds[3], 4);

  for (utils::IteratorType type : tiling.getLoopIteratorTypes())
    EXPECT_EQ(type, utils::IteratorType::parallel)
        << "every dimension of an elementwise operation is parallel";
}

// The elementwise operations are the ones whose tile generation is complete, so
// this asserts that a tile really is produced and really is the right shape.
TEST_F(NPUTilingTest, ElementwiseTileGenerationProducesATileOfEveryOperand) {
  auto relu = parseFirst<ReluOp>(R"mlir(
    func.func @f(%x: tensor<4x8x4x4xf32>, %d: tensor<4x8x4x4xf32>)
        -> tensor<4x8x4x4xf32> {
      %0 = npu.relu ins(%x : tensor<4x8x4x4xf32>) outs(%d : tensor<4x8x4x4xf32>)
           -> tensor<4x8x4x4xf32>
      return %0 : tensor<4x8x4x4xf32>
    }
  )mlir");
  ASSERT_TRUE(relu);

  auto tiling = cast<TilingInterface>(relu.getOperation());
  OpBuilder builder(&context);
  builder.setInsertionPoint(relu);

  // The second half of the batch, one channel, the whole spatial extent.
  SmallVector<OpFoldResult> offsets = {
      builder.getIndexAttr(2), builder.getIndexAttr(3), builder.getIndexAttr(0),
      builder.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes = {
      builder.getIndexAttr(2), builder.getIndexAttr(1), builder.getIndexAttr(4),
      builder.getIndexAttr(4)};

  FailureOr<TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, sizes);
  ASSERT_TRUE(succeeded(tiled));
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  ASSERT_EQ(tiled->tiledValues.size(), 1u);

  auto tiledType = cast<RankedTensorType>(tiled->tiledValues[0].getType());
  EXPECT_EQ(tiledType.getShape(), ArrayRef<int64_t>({2, 1, 4, 4}));

  SmallVector<OpFoldResult> resultOffsets, resultSizes;
  ASSERT_TRUE(succeeded(tiling.getResultTilePosition(
      builder, 0, offsets, sizes, resultOffsets, resultSizes)));
  EXPECT_EQ(getConstantIntValue(resultOffsets[0]), 2);
  EXPECT_EQ(getConstantIntValue(resultSizes[1]), 1);
}

// **This test moved with the level at P13, and what it used to say is worth
// keeping.** From P1 to P12 it was
// `WindowedTileGenerationIsDeclinedRatherThanGuessed`, and it asserted that a
// convolution declines tile generation outright, because the halo arithmetic
// belonged with the pass that would exercise it and returning a wrong tile
// would be worse than returning none. P13 is that phase, so the assertion is
// replaced rather than deleted: the tiles it now returns are checked by
// `NPUTiledImplementationTest` below, and what survives here is the one refusal
// that is permanent.
//
// **Leaving the old test in place would have been the worse mistake.** It asked
// for every size to be one, which tiles the reduction as well as the parallel
// dimensions, so it would have gone on passing against the new model for a
// reason it did not state. A test that keeps passing while the thing it names
// stops being true is the shape of D-0047.
TEST_F(NPUTilingTest, WindowedTileGenerationSplitsParallelDimensionsOnly) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<1x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                 %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                      outs(%d : tensor<1x8x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<1x8x8x8xf32>
      return %0 : tensor<1x8x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  auto tiling = cast<TilingInterface>(conv.getOperation());
  OpBuilder builder(&context);
  builder.setInsertionPoint(conv);

  // Domain (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW) = (1, 1, 8, 8, 8, 3, 3, 3).
  // The old test's request: every extent one, which splits the reduction.
  SmallVector<OpFoldResult> offsets(8, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> everyExtentOne(8, builder.getIndexAttr(1));
  EXPECT_TRUE(
      failed(tiling.getTiledImplementation(builder, offsets, everyExtentOne)))
      << "splitting the input channel or the kernel window re associates an "
         "fp32 accumulation, which Section 13.2 permits only behind "
         "allow-reduction-tiling with its own golden set";

  // The same tile with the three reduction extents left whole is generated.
  SmallVector<OpFoldResult> parallelOnly = {
      builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(1),
      builder.getIndexAttr(1), builder.getIndexAttr(1), builder.getIndexAttr(3),
      builder.getIndexAttr(3), builder.getIndexAttr(3)};
  FailureOr<TilingResult> tiled =
      tiling.getTiledImplementation(builder, offsets, parallelOnly);
  ASSERT_TRUE(succeeded(tiled))
      << "the halo arithmetic P1 declined to write is what P13 owns";
  ASSERT_EQ(tiled->tiledOps.size(), 1u);
  EXPECT_EQ(cast<RankedTensorType>(tiled->tiledValues[0].getType()).getShape(),
            ArrayRef<int64_t>({1, 1, 1, 1}));
}

//===----------------------------------------------------------------------===//
// The destination passing partition, on every compute operation.
//===----------------------------------------------------------------------===//

namespace {

/// Asserts the partition rule directly through the interface's own accessors:
/// every operand is covered exactly once by the union of ins and outs, and no
/// operand appears in both. This is the same rule the verifier holds, asserted
/// here from the outside so that a change to getDpsInitsMutable that broke the
/// verifier's own check would still be caught.
void expectInsOutsPartition(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  ASSERT_TRUE(dps) << op->getName().getStringRef().str()
                   << " must implement DestinationStyleOpInterface";

  const int64_t numOperands = op->getNumOperands();
  SmallVector<int> coverage(numOperands, 0);

  ASSERT_EQ(dps.getNumDpsInits(), 1)
      << op->getName().getStringRef().str()
      << " has exactly one destination, the trailing operand";
  ++coverage[dps.getDpsInitOperand(0)->getOperandNumber()];

  for (OpOperand *input : dps.getDpsInputOperands())
    ++coverage[input->getOperandNumber()];

  for (int64_t i = 0; i < numOperands; ++i)
    EXPECT_EQ(coverage[i], 1)
        << "operand " << i << " of " << op->getName().getStringRef().str()
        << " is covered " << coverage[i] << " times, not once";

  // The destination is the last operand, which is what makes the init range
  // consecutive and trailing.
  EXPECT_EQ(dps.getDpsInitOperand(0)->getOperandNumber(), numOperands - 1);

  // And no operand is both an input and an init.
  for (OpOperand *input : dps.getDpsInputOperands())
    EXPECT_FALSE(dps.isDpsInit(input))
        << "operand " << input->getOperandNumber() << " of "
        << op->getName().getStringRef().str() << " is both an input and an init";
}

} // namespace

TEST_F(NPUTilingTest, InsAndOutsPartitionEveryComputeOperation) {
  // One module holding every compute operation, in both its optional operand
  // forms where it has one, so that the partition is asserted on the operand
  // counts a real graph actually produces rather than on one shape each.
  module = parseSourceString<ModuleOp>(R"mlir(
    func.func @every_compute_op(
        %x: tensor<2x3x8x8xf32>, %w: tensor<8x3x3x3xf32>, %cb: tensor<8xf32>,
        %cd: tensor<2x8x8x8xf32>,
        %a: tensor<4x16xf32>, %b: tensor<16x10xf32>, %mb: tensor<10xf32>,
        %md: tensor<4x10xf32>,
        %e: tensor<2x8x4x4xf32>, %ed: tensor<2x8x4x4xf32>,
        %g: tensor<8xf32>,
        %p: tensor<2x8x8x8xf32>, %pd: tensor<2x8x4x4xf32>,
        %t: tensor<2x3x8x8xf32>, %td: tensor<2x8x8x3xf32>,
        %c1: tensor<2x4x8x8xf32>, %c2: tensor<2x6x8x8xf32>,
        %ccd: tensor<2x10x8x8xf32>) {
      %conv = npu.conv2d ins(%x, %w : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>)
                         outs(%cd : tensor<2x8x8x8xf32>)
                         {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                          dilations = array<i64: 1, 1>, group = 1 : i64}
              -> tensor<2x8x8x8xf32>
      %convb = npu.conv2d ins(%x, %w, %cb : tensor<2x3x8x8xf32>,
                                            tensor<8x3x3x3xf32>, tensor<8xf32>)
                          outs(%cd : tensor<2x8x8x8xf32>)
                          {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                           dilations = array<i64: 1, 1>, group = 1 : i64}
               -> tensor<2x8x8x8xf32>
      %mm = npu.matmul ins(%a, %b : tensor<4x16xf32>, tensor<16x10xf32>)
                       outs(%md : tensor<4x10xf32>) -> tensor<4x10xf32>
      %mmb = npu.matmul ins(%a, %b, %mb : tensor<4x16xf32>, tensor<16x10xf32>,
                                          tensor<10xf32>)
                        outs(%md : tensor<4x10xf32>) -> tensor<4x10xf32>
      %add = npu.add ins(%e, %e : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
                     outs(%ed : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
      %mul = npu.mul ins(%e, %e : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
                     outs(%ed : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
      %relu = npu.relu ins(%e : tensor<2x8x4x4xf32>)
                       outs(%ed : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
      %maxp = npu.max_pool2d ins(%p : tensor<2x8x8x8xf32>)
                             outs(%pd : tensor<2x8x4x4xf32>)
                             {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                              pads = array<i64: 0, 0, 0, 0>,
                              dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
              -> tensor<2x8x4x4xf32>
      %avgp = npu.avg_pool2d ins(%p : tensor<2x8x8x8xf32>)
                             outs(%pd : tensor<2x8x4x4xf32>)
                             {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                              pads = array<i64: 0, 0, 0, 0>,
                              dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
              -> tensor<2x8x4x4xf32>
      %tr = npu.transpose ins(%t : tensor<2x3x8x8xf32>)
                          outs(%td : tensor<2x8x8x3xf32>)
                          {permutation = array<i64: 0, 2, 3, 1>}
            -> tensor<2x8x8x3xf32>
      %cc = npu.concat ins(%c1, %c2 : tensor<2x4x8x8xf32>, tensor<2x6x8x8xf32>)
                       outs(%ccd : tensor<2x10x8x8xf32>) {axis = 1 : i64}
            -> tensor<2x10x8x8xf32>
      %bn = npu.batch_norm ins(%e, %g, %g, %g, %g : tensor<2x8x4x4xf32>,
                                                    tensor<8xf32>, tensor<8xf32>,
                                                    tensor<8xf32>, tensor<8xf32>)
                           outs(%ed : tensor<2x8x4x4xf32>)
                           {epsilon = 1.000000e-05 : f32} -> tensor<2x8x4x4xf32>
      return
    }
  )mlir",
                                      &context);
  ASSERT_TRUE(module);

  // Every destination passing operation in the module, and there must be twelve
  // of them: the ten the design names as destination passing (conv2d, matmul,
  // add, mul, relu, both pools, transpose, concat, batch_norm) plus the second
  // appearance of conv2d and matmul in their optional bias forms.
  //
  // The count is asserted rather than left implicit because the whole point of
  // this test is coverage: a walk that silently visited nine operations would
  // pass every assertion inside expectInsOutsPartition and prove nothing about
  // the tenth.
  unsigned seen = 0;
  module->walk([&](Operation *op) {
    if (!isa<DestinationStyleOpInterface>(op))
      return;
    ++seen;
    expectInsOutsPartition(op);
  });
  EXPECT_EQ(seen, 12u) << "every compute operation in the module was visited";
}

//===----------------------------------------------------------------------===//
// The tiled implementations, which arrive at P13.
//
// P1 implemented the introspection half of this interface and returned failure
// from `getTiledImplementation` for the windowed operations, with a comment
// saying that returning a wrong tile would be worse than returning none because
// a pass would consume it and produce a program with a quietly wrong answer.
// These are the tests that make the tile it now returns not wrong.
//
// **Two properties are asserted and the second is the one that matters.** The
// first is that a tile has the shape it was asked for, which is what a pass
// needs to stitch the tiles back together. The second is that a tile computes
// the same arithmetic on the same inputs, which is what makes tiling exact and
// is why the P13 gate can ask for goldens byte identical rather than inside a
// tolerance. The second is checked by recomputing, for every output position of
// every tile, which input positions its window touches, and comparing that
// against the same computation on the untiled operation.
//===----------------------------------------------------------------------===//

/// Tiles `op` and returns the operation the model built, or null.
///
/// The builder writes into a throwaway block so the tiles never join the module
/// being tested. What is under test is the operation the model produced, and
/// inserting it into the original function would leave a second definition of
/// every value beside the first.
class NPUTiledImplementationTest : public NPUTilingTest {
protected:
  /// The tile, built into a detached block that this fixture owns.
  Operation *tile(Operation *op, ArrayRef<int64_t> offsets,
                  ArrayRef<int64_t> sizes) {
    auto tiling = dyn_cast<TilingInterface>(op);
    if (!tiling)
      return nullptr;

    OpBuilder builder(&context);
    scratch = std::make_unique<Block>();
    builder.setInsertionPointToStart(scratch.get());

    SmallVector<OpFoldResult> offsetValues;
    SmallVector<OpFoldResult> sizeValues;
    for (int64_t offset : offsets)
      offsetValues.push_back(builder.getIndexAttr(offset));
    for (int64_t size : sizes)
      sizeValues.push_back(builder.getIndexAttr(size));

    FailureOr<TilingResult> result =
        tiling.getTiledImplementation(builder, offsetValues, sizeValues);
    if (failed(result) || result->tiledOps.empty())
      return nullptr;
    return result->tiledOps.front();
  }

  /// True when the model declined, which is a result rather than a failure.
  bool declines(Operation *op, ArrayRef<int64_t> offsets,
                ArrayRef<int64_t> sizes) {
    return tile(op, offsets, sizes) == nullptr;
  }

  /// Parses the next case, dropping any tile built against the previous one
  /// first.
  ///
  /// **A tile holds operands belonging to the module it was built against**, so
  /// reparsing while one is alive destroys values that still have uses, and MLIR
  /// asserts on exactly that. A test that parses one module needs none of this;
  /// the sweep below parses five, and it took the assertion to notice.
  template <typename OpTy>
  OpTy parseCase(StringRef moduleText) {
    scratch.reset();
    return parseFirst<OpTy>(moduleText);
  }

  static SmallVector<int64_t> shapeOf(Value value) {
    return llvm::to_vector(cast<RankedTensorType>(value.getType()).getShape());
  }

  std::unique_ptr<Block> scratch;
};

/// The input positions one output position's window touches, as a set of
/// indices into the operation's own input, with a negative index standing for a
/// position in the leading padding and an index at or above the extent standing
/// for one in the trailing padding.
///
/// Written out here rather than shared with the implementation, deliberately: a
/// test that called the same helper the code calls would agree with it whatever
/// either of them said.
SmallVector<int64_t> windowPositions(int64_t outputIndex, int64_t kernel,
                                     int64_t stride, int64_t dilation,
                                     int64_t padBegin) {
  SmallVector<int64_t> positions;
  for (int64_t tap = 0; tap < kernel; ++tap)
    positions.push_back(outputIndex * stride - padBegin + tap * dilation);
  return positions;
}

TEST_F(NPUTiledImplementationTest, Conv2DTilesTheOutputHeightWithItsHalo) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<2x4x8x8xf32>, %w: tensor<6x4x3x3xf32>,
                 %b: tensor<6xf32>, %d: tensor<2x6x8x8xf32>)
        -> tensor<2x6x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w, %b : tensor<2x4x8x8xf32>, tensor<6x4x3x3xf32>,
                                       tensor<6xf32>)
                      outs(%d : tensor<2x6x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<2x6x8x8xf32>
      return %0 : tensor<2x6x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  // Domain (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW) = (2, 1, 6, 8, 8, 4, 3, 3).
  // A middle tile of four output rows, everything else whole.
  Operation *tiled = tile(conv, {0, 0, 0, 2, 0, 0, 0, 0}, {2, 1, 6, 4, 8, 4, 3, 3});
  ASSERT_TRUE(tiled);
  auto tiledConv = cast<Conv2DOp>(tiled);

  // The result is the tile that was asked for.
  EXPECT_EQ(shapeOf(tiledConv.getResult()),
            (SmallVector<int64_t>{2, 6, 4, 8}));

  // The input carries the halo. Output rows 2 to 5 with a 3 tap kernel, unit
  // stride and a pad of one read input rows 1 to 6, which is six rows, and the
  // tile sits away from both edges so it carries no padding of its own on that
  // axis. The width axis is untiled and keeps the original's pad on both sides.
  EXPECT_EQ(shapeOf(tiledConv.getInput()), (SmallVector<int64_t>{2, 4, 6, 8}));
  EXPECT_EQ(tiledConv.getPads(), (ArrayRef<int64_t>{0, 1, 0, 1}));

  // The filter and the bias are whole, because only spatial axes were tiled.
  EXPECT_EQ(shapeOf(tiledConv.getFilter()),
            (SmallVector<int64_t>{6, 4, 3, 3}));
  ASSERT_TRUE(tiledConv.getBias());
  EXPECT_EQ(shapeOf(tiledConv.getBias()), (SmallVector<int64_t>{6}));
  EXPECT_EQ(tiledConv.getGroup(), 1);

  // And it is a legal operation, which is the check that would catch an extent
  // the shared windowed arithmetic disagrees with.
  EXPECT_TRUE(succeeded(tiledConv.verify()));
}

TEST_F(NPUTiledImplementationTest, Conv2DTheFirstAndLastTilesKeepTheEdgePads) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<1x2x8x8xf32>, %w: tensor<2x2x3x3xf32>,
                 %d: tensor<1x2x8x8xf32>) -> tensor<1x2x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<1x2x8x8xf32>, tensor<2x2x3x3xf32>)
                      outs(%d : tensor<1x2x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<1x2x8x8xf32>
      return %0 : tensor<1x2x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  // The first two output rows read input rows -1 to 2, so the tile keeps the
  // leading pad and slices three real rows.
  Operation *first = tile(conv, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 2, 2, 8, 2, 3, 3});
  ASSERT_TRUE(first);
  auto firstConv = cast<Conv2DOp>(first);
  EXPECT_EQ(shapeOf(firstConv.getInput()), (SmallVector<int64_t>{1, 2, 3, 8}));
  EXPECT_EQ(firstConv.getPads(), (ArrayRef<int64_t>{1, 1, 0, 1}));
  EXPECT_TRUE(succeeded(firstConv.verify()));

  // And the last two read rows 5 to 8, so it keeps the trailing pad.
  Operation *last = tile(conv, {0, 0, 0, 6, 0, 0, 0, 0}, {1, 1, 2, 2, 8, 2, 3, 3});
  ASSERT_TRUE(last);
  auto lastConv = cast<Conv2DOp>(last);
  EXPECT_EQ(shapeOf(lastConv.getInput()), (SmallVector<int64_t>{1, 2, 3, 8}));
  EXPECT_EQ(lastConv.getPads(), (ArrayRef<int64_t>{0, 1, 1, 1}));
  EXPECT_TRUE(succeeded(lastConv.verify()));
}

TEST_F(NPUTiledImplementationTest, Conv2DEveryTileReadsTheSamePositionsAsTheWhole) {
  // **This is the exactness property, and it is the reason the P13 gate asks
  // for goldens byte identical rather than inside a tolerance.** For a sweep of
  // tile sizes and a set of strides, dilations and pads, every output position
  // of every tile is checked to read the same input positions it read untiled,
  // including which of them lie in the padding. An average pool divides by the
  // number of positions that actually contributed, so a tile that turned a real
  // element into a padded one would move a divisor rather than only a sum.
  struct Case {
    int64_t kernel, stride, dilation, pad, inputExtent, outputExtent;
  };
  // Each output extent is the shared windowed arithmetic's answer, written out
  // rather than computed, so that a change to that arithmetic fails here too.
  const Case cases[] = {
      {3, 1, 1, 1, 8, 8},  // same padding, the common case
      {3, 1, 1, 0, 8, 6},  // valid padding
      {2, 2, 1, 0, 8, 4},  // a strided pool shape
      {3, 2, 1, 1, 8, 4},  // strided and padded at once
      {3, 1, 2, 2, 12, 12}, // dilated, which is dilated_stack's shape
  };

  for (const Case &shape : cases) {
    std::string text = llvm::formatv(
        R"mlir(
    func.func @f(%x: tensor<1x2x{0}x{0}xf32>, %w: tensor<2x2x{1}x{1}xf32>,
                 %d: tensor<1x2x{2}x{2}xf32>) -> tensor<1x2x{2}x{2}xf32> {{
      %0 = npu.conv2d ins(%x, %w : tensor<1x2x{0}x{0}xf32>, tensor<2x2x{1}x{1}xf32>)
                      outs(%d : tensor<1x2x{2}x{2}xf32>)
                      {{strides = array<i64: {3}, {3}>,
                       pads = array<i64: {4}, {4}, {4}, {4}>,
                       dilations = array<i64: {5}, {5}>, group = 1 : i64}
           -> tensor<1x2x{2}x{2}xf32>
      return %0 : tensor<1x2x{2}x{2}xf32>
    })mlir",
        shape.inputExtent, shape.kernel, shape.outputExtent, shape.stride,
        shape.pad, shape.dilation);

    auto conv = parseCase<Conv2DOp>(text);
    ASSERT_TRUE(conv) << text;

    for (int64_t tileSize = 1; tileSize <= shape.outputExtent; ++tileSize) {
      if (shape.outputExtent % tileSize != 0)
        continue;
      for (int64_t offset = 0; offset < shape.outputExtent;
           offset += tileSize) {
        Operation *tiled =
            tile(conv, {0, 0, 0, offset, 0, 0, 0, 0},
                 {1, 1, 2, tileSize, shape.outputExtent, 2, shape.kernel,
                  shape.kernel});
        ASSERT_TRUE(tiled)
            << "declined at extent " << shape.outputExtent << " tile "
            << tileSize << " offset " << offset;
        auto tiledConv = cast<Conv2DOp>(tiled);
        ASSERT_TRUE(succeeded(tiledConv.verify()));

        // The tile's own output extent is the one it was asked for. Without
        // this the comparison below could pass over the wrong number of rows.
        ASSERT_EQ(shapeOf(tiledConv.getResult())[2], tileSize);

        const int64_t tilePadBegin = tiledConv.getPads()[0];
        const int64_t tileInputExtent = shapeOf(tiledConv.getInput())[2];

        // Where the tile's input slice actually starts, read off the slice the
        // model built rather than derived from the pads it reported. Deriving
        // it would make the comparison below check the pads against themselves;
        // reading it makes the two halves of the model, the slice and the pads,
        // have to agree with each other about the same positions.
        auto slice =
            tiledConv.getInput().getDefiningOp<tensor::ExtractSliceOp>();
        ASSERT_TRUE(slice);
        const int64_t sliceBegin = slice.getStaticOffsets()[2];

        for (int64_t local = 0; local < tileSize; ++local) {
          SmallVector<int64_t> whole = windowPositions(
              offset + local, shape.kernel, shape.stride, shape.dilation,
              shape.pad);
          SmallVector<int64_t> inTile =
              windowPositions(local, shape.kernel, shape.stride,
                              shape.dilation, tilePadBegin);

          ASSERT_EQ(whole.size(), inTile.size());
          for (size_t tap = 0; tap < whole.size(); ++tap) {
            const int64_t globalFromTile = inTile[tap] + sliceBegin;
            // Same position, whichever way it is reached.
            EXPECT_EQ(globalFromTile, whole[tap])
                << "extent " << shape.outputExtent << " tile " << tileSize
                << " offset " << offset << " local " << local << " tap " << tap;
            // And a position that was padding in the whole is padding in the
            // tile, and one that was real is real. This is the half an average
            // pool's divisor depends on.
            const bool paddingInWhole =
                whole[tap] < 0 || whole[tap] >= shape.inputExtent;
            const bool paddingInTile =
                inTile[tap] < 0 || inTile[tap] >= tileInputExtent;
            EXPECT_EQ(paddingInWhole, paddingInTile)
                << "extent " << shape.outputExtent << " tile " << tileSize
                << " offset " << offset << " local " << local << " tap " << tap;
          }
        }
      }
    }
  }
}

TEST_F(NPUTiledImplementationTest, Conv2DTilesTheGroupOfADepthwiseLayer) {
  // A depthwise convolution, which is the shape `depthwise_separable` carries
  // and the one the group dimension exists to keep expressible. Tiling one
  // group must slice that group's own input channel and its own filter row, and
  // a model that had collapsed the two channel dimensions would slice the wrong
  // channels while producing a tile of the right shape.
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<1x8x8x8xf32>, %w: tensor<8x1x3x3xf32>,
                 %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x1x3x3xf32>)
                      outs(%d : tensor<1x8x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 8 : i64}
           -> tensor<1x8x8x8xf32>
      return %0 : tensor<1x8x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  // Domain (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW) = (1, 8, 1, 8, 8, 1, 3, 3).
  // Two whole groups, starting at group 4.
  Operation *tiled = tile(conv, {0, 4, 0, 0, 0, 0, 0, 0}, {1, 2, 1, 8, 8, 1, 3, 3});
  ASSERT_TRUE(tiled);
  auto tiledConv = cast<Conv2DOp>(tiled);

  EXPECT_EQ(tiledConv.getGroup(), 2) << "two whole groups, so two groups";
  EXPECT_EQ(shapeOf(tiledConv.getResult()), (SmallVector<int64_t>{1, 2, 8, 8}));
  EXPECT_EQ(shapeOf(tiledConv.getInput()), (SmallVector<int64_t>{1, 2, 8, 8}))
      << "one input channel per group, and two groups";
  EXPECT_EQ(shapeOf(tiledConv.getFilter()), (SmallVector<int64_t>{2, 1, 3, 3}));
  EXPECT_TRUE(succeeded(tiledConv.verify()));

  // The slice has to start at input channel 4, which is where group 4 does.
  auto slice = tiledConv.getInput().getDefiningOp<tensor::ExtractSliceOp>();
  ASSERT_TRUE(slice);
  ArrayRef<int64_t> offsets = slice.getStaticOffsets();
  ASSERT_EQ(offsets.size(), 4u);
  EXPECT_EQ(offsets[1], 4)
      << "group 4's input channels start at 4, not at 0 and not at 4 times "
         "anything else";
}

TEST_F(NPUTiledImplementationTest, Conv2DDeclinesTheThingsSectionThirteenTwoForbids) {
  auto conv = parseFirst<Conv2DOp>(R"mlir(
    func.func @f(%x: tensor<1x4x8x8xf32>, %w: tensor<6x4x3x3xf32>,
                 %d: tensor<1x6x8x8xf32>) -> tensor<1x6x8x8xf32> {
      %0 = npu.conv2d ins(%x, %w : tensor<1x4x8x8xf32>, tensor<6x4x3x3xf32>)
                      outs(%d : tensor<1x6x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<1x6x8x8xf32>
      return %0 : tensor<1x6x8x8xf32>
    }
  )mlir");
  ASSERT_TRUE(conv);

  // The whole tile, which is the identity, is accepted. Without this the three
  // refusals below would be indistinguishable from a model that refuses
  // everything.
  EXPECT_FALSE(declines(conv, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 6, 8, 8, 4, 3, 3}));

  // The input channel reduction, split in half. Under fp32 addition is not
  // associative, so this moves every golden file and Section 13.2 permits it
  // only behind `allow-reduction-tiling` with its own golden set.
  EXPECT_TRUE(declines(conv, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 6, 8, 8, 2, 3, 3}));
  // The kernel window, on either axis, for the same reason.
  EXPECT_TRUE(declines(conv, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 6, 8, 8, 4, 1, 3}));
  EXPECT_TRUE(declines(conv, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 6, 8, 8, 4, 3, 1}));
  // And a reduction offset with a whole extent, which is the same split written
  // as an offset rather than as a size.
  EXPECT_TRUE(declines(conv, {0, 0, 0, 0, 0, 2, 0, 0}, {1, 1, 6, 8, 8, 4, 3, 3}));
}

TEST_F(NPUTiledImplementationTest, PoolTilesItsOutputAndDeclinesItsWindow) {
  auto pool = parseFirst<MaxPool2DOp>(R"mlir(
    func.func @f(%x: tensor<2x8x8x8xf32>, %d: tensor<2x8x4x4xf32>)
        -> tensor<2x8x4x4xf32> {
      %0 = npu.max_pool2d ins(%x : tensor<2x8x8x8xf32>)
                          outs(%d : tensor<2x8x4x4xf32>)
                          {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                           pads = array<i64: 0, 0, 0, 0>,
                           dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
           -> tensor<2x8x4x4xf32>
      return %0 : tensor<2x8x4x4xf32>
    }
  )mlir");
  ASSERT_TRUE(pool);

  // Domain (N, C, Hout, Wout, KH, KW) = (2, 8, 4, 4, 2, 2). Two output rows of
  // a stride two pool read four input rows and nothing overlaps.
  Operation *tiled = tile(pool, {0, 0, 2, 0, 0, 0}, {2, 8, 2, 4, 2, 2});
  ASSERT_TRUE(tiled);
  auto tiledPool = cast<MaxPool2DOp>(tiled);
  EXPECT_EQ(shapeOf(tiledPool.getResult()), (SmallVector<int64_t>{2, 8, 2, 4}));
  EXPECT_EQ(shapeOf(tiledPool.getInput()), (SmallVector<int64_t>{2, 8, 4, 8}));
  EXPECT_EQ(tiledPool.getPads(), (ArrayRef<int64_t>{0, 0, 0, 0}));
  EXPECT_TRUE(succeeded(tiledPool.verify()));

  // The channel axis is parallel here and is tileable, which is the difference
  // between a pool's domain and a convolution's.
  Operation *channels = tile(pool, {0, 4, 0, 0, 0, 0}, {2, 4, 4, 4, 2, 2});
  ASSERT_TRUE(channels);
  EXPECT_EQ(shapeOf(cast<MaxPool2DOp>(channels).getResult()),
            (SmallVector<int64_t>{2, 4, 4, 4}));

  // The window is the reduction and is declined.
  EXPECT_TRUE(declines(pool, {0, 0, 0, 0, 0, 0}, {2, 8, 4, 4, 1, 2}));
  EXPECT_TRUE(declines(pool, {0, 0, 0, 0, 0, 0}, {2, 8, 4, 4, 2, 1}));
}

TEST_F(NPUTiledImplementationTest, PoolDeclinesACeilMode) {
  // `ceil_mode = 1` adds the rule that a window whose first element would start
  // inside the right padded region is dropped, so the number of windows is no
  // longer the floor formula and a tile's own extent is no longer the extent it
  // was asked for. Declined rather than approximated.
  auto pool = parseFirst<AvgPool2DOp>(R"mlir(
    func.func @f(%x: tensor<1x2x7x7xf32>, %d: tensor<1x2x4x4xf32>)
        -> tensor<1x2x4x4xf32> {
      %0 = npu.avg_pool2d ins(%x : tensor<1x2x7x7xf32>)
                          outs(%d : tensor<1x2x4x4xf32>)
                          {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                           pads = array<i64: 0, 0, 0, 0>,
                           dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
           -> tensor<1x2x4x4xf32>
      return %0 : tensor<1x2x4x4xf32>
    }
  )mlir");
  ASSERT_TRUE(pool);
  EXPECT_TRUE(declines(pool, {0, 0, 0, 0, 0, 0}, {1, 2, 2, 4, 2, 2}));
  // Including the whole tile, because the refusal is about the mode and not
  // about which dimensions were split.
  EXPECT_TRUE(declines(pool, {0, 0, 0, 0, 0, 0}, {1, 2, 4, 4, 2, 2}));
}

TEST_F(NPUTiledImplementationTest, MatMulTilesBothParallelAxesAndDeclinesTheInner) {
  auto matmul = parseFirst<MatMulOp>(R"mlir(
    func.func @f(%a: tensor<8x400xf32>, %b: tensor<400x120xf32>,
                 %c: tensor<120xf32>, %d: tensor<8x120xf32>)
        -> tensor<8x120xf32> {
      %0 = npu.matmul ins(%a, %b, %c : tensor<8x400xf32>, tensor<400x120xf32>,
                                       tensor<120xf32>)
                      outs(%d : tensor<8x120xf32>) -> tensor<8x120xf32>
      return %0 : tensor<8x120xf32>
    }
  )mlir");
  ASSERT_TRUE(matmul);

  // The shape of `lenet`'s largest weight matrix. Domain (M, N, K).
  Operation *tiled = tile(matmul, {0, 40, 0}, {8, 40, 400});
  ASSERT_TRUE(tiled);
  auto tiledMatMul = cast<MatMulOp>(tiled);
  EXPECT_EQ(shapeOf(tiledMatMul.getResult()), (SmallVector<int64_t>{8, 40}));
  EXPECT_EQ(shapeOf(tiledMatMul.getLhs()), (SmallVector<int64_t>{8, 400}))
      << "the left operand keeps every column, because K is not tiled";
  EXPECT_EQ(shapeOf(tiledMatMul.getRhs()), (SmallVector<int64_t>{400, 40}));
  ASSERT_TRUE(tiledMatMul.getBias());
  EXPECT_EQ(shapeOf(tiledMatMul.getBias()), (SmallVector<int64_t>{40}))
      << "the bias is length N and is sliced with the columns";
  EXPECT_TRUE(succeeded(tiledMatMul.verify()));

  // The rows too.
  Operation *rows = tile(matmul, {4, 0, 0}, {4, 120, 400});
  ASSERT_TRUE(rows);
  EXPECT_EQ(shapeOf(cast<MatMulOp>(rows).getResult()),
            (SmallVector<int64_t>{4, 120}));

  // And the inner dimension is declined, for the convolution's reason.
  EXPECT_TRUE(declines(matmul, {0, 0, 0}, {8, 120, 200}));
  EXPECT_TRUE(declines(matmul, {0, 0, 200}, {8, 120, 400}));
}

} // namespace
