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

#include "gtest/gtest.h"

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

// The windowed operations decline tile generation, on purpose. The halo
// arithmetic belongs with the pass that will exercise it, and returning a wrong
// tile would be worse than returning none because a pass would consume it.
TEST_F(NPUTilingTest, WindowedTileGenerationIsDeclinedRatherThanGuessed) {
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

  SmallVector<OpFoldResult> offsets(8, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes(8, builder.getIndexAttr(1));

  EXPECT_TRUE(failed(tiling.getTiledImplementation(builder, offsets, sizes)))
      << "the halo arithmetic a convolution tile needs is not implemented "
         "here, and declining is the honest report";
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

} // namespace
