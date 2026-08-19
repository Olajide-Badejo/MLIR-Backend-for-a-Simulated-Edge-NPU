//===- NPUTilingInterfaceImpl.cpp - TilingInterface models ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The TilingInterface external models for the `npu` compute operations.
//
// What is implemented and what is not, stated up front, because an interface
// whose gaps are undocumented is an interface whose gaps get discovered by a
// pass author three phases later.
//
// - `getLoopIteratorTypes` and `getIterationDomain` are complete for every
//   operation. They are the introspection half, and they are what the tiling
//   pass will read to decide what it is allowed to split.
// - `getTiledImplementation` and `getResultTilePosition` are implemented for
//   the elementwise operations, whose iteration domain is exactly their result
//   shape and whose every dimension is parallel. For those, a tile of the
//   iteration space is a tile of every operand and a tile of the result, and
//   the slicing is honest.
// - For the windowed operations, `getTiledImplementation` returns failure. The
//   halo arithmetic that a convolution tile needs is real work, it belongs with
//   the tiling pass that will exercise it, and returning a wrong tile here
//   would be worse than returning none: a pass would consume it and produce a
//   program with a quietly wrong answer. Failure is the honest report and the
//   consuming phase is where it is replaced.
//
// The iteration domains carry the two structural facts the design cares about
// and the unit tests pin both: batch is a first class dimension, so it is
// always the outermost parallel loop and never assumed to be one, and a grouped
// convolution's domain has a group dimension distinct from its per group output
// channel dimension, so a depthwise layer with group equal to the channel count
// is expressible rather than collapsing to a single channel loop.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Interfaces/NPUTilingInterfaceImpl.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

/// Builds a `Range` list from a list of static extents, all starting at zero
/// with unit step. Every iteration domain in this dialect is static, because
/// every tensor in this dialect is statically shaped, so none of these ranges
/// ever needs a value.
SmallVector<Range> staticDomain(OpBuilder &builder,
                                ArrayRef<int64_t> extents) {
  OpFoldResult zero = builder.getIndexAttr(0);
  OpFoldResult one = builder.getIndexAttr(1);
  SmallVector<Range> ranges;
  ranges.reserve(extents.size());
  for (int64_t extent : extents)
    ranges.push_back(Range{zero, builder.getIndexAttr(extent), one});
  return ranges;
}

SmallVector<utils::IteratorType> allParallel(unsigned count) {
  return SmallVector<utils::IteratorType>(count, utils::IteratorType::parallel);
}

/// Extracts a tile of `source` at the given offsets and sizes, unit strides.
Value sliceOperand(OpBuilder &builder, Location loc, Value source,
                   ArrayRef<OpFoldResult> offsets,
                   ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> strides(offsets.size(), builder.getIndexAttr(1));
  return tensor::ExtractSliceOp::create(builder, loc, source, offsets, sizes,
                                        strides);
}

//===----------------------------------------------------------------------===//
// The elementwise model.
//===----------------------------------------------------------------------===//

/// Shared by add, mul, relu and batch_norm: the iteration domain is the result
/// shape, every dimension is parallel, and a tile of the domain is the same
/// tile of every shape matched operand.
///
/// batch_norm belongs here because its per channel parameters are rank 1 and
/// are not tiled: a tile of the iteration space needs the whole parameter
/// vector's channel slice, and since the channel axis is one of the tiled
/// dimensions the parameters would need a matching one dimensional slice. That
/// is exactly the case `getTiledImplementation` declines below, so batch_norm
/// gets the introspection half and not the generation half.
template <typename OpTy>
struct ElementwiseTilingModel
    : public TilingInterface::ExternalModel<ElementwiseTilingModel<OpTy>, OpTy> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    return allParallel(resultType.getRank());
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    return staticDomain(b, resultType.getShape());
  }

  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    // The iteration domain is the result shape, so a tile of the domain is
    // exactly the same tile of the result.
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }

  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto concreteOp = cast<OpTy>(op);
    Location loc = op->getLoc();
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

    // Every operand of a tileable elementwise operation here has the result
    // shape. An operand that does not, such as a batch_norm parameter vector,
    // means this model cannot produce the tile, and saying so is better than
    // slicing something whose axes do not line up with the domain.
    for (Value operand : op->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getShape() != resultType.getShape())
        return failure();
    }

    SmallVector<Value> tiledOperands;
    tiledOperands.reserve(op->getNumOperands());
    for (Value operand : op->getOperands())
      tiledOperands.push_back(sliceOperand(b, loc, operand, offsets, sizes));

    SmallVector<int64_t> tileShape;
    for (OpFoldResult size : sizes) {
      std::optional<int64_t> constant = getConstantIntValue(size);
      if (!constant)
        return failure();
      tileShape.push_back(*constant);
    }
    auto tiledResultType = RankedTensorType::get(
        tileShape, resultType.getElementType(), resultType.getEncoding());

    Operation *tiled = clone(b, concreteOp, {tiledResultType}, tiledOperands);
    return TilingResult{{tiled},
                        SmallVector<Value>(tiled->getResults()),
                        llvm::to_vector(llvm::map_range(
                            tiledOperands, [](Value v) -> Operation * {
                              return v.getDefiningOp();
                            }))};
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }

  LogicalResult getIterationDomainTileFromResultTile(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
      SmallVectorImpl<OpFoldResult> &iterDomainSizes) const {
    iterDomainOffsets.assign(offsets.begin(), offsets.end());
    iterDomainSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
};

/// The shape only variant: the iteration domain is the result shape and every
/// dimension is parallel, but a tile of the domain is not a matching tile of
/// the operands, so only the introspection half is provided.
///
/// This covers transpose, whose operand axes are permuted relative to the
/// result, and concat, whose operands each cover a disjoint stretch of one
/// axis, and batch_norm, whose parameters are rank 1. Each of the three needs
/// its own operand side arithmetic, and each of the three gets it in the phase
/// that consumes the interface.
template <typename OpTy>
struct ShapeOnlyTilingModel
    : public TilingInterface::ExternalModel<ShapeOnlyTilingModel<OpTy>, OpTy> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    return allParallel(resultType.getRank());
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    return staticDomain(b, resultType.getShape());
  }

  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// npu.conv2d
//===----------------------------------------------------------------------===//

/// The iteration domain of a grouped convolution, in this order:
///
///     (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW)
///
/// The first five are parallel and the last three are reduction. Two things
/// about that list are deliberate and both are pinned by a unit test.
///
/// **Batch is dimension zero and is a real loop.** There is no shortcut in this
/// dialect that assumes a batch of one, and the iteration domain is where that
/// promise either holds or quietly does not.
///
/// **The group dimension is separate from the per group output channel
/// dimension.** Collapsing them into a single Cout loop would make a depthwise
/// layer, where G equals the channel count and Cout/G is 1, indistinguishable
/// from a dense convolution with the same Cout, and a tiling pass splitting the
/// collapsed loop would produce tiles that read the wrong input channels. The
/// product of the two is Cout, which is what a reader expects; the split is
/// what makes grouping visible to a pass.
struct Conv2DTilingModel
    : public TilingInterface::ExternalModel<Conv2DTilingModel, Conv2DOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    SmallVector<utils::IteratorType> types = allParallel(5);
    types.append(3, utils::IteratorType::reduction);
    return types;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto conv = cast<Conv2DOp>(op);
    auto inputType = cast<RankedTensorType>(conv.getInput().getType());
    auto filterType = cast<RankedTensorType>(conv.getFilter().getType());
    auto resultType = cast<RankedTensorType>(conv.getResult().getType());

    const int64_t group = conv.getGroup();
    const int64_t outputChannels = filterType.getDimSize(0);
    SmallVector<int64_t> outputSpatial = getSpatialExtents(resultType);

    SmallVector<int64_t> extents = {getBatchExtent(inputType),
                                    group,
                                    outputChannels / group,
                                    outputSpatial[0],
                                    outputSpatial[1],
                                    filterType.getDimSize(1),
                                    filterType.getDimSize(2),
                                    filterType.getDimSize(3)};
    return staticDomain(b, extents);
  }

  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    // The result tile is the parallel prefix of the domain tile, with the
    // group and per group channel dimensions folded back into one channel
    // extent, and laid out per the operation's layout.
    auto conv = cast<Conv2DOp>(op);
    auto resultType = cast<RankedTensorType>(conv.getResult().getType());

    std::optional<int64_t> groupOffset = getConstantIntValue(offsets[1]);
    std::optional<int64_t> groupSize = getConstantIntValue(sizes[1]);
    std::optional<int64_t> channelOffset = getConstantIntValue(offsets[2]);
    std::optional<int64_t> channelSize = getConstantIntValue(sizes[2]);
    if (!groupOffset || !groupSize || !channelOffset || !channelSize)
      return failure();

    // Folding (g, c) into a single channel index is exact only when the tile
    // covers whole groups or sits inside one group. Anything else names a set
    // of channels that is not contiguous, and a non contiguous result tile is
    // not expressible as an offset and a size.
    const int64_t channelsPerGroup =
        cast<RankedTensorType>(conv.getFilter().getType()).getDimSize(0) /
        conv.getGroup();
    if (*groupSize != 1 && *channelSize != channelsPerGroup)
      return failure();

    const int64_t foldedOffset = *groupOffset * channelsPerGroup + *channelOffset;
    const int64_t foldedSize = *groupSize * *channelSize;

    SmallVector<OpFoldResult> spatialOffsets = {offsets[3], offsets[4]};
    SmallVector<OpFoldResult> spatialSizes = {sizes[3], sizes[4]};

    if (isNHWC(resultType)) {
      resultOffsets = {offsets[0], spatialOffsets[0], spatialOffsets[1],
                       b.getIndexAttr(foldedOffset)};
      resultSizes = {sizes[0], spatialSizes[0], spatialSizes[1],
                     b.getIndexAttr(foldedSize)};
    } else {
      resultOffsets = {offsets[0], b.getIndexAttr(foldedOffset),
                       spatialOffsets[0], spatialOffsets[1]};
      resultSizes = {sizes[0], b.getIndexAttr(foldedSize), spatialSizes[0],
                     spatialSizes[1]};
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pooling operations.
//===----------------------------------------------------------------------===//

/// Domain: (N, C, Hout, Wout, KH, KW). The first four are parallel, the two
/// window dimensions are reduction, because a pool reduces over its window.
template <typename OpTy>
struct PoolTilingModel
    : public TilingInterface::ExternalModel<PoolTilingModel<OpTy>, OpTy> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    SmallVector<utils::IteratorType> types = allParallel(4);
    types.append(2, utils::IteratorType::reduction);
    return types;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto pool = cast<OpTy>(op);
    auto inputType = cast<RankedTensorType>(pool.getInput().getType());
    auto resultType = cast<RankedTensorType>(pool.getResult().getType());
    ArrayRef<int64_t> kernel = pool.getKernel();
    SmallVector<int64_t> outputSpatial = getSpatialExtents(resultType);

    SmallVector<int64_t> extents = {
        getBatchExtent(inputType), getChannelExtent(inputType),
        outputSpatial[0],          outputSpatial[1],
        kernel[0],                 kernel[1]};
    return staticDomain(b, extents);
  }

  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    auto pool = cast<OpTy>(op);
    auto resultType = cast<RankedTensorType>(pool.getResult().getType());
    if (isNHWC(resultType)) {
      resultOffsets = {offsets[0], offsets[2], offsets[3], offsets[1]};
      resultSizes = {sizes[0], sizes[2], sizes[3], sizes[1]};
    } else {
      resultOffsets = {offsets[0], offsets[1], offsets[2], offsets[3]};
      resultSizes = {sizes[0], sizes[1], sizes[2], sizes[3]};
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// npu.matmul
//===----------------------------------------------------------------------===//

/// Domain: (M, N, K). M and N are parallel, K is the reduction. M is the batch
/// dimension of a fully connected layer and is a real loop for the same reason
/// the convolution's N is.
struct MatMulTilingModel
    : public TilingInterface::ExternalModel<MatMulTilingModel, MatMulOp> {

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    return {utils::IteratorType::parallel, utils::IteratorType::parallel,
            utils::IteratorType::reduction};
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto matmul = cast<MatMulOp>(op);
    auto lhsType = cast<RankedTensorType>(matmul.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(matmul.getRhs().getType());
    SmallVector<int64_t> extents = {lhsType.getDimSize(0),
                                    rhsType.getDimSize(1),
                                    lhsType.getDimSize(1)};
    return staticDomain(b, extents);
  }

  LogicalResult getResultTilePosition(
      Operation *op, OpBuilder &b, unsigned resultNumber,
      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
      SmallVector<OpFoldResult> &resultOffsets,
      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets = {offsets[0], offsets[1]};
    resultSizes = {sizes[0], sizes[1]};
    return success();
  }
};

} // namespace

void mlir::npu::registerNPUTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, NPUDialect *dialect) {
    Conv2DOp::attachInterface<Conv2DTilingModel>(*context);
    MatMulOp::attachInterface<MatMulTilingModel>(*context);
    AddOp::attachInterface<ElementwiseTilingModel<AddOp>>(*context);
    MulOp::attachInterface<ElementwiseTilingModel<MulOp>>(*context);
    ReluOp::attachInterface<ElementwiseTilingModel<ReluOp>>(*context);
    MaxPool2DOp::attachInterface<PoolTilingModel<MaxPool2DOp>>(*context);
    AvgPool2DOp::attachInterface<PoolTilingModel<AvgPool2DOp>>(*context);
    TransposeOp::attachInterface<ShapeOnlyTilingModel<TransposeOp>>(*context);
    ConcatOp::attachInterface<ShapeOnlyTilingModel<ConcatOp>>(*context);
    BatchNormOp::attachInterface<ShapeOnlyTilingModel<BatchNormOp>>(*context);
  });

  // The models slice with tensor.extract_slice, so a context that loads the
  // npu dialect and then tiles needs the tensor dialect loaded too. Declaring
  // the dependency here means the caller does not have to know it.
  registry.insert<tensor::TensorDialect>();
}
