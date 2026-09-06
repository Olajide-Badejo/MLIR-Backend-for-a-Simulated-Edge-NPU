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
// **This file was written at P1 and completed at P13**, which is the split
// Section 7.2 asks for by name: `TilingInterface` is implemented at P1 and
// consumed at P13, deliberately, so that an interface bug and a policy bug
// cannot be mistaken for each other. P1's own note said the halo arithmetic
// belonged with the pass that would exercise it and that returning a wrong tile
// would be worse than returning none. P13 is that phase and the arithmetic is
// here now, with the property that makes it right asserted rather than
// described.
//
// - `getLoopIteratorTypes` and `getIterationDomain` are complete for every
//   operation. They are the introspection half, and they are what the tiling
//   pass reads to decide what it is allowed to split.
// - `getTiledImplementation` and `getResultTilePosition` are implemented for
//   the elementwise operations, whose iteration domain is exactly their result
//   shape and whose every dimension is parallel. For those, a tile of the
//   iteration space is a tile of every operand and a tile of the result, and
//   the slicing is honest.
// - `getTiledImplementation` is implemented for the convolution, both pools and
//   the matmul, **over the parallel dimensions only**. That restriction is
//   Section 13.2's and it lives here rather than in the pass, because the
//   interface is what knows whether a tile of the domain is expressible and a
//   pass that had to know it would be a second copy of the same judgement.
//   Under fp32 addition is not associative, so splitting the input channel, the
//   kernel window or a matmul's inner dimension re associates the accumulation
//   and moves every golden file; Section 13.2 permits that only behind
//   `allow-reduction-tiling` with a documented accumulation order and its own
//   golden set, so it is declined here and cannot be asked for by accident.
//   **Declining is a result rather than a failure**, and the caller's fallback
//   is the allocator's spilling.
// - For the three shape only operations, transpose, concat and batch_norm,
//   `getTiledImplementation` is still absent. Each needs its own operand side
//   arithmetic and none of the three is what a scratchpad budget runs out on:
//   the buffers that overflow in Section 15's suite are convolution
//   activations and one 400 by 120 weight matrix.
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

/// The same, from plain integers, which is what every extent in this dialect
/// is. Returning early on a non constant is not a case that arises here and
/// building the attributes at each call site would be the same three lines
/// written out nine times.
Value sliceOperand(OpBuilder &builder, Location loc, Value source,
                   ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
  SmallVector<OpFoldResult> offsetAttrs;
  SmallVector<OpFoldResult> sizeAttrs;
  for (int64_t offset : offsets)
    offsetAttrs.push_back(builder.getIndexAttr(offset));
  for (int64_t size : sizes)
    sizeAttrs.push_back(builder.getIndexAttr(size));
  return sliceOperand(builder, loc, source, offsetAttrs, sizeAttrs);
}

/// Every entry of an offset or size list as an integer, or nothing when one of
/// them is dynamic. Every iteration domain in this dialect is static, so a
/// dynamic entry means the caller is asking for something this dialect cannot
/// represent, and declining is the honest answer.
std::optional<SmallVector<int64_t>> constantsOf(ArrayRef<OpFoldResult> values) {
  SmallVector<int64_t> constants;
  constants.reserve(values.size());
  for (OpFoldResult value : values) {
    std::optional<int64_t> constant = getConstantIntValue(value);
    if (!constant)
      return std::nullopt;
    constants.push_back(*constant);
  }
  return constants;
}

//===----------------------------------------------------------------------===//
// The halo arithmetic, in one place.
//===----------------------------------------------------------------------===//

/// Which stretch of one input axis a tiled output range reads, and the pads
/// that stretch carries once it has been sliced out.
///
/// **This is the arithmetic P1 declined to write and named the reason for.**
/// Its comment says returning a wrong tile would be worse than returning none,
/// because a pass would consume it and produce a program with a quietly wrong
/// answer. So the property that makes it right is stated here and asserted in
/// `unittests/Dialect/NPU/TilingInterfaceTest.cpp` rather than left to a
/// reader:
///
///     the tile's own windowed extent, computed by `computeWindowedExtent` from
///     the sliced input extent and the pads below, equals `outputSize`
///
/// and the stronger property underneath it, which is what makes tiling exact:
///
///     output position `outputOffset + j` reads the same input positions in the
///     tile as it did in the whole, and the positions it reads outside the
///     input are the same ones
///
/// The second is what an average pool depends on, because it divides by the
/// number of elements that actually contributed rather than by the window area,
/// so a tile that turned a real element into a padded one would change a
/// divisor rather than only a sum.
///
/// The derivation is four lines. With `e` the effective kernel extent, the
/// tile's first window starts at `first = outputOffset * stride - padBegin` and
/// its last window ends at `last = (outputOffset + outputSize - 1) * stride -
/// padBegin + e`. Clamping those to the input gives the slice, and the clamped
/// amounts are exactly the pads the tile carries, so
///
///     size + padBegin' + padEnd' - e = last - first - e = (outputSize - 1) * stride
///
/// and the floor division by `stride` recovers `outputSize`.
struct WindowSlice {
  /// Into the untiled input, clamped at zero.
  int64_t offset = 0;
  /// The clamped extent, always at least one.
  int64_t size = 0;
  /// The padding the tile carries at its start, never more than the original's.
  int64_t padBegin = 0;
  int64_t padEnd = 0;
};

std::optional<WindowSlice> windowSlice(int64_t outputOffset, int64_t outputSize,
                                       int64_t inputExtent, int64_t kernel,
                                       int64_t stride, int64_t dilation,
                                       int64_t padBegin) {
  if (outputSize <= 0 || inputExtent <= 0 || kernel <= 0 || stride <= 0 ||
      dilation <= 0)
    return std::nullopt;

  const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
  const int64_t first = outputOffset * stride - padBegin;
  const int64_t last =
      (outputOffset + outputSize - 1) * stride - padBegin + effectiveKernel;

  const int64_t begin = std::max<int64_t>(0, first);
  const int64_t end = std::min<int64_t>(inputExtent, last);
  // A tile whose every window lies entirely in the padding has no input to
  // slice. The verifier's rule that a pad is smaller than its kernel makes this
  // unreachable for a well formed operation, and it is guarded rather than
  // asserted because an empty `tensor.extract_slice` is not a diagnosable error
  // downstream, it is a tensor of no elements that verifies.
  if (end <= begin)
    return std::nullopt;

  WindowSlice slice;
  slice.offset = begin;
  slice.size = end - begin;
  slice.padBegin = begin - first;
  slice.padEnd = last - end;
  return slice;
}

/// Places a rank 4 activation's four extents in the order its layout names.
/// Reading a shape without reading its encoding is always wrong, which is why
/// every caller here goes through this rather than indexing by hand.
SmallVector<int64_t> placeRank4(RankedTensorType like, int64_t batch,
                                int64_t channel, int64_t height,
                                int64_t width) {
  if (isNHWC(like))
    return {batch, height, width, channel};
  return {batch, channel, height, width};
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

  /// The tile, over the parallel dimensions only.
  ///
  /// **Section 13.2's rule is implemented here rather than in the pass**, and
  /// that placement is the point: the interface is what knows whether a tile of
  /// the domain is expressible, and a pass that had to know it would be a
  /// second copy of the same judgement. Parallel dimensions, which is batch,
  /// group, per group output channel and the two output spatial axes, are
  /// always tileable. **A tile that splits the reduction is declined**, because
  /// under fp32 addition is not associative and splitting the input channel or
  /// the kernel window re associates the accumulation, which moves every golden
  /// file. Section 13.2 permits that only behind `allow-reduction-tiling` with
  /// its own golden set, so it is not something this model can be asked for by
  /// accident.
  ///
  /// **Declining is a result, not a failure.** The caller's fallback is the
  /// allocator's spilling, which Section 13.2 names.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto conv = cast<Conv2DOp>(op);
    Location loc = op->getLoc();

    std::optional<SmallVector<int64_t>> offsetsOr = constantsOf(offsets);
    std::optional<SmallVector<int64_t>> sizesOr = constantsOf(sizes);
    if (!offsetsOr || !sizesOr || offsetsOr->size() != 8 ||
        sizesOr->size() != 8)
      return failure();
    ArrayRef<int64_t> tileOffsets = *offsetsOr;
    ArrayRef<int64_t> tileSizes = *sizesOr;

    auto inputType = cast<RankedTensorType>(conv.getInput().getType());
    auto filterType = cast<RankedTensorType>(conv.getFilter().getType());
    auto resultType = cast<RankedTensorType>(conv.getResult().getType());

    const int64_t group = conv.getGroup();
    const int64_t outputChannels = filterType.getDimSize(0);
    const int64_t channelsPerGroup = outputChannels / group;
    const int64_t inputChannelsPerGroup = filterType.getDimSize(1);
    const int64_t kernelHeight = filterType.getDimSize(2);
    const int64_t kernelWidth = filterType.getDimSize(3);

    // The three reduction dimensions of the domain, which this model tiles at
    // their full extent or not at all.
    const int64_t reductionExtents[3] = {inputChannelsPerGroup, kernelHeight,
                                         kernelWidth};
    for (unsigned index = 0; index < 3; ++index) {
      if (tileOffsets[5 + index] != 0 ||
          tileSizes[5 + index] != reductionExtents[index])
        return failure();
    }

    // The same folding rule `getResultTilePosition` applies, and it has to be
    // the same one: a tile whose result position that function refuses is a
    // tile whose result cannot be written back.
    const int64_t groupOffset = tileOffsets[1];
    const int64_t groupSize = tileSizes[1];
    const int64_t channelOffset = tileOffsets[2];
    const int64_t channelSize = tileSizes[2];
    if (groupSize != 1 && channelSize != channelsPerGroup)
      return failure();
    const int64_t foldedOffset = groupOffset * channelsPerGroup + channelOffset;
    const int64_t foldedSize = groupSize * channelSize;

    ArrayRef<int64_t> strides = conv.getStrides();
    ArrayRef<int64_t> pads = conv.getPads();
    ArrayRef<int64_t> dilations = conv.getDilations();
    SmallVector<int64_t> inputSpatial = getSpatialExtents(inputType);

    // `pads` is four entries in ONNX order: padTop, padLeft, padBottom,
    // padRight. Only the two begin pads take part in the window arithmetic;
    // the end pads fall out of it, which is why they are not read here.
    std::optional<WindowSlice> height =
        windowSlice(tileOffsets[3], tileSizes[3], inputSpatial[0], kernelHeight,
                    strides[0], dilations[0], pads[0]);
    std::optional<WindowSlice> width =
        windowSlice(tileOffsets[4], tileSizes[4], inputSpatial[1], kernelWidth,
                    strides[1], dilations[1], pads[1]);
    if (!height || !width)
      return failure();

    SmallVector<Operation *> generated;

    // The input: the tile's batch, the whole of every group it covers, and the
    // halo window the two axes above computed.
    const int64_t inputChannelOffset = groupOffset * inputChannelsPerGroup;
    const int64_t inputChannelSize = groupSize * inputChannelsPerGroup;
    Value tiledInput = sliceOperand(
        b, loc, conv.getInput(),
        placeRank4(inputType, tileOffsets[0], inputChannelOffset, height->offset,
                   width->offset),
        placeRank4(inputType, tileSizes[0], inputChannelSize, height->size,
                   width->size));
    generated.push_back(tiledInput.getDefiningOp());

    // The filter is not layout tagged, so its axes are always
    // (outputChannels, inputChannels / group, kernelHeight, kernelWidth).
    Value tiledFilter = sliceOperand(
        b, loc, conv.getFilter(), {foldedOffset, 0, 0, 0},
        {foldedSize, inputChannelsPerGroup, kernelHeight, kernelWidth});
    generated.push_back(tiledFilter.getDefiningOp());

    Value tiledBias;
    if (Value bias = conv.getBias()) {
      tiledBias = sliceOperand(b, loc, bias, {foldedOffset}, {foldedSize});
      generated.push_back(tiledBias.getDefiningOp());
    }

    SmallVector<int64_t> destinationOffsets = placeRank4(
        resultType, tileOffsets[0], foldedOffset, tileOffsets[3], tileOffsets[4]);
    SmallVector<int64_t> destinationSizes = placeRank4(
        resultType, tileSizes[0], foldedSize, tileSizes[3], tileSizes[4]);
    Value tiledDestination = sliceOperand(b, loc, conv.getDestination(),
                                          destinationOffsets, destinationSizes);
    generated.push_back(tiledDestination.getDefiningOp());

    auto tiledResultType =
        RankedTensorType::get(destinationSizes, resultType.getElementType(),
                              resultType.getEncoding());

    const SmallVector<int64_t> tilePads = {height->padBegin, width->padBegin,
                                           height->padEnd, width->padEnd};

    // The tile's group count is the number of whole groups it covers, which is
    // one whenever the tile sits inside a group. That is the same case
    // distinction the folding rule above makes and it is not a second one.
    auto tiled = Conv2DOp::create(b, loc, tiledResultType, tiledInput,
                                  tiledFilter, tiledBias, strides, tilePads,
                                  dilations, static_cast<uint64_t>(groupSize),
                                  tiledDestination);

    return TilingResult{{tiled.getOperation()},
                        SmallVector<Value>(tiled->getResults()), generated};
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

  /// The tile, over batch, channel and the two output spatial axes.
  ///
  /// The window arithmetic is the convolution's, because Section 7's shared
  /// extent formula is the convolution's, and a pool that computed its halo
  /// differently would be the eight ways of writing one formula that
  /// `NPUShapeUtils.h` exists to prevent.
  ///
  /// **`ceil_mode = 1` is declined.** That mode adds the rule that a window
  /// whose first element would start inside the right padded region is dropped,
  /// which changes how many windows there are rather than what one computes, so
  /// a tile's own extent is no longer the extent it was asked for. The
  /// arithmetic to carry it through a tiling is real work with no user: nothing
  /// in Section 15's suite imports a pool with a ceil mode, and the frontend
  /// would have to accept one first.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto pool = cast<OpTy>(op);
    Location loc = op->getLoc();

    if (pool.getCeilMode() != 0)
      return failure();

    std::optional<SmallVector<int64_t>> offsetsOr = constantsOf(offsets);
    std::optional<SmallVector<int64_t>> sizesOr = constantsOf(sizes);
    if (!offsetsOr || !sizesOr || offsetsOr->size() != 6 ||
        sizesOr->size() != 6)
      return failure();
    ArrayRef<int64_t> tileOffsets = *offsetsOr;
    ArrayRef<int64_t> tileSizes = *sizesOr;

    auto inputType = cast<RankedTensorType>(pool.getInput().getType());
    auto resultType = cast<RankedTensorType>(pool.getResult().getType());

    ArrayRef<int64_t> kernel = pool.getKernel();
    ArrayRef<int64_t> strides = pool.getStrides();
    ArrayRef<int64_t> pads = pool.getPads();
    ArrayRef<int64_t> dilations = pool.getDilations();

    // The window is the reduction, so it is tiled whole or not at all.
    if (tileOffsets[4] != 0 || tileSizes[4] != kernel[0] ||
        tileOffsets[5] != 0 || tileSizes[5] != kernel[1])
      return failure();

    SmallVector<int64_t> inputSpatial = getSpatialExtents(inputType);
    std::optional<WindowSlice> height =
        windowSlice(tileOffsets[2], tileSizes[2], inputSpatial[0], kernel[0],
                    strides[0], dilations[0], pads[0]);
    std::optional<WindowSlice> width =
        windowSlice(tileOffsets[3], tileSizes[3], inputSpatial[1], kernel[1],
                    strides[1], dilations[1], pads[1]);
    if (!height || !width)
      return failure();

    SmallVector<Operation *> generated;

    Value tiledInput = sliceOperand(
        b, loc, pool.getInput(),
        placeRank4(inputType, tileOffsets[0], tileOffsets[1], height->offset,
                   width->offset),
        placeRank4(inputType, tileSizes[0], tileSizes[1], height->size,
                   width->size));
    generated.push_back(tiledInput.getDefiningOp());

    SmallVector<int64_t> destinationOffsets =
        placeRank4(resultType, tileOffsets[0], tileOffsets[1], tileOffsets[2],
                   tileOffsets[3]);
    SmallVector<int64_t> destinationSizes = placeRank4(
        resultType, tileSizes[0], tileSizes[1], tileSizes[2], tileSizes[3]);
    Value tiledDestination = sliceOperand(b, loc, pool.getDestination(),
                                          destinationOffsets, destinationSizes);
    generated.push_back(tiledDestination.getDefiningOp());

    auto tiledResultType =
        RankedTensorType::get(destinationSizes, resultType.getElementType(),
                              resultType.getEncoding());

    const SmallVector<int64_t> tilePads = {height->padBegin, width->padBegin,
                                           height->padEnd, width->padEnd};

    auto tiled = OpTy::create(b, loc, tiledResultType, tiledInput, kernel,
                              strides, tilePads, dilations, /*ceil_mode=*/0,
                              tiledDestination);

    return TilingResult{{tiled.getOperation()},
                        SmallVector<Value>(tiled->getResults()), generated};
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

  /// The tile, over M and N.
  ///
  /// A matmul has no halo: a tile of the output rows reads exactly those rows
  /// of the left operand and a tile of the output columns reads exactly those
  /// columns of the right one, so the slicing is the identity on both axes and
  /// there is nothing to clamp. **K is declined for the convolution's reason**
  /// and it is the same reason stated once: splitting the inner dimension re
  /// associates an fp32 accumulation, which Section 13.2 permits only behind
  /// `allow-reduction-tiling` with its own golden set.
  ///
  /// This is the model that matters most for the fully connected tail of
  /// Section 15's suite, where a single 400 by 120 weight matrix is the largest
  /// buffer any of these programs allocates.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    auto matmul = cast<MatMulOp>(op);
    Location loc = op->getLoc();

    std::optional<SmallVector<int64_t>> offsetsOr = constantsOf(offsets);
    std::optional<SmallVector<int64_t>> sizesOr = constantsOf(sizes);
    if (!offsetsOr || !sizesOr || offsetsOr->size() != 3 ||
        sizesOr->size() != 3)
      return failure();
    ArrayRef<int64_t> tileOffsets = *offsetsOr;
    ArrayRef<int64_t> tileSizes = *sizesOr;

    auto lhsType = cast<RankedTensorType>(matmul.getLhs().getType());
    auto resultType = cast<RankedTensorType>(matmul.getResult().getType());
    const int64_t reduction = lhsType.getDimSize(1);

    if (tileOffsets[2] != 0 || tileSizes[2] != reduction)
      return failure();

    SmallVector<Operation *> generated;

    Value tiledLhs = sliceOperand(b, loc, matmul.getLhs(), {tileOffsets[0], 0},
                                  {tileSizes[0], reduction});
    generated.push_back(tiledLhs.getDefiningOp());

    Value tiledRhs = sliceOperand(b, loc, matmul.getRhs(), {0, tileOffsets[1]},
                                  {reduction, tileSizes[1]});
    generated.push_back(tiledRhs.getDefiningOp());

    Value tiledBias;
    if (Value bias = matmul.getBias()) {
      tiledBias =
          sliceOperand(b, loc, bias, {tileOffsets[1]}, {tileSizes[1]});
      generated.push_back(tiledBias.getDefiningOp());
    }

    const SmallVector<int64_t> destinationOffsets = {tileOffsets[0],
                                                     tileOffsets[1]};
    const SmallVector<int64_t> destinationSizes = {tileSizes[0], tileSizes[1]};
    Value tiledDestination = sliceOperand(b, loc, matmul.getDestination(),
                                          destinationOffsets, destinationSizes);
    generated.push_back(tiledDestination.getDefiningOp());

    auto tiledResultType =
        RankedTensorType::get(destinationSizes, resultType.getElementType(),
                              resultType.getEncoding());

    auto tiled = MatMulOp::create(b, loc, tiledResultType, tiledLhs, tiledRhs,
                                  tiledBias, tiledDestination);

    return TilingResult{{tiled.getOperation()},
                        SmallVector<Value>(tiled->getResults()), generated};
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
