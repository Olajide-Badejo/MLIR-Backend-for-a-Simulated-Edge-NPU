//===- NPUOps.cpp - npu dialect operations ----------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The verifiers, and the type inference that shares its arithmetic with them.
//
// The rule every message in this file follows: name the operation and quote the
// offending numbers. "invalid shape" tells the reader that something is wrong
// and nothing about what, which means the next step is always to open the
// verifier and read it, and at that point the diagnostic did no work at all.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::npu;

#define GET_OP_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared verification helpers.
//===----------------------------------------------------------------------===//

namespace {

/// The destination rule, verified identically on every compute operation:
/// the destination type equals the result type exactly, including any layout
/// encoding. "Exactly" is the operative word. Two tensors of the same shape and
/// element type but different layout encodings are different types here, and
/// accepting the pair would mean the tiling loop inserting an NCHW tile into an
/// NHWC destination.
LogicalResult verifyDestinationMatchesResult(Operation *op, Type destination,
                                             Type result) {
  if (destination == result)
    return success();

  return op->emitOpError()
         << "destination type must equal the result type exactly, including "
            "any layout encoding, but the destination is "
         << destination << " and the result is " << result;
}

/// The partition rule of destination passing style: `ins` and `outs` together
/// cover every operand exactly once, and no operand appears in both.
///
/// The interface's own accessors are what a pass will use, so this check runs
/// through them rather than reimplementing the split from the operand list. A
/// bug in getDpsInitsMutable is precisely what this is meant to catch, and a
/// check that recomputed the split from scratch would agree with itself and
/// miss it.
LogicalResult verifyDpsPartition(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps)
    return op->emitOpError()
           << "is a compute operation and must implement "
              "DestinationStyleOpInterface";

  const int64_t numOperands = op->getNumOperands();
  SmallVector<int> coverage(numOperands, 0);

  for (OpOperand *init : llvm::map_range(
           llvm::seq<int64_t>(0, dps.getNumDpsInits()),
           [&](int64_t i) { return dps.getDpsInitOperand(i); }))
    ++coverage[init->getOperandNumber()];

  for (OpOperand *input : dps.getDpsInputOperands())
    ++coverage[input->getOperandNumber()];

  for (int64_t i = 0; i < numOperands; ++i) {
    if (coverage[i] == 1)
      continue;
    return op->emitOpError()
           << "ins and outs must partition the operands exactly once, but "
              "operand "
           << i << " is covered " << coverage[i] << " times";
  }
  return success();
}

/// Rank 4 only for a layout encoding, and no mixed layouts among the operands.
///
/// Both halves are here rather than split, because they are the same question
/// asked of every operand: is this tensor's encoding meaningful, and does it
/// agree with the others. `referenceName` and `reference` are the operand the
/// others are compared against, which is always the first tensor operand of
/// rank 4, so that the message can name a concrete pair rather than saying that
/// the set disagrees.
LogicalResult verifyLayoutConsistency(Operation *op,
                                      ArrayRef<StringRef> names,
                                      ArrayRef<Value> values) {
  RankedTensorType reference;
  StringRef referenceName;

  for (auto [name, value] : llvm::zip_equal(names, values)) {
    if (!value)
      continue;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type)
      continue;

    if (type.getEncoding()) {
      if (!isa<LayoutAttr>(type.getEncoding()))
        return op->emitOpError()
               << "operand " << name << " carries an encoding that is not an "
               << "#npu.layout attribute: " << type.getEncoding();
      if (type.getRank() != 4)
        return op->emitOpError()
               << "a layout encoding is only meaningful on a rank 4 tensor, "
                  "but operand "
               << name << " has rank " << type.getRank() << " and carries "
               << describeLayout(type);
    }

    if (type.getRank() != 4)
      continue;

    if (!reference) {
      reference = type;
      referenceName = name;
      continue;
    }
    if (!sameLayout(reference, type))
      return op->emitOpError()
             << "operands must not mix layouts, but " << referenceName << " is "
             << describeLayout(reference) << " and " << name << " is "
             << describeLayout(type);
  }
  return success();
}

/// Checks that a windowed operation's attribute arrays have the shape the
/// arithmetic expects: two entries for kernel, strides and dilations, four for
/// pads, in ONNX order.
LogicalResult verifyWindowAttributeArity(Operation *op,
                                         ArrayRef<int64_t> kernel,
                                         ArrayRef<int64_t> strides,
                                         ArrayRef<int64_t> dilations,
                                         ArrayRef<int64_t> pads) {
  if (kernel.size() != 2)
    return op->emitOpError() << "expects a kernel of 2 entries, height then "
                                "width, but got "
                             << kernel.size();
  if (strides.size() != 2)
    return op->emitOpError()
           << "expects 2 strides, height then width, but got " << strides.size();
  if (dilations.size() != 2)
    return op->emitOpError() << "expects 2 dilations, height then width, but "
                                "got "
                             << dilations.size();
  if (pads.size() != 4)
    return op->emitOpError()
           << "expects 4 pads in ONNX order, top, left, bottom, right, but got "
           << pads.size();
  return success();
}

/// Turns a window arithmetic failure into a diagnostic that names the operation
/// and the axis and quotes the numbers.
LogicalResult emitWindowError(Operation *op, const WindowedShapeResult &shape) {
  static constexpr StringRef axisNames[] = {"height", "width"};
  StringRef axisName = shape.failedAxis < 2 ? axisNames[shape.failedAxis]
                                            : StringRef("spatial axis");
  return op->emitOpError() << "on the " << axisName << " axis, "
                           << describeWindowError(shape.error,
                                                  shape.failedParams);
}

/// The batch agreement rule of Section 7.3, applied to every operation that has
/// an input and a result of the same rank. Batch is a first class dimension
/// here from the first commit, so an operation whose input and result disagree
/// on N is refused by name rather than producing a result nobody can explain.
LogicalResult verifyBatchAgreement(Operation *op, RankedTensorType input,
                                   RankedTensorType result) {
  if (input.getRank() < 1 || result.getRank() < 1)
    return success();
  const int64_t inputBatch = input.getDimSize(0);
  const int64_t resultBatch = result.getDimSize(0);
  if (inputBatch == resultBatch)
    return success();
  return op->emitOpError() << "input and result must agree on the batch "
                              "extent, but the input has "
                           << inputBatch << " and the result has "
                           << resultBatch;
}

/// The shared inference body for the two pooling operations. Both have exactly
/// the same shape rule, so writing it twice would be writing two chances to
/// write it differently.
LogicalResult inferPoolReturnTypes(
    MLIRContext *context, std::optional<Location> location, Value input,
    ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> dilations, ArrayRef<int64_t> pads, bool ceilMode,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() != 4)
    return failure();
  if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2 ||
      pads.size() != 4)
    return failure();

  WindowedShapeResult shape = computeWindowedShape(
      getSpatialExtents(inputType), kernel, strides, dilations, pads, ceilMode);
  if (!shape.ok())
    return failure();

  SmallVector<int64_t> resultShape =
      buildNCHWLikeShape(inputType, getBatchExtent(inputType),
                         getChannelExtent(inputType), shape.extents);
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType(), inputType.getEncoding()));
  return success();
}

/// The shared verification body for the two pooling operations.
LogicalResult verifyPoolOp(Operation *op, Value input, Value destination,
                           Value result, ArrayRef<int64_t> kernel,
                           ArrayRef<int64_t> strides,
                           ArrayRef<int64_t> dilations, ArrayRef<int64_t> pads,
                           int64_t ceilMode) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(result.getType());

  if (failed(verifyDpsPartition(op)))
    return failure();
  if (failed(verifyLayoutConsistency(op, {"input", "destination", "result"},
                                     {input, destination, result})))
    return failure();
  if (failed(verifyDestinationMatchesResult(op, destination.getType(),
                                            result.getType())))
    return failure();

  if (inputType.getRank() != 4)
    return op->emitOpError()
           << "expects a rank 4 input, but got rank " << inputType.getRank();
  if (resultType.getRank() != 4)
    return op->emitOpError()
           << "expects a rank 4 result, but got rank " << resultType.getRank();

  if (ceilMode != 0 && ceilMode != 1)
    return op->emitOpError()
           << "ceil_mode must be 0 or 1, but got " << ceilMode;

  if (failed(verifyWindowAttributeArity(op, kernel, strides, dilations, pads)))
    return failure();

  WindowedShapeResult shape =
      computeWindowedShape(getSpatialExtents(inputType), kernel, strides,
                           dilations, pads, ceilMode == 1);
  if (!shape.ok())
    return emitWindowError(op, shape);

  if (failed(verifyBatchAgreement(op, inputType, resultType)))
    return failure();

  if (getChannelExtent(inputType) != getChannelExtent(resultType))
    return op->emitOpError()
           << "pooling does not change the channel count, but the input has "
           << getChannelExtent(inputType) << " channels and the result has "
           << getChannelExtent(resultType);

  SmallVector<int64_t> actual = getSpatialExtents(resultType);
  if (actual[0] != shape.extents[0] || actual[1] != shape.extents[1]) {
    SmallVector<int64_t> inputSpatial = getSpatialExtents(inputType);
    return op->emitOpError()
           << "result spatial extents must be " << shape.extents[0] << " by "
           << shape.extents[1] << ", computed from the input " << inputSpatial[0]
           << " by " << inputSpatial[1] << " with kernel " << kernel[0] << " by "
           << kernel[1] << ", strides " << strides[0] << " and " << strides[1]
           << ", dilations " << dilations[0] << " and " << dilations[1]
           << ", pads " << pads[0] << ", " << pads[1] << ", " << pads[2] << ", "
           << pads[3] << " and ceil_mode " << ceilMode << ", but got "
           << actual[0] << " by " << actual[1];
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// npu.constant
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ConstantOp::Adaptor adaptor(operands, attributes, properties, regions);
  // The dense elements attribute carries its own type, and the verifier
  // requires it to equal the result type, so inference here is a read rather
  // than a computation. This is what lets the assembly format print the type
  // once instead of twice.
  inferredReturnTypes.push_back(adaptor.getValue().getType());
  return success();
}

LogicalResult ConstantOp::verify() {
  if (getValue().getType() != getResult().getType())
    return emitOpError() << "value attribute type " << getValue().getType()
                         << " must equal the result type "
                         << getResult().getType();
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

//===----------------------------------------------------------------------===//
// npu.conv2d
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Conv2DOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto filterType = dyn_cast<RankedTensorType>(adaptor.getFilter().getType());
  if (!inputType || !filterType || inputType.getRank() != 4 ||
      filterType.getRank() != 4)
    return failure();

  ArrayRef<int64_t> strides = adaptor.getStrides();
  ArrayRef<int64_t> pads = adaptor.getPads();
  ArrayRef<int64_t> dilations = adaptor.getDilations();
  if (strides.size() != 2 || dilations.size() != 2 || pads.size() != 4)
    return failure();

  // The filter is (outputChannels, inputChannels / group, kernelHeight,
  // kernelWidth) and is never layout tagged, so its spatial extents are always
  // at dimensions 2 and 3.
  SmallVector<int64_t> kernel = {filterType.getDimSize(2),
                                 filterType.getDimSize(3)};

  // ceil_mode is fixed at 0 for convolution: ONNX has no ceil mode on Conv and
  // this dialect does not invent one. The same helper computes the pools with
  // the mode they do have, which is what stops the four windowed operations
  // from ever disagreeing on the same shape.
  WindowedShapeResult shape = computeWindowedShape(
      getSpatialExtents(inputType), kernel, strides, dilations, pads,
      /*ceilMode=*/false);
  if (!shape.ok())
    return failure();

  SmallVector<int64_t> resultShape =
      buildNCHWLikeShape(inputType, getBatchExtent(inputType),
                         filterType.getDimSize(0), shape.extents);
  inferredReturnTypes.push_back(RankedTensorType::get(
      resultShape, inputType.getElementType(), inputType.getEncoding()));
  return success();
}

LogicalResult Conv2DOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto filterType = cast<RankedTensorType>(getFilter().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyLayoutConsistency(getOperation(),
                                     {"input", "destination", "result"},
                                     {getInput(), getDestination(), getResult()})))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  if (inputType.getRank() != 4)
    return emitOpError() << "expects a rank 4 input, but got rank "
                         << inputType.getRank();
  if (filterType.getRank() != 4)
    return emitOpError() << "expects a rank 4 filter of the form (outputChannels"
                            ", inputChannels / group, kernelHeight, kernelWidth"
                            "), but got rank "
                         << filterType.getRank();
  if (resultType.getRank() != 4)
    return emitOpError() << "expects a rank 4 result, but got rank "
                         << resultType.getRank();
  // The filter is not layout tagged, and accepting an encoding on it would mean
  // inventing a meaning for an NHWC filter that nothing below this level reads.
  if (filterType.getEncoding())
    return emitOpError() << "the filter carries no layout, but got "
                         << describeLayout(filterType);

  const int64_t group = getGroup();
  if (group <= 0)
    return emitOpError() << "group must be strictly positive, but got " << group;

  const int64_t inputChannels = getChannelExtent(inputType);
  const int64_t outputChannels = filterType.getDimSize(0);

  if (inputChannels % group != 0)
    return emitOpError() << "group " << group
                         << " must divide the input channel count "
                         << inputChannels;
  if (outputChannels % group != 0)
    return emitOpError() << "group " << group
                         << " must divide the output channel count "
                         << outputChannels;
  if (filterType.getDimSize(1) != inputChannels / group)
    return emitOpError() << "the filter's second dimension must be the input "
                            "channel count divided by group, which is "
                         << inputChannels << " / " << group << " = "
                         << inputChannels / group << ", but got "
                         << filterType.getDimSize(1);

  ArrayRef<int64_t> strides = getStrides();
  ArrayRef<int64_t> pads = getPads();
  ArrayRef<int64_t> dilations = getDilations();
  SmallVector<int64_t> kernel = {filterType.getDimSize(2),
                                 filterType.getDimSize(3)};

  if (failed(verifyWindowAttributeArity(getOperation(), kernel, strides,
                                        dilations, pads)))
    return failure();

  WindowedShapeResult shape =
      computeWindowedShape(getSpatialExtents(inputType), kernel, strides,
                           dilations, pads, /*ceilMode=*/false);
  if (!shape.ok())
    return emitWindowError(getOperation(), shape);

  if (failed(verifyBatchAgreement(getOperation(), inputType, resultType)))
    return failure();

  if (getChannelExtent(resultType) != outputChannels)
    return emitOpError() << "result channel count must be the filter's output "
                            "channel count "
                         << outputChannels << ", but got "
                         << getChannelExtent(resultType);

  SmallVector<int64_t> actual = getSpatialExtents(resultType);
  if (actual[0] != shape.extents[0] || actual[1] != shape.extents[1])
    return emitOpError() << "result spatial extents must be "
                         << shape.extents[0] << " by " << shape.extents[1]
                         << ", but got " << actual[0] << " by " << actual[1];

  if (getBias()) {
    auto biasType = cast<RankedTensorType>(getBias().getType());
    if (biasType.getRank() != 1)
      return emitOpError() << "the bias must be rank 1, but got rank "
                           << biasType.getRank();
    if (biasType.getDimSize(0) != outputChannels)
      return emitOpError() << "the bias length must equal the output channel "
                              "count "
                           << outputChannels << ", but got "
                           << biasType.getDimSize(0);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// npu.matmul
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  MatMulOp::Adaptor adaptor(operands, attributes, properties, regions);

  auto lhsType = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return failure();
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return failure();

  inferredReturnTypes.push_back(RankedTensorType::get(
      {lhsType.getDimSize(0), rhsType.getDimSize(1)},
      lhsType.getElementType()));
  return success();
}

LogicalResult MatMulOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2)
    return emitOpError() << "is rank 2 by rank 2, but got ranks "
                         << lhsType.getRank() << ", " << rhsType.getRank()
                         << " and " << resultType.getRank();

  const int64_t m = lhsType.getDimSize(0);
  const int64_t k = lhsType.getDimSize(1);
  const int64_t n = rhsType.getDimSize(1);

  if (rhsType.getDimSize(0) != k)
    return emitOpError() << "the contracted dimensions must agree, but the lhs "
                            "is "
                         << m << " by " << k << " and the rhs is "
                         << rhsType.getDimSize(0) << " by " << n;

  if (resultType.getDimSize(0) != m || resultType.getDimSize(1) != n)
    return emitOpError() << "result must be " << m << " by " << n
                         << ", but got " << resultType.getDimSize(0) << " by "
                         << resultType.getDimSize(1);

  if (getBias()) {
    auto biasType = cast<RankedTensorType>(getBias().getType());
    if (biasType.getRank() != 1)
      return emitOpError() << "the bias must be rank 1, but got rank "
                           << biasType.getRank();
    if (biasType.getDimSize(0) != n)
      return emitOpError() << "the bias length must equal the output column "
                              "count "
                           << n << ", but got " << biasType.getDimSize(0);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// The elementwise operations.
//===----------------------------------------------------------------------===//

namespace {

LogicalResult verifyElementwiseBinary(Operation *op, Value lhs, Value rhs,
                                      Value destination, Value result) {
  if (failed(verifyDpsPartition(op)))
    return failure();
  if (failed(verifyLayoutConsistency(
          op, {"lhs", "rhs", "destination", "result"},
          {lhs, rhs, destination, result})))
    return failure();
  if (failed(verifyDestinationMatchesResult(op, destination.getType(),
                                            result.getType())))
    return failure();

  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  auto resultType = cast<RankedTensorType>(result.getType());

  if (lhsType.getShape() != resultType.getShape())
    return op->emitOpError()
           << "does not broadcast, so the lhs shape must equal the result "
              "shape, but the lhs is "
           << lhsType << " and the result is " << resultType;
  if (rhsType.getShape() != resultType.getShape())
    return op->emitOpError()
           << "does not broadcast, so the rhs shape must equal the result "
              "shape, but the rhs is "
           << rhsType << " and the result is " << resultType;

  return success();
}

} // namespace

LogicalResult AddOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(),
                                 getDestination(), getResult());
}

LogicalResult MulOp::verify() {
  return verifyElementwiseBinary(getOperation(), getLhs(), getRhs(),
                                 getDestination(), getResult());
}

LogicalResult ReluOp::verify() {
  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyLayoutConsistency(getOperation(),
                                     {"input", "destination", "result"},
                                     {getInput(), getDestination(), getResult()})))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (inputType.getShape() != resultType.getShape())
    return emitOpError() << "is elementwise, so the input shape must equal the "
                            "result shape, but the input is "
                         << inputType << " and the result is " << resultType;
  return success();
}

//===----------------------------------------------------------------------===//
// The pooling operations.
//===----------------------------------------------------------------------===//

LogicalResult MaxPool2DOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  MaxPool2DOp::Adaptor adaptor(operands, attributes, properties, regions);
  return inferPoolReturnTypes(context, location, adaptor.getInput(),
                              adaptor.getKernel(), adaptor.getStrides(),
                              adaptor.getDilations(), adaptor.getPads(),
                              adaptor.getCeilMode() == 1, inferredReturnTypes);
}

LogicalResult MaxPool2DOp::verify() {
  return verifyPoolOp(getOperation(), getInput(), getDestination(), getResult(),
                      getKernel(), getStrides(), getDilations(), getPads(),
                      getCeilMode());
}

LogicalResult AvgPool2DOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  AvgPool2DOp::Adaptor adaptor(operands, attributes, properties, regions);
  return inferPoolReturnTypes(context, location, adaptor.getInput(),
                              adaptor.getKernel(), adaptor.getStrides(),
                              adaptor.getDilations(), adaptor.getPads(),
                              adaptor.getCeilMode() == 1, inferredReturnTypes);
}

LogicalResult AvgPool2DOp::verify() {
  return verifyPoolOp(getOperation(), getInput(), getDestination(), getResult(),
                      getKernel(), getStrides(), getDilations(), getPads(),
                      getCeilMode());
}

//===----------------------------------------------------------------------===//
// npu.reshape
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError() << "moves no data, so the element types must match, "
                            "but the input is "
                         << inputType.getElementType() << " and the result is "
                         << resultType.getElementType();

  const int64_t inputElements = inputType.getNumElements();
  const int64_t resultElements = resultType.getNumElements();
  if (inputElements != resultElements)
    return emitOpError() << "element counts must match, but the input "
                         << inputType << " has " << inputElements
                         << " elements and the result " << resultType << " has "
                         << resultElements;

  // A reshape rearranges extents, so a layout encoding on either side would be
  // claiming that the rearrangement preserved a meaning it does not preserve.
  if (inputType.getEncoding() || resultType.getEncoding())
    return emitOpError() << "carries no layout encoding on either side, "
                            "because a reshape does not preserve one, but the "
                            "input is "
                         << describeLayout(inputType) << " and the result is "
                         << describeLayout(resultType);

  return success();
}

//===----------------------------------------------------------------------===//
// npu.transpose
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  ArrayRef<int64_t> permutation = getPermutation();

  if (inputType.getElementType() != resultType.getElementType())
    return emitOpError() << "moves data without converting it, so the element "
                            "types must match, but the input is "
                         << inputType.getElementType() << " and the result is "
                         << resultType.getElementType();

  const int64_t rank = resultType.getRank();
  if (static_cast<int64_t>(permutation.size()) != rank)
    return emitOpError() << "the permutation must have exactly the result rank "
                         << rank << " entries, but got " << permutation.size();
  if (inputType.getRank() != rank)
    return emitOpError() << "input and result must have the same rank, but got "
                         << inputType.getRank() << " and " << rank;

  // A permutation of exactly the result rank: every index appears once.
  SmallVector<bool> seen(rank, false);
  for (auto [position, index] : llvm::enumerate(permutation)) {
    if (index < 0 || index >= rank)
      return emitOpError() << "permutation entry " << position << " is "
                           << index << ", which is outside the range 0 to "
                           << rank - 1;
    if (seen[index])
      return emitOpError() << "the permutation must be a permutation, but "
                           << index << " appears more than once";
    seen[index] = true;
  }

  for (auto [position, index] : llvm::enumerate(permutation)) {
    const int64_t expected = inputType.getDimSize(index);
    const int64_t actual = resultType.getDimSize(position);
    if (expected != actual)
      return emitOpError() << "result extent " << position << " must be input "
                              "extent "
                           << index << ", which is " << expected
                           << ", but got " << actual;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// npu.concat
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  auto resultType = cast<RankedTensorType>(getResult().getType());
  const int64_t rank = resultType.getRank();
  const int64_t axis = getAxis();

  if (getInputs().empty())
    return emitOpError() << "requires at least one input";

  if (axis < 0 || axis >= rank)
    return emitOpError() << "axis must be in the range 0 to " << rank - 1
                         << " for a rank " << rank << " result, but got "
                         << axis;

  // The layout consistency check wants names, and the operand count is not
  // known until run time, so the names are built here rather than being a
  // static list.
  SmallVector<std::string> ownedNames;
  SmallVector<StringRef> names;
  SmallVector<Value> values;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    ownedNames.push_back(("input " + Twine(index)).str());
    values.push_back(input);
  }
  ownedNames.push_back("destination");
  values.push_back(getDestination());
  ownedNames.push_back("result");
  values.push_back(getResult());
  for (const std::string &name : ownedNames)
    names.push_back(name);
  if (failed(verifyLayoutConsistency(getOperation(), names, values)))
    return failure();

  int64_t concatenated = 0;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    auto inputType = cast<RankedTensorType>(input.getType());

    if (inputType.getElementType() != resultType.getElementType())
      return emitOpError() << "input " << index << " has element type "
                           << inputType.getElementType()
                           << ", which differs from the result element type "
                           << resultType.getElementType();
    if (inputType.getRank() != rank)
      return emitOpError() << "input " << index << " has rank "
                           << inputType.getRank()
                           << ", but every input must have the result rank "
                           << rank;

    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      if (dimension == axis)
        continue;
      if (inputType.getDimSize(dimension) != resultType.getDimSize(dimension))
        return emitOpError()
               << "input " << index << " has extent "
               << inputType.getDimSize(dimension) << " on axis " << dimension
               << ", but the result has " << resultType.getDimSize(dimension)
               << ", and extents must match on every axis except the "
                  "concatenation axis "
               << axis;
    }
    concatenated += inputType.getDimSize(axis);
  }

  if (concatenated != resultType.getDimSize(axis))
    return emitOpError() << "the input extents along axis " << axis
                         << " sum to " << concatenated
                         << ", but the result extent there is "
                         << resultType.getDimSize(axis);

  return success();
}

//===----------------------------------------------------------------------===//
// npu.batch_norm
//===----------------------------------------------------------------------===//

LogicalResult BatchNormOp::verify() {
  if (failed(verifyDpsPartition(getOperation())))
    return failure();
  if (failed(verifyLayoutConsistency(getOperation(),
                                     {"input", "destination", "result"},
                                     {getInput(), getDestination(), getResult()})))
    return failure();
  if (failed(verifyDestinationMatchesResult(
          getOperation(), getDestination().getType(), getResult().getType())))
    return failure();

  auto inputType = cast<RankedTensorType>(getInput().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  if (inputType.getRank() != 4)
    return emitOpError() << "expects a rank 4 input, but got rank "
                         << inputType.getRank();
  if (inputType.getShape() != resultType.getShape())
    return emitOpError() << "is elementwise over the channel parameters, so the "
                            "input shape must equal the result shape, but the "
                            "input is "
                         << inputType << " and the result is " << resultType;

  const int64_t channels = getChannelExtent(inputType);

  static constexpr StringRef parameterNames[] = {"gamma", "beta", "mean",
                                                 "variance"};
  const Value parameters[] = {getGamma(), getBeta(), getMean(), getVariance()};
  for (auto [name, parameter] : llvm::zip_equal(parameterNames, parameters)) {
    auto parameterType = cast<RankedTensorType>(parameter.getType());
    if (parameterType.getRank() != 1)
      return emitOpError() << name << " must be rank 1, but got rank "
                           << parameterType.getRank();
    if (parameterType.getDimSize(0) != channels)
      return emitOpError() << name << " must have length equal to the channel "
                                      "count "
                           << channels << ", but got "
                           << parameterType.getDimSize(0);
  }

  if (!getEpsilon().isFinite() || getEpsilon().isNegative())
    return emitOpError() << "epsilon must be finite and non negative, but got "
                         << getEpsilon().convertToFloat();

  return success();
}

//===----------------------------------------------------------------------===//
// npu.fused_op
//===----------------------------------------------------------------------===//

LogicalResult FusedOp::verify() {
  Region &body = getBody();

  if (body.empty() || body.front().empty())
    return emitOpError() << "requires a non empty region";

  Block &block = body.front();

  if (block.getNumArguments() != getInputs().size())
    return emitOpError() << "the region takes one block argument per operand, "
                            "so it must have "
                         << getInputs().size() << " block arguments, but it has "
                         << block.getNumArguments();

  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(getInputs(), block.getArguments()))) {
    Type operandType = std::get<0>(pair).getType();
    Type argumentType = std::get<1>(pair).getType();
    if (operandType != argumentType)
      return emitOpError() << "block argument " << index << " has type "
                           << argumentType << ", which differs from operand "
                           << index << " of type " << operandType;
  }

  for (Operation &nested : block.without_terminator()) {
    if (nested.getDialect() != getOperation()->getDialect())
      return emitOpError() << "the region holds only npu dialect operations, "
                              "but it contains "
                           << nested.getName();
  }

  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return emitOpError() << "the region must be terminated by npu.yield, but "
                            "it is terminated by "
                         << block.getTerminator()->getName();

  if (yield.getValue().getType() != getResult().getType())
    return emitOpError() << "the yielded type " << yield.getValue().getType()
                         << " must equal the result type "
                         << getResult().getType();

  return success();
}
