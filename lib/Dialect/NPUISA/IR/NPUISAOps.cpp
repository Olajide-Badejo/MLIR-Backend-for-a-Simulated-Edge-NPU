//===- NPUISAOps.cpp - npuisa dialect operations ----------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The verifiers, the asynchronous DMA rules of Section 8, and the one
// canonicalization this dialect has.
//
// Every message here follows the rule the npu dialect's verifiers follow: name
// the operation and quote the offending numbers. A diagnostic that says only
// that something is wrong sends the reader to the verifier source to find out
// what, and at that point the diagnostic did no work at all.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::npuisa;

#define GET_OP_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Shared verification helpers.
//===----------------------------------------------------------------------===//

namespace {

/// The memref type of a value, which every operand in this dialect has by
/// construction: the ODS constraints admit nothing else.
MemRefType memRefOf(Value value) {
  return cast<MemRefType>(value.getType());
}

/// The partition rule of destination passing style: `ins` and `outs` together
/// cover every operand exactly once, and no operand appears in both.
///
/// This runs through the interface's own accessors rather than recomputing the
/// split from the operand list, for the reason the npu dialect gives about the
/// same check: a bug in getDpsInitsMutable is precisely what it is meant to
/// catch, and a check that recomputed the split from scratch would agree with
/// itself and miss it.
LogicalResult verifyDpsPartition(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps)
    return op->emitOpError()
           << "is a compute instruction and must implement "
              "DestinationStyleOpInterface";

  const int64_t numOperands = op->getNumOperands();
  SmallVector<int> coverage(numOperands, 0);

  for (int64_t i = 0, e = dps.getNumDpsInits(); i < e; ++i)
    ++coverage[dps.getDpsInitOperand(i)->getOperandNumber()];

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

/// Element types agree across a named pair of operands.
LogicalResult verifySameElementType(Operation *op, StringRef lhsName,
                                    MemRefType lhs, StringRef rhsName,
                                    MemRefType rhs) {
  if (lhs.getElementType() == rhs.getElementType())
    return success();
  return op->emitOpError()
         << "element types must agree, but " << lhsName << " has element type "
         << lhs.getElementType() << " and " << rhsName << " has element type "
         << rhs.getElementType();
}

/// Shapes agree exactly across a named pair of operands.
LogicalResult verifySameShape(Operation *op, StringRef lhsName, MemRefType lhs,
                              StringRef rhsName, MemRefType rhs) {
  if (lhs.getShape() == rhs.getShape())
    return success();
  return op->emitOpError()
         << "shapes must agree, but " << lhsName << " is " << lhs << " and "
         << rhsName << " is " << rhs;
}

/// The number of elements a statically shaped memref holds.
int64_t elementCount(MemRefType type) {
  int64_t count = 1;
  for (int64_t extent : type.getShape())
    count *= extent;
  return count;
}

/// The bias rule, shared by matmul and conv2d: rank 1 of the expected length.
LogicalResult verifyBias(Operation *op, Value bias, int64_t expectedLength,
                         StringRef what) {
  if (!bias)
    return success();
  MemRefType type = memRefOf(bias);
  if (type.getRank() != 1)
    return op->emitOpError()
           << "the bias must be rank 1, but it is " << type;
  if (type.getShape()[0] != expectedLength)
    return op->emitOpError()
           << "the bias length must equal " << what << ", which is "
           << expectedLength << ", but the bias is " << type;
  return success();
}

/// The windowed operations share their arithmetic with the npu dialect through
/// this helper, so the two levels cannot disagree about a shape across the
/// lowering. Buffers at this level carry no layout encoding, so NCHW is the
/// order: batch, channels, height, width.
LogicalResult verifyWindowedDestination(Operation *op, MemRefType input,
                                        MemRefType destination,
                                        ArrayRef<int64_t> kernel,
                                        ArrayRef<int64_t> strides,
                                        ArrayRef<int64_t> dilations,
                                        ArrayRef<int64_t> pads, bool ceilMode,
                                        int64_t expectedChannels) {
  if (input.getRank() != 4)
    return op->emitOpError()
           << "the input must be rank 4 in NCHW order, but it is " << input;
  if (destination.getRank() != 4)
    return op->emitOpError()
           << "the destination must be rank 4 in NCHW order, but it is "
           << destination;
  if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2)
    return op->emitOpError()
           << "kernel, strides and dilations must each have 2 entries, height "
              "then width, but they have "
           << kernel.size() << ", " << strides.size() << " and "
           << dilations.size();
  if (pads.size() != 4)
    return op->emitOpError()
           << "pads must have 4 entries in ONNX order, padTop, padLeft, "
              "padBottom, padRight, but it has "
           << pads.size();

  const ArrayRef<int64_t> inputShape = input.getShape();
  const ArrayRef<int64_t> destShape = destination.getShape();

  if (destShape[0] != inputShape[0])
    return op->emitOpError()
           << "the destination batch extent must equal the input batch extent, "
              "which is "
           << inputShape[0] << ", but the destination is " << destination;
  if (destShape[1] != expectedChannels)
    return op->emitOpError()
           << "the destination channel extent must be " << expectedChannels
           << ", but the destination is " << destination;

  // The pads arrive in ONNX order and the helper wants them per axis, begin
  // then end, so height takes entries 0 and 2 and width takes 1 and 3. Getting
  // this reordering wrong is the single easiest mistake in the whole file and
  // it is why it is written once here instead of at each call site.
  const SmallVector<int64_t> perAxisPads = {pads[0], pads[2], pads[1], pads[3]};
  const SmallVector<int64_t> inputSpatial = {inputShape[2], inputShape[3]};

  npu::WindowedShapeResult windowed = npu::computeWindowedShape(
      inputSpatial, kernel, strides, dilations, perAxisPads, ceilMode);
  if (!windowed.ok())
    return op->emitOpError()
           << "the window is not representable on the "
           << (windowed.failedAxis == 0 ? "height" : "width") << " axis: "
           << npu::describeWindowError(windowed.error, windowed.failedParams);

  for (unsigned axis = 0; axis < 2; ++axis) {
    if (destShape[2 + axis] == windowed.extents[axis])
      continue;
    return op->emitOpError()
           << "the destination " << (axis == 0 ? "height" : "width")
           << " extent must be " << windowed.extents[axis]
           << ", the extent this window implies, but the destination is "
           << destination;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// The asynchronous DMA rules of Section 8.
//===----------------------------------------------------------------------===//

/// Rules 1 and 2: the token has exactly one use, that use is an `await`, and
/// the `await` is in the same block as its producer and comes after it.
///
/// Returns the await on success so the caller can run the overlap scan between
/// the two without finding it again.
FailureOr<AwaitOp> verifyTokenUse(Operation *op, Value token) {
  if (token.use_empty()) {
    op->emitOpError()
        << "the token must have exactly one use and that use must be an "
           "npuisa.await, but it has no uses. An asynchronous transfer that is "
           "never awaited is a transfer nothing waits for";
    return failure();
  }

  if (!token.hasOneUse()) {
    const auto uses = std::distance(token.use_begin(), token.use_end());
    op->emitOpError()
        << "the token must have exactly one use and that use must be an "
           "npuisa.await, but it has "
        << uses << " uses";
    return failure();
  }

  Operation *user = *token.user_begin();
  auto await = dyn_cast<AwaitOp>(user);
  if (!await) {
    op->emitOpError()
        << "the token's single use must be an npuisa.await, but it is used by "
        << user->getName();
    return failure();
  }

  if (await->getBlock() != op->getBlock()) {
    op->emitOpError() << "the npuisa.await must be in the same block as the "
                         "asynchronous operation it waits for, but it is in a "
                         "different block";
    return failure();
  }

  // Same block, so "after" is a position comparison the block itself can make.
  if (!op->isBeforeInBlock(await)) {
    op->emitOpError() << "the npuisa.await must come after the asynchronous "
                         "operation it waits for";
    return failure();
  }

  return await;
}

/// Rule 4: no operation between the async operation and its `await` may access
/// memory overlapping the destination.
///
/// The decision procedure is Section 8's, exactly: MemoryEffectOpInterface
/// effects to find which operations touch memory and which values they touch,
/// then the explicit byte range overlap test on those values. SSA identity is
/// never the question asked.
///
/// An operation between the two that does not implement MemoryEffectOpInterface
/// is treated as a possible conflict rather than as harmless, because an
/// operation that cannot say what it touches has not said it touches nothing.
LogicalResult verifyNoInterveningOverlap(Operation *op, Value destination,
                                         AwaitOp await) {
  // The destination's own range has to be computable before anything else can
  // be compared against it. Section 8 is explicit: if the offsets are not
  // static, refuse the async form with a diagnostic rather than assuming
  // disjointness.
  if (!computeBufferRange(destination))
    return op->emitOpError()
           << "the asynchronous form requires a destination whose byte range "
              "is statically known, but "
           << describeWhyNotAnalysable(destination)
           << ". Use the synchronous npuisa."
           << (isa<DmaLoadAsyncOp>(op) ? "dma_load" : "dma_store")
           << " instead, which needs no such proof";

  for (Operation *between = op->getNextNode(); between && between != await;
       between = between->getNextNode()) {
    auto effects = dyn_cast<MemoryEffectOpInterface>(between);
    if (!effects)
      return op->emitOpError()
             << "the operation " << between->getName()
             << " lies between this asynchronous transfer and its npuisa.await "
                "and does not implement MemoryEffectOpInterface, so it cannot "
                "be shown not to touch the destination buffer";

    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);

    for (const MemoryEffects::EffectInstance &instance : instances) {
      Value touched = instance.getValue();
      if (!touched) {
        // An effect with no value attached is an effect on something the
        // operation did not name. It cannot be shown disjoint from anything,
        // so it is a conflict.
        return op->emitOpError()
               << "the operation " << between->getName()
               << " lies between this asynchronous transfer and its "
                  "npuisa.await and declares a memory effect on an unnamed "
                  "value, so it cannot be shown not to touch the destination "
                  "buffer";
      }
      if (!isa<MemRefType>(touched.getType()))
        continue;

      switch (overlaps(touched, destination)) {
      case OverlapResult::Disjoint:
        continue;
      case OverlapResult::Overlaps:
        return op->emitOpError()
               << "the operation " << between->getName()
               << " lies between this asynchronous transfer and its "
                  "npuisa.await and accesses memory overlapping the "
                  "destination buffer, which is the race the token exists to "
                  "prevent";
      case OverlapResult::Unknown:
        return op->emitOpError()
               << "the operation " << between->getName()
               << " lies between this asynchronous transfer and its "
                  "npuisa.await and accesses a buffer that cannot be shown "
                  "disjoint from the destination, because "
               << describeWhyNotAnalysable(touched);
      }
    }
  }
  return success();
}

/// The three shape and direction rules the two asynchronous forms share with
/// the two synchronous ones.
LogicalResult verifyTransferShapes(Operation *op, Value source, Value dest) {
  MemRefType sourceType = memRefOf(source);
  MemRefType destType = memRefOf(dest);

  if (failed(verifySameElementType(op, "the source", sourceType,
                                   "the destination", destType)))
    return failure();
  return verifySameShape(op, "the source", sourceType, "the destination",
                         destType);
}

/// The shared body of the two asynchronous verifiers.
LogicalResult verifyAsyncTransfer(Operation *op, Value source, Value dest,
                                  Value token) {
  if (failed(verifyTransferShapes(op, source, dest)))
    return failure();

  FailureOr<AwaitOp> await = verifyTokenUse(op, token);
  if (failed(await))
    return failure();

  return verifyNoInterveningOverlap(op, dest, *await);
}

/// The shared body of the two asynchronous canonicalizations.
///
/// An asynchronous operation whose `await` is the very next operation
/// canonicalizes back to the synchronous form, so a scheduling pass that looked
/// for something to overlap the transfer with and found nothing leaves no
/// residue behind.
template <typename SyncOpTy>
LogicalResult canonicalizeImmediateAwait(Operation *op, Value source,
                                         Value dest, Value token,
                                         PatternRewriter &rewriter) {
  if (!token.hasOneUse())
    return failure();
  auto await = dyn_cast<AwaitOp>(*token.user_begin());
  if (!await)
    return failure();

  // "The very next operation" is exactly that, and nothing weaker. An await two
  // operations later with a harmless one in between is a transfer that really
  // does overlap some work, and folding it would undo a scheduling decision
  // somebody made on purpose.
  if (op->getNextNode() != await.getOperation())
    return failure();

  rewriter.setInsertionPoint(op);
  SyncOpTy::create(rewriter, op->getLoc(), source, dest);
  rewriter.eraseOp(await);
  rewriter.eraseOp(op);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// npuisa.const
//===----------------------------------------------------------------------===//

void ConstOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "cst");
}

LogicalResult ConstOp::verify() {
  auto valueType = dyn_cast<ShapedType>(getValue().getType());
  if (!valueType)
    return emitOpError() << "the value attribute must be a shaped attribute, "
                            "but its type is "
                         << getValue().getType();

  MemRefType resultType = memRefOf(getResult());

  if (valueType.getElementType() != resultType.getElementType())
    return emitOpError()
           << "the value attribute's element type must equal the result's, but "
              "the attribute has element type "
           << valueType.getElementType() << " and the result has "
           << resultType.getElementType();

  if (valueType.getShape() != resultType.getShape())
    return emitOpError()
           << "the value attribute's shape must equal the result's, but the "
              "attribute is "
           << valueType << " and the result is " << resultType;

  return success();
}

//===----------------------------------------------------------------------===//
// The synchronous DMA operations.
//===----------------------------------------------------------------------===//

LogicalResult DmaLoadOp::verify() {
  return verifyTransferShapes(*this, getSource(), getDest());
}

LogicalResult DmaStoreOp::verify() {
  return verifyTransferShapes(*this, getSource(), getDest());
}

//===----------------------------------------------------------------------===//
// The asynchronous DMA operations and the await.
//===----------------------------------------------------------------------===//

LogicalResult DmaLoadAsyncOp::verify() {
  return verifyAsyncTransfer(*this, getSource(), getDest(), getToken());
}

LogicalResult DmaLoadAsyncOp::canonicalize(DmaLoadAsyncOp op,
                                           PatternRewriter &rewriter) {
  return canonicalizeImmediateAwait<DmaLoadOp>(op, op.getSource(), op.getDest(),
                                               op.getToken(), rewriter);
}

LogicalResult DmaStoreAsyncOp::verify() {
  return verifyAsyncTransfer(*this, getSource(), getDest(), getToken());
}

LogicalResult DmaStoreAsyncOp::canonicalize(DmaStoreAsyncOp op,
                                            PatternRewriter &rewriter) {
  return canonicalizeImmediateAwait<DmaStoreOp>(
      op, op.getSource(), op.getDest(), op.getToken(), rewriter);
}

LogicalResult AwaitOp::verify() {
  // The whole rule lives on the asynchronous side, so this verifier holds only
  // what that one cannot see: that the token came from an asynchronous
  // transfer at all. Today the type system makes that unavoidable, since the
  // two asynchronous operations are the only producers of a token; the check
  // is here so that adding a third producer is a decision somebody makes rather
  // than a thing that quietly starts verifying.
  Operation *producer = getToken().getDefiningOp();
  if (!producer)
    return emitOpError() << "the token must come from an npuisa.dma_load_async "
                            "or npuisa.dma_store_async, but it is a block "
                            "argument";
  if (!isa<DmaLoadAsyncOp, DmaStoreAsyncOp>(producer))
    return emitOpError() << "the token must come from an npuisa.dma_load_async "
                            "or npuisa.dma_store_async, but it comes from "
                         << producer->getName();
  return success();
}

//===----------------------------------------------------------------------===//
// npuisa.matmul
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType lhs = memRefOf(getLhs());
  MemRefType rhs = memRefOf(getRhs());
  MemRefType dest = memRefOf(getDestination());

  if (lhs.getRank() != 2 || rhs.getRank() != 2 || dest.getRank() != 2)
    return emitOpError()
           << "matmul is rank 2 by rank 2 into rank 2, but the operands are "
           << lhs << ", " << rhs << " and " << dest;

  const int64_t m = lhs.getShape()[0];
  const int64_t k = lhs.getShape()[1];
  const int64_t n = rhs.getShape()[1];

  if (rhs.getShape()[0] != k)
    return emitOpError()
           << "the contraction extents must agree, but the left operand is "
           << lhs << " with K = " << k << " and the right operand is " << rhs
           << " with K = " << rhs.getShape()[0];

  if (dest.getShape()[0] != m || dest.getShape()[1] != n)
    return emitOpError() << "the destination must be " << m << " by " << n
                         << ", the shape this contraction implies, but it is "
                         << dest;

  if (failed(verifySameElementType(*this, "the left operand", lhs,
                                   "the right operand", rhs)))
    return failure();
  if (failed(verifySameElementType(*this, "the left operand", lhs,
                                   "the destination", dest)))
    return failure();

  return verifyBias(*this, getBias(), n, "the destination column count");
}

//===----------------------------------------------------------------------===//
// npuisa.conv2d
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType input = memRefOf(getInput());
  MemRefType filter = memRefOf(getFilter());
  MemRefType dest = memRefOf(getDestination());

  if (filter.getRank() != 4)
    return emitOpError() << "the filter must be rank 4, outputChannels by "
                            "inputChannels over group by kernelHeight by "
                            "kernelWidth, but it is "
                         << filter;
  if (input.getRank() != 4)
    return emitOpError()
           << "the input must be rank 4 in NCHW order, but it is " << input;

  const int64_t group = getGroup();
  if (group <= 0)
    return emitOpError() << "group must be positive, but it is " << group;

  const int64_t inputChannels = input.getShape()[1];
  const int64_t outputChannels = filter.getShape()[0];

  if (inputChannels % group != 0)
    return emitOpError() << "group must divide the input channel count, but "
                            "the input has "
                         << inputChannels << " channels and group is " << group;
  if (outputChannels % group != 0)
    return emitOpError() << "group must divide the output channel count, but "
                            "the filter has "
                         << outputChannels << " output channels and group is "
                         << group;
  if (filter.getShape()[1] != inputChannels / group)
    return emitOpError()
           << "the filter's second extent must be the input channel count "
              "divided by group, which is "
           << inputChannels / group << ", but the filter is " << filter;

  const SmallVector<int64_t> kernel = {filter.getShape()[2],
                                       filter.getShape()[3]};
  if (failed(verifyWindowedDestination(
          *this, input, dest, kernel, getStrides(), getDilations(), getPads(),
          /*ceilMode=*/false, outputChannels)))
    return failure();

  if (failed(verifySameElementType(*this, "the input", input, "the filter",
                                   filter)))
    return failure();
  if (failed(verifySameElementType(*this, "the input", input,
                                   "the destination", dest)))
    return failure();

  return verifyBias(*this, getBias(), outputChannels,
                    "the output channel count");
}

//===----------------------------------------------------------------------===//
// The elementwise instructions.
//===----------------------------------------------------------------------===//

namespace {

/// The shared body of the two binary elementwise verifiers.
LogicalResult verifyElementwiseBinary(Operation *op, Value lhsValue,
                                      Value rhsValue, Value destValue) {
  if (failed(verifyDpsPartition(op)))
    return failure();

  MemRefType lhs = memRefOf(lhsValue);
  MemRefType rhs = memRefOf(rhsValue);
  MemRefType dest = memRefOf(destValue);

  if (failed(verifySameShape(op, "the left operand", lhs, "the destination",
                             dest)))
    return failure();
  if (failed(verifySameShape(op, "the right operand", rhs, "the destination",
                             dest)))
    return failure();
  if (failed(verifySameElementType(op, "the left operand", lhs,
                                   "the destination", dest)))
    return failure();
  return verifySameElementType(op, "the right operand", rhs, "the destination",
                               dest);
}

} // namespace

LogicalResult AddOp::verify() {
  return verifyElementwiseBinary(*this, getLhs(), getRhs(), getDestination());
}

LogicalResult MulOp::verify() {
  return verifyElementwiseBinary(*this, getLhs(), getRhs(), getDestination());
}

LogicalResult ReluOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType input = memRefOf(getInput());
  MemRefType dest = memRefOf(getDestination());

  if (failed(verifySameShape(*this, "the input", input, "the destination",
                             dest)))
    return failure();
  return verifySameElementType(*this, "the input", input, "the destination",
                               dest);
}

//===----------------------------------------------------------------------===//
// The pooling instructions.
//===----------------------------------------------------------------------===//

namespace {

/// The shared body of the two pooling verifiers. Pooling does not change the
/// channel count, so the expected channel extent is the input's.
LogicalResult verifyPool(Operation *op, Value inputValue, Value destValue,
                         ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides,
                         ArrayRef<int64_t> dilations, ArrayRef<int64_t> pads,
                         int64_t ceilMode) {
  if (failed(verifyDpsPartition(op)))
    return failure();

  MemRefType input = memRefOf(inputValue);
  MemRefType dest = memRefOf(destValue);

  if (ceilMode != 0 && ceilMode != 1)
    return op->emitOpError()
           << "ceil_mode must be 0 or 1, but it is " << ceilMode;

  if (input.getRank() != 4)
    return op->emitOpError()
           << "the input must be rank 4 in NCHW order, but it is " << input;

  if (failed(verifyWindowedDestination(op, input, dest, kernel, strides,
                                       dilations, pads, ceilMode == 1,
                                       input.getShape()[1])))
    return failure();

  return verifySameElementType(op, "the input", input, "the destination", dest);
}

} // namespace

LogicalResult PoolMaxOp::verify() {
  return verifyPool(*this, getInput(), getDestination(), getKernel(),
                    getStrides(), getDilations(), getPads(), getCeilMode());
}

LogicalResult PoolAvgOp::verify() {
  return verifyPool(*this, getInput(), getDestination(), getKernel(),
                    getStrides(), getDilations(), getPads(), getCeilMode());
}

//===----------------------------------------------------------------------===//
// The shape instructions.
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType input = memRefOf(getInput());
  MemRefType dest = memRefOf(getDestination());

  if (failed(verifySameElementType(*this, "the input", input,
                                   "the destination", dest)))
    return failure();

  const int64_t inputElements = elementCount(input);
  const int64_t destElements = elementCount(dest);
  if (inputElements != destElements)
    return emitOpError()
           << "a reshape must preserve the element count, but the input "
           << input << " holds " << inputElements << " elements and the "
           << "destination " << dest << " holds " << destElements;
  return success();
}

LogicalResult TransposeOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType input = memRefOf(getInput());
  MemRefType dest = memRefOf(getDestination());
  const ArrayRef<int64_t> permutation = getPermutation();

  if (failed(verifySameElementType(*this, "the input", input,
                                   "the destination", dest)))
    return failure();

  const int64_t rank = dest.getRank();
  if (input.getRank() != rank)
    return emitOpError()
           << "the input and the destination must have the same rank, but the "
              "input is "
           << input << " and the destination is " << dest;

  if (static_cast<int64_t>(permutation.size()) != rank)
    return emitOpError() << "permutation must have exactly " << rank
                         << " entries, the destination rank, but it has "
                         << permutation.size();

  SmallVector<bool> seen(rank, false);
  for (int64_t entry : permutation) {
    if (entry < 0 || entry >= rank)
      return emitOpError() << "permutation entries must be in the range 0 to "
                           << rank - 1 << ", but one of them is " << entry;
    if (seen[entry])
      return emitOpError()
             << "permutation must be a permutation, with every index appearing "
                "exactly once, but "
             << entry << " appears more than once";
    seen[entry] = true;
  }

  for (int64_t axis = 0; axis < rank; ++axis) {
    const int64_t expected = input.getShape()[permutation[axis]];
    if (dest.getShape()[axis] == expected)
      continue;
    return emitOpError() << "destination extent " << axis << " must be "
                         << expected << ", the input extent at permutation["
                         << axis << "] = " << permutation[axis]
                         << ", but the destination is " << dest;
  }
  return success();
}

LogicalResult ConcatOp::verify() {
  if (failed(verifyDpsPartition(*this)))
    return failure();

  MemRefType dest = memRefOf(getDestination());
  const int64_t rank = dest.getRank();
  const int64_t axis = getAxis();

  if (getInputs().empty())
    return emitOpError()
           << "a concatenation needs at least one input, but it has none";

  if (axis < 0 || axis >= rank)
    return emitOpError() << "axis must be in the range 0 to " << rank - 1
                         << ", the destination rank being " << rank
                         << ", but it is " << axis
                         << ". A negative axis is an ONNX convention the "
                            "frontend normalises and it does not reach here";

  int64_t axisSum = 0;
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    MemRefType type = memRefOf(input);
    if (type.getRank() != rank)
      return emitOpError() << "every input must have the destination's rank, "
                              "which is "
                           << rank << ", but input " << index << " is " << type;
    if (failed(verifySameElementType(*this, "the destination", dest,
                                     "an input", type)))
      return failure();

    for (int64_t other = 0; other < rank; ++other) {
      if (other == axis)
        continue;
      if (type.getShape()[other] == dest.getShape()[other])
        continue;
      return emitOpError()
             << "the inputs and the destination must agree on every axis "
                "except axis "
             << axis << ", but input " << index << " is " << type
             << " and the destination is " << dest;
    }
    axisSum += type.getShape()[axis];
  }

  if (axisSum != dest.getShape()[axis])
    return emitOpError() << "the input extents along axis " << axis
                         << " must sum to the destination extent, which is "
                         << dest.getShape()[axis] << ", but they sum to "
                         << axisSum;
  return success();
}
