//===- NPUOps.cpp - NPU dialect operation implementations -----------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/Dialect/NPU/IR/NPUEnums.cpp.inc"

#define GET_OP_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static RankedTensorType rankedType(Value v) {
  return llvm::cast<RankedTensorType>(v.getType());
}

// Reject a batch size the backend cannot compute correctly.
//
// The simulator's convolution kernel hardcodes batch index 0, and its pooling
// kernel iterates channels without ever seeing the batch dimension, so for
// N greater than 1 both write correct data for the first image and leave the
// rest of the output holding whatever was in the scratchpad. Nothing used to
// stop that: the importer, the verifiers, the lowering, and the allocator all
// accepted it, and the wrong answer came out of the simulator with no
// diagnostic. docs/ASSESSMENT.md section 2.1 has the reproduction.
//
// Refusing is the correct behaviour until phase U6 adds the batch loops. This
// only applies to activations. A convolution weight is OIHW, so its leading
// dimension is the output channel count and has nothing to do with batch.
static LogicalResult verifyUnbatchedActivation(Operation *op, Value activation) {
  auto type = llvm::cast<RankedTensorType>(activation.getType());
  if (type.getRank() != 4 || type.isDynamicDim(0))
    return success();
  int64_t batch = type.getDimSize(0);
  if (batch == 1)
    return success();
  return op->emitOpError()
         << "batch size " << batch
         << " is not supported yet. The simulator processes only batch index 0 "
            "for this op, so N greater than 1 returns silently wrong results "
            "for every image after the first. This is a tracked limitation, "
            "not a permanent design choice: upgrade phase U6 adds real batch "
            "support";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto attrType = llvm::cast<ShapedType>(getValue().getType());
  auto resultType = llvm::cast<ShapedType>(getResult().getType());
  if (attrType != resultType)
    return emitOpError("value attribute type ")
           << attrType << " does not match result type " << resultType;
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }

//===----------------------------------------------------------------------===//
// Elementwise folders
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr, APFloat, void>(
      ArrayRef<Attribute>{adaptor.getLhs(), adaptor.getRhs()},
      [](const APFloat &a, const APFloat &b) { return a + b; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return constFoldBinaryOp<FloatAttr, APFloat, void>(
      ArrayRef<Attribute>{adaptor.getLhs(), adaptor.getRhs()},
      [](const APFloat &a, const APFloat &b) { return a * b; });
}

OpFoldResult ReluOp::fold(FoldAdaptor adaptor) {
  return constFoldUnaryOp<FloatAttr, APFloat, void>(
      ArrayRef<Attribute>{adaptor.getInput()}, [](const APFloat &a) {
        APFloat zero = APFloat::getZero(a.getSemantics());
        return a.compare(zero) == APFloat::cmpLessThan ? zero : a;
      });
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
// relu(relu(x)) folds to relu(x): the activation is idempotent.
struct FoldReluOfRelu : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto parent = op.getInput().getDefiningOp<ReluOp>();
    if (!parent)
      return failure();
    rewriter.replaceOp(op, parent.getResult());
    return success();
  }
};

// A reshape whose result type equals its input type is the identity.
struct FoldReshapeIdentity : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInput().getType() != op.getOutput().getType())
      return failure();
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
};

// reshape(reshape(x)) collapses to a single reshape to the outer result type.
struct FoldReshapeOfReshape : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto parent = op.getInput().getDefiningOp<ReshapeOp>();
    if (!parent)
      return failure();
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getOutput().getType(),
                                           parent.getInput());
    return success();
  }
};
} // namespace

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<FoldReluOfRelu>(context);
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<FoldReshapeIdentity, FoldReshapeOfReshape>(context);
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  if (rankedType(getInput()).getRank() != 4)
    return emitOpError("expects a rank 4 NCHW input");
  if (rankedType(getWeight()).getRank() != 4)
    return emitOpError("expects a rank 4 OIHW weight");
  if (rankedType(getOutput()).getRank() != 4)
    return emitOpError("expects a rank 4 NCHW output");
  if (getStrides().size() != 2)
    return emitOpError("expects a 2 element strides attribute");
  if (getDilations().size() != 2)
    return emitOpError("expects a 2 element dilations attribute");
  if (getPads().size() != 4)
    return emitOpError("expects a 4 element pads attribute");
  if (getBias() && rankedType(getBias()).getRank() != 1)
    return emitOpError("bias must be a rank 1 per channel vector");
  return verifyUnbatchedActivation(*this, getInput());
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhs = rankedType(getLhs());
  auto rhs = rankedType(getRhs());
  auto out = rankedType(getOutput());
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || out.getRank() != 2)
    return emitOpError("expects rank 2 operands and result");
  int64_t k1 = lhs.getDimSize(1), k2 = rhs.getDimSize(0);
  if (!ShapedType::isDynamic(k1) && !ShapedType::isDynamic(k2) && k1 != k2)
    return emitOpError("contraction dimensions disagree: ")
           << k1 << " vs " << k2;
  if (getBias() && rankedType(getBias()).getRank() != 1)
    return emitOpError("bias must be a rank 1 vector");
  return success();
}

//===----------------------------------------------------------------------===//
// Pooling
//===----------------------------------------------------------------------===//

template <typename PoolOp>
static LogicalResult verifyPool(PoolOp op) {
  if (rankedType(op.getInput()).getRank() != 4 ||
      rankedType(op.getOutput()).getRank() != 4)
    return op.emitOpError("expects a rank 4 NCHW input and output");
  if (op.getKernelShape().size() != 2)
    return op.emitOpError("expects a 2 element kernel_shape attribute");
  if (op.getStrides().size() != 2)
    return op.emitOpError("expects a 2 element strides attribute");
  if (op.getPads().size() != 4)
    return op.emitOpError("expects a 4 element pads attribute");
  return verifyUnbatchedActivation(op, op.getInput());
}

LogicalResult MaxPool2DOp::verify() { return verifyPool(*this); }
LogicalResult AvgPool2DOp::verify() { return verifyPool(*this); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto in = rankedType(getInput());
  auto out = rankedType(getOutput());
  if (in.hasStaticShape() && out.hasStaticShape() &&
      in.getNumElements() != out.getNumElements())
    return emitOpError("element count changes across reshape: ")
           << in.getNumElements() << " vs " << out.getNumElements();
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  auto in = rankedType(getInput());
  auto out = rankedType(getOutput());
  int64_t rank = in.getRank();
  ArrayAttr perm = getPermutation();
  if (static_cast<int64_t>(perm.size()) != rank)
    return emitOpError("permutation size ")
           << perm.size() << " does not match input rank " << rank;

  llvm::SmallVector<bool> seen(rank, false);
  llvm::SmallVector<int64_t> permVals;
  for (Attribute a : perm) {
    int64_t v = llvm::cast<IntegerAttr>(a).getInt();
    if (v < 0 || v >= rank)
      return emitOpError("permutation index out of range: ") << v;
    if (seen[v])
      return emitOpError("permutation index repeated: ") << v;
    seen[v] = true;
    permVals.push_back(v);
  }

  if (in.hasStaticShape() && out.hasStaticShape())
    for (int64_t i = 0; i < rank; ++i)
      if (in.getDimSize(permVals[i]) != out.getDimSize(i))
        return emitOpError("result shape is not the permuted input shape");
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  if (getInputs().empty())
    return emitOpError("expects at least one input");
  auto out = rankedType(getOutput());
  int64_t rank = out.getRank();
  int64_t axis = getAxis();
  if (axis < 0 || axis >= rank)
    return emitOpError("axis ") << axis << " out of range for rank " << rank;

  int64_t sum = 0;
  bool sumStatic = !out.isDynamicDim(axis);
  for (Value v : getInputs()) {
    auto t = rankedType(v);
    if (t.getRank() != rank)
      return emitOpError("all inputs must match the result rank ") << rank;
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis)
        continue;
      int64_t a = t.getDimSize(d), b = out.getDimSize(d);
      if (!ShapedType::isDynamic(a) && !ShapedType::isDynamic(b) && a != b)
        return emitOpError("non concatenated dimension ")
               << d << " disagrees with the result";
    }
    if (t.isDynamicDim(axis))
      sumStatic = false;
    else
      sum += t.getDimSize(axis);
  }
  if (sumStatic && sum != out.getDimSize(axis))
    return emitOpError("concatenated size ")
           << sum << " does not match result dimension " << out.getDimSize(axis);
  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormOp::verify() {
  auto in = rankedType(getInput());
  if (in.getRank() != 4)
    return emitOpError("expects a rank 4 NCHW input");
  if (rankedType(getOutput()) != in)
    return emitOpError("result type must equal input type");

  int64_t channels = in.getDimSize(1);
  auto checkParam = [&](Value v, StringRef name) -> LogicalResult {
    auto t = rankedType(v);
    if (t.getRank() != 1)
      return emitOpError(name) << " must be a rank 1 per channel vector";
    if (!ShapedType::isDynamic(channels) && !t.isDynamicDim(0) &&
        t.getDimSize(0) != channels)
      return emitOpError(name) << " length " << t.getDimSize(0)
                               << " does not match channel count " << channels;
    return success();
  };
  if (failed(checkParam(getScale(), "scale")) ||
      failed(checkParam(getOffset(), "offset")) ||
      failed(checkParam(getMean(), "mean")) ||
      failed(checkParam(getVariance(), "variance")))
    return failure();
  return verifyUnbatchedActivation(*this, getInput());
}
