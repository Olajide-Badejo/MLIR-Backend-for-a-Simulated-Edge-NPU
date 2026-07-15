//===- FoldBatchNorm.cpp - Fold batch norm into convolution ---------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

namespace mlir::npu {

#define GEN_PASS_DEF_NPUFOLDBATCHNORM
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

namespace {

// Reads a rank 1 or higher fp32 constant tensor into a flat float vector.
static llvm::SmallVector<float> flatten(DenseFPElementsAttr attr) {
  llvm::SmallVector<float> out;
  out.reserve(attr.getNumElements());
  for (const llvm::APFloat &f : attr.getValues<llvm::APFloat>())
    out.push_back(f.convertToFloat());
  return out;
}

// Rewrites bn(conv(x, W, b)) with no activation between the two into a single
// convolution whose weights and bias absorb the normalization. Requires the
// convolution weight and all four batch norm parameters to be constants, which
// they are for an imported inference graph.
struct FoldBatchNormIntoConv : public OpRewritePattern<BatchNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BatchNormOp bn,
                                PatternRewriter &rewriter) const override {
    auto conv = bn.getInput().getDefiningOp<Conv2DOp>();
    if (!conv)
      return rewriter.notifyMatchFailure(bn, "input is not a conv2d");
    if (conv.getActivation() != Activation::none)
      return rewriter.notifyMatchFailure(bn, "conv has a fused activation");
    if (!conv->hasOneUse())
      return rewriter.notifyMatchFailure(bn, "conv result has other uses");

    DenseFPElementsAttr weightAttr, scaleAttr, offsetAttr, meanAttr, varAttr;
    if (!matchPattern(conv.getWeight(), m_Constant(&weightAttr)) ||
        !matchPattern(bn.getScale(), m_Constant(&scaleAttr)) ||
        !matchPattern(bn.getOffset(), m_Constant(&offsetAttr)) ||
        !matchPattern(bn.getMean(), m_Constant(&meanAttr)) ||
        !matchPattern(bn.getVariance(), m_Constant(&varAttr)))
      return rewriter.notifyMatchFailure(bn, "weight or parameters not constant");

    bool hasBias = static_cast<bool>(conv.getBias());
    DenseFPElementsAttr biasAttr;
    if (hasBias && !matchPattern(conv.getBias(), m_Constant(&biasAttr)))
      return rewriter.notifyMatchFailure(bn, "conv bias not constant");

    auto weightType = llvm::cast<RankedTensorType>(weightAttr.getType());
    int64_t oc = weightType.getDimSize(0);
    if (oc == 0)
      return rewriter.notifyMatchFailure(bn, "empty weight");
    int64_t inner = weightType.getNumElements() / oc;

    llvm::SmallVector<float> scale = flatten(scaleAttr);
    llvm::SmallVector<float> offset = flatten(offsetAttr);
    llvm::SmallVector<float> mean = flatten(meanAttr);
    llvm::SmallVector<float> var = flatten(varAttr);
    llvm::SmallVector<float> bias(oc, 0.0f);
    if (hasBias)
      bias = flatten(biasAttr);

    if (static_cast<int64_t>(scale.size()) != oc ||
        static_cast<int64_t>(offset.size()) != oc ||
        static_cast<int64_t>(mean.size()) != oc ||
        static_cast<int64_t>(var.size()) != oc ||
        static_cast<int64_t>(bias.size()) != oc)
      return rewriter.notifyMatchFailure(bn, "parameter length mismatch");

    float eps = bn.getEpsilon().convertToFloat();

    llvm::SmallVector<float> coef(oc);
    for (int64_t o = 0; o < oc; ++o)
      coef[o] = scale[o] / std::sqrt(var[o] + eps);

    llvm::SmallVector<float> newWeight = flatten(weightAttr);
    for (int64_t idx = 0; idx < static_cast<int64_t>(newWeight.size()); ++idx)
      newWeight[idx] *= coef[idx / inner];

    llvm::SmallVector<float> newBias(oc);
    for (int64_t o = 0; o < oc; ++o)
      newBias[o] = coef[o] * (bias[o] - mean[o]) + offset[o];

    auto biasType = RankedTensorType::get({oc}, rewriter.getF32Type());
    auto newWeightAttr =
        DenseElementsAttr::get(weightType, llvm::ArrayRef<float>(newWeight));
    auto newBiasAttr =
        DenseElementsAttr::get(biasType, llvm::ArrayRef<float>(newBias));

    Location loc = bn.getLoc();
    auto newWeightConst = ConstantOp::create(rewriter, loc, weightType, newWeightAttr);
    auto newBiasConst = ConstantOp::create(rewriter, loc, biasType, newBiasAttr);

    rewriter.replaceOpWithNewOp<Conv2DOp>(
        bn, bn.getOutput().getType(), conv.getInput(),
        newWeightConst.getResult(), newBiasConst.getResult(),
        conv.getStridesAttr(), conv.getPadsAttr(), conv.getDilationsAttr(),
        conv.getGroupAttr(), conv.getActivationAttr());
    return success();
  }
};

struct NPUFoldBatchNormPass
    : public impl::NPUFoldBatchNormBase<NPUFoldBatchNormPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldBatchNormIntoConv>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::npu
