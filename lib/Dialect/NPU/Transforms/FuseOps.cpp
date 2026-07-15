//===- FuseOps.cpp - Fuse activation into conv/matmul ---------------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::npu {

#define GEN_PASS_DEF_NPUFUSEOPS
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

namespace {

// relu(conv2d(x, W, b, activation=none)) becomes conv2d(x, W, b, activation=relu),
// keeping the intermediate in scratchpad rather than round tripping it to DRAM.
struct FuseReluIntoConv : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter &rewriter) const override {
    auto conv = relu.getInput().getDefiningOp<Conv2DOp>();
    if (!conv)
      return rewriter.notifyMatchFailure(relu, "producer is not a conv2d");
    if (conv.getActivation() != Activation::none)
      return rewriter.notifyMatchFailure(relu, "conv already has an activation");
    if (!conv->hasOneUse())
      return rewriter.notifyMatchFailure(relu, "conv result has other uses");

    auto reluAttr = ActivationAttr::get(getContext(), Activation::relu);
    rewriter.replaceOpWithNewOp<Conv2DOp>(
        relu, relu.getType(), conv.getInput(), conv.getWeight(), conv.getBias(),
        conv.getStridesAttr(), conv.getPadsAttr(), conv.getDilationsAttr(),
        conv.getGroupAttr(), reluAttr);
    return success();
  }
};

// relu(matmul(a, b, bias, activation=none)) becomes matmul(a, b, bias, relu).
struct FuseReluIntoMatMul : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter &rewriter) const override {
    auto matmul = relu.getInput().getDefiningOp<MatMulOp>();
    if (!matmul)
      return rewriter.notifyMatchFailure(relu, "producer is not a matmul");
    if (matmul.getActivation() != Activation::none)
      return rewriter.notifyMatchFailure(relu, "matmul already has activation");
    if (!matmul->hasOneUse())
      return rewriter.notifyMatchFailure(relu, "matmul result has other uses");

    auto reluAttr = ActivationAttr::get(getContext(), Activation::relu);
    rewriter.replaceOpWithNewOp<MatMulOp>(relu, relu.getType(),
                                          matmul.getLhs(), matmul.getRhs(),
                                          matmul.getBias(), reluAttr);
    return success();
  }
};

struct NPUFuseOpsPass : public impl::NPUFuseOpsBase<NPUFuseOpsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseReluIntoConv, FuseReluIntoMatMul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::npu
