//===- Passes.td - NPU dialect transform passes ------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_TRANSFORMS_PASSES_TD
#define NPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def NPUFoldBatchNorm : Pass<"npu-fold-batchnorm", "::mlir::func::FuncOp"> {
  let summary = "Fold batch normalization into a preceding convolution.";
  let description = [{
    At inference a batch normalization's scale, offset, mean, and variance are
    constants, so `bn(conv(x, W, b))` with no activation between the two is exactly
    another convolution with rescaled weights and a rescaled bias:

      coef[o]  = scale[o] / sqrt(variance[o] + epsilon)
      W'[o]    = W[o] * coef[o]
      b'[o]    = coef[o] * (b[o] - mean[o]) + offset[o]

    This pass does that arithmetic on the constant tensors at compile time and
    replaces the pair with a single convolution, removing the batch normalization
    and its parameter constants from the graph.
  }];
  let dependentDialects = ["::mlir::npu::NPUDialect"];
}

#endif // NPU_TRANSFORMS_PASSES_TD
