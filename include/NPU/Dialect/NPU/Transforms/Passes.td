//===- Passes.td - NPU dialect transform passes ------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
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

def NPUFuseOps : Pass<"npu-fuse-ops", "::mlir::func::FuncOp"> {
  let summary = "Fuse a trailing activation into a convolution or matmul.";
  let description = [{
    A `relu` whose sole producer is a `conv2d` or `matmul` with no fused
    activation is absorbed into that op by setting its `activation` attribute,
    which already carries any bias as an operand. The result is a single fused
    `conv2d`/`matmul` that computes the convolution, the bias add, and the
    activation together.

    This fusion matters because of the memory model: the unfused form writes the
    convolution result to DRAM and reads it back for the activation, while the
    fused op keeps that intermediate in the scratchpad. The benchmark suite
    measures the resulting drop in DRAM traffic.
  }];
  let dependentDialects = ["::mlir::npu::NPUDialect"];
}

#endif // NPU_TRANSFORMS_PASSES_TD
