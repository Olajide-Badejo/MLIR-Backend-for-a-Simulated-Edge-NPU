//===- NPUOps.td - NPU dialect operations ------------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "NPU/Dialect/NPU/IR/NPUDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Common type constraints
//===----------------------------------------------------------------------===//

// Every value in the npu dialect is a ranked fp32 tensor. Keeping the element
// type fixed to fp32 matches the core scope (fp32 CNN inference).
def NPU_FloatTensor : RankedTensorOf<[F32]>;

//===----------------------------------------------------------------------===//
// npu.constant
//===----------------------------------------------------------------------===//

def NPU_ConstantOp : NPU_Op<"constant", [Pure, ConstantLike]> {
  let summary = "A constant fp32 tensor.";
  let description = [{
    Materializes a constant tensor from an elements attribute. After weights are
    loaded, convolution kernels, biases, and batch normalization parameters all
    appear as `npu.constant`, which is what makes compile time folding possible.

    Example:

    ```mlir
    %0 = npu.constant : tensor<2xf32> {value = dense<[1.0, 2.0]> : tensor<2xf32>}
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.relu
//===----------------------------------------------------------------------===//

def NPU_ReluOp : NPU_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified linear unit, elementwise max(x, 0).";
  let arguments = (ins NPU_FloatTensor:$input);
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

//===----------------------------------------------------------------------===//
// npu.add
//===----------------------------------------------------------------------===//

def NPU_AddOp : NPU_Op<"add", [Pure, SameOperandsAndResultType, Commutative]> {
  let summary = "Elementwise addition of two tensors of identical shape.";
  let arguments = (ins NPU_FloatTensor:$lhs, NPU_FloatTensor:$rhs);
  let results = (outs NPU_FloatTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// npu.mul
//===----------------------------------------------------------------------===//

def NPU_MulOp : NPU_Op<"mul", [Pure, SameOperandsAndResultType, Commutative]> {
  let summary = "Elementwise multiplication of two tensors of identical shape.";
  let arguments = (ins NPU_FloatTensor:$lhs, NPU_FloatTensor:$rhs);
  let results = (outs NPU_FloatTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // NPU_OPS_TD
