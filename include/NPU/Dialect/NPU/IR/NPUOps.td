//===- NPUOps.td - NPU dialect operations ------------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "NPU/Dialect/NPU/IR/NPUDialect.td"
include "mlir/IR/EnumAttr.td"
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
// Fused activation attribute
//===----------------------------------------------------------------------===//

// Convolution and matmul carry a fused activation rather than being followed by
// a separate activation op. This gives the fusion patterns a single clean target
// and mirrors the fused operator style used by TFLite. The set is intentionally
// small for the fp32 CNN scope and can grow later.
def NPU_Activation : I32EnumAttr<"Activation",
    "Fused activation applied after a conv or matmul",
    [
      I32EnumAttrCase<"none", 0>,
      I32EnumAttrCase<"relu", 1>,
    ]> {
  let cppNamespace = "::mlir::npu";
}

//===----------------------------------------------------------------------===//
// npu.constant
//===----------------------------------------------------------------------===//

def NPU_ConstantOp : NPU_Op<"constant", [Pure, ConstantLike]> {
  let summary = "A constant fp32 tensor.";
  let description = [{
    Materializes a constant tensor from an elements attribute. After weights are
    loaded, convolution kernels, biases, and batch normalization parameters all
    appear as `npu.constant`, which is what makes compile time folding possible.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def NPU_ReluOp : NPU_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified linear unit, elementwise max(x, 0).";
  let arguments = (ins NPU_FloatTensor:$input);
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def NPU_AddOp : NPU_Op<"add", [Pure, SameOperandsAndResultType, Commutative]> {
  let summary = "Elementwise addition of two tensors of identical shape.";
  let arguments = (ins NPU_FloatTensor:$lhs, NPU_FloatTensor:$rhs);
  let results = (outs NPU_FloatTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def NPU_MulOp : NPU_Op<"mul", [Pure, SameOperandsAndResultType, Commutative]> {
  let summary = "Elementwise multiplication of two tensors of identical shape.";
  let arguments = (ins NPU_FloatTensor:$lhs, NPU_FloatTensor:$rhs);
  let results = (outs NPU_FloatTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// npu.conv2d
//===----------------------------------------------------------------------===//

def NPU_Conv2DOp : NPU_Op<"conv2d", [Pure]> {
  let summary = "2D convolution with an optional bias and a fused activation.";
  let description = [{
    NCHW input, OIHW weight. The optional bias is a per output channel vector.
    `activation` names an activation folded into the op. Keeping bias as an
    operand and activation as an attribute is what lets the fusion pass rewrite
    conv followed by bias add followed by relu into a single op.
  }];

  let arguments = (ins
    NPU_FloatTensor:$input,
    NPU_FloatTensor:$weight,
    Optional<NPU_FloatTensor>:$bias,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    DefaultValuedAttr<I64Attr, "1">:$group,
    DefaultValuedAttr<NPU_Activation, "::mlir::npu::Activation::none">:$activation
  );
  let results = (outs NPU_FloatTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
      `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.matmul
//===----------------------------------------------------------------------===//

def NPU_MatMulOp : NPU_Op<"matmul", [Pure]> {
  let summary = "Matrix multiply with an optional bias and a fused activation.";
  let description = [{
    Computes `lhs * rhs` for rank two operands, plus an optional bias broadcast
    over rows, plus a fused activation. This is the fully connected layer.
  }];

  let arguments = (ins
    NPU_FloatTensor:$lhs,
    NPU_FloatTensor:$rhs,
    Optional<NPU_FloatTensor>:$bias,
    DefaultValuedAttr<NPU_Activation, "::mlir::npu::Activation::none">:$activation
  );
  let results = (outs NPU_FloatTensor:$output);

  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $bias^)? attr-dict
      `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Pooling
//===----------------------------------------------------------------------===//

class NPU_PoolOp<string mnemonic> : NPU_Op<mnemonic, [Pure]> {
  let arguments = (ins
    NPU_FloatTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads
  );
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}

def NPU_MaxPool2DOp : NPU_PoolOp<"max_pool2d"> {
  let summary = "2D max pooling over an NCHW input.";
}

def NPU_AvgPool2DOp : NPU_PoolOp<"avg_pool2d"> {
  let summary = "2D average pooling over an NCHW input.";
}

//===----------------------------------------------------------------------===//
// Shape manipulation
//===----------------------------------------------------------------------===//

def NPU_ReshapeOp : NPU_Op<"reshape", [Pure]> {
  let summary = "Reshape a tensor, preserving the total number of elements.";
  let arguments = (ins NPU_FloatTensor:$input);
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

def NPU_TransposeOp : NPU_Op<"transpose", [Pure]> {
  let summary = "Permute the dimensions of a tensor.";
  let arguments = (ins NPU_FloatTensor:$input, I64ArrayAttr:$permutation);
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

def NPU_ConcatOp : NPU_Op<"concat", [Pure]> {
  let summary = "Concatenate tensors along a given axis.";
  let arguments = (ins Variadic<NPU_FloatTensor>:$inputs, I64Attr:$axis);
  let results = (outs NPU_FloatTensor:$output);
  let assemblyFormat = "$inputs attr-dict `:` functional-type($inputs, $output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.batch_norm
//===----------------------------------------------------------------------===//

def NPU_BatchNormOp : NPU_Op<"batch_norm", [Pure]> {
  let summary = "Inference batch normalization with constant parameters.";
  let description = [{
    Computes `output = scale * (input - mean) / sqrt(variance + epsilon) + offset`
    per channel over an NCHW input. At inference all four parameter tensors are
    constants, which is what the dedicated BatchNorm folding pass exploits to fold
    the normalization into a preceding convolution's weights.
  }];

  let arguments = (ins
    NPU_FloatTensor:$input,
    NPU_FloatTensor:$scale,
    NPU_FloatTensor:$offset,
    NPU_FloatTensor:$mean,
    NPU_FloatTensor:$variance,
    DefaultValuedAttr<F32Attr, "1.0e-5">:$epsilon
  );
  let results = (outs NPU_FloatTensor:$output);

  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance attr-dict
      `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // NPU_OPS_TD
