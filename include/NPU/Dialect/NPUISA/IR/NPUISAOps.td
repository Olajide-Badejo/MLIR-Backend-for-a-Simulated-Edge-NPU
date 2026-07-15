//===- NPUISAOps.td - NPUISA dialect operations ------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_OPS_TD
#define NPUISA_OPS_TD

include "NPU/Dialect/NPUISA/IR/NPUISADialect.td"
include "NPU/Dialect/NPUISA/IR/NPUISATypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NPUISA_FloatTensor : RankedTensorOf<[F32]>;

// Optional scratchpad byte offset, filled in by the allocator.
def NPUISA_AddressAttr : OptionalAttr<I64Attr>;
// Fused activation: 0 = none, 1 = relu, matching the npu Activation enum storage.
def NPUISA_ActivationAttr : DefaultValuedAttr<I32Attr, "0">;

//===----------------------------------------------------------------------===//
// Control
//===----------------------------------------------------------------------===//

def NPUISA_NopOp : NPUISA_Op<"nop"> {
  let summary = "No operation.";
  let assemblyFormat = "attr-dict";
}

def NPUISA_HaltOp : NPUISA_Op<"halt", [Terminator]> {
  let summary = "Halt the instruction stream.";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// DRAM constant
//===----------------------------------------------------------------------===//

def NPUISA_ConstOp : NPUISA_Op<"const", [Pure, ConstantLike]> {
  let summary = "A constant tensor resident in DRAM.";
  let description = [{
    Holds a weight, bias, or other constant in DRAM. A dma_load brings it into the
    scratchpad before a compute instruction can use it.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs NPUISA_FloatTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// DMA between DRAM and scratchpad
//===----------------------------------------------------------------------===//

def NPUISA_DmaLoadOp : NPUISA_Op<"dma_load"> {
  let summary = "Copy a DRAM tensor into a scratchpad buffer.";
  let arguments = (ins NPUISA_FloatTensor:$source, NPUISA_AddressAttr:$address);
  let results = (outs NPUISA_BufferType:$dest);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, $dest)";
  let hasVerifier = 1;
}

def NPUISA_DmaStoreOp : NPUISA_Op<"dma_store"> {
  let summary = "Copy a scratchpad buffer back to a DRAM tensor.";
  let arguments = (ins NPUISA_BufferType:$source);
  let results = (outs NPUISA_FloatTensor:$dest);
  let assemblyFormat = "$source attr-dict `:` functional-type($source, $dest)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Compute instructions (scratchpad to scratchpad)
//===----------------------------------------------------------------------===//

def NPUISA_Conv2DOp : NPUISA_Op<"conv2d"> {
  let summary = "Convolution instruction with an optional bias and activation.";
  let arguments = (ins
    NPUISA_BufferType:$input,
    NPUISA_BufferType:$weight,
    Optional<NPUISA_BufferType>:$bias,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    DefaultValuedAttr<I64Attr, "1">:$group,
    NPUISA_ActivationAttr:$activation,
    NPUISA_AddressAttr:$address
  );
  let results = (outs NPUISA_BufferType:$output);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:` functional-type(operands, results)
  }];
}

def NPUISA_MatMulOp : NPUISA_Op<"matmul"> {
  let summary = "Matrix multiply instruction with an optional bias and activation.";
  let arguments = (ins
    NPUISA_BufferType:$lhs,
    NPUISA_BufferType:$rhs,
    Optional<NPUISA_BufferType>:$bias,
    NPUISA_ActivationAttr:$activation,
    NPUISA_AddressAttr:$address
  );
  let results = (outs NPUISA_BufferType:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $bias^)? attr-dict `:` functional-type(operands, results)
  }];
}

class NPUISA_UnaryOp<string mnemonic> : NPUISA_Op<mnemonic> {
  let arguments = (ins NPUISA_BufferType:$input, NPUISA_AddressAttr:$address);
  let results = (outs NPUISA_BufferType:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
}

def NPUISA_ReluOp : NPUISA_UnaryOp<"relu"> {
  let summary = "Elementwise relu instruction.";
}

def NPUISA_ReshapeOp : NPUISA_UnaryOp<"reshape"> {
  let summary = "Reinterpret a buffer as a different shape (no data movement).";
}

class NPUISA_BinaryOp<string mnemonic> : NPUISA_Op<mnemonic> {
  let arguments = (ins NPUISA_BufferType:$lhs, NPUISA_BufferType:$rhs,
                       NPUISA_AddressAttr:$address);
  let results = (outs NPUISA_BufferType:$output);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def NPUISA_AddOp : NPUISA_BinaryOp<"add"> {
  let summary = "Elementwise add instruction.";
}

def NPUISA_MulOp : NPUISA_BinaryOp<"mul"> {
  let summary = "Elementwise multiply instruction.";
}

class NPUISA_PoolOp<string mnemonic> : NPUISA_Op<mnemonic> {
  let arguments = (ins
    NPUISA_BufferType:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    NPUISA_AddressAttr:$address
  );
  let results = (outs NPUISA_BufferType:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
}

def NPUISA_PoolMaxOp : NPUISA_PoolOp<"pool_max"> {
  let summary = "Max pooling instruction.";
}

def NPUISA_PoolAvgOp : NPUISA_PoolOp<"pool_avg"> {
  let summary = "Average pooling instruction.";
}

#endif // NPUISA_OPS_TD
