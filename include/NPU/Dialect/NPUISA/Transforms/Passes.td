//===- Passes.td - NPUISA transform passes -----------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_TRANSFORMS_PASSES_TD
#define NPUISA_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def NPULowerToNPUISA : Pass<"npu-lower-to-npuisa", "::mlir::func::FuncOp"> {
  let summary = "Lower the npu tensor dialect to the npuisa instruction dialect.";
  let description = [{
    A representation changing lowering built on the dialect conversion framework.
    A `TypeConverter` maps every `tensor` (a DRAM value) to an `!npuisa.buffer`
    (a scratchpad value). Compute ops become their npuisa instruction
    counterparts. DMA is inserted only at the boundaries between the two memories:
    a `dma_load` materializes a scratchpad buffer from a DRAM tensor (a function
    argument or a constant), and a `dma_store` materializes a DRAM tensor from a
    scratchpad buffer (a returned value). Constants become an `npuisa.const` in
    DRAM followed by a `dma_load`.
  }];
  let dependentDialects = [
    "::mlir::npu::NPUDialect",
    "::mlir::npuisa::NPUISADialect",
  ];
}

#endif // NPUISA_TRANSFORMS_PASSES_TD
