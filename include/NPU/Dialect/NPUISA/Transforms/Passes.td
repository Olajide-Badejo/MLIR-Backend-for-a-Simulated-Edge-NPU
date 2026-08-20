//===- Passes.td - npuisa transformation passes ------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_PASSES
#define NPUISA_PASSES

include "mlir/Pass/PassBase.td"

def NPULowerToNPUISA : Pass<"npu-lower-to-npuisa", "::mlir::ModuleOp"> {
  let summary = "Lowers the npu tensor dialect to npuisa instructions on memrefs.";
  let description = [{
    The dialect conversion of Section 5.1: `npu` operations on tensors become
    `npuisa` instructions on `memref`s in the two memory spaces of Section 8.

    Function arguments and constants land in `#npu.dram`. Everything the
    machine computes on lands in `#npu.scratchpad`. Exactly one
    `npuisa.dma_load` is emitted per DRAM value that enters the scratchpad and
    exactly one `npuisa.dma_store` per returned value, and immediately after
    this pass those are the only DMA operations in the program. That is
    Section 8's scoped boundary invariant, this pass is its first permitted
    producer, and `test/Dialect/NPUISA/dma-boundaries.mlir` asserts it at
    exactly this point.

    Function results become trailing DRAM arguments. A lowered function takes
    its inputs and its outputs as parameters and returns nothing, which is the
    boundary shape `test/Dialect/NPUISA/ops-memref.mlir` has pinned since P2
    and the shape the encoder's input and output regions are read out of.

    The destination operand of every `npu` compute operation is what the
    conversion writes into: a `tensor.empty` destination becomes a
    `memref.alloc` in the scratchpad and the instruction's `outs`, so
    destination passing is preserved rather than reconstructed.

    Three rewrites happen before the conversion proper, on tensors, because
    each of them is a tensor level identity and writing it against memrefs
    would mean writing it twice:

    - an `npu.fused_op` region is flattened into its parent, which is what
      keeps a fused chain's intermediate in the scratchpad with no DMA between
      the producer and the consumer;
    - an `npu.batch_norm` the folding pass did not fold is decomposed into a
      multiply and an add with the per channel constants computed at rewrite
      time, so an unfolded batch norm is legal rather than a hard error;
    - nothing else, deliberately.

    A rank 1 right hand operand on `npu.add` or `npu.mul`, which is the channel
    broadcast of `docs/adr/0005-channel-broadcast-on-add-and-mul.md`, lowers to
    a `memref.reinterpret_cast` view of the rank 1 buffer whose extents are the
    destination's and whose strides are 1 on the channel axis and 0 everywhere
    else. That is the operand shape the record obliges: a channel stride of one
    and a spatial stride of zero, carried in the type rather than in a flag.

    This pass is **not ablatable**, per Section 12: removing it produces no
    program at all, so an ablation of it measures nothing.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::npuisa::NPUISADialect",
    "::mlir::tensor::TensorDialect",
  ];
}

#endif // NPUISA_PASSES
