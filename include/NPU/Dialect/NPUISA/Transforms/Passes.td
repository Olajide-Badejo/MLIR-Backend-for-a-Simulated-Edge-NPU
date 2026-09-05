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

def NPUAllocateScratchpad
    : Pass<"npu-allocate-scratchpad", "::mlir::func::FuncOp"> {
  let summary = "Assigns every scratchpad buffer a byte offset in one arena.";
  let description = [{
    Section 13.1. Liveness runs over the `memref.alloc` operations in the
    scratchpad space of a straight line function, each range running from the
    allocation to the last operation that reads or writes that memref. Sizes
    come from the memref type and its element type.

    Peak pressure is computed with a **sweep line**: one event list of
    `(index, +size)` at each definition and `(lastUse + 1, -size)` at each
    death, sorted once and walked once. The first index with a strictly greater
    sum wins the peak, and at equal indices deaths are ordered before
    definitions.

    **The peak is a lower bound, not a placement test.** Fragmentation means a
    program whose peak sits under the budget can still fail to place, so the
    spill trigger is "offset assignment failed" and never "peak exceeded
    budget".

    Each buffer is then assigned a byte offset and rewritten as a `memref.view`
    over one flat `memref<Nxi8, #npu.scratchpad>`, which is Section 8's rule
    that the offset is an SSA operand rather than a discardable attribute. A
    buffer whose type carries a strided layout map gets a `memref.view` at its
    extents followed by a `memref.reinterpret_cast` that restores the layout,
    because `memref.view` requires an identity layout on its result.

    When offset assignment fails, one buffer live across the pressure peak is
    spilled: a `npuisa.dma_store` after its definition and a `npuisa.dma_load`
    before each later use, with the reload replacing that use. That is the
    second of the three permitted DMA producers of Section 8, and the count is
    recorded on the function. A budget too small even after everything
    spillable has been spilled is a diagnostic with the numbers in it, never a
    wrong program.

    The function comes out carrying `npuisa.scratchpad_budget`,
    `npuisa.scratchpad_bytes`, `npuisa.scratchpad_peak_bytes`,
    `npuisa.fragmentation_ratio`, `npuisa.spill_count` and
    `npuisa.spill_dma_count`. A function that already carries
    `npuisa.scratchpad_bytes` is left alone, which makes the pass idempotent:
    the arena is itself a scratchpad allocation, and a second run without that
    guard would allocate an arena for the arena.

    This pass is **not ablatable**, per Section 12: removing it produces no
    program at all, so an ablation of it measures nothing.
  }];

  let options = [
    Option<"strategy", "strategy", "std::string", "\"pack\"",
           "The offset assignment strategy: 'pack', the greedy by size "
           "algorithm of TFLite Micro's arena planner, or 'interval', the "
           "named baseline that places in definition order. An unknown value "
           "is a diagnostic naming it and listing the accepted values.">,
    Option<"spillHeuristic", "spill-heuristic", "std::string",
           "\"longest-range\"",
           "Which buffer to spill: 'longest-range', the longest live range "
           "crossing the pressure peak, or 'cost', the Belady style rule "
           "bytes * (1 + reloads) with the smallest cost spilled. The default "
           "is provisional and Section 13.1 requires it to be chosen with the "
           "ablation data that lands at P13.">,
    Option<"budget", "budget", "int64_t", "-1",
           "The scratchpad budget in bytes. Minus one means take it from the "
           "function's npuisa.scratchpad_budget attribute, or from the default "
           "of 1048576 when the function carries none.">,
    Option<"alignment", "alignment", "int64_t", "64",
           "The byte alignment of every assigned offset. A positive power of "
           "two. The default is 64 because the array of Section 5.3 consumes "
           "a row of 16 f32 lanes at a time.">,
  ];

  let statistics = [
    Statistic<"allocatedBuffers", "allocated-buffers",
              "Scratchpad buffers given an offset">,
    Statistic<"allocatedBytes", "allocated-bytes",
              "Assigned high water mark, summed over functions">,
    Statistic<"peakBytes", "peak-bytes",
              "Sweep line peak, summed over functions">,
    Statistic<"spilledBufferCount", "spilled-buffers",
              "Buffers spilled to DRAM">,
    Statistic<"insertedDmaCount", "inserted-dma",
              "DMA operations this pass inserted, per Section 8's count">,
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::npuisa::NPUISADialect",
  ];
}

def NPUDoubleBuffer : Pass<"npu-double-buffer", "::mlir::func::FuncOp"> {
  let summary = "Overlaps a transfer with the computation before it.";
  let description = [{
    Section 5.1's optional double buffering over asynchronous DMA tokens,
    **running before allocation and not after**. That order is the whole reason
    the pass has a fixed place in the pipeline: double buffering doubles the
    working set of the prefetched operand, because the next tile's buffer is
    live while the current tile computes, and the allocator has to see the
    doubled set or it places a program that cannot fit and spills the wrong
    buffers.

    A `npuisa.dma_load` is hoisted above the computation that precedes it and
    becomes a `npuisa.dma_load_async`, with an `npuisa.await` left where the
    load was. The transfer then runs underneath that computation instead of
    after it, which is what the two port cost model of Section 5.5 charges: a
    program whose transfers and computations overlap reports an
    `overlap_fraction` above zero, and one whose transfers all wait reports
    zero.

    **Four conditions, and the third is the one that makes it safe.**

    - The load's own operands have to be available at the earlier point, so the
      hoist stops at whatever defines its source or its destination.
    - There has to be something to overlap with. A load hoisted past nothing is
      an asynchronous operation whose `await` is the next operation, which
      canonicalizes straight back to the synchronous form, so the pass declines
      rather than emitting a token nothing gains from.
    - **Nothing between the asynchronous operation and its await may touch the
      destination**, which is Section 8's rule 4 and is decided by
      `npuisa::overlaps` rather than by comparing SSA values. Two different
      views over one buffer are distinct values and one shared region of memory,
      and an identity check cannot see the race. An `Unknown` answer stops the
      hoist, because "I cannot prove these overlap" and "these are disjoint" are
      different answers and only one of them is safe.
    - The destination must not be read between the two points either, for the
      same reason stated from the other side: the hardware owns those bytes for
      the whole of that window.

    **It moves no golden by itself and that is worth saying.** The instruction
    stream is the same instructions in a different order with one token added,
    the encoder emits `DMA_LOAD` for both forms because the machine's DMA engine
    is asynchronous either way, and the arithmetic each kernel performs is
    untouched. What moves is when a transfer is charged against the compute
    timeline, which is a scheduling result and shows up in `overlap_fraction`
    and in the cycle total, never in an output tensor.
  }];

  let statistics = [
    Statistic<"prefetched", "prefetched",
              "Transfers made asynchronous and hoisted above a computation">,
    Statistic<"notHoisted", "not-hoisted",
              "Transfers with nothing safe to overlap, left synchronous">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::npuisa::NPUISADialect",
  ];
}

#endif // NPUISA_PASSES
