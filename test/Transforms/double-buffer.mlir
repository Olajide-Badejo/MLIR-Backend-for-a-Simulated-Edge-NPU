// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-double-buffer`, Section 5.1, running **before allocation**.
//
// A `npuisa.dma_load` is hoisted above the computation before it and becomes a
// `npuisa.dma_load_async` with an `npuisa.await` left where it was. The
// transfer then runs underneath that computation instead of after it.
//
// The IR here is hand written `npuisa` on memrefs rather than `npu` tensors
// piped through the pipeline, for the reason `all_ops.mlir` gives: what is
// under test is this pass, and starting at the tensor level would make the
// test depend on what the lowering happens to emit today.
//
// **The positions are asserted exactly rather than the presence of a token.**
// A test that checked only that some `dma_load_async` appeared would pass
// against a pass that hoisted nothing and merely renamed the operation, which
// is the standard the halo byte model bug set: assert the choice, not that a
// choice was made.

// RUN: npu-opt %s --npu-double-buffer | FileCheck %s
// RUN: npu-opt %s --npu-double-buffer \
// RUN:   -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

// -----------------------------------------------------------------------------
// Positive: the second load moves above the first computation.
//
// The order it starts in is load, load, compute, compute. The second load's
// destination is untouched by the first computation, so it can start while that
// computation runs, and the `await` lands immediately before the consumer that
// needs it.
//
// **The allocation moves with the transfer and that is the point, not an
// accident.** A tile's destination buffer is defined immediately before the
// load that fills it, so a hoist that left the allocation behind would move
// nothing at all; and extending that buffer's live range is exactly what double
// buffering costs, which is why Section 5.1 puts this pass before the allocator
// rather than after it.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @the_second_load_runs_under_the_first_computation
// The first transfer is untouched: there is nothing before it to hide it under.
// CHECK:         %[[A:[a-z_0-9]+]] = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
// CHECK-NEXT:    npuisa.dma_load %arg0, %[[A]]
//
// The second transfer's buffer is allocated and its transfer started **before**
// the first computation rather than after it.
// CHECK-NEXT:    %[[B:[a-z_0-9]+]] = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
// CHECK-NEXT:    %[[T:[0-9]+]] = npuisa.dma_load_async %arg1, %[[B]]
// CHECK-NEXT:    npuisa.relu ins(%[[A]]
//
// And the wait is immediately before the operation that reads it.
// CHECK:         npuisa.await %[[T]]
// CHECK-NEXT:    npuisa.relu ins(%[[B]]
func.func @the_second_load_runs_under_the_first_computation(
    %x: memref<8x8xf32, #npu.dram>, %y: memref<8x8xf32, #npu.dram>,
    %o1: memref<8x8xf32, #npu.scratchpad>,
    %o2: memref<8x8xf32, #npu.scratchpad>) {
  %a = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  npuisa.dma_load %x, %a
      : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<8x8xf32, #npu.scratchpad>)
              outs(%o1 : memref<8x8xf32, #npu.scratchpad>)
  %b = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  npuisa.dma_load %y, %b
      : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  npuisa.relu ins(%b : memref<8x8xf32, #npu.scratchpad>)
              outs(%o2 : memref<8x8xf32, #npu.scratchpad>)
  return
}

// -----------------------------------------------------------------------------
// Negative one: a load with nothing before it stays synchronous.
//
// **A pass with only positive tests is not adequately tested**, because a pass
// that rewrote every load unconditionally would pass them all. A load hoisted
// past nothing would become an asynchronous operation whose `await` is the next
// operation, which the dialect canonicalizes straight back to the synchronous
// form; declining is the same answer without the residue.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @nothing_to_hide_it_under
// CHECK-NOT:     npuisa.dma_load_async
// CHECK-NOT:     npuisa.await
// CHECK:         npuisa.dma_load %arg0
func.func @nothing_to_hide_it_under(%x: memref<8x8xf32, #npu.dram>,
                                    %o: memref<8x8xf32, #npu.scratchpad>) {
  %a = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  npuisa.dma_load %x, %a
      : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<8x8xf32, #npu.scratchpad>)
              outs(%o : memref<8x8xf32, #npu.scratchpad>)
  return
}

// -----------------------------------------------------------------------------
// Negative two: the computation writes the buffer the transfer would fill, so
// the transfer stays where it is.
//
// **This is Section 8's rule 4 and it is the one that makes the rewrite safe.**
// The hardware owns the destination for the whole window between the
// asynchronous operation and its await, so an operation in that window which
// touches those bytes is a race. Here the first `relu` writes `%a` and the
// second transfer fills `%a`, so hoisting the transfer above the `relu` would
// let the two write the same buffer at once.
//
// The decision is `npuisa::overlaps` rather than a comparison of SSA values,
// which matters more after allocation than here: two different views over one
// flat buffer are distinct values and one shared region of memory.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @the_computation_writes_what_the_transfer_would_fill
// CHECK-NOT:     npuisa.dma_load_async
// CHECK:         npuisa.relu
// CHECK:         npuisa.dma_load %arg1, %{{[a-z_0-9]+}}
func.func @the_computation_writes_what_the_transfer_would_fill(
    %x: memref<8x8xf32, #npu.dram>, %y: memref<8x8xf32, #npu.dram>,
    %o: memref<8x8xf32, #npu.scratchpad>) {
  %a = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  npuisa.dma_load %x, %a
      : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  // Writes %a, which is what the next transfer fills.
  npuisa.relu ins(%o : memref<8x8xf32, #npu.scratchpad>)
              outs(%a : memref<8x8xf32, #npu.scratchpad>)
  npuisa.dma_load %y, %a
      : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  return
}

// **One transfer hoisted out of the five in this file, and four declined.** The
// count is asserted rather than left implied, because it is the only thing that
// distinguishes a pass that made one deliberate choice from one that happened
// to leave the right operation alone. The five are: the first function's two,
// the second function's one, and the third function's two, of which exactly one
// moves, the first function's second load.
// STATS: 4 not-hoisted
// STATS: 1 prefetched
