// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The view rule of Section 13.1's spilling, at the width its reason supports.
//
// **A buffer a view is taken of used to be unspillable and now is not, and the
// difference is which views a reload can serve.** `spill` rewrites a buffer's
// later uses onto the reload, and a view's use of the buffer is its source
// operand, so the same rewrite re-bases a view without knowing it is one. What
// a reload genuinely cannot serve is a view that is **written through**: the
// write would land in the reload and the buffer the store put in DRAM would
// never see it.
//
// **D-0056 is why this file exists.** A tiled convolution reads slices of its
// input, and under the per slice convention a slice of a scratchpad value is a
// view and no transfer, so the producer stays whole resident **and** was
// unspillable. Both at once is what made one cell of the suite unplaceable at a
// budget it had always placed at.

// RUN: npu-opt %s --npu-allocate-scratchpad=budget=192 \
// RUN:   -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS
// RUN: npu-opt %s --npu-allocate-scratchpad=budget=192 | FileCheck %s
// RUN: not npu-opt %s --npu-allocate-scratchpad=budget=64 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WORKING

// -----------------------------------------------------------------------------
// A buffer read only through a view is spilled, and the view moves to the
// reload.
//
// The first buffer is written at the top, read through a `memref.subview`
// immediately after, and read again at the very end. Between those two reads
// there is a stretch where it is not used at all and three other buffers are,
// and that stretch is the pressure peak. Spilling it takes 64 bytes out of the
// peak and the program places; leaving it in, which is what the old rule forced,
// leaves the peak above the budget with nothing else legal to spill.
// -----------------------------------------------------------------------------

// **One buffer spilled**, and it is the one the view is taken of.
// STATS: NPUAllocateScratchpad
// STATS: 1 spilled-buffers

// The store after its definition and a reload before each later use, which is
// Section 13.1's spill semantics unchanged. The view below is re-based onto the
// reload by the same rewrite that moves a reader.
// CHECK-LABEL: func.func @a_view_only_read_through_does_not_block_the_spill
// CHECK:         npuisa.dma_store
// CHECK:         npuisa.dma_load
// CHECK:         memref.subview
func.func @a_view_only_read_through_does_not_block_the_spill(
    %in: memref<4x4xf32, #npu.dram> {npuisa.arg = "in"},
    %out: memref<4x4xf32, #npu.dram> {npuisa.arg = "out"}) {
  %a = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>

  // The view, read through and never written through, so a reload can serve it.
  %view = memref.subview %a[0, 0] [2, 4] [1, 1]
      : memref<4x4xf32, #npu.scratchpad>
        to memref<2x4xf32, strided<[4, 1]>, #npu.scratchpad>
  %d = memref.alloc() : memref<2x4xf32, #npu.scratchpad>
  npuisa.relu ins(%view : memref<2x4xf32, strided<[4, 1]>, #npu.scratchpad>)
              outs(%d : memref<2x4xf32, #npu.scratchpad>)

  // The stretch the first buffer is not used across, which is where the peak
  // is. Three buffers of its own size live at once here.
  %p = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %p
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  %q = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%p : memref<4x4xf32, #npu.scratchpad>)
              outs(%q : memref<4x4xf32, #npu.scratchpad>)
  %r = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.add ins(%p, %q : memref<4x4xf32, #npu.scratchpad>,
                          memref<4x4xf32, #npu.scratchpad>)
             outs(%r : memref<4x4xf32, #npu.scratchpad>)

  // And the later read that makes spilling worth anything.
  %e = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.add ins(%a, %r : memref<4x4xf32, #npu.scratchpad>,
                          memref<4x4xf32, #npu.scratchpad>)
             outs(%e : memref<4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %e, %out
      : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
  return
}

// -----------------------------------------------------------------------------
// The failure says which rule refused which candidate.
//
// A budget nothing can be placed in, so the allocator reaches its no victim
// branch and prints its working. **A message that says no buffer can be spilled
// and stops is the shape of failure this project spends sessions on**, which is
// D-0056's other half: a reader had to reconstruct five predicates over a dozen
// buffers to find out whether the allocator was right, and this is the session
// that spent a morning doing exactly that.
// -----------------------------------------------------------------------------

// WORKING: the scratchpad budget of 64 bytes is too small
// WORKING-SAME: The peak is at operation
// WORKING: bytes, live [
// WORKING-SAME: uses after the peak:
