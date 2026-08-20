// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A budget too small is a diagnostic with the numbers in it, never a wrong
// program.
//
// Section 13.1 asks for this case by name and the P5 gate asks for it under
// `-verify-diagnostics` with a counted error. The message carries four numbers,
// and each of them is there because a reader trying to fix the failure needs
// it: the budget, the size of the buffer that could not be placed, the offset
// it would have had to start at, and the sweep line peak, which is a lower
// bound on any placement and therefore says whether raising the budget could
// ever help.
//
// The error is attached to the allocation that could not be placed rather than
// to the function, so the line the reader is sent to is the buffer rather than
// the signature.

// RUN: npu-opt %s -split-input-file -verify-diagnostics \
// RUN:   --npu-allocate-scratchpad=budget=256

// =============================================================================
// Nothing is spillable, because nothing reads the buffer after it is written.
// Spilling it would emit a `dma_store` and shorten no live range, so the
// allocator refuses rather than moving DRAM traffic for nothing.
// =============================================================================

func.func @one_buffer_larger_than_the_budget(
    %in: memref<1x8x4x4xf32, #npu.dram>) {
  // expected-error @+1 {{the scratchpad budget of 256 bytes is too small: this buffer of 512 bytes could not be placed below offset 0}}
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  return
}

// -----

// =============================================================================
// The peak is quoted as a lower bound, which is what tells the reader that this
// budget cannot be made to work by a better placement. Three 512 byte buffers
// are live at the addition, so no arrangement of them fits in 256 bytes and no
// amount of spilling changes that: an instruction's operands have to be
// resident while it runs.
// =============================================================================

func.func @spilling_everything_spillable_is_still_not_enough(
    %in: memref<1x8x4x4xf32, #npu.dram>,
    %out: memref<1x8x4x4xf32, #npu.dram>) {
  %long = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %long
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %t = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%long : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%t : memref<1x8x4x4xf32, #npu.scratchpad>)
  // The buffer named is the one that could not be placed after the spill loop
  // had run out of candidates, which is the addition's destination and not the
  // first allocation in the function. That is the useful answer: it is the
  // allocation that was still homeless when the allocator gave up.
  // expected-error @+1 {{no buffer live across the pressure peak can be spilled. The sweep line peak is 1536 bytes, which is a lower bound on any placement}}
  %sum = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.add ins(%t, %long : memref<1x8x4x4xf32, #npu.scratchpad>,
                             memref<1x8x4x4xf32, #npu.scratchpad>)
             outs(%sum : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %sum, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
