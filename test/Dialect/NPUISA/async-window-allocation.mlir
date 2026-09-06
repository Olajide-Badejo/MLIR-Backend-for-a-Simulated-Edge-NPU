// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The allocator against an asynchronous transfer's window, Section 8 rule 4.
//
// **An asynchronous transfer is not finished at its issue.** Its destination
// belongs to the DMA engine from the `npuisa.dma_load_async` until the matching
// `npuisa.await`, and the verifier says so from the other side: no operation
// between the two halves may touch the destination. The allocator has to hold
// the same belief, in two places that are easy to get wrong separately.
//
//   1. **Where a spill store goes.** The store copies the buffer out, so it
//      reads it, and reading a buffer the engine is still filling copies
//      whatever part of the transfer had landed. It goes after the await.
//   2. **How long the buffer is live.** The sweep line ends a range at the last
//      operation that names the buffer, and no operation names an in flight
//      destination between the two halves, so the range ended at the issue and
//      the bytes were handed to something defined inside the window.
//
// Both were live at once and only the first was visible, because the verifier
// catches a store it can compare and stays quiet about an offset it never sees
// as an operand. `docs/DEFECT_LOG.md` D-0054 has the program that showed it.

// RUN: npu-opt %s \
// RUN:   --npu-allocate-scratchpad="strategy=pack budget=1024 spill-heuristic=longest-range" \
// RUN:   | FileCheck %s
// RUN: npu-opt %s \
// RUN:   --npu-allocate-scratchpad="strategy=pack budget=1024 spill-heuristic=longest-range" \
// RUN:   | npu-opt --verify-diagnostics
// RUN: npu-opt %s --npu-allocate-scratchpad="budget=65536" \
// RUN:   | FileCheck %s --check-prefix=ROOMY

// =============================================================================
// The spill store waits for the await.
//
// 512 bytes are prefetched, a small computation runs under the transfer, and
// the pressure comes later: the two 512 byte buffers of the middle relu, plus
// the prefetched one, want 1536 against a budget of 1024. The longest range
// crossing that peak is the prefetched buffer, so it is the victim, and the
// spill relieves the peak because the pressure is between its write and its
// read rather than at either.
//
// **The `CHECK-NOT` between the two halves is the whole test.** A store placed
// straight after the issue, which is where a spill store goes when the writer
// is taken at face value, races the transfer it is copying, and the program it
// produces is one the verifier refuses. The second run line is that refusal
// stated as a fact rather than as an expectation: the output is fed back
// through the verifier and has to survive it.
// =============================================================================

// CHECK-LABEL: func.func @the_spill_store_waits_for_the_await
// CHECK:         npuisa.dma_load_async
// CHECK-NOT:     npuisa.spill_slot
// CHECK:         npuisa.await
// CHECK:         memref.alloc() {npuisa.spill_slot} : memref<1x8x4x4xf32, #npu.dram>
// CHECK-NEXT:    npuisa.dma_store
// CHECK-SAME:      memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>

// ROOMY-LABEL: func.func @the_spill_store_waits_for_the_await
// ROOMY-SAME:    npuisa.spill_count = 0 : i64
// ROOMY-NOT:     npuisa.spill_slot

func.func @the_spill_store_waits_for_the_await(
    %small: memref<1x1x4x4xf32, #npu.dram>,
    %wide: memref<1x8x4x4xf32, #npu.dram>,
    %outSmall: memref<1x1x4x4xf32, #npu.dram>,
    %outMiddle: memref<1x8x4x4xf32, #npu.dram>,
    %outLate: memref<1x8x4x4xf32, #npu.dram>) {
  %p = memref.alloc() : memref<1x1x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %small, %p
    : memref<1x1x4x4xf32, #npu.dram> to memref<1x1x4x4xf32, #npu.scratchpad>

  %w = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  %token = npuisa.dma_load_async %wide, %w
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>

  // The computation the transfer runs underneath, on buffers of its own.
  %q = memref.alloc() : memref<1x1x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%p : memref<1x1x4x4xf32, #npu.scratchpad>)
              outs(%q : memref<1x1x4x4xf32, #npu.scratchpad>)

  npuisa.await %token

  npuisa.dma_store %q, %outSmall
    : memref<1x1x4x4xf32, #npu.scratchpad> to memref<1x1x4x4xf32, #npu.dram>

  // The pressure, after the transfer has landed and before the prefetch is
  // read: three 512 byte buffers would be live at once.
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %wide, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %b, %outMiddle
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>

  // The read the prefetch existed for.
  %z = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%w : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%z : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %z, %outLate
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// =============================================================================
// The live range reaches the await, and this function is the rule at its
// minimum.
//
// Nothing reads the prefetched buffer, so every operation that names it sits at
// the issue, and a live range read off the operands alone ends there. The bytes
// are still being written until the await, and a program that put the relu's
// output at that offset would have the DMA engine and the array writing the
// same address at the same time. **The peak is the assertion**: 1024 counts the
// prefetched 512 alongside the two 256 byte buffers the computation needs, and
// 512 would be the answer that dropped it.
//
// It is written as a function nothing reads rather than as one that reads late,
// because a later read makes the range long for a second reason and the test
// would pass without the rule under test.
// =============================================================================

// CHECK-LABEL: func.func @an_unread_prefetch_still_owns_its_window
// CHECK-SAME:    npuisa.scratchpad_peak_bytes = 1024 : i64

// ROOMY-LABEL: func.func @an_unread_prefetch_still_owns_its_window
// ROOMY-SAME:    npuisa.scratchpad_peak_bytes = 1024 : i64

func.func @an_unread_prefetch_still_owns_its_window(
    %wide: memref<1x8x4x4xf32, #npu.dram>,
    %half: memref<1x4x4x4xf32, #npu.dram>,
    %out: memref<1x4x4x4xf32, #npu.dram>) {
  %w = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  %token = npuisa.dma_load_async %wide, %w
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>

  %a = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %half, %a
    : memref<1x4x4x4xf32, #npu.dram> to memref<1x4x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x4x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x4x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x4x4x4xf32, #npu.scratchpad>)

  npuisa.await %token

  npuisa.dma_store %b, %out
    : memref<1x4x4x4xf32, #npu.scratchpad> to memref<1x4x4x4xf32, #npu.dram>
  return
}
