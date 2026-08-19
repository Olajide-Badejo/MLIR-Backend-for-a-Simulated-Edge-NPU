// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The memory model, as IR that round trips. Where ops.mlir pins what each
// operation prints, this file pins the shape of the memory the operations work
// on: two spaces, allocation in the scratchpad, and the flat buffer with views
// over it that the allocator of Section 13.1 will produce.
//
// It exists separately because these are different claims. An operation whose
// assembly format regressed breaks ops.mlir; a memory space attribute that
// stopped parsing on a memref, or a view over a flat scratchpad buffer that
// stopped verifying, breaks this one, and keeping them apart means the failure
// says which.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// -----------------------------------------------------------------------------
// The two spaces on a memref.
// -----------------------------------------------------------------------------

// The attributes print exactly as written, with no extra word. That is what the
// zero parameter attribute definition buys over one case of an enum, which
// would have printed as #npu.space<scratchpad>.
// CHECK-LABEL: func.func @the_two_spaces
// CHECK-SAME: memref<4x4xf32, #npu.scratchpad>
// CHECK-SAME: memref<4x4xf32, #npu.dram>
func.func @the_two_spaces(%s: memref<4x4xf32, #npu.scratchpad>,
                          %d: memref<4x4xf32, #npu.dram>) {
  return
}

// -----------------------------------------------------------------------------
// Scratchpad space is created by memref.alloc in that space.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @alloc_in_the_scratchpad
func.func @alloc_in_the_scratchpad() {
  // CHECK: memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  %buffer = memref.alloc() : memref<8x8xf32, #npu.scratchpad>
  // CHECK: npuisa.relu
  npuisa.relu ins(%buffer : memref<8x8xf32, #npu.scratchpad>)
              outs(%buffer : memref<8x8xf32, #npu.scratchpad>)
  memref.dealloc %buffer : memref<8x8xf32, #npu.scratchpad>
  return
}

// -----------------------------------------------------------------------------
// The flat buffer and the views over it.
//
// This is the shape AllocateScratchpad produces: one flat memref<Nxi8> in the
// scratchpad, and a memref.view per allocation at its assigned byte offset. The
// offset is an SSA operand rather than a discardable attribute, so every
// verifier and consumer can see it and a pass cannot silently drop it.
// -----------------------------------------------------------------------------

// Two views at offsets 0 and 256 over one 1024 byte buffer. Each is 8 by 8 f32,
// which is 256 bytes, so byte ranges [0, 256) and [256, 512) are adjacent and
// disjoint. Two distinct SSA values over one allocation, and they do not race.
// CHECK-LABEL: func.func @two_disjoint_views
func.func @two_disjoint_views() {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  // CHECK: memref.alloc() : memref<1024xi8, #npu.scratchpad>
  %flat = memref.alloc() : memref<1024xi8, #npu.scratchpad>
  // CHECK: memref.view
  %a = memref.view %flat[%c0][]
     : memref<1024xi8, #npu.scratchpad> to memref<8x8xf32, #npu.scratchpad>
  // CHECK: memref.view
  %b = memref.view %flat[%c256][]
     : memref<1024xi8, #npu.scratchpad> to memref<8x8xf32, #npu.scratchpad>
  // CHECK: npuisa.add
  npuisa.add ins(%a, %b : memref<8x8xf32, #npu.scratchpad>,
                          memref<8x8xf32, #npu.scratchpad>)
             outs(%a : memref<8x8xf32, #npu.scratchpad>)
  memref.dealloc %flat : memref<1024xi8, #npu.scratchpad>
  return
}

// An asynchronous transfer whose destination is a view, with a compute
// instruction on a disjoint view of the same flat buffer between it and its
// await. This is the case Section 8 is written for and the case an SSA identity
// check would get right for the wrong reason: the two values are different, and
// they are also genuinely disjoint, and only the byte range arithmetic knows
// the second half.
// CHECK-LABEL: func.func @async_over_a_disjoint_view
func.func @async_over_a_disjoint_view(%src: memref<8x8xf32, #npu.dram>) {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %flat = memref.alloc() : memref<1024xi8, #npu.scratchpad>
  %dst = memref.view %flat[%c0][]
       : memref<1024xi8, #npu.scratchpad> to memref<8x8xf32, #npu.scratchpad>
  %other = memref.view %flat[%c256][]
         : memref<1024xi8, #npu.scratchpad> to memref<8x8xf32, #npu.scratchpad>
  // CHECK: %[[T:.*]] = npuisa.dma_load_async
  %t = npuisa.dma_load_async %src, %dst
     : memref<8x8xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  // CHECK: npuisa.relu
  npuisa.relu ins(%other : memref<8x8xf32, #npu.scratchpad>)
              outs(%other : memref<8x8xf32, #npu.scratchpad>)
  // CHECK: npuisa.await %[[T]]
  npuisa.await %t
  memref.dealloc %flat : memref<1024xi8, #npu.scratchpad>
  return
}

// -----------------------------------------------------------------------------
// The boundary shape: DRAM in, scratchpad through, DRAM out.
// -----------------------------------------------------------------------------

// What a lowered function looks like: function arguments in DRAM, one load per
// value entering the scratchpad, the computation on scratchpad buffers, one
// store per value returned. Section 8 scopes that invariant to exactly this
// point in the pipeline, immediately after lowering and before allocation, and
// test/Dialect/NPUISA/dma-boundaries.mlir is what asserts it.
// CHECK-LABEL: func.func @a_lowered_function
func.func @a_lowered_function(%x: memref<1x8x4x4xf32, #npu.dram>,
                              %out: memref<1x8x4x4xf32, #npu.dram>)
    attributes {npuisa.scratchpad_bytes = 1024 : i64,
                npuisa.scratchpad_budget = 65536 : i64} {
  %in = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  %tmp = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.dma_load
  npuisa.dma_load %x, %in
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK: npuisa.relu
  npuisa.relu ins(%in : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%tmp : memref<1x8x4x4xf32, #npu.scratchpad>)
  // CHECK: npuisa.dma_store
  npuisa.dma_store %tmp, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  memref.dealloc %in : memref<1x8x4x4xf32, #npu.scratchpad>
  memref.dealloc %tmp : memref<1x8x4x4xf32, #npu.scratchpad>
  return
}
