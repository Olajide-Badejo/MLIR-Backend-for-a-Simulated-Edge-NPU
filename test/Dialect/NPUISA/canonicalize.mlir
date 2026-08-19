// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The one canonicalization this dialect has: an asynchronous transfer whose
// `await` is the very next operation folds back to the synchronous form.
//
// Section 8's reason for it is worth repeating here, because it is what decides
// how strict the pattern is. A scheduling pass emits the asynchronous form
// speculatively, looks for work to overlap the transfer with, and sometimes
// finds none. Without this fold that pass leaves a token, an await, and two
// operations where one would do, which the encoder would then have to special
// case. With it, a pass that found nothing to overlap leaves no residue.
//
// The negative half of this file is the load bearing half. A fold that fired
// whenever an await was merely somewhere later in the block would undo a
// scheduling decision somebody made on purpose, and a positive test alone
// cannot tell a correct pattern from that one.
//
// The section banners in this file are written with `=` rather than `-`. A
// comment line of dashes is what `-split-input-file` looks for, so a banner
// drawn with dashes silently becomes a split marker and every check after it
// lands in the wrong section. That cost a debugging round here and is written
// down so it does not cost another one in the next test file.

// RUN: npu-opt %s -canonicalize -split-input-file | FileCheck %s

// =============================================================================
// The fold fires: the await is the very next operation.
// =============================================================================

// CHECK-LABEL: func.func @load_async_with_immediate_await
// CHECK: npuisa.dma_load %{{[^ ]*}}, %{{[^ ]*}} : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
// CHECK-NOT: npuisa.dma_load_async
// CHECK-NOT: npuisa.await
func.func @load_async_with_immediate_await(
    %src: memref<16x16xf32, #npu.dram>,
    %dst: memref<16x16xf32, #npu.scratchpad>) {
  %t = npuisa.dma_load_async %src, %dst
     : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  npuisa.await %t
  return
}

// -----

// The store direction folds to npuisa.dma_store, not to npuisa.dma_load. The
// two are separate operations rather than one with a direction attribute, so a
// pattern that used the wrong one would produce IR that does not verify, and
// this check is what would catch a copy and paste between the two.
// CHECK-LABEL: func.func @store_async_with_immediate_await
// CHECK: npuisa.dma_store %{{[^ ]*}}, %{{[^ ]*}} : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
// CHECK-NOT: npuisa.dma_store_async
// CHECK-NOT: npuisa.await
func.func @store_async_with_immediate_await(
    %src: memref<16x16xf32, #npu.scratchpad>,
    %dst: memref<16x16xf32, #npu.dram>) {
  %t = npuisa.dma_store_async %src, %dst
     : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  npuisa.await %t
  return
}

// -----

// The destination is a view over a flat scratchpad buffer, which is the shape
// the allocator produces. The fold is about adjacency and does not care what
// the operands are derived from, and this pins that.
// CHECK-LABEL: func.func @immediate_await_over_a_view
// CHECK: npuisa.dma_load %
// CHECK-NOT: npuisa.dma_load_async
// CHECK-NOT: npuisa.await
func.func @immediate_await_over_a_view(%src: memref<4x4xf32, #npu.dram>) {
  %c0 = arith.constant 0 : index
  %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
  %dst = memref.view %flat[%c0][]
       : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.await %t
  memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
  return
}

// -----

// =============================================================================
// The fold does not fire: something real sits between the two.
// =============================================================================

// A compute instruction on a disjoint buffer between the transfer and its await
// is exactly the overlap the asynchronous form exists to express. Folding this
// one would serialise a transfer somebody deliberately overlapped with work, so
// the asynchronous form survives canonicalization unchanged.
// CHECK-LABEL: func.func @await_is_not_the_next_operation
// CHECK: npuisa.dma_load_async
// CHECK: npuisa.relu
// CHECK: npuisa.await
func.func @await_is_not_the_next_operation(
    %src: memref<16x16xf32, #npu.dram>,
    %dst: memref<16x16xf32, #npu.scratchpad>,
    %other: memref<4x4xf32, #npu.scratchpad>) {
  %t = npuisa.dma_load_async %src, %dst
     : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  npuisa.relu ins(%other : memref<4x4xf32, #npu.scratchpad>)
              outs(%other : memref<4x4xf32, #npu.scratchpad>)
  npuisa.await %t
  return
}

// -----

// The same for the store direction, so that neither pattern is over eager.
// CHECK-LABEL: func.func @store_await_is_not_the_next_operation
// CHECK: npuisa.dma_store_async
// CHECK: npuisa.relu
// CHECK: npuisa.await
func.func @store_await_is_not_the_next_operation(
    %src: memref<16x16xf32, #npu.scratchpad>,
    %dst: memref<16x16xf32, #npu.dram>,
    %other: memref<4x4xf32, #npu.scratchpad>) {
  %t = npuisa.dma_store_async %src, %dst
     : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  npuisa.relu ins(%other : memref<4x4xf32, #npu.scratchpad>)
              outs(%other : memref<4x4xf32, #npu.scratchpad>)
  npuisa.await %t
  return
}

// -----

// Two transfers in flight at once, each awaited after the other's producer.
// Neither await is the next operation after its own producer, so neither folds,
// and this is the case a pattern that searched the block for any await rather
// than checking the next operation would get wrong in both directions.
// CHECK-LABEL: func.func @two_transfers_in_flight
// CHECK: npuisa.dma_load_async
// CHECK: npuisa.dma_load_async
// CHECK: npuisa.await
// CHECK: npuisa.await
func.func @two_transfers_in_flight(%src1: memref<4x4xf32, #npu.dram>,
                                   %dst1: memref<4x4xf32, #npu.scratchpad>,
                                   %src2: memref<4x4xf32, #npu.dram>,
                                   %dst2: memref<4x4xf32, #npu.scratchpad>) {
  %t1 = npuisa.dma_load_async %src1, %dst1
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  %t2 = npuisa.dma_load_async %src2, %dst2
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.await %t1
  npuisa.await %t2
  return
}

// -----

// =============================================================================
// The compute instructions are not deleted, which is the other half of the
// memory effects being declared.
// =============================================================================

// Every compute instruction here has no results, so if it reported itself free
// of effects the canonicalizer would be entitled to delete it as dead and this
// function would canonicalize to a bare `return`. It does not, and that is the
// whole reason MemoryEffectOpInterface is on these operations rather than
// `Pure`. NPUInterfaceTests asserts the same thing directly on the interface;
// this asserts it at the level a user would notice, which is the output.
// CHECK-LABEL: func.func @compute_instructions_survive_canonicalization
// CHECK: npuisa.relu
// CHECK: npuisa.add
// CHECK: npuisa.matmul
// CHECK: npuisa.dma_store
func.func @compute_instructions_survive_canonicalization(
    %x: memref<4x4xf32, #npu.scratchpad>,
    %y: memref<4x4xf32, #npu.scratchpad>,
    %out: memref<4x4xf32, #npu.dram>) {
  npuisa.relu ins(%x : memref<4x4xf32, #npu.scratchpad>)
              outs(%y : memref<4x4xf32, #npu.scratchpad>)
  npuisa.add ins(%x, %y : memref<4x4xf32, #npu.scratchpad>,
                          memref<4x4xf32, #npu.scratchpad>)
             outs(%y : memref<4x4xf32, #npu.scratchpad>)
  npuisa.matmul ins(%x, %y : memref<4x4xf32, #npu.scratchpad>,
                             memref<4x4xf32, #npu.scratchpad>)
                outs(%y : memref<4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %y, %out
    : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
  return
}

// -----

// An unallocated scratchpad buffer with nothing but a dealloc. The memref
// dialect's own canonicalizer removes an alloc whose only use is its dealloc,
// which is fine and is not this dialect's business; what matters is that it does
// not take a transfer with it. The load stays because it writes the buffer.
// CHECK-LABEL: func.func @a_transfer_is_not_dead
// CHECK: npuisa.dma_load
func.func @a_transfer_is_not_dead(%src: memref<4x4xf32, #npu.dram>,
                                  %dst: memref<4x4xf32, #npu.scratchpad>) {
  npuisa.dma_load %src, %dst
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  return
}
