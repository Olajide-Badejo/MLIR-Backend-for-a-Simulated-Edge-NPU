// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// **Section 8's scoped DMA boundary invariant, asserted at exactly the point in
// the pipeline where it holds.**
//
// Immediately after lowering and before allocation, DMA appears only at the
// boundaries between the two memories: one `npuisa.dma_load` per DRAM value
// entering the scratchpad, one `npuisa.dma_store` per returned value, and
// nothing else. Stated without that scope the invariant is simply false,
// because two later passes legitimately add DMA. There are exactly three
// permitted producers, named so that an unexplained fourth is recognisable as a
// defect rather than debated as a style question:
//
//   1. the lowering itself, which is what this file checks;
//   2. spilling, which adds a store after a spilled definition and a load
//      before each later use;
//   3. tiling, which adds one load and one store per tile it split.
//
// The run line is what makes this file mean anything: it runs the lowering and
// nothing after it. Adding an allocation or a double buffering pass to that
// pipeline would break these checks correctly, and the fix would be to move
// them rather than to loosen them.
//
// This file was deferred from P2 on purpose, and docs/ARCHITECTURE.md says why:
// written then it could only have hand written IR already in the right shape,
// which asserts that its author can produce a correct example rather than that
// the pipeline does. Every module below is `npu` dialect input, so what is
// checked is what the lowering produced.

// RUN: npu-opt %s --npu-lower-to-npuisa | FileCheck %s

// =============================================================================
// One argument, one constant, one result: two loads in, one store out, and
// nothing in between.
// =============================================================================

// CHECK-LABEL: func.func @one_load_per_dram_value(
// CHECK-SAME:      %[[X:[^:]*]]: memref<1x3x8x8xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT:[^:]*]]: memref<1x8x8x8xf32, #npu.dram>)
//
// The argument enters the scratchpad exactly once.
// CHECK:         %[[XSP:[a-z_0-9]+]] = memref.alloc() : memref<1x3x8x8xf32, #npu.scratchpad>
// CHECK-NEXT:    npuisa.dma_load %[[X]], %[[XSP]]
//
// The weight is a DRAM buffer and enters the scratchpad exactly once.
// CHECK:         %[[W:[a-z_0-9]+]] = npuisa.const
// CHECK:         %[[WSP:[a-z_0-9]+]] = memref.alloc() : memref<8x3x3x3xf32, #npu.scratchpad>
// CHECK-NEXT:    npuisa.dma_load %[[W]], %[[WSP]]
//
// Nothing between the last load and the computation, and nothing between the
// computation and the store.
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.conv2d
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.dma_store %{{[a-z_0-9]+}}, %[[OUT]]
// CHECK-NOT:     npuisa.dma
// CHECK:         return
func.func @one_load_per_dram_value(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %d = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// =============================================================================
// A chain of four instructions with one input and one output. The intermediates
// stay in the scratchpad: one load at the head, one store at the tail, and the
// three values in the middle never touch DRAM.
//
// This is the case the whole memory model exists for. An intermediate written
// out and read back would cost two DRAM accesses per element, and a DRAM access
// costs orders of magnitude more energy than a scratchpad one, so a regression
// here would move the energy numbers the evaluation reports.
// =============================================================================

// CHECK-LABEL: func.func @a_chain_keeps_its_intermediates_on_chip(
// CHECK:         npuisa.dma_load
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.relu
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.pool_max
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.reshape
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.transpose
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.dma_store
// CHECK-NOT:     npuisa.dma
// CHECK:         return
func.func @a_chain_keeps_its_intermediates_on_chip(%x: tensor<1x4x8x8xf32>)
    -> tensor<16x4xf32> {
  %d0 = tensor.empty() : tensor<1x4x8x8xf32>
  %a = npu.relu ins(%x : tensor<1x4x8x8xf32>) outs(%d0 : tensor<1x4x8x8xf32>)
       -> tensor<1x4x8x8xf32>
  %d1 = tensor.empty() : tensor<1x4x4x4xf32>
  %p = npu.max_pool2d ins(%a : tensor<1x4x8x8xf32>)
                      outs(%d1 : tensor<1x4x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x4x4x4xf32>
  %s = npu.reshape %p : tensor<1x4x4x4xf32> to tensor<4x16xf32>
  %d2 = tensor.empty() : tensor<16x4xf32>
  %r = npu.transpose ins(%s : tensor<4x16xf32>) outs(%d2 : tensor<16x4xf32>)
                     {permutation = array<i64: 1, 0>} -> tensor<16x4xf32>
  return %r : tensor<16x4xf32>
}

// =============================================================================
// **No DMA between a convolution and its fused activation**, which is the P4
// gate's own wording. The fused region is flattened and its intermediate is an
// ordinary scratchpad buffer, so the property holds for the same reason it
// holds for the unfused chain above rather than for a special reason.
// =============================================================================

// CHECK-LABEL: func.func @nothing_between_a_convolution_and_its_activation(
// CHECK-COUNT-2: npuisa.dma_load
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.conv2d
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.relu
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.dma_store
// CHECK-NOT:     npuisa.dma
// CHECK:         return
func.func @nothing_between_a_convolution_and_its_activation(
    %x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %d0 = tensor.empty() : tensor<1x8x8x8xf32>
  %d1 = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.fused_op ins(%x, %w, %d0, %d1 : tensor<1x3x8x8xf32>,
                        tensor<8x3x3x3xf32>, tensor<1x8x8x8xf32>,
                        tensor<1x8x8x8xf32>) {
  ^bb0(%a: tensor<1x3x8x8xf32>, %f: tensor<8x3x3x3xf32>,
       %e0: tensor<1x8x8x8xf32>, %e1: tensor<1x8x8x8xf32>):
    %c = npu.conv2d ins(%a, %f : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                    outs(%e0 : tensor<1x8x8x8xf32>)
                    {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                     dilations = array<i64: 1, 1>, group = 1 : i64}
         -> tensor<1x8x8x8xf32>
    %v = npu.relu ins(%c : tensor<1x8x8x8xf32>) outs(%e1 : tensor<1x8x8x8xf32>)
         -> tensor<1x8x8x8xf32>
    npu.yield %v : tensor<1x8x8x8xf32>
  } -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// =============================================================================
// Two arguments and two results: two loads and two stores, one of each per
// value, and no fifth transfer.
// =============================================================================

// CHECK-LABEL: func.func @two_in_two_out_is_two_loads_and_two_stores(
// CHECK-SAME:      %[[A:[^:]*]]: memref<1x4x4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[B:[^:]*]]: memref<1x4x4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT0:[^:]*]]: memref<1x4x4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT1:[^:]*]]: memref<1x4x4x4xf32, #npu.dram>)
// CHECK:         npuisa.dma_load %[[A]]
// CHECK:         npuisa.dma_load %[[B]]
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.add
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.mul
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.dma_store %{{[a-z_0-9]+}}, %[[OUT0]]
// CHECK-NEXT:    npuisa.dma_store %{{[a-z_0-9]+}}, %[[OUT1]]
// CHECK-NOT:     npuisa.dma
// CHECK:         return
func.func @two_in_two_out_is_two_loads_and_two_stores(%a: tensor<1x4x4x4xf32>,
                                                      %b: tensor<1x4x4x4xf32>)
    -> (tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>) {
  %d0 = tensor.empty() : tensor<1x4x4x4xf32>
  %s = npu.add ins(%a, %b : tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>)
               outs(%d0 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %d1 = tensor.empty() : tensor<1x4x4x4xf32>
  %p = npu.mul ins(%a, %b : tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>)
               outs(%d1 : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %s, %p : tensor<1x4x4x4xf32>, tensor<1x4x4x4xf32>
}

// =============================================================================
// The rank 1 channel broadcast does not add a transfer. The view is a
// `memref.reinterpret_cast` over the buffer that was already loaded, so the
// carve out of ADR 0005 costs C floats of DRAM traffic rather than N by C by H
// by W of it, which is the reason that record gives for leaving the constant
// unexpanded in the first place.
// =============================================================================

// CHECK-LABEL: func.func @a_broadcast_view_adds_no_transfer(
// CHECK-COUNT-2: npuisa.dma_load
// CHECK-NOT:     npuisa.dma
// CHECK:         memref.reinterpret_cast
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.mul
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.dma_store
// CHECK-NOT:     npuisa.dma
// CHECK:         return
func.func @a_broadcast_view_adds_no_transfer(%x: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  %s = npu.constant dense<2.000000e+00> : tensor<8xf32>
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.mul ins(%x, %s : tensor<1x8x4x4xf32>, tensor<8xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}

// =============================================================================
// The asynchronous forms and the await are absent. This lowering emits the
// synchronous pair and nothing else; `-npu-double-buffer` is what introduces a
// token, it runs after this pass and before allocation, and its own tests are
// where the asynchronous form is asserted.
// =============================================================================

// CHECK-LABEL: func.func @the_lowering_emits_no_asynchronous_transfer(
// CHECK-NOT:     npuisa.dma_load_async
// CHECK-NOT:     npuisa.dma_store_async
// CHECK-NOT:     npuisa.await
// CHECK:         return
func.func @the_lowering_emits_no_asynchronous_transfer(%x: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.relu ins(%x : tensor<1x8x4x4xf32>) outs(%d : tensor<1x8x4x4xf32>)
       -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}
