// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// One case per lowering pattern, which is the 17.1 row for a lowering pattern,
// and three cases proving the pass does not fire where it should not, which is
// the 17.1 row for a pass. Every case carries a CHECK-NOT as well as a CHECK,
// because a pattern that fired unconditionally would satisfy every positive
// check in this file.
//
// Where the boundary DMA of Section 8 is asserted is test/Dialect/NPUISA/
// dma-boundaries.mlir, not here. This file pins what each operation becomes;
// that one pins how much DMA the whole function has. Keeping them apart means a
// failure says which of the two claims broke.

// RUN: npu-opt %s --npu-lower-to-npuisa | FileCheck %s

// =============================================================================
// The function boundary: arguments in DRAM, results as trailing DRAM arguments.
// =============================================================================

// Two arguments and two results become four arguments, in that order, and the
// function returns nothing. The out parameters are appended rather than
// prepended so that argument N of the lowered function is argument N of the
// model for every N a caller already knew about.
// CHECK-LABEL: func.func @two_in_two_out(
// CHECK-SAME:      %[[A:[^:]*]]: memref<4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[B:[^:]*]]: memref<4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT0:[^:]*]]: memref<4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT1:[^:]*]]: memref<4x4xf32, #npu.dram>) {
// CHECK-NOT:     tensor<
// CHECK:         npuisa.dma_store %{{[a-z_0-9]+}}, %[[OUT0]]
// CHECK:         npuisa.dma_store %{{[a-z_0-9]+}}, %[[OUT1]]
// CHECK:         return
// CHECK-NOT:     return %
func.func @two_in_two_out(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>)
    -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %d0 = tensor.empty() : tensor<4x4xf32>
  %r0 = npu.add ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
                outs(%d0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %d1 = tensor.empty() : tensor<4x4xf32>
  %r1 = npu.mul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>)
                outs(%d1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r0, %r1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// =============================================================================
// npu.constant becomes an npuisa.const in DRAM plus the one load that brings it
// on chip, and tensor.empty becomes a scratchpad allocation.
// =============================================================================

// CHECK-LABEL: func.func @constant_and_destination(
// CHECK:         %[[CST:[a-z_0-9]+]] = npuisa.const dense<2.000000e+00> : tensor<4x4xf32>
// CHECK-SAME:      -> memref<4x4xf32, #npu.dram>
// CHECK:         %[[SP:[a-z_0-9]+]] = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
// CHECK:         npuisa.dma_load %[[CST]], %[[SP]]
// CHECK:         memref.alloc() : memref<4x4xf32, #npu.scratchpad>
// CHECK-NOT:     npu.constant
// CHECK-NOT:     tensor.empty
func.func @constant_and_destination(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c = npu.constant dense<2.000000e+00> : tensor<4x4xf32>
  %d = tensor.empty() : tensor<4x4xf32>
  %r = npu.add ins(%a, %c : tensor<4x4xf32>, tensor<4x4xf32>)
               outs(%d : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// =============================================================================
// npu.conv2d, with and without a bias.
// =============================================================================

// CHECK-LABEL: func.func @conv_with_bias(
// CHECK:         npuisa.conv2d ins(
// CHECK-SAME:        memref<1x3x8x8xf32, #npu.scratchpad>,
// CHECK-SAME:        memref<8x3x3x3xf32, #npu.scratchpad>,
// CHECK-SAME:        memref<8xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%{{[a-z_0-9]+}} : memref<1x8x8x8xf32, #npu.scratchpad>)
// CHECK-SAME:      {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
// CHECK-SAME:       strides = array<i64: 1, 1>}
// CHECK-NOT:     npu.conv2d
func.func @conv_with_bias(%x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %b = npu.constant dense<0.000000e+00> : tensor<8xf32>
  %d = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.conv2d ins(%x, %w, %b : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>,
                                   tensor<8xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}

// A grouped convolution with no bias. The group attribute survives, and the
// instruction has two `ins` rather than three.
// CHECK-LABEL: func.func @conv_grouped_no_bias(
// CHECK:         npuisa.conv2d ins(%{{[^,]*}}, %{{[^,]*}} :
// CHECK-SAME:      outs(
// CHECK-SAME:      group = 4 : i64
// CHECK-NOT:     npu.conv2d
func.func @conv_grouped_no_bias(%x: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<4x1x3x3xf32>
  %d = tensor.empty() : tensor<1x4x8x8xf32>
  %r = npu.conv2d ins(%x, %w : tensor<1x4x8x8xf32>, tensor<4x1x3x3xf32>)
                  outs(%d : tensor<1x4x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 4 : i64}
       -> tensor<1x4x8x8xf32>
  return %r : tensor<1x4x8x8xf32>
}

// =============================================================================
// npu.matmul.
// =============================================================================

// CHECK-LABEL: func.func @matmul_with_bias(
// CHECK:         npuisa.matmul ins(
// CHECK-SAME:        memref<4x16xf32, #npu.scratchpad>,
// CHECK-SAME:        memref<16x10xf32, #npu.scratchpad>,
// CHECK-SAME:        memref<10xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%{{[a-z_0-9]+}} : memref<4x10xf32, #npu.scratchpad>)
// CHECK-NOT:     npu.matmul
func.func @matmul_with_bias(%x: tensor<4x16xf32>) -> tensor<4x10xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<16x10xf32>
  %b = npu.constant dense<5.000000e-01> : tensor<10xf32>
  %d = tensor.empty() : tensor<4x10xf32>
  %r = npu.matmul ins(%x, %w, %b : tensor<4x16xf32>, tensor<16x10xf32>,
                                   tensor<10xf32>)
                  outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %r : tensor<4x10xf32>
}

// =============================================================================
// The rank 1 channel broadcast of ADR 0005, on both npu.add and npu.mul.
//
// The record obliges a channel stride of one and a spatial stride of zero, and
// this is where that contract is written down as a type rather than as a flag.
// The view has the destination's extents, which is what npuisa.add requires of
// its operands, and reads the same C values over and over, which is what a per
// channel scale means.
// =============================================================================

// CHECK-LABEL: func.func @add_broadcasts_a_rank_one_addend(
// CHECK:         %[[BIAS:[a-z_0-9]+]] = memref.alloc() : memref<8xf32, #npu.scratchpad>
// CHECK:         npuisa.dma_load %{{[a-z_0-9]+}}, %[[BIAS]]
// CHECK:         %[[VIEW:[a-z_0-9]+]] = memref.reinterpret_cast %[[BIAS]] to
// CHECK-SAME:      offset: [0], sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
// CHECK-SAME:      : memref<8xf32, #npu.scratchpad> to
// CHECK-SAME:        memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
// CHECK:         npuisa.add ins(%{{[a-z_0-9]+}}, %[[VIEW]]
// CHECK-NOT:     npu.add
func.func @add_broadcasts_a_rank_one_addend(%x: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  %b = npu.constant dense<1.000000e+00> : tensor<8xf32>
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.add ins(%x, %b : tensor<1x8x4x4xf32>, tensor<8xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func.func @mul_broadcasts_a_rank_one_scale(
// CHECK:         %[[VIEW:[a-z_0-9]+]] = memref.reinterpret_cast
// CHECK-SAME:      strides: [0, 1, 0, 0]
// CHECK:         npuisa.mul ins(%{{[a-z_0-9]+}}, %[[VIEW]]
// CHECK-NOT:     npu.mul
func.func @mul_broadcasts_a_rank_one_scale(%x: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  %s = npu.constant dense<2.000000e+00> : tensor<8xf32>
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.mul ins(%x, %s : tensor<1x8x4x4xf32>, tensor<8xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}

// **The negative half of the broadcast rule.** A right hand operand that
// already has the destination's shape is not broadcast, and that is the case a
// pattern firing unconditionally would get wrong while passing both tests
// above.
// CHECK-LABEL: func.func @same_shaped_operands_are_not_broadcast(
// CHECK:         npuisa.add
// CHECK-NOT:     memref.reinterpret_cast
// CHECK-NOT:     strided<
func.func @same_shaped_operands_are_not_broadcast(%x: tensor<1x8x4x4xf32>,
                                                  %y: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.add ins(%x, %y : tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}

// =============================================================================
// npu.relu and the two pools. The pool mnemonics change across the lowering,
// because the tensor level takes its name from ONNX and the instruction level
// from the opcode.
// =============================================================================

// CHECK-LABEL: func.func @relu(
// CHECK:         npuisa.relu ins(%{{[a-z_0-9]+}} : memref<1x8x4x4xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%{{[a-z_0-9]+}} : memref<1x8x4x4xf32, #npu.scratchpad>)
// CHECK-NOT:     npu.relu
func.func @relu(%x: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %d = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.relu ins(%x : tensor<1x8x4x4xf32>) outs(%d : tensor<1x8x4x4xf32>)
       -> tensor<1x8x4x4xf32>
  return %r : tensor<1x8x4x4xf32>
}

// CHECK-LABEL: func.func @max_pool_becomes_pool_max(
// CHECK:         npuisa.pool_max
// CHECK-SAME:      kernel = array<i64: 2, 2>
// CHECK-SAME:      strides = array<i64: 2, 2>
// CHECK-NOT:     npu.max_pool2d
// CHECK-NOT:     npuisa.pool_avg
func.func @max_pool_becomes_pool_max(%x: tensor<1x4x8x8xf32>)
    -> tensor<1x4x4x4xf32> {
  %d = tensor.empty() : tensor<1x4x4x4xf32>
  %r = npu.max_pool2d ins(%x : tensor<1x4x8x8xf32>)
                      outs(%d : tensor<1x4x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x4x4x4xf32>
  return %r : tensor<1x4x4x4xf32>
}

// CHECK-LABEL: func.func @avg_pool_becomes_pool_avg(
// CHECK:         npuisa.pool_avg
// CHECK-SAME:      ceil_mode = 1 : i64
// CHECK-NOT:     npu.avg_pool2d
// CHECK-NOT:     npuisa.pool_max
func.func @avg_pool_becomes_pool_avg(%x: tensor<1x4x8x8xf32>)
    -> tensor<1x4x4x4xf32> {
  %d = tensor.empty() : tensor<1x4x4x4xf32>
  %r = npu.avg_pool2d ins(%x : tensor<1x4x8x8xf32>)
                      outs(%d : tensor<1x4x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
       -> tensor<1x4x4x4xf32>
  return %r : tensor<1x4x4x4xf32>
}

// =============================================================================
// The shape operations. npu.reshape has no destination and npuisa.reshape needs
// one, so the lowering allocates it; the other two thread theirs through.
// =============================================================================

// CHECK-LABEL: func.func @reshape_allocates_its_own_destination(
// CHECK:         %[[D:[a-z_0-9]+]] = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
// CHECK:         npuisa.reshape ins(%{{[a-z_0-9]+}} :
// CHECK-SAME:        memref<1x4x2x2xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%[[D]] : memref<4x4xf32, #npu.scratchpad>)
// CHECK-NOT:     npu.reshape
func.func @reshape_allocates_its_own_destination(%x: tensor<1x4x2x2xf32>)
    -> tensor<4x4xf32> {
  %r = npu.reshape %x : tensor<1x4x2x2xf32> to tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @transpose(
// CHECK:         npuisa.transpose ins(%{{[a-z_0-9]+}} : memref<2x4xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%{{[a-z_0-9]+}} : memref<4x2xf32, #npu.scratchpad>)
// CHECK-SAME:      {permutation = array<i64: 1, 0>}
// CHECK-NOT:     npu.transpose
func.func @transpose(%x: tensor<2x4xf32>) -> tensor<4x2xf32> {
  %d = tensor.empty() : tensor<4x2xf32>
  %r = npu.transpose ins(%x : tensor<2x4xf32>) outs(%d : tensor<4x2xf32>)
                     {permutation = array<i64: 1, 0>} -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// CHECK-LABEL: func.func @concat(
// CHECK:         npuisa.concat ins(
// CHECK-SAME:        memref<1x4x2x2xf32, #npu.scratchpad>,
// CHECK-SAME:        memref<1x6x2x2xf32, #npu.scratchpad>)
// CHECK-SAME:      outs(%{{[a-z_0-9]+}} : memref<1x10x2x2xf32, #npu.scratchpad>)
// CHECK-SAME:      {axis = 1 : i64}
// CHECK-NOT:     npu.concat
func.func @concat(%a: tensor<1x4x2x2xf32>, %b: tensor<1x6x2x2xf32>)
    -> tensor<1x10x2x2xf32> {
  %d = tensor.empty() : tensor<1x10x2x2xf32>
  %r = npu.concat ins(%a, %b : tensor<1x4x2x2xf32>, tensor<1x6x2x2xf32>)
                  outs(%d : tensor<1x10x2x2xf32>) {axis = 1 : i64}
       -> tensor<1x10x2x2xf32>
  return %r : tensor<1x10x2x2xf32>
}

// =============================================================================
// The batch norm decomposition.
//
// gamma 2 and 4, beta 1 and 0, mean 0 and 1, variance 3 and 3, epsilon 1. So
// invStd is 1/sqrt(4) = 0.5 on both channels, the multiplier is gamma * invStd
// = 1 and 2, and the addend is beta - mean * multiplier = 1 and -2. The
// constants are checked by value rather than by shape, because the arithmetic
// is the thing that could be wrong while the shapes stayed right.
// =============================================================================

// CHECK-LABEL: func.func @batch_norm_decomposes_into_a_multiply_and_an_add(
// CHECK-DAG:     npuisa.const dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK-DAG:     npuisa.const dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf32>
// CHECK:         npuisa.mul
// CHECK:         npuisa.add
// CHECK-NOT:     npu.batch_norm
// CHECK-NOT:     npuisa.batch_norm
func.func @batch_norm_decomposes_into_a_multiply_and_an_add(
    %x: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %g = npu.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
  %b = npu.constant dense<[1.000000e+00, 0.000000e+00]> : tensor<2xf32>
  %m = npu.constant dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>
  %v = npu.constant dense<[3.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d = tensor.empty() : tensor<1x2x2x2xf32>
  %r = npu.batch_norm ins(%x, %g, %b, %m, %v : tensor<1x2x2x2xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d : tensor<1x2x2x2xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}

// The four parameters the decomposition consumed are gone, and so is the DRAM
// traffic that would have brought them on chip for nothing. Three loads: the
// argument, the multiplier, the addend.
// CHECK-LABEL: func.func @an_unfolded_batch_norm_loads_two_constants_not_four(
// CHECK-COUNT-3: npuisa.dma_load
// CHECK-NOT:     npuisa.dma_load
// CHECK:         return
func.func @an_unfolded_batch_norm_loads_two_constants_not_four(
    %x: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %g = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %b = npu.constant dense<1.000000e+00> : tensor<2xf32>
  %m = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d = tensor.empty() : tensor<1x2x2x2xf32>
  %r = npu.batch_norm ins(%x, %g, %b, %m, %v : tensor<1x2x2x2xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d : tensor<1x2x2x2xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}

// =============================================================================
// npu.fused_op is flattened into its parent.
//
// The region's operations become ordinary instructions, its intermediate
// becomes an ordinary scratchpad buffer, and no DMA appears between the
// convolution and its activation. That last part is the whole point of fusion
// under this memory model and it is asserted again in dma-boundaries.mlir over
// the whole function.
// =============================================================================

// CHECK-LABEL: func.func @fused_op_is_flattened(
// CHECK:         npuisa.conv2d
// CHECK-NOT:     npuisa.dma
// CHECK:         npuisa.relu
// CHECK-NOT:     npu.fused_op
// CHECK-NOT:     npu.yield
func.func @fused_op_is_flattened(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
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
// The layout encoding of Section 5.5 becomes the memref's strided layout map.
//
// The tensor is written N, H, W, C because its extents are written in the order
// its layout names. The buffer is written N, C, H, W because a buffer below
// this level is always NCHW, and the permutation moves into the strides: over
// 1 by 8 by 8 by 3 bytes laid out NHWC, the channel stride is 1, the width
// stride is 3, the height stride is 24 and the batch stride is 192.
// =============================================================================

// CHECK-LABEL: func.func @nhwc_becomes_a_strided_buffer(
// CHECK-SAME:      memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.dram>
// CHECK:         memref.alloc()
// CHECK-SAME:      memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
// CHECK:         npuisa.relu
// CHECK-NOT:     memref<1x8x8x3xf32
func.func @nhwc_becomes_a_strided_buffer(
    %x: tensor<1x8x8x3xf32, #npu.layout<nhwc>>)
    -> tensor<1x8x8x3xf32, #npu.layout<nhwc>> {
  %d = tensor.empty() : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
  %r = npu.relu ins(%x : tensor<1x8x8x3xf32, #npu.layout<nhwc>>)
                outs(%d : tensor<1x8x8x3xf32, #npu.layout<nhwc>>)
       -> tensor<1x8x8x3xf32, #npu.layout<nhwc>>
  return %r : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
}

// An NCHW tensor keeps the identity layout rather than gaining a strided map
// that says the same thing at greater length. This is the other half of the
// claim above: if every buffer carried an explicit stride list then the NHWC
// case would prove nothing.
// CHECK-LABEL: func.func @nchw_carries_no_layout_map(
// CHECK-SAME:      memref<1x3x8x8xf32, #npu.dram>
// CHECK-NOT:     strided<
func.func @nchw_carries_no_layout_map(
    %x: tensor<1x3x8x8xf32, #npu.layout<nchw>>)
    -> tensor<1x3x8x8xf32, #npu.layout<nchw>> {
  %d = tensor.empty() : tensor<1x3x8x8xf32, #npu.layout<nchw>>
  %r = npu.relu ins(%x : tensor<1x3x8x8xf32, #npu.layout<nchw>>)
                outs(%d : tensor<1x3x8x8xf32, #npu.layout<nchw>>)
       -> tensor<1x3x8x8xf32, #npu.layout<nchw>>
  return %r : tensor<1x3x8x8xf32, #npu.layout<nchw>>
}

// =============================================================================
// The negative cases: where this pass must not fire.
// =============================================================================

// **A function that is already lowered is left exactly as it is.** No second
// signature conversion, no second load, no second store. This is the strongest
// negative case in the file, because a pass that fired on anything at all would
// rewrite this one, and it also makes the pass idempotent, which is what lets a
// pipeline run it without having to know whether something else already did.
// CHECK-LABEL: func.func @an_already_lowered_function_is_untouched(
// CHECK-SAME:      %[[X:[^:]*]]: memref<1x8x4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT:[^:]*]]: memref<1x8x4x4xf32, #npu.dram>)
// CHECK-NEXT:    %[[IN:[a-z_0-9]+]] = memref.alloc()
// CHECK-NEXT:    %[[TMP:[a-z_0-9]+]] = memref.alloc()
// CHECK-NEXT:    npuisa.dma_load %[[X]], %[[IN]]
// CHECK-NEXT:    npuisa.relu ins(%[[IN]]
// CHECK-NEXT:    npuisa.dma_store %[[TMP]], %[[OUT]]
// CHECK-NEXT:    return
func.func @an_already_lowered_function_is_untouched(
    %x: memref<1x8x4x4xf32, #npu.dram>, %out: memref<1x8x4x4xf32, #npu.dram>) {
  %in = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  %tmp = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %x, %in
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%in : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%tmp : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %tmp, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}

// **An argument nothing reads is not loaded.** A transfer whose destination no
// instruction ever reads is DRAM traffic that would move the published byte
// counts, and Section 8 makes unexplained DRAM traffic a defect rather than a
// style question. The argument still exists, because the signature is the
// model's and not the compiler's to change.
// CHECK-LABEL: func.func @an_unread_argument_is_not_loaded(
// CHECK-SAME:      %[[USED:[^:]*]]: memref<4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[UNUSED:[^:]*]]: memref<4x4xf32, #npu.dram>,
// CHECK-SAME:      %[[OUT:[^:]*]]: memref<4x4xf32, #npu.dram>)
// CHECK:         npuisa.dma_load %[[USED]]
// CHECK-NOT:     npuisa.dma_load %[[UNUSED]]
// CHECK:         return
func.func @an_unread_argument_is_not_loaded(%used: tensor<4x4xf32>,
                                            %unused: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %d = tensor.empty() : tensor<4x4xf32>
  %r = npu.relu ins(%used : tensor<4x4xf32>) outs(%d : tensor<4x4xf32>)
       -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}

// **A constant read twice is loaded once.** One npu.constant becomes one
// npuisa.const and one transfer however many instructions read it, which is the
// per DRAM value half of the Section 8 invariant rather than a per use one.
// CHECK-LABEL: func.func @a_constant_read_twice_is_loaded_once(
// CHECK:         npuisa.dma_load %arg0
// CHECK:         %[[CST:[a-z_0-9]+]] = npuisa.const
// CHECK-NOT:     npuisa.const
// CHECK:         %[[SP:[a-z_0-9]+]] = memref.alloc()
// CHECK:         npuisa.dma_load %[[CST]], %[[SP]]
// CHECK-NOT:     npuisa.dma_load
// CHECK:         npuisa.add ins(%{{[a-z_0-9]+}}, %[[SP]]
// CHECK:         npuisa.mul ins(%{{[a-z_0-9]+}}, %[[SP]]
// CHECK:         return
func.func @a_constant_read_twice_is_loaded_once(%x: tensor<4x4xf32>)
    -> tensor<4x4xf32> {
  %c = npu.constant dense<3.000000e+00> : tensor<4x4xf32>
  %d0 = tensor.empty() : tensor<4x4xf32>
  %a = npu.add ins(%x, %c : tensor<4x4xf32>, tensor<4x4xf32>)
               outs(%d0 : tensor<4x4xf32>) -> tensor<4x4xf32>
  %d1 = tensor.empty() : tensor<4x4xf32>
  %r = npu.mul ins(%a, %c : tensor<4x4xf32>, tensor<4x4xf32>)
               outs(%d1 : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
