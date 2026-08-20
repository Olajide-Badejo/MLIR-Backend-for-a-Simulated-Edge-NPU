// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The round trip test, one function per operation. Every one of these parses,
// verifies, prints, and parses again, and the second parse is the half that
// matters: an operation with a custom assembly format that prints something it
// cannot read back is an operation nobody can round trip through a file.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// -----------------------------------------------------------------------------
// npuisa.const
// -----------------------------------------------------------------------------

// Both types are printed, the attribute's tensor type and the result's memref
// type, because neither can be written as the other and the verifier holds them
// equal.
// CHECK-LABEL: func.func @const_f32
func.func @const_f32() {
  // CHECK: npuisa.const dense<1.000000e+00> : tensor<2x2xf32> -> memref<2x2xf32, #npu.dram>
  %0 = npuisa.const dense<1.0> : tensor<2x2xf32> -> memref<2x2xf32, #npu.dram>
  return
}

// The signless i8 of a quantized buffer is two's complement signed, range -128
// to 127, and -128 is the end of that range a sign error turns into a parse
// failure rather than a wrong number.
// CHECK-LABEL: func.func @const_i8
func.func @const_i8() {
  // CHECK: npuisa.const dense<[-128, -1, 0, 127]> : tensor<4xi8> -> memref<4xi8, #npu.dram>
  %0 = npuisa.const dense<[-128, -1, 0, 127]> : tensor<4xi8> -> memref<4xi8, #npu.dram>
  return
}

// -----------------------------------------------------------------------------
// The synchronous DMA operations.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @dma_load
func.func @dma_load(%src: memref<16x16xf32, #npu.dram>,
                    %dst: memref<16x16xf32, #npu.scratchpad>) {
  // CHECK: npuisa.dma_load %{{.*}}, %{{.*}} : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  npuisa.dma_load %src, %dst
    : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  return
}

// CHECK-LABEL: func.func @dma_store
func.func @dma_store(%src: memref<16x16xf32, #npu.scratchpad>,
                     %dst: memref<16x16xf32, #npu.dram>) {
  // CHECK: npuisa.dma_store %{{.*}}, %{{.*}} : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  npuisa.dma_store %src, %dst
    : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  return
}

// -----------------------------------------------------------------------------
// The asynchronous DMA operations and the await.
// -----------------------------------------------------------------------------

// The token prints as a plain SSA value of type !npuisa.token. The operation
// between the two is a compute instruction on an unrelated block argument
// buffer, which is what the overlap rule is there to permit: two distinct
// function arguments are two distinct allocations, so they cannot alias.
// CHECK-LABEL: func.func @dma_load_async
func.func @dma_load_async(%src: memref<16x16xf32, #npu.dram>,
                          %dst: memref<16x16xf32, #npu.scratchpad>,
                          %other: memref<4x4xf32, #npu.scratchpad>) {
  // CHECK: %[[T:.*]] = npuisa.dma_load_async %{{.*}}, %{{.*}} : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  %t = npuisa.dma_load_async %src, %dst
     : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
  // CHECK: npuisa.relu
  npuisa.relu ins(%other : memref<4x4xf32, #npu.scratchpad>)
              outs(%other : memref<4x4xf32, #npu.scratchpad>)
  // CHECK: npuisa.await %[[T]]
  npuisa.await %t
  return
}

// CHECK-LABEL: func.func @dma_store_async
func.func @dma_store_async(%src: memref<16x16xf32, #npu.scratchpad>,
                           %dst: memref<16x16xf32, #npu.dram>,
                           %other: memref<4x4xf32, #npu.scratchpad>) {
  // CHECK: %[[T:.*]] = npuisa.dma_store_async %{{.*}}, %{{.*}} : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  %t = npuisa.dma_store_async %src, %dst
     : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
  // CHECK: npuisa.relu
  npuisa.relu ins(%other : memref<4x4xf32, #npu.scratchpad>)
              outs(%other : memref<4x4xf32, #npu.scratchpad>)
  // CHECK: npuisa.await %[[T]]
  npuisa.await %t
  return
}

// The token type spelled in a function signature, so that the type itself round
// trips and not only the operations that produce one.
// CHECK-LABEL: func.func private @token_in_a_signature
// CHECK-SAME: (!npuisa.token)
func.func private @token_in_a_signature(!npuisa.token)

// -----------------------------------------------------------------------------
// npuisa.matmul
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: memref<4x16xf32, #npu.scratchpad>,
                  %b: memref<16x10xf32, #npu.scratchpad>,
                  %d: memref<4x10xf32, #npu.scratchpad>) {
  // CHECK: npuisa.matmul ins(%{{.*}}, %{{.*}} : memref<4x16xf32, #npu.scratchpad>, memref<16x10xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<4x10xf32, #npu.scratchpad>)
  npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.scratchpad>,
                             memref<16x10xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// CHECK-LABEL: func.func @matmul_with_bias
func.func @matmul_with_bias(%a: memref<4x16xf32, #npu.scratchpad>,
                            %b: memref<16x10xf32, #npu.scratchpad>,
                            %c: memref<10xf32, #npu.scratchpad>,
                            %d: memref<4x10xf32, #npu.scratchpad>) {
  // The bias is the third `ins` operand and the destination is still the single
  // `outs`, so the optional operand did not change the partition.
  // CHECK: npuisa.matmul ins(%{{[^,]*}}, %{{[^,]*}}, %{{[^ ]*}} : memref<4x16xf32, #npu.scratchpad>, memref<16x10xf32, #npu.scratchpad>, memref<10xf32, #npu.scratchpad>) outs(%{{[^ ]*}} : memref<4x10xf32, #npu.scratchpad>)
  npuisa.matmul ins(%a, %b, %c : memref<4x16xf32, #npu.scratchpad>,
                                 memref<16x10xf32, #npu.scratchpad>,
                                 memref<10xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// -----------------------------------------------------------------------------
// npuisa.conv2d
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @conv2d
func.func @conv2d(%x: memref<2x3x8x8xf32, #npu.scratchpad>,
                  %w: memref<8x3x3x3xf32, #npu.scratchpad>,
                  %d: memref<2x8x8x8xf32, #npu.scratchpad>) {
  // CHECK: npuisa.conv2d ins(%{{.*}}, %{{.*}} : memref<2x3x8x8xf32, #npu.scratchpad>, memref<8x3x3x3xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<2x8x8x8xf32, #npu.scratchpad>)
  npuisa.conv2d ins(%x, %w : memref<2x3x8x8xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<2x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// A grouped convolution, where the filter's second extent is the input channel
// count divided by the group rather than the input channel count.
// CHECK-LABEL: func.func @conv2d_grouped
func.func @conv2d_grouped(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
                          %w: memref<8x2x3x3xf32, #npu.scratchpad>,
                          %b: memref<8xf32, #npu.scratchpad>,
                          %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
  // CHECK: npuisa.conv2d ins(%{{.*}}, %{{.*}}, %{{.*}} :
  npuisa.conv2d ins(%x, %w, %b : memref<1x8x8x8xf32, #npu.scratchpad>,
                                 memref<8x2x3x3xf32, #npu.scratchpad>,
                                 memref<8xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 4 : i64}
  return
}

// Asymmetric padding, which is the case that tells the ONNX pad order apart
// from a per axis one. `pads` is padTop 0, padLeft 1, padBottom 2, padRight 0,
// so height takes entries 0 and 2 and width takes entries 1 and 3:
//
//   height = 4 + 0 + 2 - 3 + 1 = 4
//   width  = 8 + 1 + 0 - 3 + 1 = 7
//
// Read as per axis pairs instead, height would take 0 and 1 and width would
// take 2 and 0, giving 3 by 8 and rejecting this operation. That is defect
// D-0019, and every pad in every other case in this file is symmetric, which is
// exactly why it survived: under a symmetric pad the two orders agree.
// CHECK-LABEL: func.func @conv2d_asymmetric_pads
func.func @conv2d_asymmetric_pads(%x: memref<1x1x4x8xf32, #npu.scratchpad>,
                                  %w: memref<1x1x3x3xf32, #npu.scratchpad>,
                                  %d: memref<1x1x4x7xf32, #npu.scratchpad>) {
  // CHECK: npuisa.conv2d
  // CHECK-SAME: outs(%{{.*}} : memref<1x1x4x7xf32, #npu.scratchpad>)
  npuisa.conv2d ins(%x, %w : memref<1x1x4x8xf32, #npu.scratchpad>,
                             memref<1x1x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x1x4x7xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 0, 1, 2, 0>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// The same asymmetry on a pooling operation, because the two share the verifier
// body and a fix to one that missed the other would be invisible otherwise.
// CHECK-LABEL: func.func @pool_max_asymmetric_pads
func.func @pool_max_asymmetric_pads(%x: memref<1x1x4x8xf32, #npu.scratchpad>,
                                    %d: memref<1x1x4x7xf32, #npu.scratchpad>) {
  // CHECK: npuisa.pool_max
  // CHECK-SAME: outs(%{{.*}} : memref<1x1x4x7xf32, #npu.scratchpad>)
  npuisa.pool_max ins(%x : memref<1x1x4x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x1x4x7xf32, #npu.scratchpad>)
                  {kernel = array<i64: 3, 3>, strides = array<i64: 1, 1>,
                   pads = array<i64: 0, 1, 2, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  return
}

// -----------------------------------------------------------------------------
// The elementwise instructions.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @add
func.func @add(%a: memref<1x8x4x4xf32, #npu.scratchpad>,
               %b: memref<1x8x4x4xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.add ins(%{{.*}}, %{{.*}} : memref<1x8x4x4xf32, #npu.scratchpad>, memref<1x8x4x4xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.add ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                          memref<1x8x4x4xf32, #npu.scratchpad>)
             outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
  return
}

// CHECK-LABEL: func.func @mul
func.func @mul(%a: memref<1x8x4x4xf32, #npu.scratchpad>,
               %b: memref<1x8x4x4xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.mul ins(%{{.*}}, %{{.*}} :
  npuisa.mul ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                          memref<1x8x4x4xf32, #npu.scratchpad>)
             outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
  return
}

// CHECK-LABEL: func.func @relu
func.func @relu(%x: memref<1x8x4x4xf32, #npu.scratchpad>,
                %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.relu ins(%{{[^ ]*}} : memref<1x8x4x4xf32, #npu.scratchpad>) outs(%{{[^ ]*}} : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
  return
}

// An in place relu, input and destination the same buffer. It is a legal
// program and a useful one: it is what the allocator produces when it reuses a
// dead interval, and nothing in the dialect forbids it.
// CHECK-LABEL: func.func @relu_in_place
func.func @relu_in_place(%x: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // The capture is `[^ ]*` rather than `.*` on purpose. A greedy `.*` matches to
  // the end of the printed line and then the second half of the check has
  // nothing left to match against, which is how the same value appearing twice
  // stops being what this test asserts.
  // CHECK: npuisa.relu ins(%[[X:[^ ]*]] : memref<1x8x4x4xf32, #npu.scratchpad>) outs(%[[X]] : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
  return
}

// -----------------------------------------------------------------------------
// The pooling instructions.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @pool_max
func.func @pool_max(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
                    %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.pool_max ins(%{{.*}} : memref<1x8x8x8xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.pool_max ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  return
}

// ceil_mode = 1, with the arithmetic written out. Input 7, kernel 2, stride 2,
// no padding, no dilation: the effective kernel is 2, the numerator is
// 7 + 0 + 0 - 2 = 5, and ceil(5 / 2) + 1 = 3 + 1 = 4. There is no right padding
// here, so the rule that drops a window starting inside it cannot fire, and the
// extent stays 4 where ceil_mode = 0 would have given floor(5 / 2) + 1 = 3.
// CHECK-LABEL: func.func @pool_max_ceil_mode
func.func @pool_max_ceil_mode(%x: memref<1x8x7x7xf32, #npu.scratchpad>,
                              %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.pool_max
  npuisa.pool_max ins(%x : memref<1x8x7x7xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
  return
}

// CHECK-LABEL: func.func @pool_avg
func.func @pool_avg(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
                    %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // CHECK: npuisa.pool_avg ins(%{{.*}} : memref<1x8x8x8xf32, #npu.scratchpad>)
  npuisa.pool_avg ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  return
}

// -----------------------------------------------------------------------------
// The shape instructions.
// -----------------------------------------------------------------------------

// This one has a destination where npu.reshape has none, because at the
// instruction level there is no retyping available: a buffer has an address and
// a shape, and the same bytes under different extents means a copy.
// CHECK-LABEL: func.func @reshape
func.func @reshape(%x: memref<4x8x2x2xf32, #npu.scratchpad>,
                   %d: memref<4x32xf32, #npu.scratchpad>) {
  // CHECK: npuisa.reshape ins(%{{.*}} : memref<4x8x2x2xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<4x32xf32, #npu.scratchpad>)
  npuisa.reshape ins(%x : memref<4x8x2x2xf32, #npu.scratchpad>)
                 outs(%d : memref<4x32xf32, #npu.scratchpad>)
  return
}

// CHECK-LABEL: func.func @transpose
func.func @transpose(%x: memref<1x3x8x8xf32, #npu.scratchpad>,
                     %d: memref<1x8x8x3xf32, #npu.scratchpad>) {
  // CHECK: npuisa.transpose ins(%{{.*}} : memref<1x3x8x8xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<1x8x8x3xf32, #npu.scratchpad>)
  npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                   outs(%d : memref<1x8x8x3xf32, #npu.scratchpad>)
                   {permutation = array<i64: 0, 2, 3, 1>}
  return
}

// CHECK-LABEL: func.func @concat
func.func @concat(%a: memref<1x4x8x8xf32, #npu.scratchpad>,
                  %b: memref<1x6x8x8xf32, #npu.scratchpad>,
                  %d: memref<1x10x8x8xf32, #npu.scratchpad>) {
  // CHECK: npuisa.concat ins(%{{.*}}, %{{.*}} : memref<1x4x8x8xf32, #npu.scratchpad>, memref<1x6x8x8xf32, #npu.scratchpad>)
  // CHECK-SAME: outs(%{{.*}} : memref<1x10x8x8xf32, #npu.scratchpad>)
  npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                             memref<1x6x8x8xf32, #npu.scratchpad>)
                outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                {axis = 1 : i64}
  return
}

// The i8 element type on a shape operation, which is the one place the
// quantized buffers of a later phase already have to round trip.
// CHECK-LABEL: func.func @concat_i8
func.func @concat_i8(%a: memref<2x4xi8, #npu.scratchpad>,
                     %b: memref<2x4xi8, #npu.scratchpad>,
                     %d: memref<4x4xi8, #npu.scratchpad>) {
  // CHECK: npuisa.concat ins(%{{.*}}, %{{.*}} : memref<2x4xi8, #npu.scratchpad>, memref<2x4xi8, #npu.scratchpad>)
  npuisa.concat ins(%a, %b : memref<2x4xi8, #npu.scratchpad>,
                             memref<2x4xi8, #npu.scratchpad>)
                outs(%d : memref<4x4xi8, #npu.scratchpad>)
                {axis = 0 : i64}
  return
}

// -----------------------------------------------------------------------------
// The function level attributes.
// -----------------------------------------------------------------------------

// npuisa.scratchpad_bytes is what the function uses and npuisa.scratchpad_budget
// is what it was compiled against. The encoder writes both into the binary and
// the simulator reads them back, so they round trip here before either exists.
// CHECK-LABEL: func.func @carries_the_scratchpad_attributes
// CHECK-SAME: npuisa.scratchpad_budget = 65536 : i64
// CHECK-SAME: npuisa.scratchpad_bytes = 4096 : i64
func.func @carries_the_scratchpad_attributes()
    attributes {npuisa.scratchpad_bytes = 4096 : i64,
                npuisa.scratchpad_budget = 65536 : i64} {
  return
}
