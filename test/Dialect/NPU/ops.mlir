// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The round trip test, one function per operation. Every one of these parses,
// verifies, prints, and parses again, and the second parse is the half that
// matters: an operation with a custom assembly format that prints something it
// cannot read back is an operation nobody can round trip through a file, and
// that is not caught by parsing alone.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// -----------------------------------------------------------------------------
// npu.constant
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @constant_f32
func.func @constant_f32() -> tensor<2x2xf32> {
  // CHECK: npu.constant dense<1.000000e+00> : tensor<2x2xf32>
  %0 = npu.constant dense<1.0> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// The signless i8 of a quantized tensor is two's complement signed, range -128
// to 127, and -128 is the end of that range that a sign error turns into a
// parse failure rather than a wrong number.
// CHECK-LABEL: func.func @constant_i8
func.func @constant_i8() -> tensor<4xi8> {
  // CHECK: npu.constant dense<[-128, -1, 0, 127]> : tensor<4xi8>
  %0 = npu.constant dense<[-128, -1, 0, 127]> : tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----------------------------------------------------------------------------
// npu.conv2d
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @conv2d
func.func @conv2d(%x: tensor<2x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                  %d: tensor<2x8x8x8xf32>) -> tensor<2x8x8x8xf32> {
  // CHECK: npu.conv2d ins(%{{.*}}, %{{.*}} : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<2x8x8x8xf32>)
  // CHECK-SAME: -> tensor<2x8x8x8xf32>
  %0 = npu.conv2d ins(%x, %w : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<2x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<2x8x8x8xf32>
  return %0 : tensor<2x8x8x8xf32>
}

// CHECK-LABEL: func.func @conv2d_bias
func.func @conv2d_bias(%x: tensor<2x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                       %b: tensor<8xf32>, %d: tensor<2x8x8x8xf32>)
    -> tensor<2x8x8x8xf32> {
  // CHECK: npu.conv2d ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>, tensor<8xf32>)
  %0 = npu.conv2d ins(%x, %w, %b : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>,
                                   tensor<8xf32>)
                  outs(%d : tensor<2x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<2x8x8x8xf32>
  return %0 : tensor<2x8x8x8xf32>
}

// A depthwise convolution: group equals the input channel count, so each output
// channel sees exactly one input channel. This is the shape that a collapsed
// group dimension would misrepresent, so it is in the round trip suite as well
// as in the tiling unit tests.
// CHECK-LABEL: func.func @conv2d_depthwise
func.func @conv2d_depthwise(%x: tensor<2x8x8x8xf32>, %w: tensor<8x1x3x3xf32>,
                            %d: tensor<2x8x8x8xf32>) -> tensor<2x8x8x8xf32> {
  // CHECK: npu.conv2d
  // CHECK-SAME: group = 8
  %0 = npu.conv2d ins(%x, %w : tensor<2x8x8x8xf32>, tensor<8x1x3x3xf32>)
                  outs(%d : tensor<2x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 8 : i64}
       -> tensor<2x8x8x8xf32>
  return %0 : tensor<2x8x8x8xf32>
}

// A dilated convolution. Effective kernel is 1 * (3 - 1) * 2 + 1 = 5, so with
// pads of 2 on each side the extent is (8 + 4 - 5) / 1 + 1 = 8.
// CHECK-LABEL: func.func @conv2d_dilated
func.func @conv2d_dilated(%x: tensor<1x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                          %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  // CHECK: npu.conv2d
  // CHECK-SAME: dilations = array<i64: 2, 2>
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 2, 2, 2, 2>,
                   dilations = array<i64: 2, 2>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// The NHWC form. The extents are written in the order the layout names, so this
// is a batch of 2 three channel 8 by 8 images just like @conv2d above.
// CHECK-LABEL: func.func @conv2d_nhwc
func.func @conv2d_nhwc(%x: tensor<2x8x8x3xf32, #npu.layout<nhwc>>,
                       %w: tensor<8x3x3x3xf32>,
                       %d: tensor<2x8x8x8xf32, #npu.layout<nhwc>>)
    -> tensor<2x8x8x8xf32, #npu.layout<nhwc>> {
  // CHECK: npu.conv2d
  // CHECK-SAME: #npu.layout<nhwc>
  %0 = npu.conv2d ins(%x, %w : tensor<2x8x8x3xf32, #npu.layout<nhwc>>,
                               tensor<8x3x3x3xf32>)
                  outs(%d : tensor<2x8x8x8xf32, #npu.layout<nhwc>>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<2x8x8x8xf32, #npu.layout<nhwc>>
  return %0 : tensor<2x8x8x8xf32, #npu.layout<nhwc>>
}

// An explicit #npu.layout<nchw> and an absent encoding are the same layout, so
// this is legal beside a plain tensor.
// CHECK-LABEL: func.func @explicit_nchw
func.func @explicit_nchw(%x: tensor<2x8x4x4xf32, #npu.layout<nchw>>,
                         %d: tensor<2x8x4x4xf32, #npu.layout<nchw>>)
    -> tensor<2x8x4x4xf32, #npu.layout<nchw>> {
  // CHECK: npu.relu
  // CHECK-SAME: #npu.layout<nchw>
  %0 = npu.relu ins(%x : tensor<2x8x4x4xf32, #npu.layout<nchw>>)
                outs(%d : tensor<2x8x4x4xf32, #npu.layout<nchw>>)
       -> tensor<2x8x4x4xf32, #npu.layout<nchw>>
  return %0 : tensor<2x8x4x4xf32, #npu.layout<nchw>>
}

// -----------------------------------------------------------------------------
// npu.matmul
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @matmul
func.func @matmul(%a: tensor<4x16xf32>, %b: tensor<16x10xf32>,
                  %d: tensor<4x10xf32>) -> tensor<4x10xf32> {
  // CHECK: npu.matmul ins(%{{.*}}, %{{.*}} : tensor<4x16xf32>, tensor<16x10xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<4x10xf32>) -> tensor<4x10xf32>
  %0 = npu.matmul ins(%a, %b : tensor<4x16xf32>, tensor<16x10xf32>)
                  outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// CHECK-LABEL: func.func @matmul_bias
func.func @matmul_bias(%a: tensor<4x16xf32>, %b: tensor<16x10xf32>,
                       %c: tensor<10xf32>, %d: tensor<4x10xf32>)
    -> tensor<4x10xf32> {
  // CHECK: npu.matmul ins(%{{.*}}, %{{.*}}, %{{.*}} :
  %0 = npu.matmul ins(%a, %b, %c : tensor<4x16xf32>, tensor<16x10xf32>,
                                   tensor<10xf32>)
                  outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// -----------------------------------------------------------------------------
// The elementwise operations.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @add
func.func @add(%a: tensor<2x8x4x4xf32>, %b: tensor<2x8x4x4xf32>,
               %d: tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32> {
  // CHECK: npu.add ins(%{{.*}}, %{{.*}} : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
  %0 = npu.add ins(%a, %b : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
               outs(%d : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// CHECK-LABEL: func.func @mul
func.func @mul(%a: tensor<2x8x4x4xf32>, %b: tensor<2x8x4x4xf32>,
               %d: tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32> {
  // CHECK: npu.mul ins(%{{.*}}, %{{.*}} : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
  %0 = npu.mul ins(%a, %b : tensor<2x8x4x4xf32>, tensor<2x8x4x4xf32>)
               outs(%d : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// CHECK-LABEL: func.func @relu
func.func @relu(%x: tensor<2x8x4x4xf32>, %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // CHECK: npu.relu ins(%{{.*}} : tensor<2x8x4x4xf32>) outs(%{{.*}} : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
  %0 = npu.relu ins(%x : tensor<2x8x4x4xf32>) outs(%d : tensor<2x8x4x4xf32>)
       -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// -----------------------------------------------------------------------------
// The pooling operations.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @max_pool2d
func.func @max_pool2d(%x: tensor<2x8x8x8xf32>, %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // CHECK: npu.max_pool2d ins(%{{.*}} : tensor<2x8x8x8xf32>)
  // CHECK-SAME: outs(%{{.*}} : tensor<2x8x4x4xf32>)
  %0 = npu.max_pool2d ins(%x : tensor<2x8x8x8xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// CHECK-LABEL: func.func @avg_pool2d
func.func @avg_pool2d(%x: tensor<2x8x8x8xf32>, %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // CHECK: npu.avg_pool2d ins(%{{.*}} : tensor<2x8x8x8xf32>)
  %0 = npu.avg_pool2d ins(%x : tensor<2x8x8x8xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// A dilated average pool. AveragePool gained dilations at opset 19 and the
// verifier carries the term whatever the frontend decides to accept, so the
// round trip suite carries a case that uses it.
//
//   effectiveKernel = 2 * (2 - 1) + 1 = 3
//   floor((8 + 0 + 0 - 3) / 1) + 1    = 6
//
// The dilation is the whole difference: with dilations of 1 the same kernel and
// stride would give 7, so a verifier that dropped the term would compute 7 and
// refuse this shape.
// CHECK-LABEL: func.func @avg_pool2d_dilated
func.func @avg_pool2d_dilated(%x: tensor<1x8x8x8xf32>, %d: tensor<1x8x6x6xf32>)
    -> tensor<1x8x6x6xf32> {
  // CHECK: npu.avg_pool2d
  // CHECK-SAME: dilations = array<i64: 2, 2>
  %0 = npu.avg_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x6x6xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 1, 1>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 2, 2>, ceil_mode = 0 : i64}
       -> tensor<1x8x6x6xf32>
  return %0 : tensor<1x8x6x6xf32>
}

// The ceil_mode = 1 case whose last window starts in the right padding, which
// is the opset 19 rule that changes the shape and is routinely missed.
//
// The arithmetic, written out because a test that only asserts the answer does
// not say what it is asserting:
//
//   inputExtent 6, kernel 2, stride 3, padBegin 0, padEnd 1, dilation 1
//   effectiveKernel = 1 * (2 - 1) + 1                     = 2
//   numerator       = 6 + 0 + 1 - 2                       = 5
//   ceil(5 / 3) + 1 = 2 + 1                               = 3
//
// The three windows would start at input coordinates 0, 3 and 6. The real input
// ends at 6, so the third window's first element is at index 6, which is inside
// the right padded region. ONNX drops that window:
//
//   lastWindowStart = (3 - 1) * 3 = 6, and 6 >= inputExtent + padBegin = 6
//   extent = 3 - 1                                        = 2
//
// So the result is 2 by 2, not 3 by 3. Under ceil_mode = 0 the same parameters
// give floor(5 / 3) + 1 = 2 as well, which is why the drop rule is invisible
// unless it is tested with a case that the ceiling and the floor disagree on;
// the invalid suite carries the case that proves the verifier refuses 3.
// CHECK-LABEL: func.func @max_pool2d_ceil_mode_drops_right_padded_window
func.func @max_pool2d_ceil_mode_drops_right_padded_window(
    %x: tensor<1x8x6x6xf32>, %d: tensor<1x8x2x2xf32>) -> tensor<1x8x2x2xf32> {
  // CHECK: npu.max_pool2d
  // CHECK-SAME: ceil_mode = 1
  // CHECK-SAME: -> tensor<1x8x2x2xf32>
  %0 = npu.max_pool2d ins(%x : tensor<1x8x6x6xf32>)
                      outs(%d : tensor<1x8x2x2xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 3, 3>,
                       pads = array<i64: 0, 0, 1, 1>,
                       dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
       -> tensor<1x8x2x2xf32>
  return %0 : tensor<1x8x2x2xf32>
}

// The companion case, where the ceiling adds a window that does start inside
// the real input and is therefore kept.
//
//   inputExtent 7, kernel 2, stride 3, pads 0 and 0, dilation 1
//   numerator = 7 + 0 + 0 - 2 = 5, ceil(5 / 3) + 1 = 3
//   lastWindowStart = 2 * 3 = 6, and 6 < 7, so nothing is dropped: extent 3.
//
// Under ceil_mode = 0 the same parameters give floor(5 / 3) + 1 = 2, so this
// pair is where the ceiling and the floor genuinely differ.
// CHECK-LABEL: func.func @max_pool2d_ceil_mode_keeps_window_inside_input
func.func @max_pool2d_ceil_mode_keeps_window_inside_input(
    %x: tensor<1x8x7x7xf32>, %d: tensor<1x8x3x3xf32>) -> tensor<1x8x3x3xf32> {
  // CHECK: npu.max_pool2d
  // CHECK-SAME: -> tensor<1x8x3x3xf32>
  %0 = npu.max_pool2d ins(%x : tensor<1x8x7x7xf32>)
                      outs(%d : tensor<1x8x3x3xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 3, 3>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
       -> tensor<1x8x3x3xf32>
  return %0 : tensor<1x8x3x3xf32>
}

// The same seven by seven input under ceil_mode = 0, which is two by two.
// CHECK-LABEL: func.func @max_pool2d_floor_mode_same_parameters
func.func @max_pool2d_floor_mode_same_parameters(
    %x: tensor<1x8x7x7xf32>, %d: tensor<1x8x2x2xf32>) -> tensor<1x8x2x2xf32> {
  // CHECK: npu.max_pool2d
  // CHECK-SAME: -> tensor<1x8x2x2xf32>
  %0 = npu.max_pool2d ins(%x : tensor<1x8x7x7xf32>)
                      outs(%d : tensor<1x8x2x2xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 3, 3>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x2x2xf32>
  return %0 : tensor<1x8x2x2xf32>
}

// -----------------------------------------------------------------------------
// The shape operations.
// -----------------------------------------------------------------------------

// A flatten, which must produce (N, features) and never (1, N * features).
// CHECK-LABEL: func.func @reshape
func.func @reshape(%x: tensor<4x8x2x2xf32>) -> tensor<4x32xf32> {
  // CHECK: npu.reshape %{{.*}} : tensor<4x8x2x2xf32> to tensor<4x32xf32>
  %0 = npu.reshape %x : tensor<4x8x2x2xf32> to tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func.func @transpose
func.func @transpose(%x: tensor<2x3x8x8xf32>, %d: tensor<2x8x8x3xf32>)
    -> tensor<2x8x8x3xf32> {
  // CHECK: npu.transpose ins(%{{.*}} : tensor<2x3x8x8xf32>)
  // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
  %0 = npu.transpose ins(%x : tensor<2x3x8x8xf32>)
                     outs(%d : tensor<2x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<2x8x8x3xf32>
  return %0 : tensor<2x8x8x3xf32>
}

// CHECK-LABEL: func.func @concat
func.func @concat(%a: tensor<2x4x8x8xf32>, %b: tensor<2x6x8x8xf32>,
                  %d: tensor<2x10x8x8xf32>) -> tensor<2x10x8x8xf32> {
  // CHECK: npu.concat ins(%{{.*}}, %{{.*}} : tensor<2x4x8x8xf32>, tensor<2x6x8x8xf32>)
  // CHECK-SAME: axis = 1
  %0 = npu.concat ins(%a, %b : tensor<2x4x8x8xf32>, tensor<2x6x8x8xf32>)
                  outs(%d : tensor<2x10x8x8xf32>) {axis = 1 : i64}
       -> tensor<2x10x8x8xf32>
  return %0 : tensor<2x10x8x8xf32>
}

// Three inputs, and along the batch axis rather than the channel axis, because
// a concat verifier that only ever sees axis 1 is a verifier nobody has tested
// against the axis argument.
// CHECK-LABEL: func.func @concat_three_on_batch
func.func @concat_three_on_batch(%a: tensor<1x4x8x8xf32>,
                                 %b: tensor<2x4x8x8xf32>,
                                 %c: tensor<3x4x8x8xf32>,
                                 %d: tensor<6x4x8x8xf32>)
    -> tensor<6x4x8x8xf32> {
  // CHECK: npu.concat
  // CHECK-SAME: axis = 0
  %0 = npu.concat ins(%a, %b, %c : tensor<1x4x8x8xf32>, tensor<2x4x8x8xf32>,
                                   tensor<3x4x8x8xf32>)
                  outs(%d : tensor<6x4x8x8xf32>) {axis = 0 : i64}
       -> tensor<6x4x8x8xf32>
  return %0 : tensor<6x4x8x8xf32>
}

// -----------------------------------------------------------------------------
// npu.batch_norm
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @batch_norm
func.func @batch_norm(%x: tensor<2x8x4x4xf32>, %g: tensor<8xf32>,
                      %b: tensor<8xf32>, %m: tensor<8xf32>, %v: tensor<8xf32>,
                      %d: tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32> {
  // CHECK: npu.batch_norm ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : {{.*}}epsilon
  %0 = npu.batch_norm ins(%x, %g, %b, %m, %v : tensor<2x8x4x4xf32>,
                                               tensor<8xf32>, tensor<8xf32>,
                                               tensor<8xf32>, tensor<8xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {epsilon = 1.000000e-05 : f32} -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// -----------------------------------------------------------------------------
// npu.fused_op and npu.yield
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fused_op
func.func @fused_op(%x: tensor<2x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                    %d: tensor<2x8x8x8xf32>) -> tensor<2x8x8x8xf32> {
  // CHECK: npu.fused_op ins(%{{.*}}, %{{.*}}, %{{.*}} :
  // CHECK: ^bb0(
  // CHECK: npu.conv2d
  // CHECK: npu.relu
  // CHECK: npu.yield
  %0 = npu.fused_op ins(%x, %w, %d : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>,
                                     tensor<2x8x8x8xf32>) {
  ^bb0(%a: tensor<2x3x8x8xf32>, %f: tensor<8x3x3x3xf32>,
       %e: tensor<2x8x8x8xf32>):
    %c = npu.conv2d ins(%a, %f : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>)
                    outs(%e : tensor<2x8x8x8xf32>)
                    {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                     dilations = array<i64: 1, 1>, group = 1 : i64}
         -> tensor<2x8x8x8xf32>
    %r = npu.relu ins(%c : tensor<2x8x8x8xf32>) outs(%e : tensor<2x8x8x8xf32>)
         -> tensor<2x8x8x8xf32>
    npu.yield %r : tensor<2x8x8x8xf32>
  } -> tensor<2x8x8x8xf32>
  return %0 : tensor<2x8x8x8xf32>
}

// -----------------------------------------------------------------------------
// The memory space attributes.
// -----------------------------------------------------------------------------

// These are used as memref memory spaces at the npuisa level, which does not
// exist yet, so what is testable here is that they parse and print exactly as
// the design writes them, with no extra enum wrapper word.
// CHECK-LABEL: func.func @memory_spaces
func.func @memory_spaces(%s: memref<4x4xf32, #npu.scratchpad>,
                         %g: memref<4x4xf32, #npu.dram>) {
  // CHECK: memref<4x4xf32, #npu.scratchpad>
  // CHECK-SAME: memref<4x4xf32, #npu.dram>
  return
}
