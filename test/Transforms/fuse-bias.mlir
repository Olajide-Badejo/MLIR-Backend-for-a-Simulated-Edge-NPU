// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-fuse-bias`, Section 12, at -O2.
//
// The pass Section 11's broadcast carve out exists for. A rank 1 initializer of
// length C broadcasting against a rank 4 activation over the channel axis is
// left unexpanded by the importer precisely so this guard can match; an
// importer that expanded it would make this pass structurally unfireable and
// its ablation row a row of zeros.
//
// `test/Python/test_transform_passes.py` runs this pass on IR the real ONNX
// importer produced, which is the other half of the same claim: the shape here
// is the shape a model actually imports to.

// RUN: npu-opt %s --npu-fuse-bias --canonicalize | FileCheck %s

// -----------------------------------------------------------------------------
// Positive: the add disappears into the convolution's bias operand, and its
// destination goes with it once canonicalize has run. One fewer instruction and
// one fewer scratchpad buffer, which is the whole of the pass's benefit.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fuse_the_bias_add
// CHECK: %[[B:.*]] = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
// CHECK: npu.conv2d ins(%arg0, %{{.*}}, %[[B]]
// CHECK-NOT: npu.add
func.func @fuse_the_bias_add(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %b = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%c, %b : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 1, "already fused": the convolution already carries a bias, so there
// is nothing to move into and a second one is an operand the dialect does not
// have.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_already_biased
// CHECK: npu.add
func.func @no_fuse_already_biased(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %b0 = npu.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w, %b0 : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>, tensor<2xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %b1 = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%c, %b1 : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 2, "multiple uses": the convolution's result is read by something
// else as well, and that reader reads the value without the bias.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_multiple_uses
// CHECK: npu.add
func.func @no_fuse_multiple_uses(%x: tensor<1x2x4x4xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %b = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%c, %b : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %d2 = tensor.empty() : tensor<1x2x4x4xf32>
  %s = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r, %s : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 3a, "a non channel constant addend": the addend has the result's own
// shape. That is a residual add, which is a different operation with the same
// spelling, and folding it into a bias operand would be wrong rather than
// merely unhelpful.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_same_shaped_addend
// CHECK: npu.add
func.func @no_fuse_same_shaped_addend(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %b = npu.constant dense<1.000000e+00> : tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%c, %b : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 3b, the other half of the same guard: the addend is channel shaped
// and is not a constant. A bias operand is data the encoder writes into the
// binary, so a value only known at run time cannot become one.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_non_constant_addend
// CHECK: npu.add
func.func @no_fuse_non_constant_addend(%x: tensor<1x2x4x4xf32>, %b: tensor<2xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%c, %b : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 4: the producer is not a convolution at all. A relu followed by a
// channel add is a per channel shift of an activation, and there is no bias
// operand anywhere for it to become.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_producer_is_not_a_convolution
// CHECK: npu.add
func.func @no_fuse_producer_is_not_a_convolution(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %a = npu.relu ins(%x : tensor<1x2x4x4xf32>) outs(%d0 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %b = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.add ins(%a, %b : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}
