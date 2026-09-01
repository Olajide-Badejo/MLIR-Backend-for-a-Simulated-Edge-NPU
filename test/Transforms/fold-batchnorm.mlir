// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-fold-batchnorm`, Section 12, at -O2, before fusion.
//
// The numbers in the positive case are chosen so the arithmetic is checkable by
// hand rather than only by the pass. With variance 3 and epsilon 1 the
// denominator is 4 and the reciprocal square root is exactly 0.5, so:
//
//   scale = gamma * 0.5           = [2, 4] * 0.5     = [1, 2]
//   shift = beta - mean * scale   = [0.5, 1] - [1, 0.5] * [1, 2] = [-0.5, 0]
//   w'    = w * scale             = 2 * [1, 2]       = [2, 4] per filter
//   b'    = b * scale + shift     = [1, 2] * [1, 2] + [-0.5, 0] = [0.5, 4]
//
// Every value below is exact in binary floating point, so a CHECK on the
// printed digits is a check on the arithmetic and not on a rounding.

// RUN: npu-opt %s --npu-fold-batchnorm --canonicalize | FileCheck %s

// -----------------------------------------------------------------------------
// Positive: the batch norm disappears into the convolution's weights and bias,
// and the four parameter constants and the batch norm's destination go with it
// once canonicalize has run.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fold_into_the_convolution
// CHECK-DAG: %[[B:.*]] = npu.constant dense<[5.000000e-01, 4.000000e+00]> : tensor<2xf32>
// CHECK-DAG: %[[W:.*]] = npu.constant dense<{{\[}}{{\[}}{{\[}}[2.000000e+00]], {{\[}}[2.000000e+00]]], {{\[}}{{\[}}[4.000000e+00]], {{\[}}[4.000000e+00]]]]> : tensor<2x2x1x1xf32>
// CHECK: npu.conv2d ins(%arg0, %[[W]], %[[B]]
// CHECK-NOT: npu.batch_norm
func.func @fold_into_the_convolution(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %b = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w, %b : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>, tensor<2xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
  %be = npu.constant dense<[5.000000e-01, 1.000000e+00]> : tensor<2xf32>
  %m = npu.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  %v = npu.constant dense<[3.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %be, %m, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Positive: a convolution with no bias gains one, and it is the shift.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @a_biasless_convolution_gains_the_shift
// CHECK-DAG: %[[B:.*]] = npu.constant dense<[-5.000000e-01, 0.000000e+00]> : tensor<2xf32>
// CHECK: npu.conv2d ins(%arg0, %{{.*}}, %[[B]]
// CHECK-NOT: npu.batch_norm
func.func @a_biasless_convolution_gains_the_shift(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>
  %be = npu.constant dense<[5.000000e-01, 1.000000e+00]> : tensor<2xf32>
  %m = npu.constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  %v = npu.constant dense<[3.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %be, %m, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 1: the convolution has a second reader, so rewriting it in place
// would change what that reader sees.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_two_uses
// CHECK: npu.batch_norm
func.func @no_fold_two_uses(%x: tensor<1x2x4x4xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %z = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %z, %z, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  %d2 = tensor.empty() : tensor<1x2x4x4xf32>
  %s = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r, %s : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 2: the producer is not a convolution. Section 5.2 makes an unfolded
// batch norm legal, so this is a non match and never a diagnostic; the lowering
// decomposes it into a multiply and an add.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_producer_is_not_a_convolution
// CHECK: npu.batch_norm
func.func @no_fold_producer_is_not_a_convolution(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %a = npu.relu ins(%x : tensor<1x2x4x4xf32>) outs(%d0 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %z = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%a, %g, %z, %z, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 3: a parameter that is not a constant. The fold computes its
// multiplier at rewrite time, so a parameter it cannot read is a parameter it
// cannot fold.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_non_constant_parameter
// CHECK: npu.batch_norm
func.func @no_fold_non_constant_parameter(%x: tensor<1x2x4x4xf32>, %g: tensor<2xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %z = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %z, %z, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 4: the filter is not a constant, so there is nothing to scale at
// rewrite time.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_non_constant_filter
// CHECK: npu.batch_norm
func.func @no_fold_non_constant_filter(%x: tensor<1x2x4x4xf32>, %w: tensor<2x2x1x1xf32>) -> tensor<1x2x4x4xf32> {
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %z = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %z, %z, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 5: the variance plus the epsilon is not positive, so the reciprocal
// square root the fold takes does not exist. Declined here rather than
// diagnosed, because `-npu-lower-to-npuisa` is the layer that has to refuse it
// and it refuses with the numbers in the message.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_non_positive_variance
// CHECK: npu.batch_norm
func.func @no_fold_non_positive_variance(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %g = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %z = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<-1.000000e+00> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.batch_norm ins(%c, %g, %z, %z, %v : tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}
