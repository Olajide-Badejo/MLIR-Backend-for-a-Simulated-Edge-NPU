// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-calibrate` computes the weight scales from the constant each operation
// holds.
//
// Section 14's symmetric rule, one scale per output channel, `max |w_c| / 127`
// and 1 for a channel of zeros, over the filter the operation actually reads.
// That is the observer's rule, and at `-O0`, where nothing has touched a
// filter, the two agree bit for bit on every channel of the suite, which
// `test_quantized_contraction.py` holds against the committed profiles. The
// profile's weight entries are that oracle; this pass does not read them.
//
// **Why the constant and not the profile** is the last function here: at
// `-O2` the batch norm fold multiplies each output channel of the filter by
// its own factor before calibration, so the profile's entry, which describes
// the initializer, describes weights the machine no longer has. Section 14's
// argument for per channel weights is exactly the spread that fold creates.

// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-profile.json \
// RUN:   | FileCheck %s
// RUN: npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-profile.json weight-granularity=per-tensor" \
// RUN:   | FileCheck %s --check-prefix=PERTENSOR
// RUN: npu-opt %s --npu-fold-batchnorm \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-folded.json \
// RUN:   | FileCheck %s --check-prefix=FOLDED

// Two channels whose largest magnitudes are 1.27 and 2.54, one of them
// negative, so the rule takes magnitudes: 0.01 and 0.02. Per tensor, both take
// the larger.
// CHECK-LABEL: func.func @distinct
// CHECK: weight_scales = array<f32: 0.00999999977, 2.000000e-02>
// PERTENSOR-LABEL: func.func @distinct
// PERTENSOR: weight_scales = array<f32: 2.000000e-02, 2.000000e-02>
func.func @distinct(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<[[[[-1.27]], [[0.5]]], [[[2.54]], [[-1.0]]]]>
       : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  return %c : tensor<1x2x4x4xf32>
}

// A channel of zeros takes the degenerate 1, because its scale is not a
// magnitude, and per tensor it is left out of the choice rather than taken as
// the largest: taking it would quantize channel 1 to almost nothing.
// CHECK-LABEL: func.func @zero_channel
// CHECK: weight_scales = array<f32: 1.000000e+00, 2.000000e-02>
// PERTENSOR-LABEL: func.func @zero_channel
// PERTENSOR: weight_scales = array<f32: 2.000000e-02, 2.000000e-02>
func.func @zero_channel(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<[[[[0.0]], [[0.0]]], [[[2.54]], [[1.0]]]]>
       : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  return %c : tensor<1x2x4x4xf32>
}

// Every channel of zeros: the degenerate 1 everywhere, in both arms.
// CHECK-LABEL: func.func @all_zero
// CHECK: weight_scales = array<f32: 1.000000e+00, 1.000000e+00>
// PERTENSOR-LABEL: func.func @all_zero
// PERTENSOR: weight_scales = array<f32: 1.000000e+00, 1.000000e+00>
func.func @all_zero(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<0.0> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  return %c : tensor<1x2x4x4xf32>
}

// A matrix multiplication's output channel is its column, the right operand's
// fastest axis: column 0's largest magnitude is 1.27 and column 1's 2.54,
// while the rows' would have been 1.27, 2.54 and 0.5. D-0067 is what reading
// the wrong axis cost.
// CHECK-LABEL: func.func @matmul_columns
// CHECK: weight_scales = array<f32: 0.00999999977, 2.000000e-02>
func.func @matmul_columns(%a: tensor<1x3xf32>) -> tensor<1x2xf32> {
  %w = npu.constant dense<[[1.27, 0.25], [-0.5, -2.54], [0.125, 0.5]]>
       : tensor<3x2xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %y = npu.matmul ins(%a, %w : tensor<1x3xf32>, tensor<3x2xf32>)
                  outs(%d : tensor<1x2xf32>)
       -> tensor<1x2xf32> loc("Conv_0")
  return %y : tensor<1x2xf32>
}

// **After the batch norm fold, the scales are the folded filter's.** Every
// weight is 1 and the batch norm's gamma is 2 and 0.5 with unit variance and
// no epsilon, so the fold multiplies channel 0 by 2 and channel 1 by 0.5
// exactly, and the scales are 2 / 127 and 0.5 / 127. The profile's entry for
// the initializer would have said 1 / 127 for both, which would clamp channel
// 0's folded weights at the rails and waste half of channel 1's levels.
// FOLDED-LABEL: func.func @folded
// FOLDED-NOT: npu.batch_norm
// FOLDED: weight_scales = array<f32: 0.0157480314, 0.00393700786>
func.func @folded(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<1.0> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  %g = npu.constant dense<[2.0, 0.5]> : tensor<2xf32>
  %b = npu.constant dense<0.0> : tensor<2xf32>
  %m = npu.constant dense<0.0> : tensor<2xf32>
  %v = npu.constant dense<1.0> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %y = npu.batch_norm ins(%c, %g, %b, %m, %v : tensor<1x2x4x4xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d1 : tensor<1x2x4x4xf32>)
                      {epsilon = 0.0 : f32}
       -> tensor<1x2x4x4xf32> loc("BatchNormalization_1")
  return %y : tensor<1x2x4x4xf32>
}
