// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-calibrate`, Section 12, quantized mode only and never in an `-O` level.
//
// One function drives every case, because what differs between them is the
// profile rather than the program. The profiles are beside this file in
// `Inputs/`, written the way the observer writes one.

// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-profile.json \
// RUN:   | FileCheck %s
// RUN: not npu-opt %s --npu-calibrate 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOPROFILE
// RUN: not npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-profile.json calib-method=kl" \
// RUN:   2>&1 | FileCheck %s --check-prefix=BADMETHOD
// RUN: not npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-profile.json requant-mode=double" \
// RUN:   2>&1 | FileCheck %s --check-prefix=BADMODE
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-other-model.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=UNCOVERED
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-partial.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-profile.json weight-granularity=per-tensor" \
// RUN:   | FileCheck %s --check-prefix=PERTENSOR
// RUN: npu-opt %s \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-degenerate-channel.json \
// RUN:   | FileCheck %s --check-prefix=DEGENERATE
// RUN: npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-degenerate-channel.json weight-granularity=per-tensor" \
// RUN:   | FileCheck %s --check-prefix=PERTENSOR
// RUN: not npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-profile.json weight-granularity=per-layer" \
// RUN:   2>&1 | FileCheck %s --check-prefix=BADGRANULARITY
// RUN: not npu-opt %s \
// RUN:   --npu-calibrate="profile=%S/Inputs/calibrate-no-maxima.json weight-granularity=per-tensor" \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOMAXIMA

// -----------------------------------------------------------------------------
// Positive: the activations are wrapped and the graph stays f32.
//
// The input's pair comes from the profile and so does the result's, and they
// differ, which is what says the pass read two entries rather than one twice.
// The convolution itself is untouched: nothing here emits an integer
// instruction, because the contraction in the lowering is what does that.
//
// The weights are deliberately not wrapped. Section 14 makes weight scales per
// output channel and `npu.quantize` carries a single scale, so this level can
// express the activation half exactly and the weight half not at all; the per
// channel scales travel in the profile to the instruction's fourth operand.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @calibrated
// CHECK: %[[Q:.*]] = npu.quantize %arg0 {scale = 2.500000e-02 : f32, zero_point = -3 : i32} : tensor<1x2x4x4xf32> to tensor<1x2x4x4xi8>
// CHECK: %[[D:.*]] = npu.dequantize %[[Q]] {scale = 2.500000e-02 : f32, zero_point = -3 : i32} : tensor<1x2x4x4xi8> to tensor<1x2x4x4xf32>
// CHECK: %[[C:.*]] = npu.conv2d ins(%[[D]], %{{.*}} : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
// The per output channel weight scales come straight from the profile's
// weights section, because the QDQ form has nowhere to put them: npu.quantize
// carries a single scale, so this level says the activation half exactly and
// the weight half not at all.
// CHECK-SAME: weight_scales = array<f32: 0.00999999977, 2.000000e-02>
// CHECK: %[[QY:.*]] = npu.quantize %[[C]] {scale = 1.600000e-02 : f32, zero_point = -128 : i32} : tensor<1x2x4x4xf32> to tensor<1x2x4x4xi8>
// CHECK: %[[DY:.*]] = npu.dequantize %[[QY]] {scale = 1.600000e-02 : f32, zero_point = -128 : i32} : tensor<1x2x4x4xi8> to tensor<1x2x4x4xf32>
// CHECK: return %[[DY]]

// An empty profile is a failure rather than a no op, because a pass asked to
// calibrate from nothing has been misconfigured and silence would leave the
// model f32 while every report said it had been quantized.
// NOPROFILE: error: -npu-calibrate needs a calibration profile and was given none

// The two options are refused by name, with the accepted values listed.
// BADMETHOD: error: 'kl' is not a calibration method. The four are minmax, percentile, mse and entropy
// BADMODE: error: 'double' is not a requantization mode. The two are fixed and float

// A profile that names nothing here is one remark with the counts, because it
// is a real configuration: the profile of another model.
// UNCOVERED: remark: -npu-calibrate rewrote nothing in 'calibrated': of its 1 quantizable operations the profile does not name 1 and names 0 without a full set of ranges

// And one that names the operation without a range for its result is the other
// count. The two are reported separately because the fixes differ: one tensor
// was never observed, the other is missing from the file.
// PARTIAL: remark: -npu-calibrate rewrote nothing in 'calibrated': of its 1 quantizable operations the profile does not name 0 and names 1 without a full set of ranges

// -----------------------------------------------------------------------------
// The per tensor arm of Section 14's granularity ablation.
//
// Every channel takes the tensor's one symmetric scale, its largest magnitude
// over 127. The profile's channels have largest magnitudes 1.27 and 2.54, so
// that is 2.54 / 127 = 0.02, which is channel 1's own scale: the pass chooses
// it from the profile rather than dividing again.
//
// The second profile's channel 0 is all zeros, so its scale is the degenerate
// 1 and its largest magnitude is 0. Per channel it keeps the 1. Per tensor it
// is excluded, because a degenerate 1 is not a magnitude, and the answer is
// 0.02 again; taking the largest scale without asking would have answered 1
// and quantized channel 1 to almost nothing.
// -----------------------------------------------------------------------------

// PERTENSOR: weight_scales = array<f32: 2.000000e-02, 2.000000e-02>
// DEGENERATE: weight_scales = array<f32: 1.000000e+00, 2.000000e-02>

// The option is refused by name like the other two, and a profile without the
// absolute maxima cannot tell a channel of zeros from a real one, so per tensor
// is refused on it rather than guessed at.
// BADGRANULARITY: error: 'per-layer' is not a weight granularity. The two are per-channel, the default, and per-tensor
// NOMAXIMA: error: loc("Conv_0"): per-tensor weight granularity needs the profile's absolute maxima for 'w', one per channel, to tell a channel of zeros from a real one, and the profile has 0 for 2 scales

func.func @calibrated(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  return %c : tensor<1x2x4x4xf32>
}
