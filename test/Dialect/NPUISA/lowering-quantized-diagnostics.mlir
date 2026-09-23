// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Everything the QDQ contraction refuses, refused by name.
//
// **An operation the contraction cannot contract is not refused**: missing
// weight scales, non constant weights, a result read twice, all of those stay
// in the QDQ form and lower as they did before, and
// `lowering-quantized.mlir` shows each one. What is refused is an operation
// that has the calibrator's whole shape and whose arithmetic has no integer
// form. Those are calibrations the machine cannot execute, and quietly
// compiling one in f32 would put an f32 number under an INT8 label.
//
// Every refusal is emitted from stage 2b, where the plan is made on tensors,
// before any operation has been rewritten, for the reason the validation stage
// gives in `lowering-diagnostics.mlir`.

// RUN: npu-opt %s --npu-lower-to-npuisa -split-input-file -verify-diagnostics

// A multiplier at or above one. 1.0 * 1.0 / 0.5 = 2: the output scale is
// smaller than the product of the input and weight scales, and the pair's
// shift is a right shift, so there is no way to say a gain.
func.func @a_multiplier_of_two(%qx: tensor<1x3xi8>) -> tensor<1x1xi8> {
  %w = npu.constant dense<[[1.0], [1.0], [1.0]]> : tensor<3x1xf32>
  %dx = npu.dequantize %qx {scale = 1.000000e+00 : f32, zero_point = 0 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x1xf32>
  // expected-error @+1 {{cannot be contracted into an integer instruction: output channel 0 needs a requantization multiplier of 2.000000e+00 = (1.000000e+00 * 1.000000e+00) / 5.000000e-01, which has no M0 in [2^30, 2^31) and shift in [0, 31]: a multiplier of one or more needs a negative shift}}
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x1xf32>)
                  outs(%d : tensor<1x1xf32>)
                  {weight_scales = array<f32: 1.000000e+00>}
       -> tensor<1x1xf32>
  %qy = npu.quantize %y {scale = 5.000000e-01 : f32, zero_point = 0 : i32}
        : tensor<1x1xf32> to tensor<1x1xi8>
  return %qy : tensor<1x1xi8>
}

// -----

// A multiplier below 2^-32. 1e-6 * 1e-6 / 1 is about 1e-12, which needs a
// right shift of 39, past the 31 the binary format accepts. Clamping the shift
// would rescale the channel by a different number than the one calibrated.
func.func @a_multiplier_too_small(%qx: tensor<1x3xi8>) -> tensor<1x1xi8> {
  %w = npu.constant dense<[[1.0e-06], [1.0e-06], [1.0e-06]]> : tensor<3x1xf32>
  %dx = npu.dequantize %qx {scale = 9.99999997e-07 : f32, zero_point = 0 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x1xf32>
  // expected-error @+1 {{cannot be contracted into an integer instruction: output channel 0 needs a requantization multiplier of}}
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x1xf32>)
                  outs(%d : tensor<1x1xf32>)
                  {weight_scales = array<f32: 9.99999997e-07>}
       -> tensor<1x1xf32>
  %qy = npu.quantize %y {scale = 1.000000e+00 : f32, zero_point = 0 : i32}
        : tensor<1x1xf32> to tensor<1x1xi8>
  return %qy : tensor<1x1xi8>
}

// -----

// The same refusal names the shift it would have needed, which is the number
// a reader needs to see how far out the calibration is.
func.func @the_shift_it_would_need(%qx: tensor<1x3xi8>) -> tensor<1x1xi8> {
  %w = npu.constant dense<[[1.0e-06], [1.0e-06], [1.0e-06]]> : tensor<3x1xf32>
  %dx = npu.dequantize %qx {scale = 9.99999997e-07 : f32, zero_point = 0 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x1xf32>
  // expected-error @+1 {{needs a right shift of 39, past the 31 the binary format accepts}}
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x1xf32>)
                  outs(%d : tensor<1x1xf32>)
                  {weight_scales = array<f32: 9.99999997e-07>}
       -> tensor<1x1xf32>
  %qy = npu.quantize %y {scale = 1.000000e+00 : f32, zero_point = 0 : i32}
        : tensor<1x1xf32> to tensor<1x1xi8>
  return %qy : tensor<1x1xi8>
}

// -----

// A bias that leaves int32. 1e6 / (1e-3 * 1e-3) is about 1e12, and the machine
// adds the bias to an int32 accumulator, so wrapping it would make the channel
// silently wrong everywhere. The channel it names is the second, because the
// first channel's bias is zero and fits.
func.func @a_bias_outside_int32(%qx: tensor<1x1x1x1xi8>)
    -> tensor<1x2x1x1xi8> {
  %w = npu.constant dense<[[[[1.0e-03]]], [[[1.0e-03]]]]> : tensor<2x1x1x1xf32>
  %b = npu.constant dense<[0.0, 1.0e+06]> : tensor<2xf32>
  %dx = npu.dequantize %qx {scale = 1.000000e-03 : f32, zero_point = 0 : i32}
        : tensor<1x1x1x1xi8> to tensor<1x1x1x1xf32>
  %d = tensor.empty() : tensor<1x2x1x1xf32>
  // expected-error @+1 {{cannot be contracted into an integer instruction: output channel 1 has no int32 bias, because the bias with the input zero point folded in is}}
  %y = npu.conv2d ins(%dx, %w, %b : tensor<1x1x1x1xf32>, tensor<2x1x1x1xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 1.000000e-03, 1.000000e-03>}
       -> tensor<1x2x1x1xf32>
  %qy = npu.quantize %y {scale = 1.000000e+00 : f32, zero_point = 0 : i32}
        : tensor<1x2x1x1xf32> to tensor<1x2x1x1xi8>
  return %qy : tensor<1x2x1x1xi8>
}
