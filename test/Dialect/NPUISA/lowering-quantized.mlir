// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The QDQ contraction in `-npu-lower-to-npuisa`: Section 14's quantized
// convolution and matrix multiplication, from the calibrator's form to one
// integer instruction, and the shapes it leaves alone.
//
// **The numbers in the CHECK lines were worked out by hand** from the scales
// in each function, in exact fractions, before this file was first run, and
// the working is beside each one. `test/Python/test_quantized_contraction.py`
// runs the same two functions on the machine and checks the int8 results those
// numbers produce.
//
// The refusals are in `lowering-quantized-diagnostics.mlir`, because a
// `-verify-diagnostics` run and a FileCheck run cannot share a file.

// RUN: npu-opt %s --npu-lower-to-npuisa | FileCheck %s

// -----------------------------------------------------------------------------
// The convolution, worked by hand.
//
// Input: scale_x = 0.5, zp_x = 3, a nonzero zero point with padding on both
// sides of the one row. Weight scales 0.375 and 0.15625, two channels with
// different scales. Output: scale_y = 0.25, zp_y = 5.
//
// The weights, w / scale_c rounded half to even:
//   channel 0: 0.9375 / 0.375 = 2.5 -> 2,    -0.375 / 0.375 = -1
//   channel 1: 0.625 / 0.15625 = 4,   -0.234375 / 0.15625 = -1.5 -> -2
// Both ties go to the even neighbour, which rounding away from zero would not.
//
// The bias, round(b / (scale_x * scale_c)) - zp_x * sum of the channel's
// weights over the whole window:
//   channel 0: 1.0 / 0.1875 = 5.33 -> 5,  5 - 3 * (2 - 1) = 2
//   channel 1: f32(-0.4) / 0.078125 = -5.12 -> -5,  -5 - 3 * (4 - 2) = -11
//
// The rescale, M_c = scale_x * scale_c / scale_y, as M0 * 2^-(31 + shift):
//   channel 0: 0.1875 / 0.25 = 0.75,     M0 = 0.75 * 2^31 = 1610612736, shift 0
//   channel 1: 0.078125 / 0.25 = 0.3125, doubled to 0.625,
//              M0 = 0.625 * 2^31 = 1342177280, shift 1
// The pairs differ, so the table is the fourth operand, and the scalar fields
// carry channel 0's pair.
//
// What is **not** there matters as much: no DEQUANT of the input, no f32
// weights, no QUANT of the result. The i8 argument is read as it is.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @hand_computed_convolution(
// CHECK:       %[[X:.*]] = memref.alloc() : memref<2x1x1x2xi8, #npu.scratchpad>
// CHECK:       npuisa.dma_load %arg0, %[[X]]
// CHECK:       npuisa.const dense<{{\[\[\[\[}}2, -1]]], {{\[\[\[}}4, -2]]]]> : tensor<2x1x1x2xi8>
// CHECK:       npuisa.const dense<[2, -11]> : tensor<2xi32>
// CHECK:       npuisa.const dense<{{\[\[}}1610612736, 1342177280], [0, 1]]> : tensor<2x2xi32>
// CHECK:       npuisa.conv2d ins(%[[X]], %{{.*}}, %{{.*}}, %{{.*}} : memref<2x1x1x2xi8, #npu.scratchpad>, memref<2x1x1x2xi8, #npu.scratchpad>, memref<2xi32, #npu.scratchpad>, memref<2x2xi32, #npu.scratchpad>)
// CHECK-SAME:    outs(%{{.*}} : memref<2x2x1x3xi8, #npu.scratchpad>)
// CHECK-SAME:    output_zero_point = 5 : i32
// CHECK-SAME:    pads = array<i64: 0, 1, 0, 1>
// CHECK-SAME:    requant_multiplier = 1610612736 : i32, requant_shift = 0 : i32
// CHECK-SAME:    zero_point = 3 : i32
// CHECK:       npuisa.dma_store
// CHECK-NOT:   npuisa.dequant
// CHECK-NOT:   npuisa.quant
// CHECK-NOT:   f32
// CHECK:       return

func.func @hand_computed_convolution(%qx: tensor<2x1x1x2xi8>)
    -> tensor<2x2x1x3xi8> {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[0.625, -0.234375]]]]>
       : tensor<2x1x1x2xf32>
  %b = npu.constant dense<[1.0, -0.4]> : tensor<2xf32>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<2x1x1x2xi8> to tensor<2x1x1x2xf32>
  %d = tensor.empty() : tensor<2x2x1x3xf32>
  %y = npu.conv2d ins(%dx, %w, %b : tensor<2x1x1x2xf32>, tensor<2x1x1x2xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<2x2x1x3xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 1, 0, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 3.750000e-01, 1.562500e-01>}
       -> tensor<2x2x1x3xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<2x2x1x3xf32> to tensor<2x2x1x3xi8>
  return %qy : tensor<2x2x1x3xi8>
}

// -----------------------------------------------------------------------------
// The matrix multiplication, worked by hand.
//
// (2, 3) by (3, 2): K = 3 and N = 2 deliberately differ, which is the shape
// D-0067 hid behind, so a channel read from the wrong axis cannot pass by luck.
// Input: scale_x = 0.25, zp_x = -2. Weight scales 0.75 and 0.21875. Output:
// scale_y = 0.25, zp_y = -128, the zero point a post ReLU tensor gets.
//
// The weights, one column per output channel:
//   column 0: 0.75 / 0.75 = 1, -1.5 / 0.75 = -2, 1.875 / 0.75 = 2.5 -> 2
//   column 1: 0.65625 / 0.21875 = 3, 0.21875 / 0.21875 = 1,
//             -0.4375 / 0.21875 = -2
//
// The bias, with the zero point folded over all K taps:
//   column 0: 0.5 / 0.1875 = 2.67 -> 3,  3 - (-2) * (1 - 2 + 2) = 5
//   column 1: f32(-0.1) / 0.0546875 = -1.83 -> -2,  -2 - (-2) * (3 + 1 - 2) = 2
//
// The rescale:
//   column 0: 0.1875 / 0.25 = 0.75,  M0 = 1610612736, shift 0
//   column 1: 0.0546875 / 0.25 = 0.21875, doubled twice to 0.875,
//             M0 = 0.875 * 2^31 = 1879048192, shift 2
//
// There is no input zero point on the instruction and that is the opcode's
// profile, not an omission: every tap of a matrix multiplication is a real
// activation, so the whole of the zero point term is in the bias.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @hand_computed_matmul(
// CHECK:       npuisa.const dense<{{\[\[}}1, 3], [-2, 1], [2, -2]]> : tensor<3x2xi8>
// CHECK:       npuisa.const dense<[5, 2]> : tensor<2xi32>
// CHECK:       npuisa.const dense<{{\[\[}}1610612736, 1879048192], [0, 2]]> : tensor<2x2xi32>
// CHECK:       npuisa.matmul ins({{.*}} : memref<2x3xi8, #npu.scratchpad>, memref<3x2xi8, #npu.scratchpad>, memref<2xi32, #npu.scratchpad>, memref<2x2xi32, #npu.scratchpad>)
// CHECK-SAME:    outs(%{{.*}} : memref<2x2xi8, #npu.scratchpad>)
// CHECK-SAME:    {output_zero_point = -128 : i32, requant_multiplier = 1610612736 : i32, requant_shift = 0 : i32}
// CHECK-NOT:   npuisa.dequant
// CHECK:       return

func.func @hand_computed_matmul(%qx: tensor<2x3xi8>) -> tensor<2x2xi8> {
  %w = npu.constant dense<[[0.75, 0.65625], [-1.5, 0.21875], [1.875, -0.4375]]>
       : tensor<3x2xf32>
  %b = npu.constant dense<[0.5, -0.1]> : tensor<2xf32>
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<2x3xi8> to tensor<2x3xf32>
  %d = tensor.empty() : tensor<2x2xf32>
  %y = npu.matmul ins(%dx, %w, %b : tensor<2x3xf32>, tensor<3x2xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<2x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<2x2xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = -128 : i32}
        : tensor<2x2xf32> to tensor<2x2xi8>
  return %qy : tensor<2x2xi8>
}

// -----------------------------------------------------------------------------
// The per tensor arm: every channel shares one weight scale, so every
// channel's pair is the same and the instruction carries three operands.
//
// Section 14's granularity ablation compares this against the per channel
// arm, so it is a path of its own rather than a degenerate table. 0.5 * 0.375
// over 0.25 is 0.75 for both channels, M0 = 1610612736 at a shift of zero, and
// the two scalar fields say everything a table would. No bias here, so the
// folded term alone is the bias: -3 * (2 - 1) = -3 and -3 * (4 - 3) = -3, with
// channel 1's weights 1.5 / 0.375 = 4 and -1.125 / 0.375 = -3.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @the_per_tensor_arm(
// CHECK:       npuisa.const dense<{{\[\[\[\[}}2, -1]]], {{\[\[\[}}4, -3]]]]> : tensor<2x1x1x2xi8>
// CHECK:       npuisa.const dense<-3> : tensor<2xi32>
// CHECK-NOT:   tensor<2x2xi32>
// CHECK:       npuisa.conv2d ins({{.*}} : memref<1x1x1x2xi8, #npu.scratchpad>, memref<2x1x1x2xi8, #npu.scratchpad>, memref<2xi32, #npu.scratchpad>)
// CHECK-SAME:    requant_multiplier = 1610612736 : i32, requant_shift = 0 : i32

func.func @the_per_tensor_arm(%qx: tensor<1x1x1x2xi8>) -> tensor<1x2x1x1xi8> {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[1.5, -1.125]]]]>
       : tensor<2x1x1x2xf32>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<1x1x1x2xi8> to tensor<1x1x1x2xf32>
  %d = tensor.empty() : tensor<1x2x1x1xf32>
  %y = npu.conv2d ins(%dx, %w : tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>)
                  outs(%d : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 3.750000e-01, 3.750000e-01>}
       -> tensor<1x2x1x1xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<1x2x1x1xf32> to tensor<1x2x1x1xi8>
  return %qy : tensor<1x2x1x1xi8>
}

// -----------------------------------------------------------------------------
// The calibrator's whole shape, f32 on both sides of the model.
//
// The quantize in front is **not** part of the contraction and stays: it is
// the `QUANT` that turns the activation into integers, and the integer
// instruction reads its buffer. The dequantize after the trailing quantize
// stays too, because it feeds the function's f32 result. What goes is the
// pair between them: the dequantize that undid the calibrator's own quantize,
// and the quantize that the instruction's own rescale replaces.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @the_calibrators_whole_shape(
// CHECK:       npuisa.quant ins(%{{.*}} : memref<1x1x1x2xf32, #npu.scratchpad>) outs(%[[Q:.*]] : memref<1x1x1x2xi8, #npu.scratchpad>)
// CHECK-NOT:   npuisa.dequant
// CHECK:       npuisa.conv2d ins(%[[Q]], {{.*}}outs(%[[R:.*]] : memref<1x2x1x1xi8, #npu.scratchpad>)
// CHECK-NOT:   npuisa.quant
// CHECK:       npuisa.dequant ins(%[[R]] : memref<1x2x1x1xi8, #npu.scratchpad>)
// CHECK:       npuisa.dma_store

func.func @the_calibrators_whole_shape(%x: tensor<1x1x1x2xf32>)
    -> tensor<1x2x1x1xf32> {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[0.625, -0.234375]]]]>
       : tensor<2x1x1x2xf32>
  %qx = npu.quantize %x {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<1x1x1x2xf32> to tensor<1x1x1x2xi8>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<1x1x1x2xi8> to tensor<1x1x1x2xf32>
  %d = tensor.empty() : tensor<1x2x1x1xf32>
  %y = npu.conv2d ins(%dx, %w : tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>)
                  outs(%d : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 3.750000e-01, 1.562500e-01>}
       -> tensor<1x2x1x1xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<1x2x1x1xf32> to tensor<1x2x1x1xi8>
  %dy = npu.dequantize %qy {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<1x2x1x1xi8> to tensor<1x2x1x1xf32>
  return %dy : tensor<1x2x1x1xf32>
}

// -----------------------------------------------------------------------------
// An input something else also reads is not consumed.
//
// The dequantize feeds the contracted convolution and an f32 relu. The relu
// still needs an f32 value, so the DEQUANT is emitted for it, and the
// convolution reads the i8 argument beside it. The weight constant is shared
// the same way with an f32 convolution the calibrator did not touch, which
// keeps its f32 weights while the contracted one gets its own i8 copy.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @an_input_something_else_reads(
// CHECK:       npuisa.dma_load %arg0, %[[Q:[^ ]*]]
// CHECK:       npuisa.dequant ins(%[[Q]] : memref<1x1x1x2xi8, #npu.scratchpad>) outs(%[[DQ:[^ ]*]] : memref<1x1x1x2xf32, #npu.scratchpad>)
// CHECK:       npuisa.const dense<{{.*}}> : tensor<2x1x1x2xf32>
// CHECK:       npuisa.const dense<{{.*}}> : tensor<2x1x1x2xi8>
// CHECK:       npuisa.conv2d ins(%[[Q]], {{.*}}outs(%{{.*}} : memref<1x2x1x1xi8, #npu.scratchpad>)
// CHECK:       npuisa.relu ins(%[[DQ]] : memref<1x1x1x2xf32, #npu.scratchpad>)
// CHECK:       npuisa.conv2d ins(%{{.*}} : memref<1x1x1x2xf32, #npu.scratchpad>, memref<2x1x1x2xf32, #npu.scratchpad>) outs(%{{.*}} : memref<1x2x1x1xf32, #npu.scratchpad>)

func.func @an_input_something_else_reads(%qx: tensor<1x1x1x2xi8>)
    -> (tensor<1x2x1x1xi8>, tensor<1x1x1x2xf32>, tensor<1x2x1x1xf32>) {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[0.625, -0.234375]]]]>
       : tensor<2x1x1x2xf32>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<1x1x1x2xi8> to tensor<1x1x1x2xf32>
  %d = tensor.empty() : tensor<1x2x1x1xf32>
  %y = npu.conv2d ins(%dx, %w : tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>)
                  outs(%d : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 3.750000e-01, 1.562500e-01>}
       -> tensor<1x2x1x1xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<1x2x1x1xf32> to tensor<1x2x1x1xi8>
  %d1 = tensor.empty() : tensor<1x1x1x2xf32>
  %r = npu.relu ins(%dx : tensor<1x1x1x2xf32>) outs(%d1 : tensor<1x1x1x2xf32>)
       -> tensor<1x1x1x2xf32>
  %d2 = tensor.empty() : tensor<1x2x1x1xf32>
  %f = npu.conv2d ins(%r, %w : tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>)
                  outs(%d2 : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x1x1xf32>
  return %qy, %r, %f
      : tensor<1x2x1x1xi8>, tensor<1x1x1x2xf32>, tensor<1x2x1x1xf32>
}

// -----------------------------------------------------------------------------
// Partial coverage: each of these is missing one part of the calibrator's
// shape, so it is **not contracted and not refused**. It stays in the QDQ form
// and lowers to QUANT, f32 compute and DEQUANT exactly as before the
// contraction existed, which is Section 14's rule that a partially covered
// operation is never half rewritten.
// -----------------------------------------------------------------------------

// No weight scales: the profile named the operation's activations and not its
// weights, so there is nothing to quantize the weights with.
// CHECK-LABEL: func.func @no_weight_scales(
// CHECK:       npuisa.dequant
// CHECK:       npuisa.conv2d ins({{.*}} : memref<1x1x1x2xf32, #npu.scratchpad>, memref<2x1x1x2xf32, #npu.scratchpad>)
// CHECK:       npuisa.quant

func.func @no_weight_scales(%qx: tensor<1x1x1x2xi8>) -> tensor<1x2x1x1xi8> {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[0.625, -0.234375]]]]>
       : tensor<2x1x1x2xf32>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<1x1x1x2xi8> to tensor<1x1x1x2xf32>
  %d = tensor.empty() : tensor<1x2x1x1xf32>
  %y = npu.conv2d ins(%dx, %w : tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>)
                  outs(%d : tensor<1x2x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x1x1xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<1x2x1x1xf32> to tensor<1x2x1x1xi8>
  return %qy : tensor<1x2x1x1xi8>
}

// Weights that are not a constant cannot be quantized into the program, and
// this instruction set has no instruction that would quantize them at run time.
// CHECK-LABEL: func.func @weights_that_are_not_a_constant(
// CHECK:       npuisa.dequant
// CHECK:       npuisa.matmul ins({{.*}} : memref<1x3xf32, #npu.scratchpad>, memref<3x2xf32, #npu.scratchpad>)
// CHECK:       npuisa.quant

func.func @weights_that_are_not_a_constant(%qx: tensor<1x3xi8>,
                                           %w: tensor<3x2xf32>)
    -> tensor<1x2xi8> {
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x2xf32>)
                  outs(%d : tensor<1x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<1x2xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = -128 : i32}
        : tensor<1x2xf32> to tensor<1x2xi8>
  return %qy : tensor<1x2xi8>
}

// A bias that is not a constant cannot be folded into the int32 bias.
// CHECK-LABEL: func.func @a_bias_that_is_not_a_constant(
// CHECK:       npuisa.dequant
// CHECK:       npuisa.matmul ins({{.*}} : memref<1x3xf32, #npu.scratchpad>, memref<3x2xf32, #npu.scratchpad>, memref<2xf32, #npu.scratchpad>)
// CHECK:       npuisa.quant

func.func @a_bias_that_is_not_a_constant(%qx: tensor<1x3xi8>,
                                         %b: tensor<2xf32>)
    -> tensor<1x2xi8> {
  %w = npu.constant dense<[[0.75, 0.65625], [-1.5, 0.21875], [1.875, -0.4375]]>
       : tensor<3x2xf32>
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %y = npu.matmul ins(%dx, %w, %b : tensor<1x3xf32>, tensor<3x2xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<1x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<1x2xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = -128 : i32}
        : tensor<1x2xf32> to tensor<1x2xi8>
  return %qy : tensor<1x2xi8>
}

// A result read in f32 as well as quantized would need both the integer
// result and the f32 one. Contracting it would be half contracting it.
// CHECK-LABEL: func.func @a_result_read_twice(
// CHECK:       npuisa.dequant
// CHECK:       npuisa.matmul ins({{.*}} : memref<1x3xf32, #npu.scratchpad>, memref<3x2xf32, #npu.scratchpad>)
// CHECK:       npuisa.quant

func.func @a_result_read_twice(%qx: tensor<1x3xi8>)
    -> (tensor<1x2xi8>, tensor<1x2xf32>) {
  %w = npu.constant dense<[[0.75, 0.65625], [-1.5, 0.21875], [1.875, -0.4375]]>
       : tensor<3x2xf32>
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x2xf32>)
                  outs(%d : tensor<1x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<1x2xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = -128 : i32}
        : tensor<1x2xf32> to tensor<1x2xi8>
  return %qy, %y : tensor<1x2xi8>, tensor<1x2xf32>
}

// A result no quantize reads has no output scale to requantize into.
// CHECK-LABEL: func.func @no_quantize_after_it(
// CHECK:       npuisa.dequant
// CHECK:       npuisa.matmul ins({{.*}} : memref<1x3xf32, #npu.scratchpad>, memref<3x2xf32, #npu.scratchpad>)
// CHECK-NOT:   npuisa.quant
// CHECK:       return

func.func @no_quantize_after_it(%qx: tensor<1x3xi8>) -> tensor<1x2xf32> {
  %w = npu.constant dense<[[0.75, 0.65625], [-1.5, 0.21875], [1.875, -0.4375]]>
       : tensor<3x2xf32>
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<1x3xi8> to tensor<1x3xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %y = npu.matmul ins(%dx, %w : tensor<1x3xf32>, tensor<3x2xf32>)
                  outs(%d : tensor<1x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<1x2xf32>
  return %y : tensor<1x2xf32>
}
