// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-calibrate` quantizes after the relu when a relu is a calibrated
// operation's only reader.
//
// The contraction in the lowering fuses that relu into the integer instruction
// as its activation, so the instruction's result is the relu's and the output
// pair takes the relu output's range: 0.008 with the zero point at -128 here,
// where the convolution's own output is 0.016 at -9. Four conditions and a
// negative for each: the relu is the only reader, its location names a Relu in
// the profile's graph, that node reads this operation's own output tensor, and
// its output has a range. Anything the pass declines is quantized where it
// always was, on the operation's result.

// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-relu.json \
// RUN:   | FileCheck %s
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-relu-other-tensor.json \
// RUN:   | FileCheck %s --check-prefix=UNFUSED
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-relu-no-graph.json \
// RUN:   | FileCheck %s --check-prefix=UNFUSED

// CHECK-LABEL: func.func @conv_relu
// CHECK: %[[C:.*]] = npu.conv2d
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: %[[R:.*]] = npu.relu ins(%[[C]]
// CHECK-NEXT: %[[Q:.*]] = npu.quantize %[[R]] {scale = 8.000000e-03 : f32, zero_point = -128 : i32}
// CHECK-NEXT: %[[D:.*]] = npu.dequantize %[[Q]] {scale = 8.000000e-03 : f32, zero_point = -128 : i32}
// CHECK: return %[[D]]

// The relu the profile's graph names reads another tensor, so it is not the
// relu the observer saw after this operation, and the pair stays on the
// convolution's result with the convolution's own range. The same happens with
// a profile that has no graph section at all, which is how a profile written
// before the section compiles.
// UNFUSED-LABEL: func.func @conv_relu
// UNFUSED: %[[C:.*]] = npu.conv2d
// UNFUSED-NEXT: %[[Q:.*]] = npu.quantize %[[C]] {scale = 1.600000e-02 : f32, zero_point = -9 : i32}
// UNFUSED-NEXT: %[[D:.*]] = npu.dequantize %[[Q]]
// UNFUSED: npu.relu ins(%[[D]]

func.func @conv_relu(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d1 : tensor<1x2x4x4xf32>)
       -> tensor<1x2x4x4xf32> loc("Relu_1")
  return %r : tensor<1x2x4x4xf32>
}

// The convolution's result is returned as well as read by the relu, so the
// relu is not its only reader and the program needs the value before the relu.
// CHECK-LABEL: func.func @read_twice
// CHECK: %[[C:.*]] = npu.conv2d
// CHECK-NEXT: %[[Q:.*]] = npu.quantize %[[C]] {scale = 1.600000e-02 : f32, zero_point = -9 : i32}

func.func @read_twice(%x: tensor<1x2x4x4xf32>)
    -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d1 : tensor<1x2x4x4xf32>)
       -> tensor<1x2x4x4xf32> loc("Relu_1")
  return %r, %c : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}
