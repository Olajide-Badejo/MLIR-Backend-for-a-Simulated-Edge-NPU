// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-calibrate` removes the pairs it forms that compute nothing.
//
// Two calibrated convolutions in series, the first one's output the second
// one's input. Wrapping each one's activations gives the first its output pair
// and the second its input pair on the same tensor, so between them sits a
// dequantize then a quantize with the same scale and zero point: a `DEQUANT`
// and a `QUANT` that would compute nothing between two integer instructions.
// The pass folds it, because `-O0` runs no canonicalization, and the first
// convolution's quantized output reaches the second convolution's dequantize
// directly.

// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-chain.json \
// RUN:   | FileCheck %s
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-chain.json \
// RUN:   --mlir-pass-statistics 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=STATS
// RUN: npu-opt %s \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-chain-requantize.json \
// RUN:   | FileCheck %s --check-prefix=REQUANT

// CHECK-LABEL: func.func @chain
// CHECK: %[[C0:.*]] = npu.conv2d
// CHECK: %[[QY:.*]] = npu.quantize %[[C0]] {scale = 1.600000e-02 : f32, zero_point = -128 : i32}
// CHECK-NOT: npu.quantize
// CHECK: %[[DY:.*]] = npu.dequantize %[[QY]] {scale = 1.600000e-02 : f32, zero_point = -128 : i32}
// CHECK-NOT: npu.quantize
// CHECK-NOT: npu.dequantize
// CHECK: %[[C1:.*]] = npu.conv2d ins(%[[DY]], %{{.*}}
// CHECK: %[[QZ:.*]] = npu.quantize %[[C1]] {scale = 4.000000e-02 : f32, zero_point = 5 : i32}
// CHECK: %[[DZ:.*]] = npu.dequantize %[[QZ]]
// CHECK: return %[[DZ]]

// STATS: 1 folded-pairs

// The negative case. The second convolution's input is calibrated to another
// scale, which here stands for a tensor the observer saw through an operation
// the compiler does not have; the pair between the two is a requantization from
// 0.016 to 0.02 and stays.
// REQUANT-LABEL: func.func @chain
// REQUANT: %[[C0:.*]] = npu.conv2d
// REQUANT: %[[QY:.*]] = npu.quantize %[[C0]] {scale = 1.600000e-02 : f32
// REQUANT: %[[DY:.*]] = npu.dequantize %[[QY]] {scale = 1.600000e-02 : f32
// REQUANT: %[[QR:.*]] = npu.quantize %[[DY]] {scale = 2.000000e-02 : f32
// REQUANT: %[[DR:.*]] = npu.dequantize %[[QR]] {scale = 2.000000e-02 : f32
// REQUANT: npu.conv2d ins(%[[DR]]

func.func @chain(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w0 = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %y = npu.conv2d ins(%x, %w0 : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  %w1 = npu.constant dense<1.000000e+00> : tensor<2x2x1x1xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %z = npu.conv2d ins(%y, %w1 : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d1 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_1")
  return %z : tensor<1x2x4x4xf32>
}
