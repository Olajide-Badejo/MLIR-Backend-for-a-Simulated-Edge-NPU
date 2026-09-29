// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-calibrate` puts an `npu.fused_op` around an operation the profile names
// back into the block before calibrating it.
//
// At `-O2` the pass runs after `-npu-fuse-ops`, so a convolution and its relu
// arrive in a region. Calibrated inside it, the pairs would be sealed in: a
// neighbour's dequantize could not fold with this input's quantize, which would
// read a block argument, and CSE could not merge a quantize two regions share.
// Dissolved, the program is the one `-O0` calibrates, and the relu is fused by
// the contraction instead, with the output pair at the relu's range: 0.008 at
// -128 here, where the convolution's own is 0.016 at -9. A region around an
// operation the profile does not name is left as it is.

// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-relu.json \
// RUN:   | FileCheck %s
// RUN: npu-opt %s --npu-calibrate=profile=%S/Inputs/calibrate-relu.json \
// RUN:   --mlir-pass-statistics 2>&1 >/dev/null | FileCheck %s --check-prefix=STATS

// STATS: NPUCalibrate
// STATS: 1 dissolved-regions
// STATS: 0 folded-pairs
// STATS: 1 rewritten
// STATS: 0 skipped
// STATS: 1 uncovered

// CHECK-LABEL: func.func @named
// CHECK-NOT: npu.fused_op
// CHECK: %[[QX:.*]] = npu.quantize %arg0 {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
// CHECK-NEXT: %[[DX:.*]] = npu.dequantize %[[QX]]
// CHECK-NEXT: %[[C:.*]] = npu.conv2d ins(%[[DX]]
// CHECK-SAME: weight_scales = array<f32: 0.0157480314, 0.0157480314>
// CHECK-NEXT: %[[R:.*]] = npu.relu ins(%[[C]]
// CHECK-NEXT: %[[Q:.*]] = npu.quantize %[[R]] {scale = 8.000000e-03 : f32, zero_point = -128 : i32}
// CHECK-NEXT: %[[D:.*]] = npu.dequantize %[[Q]]
// CHECK-NOT: npu.fused_op
// CHECK: return %[[D]]

func.func @named(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %f = npu.fused_op ins(%x, %w, %d0, %d1 : tensor<1x2x4x4xf32>,
                        tensor<2x2x1x1xf32>, tensor<1x2x4x4xf32>,
                        tensor<1x2x4x4xf32>) {
  ^bb0(%a: tensor<1x2x4x4xf32>, %b: tensor<2x2x1x1xf32>,
       %e0: tensor<1x2x4x4xf32>, %e1: tensor<1x2x4x4xf32>):
    %c = npu.conv2d ins(%a, %b : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                    outs(%e0 : tensor<1x2x4x4xf32>)
                    {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, group = 1 : i64}
         -> tensor<1x2x4x4xf32> loc("Conv_0")
    %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%e1 : tensor<1x2x4x4xf32>)
         -> tensor<1x2x4x4xf32> loc("Relu_1")
    npu.yield %r : tensor<1x2x4x4xf32>
  } -> tensor<1x2x4x4xf32> loc("Relu_1")
  return %f : tensor<1x2x4x4xf32>
}

// The profile has no Conv_9, so the region stays and nothing is quantized.
// CHECK-LABEL: func.func @unnamed
// CHECK-NOT: npu.quantize
// CHECK: npu.fused_op
// CHECK: npu.conv2d
// CHECK-NOT: weight_scales
// CHECK: npu.relu
// CHECK: npu.yield
// CHECK-NOT: npu.quantize
// CHECK: return

func.func @unnamed(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %f = npu.fused_op ins(%x, %w, %d0, %d1 : tensor<1x2x4x4xf32>,
                        tensor<2x2x1x1xf32>, tensor<1x2x4x4xf32>,
                        tensor<1x2x4x4xf32>) {
  ^bb0(%a: tensor<1x2x4x4xf32>, %b: tensor<2x2x1x1xf32>,
       %e0: tensor<1x2x4x4xf32>, %e1: tensor<1x2x4x4xf32>):
    %c = npu.conv2d ins(%a, %b : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                    outs(%e0 : tensor<1x2x4x4xf32>)
                    {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, group = 1 : i64}
         -> tensor<1x2x4x4xf32> loc("Conv_9")
    %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%e1 : tensor<1x2x4x4xf32>)
         -> tensor<1x2x4x4xf32> loc("Relu_9")
    npu.yield %r : tensor<1x2x4x4xf32>
  } -> tensor<1x2x4x4xf32> loc("Relu_9")
  return %f : tensor<1x2x4x4xf32>
}
