// The fusion pass folds a trailing relu into a conv2d or matmul that already
// carries its bias operand, producing a single fused op with activation = relu.

// RUN: npu-opt %s -npu-fuse-ops | FileCheck %s

// CHECK-LABEL: func.func @conv_bias_relu
func.func @conv_bias_relu(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>,
                          %b: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  %0 = npu.conv2d %x, %w, %b {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]}
    : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %1 = npu.relu %0 : tensor<1x4x8x8xf32>
  return %1 : tensor<1x4x8x8xf32>
  // CHECK-NOT: npu.relu
  // CHECK: npu.conv2d %arg0, %arg1, %arg2 {activation = 1 : i32
}

// CHECK-LABEL: func.func @matmul_relu
func.func @matmul_relu(%a: tensor<8x16xf32>, %b: tensor<16x10xf32>,
                       %bias: tensor<10xf32>) -> tensor<8x10xf32> {
  %0 = npu.matmul %a, %b, %bias
    : (tensor<8x16xf32>, tensor<16x10xf32>, tensor<10xf32>) -> tensor<8x10xf32>
  %1 = npu.relu %0 : tensor<8x10xf32>
  return %1 : tensor<8x10xf32>
  // CHECK-NOT: npu.relu
  // CHECK: npu.matmul %arg0, %arg1, %arg2 {activation = 1 : i32}
}

// A relu whose producer already has an activation is left alone.
// CHECK-LABEL: func.func @no_double_fuse
func.func @no_double_fuse(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %0 = npu.conv2d %x, %w {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1], activation = 1 : i32}
    : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  %1 = npu.relu %0 : tensor<1x4x8x8xf32>
  // CHECK: npu.conv2d
  // CHECK: npu.relu
  return %1 : tensor<1x4x8x8xf32>
}
