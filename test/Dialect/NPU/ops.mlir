// Round trip test for the npu ops: parse, print, parse again, and check the
// printed form is stable.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// CHECK-LABEL: func.func @pointwise
func.func @pointwise(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: npu.constant {{.*}} : tensor<2x3xf32>
  %0 = npu.constant {value = dense<1.000000e+00> : tensor<2x3xf32>} : tensor<2x3xf32>
  // CHECK: npu.relu %arg0 : tensor<2x3xf32>
  %1 = npu.relu %arg0 : tensor<2x3xf32>
  // CHECK: npu.add %arg1, %{{.*}} : tensor<2x3xf32>
  %2 = npu.add %arg1, %1 : tensor<2x3xf32>
  // CHECK: npu.mul %{{.*}}, %{{.*}} : tensor<2x3xf32>
  %3 = npu.mul %2, %0 : tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @conv
func.func @conv(%input: tensor<1x3x8x8xf32>, %weight: tensor<4x3x3x3xf32>,
                %bias: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  // CHECK: npu.conv2d %arg0, %arg1, %arg2 {activation = 1 : i32,
  // CHECK-SAME: : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  %0 = npu.conv2d %input, %weight, %bias {
      strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1],
      activation = 1 : i32
    } : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// CHECK-LABEL: func.func @conv_no_bias
func.func @conv_no_bias(%input: tensor<1x3x8x8xf32>,
                        %weight: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  // CHECK: npu.conv2d %arg0, %arg1 {
  // CHECK-NOT: activation
  %0 = npu.conv2d %input, %weight {
      strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]
    } : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// CHECK-LABEL: func.func @fully_connected
func.func @fully_connected(%lhs: tensor<8x16xf32>, %rhs: tensor<16x10xf32>,
                           %bias: tensor<10xf32>) -> tensor<8x10xf32> {
  // CHECK: npu.matmul %arg0, %arg1, %arg2
  %0 = npu.matmul %lhs, %rhs, %bias
    : (tensor<8x16xf32>, tensor<16x10xf32>, tensor<10xf32>) -> tensor<8x10xf32>
  return %0 : tensor<8x10xf32>
}

// CHECK-LABEL: func.func @pooling
func.func @pooling(%arg0: tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: npu.max_pool2d %arg0
  %0 = npu.max_pool2d %arg0 {
      kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]
    } : (tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32>
  // CHECK: npu.avg_pool2d
  %1 = npu.avg_pool2d %0 {
      kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0]
    } : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  return %1 : tensor<1x4x4x4xf32>
}

// CHECK-LABEL: func.func @shape_ops
func.func @shape_ops(%arg0: tensor<1x4x2x2xf32>) -> tensor<2x8xf32> {
  // CHECK: npu.reshape %arg0 : tensor<1x4x2x2xf32> to tensor<16xf32>
  %0 = npu.reshape %arg0 : tensor<1x4x2x2xf32> to tensor<16xf32>
  // CHECK: npu.reshape %{{.*}} : tensor<16xf32> to tensor<2x8xf32>
  %1 = npu.reshape %0 : tensor<16xf32> to tensor<2x8xf32>
  return %1 : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @transpose_concat
func.func @transpose_concat(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x4xf32> {
  // CHECK: npu.transpose %arg0 {{.*}} : tensor<2x3xf32> to tensor<3x2xf32>
  %0 = npu.transpose %arg0 {permutation = [1, 0]} : tensor<2x3xf32> to tensor<3x2xf32>
  %1 = npu.transpose %arg1 {permutation = [1, 0]} : tensor<2x3xf32> to tensor<3x2xf32>
  // CHECK: npu.concat %{{.*}}, %{{.*}}
  %2 = npu.concat %0, %1 {axis = 1 : i64}
    : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x4xf32>
  return %2 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @bn
func.func @bn(%arg0: tensor<1x4x8x8xf32>, %s: tensor<4xf32>, %o: tensor<4xf32>,
              %m: tensor<4xf32>, %v: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  // CHECK: npu.batch_norm
  %0 = npu.batch_norm %arg0, %s, %o, %m, %v
    : (tensor<1x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}
