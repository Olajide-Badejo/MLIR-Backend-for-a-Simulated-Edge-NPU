// Dead code elimination for npu ops. Because every op carries the Pure trait,
// an op whose results are unused has no effects and is removed by the upstream
// canonicalizer. This is the whole point of putting Pure on the ops.

// RUN: npu-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @dead_pure_ops
func.func @dead_pure_ops(%x: tensor<1x3x8x8xf32>,
                         %w: tensor<4x3x3x3xf32>) -> tensor<1x3x8x8xf32> {
  // These results are never used. Being Pure, the whole chain is dead.
  %0 = npu.conv2d %x, %w {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]}
    : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  %1 = npu.relu %0 : tensor<1x4x8x8xf32>
  %2 = npu.max_pool2d %1 {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    : (tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32>
  // CHECK-NOT: npu.conv2d
  // CHECK-NOT: npu.relu
  // CHECK-NOT: npu.max_pool2d
  // CHECK: return %arg0
  return %x : tensor<1x3x8x8xf32>
}
