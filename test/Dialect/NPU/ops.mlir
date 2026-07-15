// Round trip test for the core npu ops: parse, print, parse again, and check the
// printed form is stable.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// CHECK-LABEL: func.func @ops
func.func @ops(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
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
