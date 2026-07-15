// Constant folding and canonicalization patterns, exercised through the built in
// canonicalize pass.

// RUN: npu-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_add
func.func @fold_add() -> tensor<2xf32> {
  %0 = npu.constant {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %1 = npu.constant {value = dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %2 = npu.add %0, %1 : tensor<2xf32>
  // CHECK-NOT: npu.add
  // CHECK: npu.constant {{.*}}dense<[4.000000e+00, 6.000000e+00]>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: func.func @fold_mul
func.func @fold_mul() -> tensor<2xf32> {
  %0 = npu.constant {value = dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %1 = npu.constant {value = dense<[4.000000e+00, 5.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %2 = npu.mul %0, %1 : tensor<2xf32>
  // CHECK-NOT: npu.mul
  // CHECK: npu.constant {{.*}}dense<[8.000000e+00, 1.500000e+01]>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: func.func @fold_relu
func.func @fold_relu() -> tensor<2xf32> {
  %0 = npu.constant {value = dense<[-1.000000e+00, 2.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %1 = npu.relu %0 : tensor<2xf32>
  // CHECK-NOT: npu.relu
  // CHECK: npu.constant {{.*}}dense<[0.000000e+00, 2.000000e+00]>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @relu_of_relu
func.func @relu_of_relu(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = npu.relu %arg0 : tensor<2xf32>
  %1 = npu.relu %0 : tensor<2xf32>
  // CHECK: %[[R:.*]] = npu.relu %arg0 : tensor<2xf32>
  // CHECK-NEXT: return %[[R]]
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func.func @reshape_identity
func.func @reshape_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = npu.reshape %arg0 : tensor<2x3xf32> to tensor<2x3xf32>
  // CHECK-NOT: npu.reshape
  // CHECK: return %arg0
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @reshape_of_reshape
func.func @reshape_of_reshape(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = npu.reshape %arg0 : tensor<2x3xf32> to tensor<3x2xf32>
  %1 = npu.reshape %0 : tensor<3x2xf32> to tensor<6xf32>
  // CHECK: %[[R:.*]] = npu.reshape %arg0 : tensor<2x3xf32> to tensor<6xf32>
  // CHECK-NEXT: return %[[R]]
  return %1 : tensor<6xf32>
}
