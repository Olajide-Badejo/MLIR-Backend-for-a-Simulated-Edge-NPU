// A private function that nothing calls is removed by symbol-dce, while public
// functions are kept.

// RUN: npu-opt %s -symbol-dce | FileCheck %s

// CHECK-LABEL: func.func @keep
func.func @keep(%a: tensor<2xf32>) -> tensor<2xf32> {
  %0 = npu.relu %a : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-NOT: unused_helper
func.func private @unused_helper(%a: tensor<2xf32>) -> tensor<2xf32> {
  %0 = npu.relu %a : tensor<2xf32>
  return %0 : tensor<2xf32>
}
