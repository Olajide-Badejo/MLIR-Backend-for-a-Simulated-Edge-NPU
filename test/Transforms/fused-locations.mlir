// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-fuse-bias` and `-npu-fold-batchnorm` record what they absorbed.
//
// After the bias fusion a convolution computes what the add's result was, and
// after the batch norm fold what the batch norm's was. Each pass now says so in
// the convolution's location: a fused location with the convolution's own name
// first and each absorbed node's after it, in the order they were absorbed.
//
// **The first name is unchanged, and that is what keeps fp32 where it was.**
// The encoder's debug section and `-npu-calibrate`'s join to the profile's
// `nodes` section both read the first name, so every fp32 binary of the 217
// cells is byte identical before and after, debug section included. The two
// constants the fold creates take the convolution's location as it stood when
// they were made, which after the bias fusion already names the add; their
// first name is the convolution's either way.
//
// **The last name is what calibration after the folds needs**: the tensor the
// operation's result now is, whose range the profile's `graph` section gives.
// Here that is the batch norm's output `y`, and with the relu the profile shows
// reading it, the output pair goes after the relu at the relu output's range.

// RUN: npu-opt %s --npu-fuse-bias --npu-fold-batchnorm \
// RUN:   --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s
// RUN: npu-opt %s --npu-fuse-bias --npu-fold-batchnorm \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-fused-location.json \
// RUN:   | FileCheck %s --check-prefix=CALIBRATED
// RUN: npu-opt %s --npu-fuse-bias --npu-fold-batchnorm \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-relu-no-graph.json \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOGRAPH

// CHECK-LABEL: func.func @conv_bias_bn_relu
// CHECK: npu.constant {{.*}} loc(fused["Conv_0", "Add_1"])
// CHECK: npu.constant {{.*}} loc(fused["Conv_0", "Add_1"])
// CHECK: npu.conv2d
// CHECK-SAME: loc(fused["Conv_0", "Add_1", "BatchNormalization_2"])
// CHECK-NOT: npu.batch_norm
// CHECK-NOT: npu.add
// CHECK: npu.relu

// The convolution's result is the batch norm's, so its reader, the relu the
// graph shows reading `y`, is fused, and the pair after it takes `r`'s range.
// CALIBRATED-LABEL: func.func @conv_bias_bn_relu
// CALIBRATED: npu.conv2d
// CALIBRATED: %[[R:.*]] = npu.relu
// CALIBRATED-NEXT: npu.quantize %[[R]] {scale = 1.200000e-02 : f32, zero_point = -128 : i32}

// A profile with no graph cannot say what the absorbed batch norm writes, so
// the convolution has no range for its result and is skipped and counted
// rather than quantized with the pre fold range.
// NOGRAPH: remark: -npu-calibrate rewrote nothing in 'conv_bias_bn_relu': of its 1 quantizable operations the profile does not name 0 and names 1 without a full set of ranges

func.func @conv_bias_bn_relu(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<1.0> : tensor<2x2x1x1xf32> loc("Conv_0")
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32> loc("Conv_0")
  %bias = npu.constant dense<[0.5, -0.5]> : tensor<2xf32> loc("Add_1")
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %b = npu.add ins(%c, %bias : tensor<1x2x4x4xf32>, tensor<2xf32>)
               outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> loc("Add_1")
  %g = npu.constant dense<[2.0, 0.5]> : tensor<2xf32>
  %be = npu.constant dense<0.0> : tensor<2xf32>
  %m = npu.constant dense<0.0> : tensor<2xf32>
  %v = npu.constant dense<1.0> : tensor<2xf32>
  %d2 = tensor.empty() : tensor<1x2x4x4xf32>
  %y = npu.batch_norm ins(%b, %g, %be, %m, %v : tensor<1x2x4x4xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d2 : tensor<1x2x4x4xf32>)
                      {epsilon = 0.0 : f32}
       -> tensor<1x2x4x4xf32> loc("BatchNormalization_2")
  %d3 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%y : tensor<1x2x4x4xf32>) outs(%d3 : tensor<1x2x4x4xf32>)
       -> tensor<1x2x4x4xf32> loc("Relu_3")
  return %r : tensor<1x2x4x4xf32>
}
