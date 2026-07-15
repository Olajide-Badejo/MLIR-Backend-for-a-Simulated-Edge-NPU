// The dedicated BatchNorm folding pass folds bn(conv(x, W)) into a single conv
// whose weights and bias absorb the normalization. The constants below are chosen
// so the arithmetic is exact:
//   coef[o]     = scale[o] / sqrt(var[o] + eps) = [2, 4] / sqrt(3 + 1) = [1, 2]
//   new weight  = W * coef                      = [2, 3] * [1, 2] = [2, 6]
//   new bias    = coef * (0 - mean) + offset    = [1, 2] * (-1) + 0.5 = [-0.5, -1.5]

// RUN: npu-opt %s -npu-fold-batchnorm | FileCheck %s

// CHECK-LABEL: func.func @fold_bn
func.func @fold_bn(%x: tensor<1x1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %w = npu.constant {value = dense<[[[[2.000000e+00]]], [[[3.000000e+00]]]]> : tensor<2x1x1x1xf32>} : tensor<2x1x1x1xf32>
  %conv = npu.conv2d %x, %w {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]}
    : (tensor<1x1x2x2xf32>, tensor<2x1x1x1xf32>) -> tensor<1x2x2x2xf32>
  %scale = npu.constant {value = dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %offset = npu.constant {value = dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>} : tensor<2xf32>
  %mean = npu.constant {value = dense<[1.000000e+00, 1.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %var = npu.constant {value = dense<[3.000000e+00, 3.000000e+00]> : tensor<2xf32>} : tensor<2xf32>
  %bn = npu.batch_norm %conv, %scale, %offset, %mean, %var {epsilon = 1.000000e+00 : f32}
    : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  return %bn : tensor<1x2x2x2xf32>

  // CHECK-NOT: npu.batch_norm
  // CHECK-DAG: %[[B:.*]] = npu.constant {{.*}}dense<[-5.000000e-01, -1.500000e+00]>
  // CHECK-DAG: %[[W:.*]] = npu.constant {{.*}}6.000000e+00
  // CHECK: npu.conv2d %arg0, %{{.*}}, %{{.*}} :
}
