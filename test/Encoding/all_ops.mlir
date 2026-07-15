// Exercise every op through the whole backend: lower to npuisa, allocate the
// scratchpad, encode to .nbin, and disassemble. This covers the lowering
// patterns, the encoder, and the disassembler for every opcode at once.

// RUN: npu-opt %s -npu-lower-to-npuisa -npu-allocate-scratchpad -o %t.mlir
// RUN: npu-translate %t.mlir -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// CHECK-DAG: DMA_LOAD
// CHECK-DAG: CONV2D
// CHECK-DAG: RELU
// CHECK-DAG: ADD
// CHECK-DAG: MUL
// CHECK-DAG: POOL_MAX
// CHECK-DAG: POOL_AVG
// CHECK-DAG: RESHAPE
// CHECK-DAG: MATMUL
// CHECK-DAG: DMA_STORE
// CHECK-DAG: HALT

func.func @all(%x: tensor<1x1x8x8xf32>, %w1: tensor<4x1x3x3xf32>,
               %w2: tensor<64x10xf32>) -> tensor<1x10xf32> {
  %c = npu.conv2d %x, %w1 {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]}
    : (tensor<1x1x8x8xf32>, tensor<4x1x3x3xf32>) -> tensor<1x4x8x8xf32>
  %r = npu.relu %c : tensor<1x4x8x8xf32>
  %a = npu.add %r, %c : tensor<1x4x8x8xf32>
  %m = npu.mul %a, %r : tensor<1x4x8x8xf32>
  %p = npu.max_pool2d %m {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    : (tensor<1x4x8x8xf32>) -> tensor<1x4x4x4xf32>
  %q = npu.avg_pool2d %p {kernel_shape = [1, 1], strides = [1, 1], pads = [0, 0, 0, 0]}
    : (tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
  %f = npu.reshape %q : tensor<1x4x4x4xf32> to tensor<1x64xf32>
  %y = npu.matmul %f, %w2 : (tensor<1x64xf32>, tensor<64x10xf32>) -> tensor<1x10xf32>
  return %y : tensor<1x10xf32>
}
