// Lowering from the npu tensor dialect to the npuisa instruction dialect via the
// dialect conversion framework. DRAM tensors become scratchpad buffers, with DMA
// inserted only at the DRAM boundaries.

// RUN: npu-opt %s -npu-lower-to-npuisa | FileCheck %s

// CHECK-LABEL: func.func @conv_relu
func.func @conv_relu(%x: tensor<1x1x4x4xf32>, %w: tensor<2x1x3x3xf32>,
                     %b: tensor<2xf32>) -> tensor<1x2x2x2xf32> {
  // Every function argument is DRAM, so each is loaded into the scratchpad.
  // CHECK-DAG: npuisa.dma_load %arg0
  // CHECK-DAG: npuisa.dma_load %arg1
  // CHECK-DAG: npuisa.dma_load %arg2
  %0 = npu.conv2d %x, %w, %b {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]}
    : (tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  // The conv result stays in scratchpad and feeds relu with no DMA in between.
  // CHECK: %[[C:.*]] = npuisa.conv2d
  // CHECK: %[[R:.*]] = npuisa.relu %[[C]]
  %1 = npu.relu %0 : tensor<1x2x2x2xf32>
  // The returned value is moved back to DRAM.
  // CHECK: npuisa.dma_store %[[R]]
  return %1 : tensor<1x2x2x2xf32>
}

// CHECK-LABEL: func.func @weight_constant
func.func @weight_constant(%x: tensor<1x1x4x4xf32>) -> tensor<1x2x2x2xf32> {
  // A weight constant becomes a DRAM npuisa.const that is dma_load'd to scratchpad
  // before the convolution can use it.
  // CHECK: %[[DRAM:.*]] = npuisa.const
  // CHECK: npuisa.dma_load %[[DRAM]]
  %w = npu.constant {value = dense<1.000000e+00> : tensor<2x1x3x3xf32>} : tensor<2x1x3x3xf32>
  // CHECK: npuisa.conv2d
  %0 = npu.conv2d %x, %w {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]}
    : (tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x2x2xf32>
  // CHECK: npuisa.dma_store
  return %0 : tensor<1x2x2x2xf32>
}
