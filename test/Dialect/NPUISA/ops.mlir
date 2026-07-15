// Round trip for the npuisa instruction dialect: the scratchpad buffer type and
// the DMA and compute instructions. A buffer wraps the logical tensor type, so it
// prints as !npuisa.buffer<tensor<...>>.

// RUN: npu-opt %s | npu-opt | FileCheck %s

// CHECK-LABEL: func.func @conv_block
func.func @conv_block(%x: tensor<1x1x28x28xf32>, %w: tensor<6x1x5x5xf32>,
                      %b: tensor<6xf32>) -> tensor<1x6x24x24xf32> {
  // CHECK: npuisa.dma_load %arg0 : (tensor<1x1x28x28xf32>) -> !npuisa.buffer<tensor<1x1x28x28xf32>>
  %0 = npuisa.dma_load %x : (tensor<1x1x28x28xf32>) -> !npuisa.buffer<tensor<1x1x28x28xf32>>
  %1 = npuisa.dma_load %w : (tensor<6x1x5x5xf32>) -> !npuisa.buffer<tensor<6x1x5x5xf32>>
  %2 = npuisa.dma_load %b : (tensor<6xf32>) -> !npuisa.buffer<tensor<6xf32>>
  // CHECK: npuisa.conv2d {{.*}}activation = 1 : i32
  %3 = npuisa.conv2d %0, %1, %2 {
      strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1], activation = 1 : i32
    } : (!npuisa.buffer<tensor<1x1x28x28xf32>>, !npuisa.buffer<tensor<6x1x5x5xf32>>, !npuisa.buffer<tensor<6xf32>>)
        -> !npuisa.buffer<tensor<1x6x24x24xf32>>
  // CHECK: npuisa.dma_store
  %4 = npuisa.dma_store %3 : (!npuisa.buffer<tensor<1x6x24x24xf32>>) -> tensor<1x6x24x24xf32>
  return %4 : tensor<1x6x24x24xf32>
}

// CHECK-LABEL: func.func @with_address
func.func @with_address(%x: tensor<8x16xf32>, %w: tensor<16x10xf32>) -> tensor<8x10xf32> {
  %0 = npuisa.dma_load %x {address = 0 : i64} : (tensor<8x16xf32>) -> !npuisa.buffer<tensor<8x16xf32>>
  %1 = npuisa.dma_load %w {address = 512 : i64} : (tensor<16x10xf32>) -> !npuisa.buffer<tensor<16x10xf32>>
  // CHECK: npuisa.matmul {{.*}}address = 1152
  %2 = npuisa.matmul %0, %1 {address = 1152 : i64}
    : (!npuisa.buffer<tensor<8x16xf32>>, !npuisa.buffer<tensor<16x10xf32>>) -> !npuisa.buffer<tensor<8x10xf32>>
  %3 = npuisa.relu %2 : (!npuisa.buffer<tensor<8x10xf32>>) -> !npuisa.buffer<tensor<8x10xf32>>
  %4 = npuisa.dma_store %3 : (!npuisa.buffer<tensor<8x10xf32>>) -> tensor<8x10xf32>
  return %4 : tensor<8x10xf32>
}

// CHECK-LABEL: func.func @control
func.func @control() {
  // CHECK: npuisa.nop
  npuisa.nop
  // CHECK: npuisa.halt
  npuisa.halt
}
