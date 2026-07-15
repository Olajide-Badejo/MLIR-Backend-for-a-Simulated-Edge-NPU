// Encode an allocated npuisa program to .nbin and disassemble it. This locks in
// the binary format and the npu-objdump listing.

// RUN: npu-opt %s -npu-allocate-scratchpad -o %t.mlir
// RUN: npu-translate %t.mlir -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// CHECK: scratchpad
// CHECK: .dram
// CHECK: input0
// CHECK: output0
// CHECK: .text
// CHECK: DMA_LOAD
// CHECK: RELU
// CHECK: DMA_STORE
// CHECK: HALT

func.func @main(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = npuisa.dma_load %x : (tensor<4xf32>) -> !npuisa.buffer<tensor<4xf32>>
  %1 = npuisa.relu %0 : (!npuisa.buffer<tensor<4xf32>>) -> !npuisa.buffer<tensor<4xf32>>
  %2 = npuisa.dma_store %1 : (!npuisa.buffer<tensor<4xf32>>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
