// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// A function that returns a buffer. A lowered function takes its outputs
// as trailing DRAM arguments and returns nothing.

func.func @returns_a_value(%in: memref<4x4xf32, #npu.dram> {npuisa.arg = "in"},
                           %out: memref<4x4xf32, #npu.dram> {npuisa.arg = "out"})
    -> memref<4x4xf32, #npu.dram>
    attributes {npuisa.scratchpad_bytes = 64 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<64xi8, #npu.scratchpad>
  %c0 = arith.constant 0 : index
  %v = memref.view %arena[%c0][]
     : memref<64xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %v
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_store %v, %out
    : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
  return %out : memref<4x4xf32, #npu.dram>
}
