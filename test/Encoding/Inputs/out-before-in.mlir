// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// A function whose output argument comes before an input. The order is
// the one P4 fixed and the encoder checks it rather than assuming it.

func.func @out_before_in(%out: memref<4x4xf32, #npu.dram> {npuisa.arg = "out"},
                         %in: memref<4x4xf32, #npu.dram> {npuisa.arg = "in"})
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
  return
}
