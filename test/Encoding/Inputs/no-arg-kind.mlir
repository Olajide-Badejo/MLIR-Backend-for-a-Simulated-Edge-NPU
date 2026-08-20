// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// A function whose arguments carry no npuisa.arg attribute. The binary
// declares its input and output regions separately and the encoder will not
// guess which is which.

func.func @no_arg_kind(%in: memref<4x4xf32, #npu.dram>,
                       %out: memref<4x4xf32, #npu.dram>)
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
