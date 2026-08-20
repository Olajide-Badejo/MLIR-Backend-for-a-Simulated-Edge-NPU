// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// A function holding an operation the encoder has no case for. It is
// refused by name rather than dropped: a silently skipped computation is a
// binary that runs and computes the wrong thing.

func.func @unencodable(%in: memref<4x4xf32, #npu.dram> {npuisa.arg = "in"},
                       %out: memref<4x4xf32, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 64 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<64xi8, #npu.scratchpad>
  %c0 = arith.constant 0 : index
  %v = memref.view %arena[%c0][]
     : memref<64xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %v
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  %f = arith.constant 1.000000e+00 : f32
  %z = arith.addf %f, %f : f32
  npuisa.dma_store %v, %out
    : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
  return
}
