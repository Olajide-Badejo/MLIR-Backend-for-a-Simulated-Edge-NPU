// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
// A function whose scratchpad buffers are still typed allocations. The
// encoder reads offsets out of the IR and does not assign them, so this is a
// pipeline that skipped -npu-allocate-scratchpad.

func.func @unallocated(%in: memref<4x4xf32, #npu.dram> {npuisa.arg = "in"},
                       %out: memref<4x4xf32, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 64 : i64} {
  %v = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %v
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.dma_store %v, %out
    : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
  return
}
