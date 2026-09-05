// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A sub region of a DRAM buffer, which is what a tiled program loads per tile.
//
// **The DRAM side walks the view chain from P13 and this is the test of that.**
// Before it, `dramAddressOf` was a map lookup and nothing else, and a
// `memref.subview` of an argument was refused with "this DRAM buffer has no
// address in the DRAM map". The map holds whole regions, because a region is
// what the header declares and what the loader places, so a sub region will
// never be in it; the address of one is the region's address plus the bytes the
// view chain accumulates. That is the same analysis `scratchpadAddressOf` has
// used since P5 through `computeBufferRange`, which already understood
// `memref.subview`. Only this side had not asked it.
//
// **The arithmetic is the whole assertion.** The second tile starts at row 3 of
// an 8 by 8 image with 4 channels, so it starts 3 * 8 = 24 elements into the
// buffer, which is 96 bytes, which is `0x60`. A lookup that ignored the view
// would put both loads at `0x0` and the program would read the same tile twice.

// RUN: npu-translate %s -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// CHECK: input 0: dram@0x0 1x4x8x8xf32 (1024 bytes)

// The first tile starts at the buffer's own address, and carries the parent's
// strides rather than the ones its own extents imply: a 1x4x5x8 tile of a
// 1x4x8x8 buffer steps 64 elements per channel and 8 per row, not 40 and 8.
// CHECK: DMA_LOAD sp@0x{{[0-9a-f]+}} 1x4x5x8xf32 <- dram@0x0 1x4x5x8xf32 s[256,64,8,1]

// The second starts three rows in.
// CHECK: DMA_LOAD sp@0x{{[0-9a-f]+}} 1x4x5x8xf32 <- dram@0x60 1x4x5x8xf32 s[256,64,8,1]

// The file is hand written **allocated** IR, as `all_ops.mlir` is and for the
// same reason: what is under test is the encoder, and running the allocator
// first would make the addresses its choice rather than this file's.
func.func @two_row_tiles(
    %in: memref<1x4x8x8xf32, #npu.dram> {npuisa.arg = "in"},
    %out: memref<1x4x5x8xf32, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 640 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
      : memref<640xi8, #npu.scratchpad>
  %c0 = arith.constant 0 : index

  %first = memref.subview %in[0, 0, 0, 0] [1, 4, 5, 8] [1, 1, 1, 1]
      : memref<1x4x8x8xf32, #npu.dram>
        to memref<1x4x5x8xf32, strided<[256, 64, 8, 1]>, #npu.dram>
  %tile = memref.view %arena[%c0][]
      : memref<640xi8, #npu.scratchpad> to memref<1x4x5x8xf32, #npu.scratchpad>
  npuisa.dma_load %first, %tile
      : memref<1x4x5x8xf32, strided<[256, 64, 8, 1]>, #npu.dram>
        to memref<1x4x5x8xf32, #npu.scratchpad>

  %second = memref.subview %in[0, 0, 3, 0] [1, 4, 5, 8] [1, 1, 1, 1]
      : memref<1x4x8x8xf32, #npu.dram>
        to memref<1x4x5x8xf32, strided<[256, 64, 8, 1], offset: 24>, #npu.dram>
  npuisa.dma_load %second, %tile
      : memref<1x4x5x8xf32, strided<[256, 64, 8, 1], offset: 24>, #npu.dram>
        to memref<1x4x5x8xf32, #npu.scratchpad>

  npuisa.dma_store %tile, %out
      : memref<1x4x5x8xf32, #npu.scratchpad>
        to memref<1x4x5x8xf32, #npu.dram>
  return
}
