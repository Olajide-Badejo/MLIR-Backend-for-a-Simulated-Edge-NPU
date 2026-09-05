// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A tiled result assembled in the scratchpad is refused, **by design**.
//
// This file exists so that a later reader finds that out by reading a test
// rather than by discovering it as a puzzle. It is the program P13 considered
// and did not take: one tile writes a sub region of a whole output buffer and a
// later instruction reads all of it, which is a buffer written in pieces and
// read whole.
//
// **Checks 8 and 9 refuse it and they are right to.** `WrittenSpans` records
// one span per written address and deliberately does not merge adjacent ones,
// because merging two adjacent buffers into one range would let an over read
// that runs off the end of the first and into the second pass validation, which
// is precisely the case the rule exists to catch. A tiled assembly and that
// over read are **not distinguishable by addresses**, and addresses are all a
// validator has: the binary carries region identity for DRAM, in its declared
// input, output, constant and spill regions, and none at all for the
// scratchpad, which is one arena of offsets.
//
// **So the tiling design puts the assembly in DRAM instead**, storing tiles as
// they are produced and loading the slices the next layer needs. Tiling exists
// because the value did not fit on chip, so a result that had to be split has
// no business being reassembled in the memory it did not fit in, and the round
// trip is a cost the program accepted when it tiled. `docs/PHASE_STATE.md`
// carries the decision and the reasoning.
//
// **What this test is not.** It is not a claim that the refusal is unfortunate
// or that the check should one day be relaxed. It is the record that the
// program below was written, run, and rejected for a reason somebody chose.

// RUN: npu-opt %s --npu-allocate-scratchpad -o %t.alloc.mlir
// RUN: not npu-translate %t.alloc.mlir -o %t.nbin 2>&1 | FileCheck %s
// RUN: not test -e %t.nbin

// The message names the check, the read, and the write that does not cover it.
// CHECK: operand-extent
// CHECK-SAME: reads 2048 bytes from 0
// CHECK-SAME: buffer written there ends at 512

// And the file is refused rather than written, which is the half that matters:
// a program the encoder cannot express must not become a file that runs.
// CHECK: no file has been written

func.func @assemble_in_scratchpad(
    %in: memref<1x4x8x8xf32, #npu.dram> {npuisa.arg = "in"},
    %w: memref<8x4x3x3xf32, #npu.dram> {npuisa.arg = "in"},
    %out: memref<1x8x8x8xf32, #npu.dram> {npuisa.arg = "out"}) {
  %spin = memref.alloc() : memref<1x4x8x8xf32, #npu.scratchpad>
  npuisa.dma_load %in, %spin
      : memref<1x4x8x8xf32, #npu.dram> to memref<1x4x8x8xf32, #npu.scratchpad>
  %spw = memref.alloc() : memref<8x4x3x3xf32, #npu.scratchpad>
  npuisa.dma_load %w, %spw
      : memref<8x4x3x3xf32, #npu.dram> to memref<8x4x3x3xf32, #npu.scratchpad>
  %spout = memref.alloc() : memref<1x8x8x8xf32, #npu.scratchpad>

  // One tile of the output: four of its eight channels and four of its eight
  // rows, which is a quarter of the buffer, written through a strided view.
  %o0 = memref.subview %spout[0, 0, 0, 0] [1, 4, 4, 8] [1, 1, 1, 1]
      : memref<1x8x8x8xf32, #npu.scratchpad>
        to memref<1x4x4x8xf32, strided<[512, 64, 8, 1]>, #npu.scratchpad>
  %w0 = memref.subview %spw[0, 0, 0, 0] [4, 4, 3, 3] [1, 1, 1, 1]
      : memref<8x4x3x3xf32, #npu.scratchpad>
        to memref<4x4x3x3xf32, strided<[36, 9, 3, 1]>, #npu.scratchpad>
  %i0 = memref.subview %spin[0, 0, 0, 0] [1, 4, 5, 8] [1, 1, 1, 1]
      : memref<1x4x8x8xf32, #npu.scratchpad>
        to memref<1x4x5x8xf32, strided<[256, 64, 8, 1]>, #npu.scratchpad>
  npuisa.conv2d ins(%i0, %w0 : memref<1x4x5x8xf32, strided<[256, 64, 8, 1]>, #npu.scratchpad>,
                               memref<4x4x3x3xf32, strided<[36, 9, 3, 1]>, #npu.scratchpad>)
                outs(%o0 : memref<1x4x4x8xf32, strided<[512, 64, 8, 1]>, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 0, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}

  // The read of the whole buffer that the quarter above does not cover.
  npuisa.dma_store %spout, %out
      : memref<1x8x8x8xf32, #npu.scratchpad> to memref<1x8x8x8xf32, #npu.dram>
  return
}
