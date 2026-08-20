// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The disassembler, over a file the whole pipeline produced.
//
// The listing below is **regenerated rather than hand typed**. Writing it by
// hand would mean asserting what somebody believed the disassembler printed,
// and the first time the two disagreed the file would be edited to match the
// output, which is a test that can only ever pass.
//
// Four things are checked and each one is a different claim.
//
//   DUMP      a well formed file disassembles, addresses and all, and every
//             instruction that came from an ONNX node carries its name.
//   STRIPPED  --strip-debug produces a legal file with an empty debug section
//             and no names.
//   WARN      a file that decodes but does not validate is still dumped, with
//             a warning block naming the check that rejected it. That is the
//             whole reason decodeUnvalidated() exists.
//   SHORT     a file that cannot be framed at all is refused instead. Reading
//             past the end of a buffer to show somebody what is there is the
//             bug this subsystem exists to prevent.
//
// --mlir-print-debuginfo is not decoration on the first run line. npu-opt
// prints locations only when asked, so without it the encoder sees a module
// with no NameLoc anywhere and writes an empty debug section: the pipeline
// would still work and the names would silently be gone.

// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad \
// RUN:   --mlir-print-debuginfo | npu-translate -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s --check-prefix=DUMP

// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad \
// RUN:   --mlir-print-debuginfo | npu-translate --strip-debug -o %t.stripped.nbin
// RUN: npu-objdump %t.stripped.nbin | FileCheck %s --check-prefix=STRIPPED

// Byte 44 is the first extent of the first input region: the header is 24
// bytes, the input count is 4, then the region's DRAM offset is 8, its element
// type is 4 and its rank is 4. Zeroing eight bytes there gives the region a
// zero extent, which no shape has, so the file frames correctly and fails the
// region-shape check.
// RUN: cp %t.nbin %t.corrupt.nbin
// RUN: dd if=/dev/zero of=%t.corrupt.nbin bs=1 seek=44 count=8 conv=notrunc status=none
// RUN: npu-objdump %t.corrupt.nbin | FileCheck %s --check-prefix=WARN

// RUN: head -c 40 %t.nbin > %t.short.nbin
// RUN: not npu-objdump %t.short.nbin 2>&1 | FileCheck %s --check-prefix=SHORT

// DUMP:      ; .nbin version 1, host byte order
// DUMP-NEXT: ; scratchpad 1536 bytes, dram 1536 bytes
// DUMP-NEXT: ; 1 inputs, 1 outputs, 1 constants, 0 spill slots, 6 instructions, 3 debug entries
// DUMP-NEXT: ;   input 0: dram@0x0 1x8x4x4xf32 (512 bytes)
// DUMP-NEXT: ;   output 0: dram@0x200 1x8x4x4xf32 (512 bytes)
// DUMP-NEXT: ;   constant 0: dram@0x400 1x8x4x4xf32 (512 bytes), 512 bytes of data
// DUMP-NEXT: 0000  DMA_LOAD sp@0x0 1x8x4x4xf32 <- dram@0x0 1x8x4x4xf32
// DUMP-NEXT: 0001  DMA_LOAD sp@0x200 1x8x4x4xf32 <- dram@0x400 1x8x4x4xf32    ; weights
// DUMP-NEXT: 0002  MUL sp@0x400 1x8x4x4xf32 <- sp@0x0 1x8x4x4xf32, sp@0x200 1x8x4x4xf32 activation=none    ; scale_by_two
// DUMP-NEXT: 0003  RELU sp@0x0 1x8x4x4xf32 <- sp@0x400 1x8x4x4xf32    ; clamp
// DUMP-NEXT: 0004  DMA_STORE dram@0x200 1x8x4x4xf32 <- sp@0x0 1x8x4x4xf32
// DUMP-NEXT: 0005  HALT
// DUMP-NOT:  WARNING

// STRIPPED: ; 1 inputs, 1 outputs, 1 constants, 0 spill slots, 6 instructions, 0 debug entries
// STRIPPED-NOT: scale_by_two
// STRIPPED-NOT: clamp

// WARN:      WARNING: this file did not validate
// WARN:      WARNING: region-shape: the shape [0, 8, 4, 4]
// WARN:      0000  DMA_LOAD

// SHORT: error: cannot decode
// SHORT-SAME: structure: the count of the input regions is 1, which needs at least 16 bytes and only 12 remain

func.func @chain(%x: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<1x8x4x4xf32> loc("weights")
  %d0 = tensor.empty() : tensor<1x8x4x4xf32>
  %m = npu.mul ins(%x, %w : tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>)
               outs(%d0 : tensor<1x8x4x4xf32>)
       -> tensor<1x8x4x4xf32> loc("scale_by_two")
  %d1 = tensor.empty() : tensor<1x8x4x4xf32>
  %r = npu.relu ins(%m : tensor<1x8x4x4xf32>) outs(%d1 : tensor<1x8x4x4xf32>)
       -> tensor<1x8x4x4xf32> loc("clamp")
  return %r : tensor<1x8x4x4xf32>
}
