// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The quantization pair from the tensor level to the bytes, and back out
// through the disassembler.
//
// `all_ops.mlir` is hand written `npuisa` because what it pins is the encoder
// alone. This file starts at the tensor level and goes through the whole
// pipeline, for the reason `tiled-result-returned.mlir` gives: what is under
// test here is the **agreement** between the lowering, the allocator and the
// encoder about an operation whose result element type differs from its
// operand's, and a hand written program would assert only the last of the
// three.
//
// **The three things it pins that nothing else does.** That an i8 buffer gets
// an i8 sized allocation rather than an f32 sized one, which is a byte count
// the allocator has to derive from the element type. That the scale and the
// zero point reach the binary, since Section 14 requires the quantization
// parameters to live in the file rather than in an out of band JSON. And that
// the zero point survives as a **negative** number, which is where the signless
// storage and the signed reading of Section 7.2 either agree or produce
// 4294967293.

// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad \
// RUN:   | npu-translate -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// **The scratchpad total is the arithmetic this file exists to check**, and it
// is checked first because the disassembler prints the header first. The f32
// buffer is 32 elements at four bytes, which is 128, and the i8 buffer is 32
// elements at **one**, which is 32 and fits in the 64 byte aligned span from
// 128 to 160. An allocator that sized the quantized buffer from the f32 element
// size would want 128 bytes there and report 256.
// CHECK: scratchpad 160 bytes

// The argument arrives in DRAM and is loaded whole, which is Section 8's one
// load per DRAM value entering the scratchpad.
// CHECK: 0000  DMA_LOAD sp@0x0 1x2x4x4xf32 <- dram@0x0 1x2x4x4xf32

// The quantization, with both parameters in the instruction, the result i8 and
// the operand f32, and the zero point negative.
// CHECK-NEXT: 0001  QUANT sp@0x80 1x2x4x4xi8 <- sp@0x0 1x2x4x4xf32 scale=0.025 zeroPoint=-3

// The dequantization back, reading the i8 buffer and writing an f32 one. Its
// destination is offset zero, which is the argument buffer's: that buffer is
// dead after the quantization and the allocator reuses it, which is the same
// interval reuse every other program in this suite gets and is not special here.
// CHECK-NEXT: 0002  DEQUANT sp@0x0 1x2x4x4xf32 <- sp@0x80 1x2x4x4xi8 scale=0.025 zeroPoint=-3

// CHECK-NEXT: 0003  DMA_STORE dram@0x80 1x2x4x4xf32 <- sp@0x0 1x2x4x4xf32
// CHECK-NEXT: 0004  HALT

func.func @quantize_round_trip(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %q = npu.quantize %x {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
     : tensor<1x2x4x4xf32> to tensor<1x2x4x4xi8>
  %r = npu.dequantize %q {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
     : tensor<1x2x4x4xi8> to tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}
