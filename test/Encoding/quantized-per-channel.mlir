// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A convolution that rescales per output channel, from the instruction level
// to the bytes and back out through the disassembler.
//
// `quantized-compute.mlir` is the per tensor arm: the same two instructions
// rescaling with the scalar pair the record has carried since version 1. This
// file is the other arm, and Section 14's granularity ablation compares them,
// so neither is a legacy of the other and both are kept.
//
// **What it pins that nothing else does.** That the encoder writes the fourth
// operand at all, and writes it **after the bias**, which is the order the
// opcode declares and the order the kernels read. That the operand survives as
// an i32 buffer of shape (2, C) rather than being flattened on the way. And
// that `npu-objdump` shows it, so the arithmetic of a per channel instruction
// can be read off a disassembly the way the per tensor one can.
//
// It is a separate file rather than two more instructions in the per tensor
// one, because a `.nbin` holds one program and every offset in that file is
// computed in its own comments; adding a fifth buffer there would have meant
// recomputing all of them to assert something they do not assert.

// RUN: npu-translate %s -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// The scratchpad layout, every view 64 byte aligned:
//
//   x    1x2x4x4 i8   32 bytes at 0
//   w    2x2x3x3 i8   36 bytes at 64
//   b    2       i32   8 bytes at 128
//   r    2x2     i32  16 bytes at 192
//   d    1x2x4x4 i8   32 bytes at 256
//
// CHECK: ; scratchpad 320 bytes, dram 320 bytes
// CHECK: ; 1 inputs, 1 outputs, 3 constants, 0 spill slots, 7 instructions

// The DRAM map: the input, then the output, then the three constants, each
// aligned to 64. The rescale table is 16 bytes, which is two rows of two i32.
// CHECK: ;   input 0: dram@0x0 1x2x4x4xi8 (32 bytes)
// CHECK: ;   output 0: dram@0x40 1x2x4x4xi8 (32 bytes)
// CHECK: ;   constant 0: dram@0x80 2x2x3x3xi8 (36 bytes), 36 bytes of data
// CHECK: ;   constant 1: dram@0xc0 2xi32 (8 bytes), 8 bytes of data
// CHECK: ;   constant 2: dram@0x100 2x2xi32 (16 bytes), 16 bytes of data

func.func @per_channel_convolution(
    %xd: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "in"},
    %od: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 320 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<320xi8, #npu.scratchpad>

  %wc = npuisa.const dense<1> : tensor<2x2x3x3xi8>
      -> memref<2x2x3x3xi8, #npu.dram>
  %bc = npuisa.const dense<100> : tensor<2xi32>
      -> memref<2xi32, #npu.dram>
  // Row 0 is one multiplier per output channel and row 1 is one shift. Both
  // channels take 2^30 here and the shifts differ, which is the smallest table
  // that is not the same as a scalar pair.
  %rc = npuisa.const dense<[[1073741824, 1073741824], [0, 1]]> : tensor<2x2xi32>
      -> memref<2x2xi32, #npu.dram>

  %c0 = arith.constant 0 : index
  %x = memref.view %arena[%c0][]
     : memref<320xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>
  %c64 = arith.constant 64 : index
  %w = memref.view %arena[%c64][]
     : memref<320xi8, #npu.scratchpad> to memref<2x2x3x3xi8, #npu.scratchpad>
  %c128 = arith.constant 128 : index
  %b = memref.view %arena[%c128][]
     : memref<320xi8, #npu.scratchpad> to memref<2xi32, #npu.scratchpad>
  %c192 = arith.constant 192 : index
  %r = memref.view %arena[%c192][]
     : memref<320xi8, #npu.scratchpad> to memref<2x2xi32, #npu.scratchpad>
  %c256 = arith.constant 256 : index
  %d = memref.view %arena[%c256][]
     : memref<320xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>

  // CHECK: 0000  DMA_LOAD sp@0x0 1x2x4x4xi8 <- dram@0x0 1x2x4x4xi8
  npuisa.dma_load %xd, %x
    : memref<1x2x4x4xi8, #npu.dram> to memref<1x2x4x4xi8, #npu.scratchpad>
  // CHECK-NEXT: 0001  DMA_LOAD sp@0x40 2x2x3x3xi8 <- dram@0x80 2x2x3x3xi8
  npuisa.dma_load %wc, %w
    : memref<2x2x3x3xi8, #npu.dram> to memref<2x2x3x3xi8, #npu.scratchpad>
  // CHECK-NEXT: 0002  DMA_LOAD sp@0x80 2xi32 <- dram@0xc0 2xi32
  npuisa.dma_load %bc, %b
    : memref<2xi32, #npu.dram> to memref<2xi32, #npu.scratchpad>
  // The table arrives the way every other constant does, which is the whole
  // argument for making it an operand rather than a field.
  // CHECK-NEXT: 0003  DMA_LOAD sp@0xc0 2x2xi32 <- dram@0x100 2x2xi32
  npuisa.dma_load %rc, %r
    : memref<2x2xi32, #npu.dram> to memref<2x2xi32, #npu.scratchpad>

  // The rescale prints after the bias and before the window fields, and the
  // scalar pair still prints too: an instruction carrying the table leaves the
  // fields at whatever it was given, and the machine reads the table.
  // CHECK-NEXT: 0004  CONV2D sp@0x100 1x2x4x4xi8 <- sp@0x0 1x2x4x4xi8, sp@0x40 2x2x3x3xi8 bias sp@0x80 2xi32 rescale sp@0xc0 2x2xi32 strides=[1,1] pads=[1,1,1,1] dilations=[1,1] group=1 activation=none zeroPoint=-11 requantMultiplier=1073741824 requantShift=7 outputZeroPoint=12
  npuisa.conv2d ins(%x, %w, %b, %r : memref<1x2x4x4xi8, #npu.scratchpad>,
                                     memref<2x2x3x3xi8, #npu.scratchpad>,
                                     memref<2xi32, #npu.scratchpad>,
                                     memref<2x2xi32, #npu.scratchpad>)
                outs(%d : memref<1x2x4x4xi8, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64,
                 zero_point = -11 : i32, output_zero_point = 12 : i32,
                 requant_multiplier = 1073741824 : i32, requant_shift = 7 : i32}

  // CHECK-NEXT: 0005  DMA_STORE dram@0x40 1x2x4x4xi8 <- sp@0x100 1x2x4x4xi8
  npuisa.dma_store %d, %od
    : memref<1x2x4x4xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.dram>
  // CHECK-NEXT: 0006  HALT
  return
}
