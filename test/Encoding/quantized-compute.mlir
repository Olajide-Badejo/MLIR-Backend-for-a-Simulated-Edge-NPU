// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A quantized convolution and a quantized matrix multiplication, from the
// instruction level to the bytes and back out through the disassembler.
//
// `quantized.mlir` is the pair that converts between the two number systems and
// it starts at the tensor level. This file is the arithmetic that happens
// **inside** the integer system, and it is hand written allocated `npuisa` for
// the reason `all_ops.mlir` gives: the tensor level has no quantized compute
// operation yet, so a file that started there could not reach these two
// instructions at all.
//
// **The four things it pins that nothing else does.**
//
// That an `i32` bias can reach the scratchpad. It is an `npuisa.const` in DRAM
// and a `dma_load` like every other constant, and until the transfer
// constraints admitted `i32` the compute instructions could name a bias that
// nothing in the dialect could have filled.
//
// That an `i32` buffer is sized as four bytes an element by every layer that
// sizes one: 2 elements is 8 bytes below and 3 is 12, not 2 and 3.
//
// That the output zero point reaches the binary. It is carried in the `scale`
// word, which an integer compute instruction does not otherwise use because its
// scale is folded into the rescaling pair, and the disassembler prints it as
// the zero point it is rather than as the float it is stored in.
//
// That the disassembly of an integer instruction is the arithmetic in order:
// the input zero point a padding tap contributes, the pair that rescales the
// accumulator, and the output zero point added to what the rescale produced.

// RUN: npu-translate %s -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// The scratchpad layout, every view 64 byte aligned:
//
//   x    1x2x4x4 i8    32 bytes at 0
//   w    2x2x3x3 i8    36 bytes at 64
//   b    2       i32    8 bytes at 128
//   cd   1x2x4x4 i8    32 bytes at 192
//   a    4x8     i8    32 bytes at 256
//   r    8x3     i8    24 bytes at 320
//   mb   3       i32   12 bytes at 384
//   md   4x3     i8    12 bytes at 448
//
// CHECK: ; scratchpad 512 bytes, dram 512 bytes
// CHECK: ; 2 inputs, 2 outputs, 4 constants, 0 spill slots, 11 instructions

// The DRAM map the encoder assigns: inputs, then outputs, then constants, each
// aligned to 64 bytes. The two `i32` constants are the biases, and their byte
// counts are the assertion: 8 for two elements and 12 for three.
// CHECK: ;   input 0: dram@0x0 1x2x4x4xi8 (32 bytes)
// CHECK: ;   input 1: dram@0x40 4x8xi8 (32 bytes)
// CHECK: ;   output 0: dram@0x80 1x2x4x4xi8 (32 bytes)
// CHECK: ;   output 1: dram@0xc0 4x3xi8 (12 bytes)
// CHECK: ;   constant 0: dram@0x100 2x2x3x3xi8 (36 bytes), 36 bytes of data
// CHECK: ;   constant 1: dram@0x140 2xi32 (8 bytes), 8 bytes of data
// CHECK: ;   constant 2: dram@0x180 8x3xi8 (24 bytes), 24 bytes of data
// CHECK: ;   constant 3: dram@0x1c0 3xi32 (12 bytes), 12 bytes of data

func.func @quantized_compute(
    %xd: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "in"},
    %ad: memref<4x8xi8, #npu.dram> {npuisa.arg = "in"},
    %cout: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "out"},
    %mout: memref<4x3xi8, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 512 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<512xi8, #npu.scratchpad>

  // The weights are i8 and the biases are i32, which is the pairing Section 14
  // asks for: the bias is added to the int32 accumulator, so it is the
  // accumulator's own width rather than the data's.
  %wc = npuisa.const dense<1> : tensor<2x2x3x3xi8>
      -> memref<2x2x3x3xi8, #npu.dram>
  %bc = npuisa.const dense<100> : tensor<2xi32>
      -> memref<2xi32, #npu.dram>
  %rc = npuisa.const dense<2> : tensor<8x3xi8>
      -> memref<8x3xi8, #npu.dram>
  %mbc = npuisa.const dense<50> : tensor<3xi32>
       -> memref<3xi32, #npu.dram>

  %c0 = arith.constant 0 : index
  %x = memref.view %arena[%c0][]
     : memref<512xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>
  %c64 = arith.constant 64 : index
  %w = memref.view %arena[%c64][]
     : memref<512xi8, #npu.scratchpad> to memref<2x2x3x3xi8, #npu.scratchpad>
  %c128 = arith.constant 128 : index
  %b = memref.view %arena[%c128][]
     : memref<512xi8, #npu.scratchpad> to memref<2xi32, #npu.scratchpad>
  %c192 = arith.constant 192 : index
  %cd = memref.view %arena[%c192][]
      : memref<512xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>
  %c256 = arith.constant 256 : index
  %a = memref.view %arena[%c256][]
     : memref<512xi8, #npu.scratchpad> to memref<4x8xi8, #npu.scratchpad>
  %c320 = arith.constant 320 : index
  %r = memref.view %arena[%c320][]
     : memref<512xi8, #npu.scratchpad> to memref<8x3xi8, #npu.scratchpad>
  %c384 = arith.constant 384 : index
  %mb = memref.view %arena[%c384][]
      : memref<512xi8, #npu.scratchpad> to memref<3xi32, #npu.scratchpad>
  %c448 = arith.constant 448 : index
  %md = memref.view %arena[%c448][]
      : memref<512xi8, #npu.scratchpad> to memref<4x3xi8, #npu.scratchpad>

  // CHECK: 0000  DMA_LOAD sp@0x0 1x2x4x4xi8 <- dram@0x0 1x2x4x4xi8
  npuisa.dma_load %xd, %x
    : memref<1x2x4x4xi8, #npu.dram> to memref<1x2x4x4xi8, #npu.scratchpad>
  // CHECK-NEXT: 0001  DMA_LOAD sp@0x40 2x2x3x3xi8 <- dram@0x100 2x2x3x3xi8
  npuisa.dma_load %wc, %w
    : memref<2x2x3x3xi8, #npu.dram> to memref<2x2x3x3xi8, #npu.scratchpad>
  // An i32 transfer. A DMA copies bytes and does not interpret them, which is
  // why the transfer constraints are wider than what a multiply may read.
  // CHECK-NEXT: 0002  DMA_LOAD sp@0x80 2xi32 <- dram@0x140 2xi32
  npuisa.dma_load %bc, %b
    : memref<2xi32, #npu.dram> to memref<2xi32, #npu.scratchpad>

  // The convolution carries both zero points and the rescale, and the printed
  // order is the order the machine applies them: a tap outside the input
  // contributes zeroPoint, the pair rescales the accumulator, and
  // outputZeroPoint is added to the result of that rescale.
  // CHECK-NEXT: 0003  CONV2D sp@0xc0 1x2x4x4xi8 <- sp@0x0 1x2x4x4xi8, sp@0x40 2x2x3x3xi8 bias sp@0x80 2xi32 strides=[1,1] pads=[1,1,1,1] dilations=[1,1] group=1 activation=none zeroPoint=-11 requantMultiplier=1073741824 requantShift=7 outputZeroPoint=12
  npuisa.conv2d ins(%x, %w, %b : memref<1x2x4x4xi8, #npu.scratchpad>,
                                 memref<2x2x3x3xi8, #npu.scratchpad>,
                                 memref<2xi32, #npu.scratchpad>)
                outs(%cd : memref<1x2x4x4xi8, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64,
                 zero_point = -11 : i32, output_zero_point = 12 : i32,
                 requant_multiplier = 1073741824 : i32, requant_shift = 7 : i32}

  // CHECK-NEXT: 0004  DMA_STORE dram@0x80 1x2x4x4xi8 <- sp@0xc0 1x2x4x4xi8
  npuisa.dma_store %cd, %cout
    : memref<1x2x4x4xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.dram>

  // CHECK-NEXT: 0005  DMA_LOAD sp@0x100 4x8xi8 <- dram@0x40 4x8xi8
  npuisa.dma_load %ad, %a
    : memref<4x8xi8, #npu.dram> to memref<4x8xi8, #npu.scratchpad>
  // CHECK-NEXT: 0006  DMA_LOAD sp@0x140 8x3xi8 <- dram@0x180 8x3xi8
  npuisa.dma_load %rc, %r
    : memref<8x3xi8, #npu.dram> to memref<8x3xi8, #npu.scratchpad>
  // CHECK-NEXT: 0007  DMA_LOAD sp@0x180 3xi32 <- dram@0x1c0 3xi32
  npuisa.dma_load %mbc, %mb
    : memref<3xi32, #npu.dram> to memref<3xi32, #npu.scratchpad>

  // The matrix multiplication carries one zero point where the convolution
  // carries two, and the difference is padding: every tap of a matrix
  // multiplication is in range, so the input zero point's whole contribution is
  // the compile time term already folded into the bias.
  // CHECK-NEXT: 0008  MATMUL sp@0x1c0 4x3xi8 <- sp@0x100 4x8xi8, sp@0x140 8x3xi8 bias sp@0x180 3xi32 activation=none requantMultiplier=2147483647 requantShift=3 outputZeroPoint=-7
  npuisa.matmul ins(%a, %r, %mb : memref<4x8xi8, #npu.scratchpad>,
                                  memref<8x3xi8, #npu.scratchpad>,
                                  memref<3xi32, #npu.scratchpad>)
                outs(%md : memref<4x3xi8, #npu.scratchpad>)
                {output_zero_point = -7 : i32,
                 requant_multiplier = 2147483647 : i32, requant_shift = 3 : i32}

  // CHECK-NEXT: 0009  DMA_STORE dram@0xc0 4x3xi8 <- sp@0x1c0 4x3xi8
  npuisa.dma_store %md, %mout
    : memref<4x3xi8, #npu.scratchpad> to memref<4x3xi8, #npu.dram>
  // CHECK-NEXT: 0010  HALT
  return
}
