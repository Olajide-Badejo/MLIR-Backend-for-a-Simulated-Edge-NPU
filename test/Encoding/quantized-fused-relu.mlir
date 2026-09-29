// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// A quantized matrix multiplication and a quantized convolution carrying the
// fused relu, from the instruction level to the bytes and back out through the
// disassembler.
//
// `relu` on `npuisa.matmul` and `npuisa.conv2d` is the `activation` field the
// two opcodes have carried since version 1 of the format, and until the QDQ
// contraction fused a relu into an integer instruction nothing set it. This
// pins that the attribute reaches the field, that an instruction without it
// still encodes `activation=none` and so every program written before it
// encodes to the bytes it always did, and that the round trip through the
// parser keeps it.

// RUN: npu-translate %s -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s
// RUN: npu-opt %s | npu-opt | FileCheck %s --check-prefix=ROUNDTRIP

// The scratchpad layout, every view 64 byte aligned:
//
//   a    4x8     i8    32 bytes at 0
//   r    8x3     i8    24 bytes at 64
//   mb   3       i32   12 bytes at 128
//   md   4x3     i8    12 bytes at 192
//   ne   4x3     i8    12 bytes at 256
//   x    1x2x4x4 i8    32 bytes at 320
//   w    2x2x3x3 i8    36 bytes at 384
//   b    2       i32    8 bytes at 448
//   cd   1x2x4x4 i8    32 bytes at 512
//
// CHECK: ; scratchpad 576 bytes

func.func @fused_relu(
    %ad: memref<4x8xi8, #npu.dram> {npuisa.arg = "in"},
    %xd: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "in"},
    %mout: memref<4x3xi8, #npu.dram> {npuisa.arg = "out"},
    %nout: memref<4x3xi8, #npu.dram> {npuisa.arg = "out"},
    %cout: memref<1x2x4x4xi8, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 576 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<576xi8, #npu.scratchpad>

  %rc = npuisa.const dense<2> : tensor<8x3xi8> -> memref<8x3xi8, #npu.dram>
  %mbc = npuisa.const dense<50> : tensor<3xi32> -> memref<3xi32, #npu.dram>
  %wc = npuisa.const dense<1> : tensor<2x2x3x3xi8>
      -> memref<2x2x3x3xi8, #npu.dram>
  %bc = npuisa.const dense<100> : tensor<2xi32> -> memref<2xi32, #npu.dram>

  %c0 = arith.constant 0 : index
  %a = memref.view %arena[%c0][]
     : memref<576xi8, #npu.scratchpad> to memref<4x8xi8, #npu.scratchpad>
  %c64 = arith.constant 64 : index
  %r = memref.view %arena[%c64][]
     : memref<576xi8, #npu.scratchpad> to memref<8x3xi8, #npu.scratchpad>
  %c128 = arith.constant 128 : index
  %mb = memref.view %arena[%c128][]
      : memref<576xi8, #npu.scratchpad> to memref<3xi32, #npu.scratchpad>
  %c192 = arith.constant 192 : index
  %md = memref.view %arena[%c192][]
      : memref<576xi8, #npu.scratchpad> to memref<4x3xi8, #npu.scratchpad>
  %c256 = arith.constant 256 : index
  %ne = memref.view %arena[%c256][]
      : memref<576xi8, #npu.scratchpad> to memref<4x3xi8, #npu.scratchpad>
  %c320 = arith.constant 320 : index
  %x = memref.view %arena[%c320][]
     : memref<576xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>
  %c384 = arith.constant 384 : index
  %w = memref.view %arena[%c384][]
     : memref<576xi8, #npu.scratchpad> to memref<2x2x3x3xi8, #npu.scratchpad>
  %c448 = arith.constant 448 : index
  %b = memref.view %arena[%c448][]
     : memref<576xi8, #npu.scratchpad> to memref<2xi32, #npu.scratchpad>
  %c512 = arith.constant 512 : index
  %cd = memref.view %arena[%c512][]
      : memref<576xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.scratchpad>

  npuisa.dma_load %ad, %a
    : memref<4x8xi8, #npu.dram> to memref<4x8xi8, #npu.scratchpad>
  npuisa.dma_load %rc, %r
    : memref<8x3xi8, #npu.dram> to memref<8x3xi8, #npu.scratchpad>
  npuisa.dma_load %mbc, %mb
    : memref<3xi32, #npu.dram> to memref<3xi32, #npu.scratchpad>

  // CHECK: MATMUL sp@0xc0 4x3xi8 <- sp@0x0 4x8xi8, sp@0x40 8x3xi8 bias sp@0x80 3xi32 activation=relu requantMultiplier=2147483647 requantShift=3 outputZeroPoint=-7
  // ROUNDTRIP: npuisa.matmul
  // ROUNDTRIP-SAME: relu
  npuisa.matmul ins(%a, %r, %mb : memref<4x8xi8, #npu.scratchpad>,
                                  memref<8x3xi8, #npu.scratchpad>,
                                  memref<3xi32, #npu.scratchpad>)
                outs(%md : memref<4x3xi8, #npu.scratchpad>)
                {output_zero_point = -7 : i32, relu,
                 requant_multiplier = 2147483647 : i32, requant_shift = 3 : i32}

  // The same instruction without the attribute is the one every program before
  // the fusion carried, and it still says none.
  // CHECK: MATMUL sp@0x100 4x3xi8 <- sp@0x0 4x8xi8, sp@0x40 8x3xi8 bias sp@0x80 3xi32 activation=none requantMultiplier=2147483647 requantShift=3 outputZeroPoint=-7
  // ROUNDTRIP: npuisa.matmul
  // ROUNDTRIP-NOT: relu
  // ROUNDTRIP: npuisa.dma_store
  npuisa.matmul ins(%a, %r, %mb : memref<4x8xi8, #npu.scratchpad>,
                                  memref<8x3xi8, #npu.scratchpad>,
                                  memref<3xi32, #npu.scratchpad>)
                outs(%ne : memref<4x3xi8, #npu.scratchpad>)
                {output_zero_point = -7 : i32,
                 requant_multiplier = 2147483647 : i32, requant_shift = 3 : i32}

  npuisa.dma_store %md, %mout
    : memref<4x3xi8, #npu.scratchpad> to memref<4x3xi8, #npu.dram>
  npuisa.dma_store %ne, %nout
    : memref<4x3xi8, #npu.scratchpad> to memref<4x3xi8, #npu.dram>

  npuisa.dma_load %xd, %x
    : memref<1x2x4x4xi8, #npu.dram> to memref<1x2x4x4xi8, #npu.scratchpad>
  npuisa.dma_load %wc, %w
    : memref<2x2x3x3xi8, #npu.dram> to memref<2x2x3x3xi8, #npu.scratchpad>
  npuisa.dma_load %bc, %b
    : memref<2xi32, #npu.dram> to memref<2xi32, #npu.scratchpad>

  // CHECK: CONV2D sp@0x200 1x2x4x4xi8 <- sp@0x140 1x2x4x4xi8, sp@0x180 2x2x3x3xi8 bias sp@0x1c0 2xi32 strides=[1,1] pads=[1,1,1,1] dilations=[1,1] group=1 activation=relu zeroPoint=-11 requantMultiplier=1073741824 requantShift=7 outputZeroPoint=12
  // ROUNDTRIP: npuisa.conv2d
  // ROUNDTRIP-SAME: relu
  npuisa.conv2d ins(%x, %w, %b : memref<1x2x4x4xi8, #npu.scratchpad>,
                                 memref<2x2x3x3xi8, #npu.scratchpad>,
                                 memref<2xi32, #npu.scratchpad>)
                outs(%cd : memref<1x2x4x4xi8, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64,
                 zero_point = -11 : i32, output_zero_point = 12 : i32,
                 requant_multiplier = 1073741824 : i32, requant_shift = 7 : i32,
                 relu}

  npuisa.dma_store %cd, %cout
    : memref<1x2x4x4xi8, #npu.scratchpad> to memref<1x2x4x4xi8, #npu.dram>
  return
}
