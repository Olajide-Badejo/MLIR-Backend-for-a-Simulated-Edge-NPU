// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Every instruction the compiler can emit, encoded, in one program.
//
// Three opcodes are absent and each is absent for a reason the ISA description
// records. `NOP` and `HALT` are not compiler concepts: nothing in the IR ever
// says "do nothing" or "stop", and the encoder emits the `HALT` at the end of
// this program itself. `QUANT` and `DEQUANT` have no dialect operation until
// P14. All four are covered structurally by `unittests/Encoding/PropertyTest`,
// which builds them directly rather than through the IR, and that is all the
// coverage Section 17.2 asks for at this phase.
//
// This file is hand written **allocated** `npuisa` IR rather than `npu` tensor
// IR piped through the pipeline, and the reason is coverage rather than
// preference. The lowering never emits the asynchronous DMA forms, so a file
// that started at the tensor level could not reach `dma_load_async`,
// `dma_store_async` or `await` at all, and those three are exactly the ones
// whose encoding has something to say: they encode to the synchronous opcodes
// and the `await` encodes to nothing.
//
// It is one function because a `.nbin` holds one program and `npu-translate`
// diagnoses a module with two rather than truncating it. `diagnostics.mlir`
// carries that case.
//
// Every offset below is a multiple of 64, which is what the allocator produces,
// and every buffer is written before it is read, which is what the
// `operand-defined` check requires. The arena is 12800 bytes and the arithmetic
// is written out beside each view.

// RUN: npu-translate %s -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// The DRAM map the encoder assigns: inputs, then outputs, then constants, each
// aligned to 64 bytes.
//
//   arg0     1x3x8x8   768 bytes at 0
//   arg1     1x8x4x4   512 bytes at 768
//   arg2     1x16x4x4 1024 bytes at 1280
//   arg3     16x10     640 bytes at 2304
//   filter   8x3x3x3   864 bytes at 2944
//   bias     8          32 bytes at 3840
//   rhs      16x10     640 bytes at 3904
//   rhs bias 10         40 bytes at 4544
//
// CHECK: ; scratchpad 12800 bytes, dram 4608 bytes
// CHECK: ; 2 inputs, 2 outputs, 4 constants, 0 spill slots, 19 instructions
// CHECK: ;   input 0: dram@0x0 1x3x8x8xf32 (768 bytes)
// CHECK: ;   input 1: dram@0x300 1x8x4x4xf32 (512 bytes)
// CHECK: ;   output 0: dram@0x500 1x16x4x4xf32 (1024 bytes)
// CHECK: ;   output 1: dram@0x900 16x10xf32 (640 bytes)
// CHECK: ;   constant 0: dram@0xb80 8x3x3x3xf32 (864 bytes), 864 bytes of data
// CHECK: ;   constant 1: dram@0xf00 8xf32 (32 bytes), 32 bytes of data
// CHECK: ;   constant 2: dram@0xf40 16x10xf32 (640 bytes), 640 bytes of data
// CHECK: ;   constant 3: dram@0x11c0 10xf32 (40 bytes), 40 bytes of data

func.func @every_instruction(
    %in: memref<1x3x8x8xf32, #npu.dram> {npuisa.arg = "in"},
    %side: memref<1x8x4x4xf32, #npu.dram> {npuisa.arg = "in"},
    %out0: memref<1x16x4x4xf32, #npu.dram> {npuisa.arg = "out"},
    %out1: memref<16x10xf32, #npu.dram> {npuisa.arg = "out"})
    attributes {npuisa.scratchpad_bytes = 12800 : i64} {
  %arena = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<12800xi8, #npu.scratchpad>

  // 1x3x8x8 f32 is 192 elements, 768 bytes.
  %c0 = arith.constant 0 : index
  %x = memref.view %arena[%c0][]
     : memref<12800xi8, #npu.scratchpad> to memref<1x3x8x8xf32, #npu.scratchpad>
  // 1x8x4x4 f32 is 128 elements, 512 bytes. 0 + 768 is 768.
  %c768 = arith.constant 768 : index
  %s = memref.view %arena[%c768][]
     : memref<12800xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  // 8x3x3x3 f32 is 216 elements, 864 bytes. 768 + 512 is 1280.
  %c1280 = arith.constant 1280 : index
  %w = memref.view %arena[%c1280][]
     : memref<12800xi8, #npu.scratchpad> to memref<8x3x3x3xf32, #npu.scratchpad>
  // 8 f32 is 32 bytes. 1280 + 864 is 2144, which rounds up to 2176.
  %c2176 = arith.constant 2176 : index
  %b = memref.view %arena[%c2176][]
     : memref<12800xi8, #npu.scratchpad> to memref<8xf32, #npu.scratchpad>
  // 1x8x8x8 f32 is 512 elements, 2048 bytes. 2176 + 32 is 2208, up to 2240.
  %c2240 = arith.constant 2240 : index
  %conv = memref.view %arena[%c2240][]
        : memref<12800xi8, #npu.scratchpad> to memref<1x8x8x8xf32, #npu.scratchpad>
  // 2240 + 2048 is 4288.
  %c4288 = arith.constant 4288 : index
  %act = memref.view %arena[%c4288][]
       : memref<12800xi8, #npu.scratchpad> to memref<1x8x8x8xf32, #npu.scratchpad>
  // 4288 + 2048 is 6336.
  %c6336 = arith.constant 6336 : index
  %pmax = memref.view %arena[%c6336][]
        : memref<12800xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  // 6336 + 512 is 6848.
  %c6848 = arith.constant 6848 : index
  %pavg = memref.view %arena[%c6848][]
        : memref<12800xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  // 6848 + 512 is 7360.
  %c7360 = arith.constant 7360 : index
  %sum = memref.view %arena[%c7360][]
       : memref<12800xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  // 7360 + 512 is 7872.
  %c7872 = arith.constant 7872 : index
  %prod = memref.view %arena[%c7872][]
        : memref<12800xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  // 1x16x4x4 f32 is 256 elements, 1024 bytes. 7872 + 512 is 8384.
  %c8384 = arith.constant 8384 : index
  %cat = memref.view %arena[%c8384][]
       : memref<12800xi8, #npu.scratchpad> to memref<1x16x4x4xf32, #npu.scratchpad>
  // 16x16 f32 is 256 elements, 1024 bytes. 8384 + 1024 is 9408.
  %c9408 = arith.constant 9408 : index
  %flat = memref.view %arena[%c9408][]
        : memref<12800xi8, #npu.scratchpad> to memref<16x16xf32, #npu.scratchpad>
  // 9408 + 1024 is 10432.
  %c10432 = arith.constant 10432 : index
  %flatT = memref.view %arena[%c10432][]
         : memref<12800xi8, #npu.scratchpad> to memref<16x16xf32, #npu.scratchpad>
  // 16x10 f32 is 160 elements, 640 bytes. 10432 + 1024 is 11456.
  %c11456 = arith.constant 11456 : index
  %rhs = memref.view %arena[%c11456][]
       : memref<12800xi8, #npu.scratchpad> to memref<16x10xf32, #npu.scratchpad>
  // 10 f32 is 40 bytes. 11456 + 640 is 12096.
  %c12096 = arith.constant 12096 : index
  %rbias = memref.view %arena[%c12096][]
         : memref<12800xi8, #npu.scratchpad> to memref<10xf32, #npu.scratchpad>
  // 12096 + 40 is 12136, which rounds up to 12160. 12160 + 640 is 12800, which
  // is the arena.
  %c12160 = arith.constant 12160 : index
  %mm = memref.view %arena[%c12160][]
      : memref<12800xi8, #npu.scratchpad> to memref<16x10xf32, #npu.scratchpad>

  %cw = npuisa.const dense<1.000000e+00> : tensor<8x3x3x3xf32>
      -> memref<8x3x3x3xf32, #npu.dram>
  %cb = npuisa.const dense<0.000000e+00> : tensor<8xf32>
      -> memref<8xf32, #npu.dram>
  %crhs = npuisa.const dense<2.000000e+00> : tensor<16x10xf32>
        -> memref<16x10xf32, #npu.dram>
  %crbias = npuisa.const dense<3.000000e+00> : tensor<10xf32>
          -> memref<10xf32, #npu.dram>

  // The asynchronous load encodes to DMA_LOAD, and the plain load in between is
  // the work the transfer is overlapped with. It writes a different buffer, so
  // the overlap rule is satisfied by disjointness rather than by adjacency.
  // CHECK: 0000  DMA_LOAD sp@0x300 1x8x4x4xf32 <- dram@0x300 1x8x4x4xf32
  %t0 = npuisa.dma_load_async %side, %s
      : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  // CHECK-NEXT: 0001  DMA_LOAD sp@0x0 1x3x8x8xf32 <- dram@0x0 1x3x8x8xf32
  npuisa.dma_load %in, %x
    : memref<1x3x8x8xf32, #npu.dram> to memref<1x3x8x8xf32, #npu.scratchpad>
  // The await encodes to nothing at all, which is why the next line is 0002.
  npuisa.await %t0

  // CHECK-NEXT: 0002  DMA_LOAD sp@0x500 8x3x3x3xf32 <- dram@0xb80 8x3x3x3xf32
  npuisa.dma_load %cw, %w
    : memref<8x3x3x3xf32, #npu.dram> to memref<8x3x3x3xf32, #npu.scratchpad>
  // CHECK-NEXT: 0003  DMA_LOAD sp@0x880 8xf32 <- dram@0xf00 8xf32
  npuisa.dma_load %cb, %b
    : memref<8xf32, #npu.dram> to memref<8xf32, #npu.scratchpad>

  // CHECK-NEXT: 0004  CONV2D sp@0x8c0 1x8x8x8xf32 <- sp@0x0 1x3x8x8xf32, sp@0x500 8x3x3x3xf32 bias sp@0x880 8xf32 strides=[1,1] pads=[1,1,1,1] dilations=[1,1] group=1 activation=none
  npuisa.conv2d ins(%x, %w, %b : memref<1x3x8x8xf32, #npu.scratchpad>,
                                 memref<8x3x3x3xf32, #npu.scratchpad>,
                                 memref<8xf32, #npu.scratchpad>)
                outs(%conv : memref<1x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}

  // CHECK-NEXT: 0005  RELU sp@0x10c0 1x8x8x8xf32 <- sp@0x8c0 1x8x8x8xf32
  npuisa.relu ins(%conv : memref<1x8x8x8xf32, #npu.scratchpad>)
              outs(%act : memref<1x8x8x8xf32, #npu.scratchpad>)

  // CHECK-NEXT: 0006  POOL_MAX sp@0x18c0 1x8x4x4xf32 <- sp@0x10c0 1x8x8x8xf32 kernel=[2,2] strides=[2,2] pads=[0,0,0,0] dilations=[1,1]
  npuisa.pool_max ins(%act : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%pmax : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  // CHECK-NEXT: 0007  POOL_AVG sp@0x1ac0 1x8x4x4xf32 <- sp@0x10c0 1x8x8x8xf32 kernel=[2,2] strides=[2,2] pads=[0,0,0,0] dilations=[1,1]
  npuisa.pool_avg ins(%act : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%pavg : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}

  // CHECK-NEXT: 0008  ADD sp@0x1cc0 1x8x4x4xf32 <- sp@0x18c0 1x8x4x4xf32, sp@0x300 1x8x4x4xf32 activation=none
  npuisa.add ins(%pmax, %s : memref<1x8x4x4xf32, #npu.scratchpad>,
                             memref<1x8x4x4xf32, #npu.scratchpad>)
             outs(%sum : memref<1x8x4x4xf32, #npu.scratchpad>)
  // CHECK-NEXT: 0009  MUL sp@0x1ec0 1x8x4x4xf32 <- sp@0x1ac0 1x8x4x4xf32, sp@0x300 1x8x4x4xf32 activation=none
  npuisa.mul ins(%pavg, %s : memref<1x8x4x4xf32, #npu.scratchpad>,
                             memref<1x8x4x4xf32, #npu.scratchpad>)
             outs(%prod : memref<1x8x4x4xf32, #npu.scratchpad>)

  // CHECK-NEXT: 0010  CONCAT sp@0x20c0 1x16x4x4xf32 <- sp@0x1cc0 1x8x4x4xf32, sp@0x1ec0 1x8x4x4xf32 axes=[1]
  npuisa.concat ins(%sum, %prod : memref<1x8x4x4xf32, #npu.scratchpad>,
                                  memref<1x8x4x4xf32, #npu.scratchpad>)
                outs(%cat : memref<1x16x4x4xf32, #npu.scratchpad>)
                {axis = 1 : i64}

  // CHECK-NEXT: 0011  RESHAPE sp@0x24c0 16x16xf32 <- sp@0x20c0 1x16x4x4xf32
  npuisa.reshape ins(%cat : memref<1x16x4x4xf32, #npu.scratchpad>)
                 outs(%flat : memref<16x16xf32, #npu.scratchpad>)
  // CHECK-NEXT: 0012  TRANSPOSE sp@0x28c0 16x16xf32 <- sp@0x24c0 16x16xf32 axes=[1,0]
  npuisa.transpose ins(%flat : memref<16x16xf32, #npu.scratchpad>)
                   outs(%flatT : memref<16x16xf32, #npu.scratchpad>)
                   {permutation = array<i64: 1, 0>}

  // CHECK-NEXT: 0013  DMA_LOAD sp@0x2cc0 16x10xf32 <- dram@0xf40 16x10xf32
  npuisa.dma_load %crhs, %rhs
    : memref<16x10xf32, #npu.dram> to memref<16x10xf32, #npu.scratchpad>
  // CHECK-NEXT: 0014  DMA_LOAD sp@0x2f40 10xf32 <- dram@0x11c0 10xf32
  npuisa.dma_load %crbias, %rbias
    : memref<10xf32, #npu.dram> to memref<10xf32, #npu.scratchpad>

  // CHECK-NEXT: 0015  MATMUL sp@0x2f80 16x10xf32 <- sp@0x28c0 16x16xf32, sp@0x2cc0 16x10xf32 bias sp@0x2f40 10xf32 activation=none
  npuisa.matmul ins(%flatT, %rhs, %rbias : memref<16x16xf32, #npu.scratchpad>,
                                           memref<16x10xf32, #npu.scratchpad>,
                                           memref<10xf32, #npu.scratchpad>)
                outs(%mm : memref<16x10xf32, #npu.scratchpad>)

  // The asynchronous store encodes to DMA_STORE, and the plain store in between
  // writes a different output region.
  // CHECK-NEXT: 0016  DMA_STORE dram@0x900 16x10xf32 <- sp@0x2f80 16x10xf32
  %t1 = npuisa.dma_store_async %mm, %out1
      : memref<16x10xf32, #npu.scratchpad> to memref<16x10xf32, #npu.dram>
  // CHECK-NEXT: 0017  DMA_STORE dram@0x500 1x16x4x4xf32 <- sp@0x20c0 1x16x4x4xf32
  npuisa.dma_store %cat, %out0
    : memref<1x16x4x4xf32, #npu.scratchpad> to memref<1x16x4x4xf32, #npu.dram>
  npuisa.await %t1

  // Every program ends with one, emitted by the encoder rather than written
  // here: nothing in the IR ever says "stop", because stopping is a property of
  // the encoding and not of the instruction stream a pass manipulates.
  // CHECK-NEXT: 0018  HALT
  return
}
