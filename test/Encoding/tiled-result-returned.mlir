// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The one tiled shape this binary format can express, encoded end to end.
//
// **`tiled-assembly-in-scratchpad.mlir` is the refusal and this is the
// permission**, and the pair is the whole of D-0052 made executable. Checks 8
// and 9, `operand-defined` and `operand-extent`, satisfy a read out of a
// **single** written span, and `WrittenSpans` deliberately does not merge
// adjacent spans, so a buffer written by N stores cannot be read by anything
// covering more than one of them. A tiled result is written by one store per
// tile, so **an assembled value that any instruction reads is refused**.
//
// **The shape that is left is an assembly nothing reads.** A `tensor.empty`
// whose insert chain reaches `func.return` is mapped straight to the out
// parameter the function gained for that result, the tiles are stored into the
// output region, and an output region is never read back. That is what this
// file compiles, encodes and disassembles.
//
// It starts at the tensor level and goes through the whole level rather than
// starting at hand written `npuisa`, which is the opposite of `all_ops.mlir`'s
// choice and is deliberate: what is under test here is the **agreement**
// between the tiling pass, the lowering and the encoder, and a hand written
// program would assert only the last of the three.

// RUN: npu-opt %s --pass-pipeline='builtin.module(npu-O2{budget=6464})' \
// RUN:   | npu-translate -o %t.nbin
// RUN: npu-objdump %t.nbin | FileCheck %s

// The convolution's working set is 2048 bytes of input, 2304 of filter and 2048
// of output, which is 6400 against the 6464 byte budget, and the prefetch is
// what puts it over: the pipeline tells the tiling search that
// `-npu-double-buffer` is in it, per Section 13.2. The search splits the output
// rows in two.

// One instruction per tile, and one store per tile into the out parameter.
// CHECK: CONV2D
// CHECK: DMA_STORE
// CHECK: CONV2D
// CHECK: DMA_STORE

// **And nothing loads the assembly back**, which is the property the whole file
// exists to pin. A `DMA_LOAD` whose source is the output region would be the
// read that checks 8 and 9 refuse, and `npu-translate` would have refused this
// program rather than writing it.
// CHECK-NOT: DMA_LOAD dram

func.func @tiled_conv_is_returned(%x: tensor<1x8x8x8xf32>,
                                  %w: tensor<8x8x3x3xf32>)
    -> tensor<1x8x8x8xf32> {
  %d = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %c : tensor<1x8x8x8xf32>
}
