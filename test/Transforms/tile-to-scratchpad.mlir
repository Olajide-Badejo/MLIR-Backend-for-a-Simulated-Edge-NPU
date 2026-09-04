// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-tile-to-scratchpad`, Section 13.2.
//
// Spilling a whole tensor to DRAM and reloading it is one response to a
// scratchpad that is too small; splitting the operation so each piece fits is
// the other, and it is the one that matters for an edge accelerator.
//
// **The positive and the negative are the same IR at two budgets**, which is a
// stronger pair than two different functions would be: the pass either fires or
// does not, and the only thing that differs between the two runs is the number
// it is measured against. A pass that fired unconditionally would fail the
// `ROOMY` prefix, and a pass whose trigger never fired would fail the default
// one.
//
// The tile shapes themselves are covered by
// `unittests/Dialect/NPU/TilingInterfaceTest.cpp`, including the property that
// every output position of every tile reads the same input positions it read
// untiled. What is pinned here is the policy: which operations the pass picks
// up, what it records about its choice, and that no `scf` operation survives it.

// RUN: npu-opt %s --npu-tile-to-scratchpad=budget=2048 | FileCheck %s
// RUN: npu-opt %s --npu-tile-to-scratchpad=budget=1048576 \
// RUN:   | FileCheck %s --check-prefix=ROOMY
// RUN: npu-opt %s --npu-tile-to-scratchpad=budget=2048 \
// RUN:   | FileCheck %s --check-prefix=NOSCF

// -----------------------------------------------------------------------------
// Positive: a convolution whose working set is 4256 bytes against a budget of
// 2048 is split, and the tiles carry the halo and their own pads.
//
//   input   1x4x8x8 f32 = 1024
//   filter  8x4x3x3 f32 = 1152
//   bias    8       f32 =   32
//   output  1x8x8x8 f32 = 2048
//
// The exhaustive search chooses four output channels by four rows, leaving the
// width whole: four tiles of 1872 bytes, at a makespan of 1704 cycles.
//
// **The pads are the point of this check.** The first tile of the row axis
// keeps the leading pad and not the trailing one, the last keeps the trailing
// and not the leading, and both keep the operation's column pads whole because
// the column axis was not split. Each tile's input slice is 5 rows for a 4 row
// output under a 3 tap kernel, which is the halo. A pass that copied the
// operation's pads onto every tile would produce tiles of the right shape and
// the wrong values, which is exactly the failure P1 declined to risk when it
// left this arithmetic to the phase that consumes the interface.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @convolution
// CHECK: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 4, 5, 8]
// CHECK: npu.conv2d
// CHECK-SAME: spatial_factors = array<i64: 16, 16>
// CHECK-SAME: temporal_tiles = array<i64: 1, 1, 4, 4, 8>
// CHECK-SAME: tile_bytes = 1872
// CHECK-SAME: tile_count = 4
// CHECK-SAME: pads = array<i64: 1, 1, 0, 1>
// CHECK: tensor.insert_slice
// The second row tile reads from row 3, keeps the trailing row pad and drops
// the leading one, and keeps both column pads.
// CHECK: tensor.extract_slice %arg0[0, 0, 3, 0] [1, 4, 5, 8]
// CHECK: npu.conv2d
// CHECK-SAME: pads = array<i64: 0, 1, 1, 1>

// ROOMY-LABEL: func.func @convolution
// ROOMY-NOT: tensor.extract_slice
// ROOMY-NOT: npu.tiling_choice
// ROOMY: npu.conv2d ins(%arg0, %arg1, %arg2

// NOSCF-LABEL: func.func @convolution
// NOSCF-NOT: scf.

func.func @convolution(%x: tensor<1x4x8x8xf32>, %w: tensor<8x4x3x3xf32>,
                       %b: tensor<8xf32>, %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  %0 = npu.conv2d ins(%x, %w, %b : tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>,
                                   tensor<8xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----------------------------------------------------------------------------
// Positive: a matmul splits over both parallel axes and never over K.
//
// **The left operand keeps every column in every tile.** That is the one thing
// to look at here: a tile whose left operand had lost columns would be
// computing a partial sum, and under fp32 the sum of the partials is not the
// partial of the sum. Section 13.2 permits that split only behind
// `allow-reduction-tiling` with its own golden set, and this pass does not
// offer it.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @matmul
// CHECK: tensor.extract_slice %arg0[0, 0] [2, 64]
// CHECK: tensor.extract_slice %arg1[0, 0] [64, 4]
// CHECK: npu.matmul
// CHECK-SAME: temporal_tiles = array<i64: 2, 1, 4, 1, 1>
// CHECK-SAME: tile_count = 64
// CHECK: tensor.insert_slice

// ROOMY-LABEL: func.func @matmul
// ROOMY-NOT: tensor.extract_slice
// ROOMY: npu.matmul ins(%arg0, %arg1

// NOSCF-LABEL: func.func @matmul
// NOSCF-NOT: scf.

func.func @matmul(%a: tensor<16x64xf32>, %b: tensor<64x32xf32>,
                  %d: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %0 = npu.matmul ins(%a, %b : tensor<16x64xf32>, tensor<64x32xf32>)
                  outs(%d : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
