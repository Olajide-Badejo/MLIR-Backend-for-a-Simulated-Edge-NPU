// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-tile-to-scratchpad` declines a calibrated operation, which is D-0069's
// fix.
//
// A tile of an operation that carries `weight_scales` reads a slice of the
// dequantized input rather than the dequantize, so the QDQ contraction in the
// lowering could not turn it into an integer instruction, and the tile's copy
// of the whole operation's scales would be the wrong count for a split output
// channel axis. Before the fix the pass tiled it anyway and the verifier then
// refused the tile, with a sentence saying the input was not a quantized
// compilation when it was one: quantized compilation at `-O2` and a tight
// budget failed on all seven models. Now the pass leaves the operation whole,
// counts it as declined, and says why, the way it declines a fused region over
// the budget; the allocator's spilling is the fallback. Tiling an integer
// operation properly is Checkpoint C's, with reduction tiling under INT8.
//
// The function is `tile-to-scratchpad-declines.mlir`'s `@nothing_fits` with its
// input dequantized and its result quantized, at the budget where that file
// shows the fp32 operation splitting 32 ways. The fp32 twin below it still
// does, which is the check that the decline is the attribute's and not the
// budget's.

// RUN: npu-opt %s --npu-tile-to-scratchpad='budget=768 halo=recompute' 2>&1 \
// RUN:   | FileCheck %s
// RUN: npu-opt %s --npu-tile-to-scratchpad='budget=768 halo=recompute' \
// RUN:   --mlir-pass-statistics 2>&1 >/dev/null \
// RUN:   | FileCheck %s --check-prefix=STATS

// CHECK: remark: working set of 4224 bytes exceeds the 768 byte budget, and this operation is calibrated
// CHECK-SAME: Checkpoint C
// CHECK-LABEL: func.func @calibrated
// CHECK-NOT: tensor.extract_slice
// CHECK: npu.conv2d
// CHECK-SAME: weight_scales
// CHECK-NOT: tensor.extract_slice
// CHECK: return

// CHECK-LABEL: func.func @fp32_twin
// CHECK: tensor.extract_slice
// CHECK: npu.conv2d
// CHECK-SAME: tile_count = 32

// STATS-DAG: 1 declined
// STATS-DAG: 1 tiled-ops
// STATS-DAG: 32 tiles-emitted

func.func @calibrated(%q: tensor<1x4x8x8xi8>, %w: tensor<8x4x3x3xf32>,
                      %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xi8> {
  %x = npu.dequantize %q {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
       : tensor<1x4x8x8xi8> to tensor<1x4x8x8xf32>
  %0 = npu.conv2d ins(%x, %w : tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 0.01, 0.01, 0.01, 0.01,
                                              0.02, 0.02, 0.02, 0.02>}
       -> tensor<1x8x8x8xf32>
  %qy = npu.quantize %0 {scale = 5.000000e-02 : f32, zero_point = 4 : i32}
        : tensor<1x8x8x8xf32> to tensor<1x8x8x8xi8>
  return %qy : tensor<1x8x8x8xi8>
}

func.func @fp32_twin(%x: tensor<1x4x8x8xf32>, %w: tensor<8x4x3x3xf32>,
                     %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %0 = npu.conv2d ins(%x, %w : tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}
