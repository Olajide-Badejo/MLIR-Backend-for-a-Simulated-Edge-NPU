// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The two options that decide what "will fit" means, mistyped.
//
// `-npu-double-buffer` declines a hoist whose prefetched destination would not
// place, and it answers that by running the allocator's own offset assignment.
// So it takes the allocator's strategy and the allocator's alignment, and it
// has to refuse a bad value the way the allocator refuses one: a diagnostic
// naming the offending string and listing what is accepted, then a pass
// failure. A silent fallback to a default nobody asked for would price a
// prefetch under one rule and place it under another, which is the failure the
// single budget exists to prevent, one level down.
//
// Both bad values are given at once, because the pass reports every bad option
// rather than stopping at the first. That is Section 13.1's rule, and it is the
// allocator's behaviour in `alloc-unknown-option.mlir`, which this file is the
// sibling of.

// RUN: npu-opt %s -verify-diagnostics \
// RUN:   --npu-double-buffer="strategy=greedy alignment=48"

// expected-error @+2 {{unknown strategy 'greedy'. The accepted values are: pack, interval}}
// expected-error @+1 {{the alignment must be a positive power of two, but it is 48}}
func.func @both_options_are_wrong(%in: memref<1x8x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  return
}
