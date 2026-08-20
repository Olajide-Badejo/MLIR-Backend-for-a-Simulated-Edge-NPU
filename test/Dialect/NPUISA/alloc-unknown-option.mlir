// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// An unknown option value is a diagnostic naming the offending string and
// listing the accepted values, then a pass failure.
//
// Section 13.1 asks for exactly that, and gives the reason: a typo must not
// silently select a heuristic nobody asked for. It is the reason both
// enumerated options are declared as `std::string` and parsed inside the pass
// rather than as ODS enum options. An enum option is rejected by the pass
// manager's own option parser before the pass runs, with a message that names
// neither the pass nor the accepted values in the pass's own words, and that no
// `-verify-diagnostics` test can capture.
//
// All three bad values are given at once, on purpose. The pass reports every
// bad option rather than stopping at the first, so somebody who mistyped two of
// them does not fix one, rerun, and discover the second. Three errors on one
// function is what that behaviour looks like, and each names its own offending
// string, so this file also proves the three messages cannot be confused for
// one another.

// RUN: npu-opt %s -verify-diagnostics \
// RUN:   --npu-allocate-scratchpad="strategy=greedy spill-heuristic=belady alignment=48"

// expected-error @+3 {{unknown strategy 'greedy'. The accepted values are: pack, interval}}
// expected-error @+2 {{unknown spill heuristic 'belady'. The accepted values are: longest-range, cost}}
// expected-error @+1 {{the alignment must be a positive power of two, but it is 48}}
func.func @every_option_is_wrong(%in: memref<1x8x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  return
}
