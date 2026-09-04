// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-tile-to-scratchpad` declining, which Section 13.2 calls a result rather
// than a failure.
//
// Three refusals live here and they are different in kind. The first is the
// budget being too small for any permitted tiling, where the fallback is the
// allocator's spilling. The second is Section 13.3's halo boolean refusing to
// split the spatial axes, which is the arm that pays for not creating a halo by
// having less room to manoeuvre. The third is a misspelled option, which is a
// hard failure because a typo must not silently select a policy nobody asked
// for.
//
// **The refusals are checked rather than the absence of a crash.** A pass that
// declined silently would be indistinguishable from one whose trigger never
// fired, and those two want different responses from whoever reads the run.
//
// The remark is matched with FileCheck on the diagnostic stream rather than
// with `-verify-diagnostics`, and the difference is worth a line: this pass
// emits a **remark**, and the diagnostic verifier reports one raised from
// inside a pass on a nested operation as unexpected even when an
// `expected-remark` names its line. Matching the text is the assertion that
// works, and it is the same assertion: the message has to name both numbers.

// RUN: npu-opt %s --npu-tile-to-scratchpad=budget=64 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DECLINE
// RUN: npu-opt %s --npu-tile-to-scratchpad='budget=2048 halo=cache' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHE
// RUN: npu-opt %s --npu-tile-to-scratchpad='budget=768 halo=cache' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CACHEDECLINE
// RUN: npu-opt %s --npu-tile-to-scratchpad='budget=768 halo=recompute' 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RECOMPUTE
// RUN: not npu-opt %s --npu-tile-to-scratchpad=strategy=nonsense 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BADSTRATEGY
// RUN: not npu-opt %s --npu-tile-to-scratchpad=halo=nonsense 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BADHALO

// -----------------------------------------------------------------------------
// Nothing fits, so the pass declines and says so with both numbers in the
// message.
//
// A budget of 64 bytes cannot hold this operation's filter, let alone a tile of
// it together with its input and its output. The only thing that would make it
// fit is splitting the input channel or the kernel window, which re associates
// an fp32 accumulation and moves every golden file, so the interface refuses it
// and the pass reports that rather than reaching around it.
// -----------------------------------------------------------------------------

// DECLINE: remark: working set of 4224 bytes exceeds the 64 byte budget
// DECLINE-SAME: no tiling over the parallel dimensions fits
// DECLINE-SAME: allow-reduction-tiling
// DECLINE-SAME: allocator's spilling is the fallback
// DECLINE-NOT: tensor.extract_slice

// -----------------------------------------------------------------------------
// Section 13.3's third arm, on one function, at the budget where the choice
// actually bites.
//
// **`halo=cache` refuses to split the two output spatial axes**, so no halo is
// created and none is re-read. The cost of that is the input operand: it keeps
// its full 8 by 8 extent and its 1024 bytes whatever else is split, because the
// input channel count is fixed by the reduction and only the **output** channel
// is left to divide. At 2048 bytes that is enough: two output channels at a
// time need 1824 and the pass tiles four ways.
//
// **At 768 the same policy has nowhere left to go and declines**, because one
// output channel still costs 1024 for the input alone, while `halo=recompute`
// splits the rows two at a time and fits in 720. That contrast is Section
// 13.3's third arm in one function: not creating a halo costs the ability to
// shrink the input at all, and the budget at which that stops being affordable
// is a measurement rather than an argument.
//
// What this is not is a scratchpad resident halo carried between tiles. That
// needs a transfer whose source is the previous tile's buffer and a cost model
// term for a partially reused operand, and this machine has neither. The pass
// documentation says so in the same words.
// -----------------------------------------------------------------------------

// CACHE-LABEL: func.func @nothing_fits
// CACHE: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 4, 8, 8]
// CACHE: npu.conv2d
// CACHE-SAME: temporal_tiles = array<i64: 1, 1, 2, 8, 8>
// CACHE-SAME: tile_bytes = 1824
// CACHE-SAME: tile_count = 4
// The operation's own pads survive whole, because the axes they belong to were
// never split.
// CACHE-SAME: pads = array<i64: 1, 1, 1, 1>

// CACHEDECLINE: remark: working set of 4224 bytes exceeds the 768 byte budget
// CACHEDECLINE-NOT: tensor.extract_slice

// The same budget with the halo paid for instead: the rows split two at a time
// and the input slice shrinks with them, which is the whole of the difference.
// RECOMPUTE-LABEL: func.func @nothing_fits
// RECOMPUTE: tensor.extract_slice %arg0[0, 0, 0, 0] [1, 4, 3, 8]
// RECOMPUTE: npu.conv2d
// RECOMPUTE-SAME: temporal_tiles = array<i64: 1, 1, 1, 2, 8>
// RECOMPUTE-SAME: tile_bytes = 720
// RECOMPUTE-SAME: tile_count = 32

func.func @nothing_fits(%x: tensor<1x4x8x8xf32>, %w: tensor<8x4x3x3xf32>,
                        %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %0 = npu.conv2d ins(%x, %w : tensor<1x4x8x8xf32>, tensor<8x4x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----------------------------------------------------------------------------
// An unknown option value is a diagnostic naming the offending string and
// listing the accepted values, then a pass failure.
//
// Section 13.1 states that rule for the allocator's `spill-heuristic` and the
// reason carries here unchanged: a misspelled strategy that fell back to the
// default would produce an ablation row labelled with a strategy that never
// ran, which is worse than no row at all.
// -----------------------------------------------------------------------------

// BADSTRATEGY: unknown tiling strategy 'nonsense'
// BADSTRATEGY-SAME: 'exhaustive', 'fixed' and 'largest-fit'
// BADHALO: unknown halo policy 'nonsense'
// BADHALO-SAME: 'recompute' and 'cache'
