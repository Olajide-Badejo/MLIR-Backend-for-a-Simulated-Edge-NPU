// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The three passes P13 puts into `-O2`, asserted **at the level** rather than
// one at a time.
//
// `test/Transforms/assign-layout.mlir`, `tile-to-scratchpad.mlir` and
// `double-buffer.mlir` each check their own pass in isolation, which is where a
// rewrite's details belong. What this file checks is the half those cannot: that
// the level runs them, in Section 12's positions, with the options the pipeline
// chooses rather than the passes' own defaults, and that each one still declines
// where it should when the whole `-O2` list has run ahead of it. Section 12's
// negative test rule is the reason the second half exists: a pass that fired
// unconditionally would satisfy every positive check here.
//
// **The no `scf` assertion is a statement about the lowering and that is new.**
// The tiling pass asserts it about its own function, which is the pass checking
// its own work. Here the pipeline runs the pass and then the conversion, at a
// budget where tiling actually fires, so the claim being made is that nothing
// with a loop in it reaches `-npu-lower-to-npuisa`.

// The tensor level half, at a budget where tiling fires.
// RUN: npu-opt %s '--npu-O2=budget=6464 stop-after=npu' \
// RUN:   | FileCheck %s --check-prefix=TENSOR

// The whole level at that budget: no scf reaches the lowering, and the program
// that comes out of it is instructions and transfers.
// RUN: npu-opt %s '--npu-O2=budget=6464' | FileCheck %s --check-prefix=NOSCF
// RUN: npu-opt %s '--npu-O2=budget=6464' | FileCheck %s --check-prefix=LOWERED

// The same file at the default budget, where nothing is over budget at all.
// RUN: npu-opt %s --npu-O2 | FileCheck %s --check-prefix=WIDE

// The declines, by name, from the remarks the passes emit.
// RUN: npu-opt %s '--npu-O2=budget=6464' -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REMARK

// Each of the three ablated out of the level, which is the negative that
// matters most: the row an ablation table reads is the pass being absent rather
// than the pass being present and idle.
// RUN: npu-opt %s '--npu-O2=budget=6464 ablate=npu-tile-to-scratchpad' \
// RUN:   | FileCheck %s --check-prefix=NOTILING
// RUN: npu-opt %s '--npu-O2=budget=6464 ablate=npu-double-buffer' \
// RUN:   | FileCheck %s --check-prefix=NODB
// RUN: npu-opt %s '--npu-O2=budget=6464 ablate=npu-assign-layout' \
// RUN:   | FileCheck %s --check-prefix=NOLAYOUT

// **Nothing anywhere in the level's output has a loop in it.** The prefix
// carries only a negative, which FileCheck permits and which is the whole
// assertion: the pass unrolls its grid at compile time because this ISA has no
// branches, and an `scf` operation surviving into the lowering would be that
// having failed.
// NOSCF-NOT: scf.

// -----------------------------------------------------------------------------
// Tiling, positive: over budget, and the result is the function's own.
//
// The working set is 2048 bytes of input, 2304 of filter and 2048 of output,
// which is 6400 against a 6464 byte budget, and it is the prefetch that puts it
// over: Section 13.2 makes the doubled working set the search's problem, so the
// pass is told from the pipeline that `-npu-double-buffer` is in it. That
// coupling is what this case measures, and the ablation run below is the other
// end of it.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @tiles_when_the_result_is_returned
// TENSOR:         npu.tiling_choice
// TENSOR:         tensor.insert_slice

// The tiles leave the scratchpad one store at a time, into the out parameter
// the function gained for its result. A tiled result is assembled in DRAM and
// this is that decision arriving as a program.
// LOWERED-LABEL: func.func @tiles_when_the_result_is_returned
// LOWERED:         npuisa.conv2d
// LOWERED:         npuisa.dma_store
// LOWERED:         npuisa.conv2d
// LOWERED:         npuisa.dma_store

// With the pass ablated the same convolution is one instruction and the
// allocator is left to deal with the pressure, which is the fallback Section
// 13.2 names.
// NOTILING-LABEL: func.func @tiles_when_the_result_is_returned
// NOTILING-NOT:     npu.tiling_choice

// At the default budget nothing is over budget, so nothing tiles. The pass runs
// and answers no, which is a different thing from the pass not running.
// WIDE-LABEL: func.func @tiles_when_the_result_is_returned
// WIDE-NOT:     npuisa.dma_store
// WIDE:         npuisa.conv2d

func.func @tiles_when_the_result_is_returned(
    %x: tensor<1x8x8x8xf32>, %w: tensor<8x8x3x3xf32>) -> tensor<1x8x8x8xf32> {
  %d = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %c : tensor<1x8x8x8xf32>
}

// -----------------------------------------------------------------------------
// Tiling, negative: over budget, and something reads the result.
//
// **This is a decline about the binary format rather than about the search.** A
// tiled result is assembled in DRAM by one store per tile, and the
// `operand-defined` and `operand-extent` checks satisfy a read out of a single
// written span, so an assembled value that is read back is refused by the
// encoder. D-0052 carries the reproduction. The pass declines rather than
// emitting a program this project's own encoder rejects.
//
// The convolution has two readers so `-npu-fuse-ops` leaves it alone, which is
// deliberate: a single reader would be fused into the same region and the
// decline this case is about would be the fused region's rather than this one.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @declines_when_the_result_is_read
// TENSOR-NOT:     npu.tiling_choice
// TENSOR:         npu.conv2d

// REMARK: this operation's result is read by another operation rather than returned

func.func @declines_when_the_result_is_read(
    %x: tensor<1x8x8x8xf32>, %w: tensor<8x8x3x3xf32>)
    -> (tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) {
  %d = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  %e = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.relu ins(%c : tensor<1x8x8x8xf32>)
                outs(%e : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %c, %r : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>
}

// -----------------------------------------------------------------------------
// Layout, positive: the inverse pair the level's own `-cse` did not remove.
//
// `[0, 2, 3, 1]` then `[0, 3, 1, 2]` is the NCHW to NHWC round trip. Both
// transposes go and so do the destinations they leave behind, because there is
// no canonicalization between this pass and the lowering to remove them.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @layout_folds_the_round_trip
// TENSOR-NOT:     npu.transpose
// TENSOR-NOT:     tensor.empty
// TENSOR:         return %arg0

// With the pass ablated the pair survives the whole level and becomes two
// instructions, which is the ablation row this pass has.
// NOLAYOUT-LABEL: func.func @layout_folds_the_round_trip
// NOLAYOUT:         npuisa.transpose
// NOLAYOUT:         npuisa.transpose

func.func @layout_folds_the_round_trip(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x3x8x8xf32> {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x3x8x8xf32>
  %r = npu.transpose ins(%t : tensor<1x8x8x3xf32>)
                     outs(%d1 : tensor<1x3x8x8xf32>)
                     {permutation = array<i64: 0, 3, 1, 2>}
       -> tensor<1x3x8x8xf32>
  return %r : tensor<1x3x8x8xf32>
}

// -----------------------------------------------------------------------------
// Layout, negative: a lone transpose is not a round trip and nothing rewrites
// an operation into NHWC.
//
// There is deliberately no code that materialises an NHWC operand, because the
// comparison Section 5.5 defines refuses that trade at every shape this machine
// can hold: a strided move costs 0.5 cycles per element against a permutation's
// 0.0625, so performing the transpose always beats moving the same data strided.
// The absence is asserted here rather than assumed.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @layout_leaves_a_lone_transpose_alone
// TENSOR:         npu.transpose
// TENSOR-NOT:     #npu.nhwc

func.func @layout_leaves_a_lone_transpose_alone(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x8x8x3xf32> {
  %d = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  return %t : tensor<1x8x8x3xf32>
}

// -----------------------------------------------------------------------------
// Double buffering at the level, and this case is a **measured negative**.
//
// `test/Transforms/double-buffer.mlir` shows the rewrite firing on hand written
// `npuisa`, so the pass works. What this case records is that **no transfer the
// lowering emits at `-O2` is one it can move**, which is a fact about the two
// together rather than about either alone, and it is D-0054.
//
// Two reasons and neither is the overlap being worthless. An argument's load
// sits in the entry block with the other argument loads, and the walk stops at
// another transfer, correctly: both are charged to the same DMA port, so
// lifting a load above a load moves work along a saturated timeline. A
// constant's load does have a computation before it, and it is declined because
// `npuisa.const` is not in the set of operations the transfer may take with it,
// which is `memref.alloc` and `memref.subview`, so hoisting the load alone
// would leave its own source defined after it.
//
// **The statistic is what keeps this from being silent.** The pass runs, counts
// every transfer it looked at, and answers `not-hoisted`, which reads
// differently from a pass that was never in the pipeline. Section 19.0's rule
// is that silence and success must not look alike, and a zero `prefetched`
// beside a nonzero `not-hoisted` is the pass saying which of the two this is.
// -----------------------------------------------------------------------------

// RUN: npu-opt %s '--npu-O2=budget=6464' -o /dev/null \
// RUN:   -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

// STATS: NPUDoubleBuffer
// STATS-NEXT: {{[1-9][0-9]*}} not-hoisted
// STATS-NEXT: 0 prefetched

// LOWERED-LABEL: func.func @double_buffer_declines_every_transfer_here
// LOWERED-NOT:     npuisa.dma_load_async
// LOWERED:         npuisa.conv2d

// With the pass ablated the output is the same, which is the honest shape of
// this ablation row: it is zero because the pass fires on nothing, and the
// coupling it has with the tiling search is the part of the row that is not
// about this pass at all.
// NODB-LABEL: func.func @double_buffer_declines_every_transfer_here
// NODB-NOT:     npuisa.dma_load_async

func.func @double_buffer_declines_every_transfer_here(%x: tensor<1x2x4x4xf32>)
    -> tensor<1x2x4x4xf32> {
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %a = npu.relu ins(%x : tensor<1x2x4x4xf32>)
                outs(%d0 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %w = npu.constant dense<5.000000e-01> : tensor<2x2x3x3xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%a, %w : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>)
                  outs(%d1 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  return %c : tensor<1x2x4x4xf32>
}
