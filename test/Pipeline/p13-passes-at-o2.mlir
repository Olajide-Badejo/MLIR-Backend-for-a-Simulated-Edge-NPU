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
// Tiling, negative: over budget, and the result is both returned and read.
//
// **The one shape the binary still cannot express, and it is a narrow one.** A
// returned assembly is stored straight into the out parameter the function
// gained for it, and Section 8 makes an output region a place to write and
// never a place to read. A value that is both returned and read would need it
// to be both, and assembling into a spill slot and storing from there into the
// out parameter would be a DRAM to DRAM transfer this machine has no
// instruction for. D-0052 carries the reasoning.
//
// **A result that is only read is fine from P13**, which is the case below
// this one: checks 8 and 9 answer a read inside a declared spill slot by exact
// byte coverage, so the assembly comes back on chip in one transfer.
//
// The convolution has two readers so `-npu-fuse-ops` leaves it alone, which is
// deliberate: a single reader would be fused into the same region and the
// decline this case is about would be the fused region's rather than this one.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @declines_when_the_result_is_returned_and_read
// TENSOR-NOT:     npu.tiling_choice
// TENSOR:         npu.conv2d

// REMARK: this operation's result is both returned and read by another operation

func.func @declines_when_the_result_is_returned_and_read(
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
// Tiling, positive: over budget, and the result is read by another operation.
//
// **This is the case D-0052 refused and the one the format was taught to
// accept.** The tiles are stored into a spill slot, one `npuisa.dma_store`
// each, and the operation that reads the assembly gets it back in **one**
// `npuisa.dma_load`, which is Section 8's count unchanged: the assembly is one
// DRAM value and it enters the scratchpad once, however many operations read
// it. Checks 8 and 9 admit that read because every byte of it lies inside one
// declared slot and the tiles cover all of them.
//
// The relu after the convolution is what makes the result read rather than
// returned, and the second reader is what keeps `-npu-fuse-ops` from folding
// the two into one region.
// -----------------------------------------------------------------------------

// TENSOR-LABEL: func.func @tiles_when_the_result_is_read
// TENSOR:         npu.tiling_choice
// TENSOR:         tensor.insert_slice

// One store per tile into the assembly, then one load of the whole thing back.
// LOWERED-LABEL: func.func @tiles_when_the_result_is_read
// LOWERED:         npuisa.dma_store
// LOWERED:         npuisa.dma_store
// LOWERED:         npuisa.dma_load
// LOWERED:         npuisa.relu

func.func @tiles_when_the_result_is_read(%x: tensor<1x8x8x8xf32>,
                                         %w: tensor<8x8x3x3xf32>,
                                         %y: tensor<1x8x8x8xf32>)
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
  %f = tensor.empty() : tensor<1x8x8x8xf32>
  %a = npu.add ins(%c, %y : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>)
               outs(%f : tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32>
  return %r, %a : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>
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
// Double buffering at the level, positive: the weight load runs under the relu.
//
// **This case was a measured negative for two commits and is a positive now**,
// which is the whole of D-0054's second half. An argument's load sits in the
// entry block with the other argument loads, and the walk stops at another
// transfer, correctly: both are charged to the same DMA port, so lifting a load
// above a load moves work along a saturated timeline and hides nothing. **A
// constant's load is the one transfer with a computation before it**, and it
// was declined because `npuisa.const` was not in the set of operations the
// transfer may take with it, so hoisting the load alone would have left its own
// source defined after it. The constant is in that set now: it is a pure
// definition whose result lives in DRAM, so moving it changes no scratchpad
// pressure at all.
//
// The relu is what the transfer runs under, and the `await` is left where the
// load was.
// -----------------------------------------------------------------------------

// RUN: npu-opt %s '--npu-O2=budget=6464' -o /dev/null \
// RUN:   -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

// **All three counts, and the three of them are three different answers.**
// `prefetched` is the rewrite firing, `not-hoisted` is a transfer with nothing
// safe to overlap, and `would-not-fit` is a transfer that had something to hide
// under and whose prefetched destination would not place. Section 19.0's rule
// is that silence and success must not look alike, and a pass that answered no
// has to read differently from one that was never in the pipeline.
// STATS: NPUDoubleBuffer
// STATS-NEXT: {{[0-9]+}} not-hoisted
// STATS-NEXT: {{[1-9][0-9]*}} prefetched
// STATS-NEXT: {{[1-9][0-9]*}} would-not-fit

// LOWERED-LABEL: func.func @double_buffer_prefetches_the_weight_load
// LOWERED:         npuisa.dma_load_async
// LOWERED:         npuisa.relu
// LOWERED:         npuisa.await
// LOWERED:         npuisa.conv2d

// With the pass ablated the transfer stays synchronous, which is the negative
// half of the same case and is what the ablation row measures.
// NODB-LABEL: func.func @double_buffer_prefetches_the_weight_load
// NODB-NOT:     npuisa.dma_load_async

func.func @double_buffer_prefetches_the_weight_load(%x: tensor<1x2x4x4xf32>)
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

// -----------------------------------------------------------------------------
// Double buffering at the level, negative: the prefetch would not place.
//
// **The decline rule, as a program.** The pooling holds a 5120 byte argument
// and a 320 byte result at once, which is 5440 bytes and fits the 6464 byte
// budget. Hoisting the 1080 byte weight load above the pooling makes the
// weight resident across it too, which is 6520, and the arena does not place.
// The pass runs the allocator's own offset assignment over the intervals the
// hoist would produce, gets no placement back, and declines.
//
// **The peak would have said yes and that is why the rule is not the peak.**
// Section 13.1 makes the sweep line peak a lower bound on any placement and
// makes the spill trigger "offset assignment failed", and D-0054 has the
// measurement that separates the two: `dilated_stack` accepted a prefetch whose
// peak fit its budget by 36 bytes and then needed 20 more than the arena had.
// -----------------------------------------------------------------------------

// LOWERED-LABEL: func.func @double_buffer_declines_a_prefetch_that_would_not_place
// LOWERED-NOT:     npuisa.dma_load_async
// LOWERED:         npuisa.conv2d

// At the default budget there is room for the doubled residency, so the same
// program prefetches. **The decline is about the budget and not about the
// shape**, and asserting it at both budgets is what says so.
// WIDE-LABEL: func.func @double_buffer_declines_a_prefetch_that_would_not_place
// WIDE:         npuisa.dma_load_async

func.func @double_buffer_declines_a_prefetch_that_would_not_place(
    %x: tensor<1x5x16x16xf32>) -> tensor<1x6x4x4xf32> {
  %d0 = tensor.empty() : tensor<1x5x4x4xf32>
  %a = npu.max_pool2d ins(%x : tensor<1x5x16x16xf32>)
                      outs(%d0 : tensor<1x5x4x4xf32>)
                      {kernel = array<i64: 4, 4>, strides = array<i64: 4, 4>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x5x4x4xf32>
  %w = npu.constant dense<2.500000e-01> : tensor<6x5x3x3xf32>
  %d1 = tensor.empty() : tensor<1x6x4x4xf32>
  %c = npu.conv2d ins(%a, %w : tensor<1x5x4x4xf32>, tensor<6x5x3x3xf32>)
                  outs(%d1 : tensor<1x6x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x6x4x4xf32>
  return %c : tensor<1x6x4x4xf32>
}
