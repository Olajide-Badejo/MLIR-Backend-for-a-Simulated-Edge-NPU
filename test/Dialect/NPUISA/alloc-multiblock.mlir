// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The refusals the allocator makes on the IR in front of it, as opposed to on
// the options it was given.
//
// The file is named for the first of them because Section 6's layout names it
// that, and the other two are here rather than in a file of their own for a
// mechanical reason: `-verify-diagnostics` runs one command over the whole
// file, so every case in one file has to be refused under one set of pass
// options. These three need none, and the option refusals in
// `alloc-unknown-option.mlir` need three at once, so that is where the split
// falls.

// RUN: npu-opt %s -split-input-file -verify-diagnostics \
// RUN:   --npu-allocate-scratchpad

// =============================================================================
// Section 8: multiple blocks are diagnosed, not ignored.
//
// This is not a stylistic restriction. Liveness here is an ordering of one
// straight line instruction stream, and an operation index in a second block is
// not comparable with one in the first: a buffer defined in one and read in the
// other has no live *interval*, so the sweep line would be computing a number
// that means nothing. Refusing is the only honest answer, and Section 5.2 says
// the case cannot arise anyway because this instruction set has no branches.
// =============================================================================

// expected-error @+1 {{the allocator requires a single block function body, but @two_blocks has 2 blocks}}
func.func @two_blocks(%in: memref<1x8x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  cf.br ^tail
^tail:
  return
}

// -----

// =============================================================================
// A buffer whose byte size cannot be computed.
//
// Section 13.1 takes sizes from the memref type and its element type, never
// from a hardcoded factor of four, and a dynamic extent has no size to take.
// The lowering already refuses a dynamic extent at the tensor level, so this is
// defence in depth rather than a reachable path today; it is here because the
// allocator must not be the layer that guesses.
// =============================================================================

// The allocation stands alone, because no `npuisa` instruction will take it:
// every operand type in that dialect carries `HasStaticShapePred`, so an
// instruction reading this buffer fails to verify before the pass ever runs and
// the case would then be testing the operand predicate instead of the
// allocator.
func.func @a_dynamic_extent(%n: index) {
  // expected-error @+1 {{this scratchpad allocation has no byte size the allocator can compute}}
  %a = memref.alloc(%n) : memref<?x4xf32, #npu.scratchpad>
  return
}

// -----

// =============================================================================
// A budget attribute that is not a positive integer.
//
// The attribute is data the driver wrote and the encoder later reads, so a
// malformed one is refused where it is first used rather than silently replaced
// by the default. A silent replacement would produce a perfectly valid program
// allocated against a budget nobody asked for, and the number would then travel
// into a result cell as though it had been measured.
// =============================================================================

// expected-error @+1 {{the npuisa.scratchpad_budget attribute of @a_budget_that_is_not_a_number must be a positive integer}}
func.func @a_budget_that_is_not_a_number(%in: memref<1x8x4x4xf32, #npu.dram>)
    attributes {npuisa.scratchpad_budget = "as much as it takes"} {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  return
}
