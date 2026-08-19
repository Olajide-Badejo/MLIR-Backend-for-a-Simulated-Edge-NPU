// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// One case per verifier rule, each with an expected-error on the substring the
// verifier actually emits rather than a generic match. A test that matched only
// "error" would pass against a verifier that rejected everything for the wrong
// reason, which is the failure mode a negative test exists to rule out.

// RUN: npu-opt %s -split-input-file -verify-diagnostics

// =============================================================================
// npuisa.const
// =============================================================================

func.func @const_element_type_differs() {
  // expected-error @+1 {{the value attribute's element type must equal the result's}}
  %0 = npuisa.const dense<1> : tensor<2x2xi8> -> memref<2x2xf32, #npu.dram>
  return
}

// -----

func.func @const_shape_differs() {
  // expected-error @+1 {{the value attribute's shape must equal the result's}}
  %0 = npuisa.const dense<1.0> : tensor<2x2xf32> -> memref<4x4xf32, #npu.dram>
  return
}

// -----

// A constant in the scratchpad. Section 8 says function arguments and const
// results are the two things that live in DRAM, and the operand type is where
// that is enforced, so this is a parse failure rather than a verifier one.
func.func @const_in_the_scratchpad() {
  // expected-error @+1 {{a statically shaped memref in the Dram memory space}}
  %0 = npuisa.const dense<1.0> : tensor<2x2xf32>
     -> memref<2x2xf32, #npu.scratchpad>
  return
}

// =============================================================================
// The DMA direction and shape rules.
// =============================================================================

func.func @dma_load_shape_differs(%src: memref<16x16xf32, #npu.dram>,
                                  %dst: memref<8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{shapes must agree, but the source is 'memref<16x16xf32, #npu.dram>' and the destination is 'memref<8x8xf32, #npu.scratchpad>'}}
  npuisa.dma_load %src, %dst
    : memref<16x16xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  return
}

// -----

func.func @dma_load_element_type_differs(%src: memref<4x4xi8, #npu.dram>,
                                         %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{element types must agree, but the source has element type 'i8' and the destination has element type 'f32'}}
  npuisa.dma_load %src, %dst
    : memref<4x4xi8, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  return
}

// -----

// The direction is in the operand types, so a load whose source is a scratchpad
// buffer fails to parse rather than failing to verify. That redundancy with the
// operation's name is deliberate.
func.func @dma_load_wrong_direction(%src: memref<4x4xf32, #npu.scratchpad>,
                                    %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a statically shaped memref in the Dram memory space}}
  npuisa.dma_load %src, %dst
    : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  return
}

// -----

func.func @dma_store_wrong_direction(%src: memref<4x4xf32, #npu.dram>,
                                     %dst: memref<4x4xf32, #npu.dram>) {
  // expected-error @+1 {{a statically shaped memref in the Scratchpad memory space}}
  npuisa.dma_store %src, %dst
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.dram>
  return
}

// -----

// A memref with no memory space at all. Every buffer in this dialect is in one
// of two named memories, and the default space is neither.
//
// This case is a regression test as well as a rule. The memory space predicate
// was written with `llvm::isa`, and a memref with no memory space has a *null*
// memory space attribute, which `isa` dereferences: npu-opt died here with a
// segmentation fault and no diagnostic at all rather than rejecting the operand.
// See docs/DEFECT_LOG.md D-0008. The fix is `isa_and_present`, which answers
// false for null, and the proof it worked is that this case now produces the
// ordinary operand type diagnostic below.
func.func @dma_load_into_the_default_space(%src: memref<4x4xf32, #npu.dram>,
                                           %dst: memref<4x4xf32>) {
  // expected-error @+1 {{a statically shaped memref in the Scratchpad memory space}}
  npuisa.dma_load %src, %dst
    : memref<4x4xf32, #npu.dram> to memref<4x4xf32>
  return
}

// -----

// The same null memory space on the other side of a transfer, and on a compute
// instruction, because the predicate is generated once per space and per
// element type set and a fix that reached only one of them would leave the
// others crashing.
func.func @dma_store_from_the_default_space(%src: memref<4x4xf32>,
                                            %dst: memref<4x4xf32, #npu.dram>) {
  // expected-error @+1 {{a statically shaped memref in the Scratchpad memory space}}
  npuisa.dma_store %src, %dst
    : memref<4x4xf32> to memref<4x4xf32, #npu.dram>
  return
}

// -----

func.func @relu_in_the_default_space(%x: memref<4x4xf32>,
                                     %d: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a statically shaped memref in the Scratchpad memory space}}
  npuisa.relu ins(%x : memref<4x4xf32>)
              outs(%d : memref<4x4xf32, #npu.scratchpad>)
  return
}

// -----

// The constant's result, which uses the Dram predicate rather than the
// Scratchpad one, so the same null check is exercised on the other space.
func.func @const_in_the_default_space() {
  // expected-error @+1 {{a statically shaped memref in the Dram memory space}}
  %0 = npuisa.const dense<1.0> : tensor<2x2xf32> -> memref<2x2xf32>
  return
}

// -----

// A dynamic extent. The static shape half of the same predicate, checked here so
// that the two halves of the constraint each have a case rather than sharing
// one that could pass for the wrong reason.
func.func @dma_load_with_a_dynamic_extent(%src: memref<?x4xf32, #npu.dram>,
                                          %dst: memref<?x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a statically shaped memref in the Dram memory space}}
  npuisa.dma_load %src, %dst
    : memref<?x4xf32, #npu.dram> to memref<?x4xf32, #npu.scratchpad>
  return
}

// =============================================================================
// The asynchronous rules of Section 8.
// =============================================================================

// Rule 1: the token has exactly one use.
func.func @token_never_awaited(%src: memref<4x4xf32, #npu.dram>,
                               %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the token must have exactly one use and that use must be an npuisa.await, but it has no uses}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  return
}

// -----

// Rule 1 again, from the other side: two awaits on one token.
func.func @token_awaited_twice(%src: memref<4x4xf32, #npu.dram>,
                               %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the token must have exactly one use and that use must be an npuisa.await, but it has 2 uses}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.await %t
  npuisa.await %t
  return
}

// -----

// Rule 2: the await is in the same block as its producer. The scf.execute_region
// gives a second block to put it in without needing control flow that this
// dialect does not have.
func.func @await_in_a_different_block(%src: memref<4x4xf32, #npu.dram>,
                                      %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the npuisa.await must be in the same block as the asynchronous operation it waits for}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  scf.execute_region {
    npuisa.await %t
    scf.yield
  }
  return
}

// -----

// Rule 3: source and destination shapes agree in the asynchronous form too.
func.func @async_shape_differs(%src: memref<16x16xf32, #npu.dram>,
                               %dst: memref<8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{shapes must agree, but the source is 'memref<16x16xf32, #npu.dram>' and the destination is 'memref<8x8xf32, #npu.scratchpad>'}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<16x16xf32, #npu.dram> to memref<8x8xf32, #npu.scratchpad>
  npuisa.await %t
  return
}

// -----

// Rule 4, the race. The relu writes the very buffer being transferred into.
func.func @intervening_write_to_the_destination(
    %src: memref<4x4xf32, #npu.dram>, %dst: memref<4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{accesses memory overlapping the destination buffer, which is the race the token exists to prevent}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%dst : memref<4x4xf32, #npu.scratchpad>)
              outs(%dst : memref<4x4xf32, #npu.scratchpad>)
  npuisa.await %t
  return
}

// -----

// Rule 4 again, and this is the case Section 8 singles out: two *different* SSA
// values that are two partially overlapping views of one flat buffer. The
// destination is bytes [0, 64) and the relu touches bytes [32, 96). An SSA
// identity check sees two unrelated values and reports no race; the byte range
// arithmetic sees the 32 bytes they share.
func.func @intervening_write_to_a_partially_overlapping_view(
    %src: memref<4x4xf32, #npu.dram>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
  %dst = memref.view %flat[%c0][]
       : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  %overlapping = memref.view %flat[%c32][]
               : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  // expected-error @+1 {{accesses memory overlapping the destination buffer, which is the race the token exists to prevent}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%overlapping : memref<4x4xf32, #npu.scratchpad>)
              outs(%overlapping : memref<4x4xf32, #npu.scratchpad>)
  npuisa.await %t
  memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
  return
}

// -----

// A non static offset. Section 8 is explicit that this is refused with a
// diagnostic rather than assumed disjoint, because "I cannot prove these
// overlap" and "these are disjoint" are different answers.
func.func @async_destination_with_a_dynamic_offset(
    %src: memref<4x4xf32, #npu.dram>, %offset: index) {
  %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
  %dst = memref.view %flat[%offset][]
       : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  // expected-error @+1 {{the asynchronous form requires a destination whose byte range is statically known, but its byte offset within the underlying buffer is not a constant}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.await %t
  memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
  return
}

// -----

// An intervening buffer whose offset is dynamic, over the same allocation as
// the destination. It cannot be shown disjoint, so it is refused.
func.func @intervening_buffer_with_a_dynamic_offset(
    %src: memref<4x4xf32, #npu.dram>, %offset: index) {
  %c0 = arith.constant 0 : index
  %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
  %dst = memref.view %flat[%c0][]
       : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  %unknown = memref.view %flat[%offset][]
           : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  // expected-error @+1 {{accesses a buffer that cannot be shown disjoint from the destination}}
  %t = npuisa.dma_load_async %src, %dst
     : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%unknown : memref<4x4xf32, #npu.scratchpad>)
              outs(%unknown : memref<4x4xf32, #npu.scratchpad>)
  npuisa.await %t
  memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
  return
}

// -----

// Two transfers in flight whose destinations are overlapping views of one flat
// buffer. The double buffering case with two transfers outstanding is legal and
// canonicalize.mlir has it; what makes this one a race is that the second
// transfer writes bytes the first is still writing. The intervening operation
// caught here is the other asynchronous operation, which does declare a write
// effect on its destination, so the effect based scan sees it. The intervening
// npuisa.await is skipped, and that skip is what makes the legal case legal:
// see docs/DEFECT_LOG.md D-0009.
func.func @two_transfers_racing(%src1: memref<4x4xf32, #npu.dram>,
                                %src2: memref<4x4xf32, #npu.dram>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
  %dst1 = memref.view %flat[%c0][]
        : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  %dst2 = memref.view %flat[%c32][]
        : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
  // expected-error @+1 {{accesses memory overlapping the destination buffer, which is the race the token exists to prevent}}
  %t1 = npuisa.dma_load_async %src1, %dst1
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  %t2 = npuisa.dma_load_async %src2, %dst2
      : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
  npuisa.await %t1
  npuisa.await %t2
  memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
  return
}

// -----

// The await's own rule: the token has to come from an asynchronous transfer.
// Today the type system makes that unavoidable, and the check is here so that
// adding a third producer of a token is a decision somebody makes rather than
// a thing that quietly starts verifying.
func.func @await_on_a_block_argument_token(%t: !npuisa.token) {
  // expected-error @+1 {{the token must come from an npuisa.dma_load_async or npuisa.dma_store_async, but it is a block argument}}
  npuisa.await %t
  return
}

// =============================================================================
// npuisa.matmul
// =============================================================================

func.func @matmul_contraction_extents_differ(
    %a: memref<4x16xf32, #npu.scratchpad>,
    %b: memref<8x10xf32, #npu.scratchpad>,
    %d: memref<4x10xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the contraction extents must agree}}
  npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.scratchpad>,
                             memref<8x10xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// -----

func.func @matmul_destination_shape_wrong(
    %a: memref<4x16xf32, #npu.scratchpad>,
    %b: memref<16x10xf32, #npu.scratchpad>,
    %d: memref<4x11xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the destination must be 4 by 10, the shape this contraction implies}}
  npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.scratchpad>,
                             memref<16x10xf32, #npu.scratchpad>)
                outs(%d : memref<4x11xf32, #npu.scratchpad>)
  return
}

// -----

func.func @matmul_bias_length_wrong(%a: memref<4x16xf32, #npu.scratchpad>,
                                    %b: memref<16x10xf32, #npu.scratchpad>,
                                    %c: memref<4xf32, #npu.scratchpad>,
                                    %d: memref<4x10xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the bias length must equal the destination column count, which is 10}}
  npuisa.matmul ins(%a, %b, %c : memref<4x16xf32, #npu.scratchpad>,
                                 memref<16x10xf32, #npu.scratchpad>,
                                 memref<4xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// -----

func.func @matmul_rank_wrong(%a: memref<2x4x16xf32, #npu.scratchpad>,
                             %b: memref<16x10xf32, #npu.scratchpad>,
                             %d: memref<4x10xf32, #npu.scratchpad>) {
  // expected-error @+1 {{matmul is rank 2 by rank 2 into rank 2}}
  npuisa.matmul ins(%a, %b : memref<2x4x16xf32, #npu.scratchpad>,
                             memref<16x10xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// -----

// An operand in DRAM. The compute units of this machine address the scratchpad
// and nothing else, so this is a parse failure at the point the mistake is made
// rather than an addressing fault in the simulator three phases later.
func.func @matmul_operand_in_dram(%a: memref<4x16xf32, #npu.dram>,
                                  %b: memref<16x10xf32, #npu.scratchpad>,
                                  %d: memref<4x10xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a statically shaped memref in the Scratchpad memory space}}
  npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.dram>,
                             memref<16x10xf32, #npu.scratchpad>)
                outs(%d : memref<4x10xf32, #npu.scratchpad>)
  return
}

// =============================================================================
// npuisa.conv2d
// =============================================================================

// An arithmetically impossible convolution: a 3 by 3 kernel over a 2 by 2 input
// with no padding. The effective kernel is 3, the numerator is 2 + 0 + 0 - 3 =
// -1, and the implied extent is floor(-1 / 1) + 1 = 0. The diagnostic quotes
// that implied extent rather than saying only that the shape is wrong.
func.func @conv2d_impossible(%x: memref<1x3x2x2xf32, #npu.scratchpad>,
                             %w: memref<8x3x3x3xf32, #npu.scratchpad>,
                             %d: memref<1x8x1x1xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the window is not representable on the height axis}}
  npuisa.conv2d ins(%x, %w : memref<1x3x2x2xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x1x1xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// -----

func.func @conv2d_destination_extent_wrong(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %w: memref<8x3x3x3xf32, #npu.scratchpad>,
    %d: memref<1x8x7x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the destination height extent must be 8, the extent this window implies}}
  npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x7x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// -----

func.func @conv2d_group_does_not_divide(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %w: memref<8x3x3x3xf32, #npu.scratchpad>,
    %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{group must divide the input channel count, but the input has 3 channels and group is 2}}
  npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 2 : i64}
  return
}

// -----

func.func @conv2d_filter_second_extent_wrong(
    %x: memref<1x8x8x8xf32, #npu.scratchpad>,
    %w: memref<8x8x3x3xf32, #npu.scratchpad>,
    %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the filter's second extent must be the input channel count divided by group, which is 2}}
  npuisa.conv2d ins(%x, %w : memref<1x8x8x8xf32, #npu.scratchpad>,
                             memref<8x8x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 4 : i64}
  return
}

// -----

func.func @conv2d_pads_wrong_count(%x: memref<1x3x8x8xf32, #npu.scratchpad>,
                                   %w: memref<8x3x3x3xf32, #npu.scratchpad>,
                                   %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{pads must have 4 entries in ONNX order, padTop, padLeft, padBottom, padRight, but it has 2}}
  npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// -----

func.func @conv2d_destination_channels_wrong(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %w: memref<8x3x3x3xf32, #npu.scratchpad>,
    %d: memref<1x4x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the destination channel extent must be 8}}
  npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                             memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%d : memref<1x4x8x8xf32, #npu.scratchpad>)
                {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 dilations = array<i64: 1, 1>, group = 1 : i64}
  return
}

// =============================================================================
// The elementwise instructions.
// =============================================================================

func.func @add_shapes_differ(%a: memref<1x8x4x4xf32, #npu.scratchpad>,
                             %b: memref<1x8x4x5xf32, #npu.scratchpad>,
                             %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{shapes must agree, but the right operand is 'memref<1x8x4x5xf32, #npu.scratchpad>' and the destination is 'memref<1x8x4x4xf32, #npu.scratchpad>'}}
  npuisa.add ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                          memref<1x8x4x5xf32, #npu.scratchpad>)
             outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
  return
}

// -----

func.func @relu_shapes_differ(%x: memref<1x8x4x4xf32, #npu.scratchpad>,
                              %d: memref<1x8x4x5xf32, #npu.scratchpad>) {
  // expected-error @+1 {{shapes must agree, but the input is 'memref<1x8x4x4xf32, #npu.scratchpad>' and the destination is 'memref<1x8x4x5xf32, #npu.scratchpad>'}}
  npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%d : memref<1x8x4x5xf32, #npu.scratchpad>)
  return
}

// =============================================================================
// The pooling instructions.
// =============================================================================

func.func @pool_max_destination_extent_wrong(
    %x: memref<1x8x8x8xf32, #npu.scratchpad>,
    %d: memref<1x8x5x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the destination height extent must be 4, the extent this window implies}}
  npuisa.pool_max ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x5x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  return
}

// -----

func.func @pool_avg_ceil_mode_out_of_range(
    %x: memref<1x8x8x8xf32, #npu.scratchpad>,
    %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{ceil_mode must be 0 or 1, but it is 2}}
  npuisa.pool_avg ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 2 : i64}
  return
}

// -----

// Pooling does not change the channel count, so a destination that claims a
// different one is refused.
func.func @pool_max_channels_change(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
                                    %d: memref<1x4x4x4xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the destination channel extent must be 8}}
  npuisa.pool_max ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x4x4x4xf32, #npu.scratchpad>)
                  {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                   pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
  return
}

// =============================================================================
// The shape instructions.
// =============================================================================

func.func @reshape_element_count_differs(
    %x: memref<4x8x2x2xf32, #npu.scratchpad>,
    %d: memref<4x16xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a reshape must preserve the element count}}
  npuisa.reshape ins(%x : memref<4x8x2x2xf32, #npu.scratchpad>)
                 outs(%d : memref<4x16xf32, #npu.scratchpad>)
  return
}

// -----

func.func @transpose_permutation_wrong_length(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %d: memref<1x8x8x3xf32, #npu.scratchpad>) {
  // expected-error @+1 {{permutation must have exactly 4 entries, the destination rank, but it has 3}}
  npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                   outs(%d : memref<1x8x8x3xf32, #npu.scratchpad>)
                   {permutation = array<i64: 0, 2, 3>}
  return
}

// -----

func.func @transpose_permutation_repeats(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %d: memref<1x8x8x3xf32, #npu.scratchpad>) {
  // expected-error @+1 {{permutation must be a permutation, with every index appearing exactly once, but 0 appears more than once}}
  npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                   outs(%d : memref<1x8x8x3xf32, #npu.scratchpad>)
                   {permutation = array<i64: 0, 0, 2, 3>}
  return
}

// -----

func.func @transpose_destination_extent_wrong(
    %x: memref<1x3x8x8xf32, #npu.scratchpad>,
    %d: memref<1x8x3x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{destination extent 2 must be 8, the input extent at permutation[2] = 3}}
  npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                   outs(%d : memref<1x8x3x8xf32, #npu.scratchpad>)
                   {permutation = array<i64: 0, 2, 3, 1>}
  return
}

// -----

func.func @concat_axis_out_of_range(%a: memref<1x4x8x8xf32, #npu.scratchpad>,
                                    %b: memref<1x6x8x8xf32, #npu.scratchpad>,
                                    %d: memref<1x10x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{axis must be in the range 0 to 3, the destination rank being 4, but it is 4}}
  npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                             memref<1x6x8x8xf32, #npu.scratchpad>)
                outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                {axis = 4 : i64}
  return
}

// -----

func.func @concat_extents_do_not_sum(%a: memref<1x4x8x8xf32, #npu.scratchpad>,
                                     %b: memref<1x6x8x8xf32, #npu.scratchpad>,
                                     %d: memref<1x12x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the input extents along axis 1 must sum to the destination extent, which is 12, but they sum to 10}}
  npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                             memref<1x6x8x8xf32, #npu.scratchpad>)
                outs(%d : memref<1x12x8x8xf32, #npu.scratchpad>)
                {axis = 1 : i64}
  return
}

// -----

func.func @concat_disagrees_off_axis(%a: memref<1x4x8x8xf32, #npu.scratchpad>,
                                     %b: memref<1x6x4x8xf32, #npu.scratchpad>,
                                     %d: memref<1x10x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{the inputs and the destination must agree on every axis except axis 1}}
  npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                             memref<1x6x4x8xf32, #npu.scratchpad>)
                outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                {axis = 1 : i64}
  return
}

// -----

// Written in the generic form rather than the custom one, and that is a finding
// about the assembly format rather than a stylistic choice. `ins( : )` and
// `ins()` are both parse errors: the format prints the variadic operands and
// then their types around a literal `:`, and with no operands there is nothing
// on either side of it for the parser to latch onto. So the custom syntax cannot
// spell an empty concatenation at all, and the verifier rule is reachable only
// through the generic form or through a builder. The rule stays, because a pass
// building the operation programmatically can produce exactly this, and the
// generic form is what proves the rule fires when it does.
func.func @concat_with_no_inputs(%d: memref<1x10x8x8xf32, #npu.scratchpad>) {
  // expected-error @+1 {{a concatenation needs at least one input, but it has none}}
  "npuisa.concat"(%d) <{axis = 1 : i64}>
      : (memref<1x10x8x8xf32, #npu.scratchpad>) -> ()
  return
}
