// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-assign-layout`, Section 12, with the inverse transpose fold.
//
// **The pass answers a layout question and cancels what the answer makes
// redundant.** On this machine the answer is NCHW at every extent, because
// Section 5.5 charges NHWC `kDmaStridedElementCycles` of 0.5 per element on
// every transfer and charges a permutation `1 / kElementwiseLaneWidth` of
// 0.0625, so performing the permutation is eight times cheaper than moving the
// same data strided. The `kept-nchw` statistic below is that answer, counted
// rather than assumed: an operation whose layout was scored and answered NCHW
// and one that was never looked at are indistinguishable otherwise.
//
// **The negatives carry the weight here.** A fold that fired on any pair of
// permutations that returns the extents to where they started would delete a
// relayout, which changes what the bytes mean while leaving the types looking
// right. Three of the five cases below exist to hold that line.

// RUN: npu-opt %s --npu-assign-layout | FileCheck %s
// RUN: npu-opt %s --npu-assign-layout \
// RUN:   -mlir-pass-statistics -mlir-pass-statistics-display=list 2>&1 \
// RUN:   | FileCheck %s --check-prefix=STATS

// -----------------------------------------------------------------------------
// Positive: an adjacent inverse pair is replaced by the value it permuted.
//
// `[0, 2, 3, 1]` then `[0, 3, 1, 2]` is the NCHW to NHWC round trip, which is
// the exact permutation Section 10.1's kernel test and the dilated stack's
// closing transpose both use. Both transposes go, and so do both destinations:
// **this pass cleans up after itself rather than leaving a dead `tensor.empty`
// for `-canonicalize`**, because Section 12 puts the two canonicalizations
// around fusion and there is none between this pass and the lowering. A
// destination left dead here would reach the conversion and be given a
// scratchpad buffer for a value nothing reads, so the absence is asserted
// rather than assumed to be somebody else's problem.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @round_trip_folds
// CHECK-NOT:     npu.transpose
// CHECK-NOT:     tensor.empty
// CHECK:         return %arg0
func.func @round_trip_folds(%x: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x3x8x8xf32>
  %b = npu.transpose ins(%t : tensor<1x8x8x3xf32>)
                     outs(%d1 : tensor<1x3x8x8xf32>)
                     {permutation = array<i64: 0, 3, 1, 2>}
       -> tensor<1x3x8x8xf32>
  return %b : tensor<1x3x8x8xf32>
}

// -----------------------------------------------------------------------------
// Positive: the sink is what lets the fold reach a pair that is not adjacent.
//
// The graph is transpose, relu, transpose. The relu moves above the first
// transpose, which makes the two transposes adjacent, and then they fold. What
// is left is one relu at the unpermuted type and nothing else, which is the
// exact result rather than merely fewer transposes: the assertion names the
// operand the relu ends up reading.
//
// The relu is exact under this move. A permutation reads one element to write
// one element and so does a rectified linear unit, so the two orders apply the
// same maximum to the same `f32`.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @sink_then_fold
// CHECK-NOT:     npu.transpose
// CHECK:         %[[R:.*]] = npu.relu ins(%arg0 : tensor<1x3x8x8xf32>)
// CHECK:         return %[[R]]
func.func @sink_then_fold(%x: tensor<1x3x8x8xf32>) -> tensor<1x3x8x8xf32> {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x8x8x3xf32>
  %r = npu.relu ins(%t : tensor<1x8x8x3xf32>)
                outs(%d1 : tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %d2 = tensor.empty() : tensor<1x3x8x8xf32>
  %b = npu.transpose ins(%r : tensor<1x8x8x3xf32>)
                     outs(%d2 : tensor<1x3x8x8xf32>)
                     {permutation = array<i64: 0, 3, 1, 2>}
       -> tensor<1x3x8x8xf32>
  return %b : tensor<1x3x8x8xf32>
}

// -----------------------------------------------------------------------------
// Negative: two permutations that do not compose to the identity.
//
// `[0, 2, 3, 1]` then `[0, 2, 3, 1]` again lands on N, W, C, H, which is a
// different tensor from the one it started with even though it is back to being
// rank 4. Both transposes stay.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @not_inverses_stay
// CHECK:         npu.transpose
// CHECK:         npu.transpose
func.func @not_inverses_stay(%x: tensor<1x3x8x8xf32>) -> tensor<1x8x3x8xf32> {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x8x3x8xf32>
  %b = npu.transpose ins(%t : tensor<1x8x8x3xf32>)
                     outs(%d1 : tensor<1x8x3x8xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x3x8xf32>
  return %b : tensor<1x8x3x8xf32>
}

// -----------------------------------------------------------------------------
// Negative: inverse permutations that are a relayout rather than a round trip.
//
// The extents come back to `1x3x8x8` and the permutations do compose to the
// identity, so a fold that looked only at the permutations would delete both.
// It must not: the result carries `#npu.layout<nhwc>` and the input does not,
// so the pair means "these are the same extents read the other way round",
// which is a change of layout and not the absence of one. The guard is that the
// surviving value's type must equal the replaced result's exactly, encoding
// included, and this case is what that guard is for.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @relayout_is_not_a_round_trip
// CHECK:         npu.transpose
// CHECK:         npu.transpose
func.func @relayout_is_not_a_round_trip(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x3x8x8xf32, #npu.layout<nhwc>> {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x3x8x8xf32, #npu.layout<nhwc>>
  %b = npu.transpose ins(%t : tensor<1x8x8x3xf32>)
                     outs(%d1 : tensor<1x3x8x8xf32, #npu.layout<nhwc>>)
                     {permutation = array<i64: 0, 3, 1, 2>}
       -> tensor<1x3x8x8xf32, #npu.layout<nhwc>>
  return %b : tensor<1x3x8x8xf32, #npu.layout<nhwc>>
}

// -----------------------------------------------------------------------------
// Negative: a transpose two operations read is not sunk.
//
// Sinking rewrites the transpose to consume the relu's result, so a second
// reader would find its operand changed underneath it. The alternative is to
// duplicate the transpose, and duplicating a full pass over the data to enable
// a fold that removes one is not a trade this pass makes on its own. Nothing
// moves, and the relu still reads the transpose.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @two_readers_block_the_sink
// CHECK:         %[[T:.*]] = npu.transpose ins(%arg0
// CHECK:         npu.relu ins(%[[T]]
// CHECK:         npu.mul ins(%[[T]], %[[T]]
func.func @two_readers_block_the_sink(%x: tensor<1x3x8x8xf32>)
    -> (tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>) {
  %d0 = tensor.empty() : tensor<1x8x8x3xf32>
  %t = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d0 : tensor<1x8x8x3xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32>
  %d1 = tensor.empty() : tensor<1x8x8x3xf32>
  %r = npu.relu ins(%t : tensor<1x8x8x3xf32>)
                outs(%d1 : tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %d2 = tensor.empty() : tensor<1x8x8x3xf32>
  %m = npu.mul ins(%t, %t : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>)
               outs(%d2 : tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  return %r, %m : tensor<1x8x8x3xf32>, tensor<1x8x8x3xf32>
}

// **The counts, asserted rather than left implied.**
//
// Two pairs fold, one adjacent and one after a sink, and exactly one sink
// happens: neither negative pair folds and the two reader case sinks nothing.
//
// `kept-nchw` counts the layout questions the pass answered, and the answer was
// NCHW every time. Three operations are scored here: the relu of
// `sink_then_fold`, and the relu and the mul of `two_readers_block_the_sink`.
// A transpose is not scored, because it is the mechanism of a layout change
// rather than a holder of one, and the scoring walk runs before any rewrite so
// that the count reports what the graph asked rather than what it was left as.
// The assertion is the number rather than a range, because a walk that stopped
// early would still produce a plausible looking smaller one.
// STATS: 2 folded-pairs
// STATS: 3 kept-nchw
// STATS: 1 sunk-transposes
