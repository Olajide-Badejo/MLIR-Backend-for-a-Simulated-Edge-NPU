// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-fuse-ops`, Section 12, at -O2. The pass that creates `npu.fused_op` and
// `npu.yield`, which is what closes the two entries `docs/EXEMPTIONS.md` had
// been carrying since P8: neither operation could appear in a model's IR until
// something produced one.
//
// The LOWERED prefix carries the claim that makes fusion mean what Section 5.2
// says it means. `-npu-lower-to-npuisa` flattens the region, so the fused
// program's instruction stream is the unfused one's: the same two compute
// instructions in the same order with no DMA between them. The benefit under
// this memory model is that the intermediate stays on chip, and it already did;
// what the region adds is that the fusion is *stated* in the IR, which is what
// P13's tiling and double buffering read.
//
// The two forms are not byte identical as text, and the difference is one every
// reader should be able to place: the fused form allocates both destinations
// before the region and the unfused form allocates the second one between the
// two operations. A `memref.alloc` is not an instruction, so the instruction
// stream is unmoved; `test/Python/test_transform_passes.py` asserts the
// instruction count directly, which is the form of the claim Section 10.2
// allows.

// RUN: npu-opt %s --npu-fuse-ops | FileCheck %s
// RUN: npu-opt %s --npu-fuse-ops --npu-fuse-ops | FileCheck --check-prefix=TWICE %s
// RUN: npu-opt %s --npu-fuse-ops --npu-lower-to-npuisa | FileCheck --check-prefix=LOWERED %s

// -----------------------------------------------------------------------------
// Positive: a convolution and the relu that reads it become one region. Every
// value the region reads is an operand, destinations included, because the
// operation is IsolatedFromAbove and its verifier admits only npu operations
// inside, so a tensor.empty cannot be cloned in.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fuse_conv_relu
// CHECK: %[[D0:.*]] = tensor.empty()
// CHECK: %[[D1:.*]] = tensor.empty()
// CHECK: npu.fused_op ins(%arg0, %{{.*}}, %[[D0]], %[[D1]]
// CHECK-NEXT: ^bb0(%[[A:.*]]: tensor<1x2x4x4xf32>, %[[W:.*]]: tensor<2x2x1x1xf32>, %[[E0:.*]]: tensor<1x2x4x4xf32>, %[[E1:.*]]: tensor<1x2x4x4xf32>)
// CHECK-NEXT: %[[C:.*]] = npu.conv2d ins(%[[A]], %[[W]]
// CHECK-SAME: outs(%[[E0]]
// CHECK-NEXT: %[[R:.*]] = npu.relu ins(%[[C]]
// CHECK-SAME: outs(%[[E1]]
// CHECK-NEXT: npu.yield %[[R]]

// The region is flattened at lowering, so the instruction stream holds the two
// compute instructions with no DMA between them: the activation reads the
// convolution's output straight out of the scratchpad. The `memref.alloc`
// between them is the activation's own destination, sunk to where it is written
// so the allocator sees the live range the program actually has, and it is not
// an instruction.
// LOWERED-LABEL: func.func @fuse_conv_relu
// LOWERED: npuisa.conv2d
// LOWERED-NOT: npuisa.dma
// LOWERED: npuisa.relu
// LOWERED-NEXT: npuisa.dma_store
// LOWERED-NOT: npuisa.fused_op

// Section 12's "an already fused producer is not fused again" guard, asserted
// the way it actually matters: a second run of the pass produces exactly one
// region rather than a region inside a region.
// TWICE-LABEL: func.func @fuse_conv_relu
// TWICE: npu.fused_op
// TWICE-NOT: npu.fused_op
func.func @fuse_conv_relu(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Positive: a matmul and its activation fuse the same way. The two producers
// are named in one guard rather than in two patterns, because the region does
// not care which operation filled it.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fuse_matmul_relu
// CHECK: npu.fused_op
// CHECK: npu.matmul
// CHECK: npu.relu
// CHECK: npu.yield
// TWICE-LABEL: func.func @fuse_matmul_relu
func.func @fuse_matmul_relu(%a: tensor<4x8xf32>) -> tensor<4x2xf32> {
  %w = npu.constant dense<2.000000e+00> : tensor<8x2xf32>
  %d0 = tensor.empty() : tensor<4x2xf32>
  %m = npu.matmul ins(%a, %w : tensor<4x8xf32>, tensor<8x2xf32>) outs(%d0 : tensor<4x2xf32>) -> tensor<4x2xf32>
  %d1 = tensor.empty() : tensor<4x2xf32>
  %r = npu.relu ins(%m : tensor<4x2xf32>) outs(%d1 : tensor<4x2xf32>) -> tensor<4x2xf32>
  return %r : tensor<4x2xf32>
}

// -----------------------------------------------------------------------------
// Negative 1, Section 12's "exactly one use" guard: the intermediate is read by
// a second consumer, which would have to read a value living inside a region it
// is not in.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_multiple_uses
// CHECK-NOT: npu.fused_op
func.func @no_fuse_multiple_uses(%x: tensor<1x2x4x4xf32>) -> (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) {
  %w = npu.constant dense<2.000000e+00> : tensor<2x2x1x1xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                  outs(%d0 : tensor<1x2x4x4xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %d2 = tensor.empty() : tensor<1x2x4x4xf32>
  %s = npu.relu ins(%c : tensor<1x2x4x4xf32>) outs(%d2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r, %s : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 2: the consumer's input comes from a function argument rather than
// from a fusible producer, so there is no producer to pull in.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_argument_input
// CHECK-NOT: npu.fused_op
func.func @no_fuse_argument_input(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %d = tensor.empty() : tensor<1x2x4x4xf32>
  %r = npu.relu ins(%x : tensor<1x2x4x4xf32>) outs(%d : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}

// -----------------------------------------------------------------------------
// Negative 3: a producer this pass does not fuse. Pooling reads a window rather
// than one element, so there is no elementwise activation to keep on chip with
// it, and a region around a pool alone would state a fusion that did not
// happen.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fuse_pool_producer
// CHECK-NOT: npu.fused_op
func.func @no_fuse_pool_producer(%x: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  %d0 = tensor.empty() : tensor<1x2x2x2xf32>
  %p = npu.max_pool2d ins(%x : tensor<1x2x4x4xf32>) outs(%d0 : tensor<1x2x2x2xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>, dilations = array<i64: 1, 1>,
                       ceil_mode = 0 : i64}
       -> tensor<1x2x2x2xf32>
  %d1 = tensor.empty() : tensor<1x2x2x2xf32>
  %r = npu.relu ins(%p : tensor<1x2x2x2xf32>) outs(%d1 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}
