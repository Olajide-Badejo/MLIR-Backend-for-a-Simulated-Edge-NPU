// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `-npu-constant-fold`, Section 12, at -O1 and -O2.
//
// The second RUN line is the pass as the pipeline actually runs it, with the
// canonicalization Section 12 puts beside it. The folded operation leaves its
// `tensor.empty` and its operand constants with no users, and removing those is
// canonicalize's job rather than this pass's, so a test that ran only the pass
// would leave a reader thinking the dead operands survive into the program.

// RUN: npu-opt %s --npu-constant-fold | FileCheck %s
// RUN: npu-opt %s --npu-constant-fold --canonicalize | FileCheck --check-prefix=CLEAN %s

// -----------------------------------------------------------------------------
// Positive: an add of two same shaped constants becomes one constant.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @fold_add
// CHECK: npu.constant dense<{{\[}}[1.100000e+01, 2.200000e+01], [3.300000e+01, 4.400000e+01]]>
// CHECK-NOT: npu.add

// CLEAN-LABEL: func.func @fold_add
// CLEAN-NEXT: %[[C:.*]] = npu.constant dense<{{\[}}[1.100000e+01, 2.200000e+01], [3.300000e+01, 4.400000e+01]]>
// CLEAN-NEXT: return %[[C]]
func.func @fold_add() -> tensor<2x2xf32> {
  %a = npu.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  %b = npu.constant dense<[[1.000000e+01, 2.000000e+01], [3.000000e+01, 4.000000e+01]]> : tensor<2x2xf32>
  %d = tensor.empty() : tensor<2x2xf32>
  %r = npu.add ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%d : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %r : tensor<2x2xf32>
}

// -----------------------------------------------------------------------------
// Positive: a chain folds in one run, because the second operation's operand
// has already become a constant by the time the walk reaches it.
// -----------------------------------------------------------------------------

// CLEAN-LABEL: func.func @fold_chain
// CLEAN-NEXT: %[[C:.*]] = npu.constant dense<[0.000000e+00, 1.000000e+00]>
// CLEAN-NEXT: return %[[C]]
func.func @fold_chain() -> tensor<2xf32> {
  %a = npu.constant dense<[-1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %b = npu.constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<2xf32>
  %m = npu.mul ins(%a, %b : tensor<2xf32>, tensor<2xf32>) outs(%d0 : tensor<2xf32>) -> tensor<2xf32>
  %d1 = tensor.empty() : tensor<2xf32>
  %r = npu.relu ins(%m : tensor<2xf32>) outs(%d1 : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----------------------------------------------------------------------------
// Positive: a reshape of a constant is the one case where the operand's shape
// is irrelevant and only its element count matters.
// -----------------------------------------------------------------------------

// CLEAN-LABEL: func.func @fold_reshape
// CLEAN-NEXT: %[[C:.*]] = npu.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>
// CLEAN-NEXT: return %[[C]]
func.func @fold_reshape() -> tensor<4xf32> {
  %a = npu.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  %r = npu.reshape %a : tensor<2x2xf32> to tensor<4xf32>
  return %r : tensor<4xf32>
}

// -----------------------------------------------------------------------------
// Negative 1, and the load bearing one: a rank 1 channel broadcast addend is
// not folded even though both operands are constants.
//
// Section 11 leaves that addend unexpanded so `-npu-fuse-bias` can match it.
// Folding here would materialise the N x C x H x W expansion the importer
// refuses to perform, inflate every DRAM byte count by the expansion factor,
// and leave nothing for the bias fusion to fire on.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_channel_broadcast
// CHECK: npu.add
// CLEAN-LABEL: func.func @no_fold_channel_broadcast
// CLEAN: npu.add
func.func @no_fold_channel_broadcast() -> tensor<1x2x2x2xf32> {
  %x = npu.constant dense<3.000000e+00> : tensor<1x2x2x2xf32>
  %b = npu.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %d = tensor.empty() : tensor<1x2x2x2xf32>
  %r = npu.add ins(%x, %b : tensor<1x2x2x2xf32>, tensor<2xf32>) outs(%d : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}

// -----------------------------------------------------------------------------
// Negative 2: an operand that is not a constant at all. This is the case a pass
// that fired unconditionally would get wrong, and it is here for exactly the
// reason Section 12 states.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_live_operand
// CHECK: npu.mul
// CHECK: npu.relu
func.func @no_fold_live_operand(%x: tensor<2xf32>) -> tensor<2xf32> {
  %b = npu.constant dense<[5.000000e-01, 5.000000e-01]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<2xf32>
  %m = npu.mul ins(%x, %b : tensor<2xf32>, tensor<2xf32>) outs(%d0 : tensor<2xf32>) -> tensor<2xf32>
  %d1 = tensor.empty() : tensor<2xf32>
  %r = npu.relu ins(%m : tensor<2xf32>) outs(%d1 : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// -----------------------------------------------------------------------------
// Negative 3: an operation this pass does not evaluate. A convolution over
// constant operands is foldable in principle and is not folded here, because
// folding a reduction at compile time would move the summation order and
// therefore the last bits, which is a movement this pass does not make and
// `docs/BREAKING_CHANGES.md` does not declare.
// -----------------------------------------------------------------------------

// CHECK-LABEL: func.func @no_fold_convolution
// CHECK: npu.conv2d
func.func @no_fold_convolution() -> tensor<1x1x2x2xf32> {
  %x = npu.constant dense<1.000000e+00> : tensor<1x1x2x2xf32>
  %w = npu.constant dense<5.000000e-01> : tensor<1x1x3x3xf32>
  %d = tensor.empty() : tensor<1x1x2x2xf32>
  %r = npu.conv2d ins(%x, %w : tensor<1x1x2x2xf32>, tensor<1x1x3x3xf32>)
                  outs(%d : tensor<1x1x2x2xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x1x2x2xf32>
  return %r : tensor<1x1x2x2xf32>
}
