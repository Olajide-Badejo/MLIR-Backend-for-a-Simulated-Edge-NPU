// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The four upstream passes Section 12's table puts in the `-O` levels, tested
// on this dialect's IR rather than on somebody else's.
//
// Section 12's negative test rule applies to every pass in the table, not only
// to the ones written here, and the reason is the same for all of them: a pass
// with only positive tests would be passed by a pass that fired
// unconditionally. That these four come from upstream MLIR does not change what
// they do to `npu` operations, and what they do to `npu` operations is what
// this project depends on.
//
// **`-sccp` needed a dialect change to do anything at all**, and that is
// D-0033. It computed the right lattice and had nowhere to put the answer,
// because the `npu` dialect implemented no constant materializer, so it
// reported no change on every input. The SCCP block below is the test that
// would have caught it.

// RUN: npu-opt %s --canonicalize | FileCheck --check-prefix=CANON %s
// RUN: npu-opt %s --cse | FileCheck --check-prefix=CSE %s
// RUN: npu-opt %s --sccp | FileCheck --check-prefix=SCCP %s
// RUN: npu-opt %s --symbol-dce | FileCheck --check-prefix=DCE %s

// -----------------------------------------------------------------------------
// `-canonicalize`. Positive: an `npu` operation nothing reads is removed,
// because every operation in this dialect carries `Pure`. That is the whole
// mechanism behind `eliminatesDeadCode = true` on this entry and behind
// Section 17.3a's dead subgraph injection leaving the instruction count
// unchanged at `-O1` and `-O2`.
//
// Negative: the operations the result depends on are all still there. A
// canonicalizer that removed one of those would pass a test that only asserted
// the dead one had gone.
// -----------------------------------------------------------------------------

// CANON-LABEL: func.func @dead_and_live
// CANON: npu.relu
// CANON: npu.mul
// CANON-NOT: npu.relu
// CANON: return
func.func @dead_and_live(%x: tensor<2xf32>) -> tensor<2xf32> {
  %d0 = tensor.empty() : tensor<2xf32>
  %a = npu.relu ins(%x : tensor<2xf32>) outs(%d0 : tensor<2xf32>) -> tensor<2xf32>
  %c = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<2xf32>
  %m = npu.mul ins(%a, %c : tensor<2xf32>, tensor<2xf32>) outs(%d1 : tensor<2xf32>) -> tensor<2xf32>
  %d2 = tensor.empty() : tensor<2xf32>
  %dead = npu.relu ins(%m : tensor<2xf32>) outs(%d2 : tensor<2xf32>) -> tensor<2xf32>
  return %m : tensor<2xf32>
}

// -----------------------------------------------------------------------------
// `-cse`. Positive: two identical multiplies over identical operands become
// one. Negative: a third multiply differing in one operand is not merged with
// them, which is the case a common subexpression eliminator that compared too
// little would get wrong.
// -----------------------------------------------------------------------------

// CSE-LABEL: func.func @duplicate_work
// CSE: npu.mul
// CSE: npu.mul
// CSE-NOT: npu.mul
// CSE: return
func.func @duplicate_work(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %c = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %c2 = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<2xf32>
  %a = npu.mul ins(%x, %c : tensor<2xf32>, tensor<2xf32>) outs(%d0 : tensor<2xf32>) -> tensor<2xf32>
  %d1 = tensor.empty() : tensor<2xf32>
  %b = npu.mul ins(%x, %c2 : tensor<2xf32>, tensor<2xf32>) outs(%d1 : tensor<2xf32>) -> tensor<2xf32>
  %d2 = tensor.empty() : tensor<2xf32>
  %e = npu.mul ins(%y, %c : tensor<2xf32>, tensor<2xf32>) outs(%d2 : tensor<2xf32>) -> tensor<2xf32>
  return %a, %b, %e : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// -----------------------------------------------------------------------------
// `-sccp`. Positive: a private function whose only caller passes a constant has
// that constant propagated into its body, which needs the dialect to be able to
// materialise one. Negative: a private function called twice with two different
// constants keeps its argument, because the lattice meet of two constants is
// not a constant.
// -----------------------------------------------------------------------------

// SCCP-LABEL: func.func private @one_caller
// SCCP: npu.constant dense<[2.000000e+00, 3.000000e+00]>
// SCCP: npu.mul ins(%cst, %cst
func.func private @one_caller(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d = tensor.empty() : tensor<2xf32>
  %r = npu.mul ins(%x, %c : tensor<2xf32>, tensor<2xf32>) outs(%d : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// SCCP-LABEL: func.func private @two_callers
// SCCP: npu.mul ins(%arg0
func.func private @two_callers(%x: tensor<2xf32>) -> tensor<2xf32> {
  %c = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %d = tensor.empty() : tensor<2xf32>
  %r = npu.mul ins(%x, %c : tensor<2xf32>, tensor<2xf32>) outs(%d : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

func.func @drives_sccp() -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) {
  %c0 = npu.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
  %c1 = npu.constant dense<[5.000000e+00, 7.000000e+00]> : tensor<2xf32>
  %a = call @one_caller(%c0) : (tensor<2xf32>) -> tensor<2xf32>
  %b = call @two_callers(%c0) : (tensor<2xf32>) -> tensor<2xf32>
  %e = call @two_callers(%c1) : (tensor<2xf32>) -> tensor<2xf32>
  return %a, %b, %e : tensor<2xf32>, tensor<2xf32>, tensor<2xf32>
}

// -----------------------------------------------------------------------------
// `-symbol-dce`. Positive: a private function nobody calls is removed.
// Negative: a private function somebody calls is kept, and so is every public
// one, because a public symbol may have a caller this module cannot see.
// -----------------------------------------------------------------------------

// DCE-NOT: func.func private @never_called
// DCE-LABEL: func.func private @called_once
// DCE-LABEL: func.func @calls_it
func.func private @never_called(%x: tensor<2xf32>) -> tensor<2xf32> {
  %d = tensor.empty() : tensor<2xf32>
  %r = npu.relu ins(%x : tensor<2xf32>) outs(%d : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

func.func private @called_once(%x: tensor<2xf32>) -> tensor<2xf32> {
  %d = tensor.empty() : tensor<2xf32>
  %r = npu.relu ins(%x : tensor<2xf32>) outs(%d : tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

func.func @calls_it(%x: tensor<2xf32>) -> tensor<2xf32> {
  %r = call @called_once(%x) : (tensor<2xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}
