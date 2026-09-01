//===- FuseBias.cpp - a separate bias add into the bias operand -*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-fuse-bias`, at `-O2`.
//
//     add(conv2d(x, w), b) -> conv2d(x, w, b)
//
// **A convolution and nothing else, and the dialect is what decides that.**
// `npu.matmul` also carries an optional bias, so a matmul form looks like an
// obvious second case. It is not representable: `npu.add`'s verifier admits a
// rank 1 right hand operand only when the result is rank 4, and a matmul's
// result is rank 2. `add(matmul(a, b), c)` with a rank 1 `c` does not parse,
// so a pattern for it would be code no test could reach. The one shape that
// does reach a matmul is a same shaped addend, and that is a residual rather
// than a bias.
//
// **This is the pass Section 11's broadcast carve out exists for.** A rank 1
// initializer of length C broadcasting against a rank 4 activation over the
// channel axis is left unexpanded by the importer, as a rank 1 constant,
// precisely so this guard can match. An importer that expanded it into a full
// `N x C x H x W` constant would make this pass structurally unfireable on
// every model, its ablation row a row of zeros, and the phase look done while
// doing nothing. Section 11 says so in those words and this file is the other
// end of that obligation.
//
// **It is exact, and that is measured rather than assumed.** The simulator's
// convolution kernel accumulates into an `f32` and adds the bias to that same
// `f32` before it writes, which is the value the unfused program would have
// stored and then added to. So a fused and an unfused answer agree bit for bit,
// and P9's numerics movement is attributable to `-npu-fold-batchnorm` alone.
//
// **The commuted form is explicitly not matched**, which is one of the two
// options Section 12 offers. `npu.add` refuses a rank 1 left hand operand in
// its verifier, so `add(b, conv(x, w))` is not representable in this dialect;
// the importer commutes at import and there is exactly one spelling below it.
// A match on a form the verifier rejects would be dead code carrying a test
// nobody could write.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::npu {
#define GEN_PASS_DEF_NPUFUSEBIAS
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

/// Whether `addend` is the channel shaped constant this pass folds.
///
/// Three ways to fail, and each is one of the negative cases Section 12 names:
/// the addend is not an `npu.constant`; it is a constant of the result's own
/// shape rather than a rank 1 one, which is what a residual add looks like;
/// or it is rank 1 of the wrong length, which the dialect's verifier would
/// have refused anyway and which is checked here so that the guard reads as
/// one rule rather than as a rule and an assumption.
bool isChannelConstant(Value addend, int64_t channels) {
  auto constant = addend.getDefiningOp<ConstantOp>();
  if (!constant)
    return false;
  auto type = dyn_cast<RankedTensorType>(constant.getResult().getType());
  return type && type.getRank() == 1 && type.getDimSize(0) == channels;
}

/// Moves a constant above the operation that is about to read it.
///
/// The importer emits a constant immediately before its use, so a bias
/// constant sits *after* the convolution it is about to become an operand of.
/// `npu.constant` has no operands, so moving it earlier can never break its own
/// dominance, and any other reader of it comes later still.
void hoistAbove(ConstantOp constant, Operation *consumer) {
  if (constant->getBlock() == consumer->getBlock() &&
      constant->isBeforeInBlock(consumer))
    return;
  constant->moveBefore(consumer);
}

// `impl` is qualified in full: `mlir` and `mlir::npu` both have one and both
// are in scope through the using directives above.
struct FuseBiasPass : public mlir::npu::impl::NPUFuseBiasBase<FuseBiasPass> {
  using mlir::npu::impl::NPUFuseBiasBase<FuseBiasPass>::NPUFuseBiasBase;

  void runOnOperation() override;
};

bool fuse(AddOp add) {
  auto conv = add.getLhs().getDefiningOp<Conv2DOp>();
  if (!conv)
    return false;

  // Section 12's "single use" guard. A second reader of the convolution's
  // result reads the value *without* the bias, and moving the bias into the
  // producer would silently change what that reader sees.
  if (!conv.getResult().hasOneUse())
    return false;

  // A layout encoding would move the channel axis, and layout assignment is
  // P13, so a result carrying one is declined rather than guessed at.
  auto resultType = dyn_cast<RankedTensorType>(add.getResult().getType());
  if (!resultType || resultType.getRank() != 4 || resultType.getEncoding())
    return false;
  const int64_t channels = resultType.getDimSize(1);

  // Section 12's "already fused" guard: a producer that already carries a bias
  // has nothing to move into, and adding a second one would be inventing an
  // operand the dialect does not have.
  if (conv.getBias())
    return false;

  if (!isChannelConstant(add.getRhs(), channels))
    return false;

  auto constant = add.getRhs().getDefiningOp<ConstantOp>();
  hoistAbove(constant, conv);

  // Three fixed operands and one optional, with no operand segment attribute,
  // so the operand count alone decides whether the bias is present. Growing
  // the list from three to four is the whole of adding one.
  conv->setOperands(
      {conv.getInput(), conv.getFilter(), constant.getResult(),
       conv.getDestination()});

  // The add's own destination is left with no user, and `-canonicalize`
  // removes it. That is the canonicalization Section 12 puts after fusion,
  // doing the work it is there for.
  add.getResult().replaceAllUsesWith(conv.getResult());
  add.erase();
  return true;
}

void FuseBiasPass::runOnOperation() {
  func::FuncOp function = getOperation();

  SmallVector<AddOp> candidates;
  function.walk([&](AddOp op) { candidates.push_back(op); });

  int64_t fused = 0;
  for (AddOp op : candidates)
    if (fuse(op))
      ++fused;

  fusedBiases += fused;
}

} // namespace
