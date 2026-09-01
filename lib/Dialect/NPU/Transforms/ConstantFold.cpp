//===- ConstantFold.cpp - npu level constant folding ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-constant-fold`, at `-O1` and `-O2`.
//
// **Why this is a pass and not four `fold` methods on the operations.** MLIR's
// folder runs inside `-canonicalize` and would do the same arithmetic, but
// Section 12 asks for `-npu-constant-fold` as a **named entry in the pipeline
// description** so that Section 16.2's leave one out ablation has a row for it.
// A fold hook is not something an ablation can remove; a pass is. The two are
// therefore not equivalent for this project's purposes, and the pass is the
// form the specification names.
//
// **What is folded, and the shape of the guard.** Four operations, all
// elementwise or purely structural, all of whose reads must be `npu.constant`
// **with the result's own shape**. The shape clause is the load bearing half:
// the rank 1 channel broadcast of Section 11 is a legal operand of `npu.add`
// and `npu.mul`, and folding one would mean materialising the `N x C x H x W`
// expansion the importer refuses to perform. That expansion inflates every DRAM
// byte count by the expansion factor and it makes `-npu-fuse-bias` unfireable
// on the result, which is the failure Section 11 wrote the carve out to
// prevent. So a broadcast operand is a non match rather than a fold, and a lit
// test pins it.
//
// **The arithmetic does not move a bit.** Every operation folded here is
// elementwise with no reduction, so `a + b` computed at compile time in `f32`
// and `a + b` computed by the kernel in `f32` are the same value. This pass is
// in the `-O1` set and `-O1`'s goldens are expected to equal `-O0`'s exactly;
// `docs/BREAKING_CHANGES.md` attributes P9's numerics movement to
// `-npu-fold-batchnorm` alone, and this comment is half of why that attribution
// is possible.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir::npu {
#define GEN_PASS_DEF_NPUCONSTANTFOLD
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

/// The values of an `npu.constant` whose type is exactly `expected`.
///
/// Returns nothing when the operand is not a constant at all, and equally when
/// it is a constant of a different shape, which is the broadcast case. One
/// function answers both because the caller's question is the same either way:
/// is this operand something this pass may evaluate elementwise against the
/// result.
std::optional<SmallVector<APFloat>> sameShapedConstant(Value operand,
                                                       RankedTensorType expected) {
  auto producer = operand.getDefiningOp<ConstantOp>();
  if (!producer)
    return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(producer.getValue());
  if (!dense)
    return std::nullopt;
  auto type = dyn_cast<RankedTensorType>(dense.getType());
  if (!type || type != expected)
    return std::nullopt;
  if (!isa<FloatType>(type.getElementType()))
    return std::nullopt;
  return llvm::to_vector(dense.getValues<APFloat>());
}

/// The whole of an `npu.constant`'s values, whatever its shape.
std::optional<SmallVector<APFloat>> anyConstant(Value operand) {
  auto producer = operand.getDefiningOp<ConstantOp>();
  if (!producer)
    return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(producer.getValue());
  if (!dense || !isa<FloatType>(dense.getType().getElementType()))
    return std::nullopt;
  return llvm::to_vector(dense.getValues<APFloat>());
}

/// Replaces `op` with one `npu.constant` holding `values`.
void replaceWithConstant(OpBuilder &builder, Operation *op,
                         RankedTensorType type, ArrayRef<APFloat> values) {
  builder.setInsertionPoint(op);
  Value folded = ConstantOp::create(builder, op->getLoc(), type,
                                    DenseElementsAttr::get(type, values))
                     .getResult();
  op->getResult(0).replaceAllUsesWith(folded);
  op->erase();
}

/// Whether this operation's reads are all constants of the result's own shape,
/// and if so, what they are.
///
/// The destination operand is deliberately not consulted. A destination is a
/// place to write and never a value to read, per Section 7.2, so a folded
/// operation leaves its `tensor.empty` with no user and `-canonicalize`
/// removes it.
struct FoldableBinary {
  SmallVector<APFloat> lhs;
  SmallVector<APFloat> rhs;
};

std::optional<FoldableBinary> foldableBinary(Value lhs, Value rhs,
                                             RankedTensorType resultType) {
  std::optional<SmallVector<APFloat>> left = sameShapedConstant(lhs, resultType);
  if (!left)
    return std::nullopt;
  std::optional<SmallVector<APFloat>> right =
      sameShapedConstant(rhs, resultType);
  if (!right)
    return std::nullopt;
  return FoldableBinary{std::move(*left), std::move(*right)};
}

// `impl` is qualified in full because `using namespace mlir` and
// `using namespace mlir::npu` are both in scope here and both namespaces have
// one. The alternative is to drop one of the using directives and spell every
// operation type instead, which trades one qualification for fifty.
struct ConstantFoldPass
    : public mlir::npu::impl::NPUConstantFoldBase<ConstantFoldPass> {
  using mlir::npu::impl::NPUConstantFoldBase<
      ConstantFoldPass>::NPUConstantFoldBase;

  void runOnOperation() override;
};

void ConstantFoldPass::runOnOperation() {
  func::FuncOp function = getOperation();
  OpBuilder builder(function.getContext());

  // Collected first and rewritten afterwards, in source order, so that a chain
  // of foldable operations collapses in one run: the second operation's
  // operand has already become a constant by the time it is reached. Walking
  // and mutating at once would invalidate the iterator on the erase.
  SmallVector<Operation *> candidates;
  function.walk([&](Operation *op) {
    if (isa<AddOp, MulOp, ReluOp, ReshapeOp>(op))
      candidates.push_back(op);
  });

  int64_t folded = 0;
  for (Operation *op : candidates) {
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !isa<FloatType>(resultType.getElementType()))
      continue;

    if (auto add = dyn_cast<AddOp>(op)) {
      std::optional<FoldableBinary> operands =
          foldableBinary(add.getLhs(), add.getRhs(), resultType);
      if (!operands)
        continue;
      SmallVector<APFloat> values;
      values.reserve(operands->lhs.size());
      for (auto [left, right] : llvm::zip_equal(operands->lhs, operands->rhs))
        values.push_back(
            APFloat(left.convertToFloat() + right.convertToFloat()));
      replaceWithConstant(builder, op, resultType, values);
      ++folded;
      continue;
    }

    if (auto mul = dyn_cast<MulOp>(op)) {
      std::optional<FoldableBinary> operands =
          foldableBinary(mul.getLhs(), mul.getRhs(), resultType);
      if (!operands)
        continue;
      SmallVector<APFloat> values;
      values.reserve(operands->lhs.size());
      for (auto [left, right] : llvm::zip_equal(operands->lhs, operands->rhs))
        values.push_back(
            APFloat(left.convertToFloat() * right.convertToFloat()));
      replaceWithConstant(builder, op, resultType, values);
      ++folded;
      continue;
    }

    if (auto relu = dyn_cast<ReluOp>(op)) {
      std::optional<SmallVector<APFloat>> input =
          sameShapedConstant(relu.getInput(), resultType);
      if (!input)
        continue;
      SmallVector<APFloat> values;
      values.reserve(input->size());
      for (const APFloat &element : *input)
        values.push_back(APFloat(std::max(element.convertToFloat(), 0.0f)));
      replaceWithConstant(builder, op, resultType, values);
      ++folded;
      continue;
    }

    auto reshape = cast<ReshapeOp>(op);
    // A reshape moves nothing, so the operand's shape is irrelevant and only
    // its element count matters. `sameShapedConstant` would be the wrong guard
    // here and this is the one place the difference shows.
    std::optional<SmallVector<APFloat>> input = anyConstant(reshape.getInput());
    if (!input || static_cast<int64_t>(input->size()) != resultType.getNumElements())
      continue;
    replaceWithConstant(builder, op, resultType, *input);
    ++folded;
  }

  foldedOps += folded;
}

} // namespace
