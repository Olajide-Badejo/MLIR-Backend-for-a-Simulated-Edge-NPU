//===- AssignLayout.cpp - the rank 4 layout choice ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-assign-layout`, at `-O2`, with the inverse transpose fold.
//
// **The pass answers one question and cancels what the answer makes
// redundant.** The question is which of the two layouts a rank 4 activation
// should be held in, and the answer on this machine is NCHW every time, for a
// reason that is two named constants rather than a preference. The
// cancellation is the inverse transpose fold Section 12 names, plus the sink
// through an elementwise operation that lets the fold reach a pair the graph
// did not write adjacent.
//
// **Why the answer is always NCHW here.** Section 5.5 charges layout in
// exactly one place, the non unit innermost stride penalty on a transfer.
// Section 5.5 also fixes how a layout survives bufferization: an NHWC tensor
// becomes a buffer at NCHW extents carrying permuted strides, so its innermost
// stride is the channel count and every transfer of it pays
// `kDmaStridedElementCycles` per element. An NCHW tensor is contiguous and pays
// nothing. The alternative to paying it is to perform the permutation, which is
// an elementwise pass at `1 / kElementwiseLaneWidth` cycles per element. The
// two rates are 0.5 and 0.0625, so performing the permutation is eight times
// cheaper than moving the same data strided, at every extent, and no shape in
// the suite comes close to reversing it.
//
// That is the P13 gate's layout delta, reported the way it came out. It is not
// a disappointment and it is not a shrug: it is a ratio between two constants
// that a reader can check, and `CostModelTest.cpp` asserts the crossover, in
// the file that owns both constants, so that recalibrating either fails a test
// rather than silently reversing a published conclusion.
//
// **What is deliberately absent.** There is no code here that rewrites an
// operation into NHWC. Section 12 says the pass may change a layout only by
// inserting a `npu.transpose`, and the cost comparison refuses that trade at
// every shape this machine can hold, so a materialisation path would be a
// branch no input could reach and no test could exercise. The refusal is
// counted, in `kept-nchw`, so that the difference between a decision taken and
// lost and a decision never reached is visible from outside. Absorbing a
// layout change into the transfer that was moving the bytes anyway is
// `relayout-and-move`, which Section 12 names, scopes out of this version, and
// forbids any gate from depending on.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"
#include "NPU/Simulator/CostModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::npu {
#define GEN_PASS_DEF_NPUASSIGNLAYOUT
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

//===----------------------------------------------------------------------===//
// The choice.
//===----------------------------------------------------------------------===//

/// What each layout costs an activation of `elements` elements, in the one term
/// that separates them.
///
/// Both numbers are per activation rather than per operation, because that is
/// the granularity at which the two differ: the arithmetic an operation
/// performs does not depend on how its operands are laid out, so every compute
/// term cancels out of the comparison and including it would only add noise
/// that is equal on both sides.
struct LayoutCost {
  /// What holding the activation in NHWC costs: the non unit innermost stride
  /// penalty, once per crossing of the DRAM boundary.
  double nhwc = 0.0;
  /// What holding it in NCHW costs: the permutation that has to be performed
  /// to get it there, as one elementwise pass plus the issue overhead of the
  /// instruction that performs it.
  double nchw = 0.0;
};

/// The two costs for an activation of `elements` elements whose innermost NHWC
/// stride would be `channels`.
///
/// `channels == 1` is the one case where NHWC is charged nothing, because a
/// single channel buffer's permuted strides are innermost unit strides and
/// `dmaCycles` reads only the innermost. It still does not win: it ties at
/// zero against a permutation that would not have to happen either, and the
/// comparison below is strict.
LayoutCost layoutCost(int64_t elements, int64_t channels) {
  LayoutCost cost;
  if (elements <= 0)
    return cost;
  if (channels != 1)
    cost.nhwc = static_cast<double>(elements) * nbin::kDmaStridedElementCycles;
  cost.nchw = nbin::elementwiseCycles(elements) + nbin::kIssueOverheadCycles;
  return cost;
}

/// Whether NHWC is strictly cheaper for this activation.
///
/// Strictly, because a tie is not a reason to move: the layouts differ in what
/// they cost to move and in nothing else, and rewriting an operation for no
/// gain spends compile time and makes a diff that a reader has to explain.
bool preferNHWC(int64_t elements, int64_t channels) {
  const LayoutCost cost = layoutCost(elements, channels);
  return cost.nhwc < cost.nchw;
}

//===----------------------------------------------------------------------===//
// The permutation arithmetic.
//===----------------------------------------------------------------------===//

/// Whether `outer` applied to `inner` is the identity.
///
/// `inner` maps result axis `i` to input axis `inner[i]`, and so does `outer`.
/// Composing them sends the outer result's axis `i` to the inner input's axis
/// `inner[outer[i]]`, and the pair is an inverse pair exactly when that is `i`
/// for every `i`.
bool composesToIdentity(ArrayRef<int64_t> outer, ArrayRef<int64_t> inner) {
  if (outer.size() != inner.size())
    return false;
  for (auto [position, index] : llvm::enumerate(outer)) {
    if (index < 0 || index >= static_cast<int64_t>(inner.size()))
      return false;
    if (inner[index] != static_cast<int64_t>(position))
      return false;
  }
  return true;
}

/// Whether a permutation leaves every axis where it found it.
bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (auto [position, index] : llvm::enumerate(permutation))
    if (index != static_cast<int64_t>(position))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// The rewrites.
//===----------------------------------------------------------------------===//

/// Removes a destination `tensor.empty` that a rewrite here has just orphaned.
///
/// **This pass cleans up after itself rather than leaving it to
/// `-canonicalize`, and the difference from `-npu-fuse-bias` is the pipeline
/// rather than the taste.** Section 12 puts the two canonicalizations around
/// **fusion**, so the passes that fuse have one after them and can leave a dead
/// destination for it. This pass runs after `-symbol-dce`, and nothing between
/// it and the lowering removes a dead `tensor.empty`: `-cse` merges, `-sccp`
/// propagates and `-symbol-dce` is about symbols. A destination left dead here
/// would reach the conversion and be given a buffer, which is a scratchpad
/// allocation for a value nothing reads.
///
/// Only a value this pass orphaned is passed in, and only an unread
/// `tensor.empty` is erased, so this removes nothing it did not create the
/// deadness of.
void eraseIfDeadEmpty(Value destination) {
  auto empty = destination.getDefiningOp<tensor::EmptyOp>();
  if (empty && empty.getResult().use_empty())
    empty.erase();
}

/// `transpose(transpose(%x, p), q)` with `q` composed with `p` the identity,
/// and `transpose(%x, identity)`, both replaced by `%x`.
///
/// **The type equality is the guard rather than a formality.** Two
/// permutations that compose to the identity return the extents to where they
/// started, but they do not return the *encoding*: a pair whose inner result is
/// NHWC and whose outer result carries no encoding has permuted the bytes into
/// a different layout and back out to the same extents under a different
/// reading, which is a relayout and not a round trip. Comparing the full types,
/// encoding included, is what tells the two apart, and it is why this checks
/// types rather than shapes.
///
/// **`orphaned` is an output rather than an erasure.** Folding the outer
/// transpose can leave the inner one with no readers, and a fold that left a
/// dead full pass over the data behind would be reporting a saving it did not
/// make. It cannot be erased here, because the caller is iterating a worklist
/// that may still hold it; it is handed back so the caller can drop it from
/// that worklist first.
bool foldTranspose(TransposeOp op, TransposeOp &orphaned) {
  Value input = op.getInput();

  if (isIdentityPermutation(op.getPermutation())) {
    if (input.getType() != op.getResult().getType())
      return false;
    Value destination = op.getDestination();
    op.getResult().replaceAllUsesWith(input);
    op.erase();
    eraseIfDeadEmpty(destination);
    return true;
  }

  auto inner = input.getDefiningOp<TransposeOp>();
  if (!inner)
    return false;
  if (!composesToIdentity(op.getPermutation(), inner.getPermutation()))
    return false;
  if (inner.getInput().getType() != op.getResult().getType())
    return false;

  Value destination = op.getDestination();
  op.getResult().replaceAllUsesWith(inner.getInput());
  op.erase();
  eraseIfDeadEmpty(destination);
  if (inner.getResult().use_empty())
    orphaned = inner;
  return true;
}

/// `relu(transpose(%x, p))` becomes `transpose(relu(%x), p)`.
///
/// **Exact, element for element.** A permutation moves an element without
/// reading its neighbours and a rectified linear unit reads one element to
/// write one element, so the composition in either order applies the same
/// maximum to the same `f32` and stores it at the same place. There is no
/// reassociation here and therefore nothing for the 1e-6 band to cover.
///
/// **The single use guard.** The transpose is rewritten in place to consume the
/// relu's result, so a second reader of the transpose would find its operand
/// changed underneath it. Declining that case leaves the graph exactly as it
/// was, which is the right answer: sinking a transpose that two operations read
/// would have to duplicate it, and duplicating a full pass over the data to
/// enable a fold that saves one is not a trade this pass should make on its own.
///
/// **The destinations swap with the operations.** Each operation's destination
/// must have its own result's type exactly, so moving the relu above the
/// transpose means giving the relu the transpose input's type and giving the
/// transpose the relu's old one. The two `tensor.empty` values are already
/// there, one per operation, so this exchanges them rather than allocating.
bool sinkThroughRelu(ReluOp relu) {
  auto transpose = relu.getInput().getDefiningOp<TransposeOp>();
  if (!transpose)
    return false;
  if (!transpose.getResult().hasOneUse())
    return false;

  // A relu whose result is read as a different type than its input is not the
  // elementwise identity on shape this rewrite assumes. It cannot happen under
  // the verifier, and checking it here means the rewrite states its own
  // precondition instead of inheriting it.
  if (relu.getInput().getType() != relu.getResult().getType())
    return false;

  auto reluDestination = relu.getDestination().getDefiningOp<tensor::EmptyOp>();
  auto transposeDestination =
      transpose.getDestination().getDefiningOp<tensor::EmptyOp>();
  if (!reluDestination || !transposeDestination)
    return false;
  if (!reluDestination.getResult().hasOneUse() ||
      !transposeDestination.getResult().hasOneUse())
    return false;

  Value source = transpose.getInput();
  auto sourceType = cast<RankedTensorType>(source.getType());

  OpBuilder builder(transpose);

  // The relu now runs at the source's type, so it needs a destination of that
  // type. The transpose keeps its own destination, which already carries the
  // permuted type the transpose still produces; only the relu's is replaced.
  Value newReluDestination = tensor::EmptyOp::create(
      builder, reluDestination.getLoc(), sourceType.getShape(),
      sourceType.getElementType(), sourceType.getEncoding());

  auto sunkRelu = ReluOp::create(builder, relu.getLoc(), sourceType, source,
                                 newReluDestination);

  transpose.getInputMutable().assign(sunkRelu.getResult());
  transpose->moveAfter(sunkRelu);

  relu.getResult().replaceAllUsesWith(transpose.getResult());
  relu.erase();
  if (reluDestination.getResult().use_empty())
    reluDestination.erase();
  return true;
}

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

// `impl` is qualified in full: `mlir` and `mlir::npu` both have one and both
// are in scope through the using directives above.
struct AssignLayoutPass
    : public mlir::npu::impl::NPUAssignLayoutBase<AssignLayoutPass> {
  using mlir::npu::impl::NPUAssignLayoutBase<
      AssignLayoutPass>::NPUAssignLayoutBase;

  void runOnOperation() override;
};

/// Scores one operation's result and returns whether it stays NCHW.
///
/// Only rank 4 activations have a layout question at all, and an operation that
/// already carries an encoding was assigned by something else and is left
/// alone rather than re-scored.
bool scoreStaysNCHW(Operation *op) {
  if (op->getNumResults() != 1)
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || type.getRank() != 4 || type.getEncoding())
    return false;
  return !preferNHWC(type.getNumElements(), getChannelExtent(type));
}

void AssignLayoutPass::runOnOperation() {
  func::FuncOp function = getOperation();

  // The choice, first and separately, so that the count reports what the graph
  // asked before any rewrite changed it.
  int64_t kept = 0;
  function.walk([&](Operation *op) {
    if (!isa<Conv2DOp, MaxPool2DOp, AvgPool2DOp, ReluOp, AddOp, MulOp>(op))
      return;
    if (scoreStaysNCHW(op))
      ++kept;
  });

  // The sink, then the fold, in that order and to a fixed point over the
  // whole function. Sinking moves a transpose strictly later in its block and
  // folding removes one, so neither can undo the other and the iteration ends;
  // the bound is the operation count, which is what makes that an argument
  // rather than a hope.
  int64_t sunk = 0;
  int64_t folded = 0;

  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<ReluOp> relus;
    function.walk([&](ReluOp op) { relus.push_back(op); });
    for (ReluOp op : relus) {
      if (sinkThroughRelu(op)) {
        ++sunk;
        changed = true;
      }
    }

    // The worklist is walked by index rather than by value because a fold can
    // orphan the inner transpose, which may itself still be in the list. It is
    // dropped from the list before it is erased, so nothing here reads an
    // operation that has already gone.
    SmallVector<TransposeOp> transposes;
    function.walk([&](TransposeOp op) { transposes.push_back(op); });
    for (size_t index = 0; index < transposes.size(); ++index) {
      TransposeOp op = transposes[index];
      if (!op)
        continue;
      TransposeOp orphaned;
      if (!foldTranspose(op, orphaned))
        continue;
      ++folded;
      changed = true;
      transposes[index] = nullptr;
      if (!orphaned)
        continue;
      for (TransposeOp &entry : transposes)
        if (entry == orphaned)
          entry = nullptr;
      // The destination it leaves behind has no user either, and it is erased
      // here for the reason `eraseIfDeadEmpty` gives: there is no
      // canonicalization between this pass and the lowering, so a destination
      // left dead here becomes a scratchpad allocation for a value nothing
      // reads. The order matters, because the destination has a user until the
      // transpose that reads it is gone.
      Value orphanedDestination = orphaned.getDestination();
      orphaned.erase();
      eraseIfDeadEmpty(orphanedDestination);
    }
  }

  keptNCHW += kept;
  sunkTransposes += sunk;
  foldedPairs += folded;
}

} // namespace
