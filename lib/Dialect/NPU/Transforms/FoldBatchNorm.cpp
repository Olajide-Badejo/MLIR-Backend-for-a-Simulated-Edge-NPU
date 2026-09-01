//===- FoldBatchNorm.cpp - batch norm into the convolution ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-fold-batchnorm`, at `-O2`, before fusion.
//
// **The identity.** An inference time batch norm is a per channel affine map,
// and a convolution is already an affine map with per output channel weights
// and a per output channel bias, so the composition is another convolution:
//
//     invStd = 1 / sqrt(variance + epsilon)
//     scale  = gamma * invStd
//     shift  = beta - mean * scale
//     w'[f]  = w[f] * scale[f]
//     b'[f]  = b[f] * scale[f] + shift[f]
//
// **The evaluation order is written out because it is observable.** Floating
// point multiplication is not associative and these constants are computed in
// `f32`, so a reader comparing this against onnxruntime needs to know which of
// several algebraically equal forms produced the number. It is deliberately the
// same order `-npu-lower-to-npuisa` uses for the decomposition of a batch norm
// this pass did not fold, so the two spellings of one identity agree with each
// other rather than by luck.
//
// **This is the pass that moves numbers.** Before the fold the machine
// convolves and then scales the result; after it the machine convolves with
// pre scaled weights, so every product in the reduction is scaled rather than
// the sum being scaled once at the end. Equal in exact arithmetic, different in
// the last bits of `f32`. `docs/BREAKING_CHANGES.md` declared the movement
// before the commit that turned it on, and records the measured magnitude.
//
// **What it rewrites, and what it does not create.** The convolution is
// mutated in place and the batch norm is replaced by the convolution's own
// result, rather than a second convolution being built beside the first. The
// batch norm's destination is left with no user and `-canonicalize` removes it,
// which is the canonicalization Section 12's table asks for after this pass.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

namespace mlir::npu {
#define GEN_PASS_DEF_NPUFOLDBATCHNORM
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

/// The dense f32 values behind a value, when that value is an `npu.constant`.
std::optional<SmallVector<APFloat>> constantValues(Value operand) {
  if (!operand)
    return std::nullopt;
  auto producer = operand.getDefiningOp<ConstantOp>();
  if (!producer)
    return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(producer.getValue());
  if (!dense || !isa<FloatType>(dense.getType().getElementType()))
    return std::nullopt;
  return llvm::to_vector(dense.getValues<APFloat>());
}

// `impl` is qualified in full: `mlir` and `mlir::npu` both have one and both
// are in scope through the using directives above.
struct FoldBatchNormPass
    : public mlir::npu::impl::NPUFoldBatchNormBase<FoldBatchNormPass> {
  using mlir::npu::impl::NPUFoldBatchNormBase<
      FoldBatchNormPass>::NPUFoldBatchNormBase;

  void runOnOperation() override;
};

/// Folds one batch norm, or reports why it did not.
///
/// Returning `false` is a non match and never a diagnostic. Section 5.2 makes
/// an unfolded batch norm legal rather than a hard error: the lowering
/// decomposes it into a multiply and an add, so a guard this pass declines is a
/// program that still compiles and still computes the right answer.
bool fold(BatchNormOp op, OpBuilder &builder) {
  auto conv = op.getInput().getDefiningOp<Conv2DOp>();
  if (!conv)
    return false;

  // Exactly one use, and it is this batch norm. A convolution two consumers
  // read cannot be rewritten in place without changing what the other one
  // sees, and cloning it instead would double the weights in DRAM to save one
  // scaling pass, which is the wrong trade on a machine whose bottleneck is
  // the DRAM port.
  if (!conv.getResult().hasOneUse())
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  auto convType = dyn_cast<RankedTensorType>(conv.getResult().getType());
  if (!resultType || !convType || resultType != convType)
    return false;
  if (resultType.getRank() != 4 || resultType.getEncoding())
    return false;

  auto filterType = dyn_cast<RankedTensorType>(conv.getFilter().getType());
  if (!filterType || filterType.getRank() != 4)
    return false;

  const int64_t channels = resultType.getDimSize(1);
  if (filterType.getDimSize(0) != channels || channels <= 0)
    return false;

  std::optional<SmallVector<APFloat>> filter = constantValues(conv.getFilter());
  std::optional<SmallVector<APFloat>> gamma = constantValues(op.getGamma());
  std::optional<SmallVector<APFloat>> beta = constantValues(op.getBeta());
  std::optional<SmallVector<APFloat>> mean = constantValues(op.getMean());
  std::optional<SmallVector<APFloat>> variance = constantValues(op.getVariance());
  if (!filter || !gamma || !beta || !mean || !variance)
    return false;
  if (static_cast<int64_t>(gamma->size()) != channels ||
      static_cast<int64_t>(beta->size()) != channels ||
      static_cast<int64_t>(mean->size()) != channels ||
      static_cast<int64_t>(variance->size()) != channels)
    return false;

  // A convolution with a bias folds its bias too; one without gets the shift
  // as its bias. Both are the same expression with `b` set to zero, and both
  // are written out rather than shared, because a reader checking the
  // arithmetic against the comment at the top of this file should see it once
  // per case rather than reconstruct it from a conditional.
  SmallVector<APFloat> bias;
  if (conv.getBias()) {
    std::optional<SmallVector<APFloat>> existing = constantValues(conv.getBias());
    if (!existing || static_cast<int64_t>(existing->size()) != channels)
      return false;
    bias = std::move(*existing);
  }

  const float epsilon = op.getEpsilon().convertToFloat();
  SmallVector<float> scale;
  SmallVector<float> shift;
  scale.reserve(channels);
  shift.reserve(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    const float denominator = (*variance)[channel].convertToFloat() + epsilon;
    // Not a diagnostic. `-npu-lower-to-npuisa` emits one for exactly this case
    // and it is the layer that has to, because by then the batch norm has to
    // become instructions; here the honest answer is to decline and leave it
    // for that layer to refuse with the numbers in the message.
    if (!(denominator > 0.0f))
      return false;
    const float invStd = 1.0f / std::sqrt(denominator);
    const float channelScale = (*gamma)[channel].convertToFloat() * invStd;
    scale.push_back(channelScale);
    shift.push_back((*beta)[channel].convertToFloat() -
                    (*mean)[channel].convertToFloat() * channelScale);
  }

  // The filter is `(F, C / group, KH, KW)` in row major order, so the output
  // channel of a flat index is the index divided by the size of one filter.
  const int64_t perFilter = filterType.getNumElements() / channels;
  SmallVector<APFloat> foldedFilter;
  foldedFilter.reserve(filter->size());
  for (auto [index, element] : llvm::enumerate(*filter))
    foldedFilter.push_back(APFloat(element.convertToFloat() *
                                   scale[static_cast<int64_t>(index) / perFilter]));

  SmallVector<APFloat> foldedBias;
  foldedBias.reserve(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    const float existing = bias.empty() ? 0.0f : bias[channel].convertToFloat();
    foldedBias.push_back(APFloat(existing * scale[channel] + shift[channel]));
  }

  auto biasType =
      RankedTensorType::get({channels}, resultType.getElementType());

  builder.setInsertionPoint(conv);
  Value newFilter =
      ConstantOp::create(builder, conv.getLoc(), filterType,
                         DenseElementsAttr::get(filterType, foldedFilter))
          .getResult();
  Value newBias =
      ConstantOp::create(builder, conv.getLoc(), biasType,
                         DenseElementsAttr::get(biasType, foldedBias))
          .getResult();

  // `setOperands` rather than a second convolution. The optional bias has no
  // operand segment attribute, because it is the only non fixed operand and
  // the count therefore determines its presence, so growing the list from
  // three operands to four is all that adding a bias requires.
  conv->setOperands({conv.getInput(), newFilter, newBias, conv.getDestination()});

  op.getResult().replaceAllUsesWith(conv.getResult());
  op.erase();
  return true;
}

void FoldBatchNormPass::runOnOperation() {
  func::FuncOp function = getOperation();
  OpBuilder builder(function.getContext());

  SmallVector<BatchNormOp> candidates;
  function.walk([&](BatchNormOp op) { candidates.push_back(op); });

  int64_t folded = 0;
  for (BatchNormOp op : candidates)
    if (fold(op, builder))
      ++folded;

  foldedBatchNorms += folded;
}

} // namespace
