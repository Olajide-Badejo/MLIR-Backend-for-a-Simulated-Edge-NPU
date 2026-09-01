//===- LowerNPUToNPUISA.cpp - npu to npuisa lowering ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The dialect conversion from the `npu` tensor dialect to the `npuisa`
// instruction dialect: Section 5.1's fifth line, and Section 8's memory model.
//
// **The operator map, which is this pass's contract.** Every operation the
// `npu` dialect has appears here, because law 2 says an operator that cannot be
// lowered does not get to exist, and a table with a hole in it is the honest
// place for that to be visible.
//
//   npu.constant     npuisa.const in DRAM, plus the one dma_load that brings it
//                    on chip. One load however many instructions read it.
//   npu.conv2d       npuisa.conv2d, attributes carried across unchanged.
//   npu.matmul       npuisa.matmul.
//   npu.add          npuisa.add. A rank 1 right hand operand first becomes a
//                    stride 0 broadcast view, per ADR 0005.
//   npu.mul          npuisa.mul, the same way.
//   npu.relu         npuisa.relu.
//   npu.max_pool2d   npuisa.pool_max. The mnemonic changes because the tensor
//   npu.avg_pool2d   npuisa.pool_avg. level is named after ONNX and the
//                    instruction level after the opcode.
//   npu.reshape      npuisa.reshape, with a destination this pass allocates,
//                    because the tensor form has none and the buffer form needs
//                    one.
//   npu.transpose    npuisa.transpose.
//   npu.concat       npuisa.concat.
//   npu.batch_norm   a multiply and an add over per channel constants computed
//                    at rewrite time. There is no batch norm instruction and
//                    there is not meant to be one.
//   npu.fused_op     no instruction at all: the region is flattened into its
//                    parent, which is what keeps the chain's intermediate in
//                    the scratchpad.
//   npu.yield        erased with the region it terminated.
//
// The pass is three stages rather than one set of patterns, and the split is
// worth stating because it is not the obvious shape.
//
//   1. Validate. Everything this lowering refuses is refused here, by name,
//      before a single operation has been rewritten. A diagnostic emitted from
//      inside a conversion pattern competes with the framework's own "failed to
//      legalize operation" message and loses; emitted here it is the only thing
//      the reader sees.
//
//   2. Expand, on tensors. An `npu.fused_op` region is flattened, and an
//      `npu.batch_norm` the folding pass did not fold is decomposed into a
//      multiply and an add. Both are tensor level identities. Written against
//      memrefs each would have to allocate its own intermediate and thread its
//      own destination, which is work stage 3 already does once.
//
//   3. Convert. ConversionTarget, TypeConverter, applyPartialConversion, which
//      is what this phase's roadmap entry asks for by name.
//
// The `one-shot-bufferize` attempt the roadmap requires was made, and its
// outcome is recorded with the commands and their output in
// docs/adr/0006-lowering-mechanism-and-the-bufferization-attempt.md. The short
// version, because a reader of this file should not have to go and find it:
// One-Shot Bufferize assigns one memory space per tensor **type**, through a
// hook that takes the type and nothing else, and this machine needs two spaces
// for one value plus a copy between them. That is a different question rather
// than a gap in the configuration. So the `memref` type and its memory space
// attributes are adopted, which Section 5.2 calls the consequential half, and
// the bufferization analysis is not.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <optional>
#include <string>

namespace mlir::npuisa {
#define GEN_PASS_DEF_NPULOWERTONPUISA
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"
} // namespace mlir::npuisa

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// The type conversion: a tensor becomes a buffer in a named memory space.
//===----------------------------------------------------------------------===//

/// Renders a type or an attribute into a string, for a diagnostic.
template <typename T> std::string describe(T value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << value;
  return text;
}

/// The one place in this project where a tensor becomes a buffer.
///
/// On failure `reason` is filled with a clause naming why, so that the caller
/// names the operation and the operand while this function names the type. The
/// decision and its explanation come out of the same branches on purpose: two
/// functions would eventually disagree, and a diagnostic that contradicts the
/// refusal it explains is worse than no diagnostic at all.
///
/// **The layout encoding of Section 5.5 is materialised here, as the memref's
/// strided layout map.** A tensor's extents are written in the order its layout
/// names, so an NHWC tensor is written N, H, W, C. A buffer below this level is
/// always NCHW, which is the order every `npuisa` verifier reads and the order
/// the simulator's kernels index. So the extents are permuted back into NCHW
/// and the permutation is carried in the strides instead, over the same
/// underlying bytes. An NHWC buffer therefore differs from an NCHW one in its
/// strides and not in its shape, which is exactly what the binary format's
/// operand stride fields and the cost model's stride term exist to read. The
/// alternative, keeping the extents in the order the layout wrote them, makes
/// the strides contiguous again and leaves the layout with no trace at all
/// below the tensor level, which is the outcome Section 5.5 says would make
/// `-npu-assign-layout` a pass whose delta is structurally zero.
FailureOr<MemRefType> convertTensorType(Type type, Attribute space,
                                        std::string *reason = nullptr) {
  auto refuse = [&](const Twine &why) -> FailureOr<MemRefType> {
    if (reason)
      *reason = why.str();
    return failure();
  };

  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return refuse("it is not a ranked tensor, and a memory space is a property "
                  "of a buffer");
  if (!tensor.hasStaticShape())
    return refuse("it has a dynamic extent, and nothing below this level can "
                  "represent one");

  Type element = tensor.getElementType();
  if (!element.isF32() && !element.isInteger(8))
    return refuse("its element type " + describe(element) +
                  " is not one this machine has a memory for; the two are f32 "
                  "and i8");

  Attribute encoding = tensor.getEncoding();
  if (encoding && !isa<npu::LayoutAttr>(encoding))
    return refuse("its encoding " + describe(encoding) +
                  " is not an #npu.layout, and no other encoding has a meaning "
                  "below the tensor level");
  if (encoding && tensor.getRank() != 4)
    return refuse("it carries a layout encoding at rank " +
                  Twine(tensor.getRank()) +
                  ", and only a rank 4 tensor has an NCHW or an NHWC reading");

  MLIRContext *context = tensor.getContext();
  if (!npu::isNHWC(tensor))
    return MemRefType::get(tensor.getShape(), element,
                           MemRefLayoutAttrInterface(), space);

  const ArrayRef<int64_t> written = tensor.getShape();
  const int64_t batch = written[0];
  const int64_t height = written[1];
  const int64_t width = written[2];
  const int64_t channels = written[3];

  const SmallVector<int64_t> shape = {batch, channels, height, width};
  const SmallVector<int64_t> strides = {height * width * channels, 1,
                                        width * channels, channels};
  return MemRefType::get(
      shape, element, StridedLayoutAttr::get(context, /*offset=*/0, strides),
      space);
}

/// The two spellings of a tensor's buffer type.
///
/// Both dereference without checking, and that is safe rather than sloppy:
/// stage 1 has already run `convertTensorType` over every type in the module
/// and refused the ones that do not convert, so by the time a pattern runs
/// there is no unconvertible type left to meet.
MemRefType scratchpadTypeOf(Type type) {
  return *convertTensorType(type, npu::ScratchpadAttr::get(type.getContext()));
}

MemRefType dramTypeOf(Type type) {
  return *convertTensorType(type, npu::DramAttr::get(type.getContext()));
}

//===----------------------------------------------------------------------===//
// Stage 1: validation, with a named diagnostic for everything refused.
//===----------------------------------------------------------------------===//

/// Every tensor typed operand and result of an operation converts, or the
/// operation is named along with the operand and the reason.
LogicalResult validateTypes(Operation *op) {
  Attribute space = npu::ScratchpadAttr::get(op->getContext());
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<TensorType>(operand.get().getType()))
      continue;
    std::string reason;
    if (succeeded(convertTensorType(operand.get().getType(), space, &reason)))
      continue;
    return op->emitError()
           << "operand " << operand.getOperandNumber() << " has type "
           << operand.get().getType()
           << ", which cannot be assigned a memory space, because " << reason;
  }
  for (OpResult result : op->getResults()) {
    if (!isa<TensorType>(result.getType()))
      continue;
    std::string reason;
    if (succeeded(convertTensorType(result.getType(), space, &reason)))
      continue;
    return op->emitError()
           << "result " << result.getResultNumber() << " has type "
           << result.getType()
           << ", which cannot be assigned a memory space, because " << reason;
  }
  return success();
}

/// The function level refusals: a declaration, multiple blocks, and a signature
/// carrying a type that has no buffer.
LogicalResult validateFunction(func::FuncOp function) {
  if (function.isExternal())
    return function.emitError()
           << "@" << function.getName()
           << " is a declaration, and this compiler has no calls and no "
              "linking: every function it lowers carries its own body";

  // Section 5.2: inference graphs are static directed acyclic graphs and this
  // instruction set has no branch instructions, so a lowered function is one
  // straight line stream. Multiple blocks are diagnosed rather than ignored.
  if (!function.getBody().hasOneBlock())
    return function.emitError()
           << "the lowering requires a single block function body, but @"
           << function.getName() << " has "
           << std::distance(function.getBody().begin(), function.getBody().end())
           << " blocks. This instruction set has no branch instructions, so a "
              "tiled loop is fully unrolled before lowering rather than lowered "
              "as control flow";

  // Only the tensors are asked about. A signature that already carries buffers
  // is a function this pass has already lowered, or one that was written that
  // way, and either is left alone: it is legal under the conversion target, so
  // asking whether its memrefs can be given a memory space would refuse a
  // program for the crime of already being correct. Idempotence falls out of
  // the same rule, which is what lets a pipeline run this pass without first
  // establishing whether something else already did.
  Attribute dram = npu::DramAttr::get(function.getContext());
  for (BlockArgument argument : function.getArguments()) {
    if (!isa<TensorType>(argument.getType()))
      continue;
    std::string reason;
    if (succeeded(convertTensorType(argument.getType(), dram, &reason)))
      continue;
    return function.emitError()
           << "argument " << argument.getArgNumber() << " of @"
           << function.getName() << " has type " << argument.getType()
           << ", which cannot be assigned a memory space, because " << reason;
  }
  for (auto [index, result] :
       llvm::enumerate(function.getFunctionType().getResults())) {
    if (!isa<TensorType>(result))
      continue;
    std::string reason;
    if (succeeded(convertTensorType(result, dram, &reason)))
      continue;
    return function.emitError()
           << "result " << index << " of @" << function.getName() << " has type "
           << result << ", which cannot be assigned a memory space, because "
           << reason;
  }
  return success();
}

/// The operation level refusals.
LogicalResult validateOperation(Operation *op) {
  if (Dialect *dialect = op->getDialect();
      dialect && dialect->getNamespace() == "scf")
    return op->emitError()
           << "an scf operation reaches the lowering, which this instruction "
              "set cannot represent: it has no branch instructions, so a tiled "
              "loop is fully unrolled before lowering. Section 5.2 makes this a "
              "diagnostic rather than an ignored region";

  // The two layout cases this lowering does not represent, named here so that
  // `-npu-assign-layout` arrives at a stated rule rather than at a verifier
  // failure from inside a pass. Neither is reachable from the frontend, which
  // emits no layout encodings at all. Both become reachable the day that pass
  // lands, and both are its work rather than a hole here.
  if (auto constant = dyn_cast<npu::ConstantOp>(op)) {
    auto tensor = cast<RankedTensorType>(constant.getType());
    if (Attribute encoding = tensor.getEncoding())
      return op->emitError()
             << "this constant carries the layout encoding " << encoding
             << ", and the lowering does not permute constant data. A layout "
                "assignment pass materialises the permuted constant itself "
                "rather than leaving the permutation to be done here";
  }
  if (auto transpose = dyn_cast<npu::TransposeOp>(op)) {
    auto input = cast<RankedTensorType>(transpose.getInput().getType());
    auto result = cast<RankedTensorType>(transpose.getType());
    if (input.getRank() == 4 && result.getRank() == 4 &&
        !npu::sameLayout(input, result))
      return op->emitError()
             << "this transpose changes the layout as well as the extents, from "
             << npu::describeLayout(input) << " to "
             << npu::describeLayout(result)
             << ", and the lowering represents a permutation of extents only. A "
                "layout assignment pass folds its own inverse transposes rather "
                "than leaving one here";
  }

  return validateTypes(op);
}

LogicalResult validate(ModuleOp module) {
  WalkResult walked = module.walk([](func::FuncOp function) {
    return failed(validateFunction(function)) ? WalkResult::interrupt()
                                              : WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();

  // Pre order, so that an `scf.for` is named rather than the `scf.yield`
  // inside it. A walk defaults to post order, and under that default the
  // diagnostic for a loop points at its terminator, which is the one operation
  // in the loop a reader does not need to be told about.
  walked = module.walk<WalkOrder::PreOrder>([](Operation *op) {
    return failed(validateOperation(op)) ? WalkResult::interrupt()
                                         : WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Stage 2: the two tensor level expansions.
//===----------------------------------------------------------------------===//

/// Flattens an `npu.fused_op` into its parent block.
///
/// The region's block arguments become the operation's operands and the yielded
/// value becomes the result. The operation is `IsolatedFromAbove`, so the block
/// arguments are the only route from inside the region to anything outside it,
/// and the substitution is therefore complete by construction.
///
/// **Flattening is what makes fusion mean what Section 5.2 says it means.**
/// Afterwards the chain's intermediate is an ordinary value that stage 3
/// allocates in the scratchpad and never stores to DRAM, so a fused convolution
/// and its activation have no DMA between them, which is the property
/// `test/Dialect/NPUISA/dma-boundaries.mlir` asserts and the property the energy
/// argument of Section 5.2 rests on. Nothing else in this pass has to know that
/// a fused region was ever there, and that is the point: the alternative, a
/// second set of patterns lowering operations inside a region while threading a
/// destination through the yield, would be the whole lowering written twice.
void flattenFusedOp(npu::FusedOp fused, IRRewriter &rewriter) {
  Block &body = fused.getBody().front();
  auto yield = cast<npu::YieldOp>(body.getTerminator());
  Value yielded = yield.getValue();

  rewriter.inlineBlockBefore(&body, fused.getOperation(), fused.getInputs());
  rewriter.eraseOp(yield);
  rewriter.replaceOp(fused, yielded);
}

/// One rank 1 parameter of a batch norm, as constant data.
FailureOr<DenseElementsAttr> constantParameter(npu::BatchNormOp op,
                                               Value parameter,
                                               StringRef name) {
  auto constant = parameter.getDefiningOp<npu::ConstantOp>();
  if (!constant) {
    op.emitError()
        << "the " << name
        << " operand of this batch norm is not an npu.constant, and the "
           "decomposition computes its multiplier and its addend at rewrite "
           "time. A batch norm whose parameters are computed rather than stored "
           "is not an inference graph this compiler accepts";
    return failure();
  }
  auto data = dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!data || !isa<FloatType>(data.getType().getElementType())) {
    op.emitError() << "the " << name
                   << " operand of this batch norm is an npu.constant whose "
                      "value is not dense floating point data";
    return failure();
  }
  return data;
}

/// Decomposes an `npu.batch_norm` into a multiply and an add.
///
///   invStd = 1 / sqrt(variance + epsilon)
///   scale  = gamma * invStd
///   shift  = beta - mean * scale
///   y      = x * scale + shift
///
/// **The evaluation order above is the order the code below evaluates in, and
/// it is written down because it is observable.** Floating point multiplication
/// is not associative, so a reader comparing this against onnxruntime needs to
/// know which of several algebraically equal forms produced the number.
///
/// Both halves come out as the rank 1 channel broadcast of ADR 0005, which is
/// why that record calls the relaxation cheaper than it looks: per channel
/// arithmetic against a rank 4 activation already had to be lowered for
/// `npu.add` and `npu.mul`, and this reuses that path rather than adding a
/// second one. An unfolded batch norm is therefore legal rather than a hard
/// error, exactly as Section 5.2 requires.
LogicalResult decomposeBatchNorm(npu::BatchNormOp op, IRRewriter &rewriter) {
  auto resultType = cast<RankedTensorType>(op.getType());
  const int64_t channels = npu::getChannelExtent(resultType);

  FailureOr<DenseElementsAttr> gamma =
      constantParameter(op, op.getGamma(), "gamma");
  FailureOr<DenseElementsAttr> beta = constantParameter(op, op.getBeta(), "beta");
  FailureOr<DenseElementsAttr> mean = constantParameter(op, op.getMean(), "mean");
  FailureOr<DenseElementsAttr> variance =
      constantParameter(op, op.getVariance(), "variance");
  if (failed(gamma) || failed(beta) || failed(mean) || failed(variance))
    return failure();

  const float epsilon = op.getEpsilonAttr().getValue().convertToFloat();

  const SmallVector<APFloat> gammaValues =
      llvm::to_vector(gamma->getValues<APFloat>());
  const SmallVector<APFloat> betaValues =
      llvm::to_vector(beta->getValues<APFloat>());
  const SmallVector<APFloat> meanValues =
      llvm::to_vector(mean->getValues<APFloat>());
  const SmallVector<APFloat> varianceValues =
      llvm::to_vector(variance->getValues<APFloat>());

  SmallVector<APFloat> scaleValues;
  SmallVector<APFloat> shiftValues;
  scaleValues.reserve(channels);
  shiftValues.reserve(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    const float denominator = varianceValues[channel].convertToFloat() + epsilon;
    if (!(denominator > 0.0f))
      return op.emitError()
             << "this batch norm cannot be decomposed on channel " << channel
             << ": the variance " << varianceValues[channel].convertToFloat()
             << " plus the epsilon " << epsilon << " is " << denominator
             << ", and the decomposition takes a square root of it";
    const float invStd = 1.0f / std::sqrt(denominator);
    const float scale = gammaValues[channel].convertToFloat() * invStd;
    const float shift = betaValues[channel].convertToFloat() -
                        meanValues[channel].convertToFloat() * scale;
    scaleValues.push_back(APFloat(scale));
    shiftValues.push_back(APFloat(shift));
  }

  auto parameterType = RankedTensorType::get(
      {channels},
      cast<RankedTensorType>(op.getGamma().getType()).getElementType());

  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Value scale = npu::ConstantOp::create(
      rewriter, loc, parameterType,
      DenseElementsAttr::get(parameterType, ArrayRef<APFloat>(scaleValues)));
  Value shift = npu::ConstantOp::create(
      rewriter, loc, parameterType,
      DenseElementsAttr::get(parameterType, ArrayRef<APFloat>(shiftValues)));

  Value intermediate =
      tensor::EmptyOp::create(rewriter, loc, TypeRange{resultType}, ValueRange{})
          .getResult();
  Value product = npu::MulOp::create(rewriter, loc, resultType, op.getInput(),
                                     scale, intermediate)
                      .getResult();
  Value sum = npu::AddOp::create(rewriter, loc, resultType, product, shift,
                                 op.getDestination())
                  .getResult();
  rewriter.replaceOp(op, sum);
  return success();
}

LogicalResult expand(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  // A walk is post order, so a nested fused operation is flattened into its
  // parent before that parent is flattened into the function.
  SmallVector<npu::FusedOp> fused;
  module.walk([&](npu::FusedOp op) { fused.push_back(op); });
  for (npu::FusedOp op : fused)
    flattenFusedOp(op, rewriter);

  SmallVector<npu::BatchNormOp> batchNorms;
  module.walk([&](npu::BatchNormOp op) { batchNorms.push_back(op); });
  for (npu::BatchNormOp op : batchNorms)
    if (failed(decomposeBatchNorm(op, rewriter)))
      return failure();

  // **One destination, one buffer.** *Added at P9, as the fix for D-0034.*
  //
  // A `tensor.empty` is a value with no contents, so two operations that use
  // the same one as a destination are two pure functions of the same
  // meaningless input, and at the tensor level that is entirely correct. This
  // pass is the layer at which it stops being correct: it converts one
  // `tensor.empty` into one `memref.alloc`, so a shared destination becomes two
  // instructions writing one buffer, and when the second of them also *reads*
  // that buffer through a window the program is simply wrong.
  //
  // Nothing produced this shape before `-O2` existed, because the importer
  // emits one `tensor.empty` per compute operation. `-cse` produces it in one
  // step, because two `tensor.empty` operations of the same type are identical
  // operations with no operands and merging them is exactly what a common
  // subexpression eliminator is for. The fix belongs here rather than in `-cse`
  // for the same reason the aliasing rule belongs here: this is where a value
  // becomes a buffer.
  //
  // Every use after the first gets its own clone. The clone goes above the
  // original, which is above every use of it, so dominance is preserved without
  // a walk.
  SmallVector<tensor::EmptyOp> destinations;
  module.walk([&](tensor::EmptyOp op) { destinations.push_back(op); });
  for (tensor::EmptyOp op : destinations) {
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : op.getResult().getUses())
      uses.push_back(&use);
    if (uses.size() < 2)
      continue;
    rewriter.setInsertionPoint(op);
    for (OpOperand *use : llvm::drop_begin(uses))
      use->set(rewriter.clone(*op.getOperation())->getResult(0));
  }

  // **And each destination is moved down to the operation that writes it.**
  // *Added at P9, beside the split above and for the allocator rather than for
  // correctness.*
  //
  // A `memref.alloc` is live from where it is defined, and Section 13.1's
  // liveness runs from the allocation to the last operation that touches the
  // buffer. So a destination defined earlier than it is written is a buffer the
  // allocator must keep out of everyone else's way for longer than the program
  // needs it.
  //
  // `-npu-fuse-ops` produces exactly that. A region takes every value it reads
  // as an operand, destinations included, so both of a fused chain's
  // destinations are defined *above* the region and the flattening leaves them
  // there. On LeNet at `-O2` that raised the sweep line peak from 194624 bytes
  // to 195040 and the tight budget cell of Section 15, measured at P8 and
  // frozen, stopped fitting: an optimization level that could not compile a
  // program `-O0` compiled, from a pass that changed no instruction.
  //
  // The importer already emits each `tensor.empty` immediately above its
  // writer, so this is a no operation at `-O0`. Every destination has exactly
  // one use by the time this runs, because the split above gave it one.
  // Recollected, because the split above created clones and they need moving
  // for the same reason the originals do.
  destinations.clear();
  module.walk([&](tensor::EmptyOp op) { destinations.push_back(op); });
  for (tensor::EmptyOp op : destinations) {
    if (!op.getResult().hasOneUse())
      continue;
    Operation *user = *op.getResult().getUsers().begin();
    if (user->getBlock() != op->getBlock())
      continue;
    if (user->getPrevNode() != op.getOperation())
      op->moveBefore(user);
  }

  // **A constant is moved to just above its first reader.** *Added at P9, as
  // the fix for D-0035.*
  //
  // This pass emits one `npuisa.const` and one `npuisa.dma_load` at the position
  // of each `npu.constant`, so where a constant sits in the block decides when
  // its bytes are fetched, and the two port cost model of Section 10.1 charges
  // exactly that: a transfer overlaps a computation only when the computation
  // does not depend on it. The importer emits every constant immediately above
  // its first use, so at `-O0` the loads interleave with the compute and the
  // overlap is high.
  //
  // MLIR's canonicalizer hoists every `ConstantLike` operation to the top of the
  // block, which is right for an operation whose cost is zero and wrong for one
  // that turns into a DRAM transfer. On LeNet at `-O1` it moved all eleven loads
  // above all the compute, in an order that put the *last* layer's weights
  // first, so the first convolution waited for essentially the whole 16441 cycle
  // transfer budget and the overlap fraction fell from 0.83 to 0.0005: 37 percent
  // more simulated cycles from an optimization level.
  //
  // The fix is here rather than in a scheduling pass because there is nothing to
  // schedule: this pass chooses where to put a transfer it is about to create,
  // and the answer is where the data is needed. It is a no operation at `-O0`,
  // where the importer's placement is already this one, which is why the `-O0`
  // baseline does not move.
  // **It sinks past computation and nothing else.** The insertion point is not
  // "immediately above the first user" but "above the run of constants and
  // destinations that immediately precedes the first user", which is where the
  // importer already puts one. Sinking to the tighter point would reorder the
  // `memref.alloc` operations of a program nothing had hoisted, which changes no
  // instruction and no number and would still have made every `-O0` lit test in
  // this file disagree with the IR for no reason a reader could act on.
  SmallVector<npu::ConstantOp> toSink;
  module.walk([&](npu::ConstantOp op) { toSink.push_back(op); });
  for (npu::ConstantOp op : toSink) {
    Operation *earliest = nullptr;
    for (Operation *user : op.getResult().getUsers()) {
      if (user->getBlock() != op->getBlock())
        continue;
      if (!earliest || user->isBeforeInBlock(earliest))
        earliest = user;
    }
    // No user in this block is either a dead constant, which the sweep below
    // erases, or one read from somewhere this cannot reason about, and moving
    // it would be moving it past nothing or into a place it does not dominate.
    if (!earliest)
      continue;

    Operation *anchor = earliest;
    bool alreadyPlaced = false;
    while (Operation *previous = anchor->getPrevNode()) {
      if (previous == op.getOperation()) {
        alreadyPlaced = true;
        break;
      }
      if (!isa<tensor::EmptyOp, npu::ConstantOp>(previous))
        break;
      anchor = previous;
    }
    if (!alreadyPlaced)
      op->moveBefore(anchor);
  }

  // A constant nothing reads is erased before the conversion sees it, and this
  // is load bearing rather than tidiness. The decomposition above consumes its
  // four parameters at rewrite time and leaves them behind with no uses; a
  // constant that reached stage 3 unused would still become an `npuisa.const`
  // in DRAM and an `npuisa.dma_load` bringing it on chip, because a transfer
  // has memory effects and is not dead code that anything downstream would
  // remove. That is four transfers per unfolded batch norm of data no
  // instruction reads, which is exactly the unexplained DRAM traffic Section 8
  // calls a defect. One pass suffices: a constant has no operands, so erasing
  // one cannot make another dead. See docs/DEFECT_LOG.md D-0016.
  SmallVector<npu::ConstantOp> constants;
  module.walk([&](npu::ConstantOp op) { constants.push_back(op); });
  for (npu::ConstantOp op : constants)
    if (op.getResult().use_empty())
      rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Stage 3: the conversion.
//===----------------------------------------------------------------------===//

/// The state the patterns share.
///
/// One map, and it carries half of Section 8's boundary invariant. A function
/// argument is a DRAM buffer, and the compute instructions of this machine
/// address the scratchpad and nothing else, so each argument is loaded exactly
/// once when the signature is converted and every consumer reads the loaded
/// buffer out of here. "One load per DRAM value entering the scratchpad" is
/// therefore a property of the data structure rather than a rule each pattern
/// has to remember and could forget.
struct LoweringState {
  DenseMap<Value, Value> argumentBuffers;

  void recordArgumentBuffer(Value dram, Value scratchpad) {
    argumentBuffers.try_emplace(dram, scratchpad);
  }

  /// The scratchpad buffer a converted operand names.
  Value resolve(Value converted) const {
    // The conversion driver reconciles types with unrealized casts. Looking
    // through them here keeps this map keyed on the buffer itself rather than
    // on whichever spelling of it a particular pattern was handed.
    Value value = converted;
    while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        break;
      value = cast.getInputs().front();
    }
    if (Value buffer = argumentBuffers.lookup(value))
      return buffer;
    return value;
  }
};

/// The channel broadcast operand of ADR 0005.
///
/// That record obliges this phase and P7 together: a rank 1 right hand operand
/// is read with a channel stride of one and a spatial stride of zero. Here the
/// contract is a **type** rather than a flag. The rank 1 buffer is viewed at the
/// destination's extents, with strides that are 1 on the channel axis and 0 on
/// every other one, so the instruction's operand has the destination's shape,
/// which is what `npuisa.add` and `npuisa.mul` require of it, and reads the same
/// C values over and over, which is what a per channel scale means.
///
/// Buffers are NCHW below the tensor level whatever the layout was, so the
/// channel axis is 1 and this view is identical under both layouts. That is the
/// second thing the strided layout map buys: the broadcast does not have to know
/// where the channels sit, because by the time it runs they always sit in the
/// same place.
///
/// The view is a `memref.reinterpret_cast` rather than a new instruction because
/// a stride 0 read is not a machine operation, it is an addressing mode, and
/// Section 9.1 already carries operand strides in the `Instruction` record for
/// the layout decision. One field, two uses, and no new opcode: the instruction
/// set of Section 5.4 is closed and this stays inside it.
Value channelBroadcast(OpBuilder &builder, Location loc, Value rank1Buffer,
                       MemRefType destinationType) {
  SmallVector<int64_t> strides(destinationType.getRank(), 0);
  strides[1] = 1;

  auto viewType = MemRefType::get(
      destinationType.getShape(), destinationType.getElementType(),
      StridedLayoutAttr::get(builder.getContext(), /*offset=*/0, strides),
      destinationType.getMemorySpace());
  return memref::ReinterpretCastOp::create(
      builder, loc, viewType, rank1Buffer, /*offset=*/int64_t{0},
      /*sizes=*/destinationType.getShape(),
      /*strides=*/ArrayRef<int64_t>(strides));
}

/// The base of every pattern below, carrying the shared state.
template <typename SourceOp>
class NPULowering : public OpConversionPattern<SourceOp> {
public:
  NPULowering(const TypeConverter &converter, MLIRContext *context,
              LoweringState &state)
      : OpConversionPattern<SourceOp>(converter, context), state(state) {}

protected:
  /// The scratchpad buffer an adaptor operand names.
  Value buffer(Value converted) const { return state.resolve(converted); }

  LoweringState &state;
};

//===----------------------------------------------------------------------===//
// The function boundary.
//===----------------------------------------------------------------------===//

/// Converts a function signature to the boundary shape of Section 8.
///
/// Arguments become DRAM buffers. Results become **trailing DRAM arguments**
/// and the function returns nothing.
///
/// That second half is the decision in this pass worth arguing for rather than
/// stating. A lowered function that returned a buffer would have to have
/// allocated it, and nothing below this level allocates DRAM: `memref.alloc` in
/// the scratchpad is what the allocator assigns offsets into, and there is no
/// second allocator for the other space. Making the caller supply the output
/// buffer is what `test/Dialect/NPUISA/ops-memref.mlir` has called a lowered
/// function since P2, it is where the encoder reads its output regions from, and
/// it is what lets `npu-sim` write every output rather than the first.
class FuncOpLowering : public NPULowering<func::FuncOp> {
public:
  using NPULowering<func::FuncOp>::NPULowering;

  LogicalResult
  matchAndRewrite(func::FuncOp function, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType type = function.getFunctionType();

    SmallVector<Type> inputs;
    for (Type input : type.getInputs())
      inputs.push_back(dramTypeOf(input));
    SmallVector<Type> outputs;
    for (Type result : type.getResults())
      outputs.push_back(dramTypeOf(result));

    SmallVector<Type> converted(inputs);
    llvm::append_range(converted, outputs);

    Location loc = function.getLoc();
    auto lowered = func::FuncOp::create(
        rewriter, loc, function.getName(),
        FunctionType::get(getContext(), converted, /*results=*/{}));

    // The discardable attributes travel, because npuisa.scratchpad_budget is
    // one of them and it is an input to the allocator rather than an output of
    // it. The argument and result attribute dictionaries deliberately do not:
    // the argument count changes here, so copying a per position dictionary
    // across would be copying it onto different positions.
    lowered->setDiscardableAttrs(function->getDiscardableAttrDictionary());
    if (StringAttr visibility = function.getSymVisibilityAttr())
      lowered.setSymVisibilityAttr(visibility);

    // Which arguments are inputs and which are outputs, said in the IR rather
    // than left to be counted.
    //
    // *Added at P6.* P4 fixed the shape, the first N arguments are the model's
    // and the last M are the outputs it returned, and P4's handoff flagged
    // that if the encoder wanted the split explicit then an argument attribute
    // was the place. This is that decision. Nothing about the order changes;
    // what changes is that the encoder reads the split instead of inferring
    // it, so a pass that appends an argument in the wrong place is a
    // diagnostic rather than a silently mislabelled input region. The
    // attribute is written here because this is the pass that performs the
    // split and therefore the only one that knows the answer for certain.
    auto in = rewriter.getStringAttr(npuisa::kArgKindIn);
    auto out = rewriter.getStringAttr(npuisa::kArgKindOut);
    for (size_t index = 0; index < converted.size(); ++index)
      lowered.setArgAttr(static_cast<unsigned>(index), npuisa::kArgKindAttrName,
                         index < inputs.size() ? in : out);

    // Which arguments the body actually reads has to be asked before the
    // signature is converted, because afterwards the original arguments are
    // gone. An argument nothing reads gets no load: a transfer whose result is
    // never used is DRAM traffic, and Section 8 makes unexplained DRAM traffic
    // a bug rather than a style question.
    SmallVector<bool> isRead;
    for (BlockArgument argument : function.getArguments())
      isRead.push_back(!argument.use_empty());

    rewriter.inlineRegionBefore(function.getBody(), lowered.getBody(),
                                lowered.getBody().end());

    TypeConverter::SignatureConversion signature(type.getNumInputs());
    for (auto [index, input] : llvm::enumerate(inputs))
      signature.addInputs(index, input);
    signature.addInputs(outputs);

    Block *entry = rewriter.applySignatureConversion(&lowered.getBody().front(),
                                                     signature,
                                                     getTypeConverter());

    rewriter.setInsertionPointToStart(entry);
    for (auto [index, input] : llvm::enumerate(type.getInputs())) {
      if (!isRead[index])
        continue;
      Value dram = entry->getArgument(index);
      Value scratchpad =
          memref::AllocOp::create(rewriter, loc, scratchpadTypeOf(input));
      npuisa::DmaLoadOp::create(rewriter, loc, dram, scratchpad);
      state.recordArgumentBuffer(dram, scratchpad);
    }

    rewriter.eraseOp(function);
    return success();
  }
};

/// One `npuisa.dma_store` per returned value, into the out parameter the
/// function gained for it.
class ReturnOpLowering : public NPULowering<func::ReturnOp> {
public:
  using NPULowering<func::ReturnOp>::NPULowering;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto function = op->getParentOfType<func::FuncOp>();

    // The conversion driver legalizes in pre order, so the enclosing function
    // has already gained its out parameters by the time this runs. The check is
    // here rather than assumed because that ordering is the driver's property
    // and not this pass's, and a silent read of the wrong arguments would
    // produce a program that stores into its own input.
    if (function.getFunctionType().getNumResults() != 0)
      return op.emitError()
             << "the enclosing function still has results when its return is "
                "lowered, so the out parameters this store needs do not exist "
                "yet";

    auto outParameters = function.getArguments().take_back(op.getNumOperands());
    for (auto [source, destination] :
         llvm::zip_equal(adaptor.getOperands(), outParameters))
      npuisa::DmaStoreOp::create(rewriter, op.getLoc(), buffer(source),
                                 destination);

    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, ValueRange{});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The buffer producing operations.
//===----------------------------------------------------------------------===//

/// A constant is a DRAM buffer plus the one load that brings it on chip.
///
/// One `npu.constant` becomes one `npuisa.const` and one `npuisa.dma_load`
/// however many instructions read it, because the operation is converted once
/// and its scratchpad buffer is what every consumer is handed.
class ConstantOpLowering : public NPULowering<npu::ConstantOp> {
public:
  using NPULowering<npu::ConstantOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value constant = npuisa::ConstOp::create(
        rewriter, loc, dramTypeOf(op.getType()), op.getValue());
    Value scratchpad =
        memref::AllocOp::create(rewriter, loc, scratchpadTypeOf(op.getType()));
    npuisa::DmaLoadOp::create(rewriter, loc, constant, scratchpad);
    rewriter.replaceOp(op, scratchpad);
    return success();
  }
};

/// A destination tensor is a scratchpad allocation.
///
/// No deallocation is emitted, and that is a decision rather than an omission.
/// The scratchpad of this machine is not a heap: `-npu-allocate-scratchpad`
/// assigns every allocation a byte offset into one flat buffer and derives the
/// live intervals from the instruction stream itself, so a `memref.dealloc`
/// here would be a second and weaker statement of a lifetime the allocator
/// computes exactly. It would also be one more operation for the boundary test
/// to have to look past.
///
/// There is no dynamic size check here, and its absence is a decision rather
/// than an oversight. `tensor.empty` carries exactly one dynamic size operand
/// per dynamic extent, so a destination with a dynamic size is a destination
/// with a dynamic result type, and stage 1 has already refused that by name,
/// quoting the type. A second check here would be a branch no input can reach.
class EmptyOpLowering : public NPULowering<tensor::EmptyOp> {
public:
  using NPULowering<tensor::EmptyOp>::NPULowering;

  LogicalResult
  matchAndRewrite(tensor::EmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, scratchpadTypeOf(op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The compute instructions.
//===----------------------------------------------------------------------===//

class Conv2DOpLowering : public NPULowering<npu::Conv2DOp> {
public:
  using NPULowering<npu::Conv2DOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = buffer(adaptor.getDestination());
    npuisa::Conv2DOp::create(
        rewriter, op.getLoc(), buffer(adaptor.getInput()),
        buffer(adaptor.getFilter()),
        adaptor.getBias() ? buffer(adaptor.getBias()) : Value(),
        op.getStridesAttr(), op.getPadsAttr(), op.getDilationsAttr(),
        op.getGroupAttr(), destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

class MatMulOpLowering : public NPULowering<npu::MatMulOp> {
public:
  using NPULowering<npu::MatMulOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = buffer(adaptor.getDestination());
    npuisa::MatMulOp::create(
        rewriter, op.getLoc(), buffer(adaptor.getLhs()),
        buffer(adaptor.getRhs()),
        adaptor.getBias() ? buffer(adaptor.getBias()) : Value(), destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

/// `npu.add` and `npu.mul`, including the rank 1 channel broadcast.
///
/// The rank test is the whole of the broadcast decision, and it is exact
/// because the `npu` verifier has already decided the question: the right hand
/// operand either has the result shape or is rank 1 of the channel extent
/// against a rank 4 result, and nothing else parses. So a rank that differs from
/// the destination's is the broadcast form and a rank that matches is not.
template <typename SourceOp, typename TargetOp>
class ElementwiseBinaryLowering : public NPULowering<SourceOp> {
public:
  using NPULowering<SourceOp>::NPULowering;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = this->buffer(adaptor.getDestination());
    auto destinationType = cast<MemRefType>(destination.getType());

    Value rhs = this->buffer(adaptor.getRhs());
    if (cast<MemRefType>(rhs.getType()).getRank() != destinationType.getRank())
      rhs = channelBroadcast(rewriter, op.getLoc(), rhs, destinationType);

    TargetOp::create(rewriter, op.getLoc(), this->buffer(adaptor.getLhs()), rhs,
                     destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

class ReluOpLowering : public NPULowering<npu::ReluOp> {
public:
  using NPULowering<npu::ReluOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = buffer(adaptor.getDestination());
    npuisa::ReluOp::create(rewriter, op.getLoc(), buffer(adaptor.getInput()),
                           destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

/// The two pools. The mnemonics differ across the lowering on purpose: the
/// tensor level takes its name from ONNX's `MaxPool` and the instruction level
/// from the opcode `POOL_MAX`, so a diagnostic says which level it came from.
template <typename SourceOp, typename TargetOp>
class PoolLowering : public NPULowering<SourceOp> {
public:
  using NPULowering<SourceOp>::NPULowering;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = this->buffer(adaptor.getDestination());
    TargetOp::create(rewriter, op.getLoc(), this->buffer(adaptor.getInput()),
                     op.getKernelAttr(), op.getStridesAttr(), op.getPadsAttr(),
                     op.getDilationsAttr(), op.getCeilModeAttr(), destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

/// `npu.reshape` has no destination and `npuisa.reshape` needs one.
///
/// At the tensor level a reshape is a retyping and moves nothing, so a
/// destination operand there would claim an ability the operation does not have.
/// At the instruction level a buffer has an address and a shape, so the same
/// fact is either a copy or an aliasing view, and this dialect models it as the
/// copy. The allocation therefore appears here, at the first level with anywhere
/// to put it, and the allocator turns the copy back into nothing when it can by
/// giving the source and the destination the same offset.
class ReshapeOpLowering : public NPULowering<npu::ReshapeOp> {
public:
  using NPULowering<npu::ReshapeOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value destination =
        memref::AllocOp::create(rewriter, loc, scratchpadTypeOf(op.getType()));
    npuisa::ReshapeOp::create(rewriter, loc, buffer(adaptor.getInput()),
                              destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

class TransposeOpLowering : public NPULowering<npu::TransposeOp> {
public:
  using NPULowering<npu::TransposeOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = buffer(adaptor.getDestination());
    npuisa::TransposeOp::create(rewriter, op.getLoc(),
                                buffer(adaptor.getInput()),
                                op.getPermutationAttr(), destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

class ConcatOpLowering : public NPULowering<npu::ConcatOp> {
public:
  using NPULowering<npu::ConcatOp>::NPULowering;

  LogicalResult
  matchAndRewrite(npu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value destination = buffer(adaptor.getDestination());
    SmallVector<Value> inputs;
    for (Value input : adaptor.getInputs())
      inputs.push_back(buffer(input));
    npuisa::ConcatOp::create(rewriter, op.getLoc(), inputs, op.getAxisAttr(),
                             destination);
    rewriter.replaceOp(op, destination);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

struct NPULowerToNPUISAPass
    : public npuisa::impl::NPULowerToNPUISABase<NPULowerToNPUISAPass> {
  using npuisa::impl::NPULowerToNPUISABase<
      NPULowerToNPUISAPass>::NPULowerToNPUISABase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    if (failed(validate(module)) || failed(expand(module)))
      return signalPassFailure();

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [context](RankedTensorType type) -> std::optional<Type> {
          FailureOr<MemRefType> buffer =
              convertTensorType(type, npu::ScratchpadAttr::get(context));
          if (failed(buffer))
            return std::nullopt;
          return *buffer;
        });

    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect,
                           npuisa::NPUISADialect>();
    target.addIllegalDialect<npu::NPUDialect>();
    target.addIllegalOp<tensor::EmptyOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp function) {
      return function.getFunctionType().getNumResults() == 0 &&
             llvm::none_of(function.getArgumentTypes(),
                           [](Type type) { return isa<TensorType>(type); });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return op.getNumOperands() == 0; });

    LoweringState state;
    RewritePatternSet patterns(context);
    patterns.add<FuncOpLowering, ReturnOpLowering, ConstantOpLowering,
                 EmptyOpLowering, Conv2DOpLowering, MatMulOpLowering,
                 ReluOpLowering, ReshapeOpLowering, TransposeOpLowering,
                 ConcatOpLowering>(converter, context, state);
    patterns.add<ElementwiseBinaryLowering<npu::AddOp, npuisa::AddOp>,
                 ElementwiseBinaryLowering<npu::MulOp, npuisa::MulOp>>(
        converter, context, state);
    patterns.add<PoolLowering<npu::MaxPool2DOp, npuisa::PoolMaxOp>,
                 PoolLowering<npu::AvgPool2DOp, npuisa::PoolAvgOp>>(
        converter, context, state);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
