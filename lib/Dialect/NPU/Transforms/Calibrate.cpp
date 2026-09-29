//===- Calibrate.cpp - the QDQ rewrite of Section 14 ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-calibrate`, quantized mode only and never in a default
// `-O` level.
//
// **What this pass is and what the observer is.** The observer runs the model
// and writes down what it saw; this turns what was seen into the form the rest
// of the pipeline can lower. The split is deliberate and is argued in
// `python/npu_frontend/calibration.py`: a range and the affine pair derived
// from it are arithmetic over observed data, pinned by Section 14 and held to
// hand computed cases in `test_calibration.py`, so deriving them again here
// would be a second implementation of the same rule with nothing comparing the
// two. That is the observer against kernel disagreement Section 14 opens by
// warning about. This pass reads numbers and rewrites operations.
//
// **The rewrite is the standard QDQ form.** A covered operation has its
// activations wrapped in a quantize and a dequantize pair, so the graph stays
// f32 and the integer values are interior:
//
//     %qx = npu.quantize %x            %x -> i8
//     %dx = npu.dequantize %qx         i8 -> f32
//     %y  = npu.conv2d ins(%dx, ...)
//     %qy = npu.quantize %y
//     %dy = npu.dequantize %qy         and every later reader takes %dy
//
// Nothing here emits an integer instruction. The contraction in the lowering
// is what turns `quantize(conv2d(dequantize, ...))` into one, and keeping the
// tensor level in f32 is what lets every pass after this one stay unchanged.
//
// **The weights are not quantized here, and that is a limit of the level
// rather than an omission.** Section 14 makes weight scales per output
// channel, and `npu.quantize` carries a single `scale` attribute: the QDQ form
// at this level can express per tensor activation quantization exactly and
// per channel weight quantization not at all. The operation that can hold them
// is the instruction, whose fourth operand carries one multiplier and one
// shift per output channel, so the scales travel to the contraction in the
// `weight_scales` attribute rather than being quantized per tensor here.
//
// **The weight scales are computed here, from the constant the operation
// holds**, and that is a revision rather than the design this pass started
// with. The profile's weight entries describe the ONNX initializers, and at
// `-O2` the batch norm fold rewrites a convolution's filter before this pass
// sees it, multiplying each output channel by its own factor. Section 14's
// argument for per channel weights is exactly the spread that fold creates, so
// only scales taken from the folded constant describe what the machine
// quantizes. The rule is the observer's, `max |w_c| / 127` with 1 for a
// channel of zeros, evaluated the same way, and at `-O0`, where nothing has
// touched the filter, the result equals the profile's entry bit for bit on
// every channel of every model, which `test_quantized_contraction.py` holds.
// The profile's weight entries are that oracle now, and nothing reads them
// here.
//
// **The three diagnostics are Section 14's own**, and each says something
// different. An empty profile is a pass failure, because a calibration pass
// asked to run with nothing to calibrate from has been misconfigured and a no
// op would hide it. A profile that covers no operation in the function is a
// single remark naming the function and the count, because that is a real
// configuration, the profile of another model, and it should be visible
// without being fatal. A partially covered operation is skipped and counted,
// never half rewritten, because an operation with a calibrated input and an
// uncalibrated output has no scale to quantize its result with and guessing
// one would be inventing a number.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace mlir::npu {
#define GEN_PASS_DEF_NPUCALIBRATE
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

/// The profile format this build reads. It moves when a field's meaning moves,
/// which is the rule `Program::kVersion` follows one level down.
constexpr int kProfileVersion = 1;

/// The four methods of Section 14, in the order that section lists them.
constexpr llvm::StringLiteral kMethods[] = {"minmax", "percentile", "mse",
                                            "entropy"};

/// The requantization mode this compiler has: `fixed`, the integer multiplier
/// and shift the machine applies. `float` is refused by name, which is the
/// owner's ruling of 2026-09-29, and the reason is in `docs/PASSES.md` beside
/// the option.
constexpr llvm::StringLiteral kRequantModes[] = {"fixed"};

/// The two weight granularities Section 14's ablation compares, the default
/// first.
constexpr llvm::StringLiteral kGranularities[] = {"per-channel", "per-tensor"};

/// One tensor's affine pair, as the profile records it.
struct ActivationScale {
  double scale = 1.0;
  int32_t zeroPoint = 0;
  bool degenerate = false;
};

/// One quantizable node, by the name the operation's location carries.
struct NodeRecord {
  std::string opType;
  llvm::SmallVector<std::string> inputs;
  llvm::SmallVector<std::string> outputs;
};

struct Profile {
  llvm::StringMap<NodeRecord> nodes;
  /// Every node of the graph, quantizable or not, which is what says which
  /// tensor a relu writes. Empty in a profile written before the section
  /// existed, and then no relu is fused, which is the compilation such a
  /// profile always had.
  llvm::StringMap<NodeRecord> graph;
  /// Keyed by tensor name, for the one method the pass was asked for.
  llvm::StringMap<ActivationScale> activations;
  bool empty() const { return nodes.empty(); }
};

/// The ONNX node name a location carries, or an empty string.
///
/// The same walk the encoder does one level down, for the same reason: the
/// importer gives every operation a `NameLoc` holding the node name, and a
/// `FusedLoc` appears wherever a pass decomposed one operation into several.
llvm::StringRef nameFromLocation(Location loc) {
  if (auto named = dyn_cast<NameLoc>(loc))
    return named.getName().strref();
  if (auto fused = dyn_cast<FusedLoc>(loc))
    for (Location nested : fused.getLocations())
      if (llvm::StringRef name = nameFromLocation(nested); !name.empty())
        return name;
  return {};
}

/// Every node name a location carries, in order.
///
/// A convolution that `-npu-fuse-bias` or `-npu-fold-batchnorm` absorbed a
/// node into carries a fused location: its own name first and each absorbed
/// node's after it. The first is the node the profile's `nodes` section
/// joins; the last is the node whose output the operation's result now is.
void namesFromLocation(Location loc,
                       llvm::SmallVectorImpl<llvm::StringRef> &names) {
  if (auto named = dyn_cast<NameLoc>(loc)) {
    names.push_back(named.getName().strref());
    return;
  }
  if (auto fused = dyn_cast<FusedLoc>(loc))
    for (Location nested : fused.getLocations())
      namesFromLocation(nested, names);
}

/// Reads the profile, or says what is wrong with it.
///
/// Every failure returns a message rather than a partially filled profile,
/// because a calibration that ran against half a file would quantize some
/// operations and silently skip the rest, which is the failure mode the
/// partial coverage rule exists to forbid.
llvm::Expected<Profile> readProfile(llvm::StringRef path,
                                    llvm::StringRef method) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read the calibration profile '%s'",
                                   path.str().c_str());

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();

  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the calibration profile '%s' is not a JSON object",
        path.str().c_str());

  std::optional<int64_t> version = root->getInteger("schema_version");
  if (!version || *version != kProfileVersion)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "the calibration profile '%s' is version %d and this build reads "
        "version %d",
        path.str().c_str(), version ? static_cast<int>(*version) : -1,
        kProfileVersion);

  Profile profile;
  // The node section and the graph section have one record shape, so they are
  // read by one loop over the two keys.
  for (llvm::StringRef section : {"nodes", "graph"}) {
    const llvm::json::Object *nodes = root->getObject(section);
    if (!nodes)
      continue;
    llvm::StringMap<NodeRecord> &into =
        section == "nodes" ? profile.nodes : profile.graph;
    for (const auto &entry : *nodes) {
      const llvm::json::Object *record = entry.second.getAsObject();
      if (!record)
        continue;
      NodeRecord node;
      if (std::optional<llvm::StringRef> type = record->getString("op_type"))
        node.opType = type->str();
      for (llvm::StringRef field : {"inputs", "outputs"}) {
        const llvm::json::Array *names = record->getArray(field);
        if (!names)
          continue;
        for (const llvm::json::Value &name : *names)
          if (std::optional<llvm::StringRef> text = name.getAsString())
            (field == "inputs" ? node.inputs : node.outputs)
                .push_back(text->str());
      }
      into[entry.first.str()] = std::move(node);
    }
  }

  if (const llvm::json::Object *scales = root->getObject("activation_scales")) {
    for (const auto &entry : *scales) {
      const llvm::json::Object *byMethod = entry.second.getAsObject();
      if (!byMethod)
        continue;
      const llvm::json::Object *chosen = byMethod->getObject(method);
      if (!chosen)
        continue;
      ActivationScale value;
      if (std::optional<double> scale = chosen->getNumber("scale"))
        value.scale = *scale;
      if (std::optional<int64_t> zero = chosen->getInteger("zero_point"))
        value.zeroPoint = static_cast<int32_t>(*zero);
      if (std::optional<bool> degenerate = chosen->getBoolean("degenerate"))
        value.degenerate = *degenerate;
      profile.activations[entry.first.str()] = value;
    }
  }
  return profile;
}

/// Section 14's static accumulator guard: `K * 128 * 127 < 2^31`.
///
/// Checked here and diagnosed by name, which is what that section asks for: a
/// static guard is both more honest than a wider accumulator and a better
/// error than a trap at run time.
bool accumulatorBoundHolds(int64_t reduction) {
  return reduction * 128 * 127 < (int64_t{1} << 31);
}

/// The reduction depth of the operation, which is what the guard is about.
int64_t reductionDepth(Operation *op) {
  if (auto conv = dyn_cast<Conv2DOp>(op)) {
    auto filter = cast<RankedTensorType>(conv.getFilter().getType());
    int64_t depth = 1;
    for (int64_t extent : filter.getShape().drop_front())
      depth *= extent;
    return depth;
  }
  auto matmul = cast<MatMulOp>(op);
  auto lhs = cast<RankedTensorType>(matmul.getLhs().getType());
  return lhs.getShape().back();
}

/// Wraps one value in a quantize and a dequantize pair.
///
/// The returned value is the dequantized one, which every later reader takes,
/// so the graph stays f32 and the i8 value is interior. That is the shape an
/// exported QDQ graph has and the shape the importer already understands.
Value quantizeAndBack(OpBuilder &builder, Location loc, Value value,
                      const ActivationScale &pair) {
  auto type = cast<RankedTensorType>(value.getType());
  auto quantized = RankedTensorType::get(
      type.getShape(), builder.getI8Type(), type.getEncoding());
  FloatAttr scale = builder.getF32FloatAttr(static_cast<float>(pair.scale));
  IntegerAttr zeroPoint = builder.getI32IntegerAttr(pair.zeroPoint);

  Value down = QuantizeOp::create(builder, loc, quantized, value, scale,
                                  zeroPoint);
  return DequantizeOp::create(builder, loc, type, down, scale, zeroPoint);
}

/// The value an operand stands for outside an `npu.fused_op` region, or the
/// operand itself when it is not a block argument of one.
///
/// A region is `IsolatedFromAbove`, so an operation fused into one reads every
/// value through a block argument, and the argument's position is the region's
/// operand position. This is how the pass sees through a region to the
/// constant a filter is and to the value an input is.
Value outsideValue(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return value;
  auto fused = dyn_cast_or_null<FusedOp>(argument.getOwner()->getParentOp());
  if (!fused)
    return value;
  return fused->getOperand(argument.getArgNumber());
}

/// Section 14's symmetric weight rule over the constant the operation holds:
/// one scale per output channel, `max |w_c| / 127`, and 1 for a channel whose
/// weights are all zero, whose scale is not a magnitude. With `perTensor`
/// every channel takes the tensor's one scale, the largest magnitude of the
/// whole tensor over 127, which is the scale of the channel that holds it.
///
/// **Evaluated the way the observer evaluates it**, so that the two agree bit
/// for bit where they see the same weights: the largest magnitude is exact in
/// f32, the division is in double, and the quotient is rounded once to f32,
/// which is what the profile's double becomes when the pass reads it.
///
/// Nothing when the weights are not an f32 constant. The contraction needs a
/// constant to quantize at compile time, so an operation without one is not
/// contracted, and a scale for it would be a number nothing reads.
std::optional<llvm::SmallVector<float>> weightScalesOf(Operation *op,
                                                       bool perTensor) {
  const bool column = isa<MatMulOp>(op);
  Value weights =
      column ? cast<MatMulOp>(op).getRhs() : cast<Conv2DOp>(op).getFilter();
  auto constant = outsideValue(weights).getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!dense || !dense.getElementType().isF32())
    return std::nullopt;

  // A filter is `(M, C/group, kH, kW)` and its output channel is the slowest
  // axis; a matrix is `(K, N)` and its output channel is the fastest. D-0067
  // is what reading one layout for both cost.
  auto type = cast<RankedTensorType>(dense.getType());
  const int64_t channels = column ? type.getDimSize(1) : type.getDimSize(0);
  if (channels <= 0)
    return std::nullopt;
  const int64_t perChannel = type.getNumElements() / channels;

  llvm::SmallVector<float> maxima(channels, 0.0f);
  int64_t flat = 0;
  for (float value : dense.getValues<float>()) {
    const int64_t channel = column ? flat % channels : flat / perChannel;
    ++flat;
    maxima[channel] = std::max(maxima[channel], std::fabs(value));
  }

  auto scaleOf = [](float maximum) {
    return maximum > 0.0f
               ? static_cast<float>(static_cast<double>(maximum) / 127.0)
               : 1.0f;
  };
  llvm::SmallVector<float> scales;
  scales.reserve(channels);
  if (perTensor) {
    const float largest = *std::max_element(maxima.begin(), maxima.end());
    scales.assign(channels, scaleOf(largest));
    return scales;
  }
  for (float maximum : maxima)
    scales.push_back(scaleOf(maximum));
  return scales;
}

class NPUCalibratePass
    : public mlir::npu::impl::NPUCalibrateBase<NPUCalibratePass> {
public:
  using mlir::npu::impl::NPUCalibrateBase<
      NPUCalibratePass>::NPUCalibrateBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();

    if (!llvm::is_contained(kMethods, llvm::StringRef(calibMethod))) {
      function.emitError()
          << "'" << calibMethod
          << "' is not a calibration method. The four are minmax, percentile, "
             "mse and entropy, and minmax is the default.";
      return signalPassFailure();
    }
    // **`float` is refused by name rather than accepted and ignored.** It
    // compiled to exactly what `fixed` compiles to, so a row labelled with it
    // would have measured `fixed` twice. Section 14 kept it so that a number
    // published under it would stay reproducible, and none ever was; the scale
    // word it would need carries the output zero point, by the owner's
    // decision of 2026-09-07; and a float multiplier per output element is
    // hardware Section 14 itself says the modelled machine does not have. The
    // fixed against float comparison is a reference level measurement in the
    // numpy reference at Checkpoint C instead.
    if (llvm::StringRef(requantMode) == "float") {
      function.emitError()
          << "requant-mode=float is refused. No number was ever published "
             "under it, the scale word it would need carries the output zero "
             "point, and Section 14 says a float multiplier is hardware the "
             "modelled machine does not have; the fixed against float "
             "comparison is measured in the numpy reference instead. The one "
             "mode is fixed.";
      return signalPassFailure();
    }
    if (!llvm::is_contained(kRequantModes, llvm::StringRef(requantMode))) {
      function.emitError()
          << "'" << requantMode
          << "' is not a requantization mode. The one mode is fixed, and float "
             "is refused by name.";
      return signalPassFailure();
    }
    if (!llvm::is_contained(kGranularities,
                            llvm::StringRef(weightGranularity))) {
      function.emitError()
          << "'" << weightGranularity
          << "' is not a weight granularity. The two are per-channel, the "
             "default, and per-tensor.";
      return signalPassFailure();
    }

    // **An empty profile is a failure rather than a no op**, which is Section
    // 14's own rule. A pass asked to calibrate with nothing to calibrate from
    // has been misconfigured, and a silent no op would leave the model f32
    // while every report said it had been quantized.
    if (profile.empty()) {
      function.emitError()
          << "-npu-calibrate needs a calibration profile and was given none. "
             "Pass profile=<path>; the profiles this project commits are under "
             "experiments/calibration/.";
      return signalPassFailure();
    }

    llvm::Expected<Profile> loaded = readProfile(profile, calibMethod);
    if (!loaded) {
      function.emitError() << llvm::toString(loaded.takeError());
      return signalPassFailure();
    }

    llvm::SmallVector<Operation *> candidates;
    function.walk([&](Operation *op) {
      if (isa<Conv2DOp, MatMulOp>(op))
        candidates.push_back(op);
    });

    for (Operation *op : candidates)
      if (failed(rewriteOne(op, *loaded)))
        return signalPassFailure();

    foldRoundTrips(function);

    // **A profile that covers nothing is one remark, not one per operation.**
    // It is a real configuration, the profile of another model, and a reader
    // needs to be told once with the count rather than told repeatedly.
    //
    // The two counts are reported separately because they mean different
    // things and have different fixes. An operation the profile does not name
    // was not observed; one it names without a full set of ranges was observed
    // and something is missing from the file. A single "covers nothing" would
    // send a reader looking for the wrong problem.
    if (rewritten == 0 && !candidates.empty())
      function.emitRemark()
          << "-npu-calibrate rewrote nothing in '" << function.getName()
          << "': of its " << candidates.size() << " quantizable operations the "
          << "profile does not name " << uncovered.getValue() << " and names "
          << skipped.getValue() << " without a full set of ranges.";
  }

private:
  /// Removes every dequantize then quantize pair that returns its values
  /// unchanged, which is `npu.quantize`'s folder applied here.
  ///
  /// **Here as well as in `-canonicalize`, because `-O0` runs no
  /// canonicalization** and the pairs are this pass's own making: where one
  /// calibrated operation's quantized output is the next one's input, the first
  /// operation's output pair and the second's input pair meet as a dequantize
  /// then a quantize of the same tensor with the same scale and zero point.
  /// Left in place, that is a `DEQUANT` and a `QUANT` that compute nothing
  /// between two integer instructions, and a boundary crossing Section 14 never
  /// drew. The condition is `QuantizeOp::getRoundTripSource`, written once, and
  /// a dequantize left with no reader goes too, because nothing after this pass
  /// at `-O0` would remove it.
  void foldRoundTrips(func::FuncOp function) {
    SmallVector<QuantizeOp> quantizes;
    function.walk([&](QuantizeOp op) { quantizes.push_back(op); });
    for (QuantizeOp op : quantizes) {
      Value source = op.getRoundTripSource();
      if (!source)
        continue;
      auto pair = op.getInput().getDefiningOp<DequantizeOp>();
      op.getResult().replaceAllUsesWith(source);
      op.erase();
      if (pair.getResult().use_empty())
        pair.erase();
      ++foldedPairs;
    }
  }

  /// The relu a calibrated operation's output pair goes after, with the
  /// range the profile holds for the relu's output, or nothing.
  ///
  /// **Four conditions, and each one is a reason the relu could not be the
  /// instruction's.** The operation's result has exactly one reader, and it is
  /// an `npu.relu` reading it as its input, because a second reader would need
  /// the value before the relu. The relu's location names a `Relu` node of the
  /// profile's graph. That node reads the tensor this operation's own node
  /// writes, which is the check that the relu the IR holds is the relu the
  /// observer saw after this operation rather than one that shares a name. And
  /// the node's output has a range for the method asked for. A profile without
  /// the graph section fails the second, and compiles as it always did.
  std::optional<std::pair<ReluOp, ActivationScale>>
  fusibleRelu(Operation *op, llvm::StringRef writes, const Profile &loaded) {
    Value result = op->getResult(0);
    if (!result.hasOneUse())
      return std::nullopt;
    auto relu = dyn_cast<ReluOp>(*result.getUsers().begin());
    if (!relu || relu.getInput() != result)
      return std::nullopt;

    auto found = loaded.graph.find(nameFromLocation(relu.getLoc()));
    if (found == loaded.graph.end() || found->second.opType != "Relu" ||
        found->second.inputs.empty() || found->second.outputs.empty() ||
        found->second.inputs.front() != writes)
      return std::nullopt;

    auto range = loaded.activations.find(found->second.outputs.front());
    if (range == loaded.activations.end())
      return std::nullopt;
    return std::make_pair(relu, range->second);
  }

  LogicalResult rewriteOne(Operation *op, const Profile &loaded) {
    llvm::StringRef name = nameFromLocation(op->getLoc());
    auto node = loaded.nodes.find(name);
    if (name.empty() || node == loaded.nodes.end()) {
      ++uncovered;
      return success();
    }

    if (node->second.inputs.empty() || node->second.outputs.empty()) {
      ++skipped;
      return success();
    }

    // **The tensor this operation's result is.** Its own node's output,
    // unless a fold absorbed a node into it, in which case the last absorbed
    // node's output, which the profile's graph names: after the batch norm
    // fold the convolution's result is the batch norm's, and its range is the
    // batch norm output's. A profile that does not say what the absorbed node
    // writes leaves the operation without a range for its result, and that is
    // the partial coverage below, skipped and counted.
    llvm::SmallVector<llvm::StringRef> names;
    namesFromLocation(op->getLoc(), names);
    llvm::StringRef writes = node->second.outputs.front();
    if (names.size() > 1) {
      auto absorbed = loaded.graph.find(names.back());
      writes =
          absorbed == loaded.graph.end() || absorbed->second.outputs.empty()
              ? llvm::StringRef()
              : llvm::StringRef(absorbed->second.outputs.front());
    }

    auto input = loaded.activations.find(node->second.inputs.front());
    auto output = writes.empty() ? loaded.activations.end()
                                 : loaded.activations.find(writes);
    if (input == loaded.activations.end() ||
        output == loaded.activations.end()) {
      // **Skipped and counted, never half rewritten.** An operation whose
      // output has no calibrated range has no scale to quantize its result
      // with, and inventing one is what this rule forbids.
      ++skipped;
      return success();
    }

    const int64_t depth = reductionDepth(op);
    if (!accumulatorBoundHolds(depth)) {
      op->emitError()
          << "this operation reduces over " << depth
          << " elements, and Section 14 accumulates in int32 and proves it "
             "statically: K * 128 * 127 must be below 2^31, which holds up to "
          << (((int64_t{1} << 31) - 1) / (128 * 127))
          << ". Quantizing it would produce a program the machine refuses to "
             "execute.";
      return failure();
    }

    OpBuilder builder(op);
    Location loc = op->getLoc();
    Value data = isa<Conv2DOp>(op) ? cast<Conv2DOp>(op).getInput()
                                   : cast<MatMulOp>(op).getLhs();
    Value staged = quantizeAndBack(builder, loc, data, input->second);
    if (auto conv = dyn_cast<Conv2DOp>(op))
      conv.getInputMutable().assign(staged);
    else
      cast<MatMulOp>(op).getLhsMutable().assign(staged);

    // **The output is quantized after the relu when a relu is the only
    // reader**, with the relu's own range, because the contraction fuses the
    // relu into the integer instruction and the instruction's result is the
    // relu's. Quantizing the operation's result instead would round it at its
    // own scale, dequantize it, run the relu in f32 and round it again, which
    // is a program no INT8 NPU runs and a boundary crossing Section 14 never
    // drew. `fusibleRelu` says when, and anything it declines is quantized
    // where it always was.
    Value quantized = op->getResult(0);
    Location outputLoc = loc;
    ActivationScale outputScale = output->second;
    if (auto fused = fusibleRelu(op, writes, loaded)) {
      quantized = fused->first.getResult();
      outputLoc = fused->first.getLoc();
      outputScale = fused->second;
    }
    builder.setInsertionPointAfterValue(quantized);
    Value back = quantizeAndBack(builder, outputLoc, quantized, outputScale);
    quantized.replaceAllUsesExcept(
        back, back.getDefiningOp()->getOperand(0).getDefiningOp());

    // **The per output channel weight scales, carried as an attribute because
    // the QDQ form cannot carry them.** `npu.quantize` has a single scale, so
    // this level expresses per tensor activation quantization exactly and per
    // channel weight quantization not at all. The contraction in the lowering
    // needs them, and an attribute is how they travel between the two without
    // a file read in the lowering or a per tensor weight that would discard the
    // granularity the gate measures.
    //
    // Computed from the constant the operation holds, which after the batch
    // norm fold is the folded one; the file header has why. Written only here,
    // which is what the verifier's rule that the operand must come from a
    // dequantize enforces at the other end.
    if (std::optional<llvm::SmallVector<float>> scales = weightScalesOf(
            op, llvm::StringRef(weightGranularity) == "per-tensor"))
      op->setAttr("weight_scales",
                  DenseF32ArrayAttr::get(op->getContext(), *scales));

    ++rewritten;
    if (input->second.degenerate || outputScale.degenerate)
      op->emitWarning()
          << "a tensor of this operation calibrated to a degenerate range, so "
             "the profile substituted a scale of 1 and a zero point of 0. The "
             "quantization is legal and it is not meaningful.";
    return success();
  }
};

} // namespace
