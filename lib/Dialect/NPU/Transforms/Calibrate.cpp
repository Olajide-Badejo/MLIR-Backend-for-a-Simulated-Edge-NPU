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
// per channel weight quantization not at all. The per channel scales are in
// the profile, and the operation that can hold them is the instruction, whose
// fourth operand carries one multiplier and one shift per output channel. So
// the weight half travels in the profile to the contraction, and this file
// says so rather than quietly quantizing weights per tensor and losing the
// granularity the phase gate asks to measure.
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

/// The two requantization modes. `fixed` is the machine's own arithmetic and
/// `float` exists so that a previously published number stays reproducible.
constexpr llvm::StringLiteral kRequantModes[] = {"fixed", "float"};

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
  if (const llvm::json::Object *nodes = root->getObject("nodes")) {
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
      profile.nodes[entry.first.str()] = std::move(node);
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
    if (!llvm::is_contained(kRequantModes, llvm::StringRef(requantMode))) {
      function.emitError()
          << "'" << requantMode
          << "' is not a requantization mode. The two are fixed and float.";
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

    auto input = loaded.activations.find(node->second.inputs.front());
    auto output = loaded.activations.find(node->second.outputs.front());
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

    Value result = op->getResult(0);
    builder.setInsertionPointAfter(op);
    Value back = quantizeAndBack(builder, loc, result, output->second);
    result.replaceAllUsesExcept(back, back.getDefiningOp()
                                          ->getOperand(0)
                                          .getDefiningOp());

    ++rewritten;
    if (input->second.degenerate || output->second.degenerate)
      op->emitWarning()
          << "a tensor of this operation calibrated to a degenerate range, so "
             "the profile substituted a scale of 1 and a zero point of 0. The "
             "quantization is legal and it is not meaningful.";
    return success();
  }
};

} // namespace
