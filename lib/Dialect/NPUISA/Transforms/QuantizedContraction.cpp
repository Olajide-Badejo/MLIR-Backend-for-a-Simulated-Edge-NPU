//===- QuantizedContraction.cpp - Section 14's QDQ contraction --*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// What `-npu-calibrate` leaves behind is a graph that is still f32: the
// activations are wrapped in a quantize and a dequantize pair, the weights are
// untouched, and their per output channel scales ride along as an attribute
// because the QDQ form has nowhere else to put them. **This file is where that
// becomes integer arithmetic.**
//
//     %dx = npu.dequantize %qx                  scale_x, zp_x
//     %y  = npu.conv2d ins(%dx, %w, %b) {weight_scales = [...]}
//     %qy = npu.quantize %y                     scale_y, zp_y
//
// becomes one instruction reading the i8 buffer `%qx` names, an i8 weight
// constant, an int32 bias, and a table of one multiplier and one shift per
// output channel.
//
// THE ARITHMETIC, WHICH IS SECTION 14'S AND IS PINNED
//
// The real convolution is `y = sum_k x[k] * w[k] + b`, and with
// `x[k] = scale_x * (q_x[k] - zp_x)` and `w[k] = scale_w_c * q_w[k]` that is
//
//     y_c = scale_x * scale_w_c * (sum_k q_x[k] q_w[k] - zp_x sum_k q_w[k])
//           + b_c
//
// so dividing by `scale_x * scale_w_c` gives an integer accumulation plus an
// integer bias:
//
//     acc_c    = sum_k q_x[k] q_w[k] + bias_q_c
//     bias_q_c = round(b_c / (scale_x * scale_w_c)) - zp_x * sum_k q_w[c][k]
//
// **The zero point is folded into the bias rather than subtracted per tap**,
// which is Section 14's own instruction: the cross term costs order N^2
// additions once at compile time against order N^3 inside the loop, and a
// subtract inside the multiply accumulate models a datapath no NPU has. The
// two forms agree bit for bit because integer addition is associative, and
// that is tested over seeded random int8 tensors rather than argued.
//
// The result is brought back to i8 by a fixed point rescale per channel,
//
//     M_c = (scale_x * scale_w_c) / scale_y = M0_c * 2^-(31 + shift_c)
//     q_y = clamp(requantize(acc_c, M0_c, shift_c) + zp_y, -128, 127)
//
// **Padding contributes the input zero point**, and that is a consequence of
// folding rather than a detail of the machine: the subtracted term is taken
// over the whole window, so a tap outside the input has to contribute `zp_x`
// for the two to cancel at a padded output position.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/QuantizedContraction.h"

#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::npuisa;

//===----------------------------------------------------------------------===//
// The arithmetic.
//===----------------------------------------------------------------------===//

double mlir::npuisa::roundHalfToEven(double value) {
  const double floorValue = std::floor(value);
  const double fraction = value - floorValue;
  if (fraction > 0.5)
    return floorValue + 1.0;
  if (fraction < 0.5)
    return floorValue;
  // Exactly halfway. `fmod` decides which neighbour is even without a cast to
  // an integer type that a large magnitude would overflow.
  return std::fmod(floorValue, 2.0) == 0.0 ? floorValue : floorValue + 1.0;
}

double mlir::npuisa::requantizationMultiplier(float scaleX, float scaleW,
                                              float scaleY) {
  return (static_cast<double>(scaleX) * static_cast<double>(scaleW)) /
         static_cast<double>(scaleY);
}

std::optional<RequantPair>
mlir::npuisa::decomposeMultiplier(double multiplier, std::string *reason) {
  auto refuse = [&](std::string why) -> std::optional<RequantPair> {
    if (reason)
      *reason = std::move(why);
    return std::nullopt;
  };

  if (!std::isfinite(multiplier) || multiplier <= 0.0)
    return refuse("the requantization multiplier must be finite and strictly "
                  "positive, and it is " +
                  std::to_string(multiplier));

  int32_t shift = 0;
  double scaled = multiplier;
  while (scaled >= 1.0) {
    scaled /= 2.0;
    --shift;
  }
  while (scaled < 0.5) {
    scaled *= 2.0;
    ++shift;
  }

  // `scaled` is in [0.5, 1), so this is in [2^30, 2^31] and exact before the
  // rounding: a power of two scales a double without touching its mantissa.
  constexpr double kTwoTo31 = 2147483648.0;
  double fixed = roundHalfToEven(scaled * kTwoTo31);
  if (fixed == kTwoTo31) {
    fixed /= 2.0;
    --shift;
  }

  if (shift < 0)
    return refuse("a multiplier of one or more needs a negative shift, and the "
                  "shift is a right shift. The output scale is smaller than "
                  "the product of the input and weight scales, which no "
                  "calibration of a tensor's own range produces");
  if (shift > 31)
    return refuse("a multiplier this small needs a right shift of " +
                  std::to_string(shift) +
                  ", past the 31 the binary format accepts, so every result "
                  "of the channel would round to the output zero point");
  return RequantPair{static_cast<int32_t>(fixed), shift};
}

int8_t mlir::npuisa::quantizeWeight(float weight, float scale) {
  const double scaled =
      static_cast<double>(weight) / static_cast<double>(scale);
  if (std::isnan(scaled))
    return 0;
  if (!std::isfinite(scaled))
    return scaled > 0.0 ? 127 : -128;
  const double rounded = roundHalfToEven(scaled);
  if (rounded <= -128.0)
    return -128;
  if (rounded >= 127.0)
    return 127;
  return static_cast<int8_t>(rounded);
}

std::optional<int32_t> mlir::npuisa::foldedBias(std::optional<float> bias,
                                                float scaleX, float scaleW,
                                                int32_t inputZeroPoint,
                                                int64_t weightSum,
                                                std::string *reason) {
  auto refuse = [&](std::string why) -> std::optional<int32_t> {
    if (reason)
      *reason = std::move(why);
    return std::nullopt;
  };

  // Bounded well inside int64: a zero point is an i8 and the sum is at most
  // 128 times the reduction depth.
  const int64_t fold = -static_cast<int64_t>(inputZeroPoint) * weightSum;

  int64_t quantized = 0;
  if (bias) {
    const double scaled =
        static_cast<double>(*bias) /
        (static_cast<double>(scaleX) * static_cast<double>(scaleW));
    if (!std::isfinite(scaled))
      return refuse("the bias " + std::to_string(*bias) +
                    " divided by its scale is not a finite number");
    const double rounded = roundHalfToEven(scaled);
    // Anything this far out is outside int32 whatever the fold adds, and
    // checking in double first keeps the conversion below defined.
    constexpr double kFarOutside = 4611686018427387904.0; // 2^62
    if (std::abs(rounded) >= kFarOutside)
      return refuse("the bias quantizes to " + std::to_string(rounded) +
                    ", which does not fit in the int32 the instruction adds to "
                    "its accumulator");
    quantized = static_cast<int64_t>(rounded);
  }

  const int64_t total = quantized + fold;
  if (total < std::numeric_limits<int32_t>::min() ||
      total > std::numeric_limits<int32_t>::max())
    return refuse("the bias with the input zero point folded in is " +
                  std::to_string(total) +
                  ", which does not fit in the int32 the instruction adds to "
                  "its accumulator");
  return static_cast<int32_t>(total);
}

//===----------------------------------------------------------------------===//
// The planner.
//===----------------------------------------------------------------------===//

namespace {

/// An operation the calibrator left whole, with its parts found.
struct Cluster {
  npu::DequantizeOp lead;
  npu::QuantizeOp sink;
  npu::ConstantOp weightOp;
  DenseFPElementsAttr weights;
  /// Both null when the operation has no bias.
  npu::ConstantOp biasOp;
  DenseFPElementsAttr bias;
  Operation *destination = nullptr;
  ArrayRef<float> scales;
  int64_t channels = 0;
  /// A matrix is `(K, N)`, so its output channel is the **fastest** axis and a
  /// channel's weights are a column; a filter is `(M, C/group, kH, kW)`, so its
  /// output channel is the slowest and a channel's weights are contiguous.
  /// D-0067 is what reading one layout for both costs.
  bool channelIsColumn = false;
};

/// The dense f32 value of a constant operand, or null when it is not one.
DenseFPElementsAttr f32Constant(Value value, npu::ConstantOp &definer) {
  definer = value ? value.getDefiningOp<npu::ConstantOp>() : npu::ConstantOp();
  if (!definer)
    return {};
  auto dense = dyn_cast<DenseFPElementsAttr>(definer.getValue());
  if (!dense || !dense.getElementType().isF32())
    return {};
  return dense;
}

/// The cluster around a calibrated operation, or nothing when the calibrator's
/// shape is not all there. Nothing is not a refusal: the operation stays in the
/// QDQ form and lowers the way it did before the contraction existed.
std::optional<Cluster> clusterOf(Operation *op) {
  Cluster cluster;
  Value data;
  Value weights;
  Value bias;
  Value destination;
  std::optional<ArrayRef<float>> scales;
  if (auto conv = dyn_cast<npu::Conv2DOp>(op)) {
    data = conv.getInput();
    weights = conv.getFilter();
    bias = conv.getBias();
    destination = conv.getDestination();
    scales = conv.getWeightScales();
    cluster.channels = cast<RankedTensorType>(weights.getType()).getDimSize(0);
  } else if (auto matmul = dyn_cast<npu::MatMulOp>(op)) {
    data = matmul.getLhs();
    weights = matmul.getRhs();
    bias = matmul.getBias();
    destination = matmul.getDestination();
    scales = matmul.getWeightScales();
    cluster.channels = cast<RankedTensorType>(weights.getType()).getDimSize(1);
    cluster.channelIsColumn = true;
  } else {
    return std::nullopt;
  }

  // The attribute only `-npu-calibrate` writes, which is what says this
  // operation was calibrated at all.
  if (!scales)
    return std::nullopt;
  cluster.scales = *scales;

  // The verifier already refuses `weight_scales` on an operation whose data
  // operand is not a dequantize, so this holds on anything that verified.
  cluster.lead = data.getDefiningOp<npu::DequantizeOp>();
  if (!cluster.lead)
    return std::nullopt;

  // **The one reader must be the quantize the calibrator put after it**,
  // because that is where the output scale and zero point are. A result that
  // is also read in f32 elsewhere would need both the integer result and the
  // f32 one, and contracting it would be half contracting it.
  Value result = op->getResult(0);
  if (!result.hasOneUse())
    return std::nullopt;
  cluster.sink = dyn_cast<npu::QuantizeOp>(*result.getUsers().begin());
  if (!cluster.sink)
    return std::nullopt;

  // The weights are quantized once, into the program, so they have to be
  // known at compile time; this instruction set has no instruction that would
  // quantize a weight computed at run time.
  cluster.weights = f32Constant(weights, cluster.weightOp);
  if (!cluster.weights)
    return std::nullopt;
  if (bias) {
    cluster.bias = f32Constant(bias, cluster.biasOp);
    if (!cluster.bias)
      return std::nullopt;
  }

  // The destination is dropped, because it has the f32 result's element type
  // and the instruction writes i8, so it has to be a destination and nothing
  // else.
  cluster.destination = destination.getDefiningOp<tensor::EmptyOp>();
  if (!cluster.destination)
    return std::nullopt;

  return cluster;
}

/// The plan for one cluster, or a diagnostic naming the arithmetic that has no
/// integer form.
FailureOr<QuantizedPlan> planOne(Operation *op, Cluster cluster) {
  const float scaleX = cluster.lead.getScale().convertToFloat();
  const float scaleY = cluster.sink.getScale().convertToFloat();
  const int32_t zeroPointX = cluster.lead.getZeroPoint();
  const int32_t zeroPointY = cluster.sink.getZeroPoint();
  const int64_t channels = cluster.channels;
  const int64_t count = cluster.weights.getNumElements();
  const int64_t perChannel = channels == 0 ? 0 : count / channels;

  // ---- The weights, and the sum per channel the fold needs. ----------------
  SmallVector<int8_t> quantized;
  quantized.reserve(count);
  SmallVector<int64_t> sums(channels, 0);
  int64_t flat = 0;
  for (float value : cluster.weights.getValues<float>()) {
    const int64_t channel =
        cluster.channelIsColumn ? flat % channels : flat / perChannel;
    ++flat;
    const int8_t q = quantizeWeight(value, cluster.scales[channel]);
    quantized.push_back(q);
    sums[channel] += q;
  }

  // ---- The int32 bias, with the input zero point folded in. ----------------
  SmallVector<float> biasValues;
  if (cluster.bias)
    biasValues = llvm::to_vector(cluster.bias.getValues<float>());

  SmallVector<int32_t> biases;
  biases.reserve(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    std::string reason;
    std::optional<float> real;
    if (!biasValues.empty())
      real = biasValues[channel];
    std::optional<int32_t> folded =
        foldedBias(real, scaleX, cluster.scales[channel], zeroPointX,
                   sums[channel], &reason);
    if (!folded)
      return op->emitError()
             << "cannot be contracted into an integer instruction: output "
                "channel "
             << channel << " has no int32 bias, because " << reason
             << ". Section 14 accumulates in int32, so this is a calibration "
                "the machine cannot execute rather than a number to wrap";
    biases.push_back(*folded);
  }

  // ---- The rescale, one pair per output channel. ---------------------------
  SmallVector<RequantPair> pairs;
  pairs.reserve(channels);
  for (int64_t channel = 0; channel < channels; ++channel) {
    const double multiplier =
        requantizationMultiplier(scaleX, cluster.scales[channel], scaleY);
    std::string reason;
    std::optional<RequantPair> pair = decomposeMultiplier(multiplier, &reason);
    if (!pair)
      return op->emitError()
             << "cannot be contracted into an integer instruction: output "
                "channel "
             << channel << " needs a requantization multiplier of "
             << multiplier << " = (" << scaleX << " * "
             << cluster.scales[channel] << ") / " << scaleY
             << ", which has no M0 in [2^30, 2^31) and shift in [0, 31]: "
             << reason;
    pairs.push_back(*pair);
  }

  MLIRContext *context = op->getContext();
  auto i8 = IntegerType::get(context, 8);
  auto i32 = IntegerType::get(context, 32);

  QuantizedPlan plan;
  plan.quantizedInput = cluster.lead.getInput();
  plan.sink = cluster.sink;
  plan.inputZeroPoint = zeroPointX;
  plan.outputZeroPoint = zeroPointY;
  plan.weights = DenseElementsAttr::get(
      RankedTensorType::get(cluster.weights.getType().getShape(), i8),
      ArrayRef<int8_t>(quantized));
  plan.bias = DenseElementsAttr::get(RankedTensorType::get({channels}, i32),
                                     ArrayRef<int32_t>(biases));
  plan.scalar = pairs.empty() ? RequantPair{} : pairs.front();

  // **The table exists exactly when the scalar pair cannot say it.** Section
  // 14's per tensor arm quantizes every channel with one weight scale, so every
  // channel's pair is the same and the two scalar fields express all of them;
  // the fourth operand is what the machine needs when they differ, and nothing
  // else. Both forms execute to the same integers, which is the machine's own
  // contract for an instruction without the operand.
  const bool uniform = llvm::all_equal(pairs);
  if (!uniform) {
    SmallVector<int32_t> table(2 * channels, 0);
    for (auto [channel, pair] : llvm::enumerate(pairs)) {
      table[channel] = pair.multiplier;
      table[channels + channel] = pair.shift;
    }
    plan.rescale = DenseElementsAttr::get(
        RankedTensorType::get({2, channels}, i32), ArrayRef<int32_t>(table));
  }

  plan.inputs.push_back(cluster.lead);
  plan.inputs.push_back(cluster.weightOp);
  if (cluster.biasOp)
    plan.inputs.push_back(cluster.biasOp);
  plan.inputs.push_back(cluster.destination);
  return plan;
}

} // namespace

LogicalResult
mlir::npuisa::planQuantizedContractions(ModuleOp module,
                                        ContractionPlans &contractions) {
  WalkResult walk = module.walk([&](Operation *op) -> WalkResult {
    std::optional<Cluster> cluster = clusterOf(op);
    if (!cluster)
      return WalkResult::advance();
    FailureOr<QuantizedPlan> plan = planOne(op, *cluster);
    if (failed(plan))
      return WalkResult::interrupt();
    contractions.plans.try_emplace(op, std::move(*plan));
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();

  // **An input is consumed only when every one of its readers contracts.** A
  // dequantize that also feeds an f32 operation is still needed as an f32
  // value, and a constant another operation reads in f32 is still needed as
  // f32 data; both lower as they always did, and the contraction reads the i8
  // value and its own i8 weights beside them.
  for (const auto &entry : contractions.plans) {
    for (Operation *input : entry.second.inputs) {
      const bool onlyContracted =
          llvm::all_of(input->getUsers(), [&](Operation *user) {
            return contractions.plans.contains(user);
          });
      if (onlyContracted)
        contractions.consumed.insert(input);
    }
  }
  return success();
}
