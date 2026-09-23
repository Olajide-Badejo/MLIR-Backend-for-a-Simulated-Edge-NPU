//===- QuantizedContraction.h - Section 14's QDQ contraction ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The half of Section 14's quantized path that turns a calibrated f32 graph
// into integer arithmetic. `-npu-calibrate` leaves a convolution or a matrix
// multiplication wrapped in the QDQ form with its per output channel weight
// scales in an attribute; this works out the i8 weights, the int32 bias with
// the input zero point folded into it, and the requantization pair of every
// output channel, and `-npu-lower-to-npuisa` builds the one integer instruction
// that carries them.
//
// **Two halves, for the reason `ScratchpadAllocation.h` is its own file.** The
// arithmetic is integers and doubles in and integers out, and the failures
// worth testing are arithmetic failures, so the first half takes no IR and is
// unit tested in `unittests/Dialect/NPUISA/QuantizedContractionTest.cpp`. The
// second half reads the QDQ form and decides which operations contract.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_TRANSFORMS_QUANTIZEDCONTRACTION_H
#define NPU_DIALECT_NPUISA_TRANSFORMS_QUANTIZEDCONTRACTION_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::npuisa {

//===----------------------------------------------------------------------===//
// The arithmetic, which takes no IR.
//===----------------------------------------------------------------------===//

/// One output channel's requantization pair, `M = M0 * 2^-(31 + shift)`.
struct RequantPair {
  int32_t multiplier = 0;
  int32_t shift = 0;

  friend bool operator==(const RequantPair &, const RequantPair &) = default;
};

/// Round half to even, the tie rule Section 14 pins for every quantization.
///
/// Written out rather than taken from `std::rint` or `std::nearbyint`, which
/// honour the dynamic rounding mode, and returned as a `double` so that a
/// magnitude no integer type holds is still a number the caller can compare
/// rather than a conversion with undefined behaviour. The same rule, written
/// the same way, is the machine's own in `lib/Simulator/Kernels.cpp`.
double roundHalfToEven(double value);

/// `M = (scale_x * scale_w) / scale_y`, from the three f32 values the IR
/// carries, evaluated in double in exactly that order.
///
/// **The order is part of the arithmetic, not a detail of it.** A double
/// product and quotient each round once, so `(a * b) / c` and `a * (b / c)` can
/// differ in the last bit, and a last bit here can move `M0` by one. Section 14
/// pins this arithmetic once because two implementations that disagree about it
/// produce an accuracy bug nobody can localize, so the order is written down
/// here and `npu_frontend.calibration.requantization_multiplier` evaluates the
/// same expression the same way.
double requantizationMultiplier(float scaleX, float scaleW, float scaleY);

/// `M` as Section 14 decomposes it, `M0` in `[2^30, 2^31)` and the shift in
/// `[0, 31]`, or the sentence saying why it has no such form.
///
/// The same steps as `npu_frontend.calibration.decompose_multiplier`, which is
/// the Python half of the pinned arithmetic: halve while at or above one,
/// double while below one half, round `scaled * 2^31` half to even, and
/// renormalise when that rounds up to `2^31`. Every step before the rounding is
/// a multiplication or division by two and is exact in binary floating point,
/// so the two agree bit for bit rather than nearly; a test holds them to it
/// over every channel of every committed profile.
///
/// **Two ways it refuses, and neither is clamped.** A multiplier at or above
/// one, including one that only rounds up to one, needs a negative shift, and
/// the shift is a right shift. A multiplier below `2^-32` needs a shift past
/// 31, which the binary format's own check refuses.
std::optional<RequantPair> decomposeMultiplier(double multiplier,
                                               std::string *reason);

/// `q = clamp(rint(w / scale), -128, 127)` with a zero point of zero.
///
/// The pinned quantize rule, applied to a weight: symmetric, so the zero point
/// is zero and the rule is the activation's with that term gone. The rails are
/// the rule's `[-128, 127]`. A weight quantized with its own channel's scale
/// never reaches either, because the scale is its channel's largest magnitude
/// over 127; the clamp is there because the scale arrives in an attribute.
///
/// A non finite quotient saturates, and a NaN maps to the zero point, which is
/// what the machine's quantizer does with the same inputs.
int8_t quantizeWeight(float weight, float scale);

/// One output channel's int32 bias with the input zero point folded in:
/// `round(b / (scale_x * scale_w)) - zp_x * sum_k q_w[k]`, or the reason it
/// does not fit.
///
/// **The sum is over the whole window**, every tap of the filter, which is why
/// padding has to contribute `zp_x` for the fold to be exact at a padded output
/// position. With no bias the folded term alone is the bias.
///
/// A value outside int32 is refused rather than wrapped, the way Section 14's
/// accumulator guard refuses rather than widening: the machine adds this to an
/// int32 accumulator, so a bias that does not fit is a calibration that cannot
/// be executed rather than a number to truncate.
std::optional<int32_t> foldedBias(std::optional<float> bias, float scaleX,
                                  float scaleW, int32_t inputZeroPoint,
                                  int64_t weightSum, std::string *reason);

//===----------------------------------------------------------------------===//
// The planner, which reads the QDQ form.
//===----------------------------------------------------------------------===//

/// Everything the lowering needs to replace one calibrated operation, and the
/// quantize after it, with a single integer instruction.
struct QuantizedPlan {
  /// The i8 value the leading `npu.dequantize` reads. The instruction reads
  /// the buffer this value converts to, so whatever produced it stays: a
  /// quantize becomes the `QUANT` that turns the activation into integers, and
  /// an integer argument or another contraction's result is read as it is.
  Value quantizedInput;

  /// The trailing `npu.quantize`. The instruction's own i8 result replaces
  /// it, because an integer convolution already is a convolution and a
  /// requantization.
  Operation *sink = nullptr;

  /// The operations whose results only this operation reads and that the
  /// contraction therefore makes dead: the leading dequantize, the f32 weight
  /// and bias constants, and the f32 destination. Collected per plan and
  /// consumed only when every one of their readers contracts.
  SmallVector<Operation *, 4> inputs;

  /// The i8 weights, in the operation's own layout.
  DenseElementsAttr weights;

  /// The int32 bias, one per output channel, with the input zero point folded
  /// in. Present even when the operation had no bias, because the folded term
  /// has to live somewhere and because the operand list is positional.
  DenseElementsAttr bias;

  /// The per output channel table, i32 of shape (2, C), multipliers in row 0
  /// and shifts in row 1. **Null when every channel's pair is the same**, and
  /// that is the per tensor arm: the scalar pair then says everything the
  /// table would, so the instruction carries three operands.
  DenseElementsAttr rescale;

  int32_t inputZeroPoint = 0;
  int32_t outputZeroPoint = 0;

  /// The pair the instruction's two scalar fields carry, which the binary
  /// format requires at every integer result. Channel 0's: with no table it is
  /// every channel's, and with one it is the pair the machine applies to
  /// channel 0 through the table, so the field is never a number that is true
  /// of nothing.
  RequantPair scalar;
};

/// The contractions of one module, and the operations they make dead.
struct ContractionPlans {
  llvm::DenseMap<Operation *, QuantizedPlan> plans;
  /// Only the operations every reader of which contracts. A constant or a
  /// dequantize that something outside the contraction still reads is lowered
  /// as it always was.
  llvm::DenseSet<Operation *> consumed;
};

/// Plans every contraction in the module, or refuses one by name.
///
/// **An operation contracts when the calibrator left it whole**: it carries
/// `weight_scales`, its data operand is an `npu.dequantize`, its one reader is
/// an `npu.quantize`, and its weights and any bias are f32 constants. An
/// operation missing any of those is **not contracted and not refused**: it
/// stays in the QDQ form and lowers to `QUANT`, f32 compute and `DEQUANT` as it
/// did before this existed, which is Section 14's partial coverage rule. It is
/// never half contracted.
///
/// What **is** refused is an operation that has everything and cannot be
/// expressed: a folded bias outside int32, or a channel whose multiplier has no
/// `M0` and shift. Those are calibrations the machine cannot execute, and each
/// is a diagnostic naming the operation and the channel.
LogicalResult planQuantizedContractions(ModuleOp module,
                                        ContractionPlans &plans);

} // namespace mlir::npuisa

#endif // NPU_DIALECT_NPUISA_TRANSFORMS_QUANTIZEDCONTRACTION_H
