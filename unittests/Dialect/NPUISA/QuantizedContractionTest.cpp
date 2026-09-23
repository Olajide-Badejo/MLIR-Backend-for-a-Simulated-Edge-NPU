//===- QuantizedContractionTest.cpp - Contraction arithmetic ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 14's pinned arithmetic as `-npu-lower-to-npuisa` computes it when it
// contracts a calibrated operation into an integer instruction.
//
// **Every expected value here was worked out by hand**, from the section's own
// words and exact fractions, before the code under test ran, and the working is
// beside each one. None is derived from the functions under test, which is
// what makes a case a test of them rather than a restatement.
//
// **Why these are unit tests rather than lit tests.** The lit files check what
// the contraction writes into a program, and they carry the same hand computed
// numbers for two real operations. What they cannot reach is the edges: a
// multiplier within a rounding of two to the 31, a shift of exactly 31, a fold
// that lands on the last int32. Writing those as programs would mean finding
// f32 scales whose double quotient lands on the edge, which is a slow way of
// writing down one double.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/QuantizedContraction.h"

#include "gtest/gtest.h"

#include <cfenv>
#include <cmath>
#include <limits>
#include <string>

using namespace mlir::npuisa;

namespace {

/// `2^exponent` as a double, exactly.
double twoTo(int exponent) { return std::ldexp(1.0, exponent); }

//===----------------------------------------------------------------------===//
// The tie rule.
//===----------------------------------------------------------------------===//

TEST(QuantizedArithmetic, ATieGoesToTheEvenNeighbour) {
  EXPECT_EQ(roundHalfToEven(0.5), 0.0);
  EXPECT_EQ(roundHalfToEven(1.5), 2.0);
  EXPECT_EQ(roundHalfToEven(2.5), 2.0);
  EXPECT_EQ(roundHalfToEven(3.5), 4.0);
  EXPECT_EQ(roundHalfToEven(-0.5), 0.0);
  EXPECT_EQ(roundHalfToEven(-1.5), -2.0);
  EXPECT_EQ(roundHalfToEven(-2.5), -2.0);
  // Everything that is not a tie goes to the nearest neighbour.
  EXPECT_EQ(roundHalfToEven(2.4999999), 2.0);
  EXPECT_EQ(roundHalfToEven(2.5000001), 3.0);
  EXPECT_EQ(roundHalfToEven(-2.5000001), -3.0);
  // A magnitude no int64 holds is still a number, not a conversion.
  EXPECT_EQ(roundHalfToEven(1e30), 1e30);
}

TEST(QuantizedArithmetic, TheTieRuleSurvivesAChangedRoundingMode) {
  // Section 14's own test, for this copy of the rule: set `FE_TOWARDZERO` and
  // the answers do not move. `std::rint` would answer 2.5 with 2 and 3.5 with
  // 3 in this mode, which is the trap the section names.
  const int previous = std::fegetround();
  ASSERT_EQ(std::fesetround(FE_TOWARDZERO), 0);
  const double tieUp = roundHalfToEven(3.5);
  const double tieDown = roundHalfToEven(2.5);
  const double above = roundHalfToEven(2.6);
  const double negative = roundHalfToEven(-2.6);
  const int8_t weight = quantizeWeight(0.9375f, 0.375f);
  std::fesetround(previous);

  EXPECT_EQ(tieUp, 4.0);
  EXPECT_EQ(tieDown, 2.0);
  EXPECT_EQ(above, 3.0);
  EXPECT_EQ(negative, -3.0);
  // 0.9375 / 0.375 is exactly 2.5, so the division has no rounding for the
  // mode to change and the tie is the rule's alone.
  EXPECT_EQ(weight, 2);
}

//===----------------------------------------------------------------------===//
// M, and the order it is evaluated in.
//===----------------------------------------------------------------------===//

TEST(QuantizedArithmetic, TheMultiplierIsTheProductOverTheOutputScale) {
  // (0.5 * 0.375) / 0.25 = 0.75, and every one of those is exact in binary.
  EXPECT_EQ(requantizationMultiplier(0.5f, 0.375f, 0.25f), 0.75);
  EXPECT_EQ(requantizationMultiplier(0.25f, 0.21875f, 0.25f), 0.21875);
}

TEST(QuantizedArithmetic, TheProductIsTakenBeforeTheQuotient) {
  // The order is part of the pinned arithmetic. With the f32 values of 0.7,
  // 0.1 and 0.7, product first returns the f32 value of 0.1 exactly,
  // 0x1.99999a0000000p-4, and quotient first returns one unit in the last
  // place below it, 0x1.999999fffffffp-4. The Python half evaluates product
  // first, so this one does, and a change of association here would be a
  // disagreement between the two halves that no accuracy number could
  // localize.
  const double productFirst = requantizationMultiplier(0.7f, 0.1f, 0.7f);
  EXPECT_EQ(productFirst, static_cast<double>(0.1f));
  const double quotientFirst =
      static_cast<double>(0.7f) *
      (static_cast<double>(0.1f) / static_cast<double>(0.7f));
  EXPECT_NE(productFirst, quotientFirst);
}

//===----------------------------------------------------------------------===//
// The decomposition.
//===----------------------------------------------------------------------===//

/// The pair, or a failure naming the reason.
RequantPair decompose(double multiplier) {
  std::string reason;
  std::optional<RequantPair> pair = decomposeMultiplier(multiplier, &reason);
  EXPECT_TRUE(pair.has_value()) << multiplier << ": " << reason;
  return pair.value_or(RequantPair{});
}

TEST(QuantizedArithmetic, HandComputedMultipliersDecompose) {
  // M = M0 * 2^-(31 + shift), M0 in [2^30, 2^31).
  //
  //   0.5      already in [0.5, 1)        M0 = 0.5 * 2^31     = 1073741824
  //   0.75     already in [0.5, 1)        M0 = 0.75 * 2^31    = 1610612736
  //   0.3125   doubled once to 0.625      M0 = 0.625 * 2^31   = 1342177280
  //   0.21875  doubled twice to 0.875     M0 = 0.875 * 2^31   = 1879048192
  //   2^-32    doubled 31 times to 0.5    M0 = 2^30, the largest legal shift
  EXPECT_EQ(decompose(0.5), (RequantPair{1073741824, 0}));
  EXPECT_EQ(decompose(0.75), (RequantPair{1610612736, 0}));
  EXPECT_EQ(decompose(0.3125), (RequantPair{1342177280, 1}));
  EXPECT_EQ(decompose(0.21875), (RequantPair{1879048192, 2}));
  EXPECT_EQ(decompose(twoTo(-32)), (RequantPair{1073741824, 31}));
}

TEST(QuantizedArithmetic, AThirdRoundsToTheNearestMultiplier) {
  // The double nearest 1/3 doubles once to the double nearest 2/3, and
  // 2/3 * 2^31 is 1431655765.33..., which rounds down.
  EXPECT_EQ(decompose(1.0 / 3.0), (RequantPair{1431655765, 1}));
}

TEST(QuantizedArithmetic, ATieInTheMultiplierRoundsToEven) {
  // (2^31 + 1) / 2^32 puts scaled * 2^31 at 1073741824.5 exactly, and the even
  // neighbour is below; (2^31 + 3) / 2^32 puts it at 1073741825.5, and the even
  // neighbour is above.
  EXPECT_EQ(decompose((twoTo(31) + 1.0) / twoTo(32)),
            (RequantPair{1073741824, 0}));
  EXPECT_EQ(decompose((twoTo(31) + 3.0) / twoTo(32)),
            (RequantPair{1073741826, 0}));
}

TEST(QuantizedArithmetic, AMultiplierThatRoundsUpToTwoToThe31Renormalises) {
  // 0.5 - 2^-40 is below one half, so it doubles to 1 - 2^-39 at a shift of
  // one, and (1 - 2^-39) * 2^31 = 2^31 - 2^-8 rounds to 2^31, which has no
  // int32. The step that halves it back gives 2^30 at a shift of zero, which
  // is the representation of one half: the nearest the pair can say.
  EXPECT_EQ(decompose(0.5 - twoTo(-40)), (RequantPair{1073741824, 0}));
}

TEST(QuantizedArithmetic, OneAndAboveHaveNoRightShift) {
  // 1.0 halves to 0.5 at a shift of -1. 1 - 2^-40 stays at a shift of zero
  // and then rounds up to 2^31, which renormalises to a shift of -1 as well:
  // a multiplier that only rounds to one is refused like one.
  for (double multiplier : {1.0, 1.5, 1.0 - twoTo(-40)}) {
    std::string reason;
    EXPECT_FALSE(decomposeMultiplier(multiplier, &reason).has_value())
        << multiplier;
    EXPECT_NE(reason.find("negative shift"), std::string::npos) << reason;
  }
}

TEST(QuantizedArithmetic, BelowTwoToTheMinus32NeedsAShiftPast31) {
  // 2^-33 doubles 32 times to reach one half.
  std::string reason;
  EXPECT_FALSE(decomposeMultiplier(twoTo(-33), &reason).has_value());
  EXPECT_NE(reason.find("right shift of 32"), std::string::npos) << reason;
}

TEST(QuantizedArithmetic, ANonPositiveOrNonFiniteMultiplierIsRefused) {
  for (double multiplier : {0.0, -0.5, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
    std::string reason;
    EXPECT_FALSE(decomposeMultiplier(multiplier, &reason).has_value())
        << multiplier;
    EXPECT_NE(reason.find("finite and strictly positive"), std::string::npos)
        << reason;
  }
  // The reason is optional to ask for.
  EXPECT_FALSE(decomposeMultiplier(0.0, nullptr).has_value());
}

//===----------------------------------------------------------------------===//
// The weights.
//===----------------------------------------------------------------------===//

TEST(QuantizedArithmetic, AWeightRoundsHalfToEven) {
  // The two ties the lit case carries: 0.9375 / 0.375 = 2.5 goes to 2, and
  // -0.234375 / 0.15625 = -1.5 goes to -2. Rounding half away from zero would
  // answer 3 and -2, and rounding half up 3 and -1.
  EXPECT_EQ(quantizeWeight(0.9375f, 0.375f), 2);
  EXPECT_EQ(quantizeWeight(-0.234375f, 0.15625f), -2);
  EXPECT_EQ(quantizeWeight(0.65625f, 0.21875f), 3);
}

TEST(QuantizedArithmetic, AWeightSaturatesAtThePinnedRails) {
  // The quantize rule's rails, [-128, 127], with the zero point at zero.
  EXPECT_EQ(quantizeWeight(1000.0f, 1.0f), 127);
  EXPECT_EQ(quantizeWeight(127.4f, 1.0f), 127);
  EXPECT_EQ(quantizeWeight(-1000.0f, 1.0f), -128);
  EXPECT_EQ(quantizeWeight(-128.4f, 1.0f), -128);
  EXPECT_EQ(quantizeWeight(-127.6f, 1.0f), -128);
  EXPECT_EQ(quantizeWeight(-127.4f, 1.0f), -127);
}

TEST(QuantizedArithmetic, ANonFiniteWeightTakesTheMachinesAnswer) {
  // What the machine's quantizer does with the same inputs: a NaN is the zero
  // point and an infinity is the rail on its side.
  EXPECT_EQ(quantizeWeight(std::numeric_limits<float>::quiet_NaN(), 1.0f), 0);
  EXPECT_EQ(quantizeWeight(std::numeric_limits<float>::infinity(), 1.0f), 127);
  EXPECT_EQ(quantizeWeight(-std::numeric_limits<float>::infinity(), 1.0f),
            -128);
}

//===----------------------------------------------------------------------===//
// The bias, with the input zero point folded in.
//===----------------------------------------------------------------------===//

/// The folded bias, or a failure naming the reason.
int32_t fold(std::optional<float> bias, float scaleX, float scaleW,
             int32_t zeroPoint, int64_t sum) {
  std::string reason;
  std::optional<int32_t> value =
      foldedBias(bias, scaleX, scaleW, zeroPoint, sum, &reason);
  EXPECT_TRUE(value.has_value()) << reason;
  return value.value_or(0);
}

TEST(QuantizedArithmetic, TheBiasFoldsTheInputZeroPointOverTheWholeWindow) {
  // The lit convolution's two channels, zp_x = 3 and scale_x = 0.5:
  //   1.0 / (0.5 * 0.375) = 5.33 rounds to 5, the weights sum to 2 - 1 = 1,
  //   so 5 - 3 * 1 = 2.
  //   f32(-0.4) / (0.5 * 0.15625) = -5.12 rounds to -5, the weights sum to
  //   4 - 2 = 2, so -5 - 3 * 2 = -11.
  EXPECT_EQ(fold(1.0f, 0.5f, 0.375f, 3, 1), 2);
  EXPECT_EQ(fold(-0.4f, 0.5f, 0.15625f, 3, 2), -11);
  // The lit matrix multiplication's two columns, zp_x = -2 and scale_x = 0.25:
  //   0.5 / 0.1875 = 2.67 rounds to 3, and 3 + 2 * 1 = 5.
  //   f32(-0.1) / 0.0546875 = -1.83 rounds to -2, and -2 + 2 * 2 = 2.
  EXPECT_EQ(fold(0.5f, 0.25f, 0.75f, -2, 1), 5);
  EXPECT_EQ(fold(-0.1f, 0.25f, 0.21875f, -2, 2), 2);
}

TEST(QuantizedArithmetic, WithNoBiasTheFoldedTermAloneIsTheBias) {
  EXPECT_EQ(fold(std::nullopt, 0.5f, 0.375f, 3, 1), -3);
  EXPECT_EQ(fold(std::nullopt, 0.5f, 0.375f, -128, 5), 640);
  EXPECT_EQ(fold(std::nullopt, 0.5f, 0.375f, 0, 12345), 0);
}

TEST(QuantizedArithmetic, TheFoldReachesBothInt32ExtremesAndNoFurther) {
  // -(-1) * (2^31 - 1) is the largest int32, and one more is refused;
  // -(1) * 2^31 is the smallest, and one more is refused.
  EXPECT_EQ(fold(std::nullopt, 1.0f, 1.0f, -1, 2147483647),
            std::numeric_limits<int32_t>::max());
  EXPECT_EQ(fold(std::nullopt, 1.0f, 1.0f, 1, 2147483648),
            std::numeric_limits<int32_t>::min());

  std::string reason;
  EXPECT_FALSE(foldedBias(std::nullopt, 1.0f, 1.0f, -1, 2147483648, &reason)
                   .has_value());
  EXPECT_NE(reason.find("2147483648"), std::string::npos) << reason;
  EXPECT_FALSE(
      foldedBias(std::nullopt, 1.0f, 1.0f, 1, 2147483649, &reason).has_value());
  EXPECT_NE(reason.find("-2147483649"), std::string::npos) << reason;
}

TEST(QuantizedArithmetic, ABiasOutsideInt32IsRefusedRatherThanWrapped) {
  // 1e6 / (1e-3 * 1e-3) is about 1e12, well past 2^31. Wrapped, it would be
  // an unrelated int32 and the channel would be silently wrong everywhere.
  std::string reason;
  EXPECT_FALSE(foldedBias(1e6f, 1e-3f, 1e-3f, 0, 0, &reason).has_value());
  EXPECT_NE(reason.find("does not fit in the int32"), std::string::npos)
      << reason;

  // Far enough out that no integer type holds it: refused in double, before
  // any conversion.
  EXPECT_FALSE(foldedBias(3e38f, 1e-30f, 1e-30f, 0, 0, &reason).has_value());
  EXPECT_NE(reason.find("does not fit in the int32"), std::string::npos)
      << reason;

  EXPECT_FALSE(foldedBias(std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f,
                          0, 0, &reason)
                   .has_value());
  EXPECT_NE(reason.find("not a finite number"), std::string::npos) << reason;
}

} // namespace
