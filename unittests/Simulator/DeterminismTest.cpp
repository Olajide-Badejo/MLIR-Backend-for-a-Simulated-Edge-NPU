//===- DeterminismTest.cpp - one thread and many agree --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 10.3, asserted: the same input under one thread and under the maximum
// thread count produces **bitwise equal** output buffers.
//
// Bitwise, not close. The accumulation order inside each output element
// determines its last bits, the golden files of Phase P8 depend on that order,
// and a test written with a tolerance would pass on the day somebody moved the
// reduction into the parallel region and left the goldens to fail three phases
// later on a machine with a different core count.
//
// The kernel parallelises over the batch and output channel dimensions only.
// Each thread writes a disjoint output region, so there is no reduction race
// and no atomics, and the loops over input channel and kernel window stay
// strictly sequential and in their original order. That is why this test can
// assert bitwise equality at all: a parallel reduction could not.
//
// **When OpenMP is absent** this test still runs and still passes, because both
// runs are then single threaded. It says so in its output rather than being
// skipped, because a test that silently becomes vacuous is worse than no test.
//
// **And that last sentence is why `TheKernelsAgreeWithThisTestAboutOpenMP`
// exists.** Saying so in the output is not enough on its own, because between
// P7 and P12 the output said the opposite of the truth: this file was compiled
// with `-fopenmp` and `lib/Simulator/Kernels.cpp` was not, so the line below
// printed a thread count of 28 while the kernel it exercises had no parallel
// region in it at all, and the bitwise comparison ran twice over the same
// serial code path. Nothing here could see that, because `_OPENMP` is a
// property of a translation unit and this translation unit's answer was
// correct about itself. D-0047. The first test in this file now asks the
// kernels rather than the preprocessor, and it is red in exactly the build
// shape that fault produced.
//
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "NPU/Simulator/Simulator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <iostream>
#include <vector>

using namespace nbin;
using namespace npusim;

namespace {

/// A deterministic stream of values in [-1, 1).
///
/// A fixed seed and an explicit generator rather than `std::mt19937` with a
/// default seed: this test's whole subject is reproducibility, and a sequence
/// that could differ between standard libraries would undermine it.
///
/// The shift is 32 and not 33, which is D-0029. One bit more leaves the values
/// in [-1, 0), and a convolution over inputs and weights that are all negative
/// has every product positive, so the reduction this test exists to hold still
/// would have been running over terms that all carry the same sign. That is the
/// easiest possible case for a summation order to survive.
class Stream {
public:
  float next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t bits = static_cast<uint32_t>(state >> 32);
    return static_cast<float>(bits) / 2147483648.0f - 1.0f;
  }

private:
  uint64_t state = 0x6e70753750374445ull;
};

/// A convolution big enough that the parallel loop has work to distribute:
/// batch 4 times eight output channels is thirty two iterations of the
/// collapsed loop, and a reduction over eight input channels and a three by
/// three window is long enough that a reordering would show.
Program buildConvolution() {
  Stream stream;
  std::vector<float> input(4 * 8 * 8 * 8);
  for (float &value : input)
    value = stream.next();
  std::vector<float> filter(8 * 8 * 3 * 3);
  for (float &value : filter)
    value = stream.next();

  Builder builder;
  const std::vector<int64_t> inputShape = {4, 8, 8, 8};
  const std::vector<int64_t> filterShape = {8, 8, 3, 3};
  const std::vector<int64_t> resultShape = {4, 8, 8, 8};

  const int64_t inputRegion = builder.constant(inputShape, input);
  const int64_t filterRegion = builder.constant(filterShape, filter);
  const int64_t inputBuffer = builder.scratch(4 * 8 * 8 * 8);
  const int64_t filterBuffer = builder.scratch(8 * 8 * 3 * 3);
  const int64_t resultBuffer = builder.scratch(4 * 8 * 8 * 8);
  const int64_t sink = builder.output(resultShape);

  builder.add(dmaLoad(inputBuffer, inputShape,
                      at(MemSpace::Dram, inputRegion, inputShape)));
  builder.add(dmaLoad(filterBuffer, filterShape,
                      at(MemSpace::Dram, filterRegion, filterShape)));

  Instruction convolution =
      compute(Opcode::CONV2D, resultBuffer, resultShape,
              {at(MemSpace::Scratchpad, inputBuffer, inputShape),
               at(MemSpace::Scratchpad, filterBuffer, filterShape)});
  convolution.strides = {1, 1};
  convolution.pads = {1, 1, 1, 1};
  convolution.dilations = {1, 1};
  convolution.group = 1;
  builder.add(std::move(convolution));

  builder.add(dmaStore(sink, resultShape,
                       at(MemSpace::Scratchpad, resultBuffer, resultShape)));
  builder.add(halt());

  // Scratchpad: input 4 * 8 * 8 * 8 = 2048 elements, filter 8 * 8 * 3 * 3 = 576,
  // result 2048. 4672 elements * 4 bytes = 18688 bytes.
  return builder.finish(18688);
}

std::vector<float> runWith(int threads) {
#ifdef _OPENMP
  omp_set_num_threads(threads);
#else
  (void)threads;
#endif
  Harness harness(buildConvolution());
  const SimResult result = harness.run();
  EXPECT_TRUE(result.ok()) << result.error.value_or("");
  return harness.outputF32(0);
}

TEST(Determinism, TheKernelsAgreeWithThisTestAboutOpenMP) {
  // The two halves of D-0047, compared. `_OPENMP` below is this file's; the
  // function is `Kernels.cpp`'s. They can only disagree in one direction and
  // that direction has happened: this file links NPUSimulator and therefore
  // picks up the OpenMP usage requirement attached to it, while the sources
  // NPUSimulator is built from compile in a separate object library that the
  // requirement never reached.
  //
  // The reverse cannot happen and is asserted anyway, because an assertion
  // that only holds one way is one somebody eventually inverts.
#ifdef _OPENMP
  EXPECT_TRUE(nbin::kernelsUseOpenMP())
      << "this test was compiled with OpenMP and lib/Simulator/Kernels.cpp was "
         "not, so the convolution has no parallel region and every thread "
         "count assertion in this file is vacuous. That is D-0047: the OpenMP "
         "usage requirement is attached to the NPUSimulator target and the "
         "sources compile in obj.NPUSimulator, which does not receive it. The "
         "fix is the directory scope add_compile_options in "
         "lib/Simulator/CMakeLists.txt";
  std::cout << "[          ] the kernels report " << nbin::kernelThreadCount()
            << " threads, read inside Kernels.cpp.\n";
#else
  EXPECT_FALSE(nbin::kernelsUseOpenMP())
      << "the kernels were compiled with OpenMP and this test was not, which "
         "no arrangement of this project's CMake produces and which would mean "
         "the two are being configured from different places";
  EXPECT_EQ(nbin::kernelThreadCount(), 1);
#endif
}

TEST(Determinism, OneThreadAndMaxThreadsAgreeBitwise) {
#ifdef _OPENMP
  const int maximum = omp_get_max_threads();
  std::cout << "[          ] OpenMP is on and reports " << maximum
            << " threads available, and the kernels report "
            << nbin::kernelThreadCount() << ".\n";
#else
  const int maximum = 1;
  std::cout << "[          ] OpenMP is off in this build, so both runs below "
               "are single threaded and this test asserts less than it does "
               "where OpenMP is present.\n";
#endif

  const std::vector<float> single = runWith(1);
  const std::vector<float> many = runWith(maximum);

  ASSERT_EQ(single.size(), many.size());
  ASSERT_FALSE(single.empty());

  // memcmp rather than a loop of float comparisons. A NaN produced by a bug
  // compares unequal to itself under `==`, so a value comparison would report a
  // difference that is not one, and a tolerance would hide the difference this
  // test exists to find.
  EXPECT_EQ(std::memcmp(single.data(), many.data(),
                        single.size() * sizeof(float)),
            0)
      << "the convolution produced different bits at " << maximum
      << " threads than at one, which means a reduction moved into the "
         "parallel region";

#ifdef _OPENMP
  // And put the thread count back, so a test that runs after this one is not
  // quietly running under whatever this one left behind.
  omp_set_num_threads(maximum);
#endif
}

TEST(Determinism, TheCycleCountDoesNotDependOnTheThreadCount) {
  // The cost model is arithmetic over the shapes, so the host's core count has
  // no business in it. Stating that as a test is cheap and it is the assertion
  // that would fail if somebody ever charged cycles from wall clock time.
#ifdef _OPENMP
  // Captured before anything is set, because `omp_get_max_threads` reports the
  // count the next region would use, which is whatever was last set. Reading it
  // after setting one would read back the one.
  const int maximum = omp_get_max_threads();
#endif

  Harness first(buildConvolution());
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  const SimResult single = first.run();

  Harness second(buildConvolution());
#ifdef _OPENMP
  omp_set_num_threads(maximum);
#endif
  const SimResult many = second.run();

  ASSERT_TRUE(single.ok()) << single.error.value_or("");
  ASSERT_TRUE(many.ok()) << many.error.value_or("");
  EXPECT_DOUBLE_EQ(single.stats.cycles, many.stats.cycles);
  EXPECT_EQ(single.stats.macs, many.stats.macs);
}

//===----------------------------------------------------------------------===//
// The integer kernel, under the same two assertions.
//
// Section 10.3's determinism rule is about the parallel region, not about the
// element type, so the integer convolution is held to it too. The two claims
// are not the same claim twice, and saying why is the point of the pair:
//
// **The f32 case can fail for two reasons and the integer case for one.** In
// f32 a reduction moved into the parallel region changes the summation order
// and therefore the bits, so the bitwise comparison catches an ordering bug and
// a race alike. Integer addition is associative, so a reordering alone cannot
// move a bit; what is left for this case to catch is the **race**, two threads
// writing the same output element, which no amount of associativity forgives.
//
// A test that only ran the f32 case would leave the integer kernel's own team
// cap and its own `collapse(2)` unexercised, and those are two clauses this
// phase wrote rather than inherited.
//===----------------------------------------------------------------------===//

/// The integer twin of `buildConvolution`, at the same extents.
///
/// The same thirty two iterations of the collapsed loop and the same eight
/// channel by three by three reduction, so the two are comparable, and a padded
/// window at a non zero input zero point so that the padding rule runs inside
/// the parallel region rather than beside it.
Program buildIntegerConvolution() {
  Stream stream;
  std::vector<int8_t> input(4 * 8 * 8 * 8);
  for (int8_t &value : input)
    value = static_cast<int8_t>(std::floor(stream.next() * 128.0f));
  std::vector<int8_t> filter(8 * 8 * 3 * 3);
  for (int8_t &value : filter)
    value = static_cast<int8_t>(std::floor(stream.next() * 128.0f));
  std::vector<int32_t> bias(8);
  for (int32_t &value : bias)
    value = static_cast<int32_t>(std::floor(stream.next() * 4096.0f));

  Builder builder;
  const std::vector<int64_t> inputShape = {4, 8, 8, 8};
  const std::vector<int64_t> filterShape = {8, 8, 3, 3};
  const std::vector<int64_t> biasShape = {8};
  const std::vector<int64_t> resultShape = {4, 8, 8, 8};

  const int64_t inputRegion = builder.constantI8(inputShape, input);
  const int64_t filterRegion = builder.constantI8(filterShape, filter);
  const int64_t biasRegion = builder.constantI32(biasShape, bias);

  // The i32 bias first, at offset zero: a four byte element is read through an
  // accessor that requires a four byte aligned address, and this builder packs
  // tightly where the real allocator aligns to 64.
  const int64_t biasBuffer = builder.scratch(8, ElemType::I32);
  const int64_t inputBuffer = builder.scratch(4 * 8 * 8 * 8, ElemType::I8);
  const int64_t filterBuffer = builder.scratch(8 * 8 * 3 * 3, ElemType::I8);
  const int64_t resultBuffer = builder.scratch(4 * 8 * 8 * 8, ElemType::I8);
  const int64_t sink = builder.output(resultShape, ElemType::I8);

  builder.add(dmaLoad(biasBuffer, biasShape,
                      at(MemSpace::Dram, biasRegion, biasShape,
                         ElemType::I32)));
  builder.add(dmaLoad(inputBuffer, inputShape,
                      at(MemSpace::Dram, inputRegion, inputShape,
                         ElemType::I8)));
  builder.add(dmaLoad(filterBuffer, filterShape,
                      at(MemSpace::Dram, filterRegion, filterShape,
                         ElemType::I8)));

  Instruction convolution =
      compute(Opcode::CONV2D, resultBuffer, resultShape,
              {at(MemSpace::Scratchpad, inputBuffer, inputShape, ElemType::I8),
               at(MemSpace::Scratchpad, filterBuffer, filterShape,
                  ElemType::I8),
               at(MemSpace::Scratchpad, biasBuffer, biasShape, ElemType::I32)},
              ElemType::I8);
  convolution.strides = {1, 1};
  convolution.pads = {1, 1, 1, 1};
  convolution.dilations = {1, 1};
  convolution.group = 1;
  convolution.zeroPoint = -19;
  convolution.requantMultiplier = kHalfMultiplier;
  convolution.requantShift = 4;
  builder.add(std::move(convolution));

  builder.add(dmaStore(sink, resultShape,
                       at(MemSpace::Scratchpad, resultBuffer, resultShape,
                          ElemType::I8)));
  builder.add(halt());

  // Scratchpad: the bias is 8 int32, which is 32 bytes, and the three i8
  // buffers are 2048, 576 and 2048 elements at one byte each. 32 + 4672 = 4704.
  return builder.finish(4704);
}

std::vector<int32_t> runIntegerWith(int threads) {
#ifdef _OPENMP
  omp_set_num_threads(threads);
#else
  (void)threads;
#endif
  Harness harness(buildIntegerConvolution());
  const SimResult result = harness.run();
  EXPECT_TRUE(result.ok()) << result.error.value_or("");
  return harness.outputI8(0);
}

TEST(Determinism, TheIntegerKernelAgreesWithItselfAtEveryThreadCount) {
#ifdef _OPENMP
  const int maximum = omp_get_max_threads();
#else
  const int maximum = 1;
#endif

  const std::vector<int32_t> single = runIntegerWith(1);
  const std::vector<int32_t> many = runIntegerWith(maximum);

  ASSERT_EQ(single.size(), many.size());
  ASSERT_FALSE(single.empty());
  EXPECT_EQ(single, many)
      << "the quantized convolution produced different bytes at " << maximum
      << " threads than at one. Integer addition is associative, so this is "
         "not a summation order: it is two threads writing the same output "
         "element";

  // The output has to be worth comparing. A quantized convolution whose every
  // element saturated to one rail would compare a constant against a constant
  // and pass whatever the kernel did, and the requantization shift above is
  // chosen so that it does not.
  const int32_t low = *std::min_element(single.begin(), single.end());
  const int32_t high = *std::max_element(single.begin(), single.end());
  EXPECT_LT(low, 0) << "every output is non negative, so this comparison is "
                       "weaker than it looks";
  EXPECT_GT(high, 0);
  EXPECT_GT(high - low, 8)
      << "the outputs span " << (high - low)
      << " counts, which is close enough to constant that comparing two runs "
         "of it asserts very little";

#ifdef _OPENMP
  omp_set_num_threads(maximum);
#endif
}

TEST(Determinism, TheIntegerCycleCountDoesNotDependOnTheThreadCount) {
#ifdef _OPENMP
  const int maximum = omp_get_max_threads();
#endif

  Harness first(buildIntegerConvolution());
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  const SimResult single = first.run();

  Harness second(buildIntegerConvolution());
#ifdef _OPENMP
  omp_set_num_threads(maximum);
#endif
  const SimResult many = second.run();

  ASSERT_TRUE(single.ok()) << single.error.value_or("");
  ASSERT_TRUE(many.ok()) << many.error.value_or("");
  EXPECT_DOUBLE_EQ(single.stats.cycles, many.stats.cycles);
  EXPECT_EQ(single.stats.macs, many.stats.macs);

  // And the integer MAC counter is the one that separates an int8 cell from an
  // f32 one in the result schema, so it is asserted here rather than assumed:
  // an integer convolution counts its multiplies in both fields, and an f32 one
  // counts them in neither of the two ways this would catch.
  EXPECT_EQ(single.stats.int8Macs, single.stats.macs);
  EXPECT_GT(single.stats.int8Macs, 0u);
}

} // namespace
