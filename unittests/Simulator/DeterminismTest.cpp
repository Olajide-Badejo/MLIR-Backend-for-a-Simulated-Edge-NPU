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
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "gtest/gtest.h"

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

TEST(Determinism, OneThreadAndMaxThreadsAgreeBitwise) {
#ifdef _OPENMP
  const int maximum = omp_get_max_threads();
  std::cout << "[          ] OpenMP is on and reports " << maximum
            << " threads available.\n";
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

} // namespace
