//===- PropertyTest.cpp - Randomized encode/decode round trip -------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// The encoder had four hand written tests, all of them shapes the LeNet path
// happens to produce. This generates randomized programs instead and asserts
// that encode then decode returns something structurally identical.
//
// The seed is fixed so a failure reproduces. When one does, the iteration index
// is printed, and re-running with that iteration is enough to get the same
// program back.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "gtest/gtest.h"

#include <random>

using namespace npu;

namespace {

constexpr unsigned kSeed = 0x5eed1234;
constexpr int kIterations = 1000;

std::vector<int64_t> randomShape(std::mt19937 &rng) {
  std::uniform_int_distribution<int> rank(1, 4);
  std::uniform_int_distribution<int> extent(1, 12);
  std::vector<int64_t> shape;
  int r = rank(rng);
  for (int i = 0; i < r; ++i)
    shape.push_back(extent(rng));
  return shape;
}

std::vector<int64_t> randomAddrs(std::mt19937 &rng, int count) {
  std::uniform_int_distribution<int64_t> addr(0, 1 << 16);
  std::vector<int64_t> v;
  for (int i = 0; i < count; ++i)
    v.push_back(addr(rng) * 4); // addresses are fp32 aligned
  return v;
}

// A random but structurally plausible program. This deliberately does not aim
// to be a program validate() would accept: the round trip is a property of the
// byte format, not of semantic validity, and restricting the generator to valid
// programs would stop it exercising the odd corners of the encoding.
Program randomProgram(std::mt19937 &rng) {
  std::uniform_int_distribution<int> small(0, 3);
  std::uniform_int_distribution<int> instructionCount(0, 12);
  std::uniform_int_distribution<uint16_t> anyOpcode(0, kMaxOpcode);
  std::uniform_int_distribution<int64_t> bytes(0, 1 << 20);
  std::uniform_real_distribution<float> value(-1e3f, 1e3f);

  Program p;
  p.version = Program::kVersion;
  p.scratchpadBytes = bytes(rng);
  p.dramBytes = bytes(rng);

  int nIn = small(rng);
  for (int i = 0; i < nIn; ++i)
    p.inputs.push_back({bytes(rng), randomShape(rng)});
  int nOut = small(rng);
  for (int i = 0; i < nOut; ++i)
    p.outputs.push_back({bytes(rng), randomShape(rng)});

  int nConst = small(rng);
  for (int i = 0; i < nConst; ++i) {
    MemRegion region{bytes(rng), randomShape(rng)};
    int64_t n = 1;
    for (int64_t d : region.shape)
      n *= d;
    std::vector<float> data;
    for (int64_t k = 0; k < n; ++k)
      data.push_back(value(rng));
    p.constants.push_back(region);
    p.constantData.push_back(std::move(data));
  }

  int nInstr = instructionCount(rng);
  for (int i = 0; i < nInstr; ++i) {
    Instruction in;
    in.op = static_cast<Opcode>(anyOpcode(rng));
    in.resultAddr = randomAddrs(rng, 1)[0];
    in.resultShape = randomShape(rng);
    in.operandAddrs = randomAddrs(rng, small(rng));
    in.dramAddr = bytes(rng);
    in.activation = small(rng) % 2;
    in.group = 1 + small(rng);
    in.strides = {1 + small(rng), 1 + small(rng)};
    in.pads = {small(rng), small(rng), small(rng), small(rng)};
    in.dilations = {1 + small(rng), 1 + small(rng)};
    in.kernelShape = {1 + small(rng), 1 + small(rng)};
    p.instructions.push_back(std::move(in));
  }
  return p;
}

bool sameInstruction(const Instruction &a, const Instruction &b) {
  return a.op == b.op && a.resultAddr == b.resultAddr &&
         a.resultShape == b.resultShape && a.operandAddrs == b.operandAddrs &&
         a.dramAddr == b.dramAddr && a.activation == b.activation &&
         a.group == b.group && a.strides == b.strides && a.pads == b.pads &&
         a.dilations == b.dilations && a.kernelShape == b.kernelShape;
}

bool sameRegion(const MemRegion &a, const MemRegion &b) {
  return a.dramOffset == b.dramOffset && a.shape == b.shape;
}

} // namespace

TEST(EncodingProperty, RoundTripIsLossless) {
  std::mt19937 rng(kSeed);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    Program original = randomProgram(rng);
    std::vector<uint8_t> bytes = original.encode();

    // decodeUnvalidated, because the generator produces structurally plausible
    // programs rather than semantically valid ones. What is under test is the
    // byte format.
    auto decoded = Program::decodeUnvalidated(bytes);
    ASSERT_TRUE(decoded.has_value()) << "iteration " << iteration;

    EXPECT_EQ(decoded->version, original.version) << "iteration " << iteration;
    EXPECT_EQ(decoded->scratchpadBytes, original.scratchpadBytes)
        << "iteration " << iteration;
    EXPECT_EQ(decoded->dramBytes, original.dramBytes)
        << "iteration " << iteration;

    ASSERT_EQ(decoded->inputs.size(), original.inputs.size())
        << "iteration " << iteration;
    for (size_t i = 0; i < original.inputs.size(); ++i)
      EXPECT_TRUE(sameRegion(decoded->inputs[i], original.inputs[i]))
          << "iteration " << iteration << " input " << i;

    ASSERT_EQ(decoded->outputs.size(), original.outputs.size())
        << "iteration " << iteration;
    for (size_t i = 0; i < original.outputs.size(); ++i)
      EXPECT_TRUE(sameRegion(decoded->outputs[i], original.outputs[i]))
          << "iteration " << iteration << " output " << i;

    ASSERT_EQ(decoded->constants.size(), original.constants.size())
        << "iteration " << iteration;
    ASSERT_EQ(decoded->constantData.size(), original.constantData.size())
        << "iteration " << iteration;
    for (size_t i = 0; i < original.constants.size(); ++i) {
      EXPECT_TRUE(sameRegion(decoded->constants[i], original.constants[i]))
          << "iteration " << iteration << " constant " << i;
      // fp32 survives the round trip exactly; it is copied, not converted.
      EXPECT_EQ(decoded->constantData[i], original.constantData[i])
          << "iteration " << iteration << " constant data " << i;
    }

    ASSERT_EQ(decoded->instructions.size(), original.instructions.size())
        << "iteration " << iteration;
    for (size_t i = 0; i < original.instructions.size(); ++i)
      EXPECT_TRUE(
          sameInstruction(decoded->instructions[i], original.instructions[i]))
          << "iteration " << iteration << " instruction " << i;
  }
}

TEST(EncodingProperty, ReEncodingProducesIdenticalBytes) {
  // Stronger than field equality: the format has no padding or ordering
  // freedom, so decode then encode has to reproduce the original stream.
  std::mt19937 rng(kSeed);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    Program original = randomProgram(rng);
    std::vector<uint8_t> bytes = original.encode();
    auto decoded = Program::decodeUnvalidated(bytes);
    ASSERT_TRUE(decoded.has_value()) << "iteration " << iteration;
    EXPECT_EQ(decoded->encode(), bytes) << "iteration " << iteration;
  }
}
