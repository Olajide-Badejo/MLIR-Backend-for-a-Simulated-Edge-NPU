//===- DifferentialExport.cpp - cases for the reference oracle *- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Writes the cases `test/Python/test_refexec_differential.py` compares the
// simulator against `python/npu_frontend/refexec.py` on.
//
// **What is and is not independent here matters, so it is stated.** The oracle
// is `refexec.py`, written from the ODS descriptions in numpy, and it never
// sees a line of the C++ kernels. This file is not part of the oracle: it is
// the harness that states an intention once, in the manifest, and hands the
// same intention to two implementations. The manifest names the `npu` operation
// and its attributes; the `.nbin` beside it is that operation encoded for this
// machine; the `.bin` files are the inputs both sides read. Agreement between
// the two answers is the test, and a manifest that described the wrong
// operation would make both sides wrong in the same way, which is why the
// manifests are small enough to read.
//
// It is guarded on an environment variable and skipped otherwise, in the same
// shape as the corpus export of Phase P6: the only caller is a test that knows
// where it wants the files, and a unit test binary that scattered files into
// the working directory of whoever ran it would be a nuisance.
//
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace nbin;
using namespace npusim;

namespace {

/// A deterministic stream of values in [-1, 1), so that a disagreement
/// reproduces from the case name alone.
///
/// **The shift is 32 and not 33, and the difference is D-0029.** A 33 bit shift
/// leaves 31 significant bits, which divided by 2^31 lands in [0, 1) and after
/// the subtraction in [-1, 0). Every input this file exported was negative, the
/// relu case compared all zeros against all zeros, and the two pooling cases
/// never saw a window a maximum could be positive in. `TheStreamSpansBothSigns`
/// below is the guard, because a generator that silently halves its range is
/// not something a reader of the exported bytes would notice.
class Stream {
public:
  explicit Stream(uint64_t seed) : state(seed) {}

  float next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t bits = static_cast<uint32_t>(state >> 32);
    return static_cast<float>(bits) / 2147483648.0f - 1.0f;
  }

  std::vector<float> values(int64_t count) {
    std::vector<float> out(static_cast<size_t>(count));
    for (float &value : out)
      value = next();
    return out;
  }

private:
  uint64_t state;
};

/// One operand of an exported case.
struct Input {
  std::vector<int64_t> shape;
  std::vector<float> data;
  /// How the compute instruction reads the buffer, when that differs from the
  /// buffer itself. This is the channel broadcast of ADR 0005.
  std::vector<int64_t> viewShape;
  std::vector<int64_t> viewStrides;
};

/// One case: an `npu` level intention, and the machine level program that is
/// supposed to carry it out.
struct Case {
  std::string name;
  /// The `npu` operation mnemonic, which is what `refexec.py` dispatches on.
  std::string operation;
  /// The attributes, already rendered as JSON object members.
  std::string attributes;
  std::vector<Input> inputs;
  std::vector<int64_t> resultShape;
  Opcode opcode;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> kernel;
  std::vector<int64_t> axes;
  int64_t group = 0;
};

std::string shapeJson(llvm::ArrayRef<int64_t> shape) {
  std::string out = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index)
      out += ", ";
    out += std::to_string(shape[index]);
  }
  return out + "]";
}

/// Builds the program for a case: one input region per operand, one load each,
/// the instruction, one store, and a HALT.
Program buildProgram(const Case &entry, uint64_t &scratchpadBytes) {
  Builder builder;
  std::vector<int64_t> regions;
  std::vector<int64_t> buffers;
  int64_t total = 0;
  for (const Input &operand : entry.inputs) {
    regions.push_back(builder.input(operand.shape));
    buffers.push_back(builder.scratch(elements(operand.shape)));
    total += elements(operand.shape);
  }
  const int64_t resultBuffer = builder.scratch(elements(entry.resultShape));
  total += elements(entry.resultShape);
  const int64_t sink = builder.output(entry.resultShape);

  for (size_t index = 0; index < entry.inputs.size(); ++index)
    builder.add(dmaLoad(buffers[index], entry.inputs[index].shape,
                        at(MemSpace::Dram, regions[index],
                           entry.inputs[index].shape)));

  std::vector<Operand> operands;
  for (size_t index = 0; index < entry.inputs.size(); ++index) {
    const Input &operand = entry.inputs[index];
    if (operand.viewShape.empty())
      operands.push_back(
          at(MemSpace::Scratchpad, buffers[index], operand.shape));
    else
      operands.push_back(strided(MemSpace::Scratchpad, buffers[index],
                                 operand.viewShape, operand.viewStrides));
  }

  Instruction instruction = compute(entry.opcode, resultBuffer,
                                    entry.resultShape, std::move(operands));
  instruction.pads = entry.pads;
  instruction.strides = entry.strides;
  instruction.dilations = entry.dilations;
  instruction.kernel = entry.kernel;
  instruction.axes = entry.axes;
  instruction.group = entry.group;
  builder.add(std::move(instruction));

  builder.add(dmaStore(sink, entry.resultShape,
                       at(MemSpace::Scratchpad, resultBuffer,
                          entry.resultShape)));
  builder.add(halt());

  // The scratchpad is the sum of the operand buffers and the result buffer,
  // four bytes an element, and nothing else. It is computed here rather than
  // written out per case because there are twenty of them; `Builder::finish`
  // still asserts that the number and the buffers agree, which is the property
  // Section 9.3 is after.
  scratchpadBytes = static_cast<uint64_t>(total * 4);
  return builder.finish(scratchpadBytes);
}

/// The convolution attribute set, as JSON.
std::string convAttributes(const Case &entry) {
  return "\"strides\": " + shapeJson(entry.strides) + ", \"pads\": " +
         shapeJson(entry.pads) + ", \"dilations\": " +
         shapeJson(entry.dilations) + ", \"group\": " +
         std::to_string(entry.group);
}

/// The pooling attribute set, as JSON.
std::string poolAttributes(const Case &entry) {
  return "\"kernel\": " + shapeJson(entry.kernel) + ", \"strides\": " +
         shapeJson(entry.strides) + ", \"pads\": " + shapeJson(entry.pads) +
         ", \"dilations\": " + shapeJson(entry.dilations);
}

/// Every case the differential test runs.
///
/// The set is the P7 coverage list again, at randomized inputs this time. The
/// hand computed tests next door prove the kernels against arithmetic a reader
/// can redo; these prove them against a second implementation over values
/// nobody chose.
std::vector<Case> cases() {
  Stream stream(0x6e7075503744494ull);
  std::vector<Case> all;

  auto conv = [&](std::string name, std::vector<int64_t> inputShape,
                  std::vector<int64_t> filterShape,
                  std::vector<int64_t> resultShape, std::vector<int64_t> strides,
                  std::vector<int64_t> pads, std::vector<int64_t> dilations,
                  int64_t group, bool bias) {
    Case entry;
    entry.name = std::move(name);
    entry.operation = "conv2d";
    entry.opcode = Opcode::CONV2D;
    entry.inputs.push_back({inputShape, stream.values(elements(inputShape)),
                            {}, {}});
    entry.inputs.push_back({filterShape, stream.values(elements(filterShape)),
                            {}, {}});
    if (bias) {
      const std::vector<int64_t> biasShape = {resultShape[1]};
      entry.inputs.push_back({biasShape, stream.values(resultShape[1]), {}, {}});
    }
    entry.resultShape = resultShape;
    entry.strides = std::move(strides);
    entry.pads = std::move(pads);
    entry.dilations = std::move(dilations);
    entry.group = group;
    entry.attributes = convAttributes(entry);
    all.push_back(std::move(entry));
  };

  conv("conv2d_dense", {2, 3, 6, 6}, {4, 3, 3, 3}, {2, 4, 6, 6}, {1, 1},
       {1, 1, 1, 1}, {1, 1}, 1, false);
  conv("conv2d_dense_bias", {1, 3, 5, 5}, {4, 3, 3, 3}, {1, 4, 3, 3}, {1, 1},
       {0, 0, 0, 0}, {1, 1}, 1, true);
  conv("conv2d_grouped", {1, 4, 5, 5}, {6, 2, 3, 3}, {1, 6, 5, 5}, {1, 1},
       {1, 1, 1, 1}, {1, 1}, 2, false);
  conv("conv2d_depthwise", {2, 6, 5, 5}, {6, 1, 3, 3}, {2, 6, 5, 5}, {1, 1},
       {1, 1, 1, 1}, {1, 1}, 6, true);
  conv("conv2d_dilated", {1, 2, 7, 7}, {3, 2, 3, 3}, {1, 3, 3, 3}, {1, 1},
       {0, 0, 0, 0}, {2, 2}, 1, false);
  conv("conv2d_asymmetric_padding", {1, 2, 5, 5}, {3, 2, 3, 3}, {1, 3, 5, 4},
       {1, 1}, {2, 0, 0, 1}, {1, 1}, 1, false);
  conv("conv2d_strided", {1, 2, 7, 7}, {3, 2, 3, 3}, {1, 3, 3, 3}, {2, 2},
       {0, 0, 0, 0}, {1, 1}, 1, true);
  conv("conv2d_batch_four", {4, 3, 4, 4}, {5, 3, 3, 3}, {4, 5, 4, 4}, {1, 1},
       {1, 1, 1, 1}, {1, 1}, 1, true);

  auto matmul = [&](std::string name, int64_t m, int64_t k, int64_t n,
                    bool bias) {
    Case entry;
    entry.name = std::move(name);
    entry.operation = "matmul";
    entry.opcode = Opcode::MATMUL;
    entry.inputs.push_back({{m, k}, stream.values(m * k), {}, {}});
    entry.inputs.push_back({{k, n}, stream.values(k * n), {}, {}});
    if (bias)
      entry.inputs.push_back({{n}, stream.values(n), {}, {}});
    entry.resultShape = {m, n};
    entry.attributes = "";
    all.push_back(std::move(entry));
  };

  matmul("matmul_square", 8, 8, 8, false);
  // A narrow tile, which is the shape Section 5.5's utilization term exists for
  // and which the arithmetic has to get right whatever the cost model says.
  matmul("matmul_narrow_bias", 5, 19, 3, true);

  auto elementwise = [&](std::string name, std::string operation,
                         Opcode opcode, bool broadcast) {
    Case entry;
    entry.name = std::move(name);
    entry.operation = std::move(operation);
    entry.opcode = opcode;
    const std::vector<int64_t> shape = {2, 3, 4, 4};
    entry.inputs.push_back({shape, stream.values(elements(shape)), {}, {}});
    if (broadcast)
      entry.inputs.push_back({{3}, stream.values(3), shape, {0, 1, 0, 0}});
    else
      entry.inputs.push_back({shape, stream.values(elements(shape)), {}, {}});
    entry.resultShape = shape;
    entry.attributes = "";
    all.push_back(std::move(entry));
  };

  elementwise("add_same_shape", "add", Opcode::ADD, false);
  elementwise("add_channel_broadcast", "add", Opcode::ADD, true);
  elementwise("mul_same_shape", "mul", Opcode::MUL, false);
  elementwise("mul_channel_broadcast", "mul", Opcode::MUL, true);

  {
    Case entry;
    entry.name = "relu";
    entry.operation = "relu";
    entry.opcode = Opcode::RELU;
    const std::vector<int64_t> shape = {4, 2, 3, 3};
    entry.inputs.push_back({shape, stream.values(elements(shape)), {}, {}});
    entry.resultShape = shape;
    entry.attributes = "";
    all.push_back(std::move(entry));
  }

  auto pool = [&](std::string name, std::string operation, Opcode opcode,
                  std::vector<int64_t> inputShape,
                  std::vector<int64_t> resultShape, std::vector<int64_t> kernel,
                  std::vector<int64_t> strides, std::vector<int64_t> pads) {
    Case entry;
    entry.name = std::move(name);
    entry.operation = std::move(operation);
    entry.opcode = opcode;
    entry.inputs.push_back({inputShape, stream.values(elements(inputShape)),
                            {}, {}});
    entry.resultShape = std::move(resultShape);
    entry.kernel = std::move(kernel);
    entry.strides = std::move(strides);
    entry.pads = std::move(pads);
    entry.dilations = {1, 1};
    entry.attributes = poolAttributes(entry);
    all.push_back(std::move(entry));
  };

  pool("max_pool2d", "max_pool2d", Opcode::POOL_MAX, {2, 3, 6, 6}, {2, 3, 3, 3},
       {2, 2}, {2, 2}, {0, 0, 0, 0});
  pool("max_pool2d_padded", "max_pool2d", Opcode::POOL_MAX, {1, 2, 5, 5},
       {1, 2, 5, 5}, {3, 3}, {1, 1}, {1, 1, 1, 1});
  pool("avg_pool2d", "avg_pool2d", Opcode::POOL_AVG, {4, 2, 4, 4},
       {4, 2, 2, 2}, {2, 2}, {2, 2}, {0, 0, 0, 0});
  // The padded average is where count_include_pad = 0 shows: the divisor is the
  // number of elements that actually contributed, and the border windows have
  // fewer.
  pool("avg_pool2d_padded", "avg_pool2d", Opcode::POOL_AVG, {1, 2, 5, 5},
       {1, 2, 5, 5}, {3, 3}, {1, 1}, {1, 1, 1, 1});

  {
    Case entry;
    entry.name = "reshape";
    entry.operation = "reshape";
    entry.opcode = Opcode::RESHAPE;
    entry.inputs.push_back({{2, 3, 4}, stream.values(24), {}, {}});
    entry.resultShape = {4, 6};
    // The target extents are an attribute of the `npu` operation even though
    // the machine reads them off the result, because `refexec.reshape` takes
    // the shape the way the dialect does.
    entry.attributes = "\"shape\": [4, 6]";
    all.push_back(std::move(entry));
  }

  {
    Case entry;
    entry.name = "transpose_nchw_to_nhwc";
    entry.operation = "transpose";
    entry.opcode = Opcode::TRANSPOSE;
    entry.inputs.push_back({{2, 3, 4, 5}, stream.values(120), {}, {}});
    entry.resultShape = {2, 4, 5, 3};
    entry.axes = {0, 2, 3, 1};
    entry.attributes = "\"permutation\": [0, 2, 3, 1]";
    all.push_back(std::move(entry));
  }

  {
    Case entry;
    entry.name = "transpose_identity";
    entry.operation = "transpose";
    entry.opcode = Opcode::TRANSPOSE;
    entry.inputs.push_back({{3, 5}, stream.values(15), {}, {}});
    entry.resultShape = {3, 5};
    entry.axes = {0, 1};
    entry.attributes = "\"permutation\": [0, 1]";
    all.push_back(std::move(entry));
  }

  {
    Case entry;
    entry.name = "concat_channel_axis";
    entry.operation = "concat";
    entry.opcode = Opcode::CONCAT;
    entry.inputs.push_back({{4, 2, 3, 3}, stream.values(72), {}, {}});
    entry.inputs.push_back({{4, 3, 3, 3}, stream.values(108), {}, {}});
    entry.resultShape = {4, 5, 3, 3};
    entry.axes = {1};
    entry.attributes = "\"axis\": 1";
    all.push_back(std::move(entry));
  }

  {
    Case entry;
    entry.name = "concat_last_axis_three_operands";
    entry.operation = "concat";
    entry.opcode = Opcode::CONCAT;
    entry.inputs.push_back({{3, 2}, stream.values(6), {}, {}});
    entry.inputs.push_back({{3, 1}, stream.values(3), {}, {}});
    entry.inputs.push_back({{3, 4}, stream.values(12), {}, {}});
    entry.resultShape = {3, 7};
    entry.axes = {1};
    entry.attributes = "\"axis\": 1";
    all.push_back(std::move(entry));
  }

  return all;
}

/// Writes `bytes` to `path`, failing the test rather than the run.
void writeFile(const std::string &path, const void *data, size_t size) {
  FILE *file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr) << path;
  if (size)
    std::fwrite(data, 1, size, file);
  std::fclose(file);
}

TEST(Differential, TheCasesCanBeWrittenOutForTheReferenceInterpreter) {
  const char *directory = std::getenv("NPU_DIFFERENTIAL_OUT");
  if (!directory) {
    std::cout << "[          ] NPU_DIFFERENTIAL_OUT is not set, so the "
                 "differential cases were not written out. Set it to a "
                 "directory to export them.\n";
    GTEST_SKIP();
  }

  const std::vector<Case> all = cases();
  std::string manifest = "{\n  \"cases\": [\n";

  for (size_t index = 0; index < all.size(); ++index) {
    const Case &entry = all[index];
    SCOPED_TRACE(entry.name);

    uint64_t scratchpadBytes = 0;
    const Program program = buildProgram(entry, scratchpadBytes);

    // The program is validated here as well as in the simulator, so that a case
    // this file got wrong fails in this binary with the case name attached
    // rather than in a pytest three steps downstream.
    const std::optional<ProgramError> failure = program.validate();
    ASSERT_FALSE(failure.has_value())
        << entry.name << ": " << (failure ? failure->toString() : "");

    const std::string base = std::string(directory) + "/" + entry.name;
    const std::vector<uint8_t> encoded = program.encode();
    writeFile(base + ".nbin", encoded.data(), encoded.size());

    manifest += "    {\n";
    manifest += "      \"name\": \"" + entry.name + "\",\n";
    manifest += "      \"operation\": \"" + entry.operation + "\",\n";
    manifest += "      \"program\": \"" + entry.name + ".nbin\",\n";
    manifest += "      \"attributes\": {" + entry.attributes + "},\n";
    manifest += "      \"inputs\": [\n";
    for (size_t operand = 0; operand < entry.inputs.size(); ++operand) {
      const Input &input = entry.inputs[operand];
      const std::string name =
          entry.name + ".in" + std::to_string(operand) + ".bin";
      writeFile(std::string(directory) + "/" + name, input.data.data(),
                input.data.size() * sizeof(float));
      manifest += "        {\"file\": \"" + name + "\", \"shape\": " +
                  shapeJson(input.shape) + "}";
      manifest += operand + 1 < entry.inputs.size() ? ",\n" : "\n";
    }
    manifest += "      ],\n";
    manifest +=
        "      \"result_shape\": " + shapeJson(entry.resultShape) + "\n";
    manifest += index + 1 < all.size() ? "    },\n" : "    }\n";
  }
  manifest += "  ]\n}\n";

  writeFile(std::string(directory) + "/manifest.json", manifest.data(),
            manifest.size());
  std::cout << "[          ] wrote " << all.size() << " differential cases to "
            << directory << "\n";
}

TEST(Differential, TheStreamSpansBothSigns) {
  // D-0029's guard. The generator above shifts by 32 to keep all thirty two
  // bits, and one bit more would halve the range to [-1, 0) without changing
  // anything a reader of the exported files could see: the values would still
  // look random, still be deterministic, and still reproduce. What they would
  // not do is exercise a relu, or a maximum whose answer is positive, and the
  // differential suite would pass while comparing zeros against zeros.
  Stream stream(0x6e7075503744494ull);
  float low = 1.0f;
  float high = -1.0f;
  for (int index = 0; index < 100000; ++index) {
    const float value = stream.next();
    ASSERT_GE(value, -1.0f);
    ASSERT_LT(value, 1.0f);
    low = std::min(low, value);
    high = std::max(high, value);
  }
  // Both ends, not merely both signs: a generator that produced [-1, 0.01)
  // would satisfy "some value is positive" and still be the same bug.
  EXPECT_LT(low, -0.99f) << "the stream's lowest value in 100000 draws";
  EXPECT_GT(high, 0.99f) << "the stream's highest value in 100000 draws";
}

TEST(Differential, EveryExportedCaseRunsCleanly) {
  // The export is guarded on an environment variable, so without this the whole
  // case set would go unexercised in a normal run and a case that no longer
  // validates would be found only by whoever next ran the pytest. Here every
  // case is built and run in this binary, on every run, with no files written.
  const std::vector<Case> all = cases();
  ASSERT_FALSE(all.empty());

  for (const Case &entry : all) {
    SCOPED_TRACE(entry.name);
    uint64_t scratchpadBytes = 0;
    Harness harness(buildProgram(entry, scratchpadBytes));

    for (size_t operand = 0; operand < entry.inputs.size(); ++operand) {
      const std::vector<float> &data = entry.inputs[operand].data;
      std::string failure;
      ASSERT_TRUE(harness.sim().loadInput(
          operand,
          llvm::ArrayRef<uint8_t>(
              reinterpret_cast<const uint8_t *>(data.data()),
              data.size() * sizeof(float)),
          failure))
          << failure;
    }

    const SimResult result = harness.run();
    ASSERT_TRUE(result.ok()) << result.error.value_or("");
    EXPECT_TRUE(result.reachedHalt);
    EXPECT_EQ(harness.outputF32(0).size(),
              static_cast<size_t>(elements(entry.resultShape)));
  }
}

} // namespace
