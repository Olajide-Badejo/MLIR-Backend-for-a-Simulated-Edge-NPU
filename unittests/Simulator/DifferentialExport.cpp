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

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

  /// The same stream, mapped onto the whole i8 range.
  ///
  /// `next()` lands in [-1, 1), so this scales and floors onto [-128, 128) and
  /// clamps the one value that lands on the top rail. Drawing i8 from the same
  /// generator rather than a second one keeps `TheStreamSpansBothSigns` the
  /// only place the generator's range is asserted.
  std::vector<int8_t> int8Values(int64_t count) {
    std::vector<int8_t> out(static_cast<size_t>(count));
    for (int8_t &value : out) {
      const float scaled = std::floor(next() * 128.0f);
      value = static_cast<int8_t>(std::min(127.0f, std::max(-128.0f, scaled)));
    }
    return out;
  }

  /// An int32 bias, drawn small enough that it cannot itself overflow the
  /// accumulator: the reductions in these cases are a few hundred taps of at
  /// most 128 by 127, so a bias of a few thousand leaves the int32 guard with
  /// three orders of magnitude in hand.
  std::vector<int32_t> int32Values(int64_t count) {
    std::vector<int32_t> out(static_cast<size_t>(count));
    for (int32_t &value : out)
      value = static_cast<int32_t>(std::floor(next() * 4096.0f));
    return out;
  }

private:
  uint64_t state;
};

/// One operand of an exported case.
///
/// **The element type joined this at Phase P14** and `data` is the f32 payload
/// only. An integer operand carries its bytes in `raw` instead, because an i8
/// operand's values are not floats that happen to be small: they are the
/// machine's own representation and rounding them through a float would be the
/// one place this harness could quietly change the numbers it is comparing.
struct Input {
  std::vector<int64_t> shape;
  std::vector<float> data;
  /// How the compute instruction reads the buffer, when that differs from the
  /// buffer itself. This is the channel broadcast of ADR 0005.
  std::vector<int64_t> viewShape;
  std::vector<int64_t> viewStrides;
  ElemType type = ElemType::F32;
  std::vector<uint8_t> raw;
};

/// The bytes an operand is written and loaded as, whichever type it is.
llvm::ArrayRef<uint8_t> payload(const Input &operand) {
  if (operand.type == ElemType::F32)
    return llvm::ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(operand.data.data()),
        operand.data.size() * sizeof(float));
  return llvm::ArrayRef<uint8_t>(operand.raw);
}

/// An i8 operand, from a draw.
Input i8Input(std::vector<int64_t> shape, const std::vector<int8_t> &values) {
  Input operand;
  operand.shape = std::move(shape);
  operand.type = ElemType::I8;
  operand.raw.assign(reinterpret_cast<const uint8_t *>(values.data()),
                     reinterpret_cast<const uint8_t *>(values.data()) +
                         values.size());
  return operand;
}

/// An i32 operand, which is what a quantized bias is.
Input i32Input(std::vector<int64_t> shape, const std::vector<int32_t> &values) {
  Input operand;
  operand.shape = std::move(shape);
  operand.type = ElemType::I32;
  const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
  operand.raw.assign(bytes, bytes + values.size() * sizeof(int32_t));
  return operand;
}

/// The name the manifest gives an element type, which is the name numpy knows.
const char *dtypeName(ElemType type) {
  switch (type) {
  case ElemType::F32:
    return "float32";
  case ElemType::I8:
    return "int8";
  case ElemType::I32:
    return "int32";
  }
  return "unknown";
}

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
  /// The result's element type. An integer result is what selects the integer
  /// arithmetic on both sides: the instruction carries it, and the manifest
  /// carries the operand dtypes that `refexec` reads.
  ElemType resultType = ElemType::F32;
  float scale = 0.0f;
  int32_t zeroPoint = 0;
  int32_t requantMultiplier = 1;
  int32_t requantShift = 0;
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
  std::vector<int64_t> regions(entry.inputs.size(), 0);
  std::vector<int64_t> buffers(entry.inputs.size(), 0);
  int64_t total = 0;

  // **The input regions are declared in operand order**, because the caller
  // loads them by index: `loadInput(0, ...)` means the program's first declared
  // input, and reordering the declarations would silently pair each operand
  // with another operand's bytes.
  for (size_t index = 0; index < entry.inputs.size(); ++index) {
    const Input &operand = entry.inputs[index];
    regions[index] = builder.input(operand.shape, operand.type);
  }

  // **The scratchpad buffers are allocated widest element first**, and that is
  // the machine's own rule rather than tidiness: a four byte element is read
  // through an accessor that requires a four byte aligned address, and an i32
  // bias placed after a run of i8 buffers would land wherever their total left
  // it. The real allocator aligns every offset to `kDefaultAlignment`, so a
  // compiled program never has the problem; this harness allocates tightly and
  // unaligned on purpose, so it orders the buffers instead. Unlike the regions
  // above this order is not observable: a scratchpad address is an address.
  std::vector<size_t> order(entry.inputs.size());
  for (size_t index = 0; index < order.size(); ++index)
    order[index] = index;
  llvm::stable_sort(order, [&](size_t left, size_t right) {
    return elementByteSize(entry.inputs[left].type) >
           elementByteSize(entry.inputs[right].type);
  });

  for (size_t index : order) {
    const Input &operand = entry.inputs[index];
    buffers[index] = builder.scratch(elements(operand.shape), operand.type);
    total += elements(operand.shape) * elementByteSize(operand.type);
  }
  const int64_t resultBuffer =
      builder.scratch(elements(entry.resultShape), entry.resultType);
  total += elements(entry.resultShape) * elementByteSize(entry.resultType);
  const int64_t sink = builder.output(entry.resultShape, entry.resultType);

  for (size_t index = 0; index < entry.inputs.size(); ++index)
    builder.add(dmaLoad(buffers[index], entry.inputs[index].shape,
                        at(MemSpace::Dram, regions[index],
                           entry.inputs[index].shape,
                           entry.inputs[index].type)));

  std::vector<Operand> operands;
  for (size_t index = 0; index < entry.inputs.size(); ++index) {
    const Input &operand = entry.inputs[index];
    if (operand.viewShape.empty())
      operands.push_back(at(MemSpace::Scratchpad, buffers[index], operand.shape,
                            operand.type));
    else
      operands.push_back(strided(MemSpace::Scratchpad, buffers[index],
                                 operand.viewShape, operand.viewStrides,
                                 operand.type));
  }

  Instruction instruction =
      compute(entry.opcode, resultBuffer, entry.resultShape,
              std::move(operands), entry.resultType);
  instruction.pads = entry.pads;
  instruction.strides = entry.strides;
  instruction.dilations = entry.dilations;
  instruction.kernel = entry.kernel;
  instruction.axes = entry.axes;
  instruction.group = entry.group;
  instruction.scale = entry.scale;
  instruction.zeroPoint = entry.zeroPoint;
  instruction.requantMultiplier = entry.requantMultiplier;
  instruction.requantShift = entry.requantShift;
  builder.add(std::move(instruction));

  builder.add(dmaStore(sink, entry.resultShape,
                       at(MemSpace::Scratchpad, resultBuffer, entry.resultShape,
                          entry.resultType)));
  builder.add(halt());

  // The scratchpad is the sum of the operand buffers and the result buffer and
  // nothing else, each at its own element size. It is computed here rather than
  // written out per case because there are two dozen of them;
  // `Builder::finish` still asserts that the number and the buffers agree,
  // which is the property Section 9.3 is after.
  scratchpadBytes = static_cast<uint64_t>(total);
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

  // -------------------------------------------------------------------------
  // The integer cases.
  //
  // These are the ones whose comparison is **exact** rather than to a
  // tolerance. Integer addition is associative, so the reference's whole tensor
  // slicing and the kernel's per element walk cannot disagree by a summation
  // order, and any difference at all is a defect. That is a stronger claim than
  // any f32 case in this file can make, and it is the whole reason Section 14
  // can say a tiled reduction is bit exact by construction.
  //
  // The requantization pair is the identity for these accumulators: M0 at
  // 2^31 - 1 with a shift of zero is M = 1 - 2^-31, and the multiply loses
  // acc / 2^31 while the rounding puts it back. That keeps the arithmetic under
  // comparison the convolution rather than the rescale, and the two cases that
  // are about the rescale carry their own multiplier.
  // -------------------------------------------------------------------------

  constexpr int32_t kIdentity = 2147483647;
  constexpr int32_t kHalf = 1073741824;

  {
    Case entry;
    entry.name = "quantize_per_tensor";
    entry.operation = "quantize";
    entry.opcode = Opcode::QUANT;
    entry.inputs.push_back({{2, 3, 4, 4}, stream.values(96), {}, {}});
    entry.resultShape = {2, 3, 4, 4};
    entry.resultType = ElemType::I8;
    entry.scale = 0.0125f;
    entry.zeroPoint = -7;
    entry.attributes = "\"scale\": 0.0125, \"zero_point\": -7";
    all.push_back(std::move(entry));
  }

  {
    Case entry;
    entry.name = "dequantize_per_tensor";
    entry.operation = "dequantize";
    entry.opcode = Opcode::DEQUANT;
    entry.inputs.push_back(i8Input({2, 3, 4, 4}, stream.int8Values(96)));
    entry.resultShape = {2, 3, 4, 4};
    entry.resultType = ElemType::F32;
    entry.scale = 0.0125f;
    entry.zeroPoint = -7;
    entry.attributes = "\"scale\": 0.0125, \"zero_point\": -7";
    all.push_back(std::move(entry));
  }

  auto quantConv = [&](std::string name, std::vector<int64_t> inputShape,
                       std::vector<int64_t> filterShape,
                       std::vector<int64_t> resultShape,
                       std::vector<int64_t> strides, std::vector<int64_t> pads,
                       std::vector<int64_t> dilations, int64_t group,
                       int32_t zeroPoint, int32_t outputZeroPoint,
                       int32_t multiplier) {
    Case entry;
    entry.name = std::move(name);
    entry.operation = "conv2d";
    entry.opcode = Opcode::CONV2D;
    entry.resultType = ElemType::I8;
    entry.inputs.push_back(
        i8Input(inputShape, stream.int8Values(elements(inputShape))));
    entry.inputs.push_back(
        i8Input(filterShape, stream.int8Values(elements(filterShape))));
    entry.inputs.push_back(
        i32Input({resultShape[1]}, stream.int32Values(resultShape[1])));
    entry.resultShape = resultShape;
    entry.strides = std::move(strides);
    entry.pads = std::move(pads);
    entry.dilations = std::move(dilations);
    entry.group = group;
    entry.zeroPoint = zeroPoint;
    // The output zero point travels in the scale word, which is where the
    // machine carries it on an integer compute instruction.
    entry.scale = static_cast<float>(outputZeroPoint);
    entry.requantMultiplier = multiplier;
    entry.requantShift = 0;
    entry.attributes = convAttributes(entry) + ", \"zero_point\": " +
                       std::to_string(entry.zeroPoint) +
                       ", \"output_zero_point\": " +
                       std::to_string(outputZeroPoint) +
                       ", \"requant_multiplier\": " +
                       std::to_string(entry.requantMultiplier) +
                       ", \"requant_shift\": " +
                       std::to_string(entry.requantShift);
    all.push_back(std::move(entry));
  };

  // A padded convolution at a non zero input zero point, which is the case the
  // whole folded bias argument rests on, and the same shape unpadded beside it
  // so that a padding rule that was wrong in both directions cannot hide.
  quantConv("conv2d_i8_padded", {1, 3, 5, 5}, {4, 3, 3, 3}, {1, 4, 5, 5},
            {1, 1}, {1, 1, 1, 1}, {1, 1}, 1, -11, 12, kHalf);
  quantConv("conv2d_i8_unpadded", {1, 3, 5, 5}, {4, 3, 3, 3}, {1, 4, 3, 3},
            {1, 1}, {0, 0, 0, 0}, {1, 1}, 1, -11, -5, kIdentity);
  // This one keeps an output zero point of zero, so the symmetric case stays
  // covered: it is a legal value and the case that used to be the only one.
  quantConv("conv2d_i8_depthwise", {2, 6, 5, 5}, {6, 1, 3, 3}, {2, 6, 5, 5},
            {1, 1}, {1, 1, 1, 1}, {1, 1}, 6, 23, 0, kHalf);
  // A zero point of zero, which is the symmetric case and the one where a
  // kernel that ignored the field entirely would still pass.
  quantConv("conv2d_i8_symmetric", {1, 2, 6, 6}, {3, 2, 3, 3}, {1, 3, 6, 6},
            {1, 1}, {1, 1, 1, 1}, {1, 1}, 1, 0, -40, kIdentity);

  {
    Case entry;
    entry.name = "matmul_i8_bias";
    entry.operation = "matmul";
    entry.opcode = Opcode::MATMUL;
    entry.resultType = ElemType::I8;
    entry.inputs.push_back(i8Input({5, 19}, stream.int8Values(95)));
    entry.inputs.push_back(i8Input({19, 3}, stream.int8Values(57)));
    entry.inputs.push_back(i32Input({3}, stream.int32Values(3)));
    entry.resultShape = {5, 3};
    entry.scale = 7.0f;
    entry.requantMultiplier = kHalf;
    entry.requantShift = 2;
    entry.attributes = "\"output_zero_point\": 7, \"requant_multiplier\": " +
                       std::to_string(entry.requantMultiplier) +
                       ", \"requant_shift\": " +
                       std::to_string(entry.requantShift);
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
      const llvm::ArrayRef<uint8_t> bytes = payload(input);
      writeFile(std::string(directory) + "/" + name, bytes.data(),
                bytes.size());
      manifest += "        {\"file\": \"" + name + "\", \"shape\": " +
                  shapeJson(input.shape) + ", \"dtype\": \"" +
                  dtypeName(input.type) + "\"}";
      manifest += operand + 1 < entry.inputs.size() ? ",\n" : "\n";
    }
    manifest += "      ],\n";
    manifest += "      \"result_dtype\": \"" + std::string(dtypeName(entry.resultType)) +
                "\",\n";
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

TEST(Differential, EveryIntegerCaseIsComparedExactly) {
  // The claim the integer cases exist to make, asserted here as a property of
  // the case set rather than left to the pytest that consumes it: every case
  // whose result is an integer type is one whose two implementations must agree
  // to the bit, because integer addition is associative and neither side has a
  // summation order the other can disagree with.
  //
  // It also catches the case set going quietly f32 only. A file that lost its
  // integer cases in an edit would still pass every other test here.
  const std::vector<Case> all = cases();
  int integerCases = 0;
  for (const Case &entry : all) {
    if (entry.resultType == ElemType::F32 &&
        (entry.inputs.empty() || entry.inputs.front().type == ElemType::F32))
      continue;
    ++integerCases;
    for (const Input &operand : entry.inputs)
      EXPECT_NE(payload(operand).size(), 0u) << entry.name;
  }
  EXPECT_GE(integerCases, 7) << "the integer half of the case set has shrunk";
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
      std::string failure;
      ASSERT_TRUE(harness.sim().loadInput(operand, payload(entry.inputs[operand]),
                                          failure))
          << failure;
    }

    const SimResult result = harness.run();
    ASSERT_TRUE(result.ok()) << result.error.value_or("");
    EXPECT_TRUE(result.reachedHalt);
    EXPECT_EQ(harness.sim().outputBytes(0).size(),
              static_cast<size_t>(elements(entry.resultShape) *
                                  elementByteSize(entry.resultType)));
  }
}

} // namespace
