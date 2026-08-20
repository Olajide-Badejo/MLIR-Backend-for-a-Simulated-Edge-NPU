//===- PropertyTest.cpp - randomized round trips --------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 17.2. Randomized `Program` structures with bounded sizes, valid
// opcodes, plausible shapes, every element type, every opcode including the
// quantization pair, and sometimes a debug section; encoded, decoded, and
// asserted structurally equal, with re-encoding byte identical. A thousand
// iterations at a fixed seed, so a failure reproduces from the case number in
// the message.
//
// **The generated programs are valid, not merely well formed.** That costs the
// generator a bump allocator and a record of what it has written, and it buys
// two things. The round trip goes through `Program::decode`, which validates,
// so a thousand randomized programs exercise the validator's accepting path as
// well as the encoder; and a generator that produced invalid programs would
// have to use `decodeUnvalidated`, which would quietly stop testing the half of
// the decoder that everything except `npu-objdump` uses.
//
// **Covering the quantization opcodes here is structural, and at this phase
// that is all it can be.** The generator builds them, encodes them, decodes
// them, and asserts round trip equality including `requantMultiplier` and
// `requantShift`. It asserts nothing about what they compute, because no
// integer kernel exists until P14.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace nbin;
using namespace npu_test;

namespace {

/// The seed. A constant in the source, so a red run reproduces exactly from the
/// case number the failure message carries.
constexpr uint64_t kSeed = 0x6E62696E5031365Full;

/// How many programs to build. Section 17.2 asks for at least a thousand.
constexpr int kIterations = 1000;

/// Scratchpad offsets are aligned the way the allocator aligns them, so the
/// addresses a generated program carries look like addresses a real one would.
constexpr int64_t kAlignment = 64;

int64_t alignUp(int64_t value) {
  int64_t remainder = value % kAlignment;
  return remainder == 0 ? value : value + (kAlignment - remainder);
}

/// Builds one random program, and remembers what it has written so that every
/// operand it emits reads a buffer that exists.
class Generator {
public:
  explicit Generator(uint64_t seed) : rng(seed) {}

  /// What the generator covered, so the suite can assert it covered
  /// everything.
  struct Coverage {
    std::set<uint32_t> opcodes;
    std::set<uint32_t> elementTypes;
    int programsWithDebug = 0;
  };

  Program build(Coverage &coverage);

private:
  int64_t pick(int64_t low, int64_t high) {
    return std::uniform_int_distribution<int64_t>(low, high)(rng);
  }
  bool coin(int percent = 50) { return pick(1, 100) <= percent; }

  ElemType pickElemType(bool movementOpcode) {
    if (!movementOpcode)
      return ElemType::F32;
    static const ElemType all[] = {ElemType::F32, ElemType::I8, ElemType::I32};
    return all[pick(0, 2)];
  }

  std::vector<int64_t> pickShape(size_t rank) {
    std::vector<int64_t> shape(rank);
    for (int64_t &extent : shape)
      extent = pick(1, 4);
    return shape;
  }

  /// Reserves a scratchpad buffer and returns its address.
  int64_t reserveScratchpad(int64_t bytes) {
    int64_t address = alignUp(scratchpadCursor);
    scratchpadCursor = address + bytes;
    return address;
  }

  int64_t reserveDram(int64_t bytes) {
    int64_t address = alignUp(dramCursor);
    dramCursor = address + bytes;
    return address;
  }

  /// Puts a buffer of `shape` and `type` into the scratchpad, by declaring a
  /// constant that holds it and emitting the `DMA_LOAD` that brings it on
  /// chip. Returns the scratchpad address.
  ///
  /// This is how the generator satisfies `operand-defined` without having to
  /// reason about what the program has already computed: every operand it
  /// hands out was written by an instruction it emitted a moment earlier.
  int64_t materialise(const std::vector<int64_t> &shape, ElemType type);

  Operand readBuffer(int64_t address, const std::vector<int64_t> &shape,
                     ElemType type) {
    return operand(MemSpace::Scratchpad, type, address, shape);
  }

  /// Fills the fields an opcode gives meaning to, reading the generated table
  /// rather than a second copy of the rules.
  void fillGeneratedFields(Instruction &item);

  Instruction buildFor(Opcode opcode);

  std::mt19937_64 rng;
  Program program;
  int64_t scratchpadCursor = 0;
  int64_t dramCursor = 0;
};

int64_t Generator::materialise(const std::vector<int64_t> &shape,
                               ElemType type) {
  int64_t elements = checkedElementCount(shape);
  int64_t bytes = elements * elementByteSize(type);

  Constant constant;
  constant.region.offset = static_cast<uint64_t>(reserveDram(bytes));
  constant.region.elementType = type;
  constant.region.shape = shape;
  constant.data.assign(static_cast<size_t>(bytes), 0);
  program.constants.push_back(constant);

  int64_t address = reserveScratchpad(bytes);
  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad, type,
                                 address, shape);
  load.operands.push_back(operand(
      MemSpace::Dram, type, static_cast<int64_t>(constant.region.offset),
      shape));
  program.instructions.push_back(load);
  return address;
}

void Generator::fillGeneratedFields(Instruction &item) {
  const OpcodeInfo &info = opcodeInfo(item.opcode);
  if (info.fieldMask & kFieldActivation)
    item.activation = coin() ? Activation::Relu : Activation::None;
  if (info.fieldMask & kFieldRequantize) {
    item.requantMultiplier = static_cast<int32_t>(pick(1, 2147483647));
    item.requantShift = static_cast<int32_t>(pick(0, 31));
  }
  if (info.fieldMask & kFieldPads)
    item.pads = {pick(0, 2), pick(0, 2), pick(0, 2), pick(0, 2)};
  if (info.fieldMask & kFieldStrides)
    item.strides = {pick(1, 3), pick(1, 3)};
  if (info.fieldMask & kFieldDilations)
    item.dilations = {pick(1, 3), pick(1, 3)};
  if (info.fieldMask & kFieldKernel)
    item.kernel = {pick(1, 3), pick(1, 3)};
  if (info.fieldMask & kFieldGroup)
    item.group = 1;
  if (info.fieldMask & kFieldScale) {
    // A finite positive scale, spread across a few orders of magnitude so the
    // float round trip is exercised on more than one exponent.
    item.scale = std::ldexp(1.0f, static_cast<int>(pick(-20, 4)));
    item.zeroPoint = static_cast<int32_t>(pick(-128, 127));
  }
}

/// One instruction of the given opcode, with every operand already written.
///
/// **This switch has no `default` label, on purpose.** It is one of the two
/// hand written switches over the generated `Opcode` enum that Section 9.4's
/// claim rests on: adding an opcode to the ISA description and nothing else
/// stops this file compiling under `-Werror=switch` until somebody has said how
/// to build one. A property test that silently skipped a new opcode would
/// report full coverage of an instruction set it had never seen.
Instruction Generator::buildFor(Opcode opcode) {
  switch (opcode) {
  case Opcode::NOP:
  case Opcode::HALT: {
    Instruction item;
    item.opcode = opcode;
    return item;
  }

  case Opcode::DMA_LOAD: {
    ElemType type = pickElemType(true);
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 3)));
    int64_t bytes =
        checkedElementCount(shape) * elementByteSize(type);
    Constant constant;
    constant.region.offset = static_cast<uint64_t>(reserveDram(bytes));
    constant.region.elementType = type;
    constant.region.shape = shape;
    constant.data.assign(static_cast<size_t>(bytes), 0);
    program.constants.push_back(constant);

    Instruction item =
        instruction(opcode, MemSpace::Scratchpad, type,
                    reserveScratchpad(bytes), shape);
    item.operands.push_back(operand(
        MemSpace::Dram, type, static_cast<int64_t>(constant.region.offset),
        shape));
    return item;
  }

  case Opcode::DMA_STORE: {
    ElemType type = pickElemType(true);
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 3)));
    int64_t source = materialise(shape, type);
    int64_t bytes = checkedElementCount(shape) * elementByteSize(type);
    Instruction item =
        instruction(opcode, MemSpace::Dram, type, reserveDram(bytes), shape);
    item.operands.push_back(readBuffer(source, shape, type));
    return item;
  }

  case Opcode::MATMUL: {
    int64_t m = pick(1, 4);
    int64_t k = pick(1, 4);
    int64_t n = pick(1, 4);
    std::vector<int64_t> lhs = {m, k};
    std::vector<int64_t> rhs = {k, n};
    std::vector<int64_t> result = {m, n};
    int64_t lhsAt = materialise(lhs, ElemType::F32);
    int64_t rhsAt = materialise(rhs, ElemType::F32);
    Instruction item = instruction(
        opcode, MemSpace::Scratchpad, ElemType::F32,
        reserveScratchpad(m * n * 4), result);
    item.operands.push_back(readBuffer(lhsAt, lhs, ElemType::F32));
    item.operands.push_back(readBuffer(rhsAt, rhs, ElemType::F32));
    if (coin()) {
      std::vector<int64_t> bias = {n};
      item.operands.push_back(
          readBuffer(materialise(bias, ElemType::F32), bias, ElemType::F32));
    }
    return item;
  }

  case Opcode::CONV2D: {
    int64_t batch = pick(1, 2);
    int64_t channels = pick(1, 3);
    int64_t filters = pick(1, 3);
    int64_t height = pick(2, 4);
    int64_t width = pick(2, 4);
    std::vector<int64_t> input = {batch, channels, height, width};
    std::vector<int64_t> filter = {filters, channels, 1, 1};
    std::vector<int64_t> result = {batch, filters, height, width};
    int64_t inputAt = materialise(input, ElemType::F32);
    int64_t filterAt = materialise(filter, ElemType::F32);
    Instruction item = instruction(
        opcode, MemSpace::Scratchpad, ElemType::F32,
        reserveScratchpad(batch * filters * height * width * 4), result);
    item.operands.push_back(readBuffer(inputAt, input, ElemType::F32));
    item.operands.push_back(readBuffer(filterAt, filter, ElemType::F32));
    if (coin()) {
      std::vector<int64_t> bias = {filters};
      item.operands.push_back(
          readBuffer(materialise(bias, ElemType::F32), bias, ElemType::F32));
    }
    return item;
  }

  case Opcode::ADD:
  case Opcode::MUL: {
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 4)));
    int64_t bytes = checkedElementCount(shape) * 4;
    int64_t lhs = materialise(shape, ElemType::F32);
    int64_t rhs = materialise(shape, ElemType::F32);
    Instruction item = instruction(opcode, MemSpace::Scratchpad, ElemType::F32,
                                   reserveScratchpad(bytes), shape);
    item.operands.push_back(readBuffer(lhs, shape, ElemType::F32));
    item.operands.push_back(readBuffer(rhs, shape, ElemType::F32));
    return item;
  }

  case Opcode::RELU: {
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 4)));
    int64_t bytes = checkedElementCount(shape) * 4;
    int64_t source = materialise(shape, ElemType::F32);
    Instruction item = instruction(opcode, MemSpace::Scratchpad, ElemType::F32,
                                   reserveScratchpad(bytes), shape);
    item.operands.push_back(readBuffer(source, shape, ElemType::F32));
    return item;
  }

  case Opcode::POOL_MAX:
  case Opcode::POOL_AVG: {
    int64_t batch = pick(1, 2);
    int64_t channels = pick(1, 3);
    int64_t height = pick(2, 4);
    int64_t width = pick(2, 4);
    std::vector<int64_t> input = {batch, channels, height, width};
    std::vector<int64_t> result = {batch, channels, 1, 1};
    int64_t source = materialise(input, ElemType::F32);
    Instruction item = instruction(opcode, MemSpace::Scratchpad, ElemType::F32,
                                   reserveScratchpad(batch * channels * 4),
                                   result);
    item.operands.push_back(readBuffer(source, input, ElemType::F32));
    return item;
  }

  case Opcode::RESHAPE: {
    ElemType type = pickElemType(true);
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 3)));
    int64_t elements = checkedElementCount(shape);
    int64_t bytes = elements * elementByteSize(type);
    int64_t source = materialise(shape, type);
    Instruction item = instruction(opcode, MemSpace::Scratchpad, type,
                                   reserveScratchpad(bytes), {elements});
    item.operands.push_back(readBuffer(source, shape, type));
    return item;
  }

  case Opcode::TRANSPOSE: {
    ElemType type = pickElemType(true);
    size_t rank = static_cast<size_t>(pick(2, 3));
    std::vector<int64_t> input = pickShape(rank);
    std::vector<int64_t> axes(rank);
    for (size_t index = 0; index < rank; ++index)
      axes[index] = static_cast<int64_t>(index);
    std::shuffle(axes.begin(), axes.end(), rng);
    std::vector<int64_t> result(rank);
    for (size_t index = 0; index < rank; ++index)
      result[index] = input[static_cast<size_t>(axes[index])];

    int64_t bytes = checkedElementCount(input) * elementByteSize(type);
    int64_t source = materialise(input, type);
    Instruction item = instruction(opcode, MemSpace::Scratchpad, type,
                                   reserveScratchpad(bytes), result);
    item.operands.push_back(readBuffer(source, input, type));
    item.axes = axes;
    return item;
  }

  case Opcode::CONCAT: {
    ElemType type = pickElemType(true);
    size_t rank = static_cast<size_t>(pick(1, 3));
    std::vector<int64_t> base = pickShape(rank);
    int64_t axis = pick(0, static_cast<int64_t>(rank) - 1);
    int64_t parts = pick(1, 3);

    std::vector<std::vector<int64_t>> operandShapes;
    int64_t total = 0;
    for (int64_t part = 0; part < parts; ++part) {
      std::vector<int64_t> shape = base;
      shape[static_cast<size_t>(axis)] = pick(1, 3);
      total += shape[static_cast<size_t>(axis)];
      operandShapes.push_back(shape);
    }
    std::vector<int64_t> result = base;
    result[static_cast<size_t>(axis)] = total;

    Instruction item = instruction(
        opcode, MemSpace::Scratchpad, type,
        0, result);
    for (const std::vector<int64_t> &shape : operandShapes)
      item.operands.push_back(
          readBuffer(materialise(shape, type), shape, type));
    // The result is reserved after the operands, because materialising an
    // operand moves the cursor and an address handed out before that would
    // overlap one of them.
    item.resultAddress =
        reserveScratchpad(checkedElementCount(result) * elementByteSize(type));
    item.axes = {axis};
    return item;
  }

  case Opcode::QUANT:
  case Opcode::DEQUANT: {
    bool down = opcode == Opcode::QUANT;
    ElemType from = down ? ElemType::F32 : ElemType::I8;
    ElemType to = down ? ElemType::I8 : ElemType::F32;
    std::vector<int64_t> shape = pickShape(static_cast<size_t>(pick(1, 3)));
    int64_t elements = checkedElementCount(shape);
    int64_t source = materialise(shape, from);
    Instruction item = instruction(
        opcode, MemSpace::Scratchpad, to,
        reserveScratchpad(elements * elementByteSize(to)), shape);
    item.operands.push_back(readBuffer(source, shape, from));
    return item;
  }
  }
  // Unreachable: the switch above names every enumerator. The return is here
  // because a scoped enum can hold a value no enumerator names.
  return Instruction();
}

Program Generator::build(Coverage &coverage) {
  program = Program();
  scratchpadCursor = 0;
  dramCursor = 0;

  int64_t count = pick(1, 6);
  for (int64_t index = 0; index < count; ++index) {
    Opcode opcode = static_cast<Opcode>(pick(0, kMaxOpcode));
    Instruction item = buildFor(opcode);
    fillGeneratedFields(item);
    coverage.opcodes.insert(static_cast<uint32_t>(opcode));
    coverage.elementTypes.insert(
        static_cast<uint32_t>(item.resultElementType));
    for (const Operand &value : item.operands)
      coverage.elementTypes.insert(static_cast<uint32_t>(value.elementType));
    program.instructions.push_back(item);
  }
  program.instructions.push_back(halt());

  // Section 9.3: an explicit, tight scratchpad. The cursor is the high water
  // mark of everything the generator placed, and nothing is placed after this.
  program.scratchpadBytes = static_cast<uint64_t>(scratchpadCursor);
  program.dramBytes = static_cast<uint64_t>(dramCursor);

  // Sometimes a debug section, with strictly increasing program counters and
  // ASCII names, which is what the format requires of a real one.
  if (coin(40)) {
    ++coverage.programsWithDebug;
    for (size_t pc = 0; pc < program.instructions.size(); ++pc) {
      if (!coin(50))
        continue;
      program.debug.push_back(
          DebugEntry{static_cast<uint32_t>(pc),
                     "node_" + std::to_string(pc) + "_" +
                         std::to_string(pick(0, 999))});
    }
  }

  return program;
}

//===----------------------------------------------------------------------===//
// The property.
//===----------------------------------------------------------------------===//

TEST(EncodingProperty, EveryRandomProgramRoundTripsExactly) {
  Generator generator(kSeed);
  Generator::Coverage coverage;

  size_t largest = 0;
  size_t instructions = 0;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    Program program = generator.build(coverage);
    instructions += program.instructions.size();

    // The generator's own output is valid. This is asserted rather than
    // assumed, because a generator that quietly drifted into producing invalid
    // programs would turn every round trip below into a test of the decoder's
    // error path.
    std::optional<ProgramError> invalid = program.validate();
    ASSERT_FALSE(invalid.has_value())
        << "case " << iteration << " of " << kIterations << " at seed 0x"
        << std::hex << kSeed << std::dec << " built an invalid program: "
        << invalid->toString();

    std::vector<uint8_t> bytes = program.encode();
    largest = std::max(largest, bytes.size());

    Program decoded;
    std::optional<ProgramError> error = Program::decode(bytes, decoded);
    ASSERT_FALSE(error.has_value())
        << "case " << iteration << " of " << kIterations << " at seed 0x"
        << std::hex << kSeed << std::dec << " failed to decode: "
        << error->toString();

    ASSERT_TRUE(program == decoded)
        << "case " << iteration << " of " << kIterations << " at seed 0x"
        << std::hex << kSeed << std::dec
        << " decoded to a structurally different program";

    // The stronger half. Structural equality alone would tolerate an encoder
    // that wrote one field two different ways.
    ASSERT_EQ(bytes, decoded.encode())
        << "case " << iteration << " of " << kIterations << " at seed 0x"
        << std::hex << kSeed << std::dec << " re-encoded differently";
  }

  // Printed rather than only asserted, because the phase gate quotes these
  // numbers and a gate that quotes a number nobody printed is a gate somebody
  // has to reconstruct.
  std::cout << "[          ] " << kIterations << " programs at seed 0x"
            << std::hex << kSeed << std::dec << ", " << instructions
            << " instructions, largest file " << largest << " bytes\n";
}

TEST(EncodingProperty, TheGeneratorCoversEveryOpcodeAndEveryElementType) {
  Generator generator(kSeed);
  Generator::Coverage coverage;
  for (int iteration = 0; iteration < kIterations; ++iteration)
    generator.build(coverage);

  std::vector<std::string> missing;
  for (uint32_t raw = 0; raw <= kMaxOpcode; ++raw)
    if (!coverage.opcodes.count(raw))
      missing.push_back(opcodeName(static_cast<Opcode>(raw)));
  EXPECT_TRUE(missing.empty())
      << "opcodes never generated: " << missing.size();
  EXPECT_EQ(coverage.opcodes.size(), kNumOpcodes);

  // Every element type the format carries, which Section 17.2 asks for by
  // name. i32 reaches a program only through the movement opcodes at this
  // phase, because no compute opcode accepts it until P14.
  EXPECT_EQ(coverage.elementTypes.size(), 3u)
      << "the generator never produced one of f32, i8 and i32";
  EXPECT_GT(coverage.programsWithDebug, 0);
  EXPECT_LT(coverage.programsWithDebug, kIterations)
      << "every program had a debug section, so the empty case is untested";

  std::cout << "[          ] covered " << coverage.opcodes.size()
            << " opcodes, " << coverage.elementTypes.size()
            << " element types, " << coverage.programsWithDebug
            << " programs with a debug section\n";
}

// The same seed twice gives the same programs. Section 3's determinism rule
// applies to test generators too: a property test whose failures do not
// reproduce is a property test nobody can act on.
TEST(EncodingProperty, TheSeedIsTheWholeState) {
  Generator::Coverage first;
  Generator::Coverage second;
  Generator one(kSeed);
  Generator two(kSeed);
  for (int iteration = 0; iteration < 32; ++iteration)
    EXPECT_EQ(one.build(first).encode(), two.build(second).encode())
        << "iteration " << iteration;
}

} // namespace
