//===- EncodingTest.cpp - round trips and frozen constants ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The hand written half of the encoding tests: a few programs whose round trip
// is checked field by field, the arithmetic the format rests on, and the
// constants that must never move.
//
// The randomized half is `PropertyTest.cpp` and it covers far more ground.
// These cases exist for the things a generator cannot assert: that `kVersion`
// is 1, that `MATMUL` is opcode 4 and always will be, and that a failure
// message names the check it came from.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/Program.h"

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace nbin;
using namespace npu_test;

namespace {

/// Encodes, decodes, and returns the decoded program. Fails the test rather
/// than returning on a decode error, so every caller can assume success.
Program roundTrip(const Program &program) {
  std::vector<uint8_t> bytes = program.encode();
  Program decoded;
  std::optional<ProgramError> error = Program::decode(bytes, decoded);
  EXPECT_FALSE(error.has_value())
      << (error ? error->toString() : std::string());
  return decoded;
}

//===----------------------------------------------------------------------===//
// The constants that must never move.
//===----------------------------------------------------------------------===//

// Opcode numeric values are assigned once and are never renumbered. This test
// is the guard on that sentence and it is written out by hand on purpose:
// reading the values back out of the generated table would assert that the
// table equals itself. A `.nbin` written by an older build has to keep meaning
// what it meant, and a renumbering would silently reinterpret every seed in the
// fuzz corpus rather than rejecting them.
TEST(FrozenConstants, OpcodeValuesNeverMove) {
  EXPECT_EQ(static_cast<uint32_t>(Opcode::NOP), 0u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::HALT), 1u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::DMA_LOAD), 2u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::DMA_STORE), 3u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::MATMUL), 4u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::CONV2D), 5u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::ADD), 6u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::MUL), 7u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::RELU), 8u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::POOL_MAX), 9u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::POOL_AVG), 10u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::RESHAPE), 11u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::TRANSPOSE), 12u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::CONCAT), 13u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::QUANT), 14u);
  EXPECT_EQ(static_cast<uint32_t>(Opcode::DEQUANT), 15u);
  EXPECT_EQ(kMaxOpcode, 15u);
  EXPECT_EQ(kNumOpcodes, 16u);
}

TEST(FrozenConstants, TheFormatsNumbers) {
  // **2 from Phase P13, and the assertion moves with the constant in the same
  // commit**, which is the P9 pattern: a frozen constant test that lagged the
  // constant it freezes would be a test nobody could trust in either
  // direction. `docs/BREAKING_CHANGES.md` carries the declaration, written
  // before the commit that caused it. The reason is D-0050: the format could
  // not express a buffer written in pieces, which is what a tiled program does,
  // and `Instruction` gained `resultStrides` for it.
  EXPECT_EQ(Program::kVersion, 2u);
  EXPECT_EQ(Program::kMagic, 0x4E49424Eu);
  EXPECT_EQ(Program::kMaxCount, 1u << 28);
  EXPECT_EQ(Program::kMaxCount, 268435456u);
  EXPECT_EQ(Program::kShapeLimit, int64_t{1} << 40);
}

// Section 9.1 pins these from the first version, and pins them narrowly: they
// and nothing broader are what let P14 land without a version bump.
TEST(FrozenConstants, ElementTypesFromTheFirstVersion) {
  EXPECT_EQ(static_cast<uint32_t>(ElemType::F32), 0u);
  EXPECT_EQ(static_cast<uint32_t>(ElemType::I8), 1u);
  EXPECT_EQ(static_cast<uint32_t>(ElemType::I32), 2u);
  EXPECT_EQ(elementByteSize(ElemType::F32), 4);
  EXPECT_EQ(elementByteSize(ElemType::I8), 1);
  EXPECT_EQ(elementByteSize(ElemType::I32), 4);
}

TEST(FrozenConstants, EveryCheckHasADistinctName) {
  std::set<std::string> names;
  for (uint32_t raw = 0; raw < kNumChecks; ++raw) {
    std::string name = checkName(static_cast<Check>(raw));
    EXPECT_FALSE(name.empty()) << "check " << raw;
    EXPECT_NE(name, "<unknown check>") << "check " << raw;
    EXPECT_TRUE(names.insert(name).second) << "duplicate check name " << name;
  }
  // Section 9.2 lists thirty three names. A count assertion is what catches a
  // check being deleted from the description, which nothing else here would.
  EXPECT_EQ(kNumChecks, 33u);
}

//===----------------------------------------------------------------------===//
// The arithmetic.
//===----------------------------------------------------------------------===//

// Section 9.2 rule 1. The guard tests before multiplying, so a shape whose
// product would overflow is refused whichever order the extents come in. A
// guard that multiplied first would accept the first of these two.
TEST(ShapeArithmetic, OverflowIsRefusedFromBothSides) {
  EXPECT_EQ(checkedElementCount({int64_t{1} << 40, int64_t{1} << 24}), -1);
  EXPECT_EQ(checkedElementCount({int64_t{1} << 24, int64_t{1} << 40}), -1);
  EXPECT_EQ(checkedElementCount({int64_t{1} << 20, int64_t{1} << 20}),
            int64_t{1} << 40);
  EXPECT_EQ(checkedElementCount({int64_t{1} << 20, int64_t{1} << 21}), -1);
}

TEST(ShapeArithmetic, NonPositiveExtentsAreRefused) {
  EXPECT_EQ(checkedElementCount({}), -1);
  EXPECT_EQ(checkedElementCount({4, 0}), -1);
  EXPECT_EQ(checkedElementCount({4, -1}), -1);
  EXPECT_EQ(checkedElementCount({4, 4}), 16);
}

// The reason `byteSize` exists at all. An i8 region is a quarter of the bytes
// an f32 one of the same shape is, and a format that assumed four bytes per
// element would size every quantized buffer wrongly at P14.
TEST(MemRegionArithmetic, ByteSizeMultipliesByTheElementSize) {
  EXPECT_EQ(region(0, ElemType::F32, {4, 4}).byteSize(), 64);
  EXPECT_EQ(region(0, ElemType::I8, {4, 4}).byteSize(), 16);
  EXPECT_EQ(region(0, ElemType::I32, {4, 4}).byteSize(), 64);
  EXPECT_EQ(region(0, ElemType::F32, {4, 4}).elementCount(), 16);
  EXPECT_EQ(region(0, ElemType::I8, {4, 4}).elementCount(), 16);
}

TEST(MemRegionArithmetic, AnImpossibleShapeHasNoByteSize) {
  EXPECT_EQ(region(0, ElemType::F32, {int64_t{1} << 40, 4}).byteSize(), -1);
  EXPECT_EQ(region(0, ElemType::F32, {}).byteSize(), -1);
}

// The rule docs/ARCHITECTURE.md fixed at P5, carried into the format. A stride
// 0 broadcast over eight channels addresses eight elements, not the hundred and
// twenty eight its extents suggest, and taking the span from the extents would
// make every per channel scale appear to collide with its neighbours.
TEST(OperandArithmetic, TheSpanComesFromTheStrides) {
  Operand broadcast;
  broadcast.space = MemSpace::Scratchpad;
  broadcast.elementType = ElemType::F32;
  broadcast.address = 0;
  broadcast.shape = {1, 8, 4, 4};
  broadcast.strides = {0, 1, 0, 0};
  EXPECT_EQ(broadcast.addressedByteSpan(), 8 * 4);

  Operand contiguous =
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 8, 4, 4});
  EXPECT_EQ(contiguous.addressedByteSpan(), 128 * 4);
}

// The NHWC case of Section 5.5: NCHW extents with permuted strides span exactly
// the contiguous buffer they are laid over.
TEST(OperandArithmetic, APermutedLayoutSpansItsBuffer) {
  Operand nhwc;
  nhwc.space = MemSpace::Scratchpad;
  nhwc.elementType = ElemType::F32;
  nhwc.address = 0;
  nhwc.shape = {1, 3, 8, 8};
  nhwc.strides = {192, 1, 24, 3};
  EXPECT_EQ(nhwc.addressedByteSpan(), 192 * 4);
}

TEST(OperandArithmetic, AMismatchedStrideVectorHasNoSpan) {
  Operand bad;
  bad.shape = {4, 4};
  bad.strides = {4};
  EXPECT_EQ(bad.addressedByteSpan(), -1);
}

//===----------------------------------------------------------------------===//
// Round trips.
//===----------------------------------------------------------------------===//

TEST(RoundTrip, TheSmallestProgram) {
  Program program = emptyProgram();
  Program decoded = roundTrip(program);
  EXPECT_TRUE(program == decoded);
  ASSERT_EQ(decoded.instructions.size(), 1u);
  EXPECT_EQ(decoded.instructions.front().opcode, Opcode::HALT);
  EXPECT_TRUE(decoded.debug.empty());
}

TEST(RoundTrip, AChainWithItsRegionsAndDebugSection) {
  Program program = chainProgram();
  Program decoded = roundTrip(program);

  EXPECT_EQ(decoded.scratchpadBytes, 128u);
  EXPECT_EQ(decoded.dramBytes, 192u);
  ASSERT_EQ(decoded.inputs.size(), 1u);
  EXPECT_EQ(decoded.inputs.front().offset, 0u);
  EXPECT_EQ(decoded.inputs.front().byteSize(), 64);
  ASSERT_EQ(decoded.constants.size(), 1u);
  EXPECT_EQ(decoded.constants.front().data.size(), 64u);
  ASSERT_EQ(decoded.instructions.size(), 4u);
  EXPECT_EQ(decoded.instructions[1].opcode, Opcode::RELU);
  ASSERT_EQ(decoded.debug.size(), 1u);
  EXPECT_EQ(decoded.debug.front().pc, 1u);
  EXPECT_EQ(decoded.debug.front().name, "relu");
  EXPECT_TRUE(program == decoded);
}

// Re-encoding is byte identical, which is the stronger half of Section 17.2's
// round trip claim: structural equality alone would tolerate an encoder that
// wrote a field two different ways.
TEST(RoundTrip, ReEncodingIsByteIdentical) {
  Program program = chainProgram();
  std::vector<uint8_t> first = program.encode();
  Program decoded;
  ASSERT_FALSE(Program::decode(first, decoded).has_value());
  EXPECT_EQ(first, decoded.encode());
}

TEST(RoundTrip, EveryFieldSurvives) {
  Program program;
  program.scratchpadBytes = 4096;
  program.dramBytes = 4096;
  program.inputs.push_back(region(0, ElemType::I8, {2, 3}));
  program.spillSlots.push_back(region(2048, ElemType::I32, {8}));

  Instruction conv = instruction(Opcode::CONV2D, MemSpace::Scratchpad,
                                 ElemType::F32, 0, {1, 2, 2, 2});
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 512, {1, 2, 2, 2}));
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 1024, {2, 2, 1, 1}));
  conv.pads = {1, 2, 3, 4};
  conv.strides = {2, 3};
  conv.dilations = {4, 5};
  conv.kernel = {};
  conv.group = 2;
  conv.activation = Activation::Relu;
  conv.requantMultiplier = 1073741824;
  conv.requantShift = 31;
  program.instructions.push_back(conv);

  Instruction quant = instruction(Opcode::QUANT, MemSpace::Scratchpad,
                                  ElemType::I8, 2048, {4});
  quant.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 3072, {4}));
  quant.scale = 0.0078125f;
  quant.zeroPoint = -128;
  program.instructions.push_back(quant);

  program.debug.push_back(DebugEntry{0, "conv"});
  program.debug.push_back(DebugEntry{1, "quantize"});

  // This program is deliberately not valid: it reads buffers nothing wrote, so
  // that every field can hold an interesting value without the operand rules
  // constraining it. What is under test here is the encoding, so the decode is
  // the unvalidated one.
  std::vector<uint8_t> bytes = program.encode();
  Program decoded;
  ASSERT_FALSE(Program::decodeUnvalidated(bytes, decoded).has_value());
  EXPECT_TRUE(program == decoded);
  EXPECT_EQ(bytes, decoded.encode());

  EXPECT_EQ(decoded.instructions[0].pads, (std::vector<int64_t>{1, 2, 3, 4}));
  EXPECT_EQ(decoded.instructions[0].group, 2);
  EXPECT_EQ(decoded.instructions[0].activation, Activation::Relu);
  EXPECT_EQ(decoded.instructions[0].requantMultiplier, 1073741824);
  EXPECT_EQ(decoded.instructions[0].requantShift, 31);
  EXPECT_EQ(decoded.instructions[1].scale, 0.0078125f);
  EXPECT_EQ(decoded.instructions[1].zeroPoint, -128);
  EXPECT_EQ(decoded.spillSlots.front().elementType, ElemType::I32);
}

// A stripped binary is legal, and it is shorter than the one it came from by
// exactly the debug section it lost.
TEST(RoundTrip, AStrippedDebugSectionIsLegal) {
  Program program = chainProgram();
  Program stripped = program;
  stripped.debug.clear();

  std::vector<uint8_t> full = program.encode();
  std::vector<uint8_t> bare = stripped.encode();
  EXPECT_LT(bare.size(), full.size());
  // The entry was a u32 program counter, a u32 length and four characters.
  EXPECT_EQ(full.size() - bare.size(), 4u + 4u + 4u);

  Program decoded;
  ASSERT_FALSE(Program::decode(bare, decoded).has_value());
  EXPECT_TRUE(decoded.debug.empty());
  EXPECT_TRUE(decoded.debugNameFor(1).empty());
}

//===----------------------------------------------------------------------===//
// The decoder's framing.
//===----------------------------------------------------------------------===//

TEST(Framing, AWrongMagicIsRefused) {
  std::vector<uint8_t> bytes = chainProgram().encode();
  writeU32(bytes, 0, 0xDEADBEEFu);
  Program decoded;
  std::optional<ProgramError> error = Program::decode(bytes, decoded);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::Structure);
}

TEST(Framing, AnUnknownVersionIsRefusedByName) {
  std::vector<uint8_t> bytes = chainProgram().encode();
  writeU32(bytes, 4, Program::kVersion + 1);
  Program decoded;
  std::optional<ProgramError> error = Program::decode(bytes, decoded);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::Version);
  EXPECT_NE(error->toString().find("version"), std::string::npos);
}

TEST(Framing, TrailingBytesAreRefused) {
  std::vector<uint8_t> bytes = chainProgram().encode();
  bytes.push_back(0);
  Program decoded;
  std::optional<ProgramError> error = Program::decode(bytes, decoded);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::Structure);
}

TEST(Framing, AnEmptyFileIsRefused) {
  Program decoded;
  std::optional<ProgramError> error = Program::decode({}, decoded);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::Structure);
}

// The whole reason decodeUnvalidated exists: a file that frames correctly and
// fails a semantic rule still decodes, so npu-objdump can show it.
TEST(Framing, DecodeUnvalidatedSkipsTheSemanticRulesAndNotTheFraming) {
  Program program = chainProgram();
  program.instructions[1].resultAddress = 1 << 20;
  std::vector<uint8_t> bytes = program.encode();

  Program decoded;
  EXPECT_FALSE(Program::decodeUnvalidated(bytes, decoded).has_value());
  EXPECT_EQ(decoded.instructions[1].resultAddress, 1 << 20);
  EXPECT_TRUE(decoded.validate().has_value());

  bytes.resize(bytes.size() - 1);
  Program truncated;
  std::optional<ProgramError> error =
      Program::decodeUnvalidated(bytes, truncated);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::Structure);
}

//===----------------------------------------------------------------------===//
// Messages.
//===----------------------------------------------------------------------===//

TEST(Messages, EveryFailureNamesItsCheckAndItsInstruction) {
  Program program = chainProgram();
  program.instructions[1].resultAddress = 1 << 20;
  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::ResultInRange);
  EXPECT_EQ(error->instructionIndex, 1);

  std::string text = error->toString();
  EXPECT_EQ(text.rfind("result-in-range: ", 0), 0u);
  EXPECT_NE(text.find("(instruction 1)"), std::string::npos);
}

TEST(Messages, ARegionFailureNamesTheRegion) {
  Program program = chainProgram();
  program.outputs.front().shape = {0, 4};
  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(error->check, Check::RegionShape);
  EXPECT_NE(error->toString().find("(output region 0)"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// The disassembler.
//===----------------------------------------------------------------------===//

TEST(Disassembly, AValidFileHasNoWarningBlock) {
  std::string text = disassemble(chainProgram(), std::nullopt);
  EXPECT_EQ(text.find("WARNING"), std::string::npos);
  EXPECT_NE(text.find("DMA_LOAD"), std::string::npos);
  EXPECT_NE(text.find("; relu"), std::string::npos);
}

TEST(Disassembly, ASuspectFileIsPrefixedWithAWarning) {
  Program program = chainProgram();
  program.instructions[1].resultAddress = 1 << 20;
  std::string text = disassemble(program, program.validate());
  EXPECT_EQ(text.rfind("; =", 0), 0u);
  EXPECT_NE(text.find("WARNING: this file did not validate"),
            std::string::npos);
  EXPECT_NE(text.find("result-in-range"), std::string::npos);
  // It still shows the instructions, which is the point.
  EXPECT_NE(text.find("0000  DMA_LOAD"), std::string::npos);
}

TEST(Disassembly, AnUnknownOpcodePrintsRatherThanCrashing) {
  Program program = chainProgram();
  program.instructions[1].opcode = static_cast<Opcode>(4242);
  std::string text = disassemble(program, program.validate());
  EXPECT_NE(text.find("<opcode 4242>"), std::string::npos);
}

// An optional operand that is absent takes its whole group out of the listing
// rather than printing an empty slot.
TEST(Disassembly, AnAbsentBiasLeavesNoHole) {
  Program program;
  program.scratchpadBytes = 4096;
  Instruction matmul = instruction(Opcode::MATMUL, MemSpace::Scratchpad,
                                   ElemType::F32, 0, {2, 2});
  matmul.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 512, {2, 2}));
  matmul.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 1024, {2, 2}));
  program.instructions.push_back(matmul);

  std::string text = disassemble(program, std::nullopt);
  EXPECT_EQ(text.find("bias"), std::string::npos);
  EXPECT_NE(text.find("MATMUL sp@0x0 2x2xf32 <- sp@0x200 2x2xf32, "
                      "sp@0x400 2x2xf32 activation=none"),
            std::string::npos);
}

// D-0023. An instruction whose mandatory operand is missing still prints.
//
// This is the whole point of npu-objdump: the file somebody is looking at is
// the file that does not validate, and an instruction that renders as a blank
// line is the one instruction they wanted to see. Before the fix, a missing
// top level operand reference made the renderer discard everything it had
// built for that instruction, opcode included, and the listing carried an
// empty line where a DMA_LOAD should have been.
//
// Found by reading the listing of a crash input the coverage guided target
// minimized, which is a use for a disassembler that a test would not have
// thought of.
TEST(Disassembly, AMissingMandatoryOperandStillPrintsTheInstruction) {
  Program program;
  program.scratchpadBytes = 128;
  program.dramBytes = 128;
  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, {4, 4});
  // No operands at all, which is what a corrupt file can carry and what the
  // arity check refuses.
  program.instructions.push_back(load);

  std::string text = disassemble(program, program.validate());
  EXPECT_NE(text.find("DMA_LOAD"), std::string::npos) << text;
  EXPECT_NE(text.find("<missing operand 0>"), std::string::npos) << text;
  // The line is not blank, which is the regression.
  EXPECT_EQ(text.find("0000  \n"), std::string::npos) << text;
}

// D-0022. A stride vector that claims the contiguous layout of a shape whose
// product overflows.
//
// The disassembler runs on the unvalidated path, so the extents are whatever
// the file said. The running product used to be multiplied without a guard,
// which is signed overflow rather than a large number, and a compiler is
// entitled to assume it cannot happen.
TEST(Disassembly, AStrideProductThatOverflowsDoesNotUndefineTheListing) {
  Program program;
  program.scratchpadBytes = 128;
  Instruction relu = instruction(Opcode::RELU, MemSpace::Scratchpad,
                                 ElemType::F32, 0, {4, 4});
  Operand overflowing;
  overflowing.space = MemSpace::Scratchpad;
  overflowing.elementType = ElemType::F32;
  overflowing.address = 0;
  // The strides are exactly the contiguous ones for this shape, so the walk
  // gets all the way to the last extent before the product would overflow.
  overflowing.shape = {8935141660703064067, 3};
  overflowing.strides = {3, 1};
  relu.operands.push_back(overflowing);
  program.instructions.push_back(relu);

  std::string text = disassemble(program, program.validate());
  EXPECT_NE(text.find("RELU"), std::string::npos) << text;
  // Not the contiguous layout as far as the listing is concerned, so the
  // strides are printed rather than elided.
  EXPECT_NE(text.find("s[3,1]"), std::string::npos) << text;
}

} // namespace
