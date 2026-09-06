//===- ValidationTest.cpp - every named check of Section 9.2 --*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Every check name Section 9.2 lists, triggered at least once, from a program
// that differs from a valid one in exactly the field the check is about.
//
// **The last test in this file is the one that makes the rest of it a rule
// rather than a habit.** It asserts that every name in the generated `Check`
// enum was reached by some case above, so a check added to the ISA description
// and never triggered fails this file instead of sitting untested.
//
// Each case asserts the check rather than the wording. The names come from the
// description and are stable; the messages are prose and are not.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace nbin;
using namespace npu_test;

namespace {

/// The set of checks any case in this file has produced.
std::set<Check> &reachedChecks() {
  static std::set<Check> reached;
  return reached;
}

/// Validates `program`, records which check fired, and returns it.
///
/// Returns `kNumChecks` worth of nothing on success, which every caller treats
/// as a failure: a validation case whose program validates is a case that tests
/// nothing.
[[nodiscard]] Check expectRejected(const Program &program) {
  std::optional<ProgramError> error = program.validate();
  EXPECT_TRUE(error.has_value()) << "this program was expected to be rejected";
  if (!error)
    return Check::Structure;
  reachedChecks().insert(error->check);
  // Every message begins with the stable name and a colon. That shape is what
  // lets a reader grep a log for a check without knowing the wording.
  std::string text = error->toString();
  EXPECT_EQ(text.rfind(std::string(checkName(error->check)) + ": ", 0), 0u)
      << text;
  return error->check;
}

/// The same, through the decoder, for the two checks a Program in memory cannot
/// reach: the version word and the count cap both live in the bytes.
[[nodiscard]] Check expectRejectedBytes(const std::vector<uint8_t> &bytes) {
  Program decoded;
  std::optional<ProgramError> error = Program::decode(bytes, decoded);
  EXPECT_TRUE(error.has_value()) << "this file was expected to be rejected";
  if (!error)
    return Check::Structure;
  reachedChecks().insert(error->check);
  return error->check;
}

/// A valid program is a valid program. Every case below starts from this, so if
/// this one ever fails the rest of the file is testing nothing in particular.
TEST(Validation, TheBaseProgramValidates) {
  EXPECT_FALSE(chainProgram().validate().has_value());
  EXPECT_FALSE(emptyProgram().validate().has_value());
}

//===----------------------------------------------------------------------===//
// The file as a whole.
//===----------------------------------------------------------------------===//

TEST(Validation, Version) {
  std::vector<uint8_t> bytes = chainProgram().encode();
  writeU32(bytes, 4, 99u);
  EXPECT_EQ(expectRejectedBytes(bytes), Check::Version);
}

TEST(Validation, CountCap) {
  std::vector<uint8_t> bytes = chainProgram().encode();
  // The input region count, at the first word after the header.
  ASSERT_EQ(readU32(bytes, 24), 1u);
  writeU32(bytes, 24, Program::kMaxCount + 1);
  EXPECT_EQ(expectRejectedBytes(bytes), Check::CountCap);
}

TEST(Validation, Structure) {
  Program program = chainProgram();
  program.instructions[1].opcode = static_cast<Opcode>(kMaxOpcode + 1);
  EXPECT_EQ(expectRejected(program), Check::Structure);
}

//===----------------------------------------------------------------------===//
// The memory regions.
//===----------------------------------------------------------------------===//

TEST(Validation, RegionShape) {
  Program program = chainProgram();
  program.inputs.front().shape = {0, 4};
  EXPECT_EQ(expectRejected(program), Check::RegionShape);

  Program overflowing = chainProgram();
  overflowing.inputs.front().shape = {int64_t{1} << 40, int64_t{1} << 24};
  EXPECT_EQ(expectRejected(overflowing), Check::RegionShape);
}

TEST(Validation, RegionOffset) {
  Program program = chainProgram();
  // An offset high enough that adding the region's own bytes overflows a
  // signed 64 bit integer, which is a different failure from running past the
  // declared DRAM size.
  program.inputs.front().offset =
      static_cast<uint64_t>(INT64_MAX) - 8;
  EXPECT_EQ(expectRejected(program), Check::RegionOffset);
}

TEST(Validation, RegionInRange) {
  Program program = chainProgram();
  program.inputs.front().offset = program.dramBytes;
  EXPECT_EQ(expectRejected(program), Check::RegionInRange);
}

TEST(Validation, ConstantData) {
  Program program = chainProgram();
  program.constants.front().data.pop_back();
  EXPECT_EQ(expectRejected(program), Check::ConstantData);

  // The reason `byteSize` multiplies by the element size. Sixteen i8 elements
  // are sixteen bytes, and a file that carried sixty four is a file whose
  // author assumed four bytes per element.
  Program mismatched = chainProgram();
  mismatched.constants.front().region.elementType = ElemType::I8;
  EXPECT_EQ(expectRejected(mismatched), Check::ConstantData);
}

//===----------------------------------------------------------------------===//
// The instruction, structurally.
//===----------------------------------------------------------------------===//

TEST(Validation, Arity) {
  Program program = chainProgram();
  program.instructions[1].operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  EXPECT_EQ(expectRejected(program), Check::Arity);

  Program none = chainProgram();
  none.instructions[1].operands.clear();
  EXPECT_EQ(expectRejected(none), Check::Arity);
}

TEST(Validation, ResultShape) {
  Program program = chainProgram();
  program.instructions[1].resultShape = {4, 0};
  EXPECT_EQ(expectRejected(program), Check::ResultShape);

  // HALT writes no result, so a result shape on one is a field that should
  // have held its neutral value.
  Program halting = chainProgram();
  halting.instructions[3].resultShape = {1};
  EXPECT_EQ(expectRejected(halting), Check::ResultShape);
}

// D-0020. A reshape that loses elements, with nothing else wrong anywhere in
// the program: the store reads exactly what the reshape wrote and the output
// region is the size the store writes, so the only rule broken is the one the
// manual states for RESHAPE and which nothing enforced until this check existed.
//
// The program is built carefully rather than casually for that reason. The
// first attempt at this case changed only the reshape, which left the store
// reading sixty four bytes from a sixteen byte buffer, and `operand-extent`
// rejected the file one instruction later. It passed either way and proved
// nothing.
TEST(Validation, ResultShapeCatchesAReshapeThatLosesElements) {
  Program program = chainProgram();

  Instruction reshape = instruction(Opcode::RESHAPE, MemSpace::Scratchpad,
                                    ElemType::F32, 64, {4});
  reshape.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  program.instructions[1] = reshape;

  // Sixteen elements in, four out, so the store moves four.
  program.instructions[2].operands.front() =
      operand(MemSpace::Scratchpad, ElemType::F32, 64, {4});
  program.instructions[2].resultShape = {4};
  program.instructions[2].resultStrides = {1};
  program.outputs.front().shape = {4};

  EXPECT_EQ(expectRejected(program), Check::ResultShape);

  // The same program with the counts agreeing is valid, which is what makes the
  // assertion above about the element count and not about anything else.
  Program agreeing = program;
  agreeing.instructions[1].resultShape = {16};
  agreeing.instructions[1].resultStrides = {1};
  agreeing.instructions[2].operands.front() =
      operand(MemSpace::Scratchpad, ElemType::F32, 64, {16});
  agreeing.instructions[2].resultShape = {16};
  agreeing.instructions[2].resultStrides = {1};
  agreeing.outputs.front().shape = {16};
  EXPECT_FALSE(agreeing.validate().has_value());
}

TEST(Validation, ResultAddress) {
  Program program = chainProgram();
  program.instructions[1].resultAddress = -64;
  EXPECT_EQ(expectRejected(program), Check::ResultAddress);

  Program halting = chainProgram();
  halting.instructions[3].resultAddress = 8;
  EXPECT_EQ(expectRejected(halting), Check::ResultAddress);

  // A RELU writing DRAM is addressing the wrong memory rather than the wrong
  // offset inside the right one.
  Program space = chainProgram();
  space.instructions[1].resultSpace = MemSpace::Dram;
  EXPECT_EQ(expectRejected(space), Check::ResultAddress);
}

TEST(Validation, ResultInRange) {
  Program program = chainProgram();
  // The scratchpad is 128 bytes and the result is 64, so 96 runs off the end.
  program.instructions[1].resultAddress = 96;
  EXPECT_EQ(expectRejected(program), Check::ResultInRange);
}

TEST(Validation, DramInRange) {
  Program program = chainProgram();
  // The store writes 64 bytes into a 192 byte DRAM, so 160 runs off the end.
  program.instructions[2].resultAddress = 160;
  EXPECT_EQ(expectRejected(program), Check::DramInRange);
}

TEST(Validation, DramAddress) {
  Program program = chainProgram();
  program.instructions[0].operands.front().address = -8;
  EXPECT_EQ(expectRejected(program), Check::DramAddress);
}

TEST(Validation, OperandInRange) {
  Program program = chainProgram();
  program.instructions[1].operands.front().address = 96;
  EXPECT_EQ(expectRejected(program), Check::OperandInRange);

  // A RELU reading DRAM. The compute units address the scratchpad and nothing
  // else, so this is not an offset mistake.
  Program space = chainProgram();
  space.instructions[1].operands.front().space = MemSpace::Dram;
  EXPECT_EQ(expectRejected(space), Check::OperandInRange);
}

TEST(Validation, OperandDefined) {
  Program program = chainProgram();
  // Nothing has written the second half of the scratchpad when the relu runs.
  program.instructions[1].operands.front().address = 64;
  EXPECT_EQ(expectRejected(program), Check::OperandDefined);

  // An output region is a place to write. Reading one before writing it is
  // reading whatever the loader left there.
  Program early = chainProgram();
  early.instructions[0].operands.front().address = 64;
  EXPECT_EQ(expectRejected(early), Check::OperandDefined);
}

// Section 9.2 rule 2, and the case it names. Membership alone would pass this:
// address 0 was written. What fails is the extent.
TEST(Validation, OperandExtent) {
  Program program = chainProgram();
  program.instructions[1].operands.front().shape = {10, 10};
  program.instructions[1].operands.front().strides = {10, 1};
  program.instructions[1].resultShape = {10, 10};
  program.scratchpadBytes = 4096;
  EXPECT_EQ(expectRejected(program), Check::OperandExtent);

  Program mismatched = chainProgram();
  mismatched.instructions[1].operands.front().strides = {4};
  EXPECT_EQ(expectRejected(mismatched), Check::OperandExtent);
}

//===----------------------------------------------------------------------===//
// The fields.
//===----------------------------------------------------------------------===//

TEST(Validation, AttributeSize) {
  // A relu gives `pads` no meaning, so it must be empty.
  Program program = chainProgram();
  program.instructions[1].pads = {0, 0, 0, 0};
  EXPECT_EQ(expectRejected(program), Check::AttributeSize);

  // A convolution needs four pads and has three.
  Program conv = chainProgram();
  Instruction convolution =
      instruction(Opcode::CONV2D, MemSpace::Scratchpad, ElemType::F32, 64,
                  {1, 1, 2, 2});
  convolution.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 4, 4}));
  convolution.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 3, 3}));
  convolution.pads = {1, 1, 1};
  convolution.strides = {1, 1};
  convolution.dilations = {1, 1};
  convolution.group = 1;
  conv.instructions[1] = convolution;
  EXPECT_EQ(expectRejected(conv), Check::AttributeSize);
}

TEST(Validation, AttributeValue) {
  Program program = chainProgram();
  Instruction pool = instruction(Opcode::POOL_MAX, MemSpace::Scratchpad,
                                 ElemType::F32, 64, {1, 1, 2, 2});
  pool.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 4, 4}));
  pool.kernel = {2, 2};
  pool.strides = {0, 2};
  pool.pads = {0, 0, 0, 0};
  pool.dilations = {1, 1};
  program.instructions[1] = pool;
  EXPECT_EQ(expectRejected(program), Check::AttributeValue);

  // A relu has no channel groups, so its group field holds zero.
  Program relu = chainProgram();
  relu.instructions[1].group = 1;
  EXPECT_EQ(expectRejected(relu), Check::AttributeValue);
}

TEST(Validation, Activation) {
  Program program = chainProgram();
  program.instructions[1].activation = static_cast<Activation>(7);
  EXPECT_EQ(expectRejected(program), Check::Activation);

  // A relu fuses no activation of its own.
  Program fused = chainProgram();
  fused.instructions[1].activation = Activation::Relu;
  EXPECT_EQ(expectRejected(fused), Check::Activation);
}

TEST(Validation, ElementType) {
  Program program = chainProgram();
  program.instructions[1].resultElementType = static_cast<ElemType>(9);
  EXPECT_EQ(expectRejected(program), Check::ElementType);

  Program regionType = chainProgram();
  regionType.inputs.front().elementType = static_cast<ElemType>(200);
  EXPECT_EQ(expectRejected(regionType), Check::ElementType);
}

TEST(Validation, ElementTypeSupported) {
  // No integer kernel exists until P14, so the description says RELU takes f32
  // and the check says so too.
  Program program = chainProgram();
  program.instructions[1].resultElementType = ElemType::I32;
  program.instructions[1].operands.front().elementType = ElemType::I32;
  EXPECT_EQ(expectRejected(program), Check::ElementTypeSupported);

  // A DMA moves bytes and does not convert them.
  Program mixed = chainProgram();
  mixed.instructions[0].operands.front().elementType = ElemType::I8;
  EXPECT_EQ(expectRejected(mixed), Check::ElementTypeSupported);
}

//===----------------------------------------------------------------------===//
// Quantization, which is structural at this phase.
//===----------------------------------------------------------------------===//

/// A program whose second instruction quantizes the loaded buffer.
Program quantProgram() {
  Program program = chainProgram();
  Instruction quant = instruction(Opcode::QUANT, MemSpace::Scratchpad,
                                  ElemType::I8, 64, {4, 4});
  quant.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  quant.scale = 0.0078125f;
  quant.zeroPoint = -128;
  program.instructions[1] = quant;
  // The store now reads sixteen i8 elements, which is sixteen bytes.
  program.instructions[2].operands.front().elementType = ElemType::I8;
  program.instructions[2].resultElementType = ElemType::I8;
  program.outputs.front().elementType = ElemType::I8;
  return program;
}

TEST(Validation, TheQuantProgramValidates) {
  EXPECT_FALSE(quantProgram().validate().has_value());
}

TEST(Validation, QuantScale) {
  Program zero = quantProgram();
  zero.instructions[1].scale = 0.0f;
  EXPECT_EQ(expectRejected(zero), Check::QuantScale);

  Program negative = quantProgram();
  negative.instructions[1].scale = -1.0f;
  EXPECT_EQ(expectRejected(negative), Check::QuantScale);

  Program nan = quantProgram();
  nan.instructions[1].scale = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(expectRejected(nan), Check::QuantScale);

  Program infinite = quantProgram();
  infinite.instructions[1].scale = std::numeric_limits<float>::infinity();
  EXPECT_EQ(expectRejected(infinite), Check::QuantScale);

  // A relu does not quantize, so its scale is zero.
  Program relu = chainProgram();
  relu.instructions[1].scale = 1.0f;
  EXPECT_EQ(expectRejected(relu), Check::QuantScale);
}

TEST(Validation, QuantZeroPoint) {
  Program program = quantProgram();
  program.instructions[1].zeroPoint = 200;
  EXPECT_EQ(expectRejected(program), Check::QuantZeroPoint);

  Program relu = chainProgram();
  relu.instructions[1].zeroPoint = 1;
  EXPECT_EQ(expectRejected(relu), Check::QuantZeroPoint);
}

TEST(Validation, QuantTypes) {
  Program program = quantProgram();
  program.instructions[1].operands.front().elementType = ElemType::I8;
  EXPECT_EQ(expectRejected(program), Check::QuantTypes);
}

TEST(Validation, QuantShape) {
  Program program = quantProgram();
  program.instructions[1].operands.front().shape = {16};
  program.instructions[1].operands.front().strides = {1};
  EXPECT_EQ(expectRejected(program), Check::QuantShape);
}

// The pair Section 9.1 adds and Section 9.2 bounds. The shift is within
// [0, 31] and the multiplier is a positive int32, which is the range the fixed
// point decomposition of Section 14 produces.
TEST(Validation, QuantRequantize) {
  Program zero = chainProgram();
  zero.instructions[1].requantMultiplier = 0;
  EXPECT_EQ(expectRejected(zero), Check::QuantRequantize);

  Program negative = chainProgram();
  negative.instructions[1].requantMultiplier = -1;
  EXPECT_EQ(expectRejected(negative), Check::QuantRequantize);

  Program shifted = chainProgram();
  shifted.instructions[1].requantShift = 32;
  EXPECT_EQ(expectRejected(shifted), Check::QuantRequantize);

  Program negativeShift = chainProgram();
  negativeShift.instructions[1].requantShift = -1;
  EXPECT_EQ(expectRejected(negativeShift), Check::QuantRequantize);

  // A relu does not requantize, so the pair is the identity.
  Program relu = chainProgram();
  relu.instructions[1].requantMultiplier = 2;
  EXPECT_EQ(expectRejected(relu), Check::QuantRequantize);
}

//===----------------------------------------------------------------------===//
// The shape relations.
//===----------------------------------------------------------------------===//

/// A program whose second instruction transposes the loaded buffer.
Program transposeProgram() {
  Program program = chainProgram();
  Instruction transpose = instruction(Opcode::TRANSPOSE, MemSpace::Scratchpad,
                                      ElemType::F32, 64, {4, 4});
  transpose.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  transpose.axes = {1, 0};
  program.instructions[1] = transpose;
  return program;
}

TEST(Validation, TheTransposeProgramValidates) {
  EXPECT_FALSE(transposeProgram().validate().has_value());
}

TEST(Validation, AxesPermutation) {
  Program repeated = transposeProgram();
  repeated.instructions[1].axes = {0, 0};
  EXPECT_EQ(expectRejected(repeated), Check::AxesPermutation);

  Program outOfRange = transposeProgram();
  outOfRange.instructions[1].axes = {0, 5};
  EXPECT_EQ(expectRejected(outOfRange), Check::AxesPermutation);

  Program wrongLength = transposeProgram();
  wrongLength.instructions[1].axes = {0};
  EXPECT_EQ(expectRejected(wrongLength), Check::AxesPermutation);
}

/// A program whose second instruction concatenates two halves of the loaded
/// buffer. The scratchpad is widened so that the result fits.
Program concatProgram() {
  Program program = chainProgram();
  program.scratchpadBytes = 256;
  Instruction concat = instruction(Opcode::CONCAT, MemSpace::Scratchpad,
                                   ElemType::F32, 64, {8, 4});
  concat.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  concat.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  concat.axes = {0};
  program.instructions[1] = concat;
  program.instructions[2].operands.front().shape = {8, 4};
  program.instructions[2].operands.front().strides = {4, 1};
  program.instructions[2].resultShape = {8, 4};
  program.outputs.front().shape = {8, 4};
  program.dramBytes = 256;
  program.constants.front().region.offset = 192;
  return program;
}

TEST(Validation, TheConcatProgramValidates) {
  EXPECT_FALSE(concatProgram().validate().has_value());
}

TEST(Validation, ConcatAxis) {
  Program two = concatProgram();
  two.instructions[1].axes = {0, 1};
  EXPECT_EQ(expectRejected(two), Check::ConcatAxis);

  Program outOfRange = concatProgram();
  outOfRange.instructions[1].axes = {4};
  EXPECT_EQ(expectRejected(outOfRange), Check::ConcatAxis);
}

TEST(Validation, ConcatExtents) {
  Program shortSum = concatProgram();
  shortSum.instructions[1].operands.pop_back();
  EXPECT_EQ(expectRejected(shortSum), Check::ConcatExtents);

  Program wrongRank = concatProgram();
  wrongRank.instructions[1].operands.back().shape = {16};
  wrongRank.instructions[1].operands.back().strides = {1};
  EXPECT_EQ(expectRejected(wrongRank), Check::ConcatExtents);

  Program offAxis = concatProgram();
  offAxis.instructions[1].operands.back().shape = {4, 2};
  offAxis.instructions[1].operands.back().strides = {2, 1};
  EXPECT_EQ(expectRejected(offAxis), Check::ConcatExtents);
}

//===----------------------------------------------------------------------===//
// The debug section.
//===----------------------------------------------------------------------===//

TEST(Validation, DebugPc) {
  Program program = chainProgram();
  program.debug.front().pc = 99;
  EXPECT_EQ(expectRejected(program), Check::DebugPc);
}

TEST(Validation, DebugOrder) {
  Program program = chainProgram();
  program.debug.push_back(DebugEntry{1, "again"});
  EXPECT_EQ(expectRejected(program), Check::DebugOrder);

  Program backwards = chainProgram();
  backwards.debug.front().pc = 2;
  backwards.debug.push_back(DebugEntry{0, "earlier"});
  EXPECT_EQ(expectRejected(backwards), Check::DebugOrder);
}

TEST(Validation, DebugName) {
  Program embedded = chainProgram();
  embedded.debug.front().name = std::string("re\0lu", 5);
  EXPECT_EQ(expectRejected(embedded), Check::DebugName);

  Program high = chainProgram();
  high.debug.front().name = "rel\xc3\xbc";
  EXPECT_EQ(expectRejected(high), Check::DebugName);
}

TEST(Validation, DebugSize) {
  Program program = chainProgram();
  program.debug.front().name =
      std::string(Program::kMaxDebugNameBytes + 1, 'a');
  EXPECT_EQ(expectRejected(program), Check::DebugSize);
}

//===----------------------------------------------------------------------===//
// The gate on this file.
//===----------------------------------------------------------------------===//

// gtest runs tests in declaration order within a suite by default, and this one
// depends on every case above having run. It is in its own suite, named to sort
// after `Validation`, because gtest orders suites by first declaration and this
// file declares `Validation` first.
TEST(ValidationCoverage, EveryCheckWasTriggered) {
  std::vector<std::string> missing;
  for (uint32_t raw = 0; raw < kNumChecks; ++raw) {
    Check check = static_cast<Check>(raw);
    if (!reachedChecks().count(check))
      missing.push_back(checkName(check));
  }
  EXPECT_TRUE(missing.empty())
      << "these checks were never triggered by any case in ValidationTest.cpp: "
      << [&] {
           std::string text;
           for (const std::string &name : missing) {
             if (!text.empty())
               text += ", ";
             text += name;
           }
           return text;
         }();
  EXPECT_EQ(reachedChecks().size(), kNumChecks);
}

} // namespace
