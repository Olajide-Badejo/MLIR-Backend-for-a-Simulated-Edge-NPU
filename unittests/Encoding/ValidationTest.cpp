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
// The integer profile of the compute opcodes.
//
// Section 14 puts the quantized convolution's data operands at i8, its bias at
// i32 and its result at i8, which is the one place in this format where an
// operand's element type is deliberately not the result's. The rules below are
// declared in the ISA description as `integerOperandTypes` and `integerFields`,
// and they apply **only** at an integer result, so every f32 program validates
// exactly as it did before an integer path existed. The pairs of cases here are
// what say so: each rule is driven at i8 and then the same field is driven on
// the same opcode at f32, where it must be refused.
//===----------------------------------------------------------------------===//

/// A load, a quantized convolution with an int32 bias, a store and a `HALT`.
///
/// The scratchpad holds an i8 input of 1 by 1 by 4 by 4, an i8 filter of
/// 1 by 1 by 1 by 1, an i32 bias of one element and an i8 result of the input's
/// shape. Sixteen plus one plus four plus sixteen is 37 bytes, and the bias sits
/// at offset 0 because a four byte element needs a four byte aligned address.
inline Program quantizedConvProgram() {
  Program program;
  program.scratchpadBytes = 40;
  program.dramBytes = 128;

  program.inputs.push_back(region(0, ElemType::I8, {1, 1, 4, 4}));
  program.outputs.push_back(region(64, ElemType::I8, {1, 1, 4, 4}));

  Constant filter;
  filter.region = region(32, ElemType::I8, {1, 1, 1, 1});
  filter.data.assign(1, 1);
  program.constants.push_back(filter);

  Constant bias;
  bias.region = region(96, ElemType::I32, {1});
  bias.data.assign(4, 0);
  program.constants.push_back(bias);

  // The bias first, at offset 0, then the two i8 buffers and the result.
  Instruction loadBias = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                     ElemType::I32, 0, {1});
  loadBias.operands.push_back(operand(MemSpace::Dram, ElemType::I32, 96, {1}));
  program.instructions.push_back(loadBias);

  Instruction loadInput = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                      ElemType::I8, 4, {1, 1, 4, 4});
  loadInput.operands.push_back(
      operand(MemSpace::Dram, ElemType::I8, 0, {1, 1, 4, 4}));
  program.instructions.push_back(loadInput);

  Instruction loadFilter = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                       ElemType::I8, 20, {1, 1, 1, 1});
  loadFilter.operands.push_back(
      operand(MemSpace::Dram, ElemType::I8, 32, {1, 1, 1, 1}));
  program.instructions.push_back(loadFilter);

  Instruction conv = instruction(Opcode::CONV2D, MemSpace::Scratchpad,
                                 ElemType::I8, 21, {1, 1, 4, 4});
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::I8, 4, {1, 1, 4, 4}));
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::I8, 20, {1, 1, 1, 1}));
  conv.operands.push_back(operand(MemSpace::Scratchpad, ElemType::I32, 0, {1}));
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  conv.group = 1;
  conv.zeroPoint = -11;
  conv.requantMultiplier = 1073741824;
  conv.requantShift = 0;
  program.instructions.push_back(conv);

  Instruction store = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::I8, 64, {1, 1, 4, 4});
  store.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::I8, 21, {1, 1, 4, 4}));
  program.instructions.push_back(store);

  program.instructions.push_back(halt());
  return program;
}

TEST(Validation, TheQuantizedConvolutionProgramValidates) {
  const std::optional<ProgramError> failure =
      quantizedConvProgram().validate();
  EXPECT_FALSE(failure.has_value())
      << (failure ? failure->toString() : std::string());
}

TEST(Validation, AQuantizedConvolutionTakesAnInt32Bias) {
  // The bias is the one operand whose type is deliberately not the result's,
  // and both of the wrong answers are refused: an f32 bias, which is what a
  // copy of the f32 rule would produce, and an i8 one, which is what "every
  // operand takes the result's type" would produce.
  Program asFloat = quantizedConvProgram();
  asFloat.instructions[3].operands[2].elementType = ElemType::F32;
  EXPECT_EQ(expectRejected(asFloat), Check::ElementTypeSupported);

  Program asInt8 = quantizedConvProgram();
  asInt8.instructions[3].operands[2].elementType = ElemType::I8;
  EXPECT_EQ(expectRejected(asInt8), Check::ElementTypeSupported);

  // And the data operands stay i8: an i32 activation is the same rule read
  // from the other end.
  Program wideInput = quantizedConvProgram();
  wideInput.instructions[3].operands[0].elementType = ElemType::I32;
  EXPECT_EQ(expectRejected(wideInput), Check::ElementTypeSupported);
}

TEST(Validation, TheZeroPointIsMeaningfulOnlyAtAnIntegerResult) {
  // At an i8 result the convolution carries the zero point of its input, which
  // is what its padding contributes. The program above already validates with
  // one, so what is left to check is the bound and the f32 refusal.
  Program outOfRange = quantizedConvProgram();
  outOfRange.instructions[3].zeroPoint = -129;
  EXPECT_EQ(expectRejected(outOfRange), Check::QuantZeroPoint);

  // The same field on the same opcode at f32 is refused, which is what makes
  // this a profile rather than a widening: nothing about the f32 path moved.
  Program asFloat = chainProgram();
  Instruction conv = instruction(Opcode::CONV2D, MemSpace::Scratchpad,
                                 ElemType::F32, 64, {1, 1, 2, 2});
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 2, 2}));
  conv.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 1, 1}));
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  conv.group = 1;
  conv.zeroPoint = 3;
  asFloat.instructions[1] = conv;
  EXPECT_EQ(expectRejected(asFloat), Check::QuantZeroPoint);
}

TEST(Validation, TheOpcodesSectionFourteenScopesOutRejectI8) {
  // Section 14: ADD, MUL, POOL_AVG, RELU and POOL_MAX reject I8 operands with a
  // named diagnostic, because the QDQ rewrite only quantizes convolution and
  // matrix multiplication. That boundary is deliberate and its cost is measured
  // rather than asserted, so it needs to be a refusal the format enforces and
  // not a convention the compiler happens to follow.
  //
  // Each of the five is driven, rather than one of them standing for the set,
  // because the rule is per opcode in the description and an opcode that gained
  // `[F32, I8]` by a copy paste would be invisible behind any other.
  const Opcode rejecting[] = {Opcode::ADD, Opcode::MUL, Opcode::RELU,
                              Opcode::POOL_MAX, Opcode::POOL_AVG};
  for (Opcode opcode : rejecting) {
    Program program = chainProgram();
    Instruction subject = instruction(opcode, MemSpace::Scratchpad,
                                      ElemType::I8, 64, {4, 4});
    subject.operands.push_back(
        operand(MemSpace::Scratchpad, ElemType::I8, 0, {4, 4}));
    if (opcode == Opcode::ADD || opcode == Opcode::MUL)
      subject.operands.push_back(
          operand(MemSpace::Scratchpad, ElemType::I8, 0, {4, 4}));
    if (opcode == Opcode::POOL_MAX || opcode == Opcode::POOL_AVG) {
      subject.resultShape = {1, 1, 4, 4};
      subject.resultStrides = contiguousStrides(subject.resultShape);
      subject.operands.front().shape = {1, 1, 4, 4};
      subject.operands.front().strides =
          contiguousStrides(subject.operands.front().shape);
      subject.kernel = {1, 1};
      subject.strides = {1, 1};
      subject.pads = {0, 0, 0, 0};
      subject.dilations = {1, 1};
    }
    program.instructions[1] = subject;
    program.instructions[2].operands.front().elementType = ElemType::I8;
    program.instructions[2].resultElementType = ElemType::I8;
    program.outputs.front().elementType = ElemType::I8;
    EXPECT_EQ(expectRejected(program), Check::ElementTypeSupported)
        << opcodeInfo(opcode).name;
  }
}

TEST(Validation, TheQuantizationPhaseDidNotMoveTheFormatVersion) {
  // Section 9.1's narrow claim, asserted rather than described: the element
  // types and the requantization pair have been present since version one, so
  // the phase that uses them bumps nothing. Version 2 is P13's, for
  // `resultStrides`, and the number this gate counts from is that one.
  //
  // It is here rather than in a document because a version bump in this phase
  // would invalidate `test_binary_stability` and every seed in the fuzz corpus
  // in the same commit that introduced quantization, which is the worst moment
  // to lose the format's regression net.
  EXPECT_EQ(Program::kVersion, 2u);

  // And the fields it is a claim about are on every instruction, at their
  // neutral values, on a program that has nothing to do with quantization.
  const Program program = chainProgram();
  for (const Instruction &carried : program.instructions) {
    EXPECT_EQ(carried.scale, 0.0f);
    EXPECT_EQ(carried.zeroPoint, 0);
    EXPECT_EQ(carried.requantMultiplier, 1);
    EXPECT_EQ(carried.requantShift, 0);
  }
}

TEST(Validation, AnIntegerComputeInstructionCarriesItsOutputZeroPointInTheScaleWord) {
  // The owner's decision of 2026-09-07, declared in `docs/BREAKING_CHANGES.md`
  // before the commit that made it: on `CONV2D` and `MATMUL`, and only at an
  // integer result, the scale word holds the output zero point. The program
  // this file already validates carries zero there, which is the symmetric case
  // and stays legal; what is new is that a whole number inside the i8 range is
  // legal too, and that the two ways of getting it wrong are refused by name.
  for (float zeroPoint : {-128.0f, -7.0f, 0.0f, 127.0f}) {
    Program program = quantizedConvProgram();
    program.instructions[3].scale = zeroPoint;
    const std::optional<ProgramError> failure = program.validate();
    EXPECT_FALSE(failure.has_value())
        << "output zero point " << zeroPoint << ": "
        << (failure ? failure->toString() : std::string());
  }

  // A fraction is a corrupt file rather than a finer grained zero point: the
  // value is added to an integer after the requantization, so a half has no
  // representation on the way out.
  for (float fractional : {1.5f, -0.5f, 0.0078125f}) {
    Program program = quantizedConvProgram();
    program.instructions[3].scale = fractional;
    EXPECT_EQ(expectRejected(program), Check::QuantZeroPoint) << fractional;
  }

  Program notFinite = quantizedConvProgram();
  notFinite.instructions[3].scale = std::numeric_limits<float>::infinity();
  EXPECT_EQ(expectRejected(notFinite), Check::QuantZeroPoint);

  Program nan = quantizedConvProgram();
  nan.instructions[3].scale = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(expectRejected(nan), Check::QuantZeroPoint);

  // One past each rail. Both, because a bound written with the wrong comparison
  // fails on one side and passes on the other.
  Program above = quantizedConvProgram();
  above.instructions[3].scale = 128.0f;
  EXPECT_EQ(expectRejected(above), Check::QuantZeroPoint);

  Program below = quantizedConvProgram();
  below.instructions[3].scale = -129.0f;
  EXPECT_EQ(expectRejected(below), Check::QuantZeroPoint);
}

TEST(Validation, TheF32PathKeepsTheRuleItAlwaysHadForTheScaleWord) {
  // The other half of the pair, and the half that says nothing moved. At an f32
  // result the same opcode gives the word no meaning and it holds zero, refused
  // by the same check with the same message as before this phase existed.
  Program asFloat = chainProgram();
  asFloat.instructions[1].scale = 1.0f;
  EXPECT_EQ(expectRejected(asFloat), Check::QuantScale);

  // And an f32 convolution, which is the opcode that changed, rather than the
  // relu that never could.
  Program conv = chainProgram();
  Instruction item = instruction(Opcode::CONV2D, MemSpace::Scratchpad,
                                 ElemType::F32, 64, {1, 1, 2, 2});
  item.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 2, 2}));
  item.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1, 1, 1}));
  item.strides = {1, 1};
  item.pads = {0, 0, 0, 0};
  item.dilations = {1, 1};
  item.group = 1;
  item.scale = -7.0f;
  conv.instructions[1] = item;
  EXPECT_EQ(expectRejected(conv), Check::QuantScale);
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
