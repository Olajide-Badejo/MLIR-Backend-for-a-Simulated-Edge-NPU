//===- ValidationTest.cpp - One test per Program::validate rule -----------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// Program::decode used to validate the magic, cap vector lengths, and stop.
// Everything downstream then trusted the result: the simulator did raw pointer
// arithmetic on unchecked addresses, Conv2D indexed operandAddrs[1] whether or
// not it existed, and an out of range opcode fell through a switch as undefined
// behaviour. See docs/ASSESSMENT.md section 2.3.
//
// Each test here breaks exactly one invariant of an otherwise valid program and
// asserts both that it is rejected and which rule caught it. Asserting the rule
// name matters: a test that only checks "rejected" passes even when a different
// bug rejects the program for an unrelated reason.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Simulator.h"

#include "gtest/gtest.h"

using namespace npu;

namespace {

// The same well formed program the round trip test uses: load input, load
// weights, convolve, halt.
Program validProgram() {
  Program p;
  p.scratchpadBytes = 32768;
  p.dramBytes = 8192;
  p.inputs.push_back({0, {1, 1, 28, 28}});
  p.constants.push_back({4096, {6, 1, 5, 5}});
  p.constantData.push_back(std::vector<float>(150, 0.5f));
  p.outputs.push_back({8000, {1, 10}});

  Instruction loadInput;
  loadInput.op = Opcode::DmaLoad;
  loadInput.resultAddr = 0;
  loadInput.resultShape = {1, 1, 28, 28};
  loadInput.dramAddr = 0;
  p.instructions.push_back(loadInput);

  Instruction loadWeight;
  loadWeight.op = Opcode::DmaLoad;
  loadWeight.resultAddr = 3136;
  loadWeight.resultShape = {6, 1, 5, 5};
  loadWeight.dramAddr = 4096;
  p.instructions.push_back(loadWeight);

  Instruction conv;
  conv.op = Opcode::Conv2D;
  conv.resultAddr = 3736;
  conv.resultShape = {1, 6, 24, 24};
  conv.operandAddrs = {0, 3136};
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  p.instructions.push_back(conv);

  p.instructions.push_back(Instruction{Opcode::Halt});
  return p;
}

// A valid program with a DMA_STORE appended that reads `read` elements from an
// address where an appended DMA_LOAD wrote `written`. Those two counts are
// exactly what the operand extent rule compares, and everything else about the
// program stays valid so that nothing else can be what rejects it.
Program storeReadingFrom(int64_t addr, int64_t written, int64_t read) {
  Program p = validProgram();

  Instruction load;
  load.op = Opcode::DmaLoad;
  load.resultAddr = addr;
  load.resultShape = {written};
  load.dramAddr = 0;

  Instruction store;
  store.op = Opcode::DmaStore;
  store.resultShape = {read};
  store.operandAddrs = {addr};
  store.dramAddr = 0;

  // Before the halt, so the walk sees the write and then the read.
  p.instructions.insert(p.instructions.end() - 1, {load, store});
  return p;
}

// Assert the program is rejected, and by the rule we meant to break.
void expectRejected(const Program &p, const char *check) {
  std::optional<ValidationError> error = p.validate();
  ASSERT_TRUE(error.has_value()) << "expected " << check << " to reject this";
  EXPECT_EQ(error->check, check) << "rejected, but by the wrong rule: "
                                << error->toString();
  // The message has to be usable by whoever has to fix the compiler bug.
  EXPECT_FALSE(error->detail.empty());
  EXPECT_NE(error->toString().find(check), std::string::npos);
}

} // namespace

TEST(Validation, TheBaselineProgramIsValid) {
  // If this fails every other test here is meaningless.
  std::optional<ValidationError> error = validProgram().validate();
  ASSERT_FALSE(error.has_value()) << error->toString();
}

TEST(Validation, RejectsUnknownVersion) {
  Program p = validProgram();
  p.version = Program::kVersion + 1;
  expectRejected(p, "version");
}

TEST(Validation, RejectsOutOfRangeOpcode) {
  Program p = validProgram();
  // The value a corrupt u16 in the byte stream becomes.
  p.instructions[2].op = static_cast<Opcode>(9999);
  expectRejected(p, "opcode");
}

TEST(Validation, RejectsConv2DWithTooFewOperands) {
  // The specific shape of ASSESSMENT 2.3: Conv2D read operandAddrs[1]
  // unconditionally, so a one operand CONV2D indexed out of bounds.
  Program p = validProgram();
  p.instructions[2].operandAddrs = {0};
  expectRejected(p, "arity");
}

TEST(Validation, RejectsConv2DWithTooManyOperands) {
  Program p = validProgram();
  p.instructions[2].operandAddrs = {0, 3136, 3136, 3136};
  expectRejected(p, "arity");
}

TEST(Validation, RejectsNegativeResultAddress) {
  Program p = validProgram();
  p.instructions[2].resultAddr = -64;
  expectRejected(p, "result-address");
}

TEST(Validation, RejectsResultPastTheEndOfTheScratchpad) {
  Program p = validProgram();
  p.scratchpadBytes = 4096; // the conv result needs 3736 + 13824
  expectRejected(p, "result-in-range");
}

TEST(Validation, RejectsOperandOutsideTheScratchpad) {
  Program p = validProgram();
  p.instructions[2].operandAddrs[1] = 1 << 30;
  expectRejected(p, "operand-in-range");
}

TEST(Validation, RejectsNegativeOperandAddress) {
  Program p = validProgram();
  p.instructions[2].operandAddrs[0] = -4;
  expectRejected(p, "operand-in-range");
}

TEST(Validation, RejectsReadingAnAddressNothingWrote) {
  // In range, but no earlier instruction produced it, so the simulator would
  // have no shape for it and would read zeros while looking correct.
  Program p = validProgram();
  p.instructions[2].operandAddrs[1] = 20000;
  expectRejected(p, "operand-defined");
}

TEST(Validation, RejectsDmaPastTheEndOfDram) {
  Program p = validProgram();
  p.instructions[0].dramAddr = 8000; // 3136 bytes from 8000 exceeds 8192
  expectRejected(p, "dram-in-range");
}

TEST(Validation, RejectsNegativeDramAddress) {
  Program p = validProgram();
  p.instructions[0].dramAddr = -1;
  expectRejected(p, "dram-address");
}

TEST(Validation, RejectsEmptyResultShape) {
  Program p = validProgram();
  p.instructions[2].resultShape = {};
  expectRejected(p, "result-shape");
}

TEST(Validation, RejectsNonPositiveExtent) {
  Program p = validProgram();
  p.instructions[2].resultShape = {1, 6, 0, 24};
  expectRejected(p, "result-shape");
}

TEST(Validation, RejectsShapeThatWouldOverflow) {
  // Extents whose product wraps int64. Without the overflow guard the byte size
  // compares as small and the bounds check passes.
  Program p = validProgram();
  p.instructions[2].resultShape = {1LL << 40, 1LL << 40, 1LL << 40, 1LL << 40};
  expectRejected(p, "result-shape");
}

TEST(Validation, RejectsAnOperandReadLargerThanWhatWasWritten) {
  // The case from ASSESSMENT 13.2 item 4: a DMA_STORE reading 100 elements from
  // a 4 element buffer near the top of the scratchpad. Membership alone accepted
  // this, and the simulator then trapped at run time, which contradicts the
  // contract on Program.h that validate checks what the simulator relies on.
  // The read spans [32000, 32400) against a 32768 byte scratchpad.
  expectRejected(storeReadingFrom(32000, 4, 100), "operand-extent");
}

TEST(Validation, RejectsAnInteriorOverRead) {
  // The same over read, moved down so it stays comfortably inside the
  // scratchpad: [20480, 20880) of 32768. Nothing traps here and the simulator
  // would run to completion, quietly folding 96 elements of whatever happened to
  // be adjacent into the result. Being in bounds is not the same as being
  // defined, so this has to be rejected by the same rule.
  expectRejected(storeReadingFrom(20480, 4, 100), "operand-extent");
}

TEST(Validation, AcceptsAnExactExtentRead) {
  // The control. A consumer reading exactly what was written is the normal case
  // and has to keep validating, otherwise the rule above is indistinguishable
  // from refusing every DMA_STORE.
  Program p = storeReadingFrom(20480, 4, 4);
  std::optional<ValidationError> error = p.validate();
  ASSERT_FALSE(error.has_value()) << error->toString();
}

TEST(Validation, RejectsShapeAtTheOverflowBoundary) {
  // Pin the boundary the overflow guard defines, so that a later edit cannot
  // move the cap without a test saying so. The cap is 2^40 elements inclusive:
  // at exactly the cap the shape rule has nothing to say and the size check
  // downstream is what refuses the program, while one bit past it the shape
  // rule fires. Asserting the rule name is what makes this a boundary test
  // rather than two more ways of observing that a huge shape is rejected.
  Program p = validProgram();
  p.instructions[2].resultShape = {1 << 20, 1 << 20}; // 2^40, exactly the cap
  expectRejected(p, "result-in-range");

  p.instructions[2].resultShape = {1 << 20, 1 << 21}; // 2^41, one bit past it
  expectRejected(p, "result-shape");
}

TEST(Validation, RejectsRegionPastTheEndOfDram) {
  Program p = validProgram();
  // This was 8190, which is not 4 byte aligned, so the alignment rule fired
  // first and the test never reached the rule it names. The offset has to break
  // the range rule and nothing else.
  p.outputs[0].dramOffset = 8160; // aligned; 10 fp32 span [8160, 8200) > 8192
  expectRejected(p, "region-in-range");
}

TEST(Validation, RejectsNegativeRegionOffset) {
  Program p = validProgram();
  p.inputs[0].dramOffset = -16;
  expectRejected(p, "region-offset");
}

TEST(Validation, RejectsEmptyRegionShape) {
  Program p = validProgram();
  p.outputs[0].shape = {};
  expectRejected(p, "region-shape");
}

TEST(Validation, RejectsConstantDataThatDoesNotMatchItsShape) {
  Program p = validProgram();
  p.constantData[0].resize(149);
  expectRejected(p, "constant-data");
}

TEST(Validation, RejectsMissingConstantData) {
  Program p = validProgram();
  p.constantData.clear();
  expectRejected(p, "constant-data");
}

TEST(Validation, RejectsWrongStridesLength) {
  Program p = validProgram();
  p.instructions[2].strides = {1};
  expectRejected(p, "attribute-size");
}

TEST(Validation, RejectsWrongPadsLength) {
  Program p = validProgram();
  p.instructions[2].pads = {0, 0};
  expectRejected(p, "attribute-size");
}

TEST(Validation, RejectsZeroStride) {
  // A zero stride is an infinite loop in every convolution kernel ever written.
  Program p = validProgram();
  p.instructions[2].strides = {0, 1};
  expectRejected(p, "attribute-value");
}

TEST(Validation, RejectsZeroDilation) {
  Program p = validProgram();
  p.instructions[2].dilations = {1, 0};
  expectRejected(p, "attribute-value");
}

TEST(Validation, RejectsNegativePad) {
  Program p = validProgram();
  p.instructions[2].pads = {0, -1, 0, 0};
  expectRejected(p, "attribute-value");
}

TEST(Validation, RejectsZeroGroup) {
  // outPerGroup = O / group, so group zero divides by zero.
  Program p = validProgram();
  p.instructions[2].group = 0;
  expectRejected(p, "group");
}

TEST(Validation, RejectsUnknownActivation) {
  Program p = validProgram();
  p.instructions[2].activation = 7;
  expectRejected(p, "activation");
}

TEST(Validation, RejectsNegativeScratchpadSize) {
  Program p = validProgram();
  p.scratchpadBytes = -1;
  expectRejected(p, "scratchpad-size");
}

TEST(Validation, RejectsScratchpadSizeAboveTheFormatLimit) {
  // The other half of the scratchpad-size rule. A declared size the format does
  // not permit has to be refused before the simulator tries to allocate it.
  Program p = validProgram();
  p.scratchpadBytes = kMaxScratchpadBytes + 4;
  expectRejected(p, "scratchpad-size");
}

TEST(Validation, RejectsNegativeDramSize) {
  // The dram-size rule had no test at all until the manual was written against
  // the fail() call sites and the gap showed up. Both of its branches are here.
  Program p = validProgram();
  p.dramBytes = -1;
  expectRejected(p, "dram-size");
}

TEST(Validation, RejectsDramSizeAboveTheFormatLimit) {
  Program p = validProgram();
  p.dramBytes = kMaxDramBytes + 4;
  expectRejected(p, "dram-size");
}

TEST(Validation, DecodeRejectsAndReportsWhy) {
  Program p = validProgram();
  p.instructions[2].operandAddrs = {0};
  std::vector<uint8_t> bytes = p.encode();

  ValidationError error;
  EXPECT_FALSE(Program::decode(bytes, &error).has_value());
  EXPECT_EQ(error.check, "arity");
  EXPECT_EQ(error.instructionIndex, 2u);
  EXPECT_NE(error.toString().find("instruction 2"), std::string::npos);
}

TEST(Validation, DecodeUnvalidatedStillAcceptsIt) {
  // npu-objdump depends on this: dumping a file you suspect is broken is the
  // whole point of a disassembler.
  Program p = validProgram();
  p.instructions[2].operandAddrs = {0};
  auto raw = Program::decodeUnvalidated(p.encode());
  ASSERT_TRUE(raw.has_value());
  EXPECT_TRUE(raw->validate().has_value());
}

TEST(Validation, TruncatedStreamIsReportedAsStructural) {
  Program p = validProgram();
  std::vector<uint8_t> bytes = p.encode();
  bytes.resize(bytes.size() / 2);

  ValidationError error;
  EXPECT_FALSE(Program::decode(bytes, &error).has_value());
  EXPECT_EQ(error.check, "structure");
}

TEST(Validation, SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory) {
  // The simulator is reachable as a library, so it defends itself rather than
  // assuming its caller validated. Under ASan this test is the one that would
  // catch a regression in the bounds checking.
  Program p = validProgram();
  p.instructions[2].resultAddr = p.scratchpadBytes * 4; // far past the end

  Simulator sim(p);
  SimResult result = sim.run({std::vector<float>(784, 1.0f)});
  EXPECT_FALSE(result.error.empty());
  EXPECT_NE(result.error.find("instruction 2"), std::string::npos);
}
