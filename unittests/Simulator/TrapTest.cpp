//===- TrapTest.cpp - the bounds checked accessors ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 9.3: memory access goes through bounds checked accessors in **every
// build mode**, there is no `assert(false)` on the trap path, an out of range
// access records the first trap message and returns a null pointer, the caller
// skips the access, and a test proves it in a release build and an assertions
// build alike.
//
// **Why these tests use `runUnvalidated`.** A program that has passed
// `Program::validate()` cannot reach the trap path, which is exactly what the
// validator is for. A test that could only submit validated programs would
// therefore be asserting that a mechanism exists rather than that it works, and
// the last line of defence would be the one line nothing ever exercised. So the
// programs below are submitted twice: once through `run`, which refuses them at
// the validator and names the check, and once through `runUnvalidated`, which
// is the only caller of that entry point in the project and which walks them
// into the accessors. The pair is the proof: the validator is the first line,
// the accessor is the last, and both hold.
//
// **The four runs the gate asks for** are the two out of range tests below in
// each of two build directories: `build`, which is an assertions build, and
// `build-ndebug`, which is not. `docs/PHASE_STATE.md` records all four
// invocations and their output and `docs/BUILD.md` carries the configure line.
//
// **`-DCMAKE_BUILD_TYPE=Release` is not that configure line**, and the reason
// is D-0028 rather than an oversight. The LLVM this project builds against is
// an assertions build, `LLVMConfig.cmake` says so in a variable that shadows
// anything the command line offers, and `HandleLLVMOptions` then appends an
// explicit `-UNDEBUG` **after** the `-DNDEBUG` a Release configuration
// supplies. A Release build against that LLVM keeps its assertions and looks
// like a release build from the outside, which is what the first attempt at
// this gate measured. `-DNPU_FORCE_NDEBUG=ON` is the switch, and the top level
// `CMakeLists.txt` carries the whole of why it exists.
//
//===----------------------------------------------------------------------===//

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <string>

using namespace nbin;
using namespace npusim;

namespace {

/// The scratchpad byte address every program below aims at: far outside a
/// thirty two byte scratchpad, and four byte aligned so that the failure is
/// unambiguously a range failure rather than an alignment one.
constexpr int64_t kOutOfRange = 4096;

/// A program whose relu writes its result past the end of the scratchpad.
Program resultAddressOutOfRange() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, -2, 3, -4});
  const int64_t buffer = builder.scratch(4);
  // The second buffer is allocated so that the declared scratchpad is the tight
  // size of a program that would have been correct. The relu below then writes
  // somewhere else entirely, which is the fault under test.
  builder.scratch(4);

  builder.add(dmaLoad(buffer, shape, at(MemSpace::Dram, source, shape)));
  Instruction relu = compute(Opcode::RELU, kOutOfRange, shape,
                             {at(MemSpace::Scratchpad, buffer, shape)});
  builder.add(std::move(relu));
  builder.add(halt());

  // Scratchpad: two buffers of 4 f32 elements. 8 * 4 = 32 bytes. It is
  // deliberately **not** grown to cover the write at 4096: Section 9.3 forbids
  // sizing the scratchpad from the writes found in the instruction stream,
  // because doing so absorbs exactly this fault.
  return builder.finish(32);
}

/// A program whose relu reads its operand from past the end of the scratchpad.
Program operandAddressOutOfRange() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, -2, 3, -4});
  const int64_t buffer = builder.scratch(4);
  const int64_t result = builder.scratch(4);

  builder.add(dmaLoad(buffer, shape, at(MemSpace::Dram, source, shape)));
  builder.add(compute(Opcode::RELU, result, shape,
                      {at(MemSpace::Scratchpad, kOutOfRange, shape)}));
  builder.add(halt());

  // Scratchpad: two buffers of 4 f32 elements. 8 * 4 = 32 bytes.
  return builder.finish(32);
}

/// A program whose relu reads from an address this machine cannot form a load
/// from: inside the scratchpad, but not on a four byte boundary.
Program operandAddressMisaligned() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t source = builder.constant(shape, {1, -2, 3, -4});
  const int64_t buffer = builder.scratch(4);
  const int64_t result = builder.scratch(4);

  builder.add(dmaLoad(buffer, shape, at(MemSpace::Dram, source, shape)));
  builder.add(compute(Opcode::RELU, result, shape,
                      {at(MemSpace::Scratchpad, buffer + 2, shape)}));
  builder.add(halt());

  // Scratchpad: two buffers of 4 f32 elements. 8 * 4 = 32 bytes.
  return builder.finish(32);
}

//===----------------------------------------------------------------------===//
// The first line: the validator.
//===----------------------------------------------------------------------===//

TEST(Trap, TheValidatorRefusesAnOutOfRangeResultAddress) {
  Harness harness(resultAddressOutOfRange());
  const SimResult result = harness.run();

  ASSERT_FALSE(result.ok());
  // `validate()` is called again before execution by contract, and this is what
  // that second call buys: the program never reaches a kernel.
  EXPECT_NE(result.error->find("result-in-range"), std::string::npos)
      << *result.error;
  EXPECT_EQ(result.stats.instructions, 0u);
}

TEST(Trap, TheValidatorRefusesAnOutOfRangeOperandAddress) {
  Harness harness(operandAddressOutOfRange());
  const SimResult result = harness.run();

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("operand-in-range"), std::string::npos)
      << *result.error;
  EXPECT_EQ(result.stats.instructions, 0u);
}

//===----------------------------------------------------------------------===//
// The last line: the accessors, with the validator stepped around.
//===----------------------------------------------------------------------===//

TEST(Trap, AnOutOfRangeResultAddressTrapsGracefully) {
  Harness harness(resultAddressOutOfRange());
  const SimResult result = harness.runUnvalidated();

  // Graceful means all four of these at once: no crash, no assertion, a message
  // that says what happened, and a run that stopped rather than carrying on
  // over memory it did not write.
  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("out of range"), std::string::npos)
      << *result.error;
  EXPECT_NE(result.error->find("scratchpad"), std::string::npos)
      << *result.error;
  EXPECT_NE(result.error->find(std::to_string(kOutOfRange)), std::string::npos)
      << *result.error;
  // The load ran and the relu was reached, so this is the kernel trapping and
  // not the program being rejected before it started.
  EXPECT_EQ(result.stats.instructions, 2u);
  EXPECT_FALSE(result.reachedHalt);

  // The scratchpad is still exactly the size the program declared. It was never
  // grown to cover the write, which is the behaviour that would have turned
  // this fault into silently correct output.
  EXPECT_EQ(harness.sim().machine().scratchpadBytes(), 32u);
}

TEST(Trap, AnOutOfRangeOperandAddressTrapsGracefully) {
  Harness harness(operandAddressOutOfRange());
  const SimResult result = harness.runUnvalidated();

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("out of range"), std::string::npos)
      << *result.error;
  EXPECT_NE(result.error->find(std::to_string(kOutOfRange)), std::string::npos)
      << *result.error;
  EXPECT_EQ(result.stats.instructions, 2u);
}

TEST(Trap, AMisalignedAddressTrapsGracefully) {
  Harness harness(operandAddressMisaligned());
  const SimResult result = harness.runUnvalidated();

  ASSERT_FALSE(result.ok());
  // A different message from a range failure, because "the address does not
  // exist" and "the address is not one this machine can load from" send a
  // reader to two different places.
  EXPECT_NE(result.error->find("unaddressable"), std::string::npos)
      << *result.error;
  EXPECT_NE(result.error->find("four byte aligned"), std::string::npos)
      << *result.error;
}

TEST(Trap, TheFirstTrapIsTheOneReported) {
  // A relu whose result is out of range traps on every one of its four
  // elements. The message is the first one, not the fourth: a program that ran
  // off the end once usually does it a thousand times, and the thousandth
  // message says nothing the first did not.
  Harness harness(resultAddressOutOfRange());
  const SimResult result = harness.runUnvalidated();

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find(std::to_string(kOutOfRange)), std::string::npos)
      << *result.error;
  // Element zero, so the address in the message is the buffer's own and not one
  // three elements further on.
  EXPECT_EQ(result.error->find(std::to_string(kOutOfRange + 4)),
            std::string::npos)
      << *result.error;
}

/// A program whose relu carries no operand at all.
///
/// This is D-0026's case. `Program::validate()` refuses it and names the
/// `arity` check, so the only way to reach a kernel with it is
/// `runUnvalidated`, which is the entry point that exists to prove the last
/// line of defence actually holds. Before the fix the kernel reached
/// `operands.front()` on an empty vector and the machine aborted inside the
/// standard library's hardened `vector`, which is exactly the assertion
/// Section 9.3 says the trap path must not have.
Program reluWithNoOperands() {
  Builder builder;
  const std::vector<int64_t> shape = {4};
  const int64_t result = builder.scratch(4);

  Instruction relu;
  relu.opcode = Opcode::RELU;
  relu.resultSpace = MemSpace::Scratchpad;
  relu.resultElementType = ElemType::F32;
  relu.resultAddress = result;
  relu.resultShape = shape;
  builder.add(std::move(relu));
  builder.add(halt());

  // Scratchpad: one buffer of 4 f32 elements. 4 * 4 = 16 bytes.
  return builder.finish(16);
}

TEST(Trap, TheValidatorRefusesAnInstructionWithTooFewOperands) {
  Harness harness(reluWithNoOperands());
  const SimResult result = harness.run();

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("arity"), std::string::npos) << *result.error;
  EXPECT_EQ(result.stats.instructions, 0u);
}

TEST(Trap, AnInstructionWithTooFewOperandsTrapsGracefully) {
  // The same program with the validator stepped around. Graceful is the whole
  // assertion: no abort, no crash, a message naming both numbers, and a run
  // that stopped rather than reading an operand that is not there.
  Harness harness(reluWithNoOperands());
  const SimResult result = harness.runUnvalidated();

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("RELU"), std::string::npos) << *result.error;
  EXPECT_NE(result.error->find("at least 1"), std::string::npos)
      << *result.error;
  EXPECT_NE(result.error->find("has 0"), std::string::npos) << *result.error;
  EXPECT_EQ(result.stats.instructions, 1u);
  EXPECT_FALSE(result.reachedHalt);
}

TEST(Trap, AnOversizedMemoryIsRefusedRatherThanClamped) {
  // A file is free to declare a scratchpad of nearly 2^64 bytes and Section
  // 9.2 has no rule against it: the format's job is to describe a machine, not
  // to know how much memory this host has. So the refusal lives where the
  // allocation is, and it is a refusal rather than a clamp, because a clamped
  // scratchpad would be a different machine from the one the file describes and
  // every bounds check afterwards would be checking against the wrong number.
  Program program;
  program.scratchpadBytes = uint64_t{1} << 50;
  program.dramBytes = 0;
  program.instructions.push_back(halt());

  Simulator simulator(program);
  const SimResult result = simulator.run(SimOptions{});

  ASSERT_FALSE(result.ok());
  EXPECT_NE(result.error->find("refuses to allocate"), std::string::npos)
      << *result.error;
  EXPECT_EQ(result.stats.instructions, 0u);
}

} // namespace
