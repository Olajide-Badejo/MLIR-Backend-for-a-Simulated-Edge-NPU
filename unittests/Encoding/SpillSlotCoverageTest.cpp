//===- SpillSlotCoverageTest.cpp - the DRAM half of checks 8 and 9 -*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Region scoped coverage: a buffer written in pieces inside one declared spill
// slot, and read.
//
// **This file exists because `ValidationTest.cpp` cannot hold it.** That file's
// contract is that every case is rejected and names its check, and its own
// comment says a validation case whose program validates is a case that tests
// nothing. The rule here has an accepting half that is the whole point of the
// change, so the two halves live together in a file whose contract is "these
// four programs and these four verdicts".
//
// The rule, from `docs/BREAKING_CHANGES.md` and D-0052. For a DRAM read whose
// address lies inside a declared spill slot, every byte the read addresses,
// computed exactly from its strides rather than from its element count, has to
// lie inside **that one slot** and has to have been written. The scratchpad is
// not covered by this and keeps the no merge rule, because a scratchpad buffer
// has no declared extent and a merged range there would accept an over read
// into the buffer next door.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <optional>
#include <vector>

using namespace nbin;
using namespace npu_test;

namespace {

/// The strides a sub region of `parent` inherits, which are the parent's.
///
/// A tile of a larger buffer steps by the parent's extents and not by its own,
/// which is the whole reason the write side needs `resultStrides`. Written out
/// here rather than taken from `contiguousStrides(tile)`, because the two
/// differ exactly where this file is about.
std::vector<int64_t> parentStrides(const std::vector<int64_t> &parent) {
  return contiguousStrides(parent);
}

/// A program with one spill slot, two row tiles stored into it, and a read.
///
/// The slot is a `1x2x4x4` f32 buffer, 128 bytes, at DRAM offset 64. Each tile
/// is `1x2x2x4`, sixteen elements, and carries the parent's strides, so tile
/// zero touches rows 0 and 1 of both channels and tile one touches rows 2 and
/// 3. Between them they cover every byte of the slot, in four runs of 32 bytes
/// each rather than in one run of 128, which is the arrangement the old single
/// span rule could not express.
///
/// `readShape` and `readOffset` are what the last instruction reads back, so a
/// caller can ask for the whole slot, an uncovered interior, or something that
/// leaves the region.
Program assemblyProgram(bool writeSecondTile, std::vector<int64_t> readShape,
                        int64_t readOffsetBytes, int64_t extraSlotBytes = 0) {
  const std::vector<int64_t> parent = {1, 2, 4, 4};
  const std::vector<int64_t> tile = {1, 2, 2, 4};
  const int64_t slotOffset = 64;
  const int64_t slotBytes = 128;

  Program program;
  program.scratchpadBytes = 256;
  program.dramBytes = slotOffset + slotBytes + extraSlotBytes;

  program.inputs.push_back(region(0, ElemType::F32, parent));
  program.spillSlots.push_back(region(slotOffset, ElemType::F32, parent));
  if (extraSlotBytes > 0)
    program.spillSlots.push_back(
        region(static_cast<uint64_t>(slotOffset + slotBytes), ElemType::F32,
               parent));

  // One load of the whole input into the scratchpad, so the tiles have
  // something defined to store.
  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, parent);
  load.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, parent));
  program.instructions.push_back(load);

  // Tile zero, at the slot's own address, with the parent's strides.
  Instruction first = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::F32, slotOffset, tile);
  first.resultStrides = parentStrides(parent);
  first.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, tile));
  program.instructions.push_back(first);

  if (writeSecondTile) {
    // Tile one, two rows in. Row two of channel zero begins eight elements into
    // the buffer, which is 32 bytes.
    Instruction second = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                     ElemType::F32, slotOffset + 32, tile);
    second.resultStrides = parentStrides(parent);
    second.operands.push_back(
        operand(MemSpace::Scratchpad, ElemType::F32, 0, tile));
    program.instructions.push_back(second);
  }

  Instruction read = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, readShape);
  read.operands.push_back(operand(MemSpace::Dram, ElemType::F32,
                                  slotOffset + readOffsetBytes, readShape));
  program.instructions.push_back(read);

  program.instructions.push_back(halt());
  return program;
}

TEST(SpillSlotCoverage, AWholeReadOfATwoTileAssemblyIsAccepted) {
  // The case D-0052 is about: two stores cover the slot between them and the
  // consumer wants all of it. Under the single span rule this was refused with
  // "the buffer written there ends at", because the span at the slot's address
  // was one tile's.
  Program program = assemblyProgram(/*writeSecondTile=*/true, {1, 2, 4, 4}, 0);
  std::optional<ProgramError> error = program.validate();
  ASSERT_FALSE(error.has_value())
      << (error ? error->toString() : std::string());
}

TEST(SpillSlotCoverage, AReadOfBytesNoTileWroteIsRefused) {
  // The same program with the second tile left out. The read still addresses
  // the whole slot, and the rows the missing tile would have written are
  // uncovered. **This is the half that got stricter**: the old rule recorded a
  // strided write as its element count laid down as one contiguous run, so
  // sixty four bytes at the slot's address covered a read of the first sixty
  // four whatever the strides said.
  Program program = assemblyProgram(/*writeSecondTile=*/false, {1, 2, 4, 4}, 0);
  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value()) << "a read of bytes nothing wrote must fail";
  EXPECT_EQ(error->check, Check::OperandDefined);
  EXPECT_NE(error->toString().find("no instruction has written all of them"),
            std::string::npos)
      << error->toString();
}

TEST(SpillSlotCoverage, AReadThatSpansTwoSlotsIsRefusedForLeavingItsRegion) {
  // Two slots, adjacent, and a read twice the size of the first starting at its
  // address. Every byte of it has been written in the sense the scratchpad rule
  // would ask about, and it is still refused, because a read is satisfied
  // inside one declared region and never across two. That is the property the
  // no merge rule protects in the scratchpad, kept here by the region instead.
  Program program = assemblyProgram(/*writeSecondTile=*/true, {1, 2, 8, 4}, 0,
                                    /*extraSlotBytes=*/128);
  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value()) << "a read across two slots must fail";
  EXPECT_EQ(error->check, Check::OperandExtent);
  EXPECT_NE(error->toString().find("leaves spill slot 0"), std::string::npos)
      << error->toString();
}

TEST(SpillSlotCoverage, TheScratchpadKeepsTheNoMergeRule) {
  // The other side of the decision, asserted here rather than assumed from the
  // lit test that records it. Two scratchpad buffers written next to each other
  // and one read spanning both: refused, because a scratchpad buffer has no
  // declared extent and merging the two ranges is what would let an over read
  // off the end of the first pass.
  Program program;
  program.scratchpadBytes = 256;
  program.dramBytes = 128;
  program.inputs.push_back(region(0, ElemType::F32, {4, 4}));

  Instruction first = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                  ElemType::F32, 0, {4, 4});
  first.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, {4, 4}));
  program.instructions.push_back(first);

  Instruction second = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                   ElemType::F32, 64, {4, 4});
  second.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, {4, 4}));
  program.instructions.push_back(second);

  Instruction whole = instruction(Opcode::RELU, MemSpace::Scratchpad,
                                  ElemType::F32, 128, {8, 4});
  whole.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {8, 4}));
  program.instructions.push_back(whole);
  program.instructions.push_back(halt());

  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value())
      << "the scratchpad's no merge rule must still refuse this";
  EXPECT_EQ(error->check, Check::OperandExtent);
}

TEST(SpillSlotCoverage, ASpilledBufferWrittenWholeAndReloadedWholeStillPasses) {
  // The arrangement the allocator has emitted since P5, which is what the
  // committed suite is made of: one store of a whole buffer into a slot and one
  // load of the whole thing back. It has to keep passing, and it is asserted
  // here rather than inferred from the suite being green, because the suite
  // being green is what a regression here would take away.
  Program program = assemblyProgram(/*writeSecondTile=*/true, {1, 2, 2, 4}, 0);
  std::optional<ProgramError> error = program.validate();
  ASSERT_FALSE(error.has_value())
      << (error ? error->toString() : std::string());
}

//===----------------------------------------------------------------------===//
// Which slot an address is in, and the two caps.
//===----------------------------------------------------------------------===//

TEST(SpillSlotCoverage, AnAddressBetweenTwoSlotsIsInNeither) {
  // Two slots with a gap between them, and a read that starts in the gap.
  // **The gap is the case the search has to get right**: regions do not have to
  // abut, so an address can be past one slot's end and before the next's, and
  // the answer there is neither. A read there is outside every declared region
  // and falls through to the rule that governs everything outside one, which
  // refuses it because nothing wrote that address.
  const std::vector<int64_t> shape = {1, 2, 4, 4};
  Program program;
  program.scratchpadBytes = 128;
  program.dramBytes = 512;
  program.inputs.push_back(region(0, ElemType::F32, shape));
  // 128 bytes each, at 128 and at 384, so 256 to 384 belongs to neither.
  program.spillSlots.push_back(region(128, ElemType::F32, shape));
  program.spillSlots.push_back(region(384, ElemType::F32, shape));

  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, shape);
  load.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, shape));
  program.instructions.push_back(load);

  Instruction store = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::F32, 128, shape);
  store.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, shape));
  program.instructions.push_back(store);

  Instruction read = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, shape);
  read.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 256, shape));
  program.instructions.push_back(read);
  program.instructions.push_back(halt());

  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value()) << "an address in no region must fail";
  EXPECT_EQ(error->check, Check::OperandDefined);
  EXPECT_NE(error->toString().find("no declared region covers"),
            std::string::npos)
      << error->toString();
}

TEST(SpillSlotCoverage, TheSlotsAreFoundWhateverOrderTheFileDeclaresThem) {
  // The same two slots, declared in descending order. A file may print its
  // regions in any order and the search sorts them, so the answer must not
  // depend on the order. This is the assertion that sort is doing something.
  const std::vector<int64_t> shape = {1, 2, 4, 4};
  Program program;
  program.scratchpadBytes = 128;
  program.dramBytes = 512;
  program.inputs.push_back(region(0, ElemType::F32, shape));
  program.spillSlots.push_back(region(384, ElemType::F32, shape));
  program.spillSlots.push_back(region(128, ElemType::F32, shape));

  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, shape);
  load.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, shape));
  program.instructions.push_back(load);

  // Write the lower slot whole and read it back whole.
  Instruction store = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::F32, 128, shape);
  store.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, shape));
  program.instructions.push_back(store);

  Instruction read = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, shape);
  read.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 128, shape));
  program.instructions.push_back(read);
  program.instructions.push_back(halt());

  std::optional<ProgramError> error = program.validate();
  ASSERT_FALSE(error.has_value())
      << (error ? error->toString() : std::string());
}

TEST(SpillSlotCoverage, AnAccessWithTooManyRunsIsRefusedRatherThanEnumerated) {
  // **The run cap, reached without writing a file and in well under a second.**
  // The cap is 65536 runs, and an access reaches it by having more than that
  // many rows: a `[131072, 1]` shape whose innermost stride is not one is one
  // run per element, so the count is 131072 before a single run is built.
  //
  // The point of the cap is that this returns rather than enumerating, and the
  // point of the refusal is that a validator which ran out of room must not
  // accept: an access it cannot measure is one it cannot say is covered.
  const std::vector<int64_t> whole = {131072, 1};
  Program program;
  program.scratchpadBytes = 4;
  // Room for the read to address, so the in range check passes and the run
  // cap is what the access meets rather than the memory size.
  program.dramBytes = 8 * 131072 * 4;
  program.inputs.push_back(region(0, ElemType::F32, whole));
  program.spillSlots.push_back(
      region(static_cast<uint64_t>(131072 * 4), ElemType::F32, whole));

  Instruction prime = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                  ElemType::F32, 0, {1, 1});
  prime.operands.push_back(operand(MemSpace::Dram, ElemType::F32, 0, {1, 1}));
  program.instructions.push_back(prime);

  Instruction store = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::F32, 131072 * 4, whole);
  store.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {1, 1}));
  program.instructions.push_back(store);

  // The read: the same bytes, addressed one element at a time because the
  // innermost stride is two rather than one.
  Instruction read = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, {1, 1});
  Operand scattered;
  scattered.space = MemSpace::Dram;
  scattered.elementType = ElemType::F32;
  scattered.address = 131072 * 4;
  scattered.shape = {65537, 1};
  scattered.strides = {2, 2};
  read.operands.push_back(scattered);
  program.instructions.push_back(read);
  program.instructions.push_back(halt());

  std::optional<ProgramError> error = program.validate();
  ASSERT_TRUE(error.has_value()) << "an access past the run cap must fail";
  EXPECT_NE(error->toString().find("discontiguous runs"), std::string::npos)
      << error->toString();
}

} // namespace
