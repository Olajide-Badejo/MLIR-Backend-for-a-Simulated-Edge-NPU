//===- TestPrograms.h - programs the encoding tests share -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// A small valid program and the pieces to build others from, shared by the four
// test files in this directory.
//
// It is a header rather than a fourth translation unit because LLVM's build
// system wants one target per directory and the four test files are one binary;
// a header keeps the builders next to the tests that use them without adding a
// source file that belongs to nobody.
//
// **Every program these build sets a tight, explicit `scratchpadBytes` with the
// arithmetic in a comment.** Section 9.3 asks for that by name, and the reason
// is that a generous scratchpad hides exactly the bugs the bounds checks exist
// to catch: a result address one buffer past the end of the program's real
// footprint is an error only if the declared footprint is the real one.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_UNITTESTS_ENCODING_TESTPROGRAMS_H
#define NPU_UNITTESTS_ENCODING_TESTPROGRAMS_H

#include "NPU/Encoding/Program.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

//===----------------------------------------------------------------------===//
// Structural equality.
//
// Section 17.2 asks the property test to assert that a decoded program is
// structurally equal to the one that was encoded, so equality has to mean every
// field of every record. These live in the test rather than in Program.h
// because nothing in the compiler compares two programs; the encoder writes one
// and the simulator runs one.
//
// They are declared **in the format's own namespace** rather than in the test's,
// and that is not a style choice. Comparing two `std::vector<Operand>` reaches
// `operator==` through argument dependent lookup, which searches the namespace
// `Operand` lives in and nowhere else; defined next door in `npu_test` they
// would be invisible to every container comparison in this directory.
//===----------------------------------------------------------------------===//

namespace nbin {

inline bool operator==(const MemRegion &lhs, const MemRegion &rhs) {
  return lhs.offset == rhs.offset && lhs.elementType == rhs.elementType &&
         lhs.shape == rhs.shape;
}

inline bool operator==(const Constant &lhs, const Constant &rhs) {
  return lhs.region == rhs.region && lhs.data == rhs.data;
}

inline bool operator==(const Operand &lhs, const Operand &rhs) {
  return lhs.space == rhs.space && lhs.elementType == rhs.elementType &&
         lhs.address == rhs.address && lhs.shape == rhs.shape &&
         lhs.strides == rhs.strides;
}

inline bool operator==(const Instruction &lhs, const Instruction &rhs) {
  return lhs.opcode == rhs.opcode && lhs.activation == rhs.activation &&
         lhs.resultSpace == rhs.resultSpace &&
         lhs.resultElementType == rhs.resultElementType &&
         lhs.resultAddress == rhs.resultAddress &&
         lhs.resultShape == rhs.resultShape && lhs.operands == rhs.operands &&
         lhs.pads == rhs.pads && lhs.strides == rhs.strides &&
         lhs.dilations == rhs.dilations && lhs.kernel == rhs.kernel &&
         lhs.group == rhs.group && lhs.axes == rhs.axes &&
         // The bit pattern rather than the value, because a scale that round
         // tripped through a different NaN payload is a format that lost
         // information even though `==` on two NaNs would have said nothing.
         std::memcmp(&lhs.scale, &rhs.scale, sizeof(float)) == 0 &&
         lhs.zeroPoint == rhs.zeroPoint &&
         lhs.requantMultiplier == rhs.requantMultiplier &&
         lhs.requantShift == rhs.requantShift;
}

inline bool operator==(const DebugEntry &lhs, const DebugEntry &rhs) {
  return lhs.pc == rhs.pc && lhs.name == rhs.name;
}

inline bool operator==(const Program &lhs, const Program &rhs) {
  return lhs.scratchpadBytes == rhs.scratchpadBytes &&
         lhs.dramBytes == rhs.dramBytes && lhs.inputs == rhs.inputs &&
         lhs.outputs == rhs.outputs && lhs.constants == rhs.constants &&
         lhs.spillSlots == rhs.spillSlots &&
         lhs.instructions == rhs.instructions && lhs.debug == rhs.debug;
}

} // namespace nbin

namespace npu_test {

using namespace nbin;

/// A region at `offset` holding `shape` elements of `type`.
inline MemRegion region(uint64_t offset, ElemType type,
                        std::vector<int64_t> shape) {
  MemRegion out;
  out.offset = offset;
  out.elementType = type;
  out.shape = std::move(shape);
  return out;
}

/// The contiguous row major strides for `shape`.
inline std::vector<int64_t> contiguousStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t index = shape.size(); index-- > 1;)
    strides[index - 1] = strides[index] * shape[index];
  return strides;
}

/// An operand reading `shape` contiguously from `address`.
inline Operand operand(MemSpace space, ElemType type, int64_t address,
                       std::vector<int64_t> shape) {
  Operand out;
  out.space = space;
  out.elementType = type;
  out.address = address;
  out.strides = contiguousStrides(shape);
  out.shape = std::move(shape);
  return out;
}

/// An instruction with its result already filled in and no operands yet.
inline Instruction instruction(Opcode opcode, MemSpace resultSpace,
                               ElemType type, int64_t address,
                               std::vector<int64_t> shape) {
  Instruction out;
  out.opcode = opcode;
  out.resultSpace = resultSpace;
  out.resultElementType = type;
  out.resultAddress = address;
  out.resultShape = std::move(shape);
  return out;
}

/// `HALT` on its own, with every field at its neutral value.
inline Instruction halt() {
  Instruction out;
  out.opcode = Opcode::HALT;
  return out;
}

/// The smallest program that validates: no memory, no work, one `HALT`.
inline Program emptyProgram() {
  Program program;
  program.scratchpadBytes = 0;
  program.dramBytes = 0;
  program.instructions.push_back(halt());
  return program;
}

/// A load, a relu, a store and a `HALT`, over one 4 by 4 f32 buffer.
///
/// The scratchpad is 128 bytes: two buffers of 4 by 4 f32, which is 16 elements
/// at 4 bytes, so 64 bytes each. DRAM is 192 bytes: the input at 0, the output
/// at 64, and a constant at 128, 64 bytes apiece.
///
/// This is the base every malformed case in `MalformedInputTest` is a mutation
/// of, so it is deliberately small: a corpus built on a large program is a
/// corpus whose failures are hard to read.
inline Program chainProgram() {
  Program program;
  program.scratchpadBytes = 128;
  program.dramBytes = 192;

  program.inputs.push_back(region(0, ElemType::F32, {4, 4}));
  program.outputs.push_back(region(64, ElemType::F32, {4, 4}));

  Constant constant;
  constant.region = region(128, ElemType::F32, {4, 4});
  constant.data.assign(64, 0);
  program.constants.push_back(constant);

  Instruction load = instruction(Opcode::DMA_LOAD, MemSpace::Scratchpad,
                                 ElemType::F32, 0, {4, 4});
  load.operands.push_back(
      operand(MemSpace::Dram, ElemType::F32, 0, {4, 4}));
  program.instructions.push_back(load);

  Instruction relu = instruction(Opcode::RELU, MemSpace::Scratchpad,
                                 ElemType::F32, 64, {4, 4});
  relu.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  program.instructions.push_back(relu);

  Instruction store = instruction(Opcode::DMA_STORE, MemSpace::Dram,
                                  ElemType::F32, 64, {4, 4});
  store.operands.push_back(
      operand(MemSpace::Scratchpad, ElemType::F32, 64, {4, 4}));
  program.instructions.push_back(store);

  program.instructions.push_back(halt());

  program.debug.push_back(DebugEntry{1, "relu"});
  return program;
}

/// Reads a `u32` out of an encoded file.
inline uint32_t readU32(const std::vector<uint8_t> &bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

/// Writes a `u32` into an encoded file.
inline void writeU32(std::vector<uint8_t> &bytes, size_t offset,
                     uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

/// One `u32` count field of an encoded file: where it is and what it holds.
struct CountField {
  size_t offset;
  uint32_t value;
  std::string name;
};

/// Every `u32` count field in the encoding of `program`, in file order.
///
/// This walks the layout a second time rather than asking the encoder where it
/// put things, and the duplication is the point. Section 9.2's cap applies to
/// **every** count field, so a test that probed the ones it happened to know
/// about would be a test of the author's memory. Each entry carries the value
/// it expects to find, and `MalformedInputTest` asserts that before mutating,
/// so a layout change that this walker does not follow fails loudly instead of
/// quietly probing the wrong bytes.
inline std::vector<CountField> countFields(const Program &program) {
  std::vector<CountField> fields;
  // magic, version, scratchpadBytes, dramBytes.
  size_t at = 4 + 4 + 8 + 8;

  auto count = [&](uint32_t value, const std::string &name) {
    fields.push_back({at, value, name});
    at += 4;
  };
  auto regionAt = [&](const MemRegion &memRegion, const std::string &name) {
    at += 8;  // offset
    at += 4;  // element type
    count(static_cast<uint32_t>(memRegion.shape.size()), name + " rank");
    at += 8 * memRegion.shape.size();
  };
  auto vectorAt = [&](const std::vector<int64_t> &values,
                      const std::string &name) {
    count(static_cast<uint32_t>(values.size()), name);
    at += 8 * values.size();
  };

  count(static_cast<uint32_t>(program.inputs.size()), "inputs");
  for (size_t index = 0; index < program.inputs.size(); ++index)
    regionAt(program.inputs[index], "input " + std::to_string(index));

  count(static_cast<uint32_t>(program.outputs.size()), "outputs");
  for (size_t index = 0; index < program.outputs.size(); ++index)
    regionAt(program.outputs[index], "output " + std::to_string(index));

  count(static_cast<uint32_t>(program.constants.size()), "constants");
  for (size_t index = 0; index < program.constants.size(); ++index) {
    const Constant &constant = program.constants[index];
    regionAt(constant.region, "constant " + std::to_string(index));
    count(static_cast<uint32_t>(constant.data.size()),
          "constant " + std::to_string(index) + " data");
    at += constant.data.size();
  }

  count(static_cast<uint32_t>(program.spillSlots.size()), "spill slots");
  for (size_t index = 0; index < program.spillSlots.size(); ++index)
    regionAt(program.spillSlots[index], "spill slot " + std::to_string(index));

  count(static_cast<uint32_t>(program.instructions.size()), "instructions");
  for (size_t index = 0; index < program.instructions.size(); ++index) {
    const Instruction &item = program.instructions[index];
    std::string where = "instruction " + std::to_string(index);
    at += 4 + 4 + 4 + 4;  // opcode, activation, result space, result type
    at += 8;              // result address
    vectorAt(item.resultShape, where + " result rank");
    count(static_cast<uint32_t>(item.operands.size()), where + " operands");
    for (size_t operandIndex = 0; operandIndex < item.operands.size();
         ++operandIndex) {
      const Operand &value = item.operands[operandIndex];
      std::string operandWhere =
          where + " operand " + std::to_string(operandIndex);
      at += 4 + 4 + 8;  // space, element type, address
      vectorAt(value.shape, operandWhere + " rank");
      vectorAt(value.strides, operandWhere + " strides");
    }
    vectorAt(item.pads, where + " pads");
    vectorAt(item.strides, where + " strides");
    vectorAt(item.dilations, where + " dilations");
    vectorAt(item.kernel, where + " kernel");
    at += 8;  // group
    vectorAt(item.axes, where + " axes");
    at += 4 + 4 + 4 + 4;  // scale, zero point, multiplier, shift
  }

  count(static_cast<uint32_t>(program.debug.size()), "debug entries");
  for (size_t index = 0; index < program.debug.size(); ++index) {
    at += 4;  // program counter
    count(static_cast<uint32_t>(program.debug[index].name.size()),
          "debug " + std::to_string(index) + " name length");
    at += program.debug[index].name.size();
  }

  return fields;
}

} // namespace npu_test

#endif // NPU_UNITTESTS_ENCODING_TESTPROGRAMS_H
