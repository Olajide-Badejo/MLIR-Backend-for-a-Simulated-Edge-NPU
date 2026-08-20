//===- Program.cpp - encoding and decoding the .nbin format ---*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The two halves of Section 9.1: turning a `Program` into bytes, and turning
// bytes back into a `Program` without ever trusting them.
//
// The decoder's whole job is to be paranoid in a specific way. Every count is
// checked against `kMaxCount` and then against the bytes that actually remain,
// **before** anything is reserved. That ordering is the point: a count of
// `(1 << 28) - 1` passes the cap and would still reserve a quarter of a
// gigabyte if the second test came later, and a length prefixed string in the
// debug section is exactly the shape of bug that turns a malformed file into a
// heap overflow.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "llvm/ADT/StringRef.h"

#include <cstring>
#include <limits>

using namespace nbin;

//===----------------------------------------------------------------------===//
// ProgramError.
//===----------------------------------------------------------------------===//

std::string ProgramError::toString() const {
  std::string out = checkName(check);
  out += ": ";
  out += detail;
  if (instructionIndex >= 0) {
    out += " (instruction ";
    out += std::to_string(instructionIndex);
    out += ")";
  }
  if (elementIndex >= 0) {
    out += " (";
    out += where.empty() ? "element" : where;
    out += " ";
    out += std::to_string(elementIndex);
    out += ")";
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Shared arithmetic.
//===----------------------------------------------------------------------===//

int64_t nbin::checkedElementCount(llvm::ArrayRef<int64_t> extents) {
  if (extents.empty())
    return -1;
  int64_t product = 1;
  for (int64_t extent : extents) {
    if (extent <= 0)
      return -1;
    // The test that has to come first. Dividing the remaining headroom by the
    // running product and comparing against the next extent asks the question
    // without performing the multiplication that would answer it wrongly.
    if (extent > Program::kShapeLimit / product)
      return -1;
    product *= extent;
  }
  return product;
}

int64_t MemRegion::elementCount() const {
  return checkedElementCount(shape);
}

int64_t MemRegion::byteSize() const {
  int64_t count = elementCount();
  if (count < 0)
    return -1;
  int64_t size = elementByteSize(elementType);
  if (size <= 0)
    return -1;
  // kShapeLimit is 2^40 and the largest element is 4 bytes, so the product is
  // at most 2^42 and cannot overflow an int64_t. The assertion is written as a
  // guard rather than a comment because the element size comes from a decoded
  // enum that validation has not necessarily looked at yet.
  if (count > Program::kShapeLimit / size)
    return -1;
  return count * size;
}

int64_t Operand::addressedByteSpan() const {
  if (shape.size() != strides.size() || shape.empty())
    return -1;
  int64_t elementSize = elementByteSize(elementType);
  if (elementSize <= 0)
    return -1;
  // The rule docs/ARCHITECTURE.md fixed at P5: a view's byte range comes from
  // its strides, so a stride 0 broadcast spans what it addresses and not what
  // its extents suggest. The hull is closed rather than exact, which is the
  // one approximation in the analysis and is in the safe direction.
  int64_t span = 1;
  for (size_t index = 0; index < shape.size(); ++index) {
    int64_t extent = shape[index];
    int64_t stride = strides[index];
    if (extent <= 0 || stride < 0)
      return -1;
    int64_t reach = extent - 1;
    if (stride != 0 && reach > Program::kShapeLimit / stride)
      return -1;
    int64_t contribution = reach * stride;
    if (contribution > Program::kShapeLimit - span)
      return -1;
    span += contribution;
  }
  if (span > Program::kShapeLimit / elementSize)
    return -1;
  return span * elementSize;
}

int64_t Instruction::resultElementCount() const {
  return checkedElementCount(resultShape);
}

int64_t Instruction::resultByteSize() const {
  int64_t count = resultElementCount();
  if (count < 0)
    return -1;
  int64_t size = elementByteSize(resultElementType);
  if (size <= 0)
    return -1;
  if (count > Program::kShapeLimit / size)
    return -1;
  return count * size;
}

llvm::StringRef Program::debugNameFor(uint32_t pc) const {
  // The entries are strictly increasing by program counter, which the
  // `debug-order` check enforces, so this could be a binary search. It is a
  // scan because the only caller is the disassembler walking the instructions
  // in order, and a scan that starts from the beginning each time is still
  // faster than a search over a section that is usually empty.
  for (const DebugEntry &entry : debug)
    if (entry.pc == pc)
      return entry.name;
  return {};
}

//===----------------------------------------------------------------------===//
// The writer.
//===----------------------------------------------------------------------===//

namespace {

/// Appends the object representation of a value to a byte vector.
///
/// Host byte order, by construction: this copies the bytes the machine already
/// has rather than serialising through shifts. Section 9.1 chooses host order
/// and `docs/ISA_MANUAL.md` says so plainly, so this is the honest
/// implementation of a stated policy rather than an oversight.
template <typename T> void put(std::vector<uint8_t> &out, T value) {
  static_assert(std::is_trivially_copyable_v<T>,
                "only trivially copyable values have an object representation "
                "worth writing");
  const auto *raw = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), raw, raw + sizeof(T));
}

void putCount(std::vector<uint8_t> &out, size_t count) {
  put<uint32_t>(out, static_cast<uint32_t>(count));
}

void putI64Vector(std::vector<uint8_t> &out,
                  const std::vector<int64_t> &values) {
  putCount(out, values.size());
  for (int64_t value : values)
    put<int64_t>(out, value);
}

void putRegion(std::vector<uint8_t> &out, const MemRegion &region) {
  put<uint64_t>(out, region.offset);
  put<uint32_t>(out, static_cast<uint32_t>(region.elementType));
  putI64Vector(out, region.shape);
}

void putOperand(std::vector<uint8_t> &out, const Operand &operand) {
  put<uint32_t>(out, static_cast<uint32_t>(operand.space));
  put<uint32_t>(out, static_cast<uint32_t>(operand.elementType));
  put<int64_t>(out, operand.address);
  putI64Vector(out, operand.shape);
  putI64Vector(out, operand.strides);
}

void putInstruction(std::vector<uint8_t> &out,
                    const Instruction &instruction) {
  put<uint32_t>(out, static_cast<uint32_t>(instruction.opcode));
  put<uint32_t>(out, static_cast<uint32_t>(instruction.activation));
  put<uint32_t>(out, static_cast<uint32_t>(instruction.resultSpace));
  put<uint32_t>(out, static_cast<uint32_t>(instruction.resultElementType));
  put<int64_t>(out, instruction.resultAddress);
  putI64Vector(out, instruction.resultShape);
  putCount(out, instruction.operands.size());
  for (const Operand &operand : instruction.operands)
    putOperand(out, operand);
  putI64Vector(out, instruction.pads);
  putI64Vector(out, instruction.strides);
  putI64Vector(out, instruction.dilations);
  putI64Vector(out, instruction.kernel);
  put<int64_t>(out, instruction.group);
  putI64Vector(out, instruction.axes);
  put<float>(out, instruction.scale);
  put<int32_t>(out, instruction.zeroPoint);
  put<int32_t>(out, instruction.requantMultiplier);
  put<int32_t>(out, instruction.requantShift);
}

} // namespace

std::vector<uint8_t> Program::encode() const {
  std::vector<uint8_t> out;

  put<uint32_t>(out, kMagic);
  put<uint32_t>(out, kVersion);
  put<uint64_t>(out, scratchpadBytes);
  put<uint64_t>(out, dramBytes);

  putCount(out, inputs.size());
  for (const MemRegion &region : inputs)
    putRegion(out, region);

  putCount(out, outputs.size());
  for (const MemRegion &region : outputs)
    putRegion(out, region);

  putCount(out, constants.size());
  for (const Constant &constant : constants) {
    putRegion(out, constant.region);
    putCount(out, constant.data.size());
    out.insert(out.end(), constant.data.begin(), constant.data.end());
  }

  putCount(out, spillSlots.size());
  for (const MemRegion &region : spillSlots)
    putRegion(out, region);

  putCount(out, instructions.size());
  for (const Instruction &instruction : instructions)
    putInstruction(out, instruction);

  putCount(out, debug.size());
  for (const DebugEntry &entry : debug) {
    put<uint32_t>(out, entry.pc);
    putCount(out, entry.name.size());
    out.insert(out.end(), entry.name.begin(), entry.name.end());
  }

  return out;
}

//===----------------------------------------------------------------------===//
// The reader.
//===----------------------------------------------------------------------===//

namespace {

/// The minimum bytes each kind of record occupies, used to reject a count
/// whose payload cannot possibly fit before anything is reserved for it.
///
/// They are minimums rather than sizes because three of the four records carry
/// length prefixed vectors of their own. Under-counting here is safe and
/// over-counting is not: the number's only job is to make a count of a hundred
/// million in a hundred byte file impossible, and the per element reads that
/// follow do the exact bounds checking.
constexpr size_t kMinRegionBytes = 8 + 4 + 4;
constexpr size_t kMinConstantBytes = kMinRegionBytes + 4;
constexpr size_t kMinOperandBytes = 4 + 4 + 8 + 4 + 4;
constexpr size_t kMinInstructionBytes =
    4 + 4 + 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4 + 4 + 4 + 4;
constexpr size_t kMinDebugEntryBytes = 4 + 4;

/// A bounds checked cursor over the file.
///
/// Nothing in here ever reads past the end, and nothing reserves memory for a
/// count it has not first compared against the bytes that remain. Both halves
/// matter: the first stops an out of bounds read and the second stops a
/// malformed file from being an allocation request.
class Reader {
public:
  Reader(llvm::ArrayRef<uint8_t> bytes) : bytes(bytes) {}

  size_t remaining() const { return bytes.size() - cursor; }
  size_t offset() const { return cursor; }
  bool atEnd() const { return cursor == bytes.size(); }

  template <typename T> bool read(T &value) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "only trivially copyable values can be read this way");
    if (remaining() < sizeof(T))
      return false;
    std::memcpy(&value, bytes.data() + cursor, sizeof(T));
    cursor += sizeof(T);
    return true;
  }

  bool readBytes(void *dest, size_t count) {
    if (remaining() < count)
      return false;
    if (count)
      std::memcpy(dest, bytes.data() + cursor, count);
    cursor += count;
    return true;
  }

  bool skip(size_t count) {
    if (remaining() < count)
      return false;
    cursor += count;
    return true;
  }

private:
  llvm::ArrayRef<uint8_t> bytes;
  size_t cursor = 0;
};

/// Everything the decode needs to fail with a stable name.
struct DecodeState {
  Reader reader;
  ProgramError error;

  explicit DecodeState(llvm::ArrayRef<uint8_t> bytes) : reader(bytes) {}

  bool fail(Check check, std::string detail) {
    error.check = check;
    error.detail = std::move(detail);
    return false;
  }

  bool truncated(const std::string &what) {
    return fail(Check::Structure,
                "the file ends in the middle of " + what + ", at byte " +
                    std::to_string(reader.offset()) + " of " +
                    std::to_string(reader.offset() + reader.remaining()));
  }

  /// Reads a `u32` count and refuses it two ways before the caller reserves
  /// anything for it.
  ///
  /// The order is the whole point and is Section 9.2 rule 3. The cap comes
  /// first because it is the stated policy and it has its own check name. The
  /// remaining bytes test comes second because it is what makes the cap
  /// sufficient: `kMaxCount` elements of a sixteen byte record is four
  /// gigabytes, so a file that claims that many and is a hundred bytes long
  /// has to be refused before the vector is sized.
  bool readCount(uint32_t &count, size_t minBytesPerElement,
                 const std::string &what) {
    if (!reader.read(count))
      return truncated("the count of " + what);
    if (count > Program::kMaxCount)
      return fail(Check::CountCap,
                  "the count of " + what + " is " + std::to_string(count) +
                      ", above the cap of " +
                      std::to_string(Program::kMaxCount));
    if (minBytesPerElement != 0) {
      size_t needed = static_cast<size_t>(count) * minBytesPerElement;
      if (needed > reader.remaining())
        return fail(Check::Structure,
                    "the count of " + what + " is " + std::to_string(count) +
                        ", which needs at least " + std::to_string(needed) +
                        " bytes and only " +
                        std::to_string(reader.remaining()) + " remain");
    }
    return true;
  }
};

bool readI64Vector(DecodeState &state, std::vector<int64_t> &out,
                   const std::string &what) {
  uint32_t count = 0;
  if (!state.readCount(count, sizeof(int64_t), what))
    return false;
  out.resize(count);
  for (uint32_t index = 0; index < count; ++index)
    if (!state.reader.read(out[index]))
      return state.truncated(what);
  return true;
}

bool readRegion(DecodeState &state, MemRegion &region,
                const std::string &what) {
  if (!state.reader.read(region.offset))
    return state.truncated(what + " offset");
  uint32_t rawType = 0;
  if (!state.reader.read(rawType))
    return state.truncated(what + " element type");
  region.elementType = static_cast<ElemType>(rawType);
  return readI64Vector(state, region.shape, what + " shape");
}

bool readOperand(DecodeState &state, Operand &operand) {
  uint32_t rawSpace = 0;
  uint32_t rawType = 0;
  if (!state.reader.read(rawSpace))
    return state.truncated("an operand memory space");
  if (!state.reader.read(rawType))
    return state.truncated("an operand element type");
  operand.space = static_cast<MemSpace>(rawSpace);
  operand.elementType = static_cast<ElemType>(rawType);
  if (!state.reader.read(operand.address))
    return state.truncated("an operand address");
  if (!readI64Vector(state, operand.shape, "an operand shape"))
    return false;
  return readI64Vector(state, operand.strides, "an operand stride vector");
}

bool readInstruction(DecodeState &state, Instruction &instruction) {
  uint32_t raw = 0;
  if (!state.reader.read(raw))
    return state.truncated("an opcode");
  instruction.opcode = static_cast<Opcode>(raw);
  if (!state.reader.read(raw))
    return state.truncated("an activation");
  instruction.activation = static_cast<Activation>(raw);
  if (!state.reader.read(raw))
    return state.truncated("a result memory space");
  instruction.resultSpace = static_cast<MemSpace>(raw);
  if (!state.reader.read(raw))
    return state.truncated("a result element type");
  instruction.resultElementType = static_cast<ElemType>(raw);
  if (!state.reader.read(instruction.resultAddress))
    return state.truncated("a result address");
  if (!readI64Vector(state, instruction.resultShape, "a result shape"))
    return false;

  uint32_t operandCount = 0;
  if (!state.readCount(operandCount, kMinOperandBytes, "the operands"))
    return false;
  instruction.operands.resize(operandCount);
  for (uint32_t index = 0; index < operandCount; ++index)
    if (!readOperand(state, instruction.operands[index]))
      return false;

  if (!readI64Vector(state, instruction.pads, "the pads"))
    return false;
  if (!readI64Vector(state, instruction.strides, "the strides"))
    return false;
  if (!readI64Vector(state, instruction.dilations, "the dilations"))
    return false;
  if (!readI64Vector(state, instruction.kernel, "the kernel shape"))
    return false;
  if (!state.reader.read(instruction.group))
    return state.truncated("a group count");
  if (!readI64Vector(state, instruction.axes, "the axes"))
    return false;
  if (!state.reader.read(instruction.scale))
    return state.truncated("a quantization scale");
  if (!state.reader.read(instruction.zeroPoint))
    return state.truncated("a quantization zero point");
  if (!state.reader.read(instruction.requantMultiplier))
    return state.truncated("a requantization multiplier");
  if (!state.reader.read(instruction.requantShift))
    return state.truncated("a requantization shift");
  return true;
}

bool readProgram(DecodeState &state, Program &out) {
  uint32_t magic = 0;
  if (!state.reader.read(magic))
    return state.truncated("the header");
  if (magic != Program::kMagic)
    return state.fail(Check::Structure,
                      "the file does not begin with the .nbin magic word");

  uint32_t version = 0;
  if (!state.reader.read(version))
    return state.truncated("the header");
  // The version is checked here rather than in validate() because everything
  // after it is laid out according to it. Reading a section at this version's
  // layout and then reporting a version mismatch would mean the reported
  // failure came after an unbounded amount of nonsense had been believed.
  if (version != Program::kVersion)
    return state.fail(Check::Version,
                      "the file is at version " + std::to_string(version) +
                          " and this build reads version " +
                          std::to_string(Program::kVersion));

  if (!state.reader.read(out.scratchpadBytes))
    return state.truncated("the header");
  if (!state.reader.read(out.dramBytes))
    return state.truncated("the header");

  uint32_t count = 0;

  if (!state.readCount(count, kMinRegionBytes, "the input regions"))
    return false;
  out.inputs.resize(count);
  for (uint32_t index = 0; index < count; ++index)
    if (!readRegion(state, out.inputs[index], "an input region"))
      return false;

  if (!state.readCount(count, kMinRegionBytes, "the output regions"))
    return false;
  out.outputs.resize(count);
  for (uint32_t index = 0; index < count; ++index)
    if (!readRegion(state, out.outputs[index], "an output region"))
      return false;

  if (!state.readCount(count, kMinConstantBytes, "the constants"))
    return false;
  out.constants.resize(count);
  for (uint32_t index = 0; index < count; ++index) {
    Constant &constant = out.constants[index];
    if (!readRegion(state, constant.region, "a constant region"))
      return false;
    uint32_t dataSize = 0;
    if (!state.readCount(dataSize, 1, "a constant's data"))
      return false;
    constant.data.resize(dataSize);
    if (!state.reader.readBytes(constant.data.data(), dataSize))
      return state.truncated("a constant's data");
  }

  if (!state.readCount(count, kMinRegionBytes, "the spill slots"))
    return false;
  out.spillSlots.resize(count);
  for (uint32_t index = 0; index < count; ++index)
    if (!readRegion(state, out.spillSlots[index], "a spill slot"))
      return false;

  if (!state.readCount(count, kMinInstructionBytes, "the instructions"))
    return false;
  out.instructions.resize(count);
  for (uint32_t index = 0; index < count; ++index) {
    if (!readInstruction(state, out.instructions[index])) {
      state.error.instructionIndex = index;
      return false;
    }
  }

  if (!state.readCount(count, kMinDebugEntryBytes, "the debug entries"))
    return false;
  out.debug.resize(count);
  for (uint32_t index = 0; index < count; ++index) {
    DebugEntry &entry = out.debug[index];
    if (!state.reader.read(entry.pc))
      return state.truncated("a debug entry's program counter");
    uint32_t length = 0;
    if (!state.readCount(length, 1, "a debug name")) {
      state.error.elementIndex = index;
      state.error.where = "debug entry";
      return false;
    }
    // The second cap, and the one that stops a legal file from being absurd.
    // kMaxCount alone would let a name be a quarter of a gigabyte, which is a
    // file nobody wrote by accident.
    if (length > Program::kMaxDebugNameBytes) {
      state.error.elementIndex = index;
      state.error.where = "debug entry";
      return state.fail(Check::DebugSize,
                        "the name is " + std::to_string(length) +
                            " bytes, above the limit of " +
                            std::to_string(Program::kMaxDebugNameBytes));
    }
    entry.name.resize(length);
    if (!state.reader.readBytes(entry.name.data(), length))
      return state.truncated("a debug name");
  }

  // Nothing may follow the last section. A file with trailing bytes is a file
  // whose author believed in a field this format does not have, and the format
  // has no tags, so there is no honest way to skip them.
  if (!state.reader.atEnd())
    return state.fail(Check::Structure,
                      std::to_string(state.reader.remaining()) +
                          " bytes follow the last section, and this format has "
                          "no tags, so there is nothing they could be");

  return true;
}

} // namespace

std::optional<ProgramError>
Program::decodeUnvalidated(llvm::ArrayRef<uint8_t> bytes, Program &out) {
  out = Program();
  DecodeState state(bytes);
  if (!readProgram(state, out))
    return state.error;
  return std::nullopt;
}

std::optional<ProgramError> Program::decode(llvm::ArrayRef<uint8_t> bytes,
                                            Program &out) {
  if (std::optional<ProgramError> error = decodeUnvalidated(bytes, out))
    return error;
  return out.validate();
}
