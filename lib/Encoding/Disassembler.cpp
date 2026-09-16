//===- Disassembler.cpp - printing a .nbin ------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The disassembler drives itself from the generated format strings, so the
// only thing here that knows about a particular opcode is the substitution
// machinery, and adding an opcode to the description adds its disassembly with
// it.
//
// The grammar of a format string, which `NPUISADescription.td` documents from
// the other side:
//
//   %r       the result: space, address, shape and element type
//   %0 %1 %2 the operand at that position
//   %n       every operand, comma separated, for a variadic opcode
//   {...}    a group. A group naming an operand that the instruction does not
//            have is dropped whole, which is how an optional bias disappears
//            rather than printing as an empty slot. Inside a group, a bare
//            word that names a field prints as `word=value`.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"

#include "llvm/ADT/StringRef.h"

#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

using namespace nbin;

namespace {

std::string hex(int64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "0x%" PRIx64, value);
  return buffer;
}

std::string shapeText(const std::vector<int64_t> &shape, ElemType type) {
  // The `4x4xf32` spelling MLIR uses, so that a listing and the IR it came
  // from read the same way. A rank 0 shape is not a shape this machine has an
  // address for, but a disassembly of a suspect file has to print one anyway.
  std::string out;
  for (int64_t extent : shape) {
    out += std::to_string(extent);
    out += "x";
  }
  if (shape.empty())
    out += "<rank 0>";
  out += elemTypeName(type);
  return out;
}

std::string vectorText(const std::vector<int64_t> &values) {
  std::string out = "[";
  for (size_t index = 0; index < values.size(); ++index) {
    if (index)
      out += ",";
    out += std::to_string(values[index]);
  }
  out += "]";
  return out;
}

std::string spaceAbbreviation(MemSpace space) {
  return space == MemSpace::Scratchpad ? "sp" : "dram";
}

/// Whether `strides` is the contiguous layout `shape` implies.
///
/// Shared by the operand and the result printers from version 2, when the
/// result gained strides of its own. The rule it exists for is that strides are
/// printed only when they are **not** the contiguous ones: a stride vector on
/// every operand and every result would double the width of a listing to repeat
/// what the shape already implies, and the cases that matter, the NHWC
/// permutation, the stride 0 broadcast and now the tile of a larger buffer, are
/// exactly the ones that differ.
///
/// **The running product is guarded the way the validator's is, and it has to
/// be.** Everything else in this file runs after `validate()` has approved a
/// program; this does not. `npu-objdump` decodes without validating so that a
/// suspect file can be shown at all, so every extent here is whatever the file
/// claimed, and `expected *= extent` on a claimed extent of nine quintillion is
/// signed overflow rather than a large number. The coverage guided target found
/// that one, which is D-0022.
bool isContiguousLayout(const std::vector<int64_t> &shape,
                        const std::vector<int64_t> &strides) {
  if (shape.size() != strides.size())
    return false;
  int64_t expected = 1;
  for (size_t index = shape.size(); index-- > 0;) {
    if (strides[index] != expected)
      return false;
    int64_t extent = shape[index];
    if (extent <= 0 || expected > Program::kShapeLimit / extent) {
      // Not a shape this machine has an address for, so it is not the
      // contiguous layout either, and the strides get printed.
      return false;
    }
    expected *= extent;
  }
  return true;
}

std::string resultText(const Instruction &instruction) {
  std::string out = spaceAbbreviation(instruction.resultSpace) + "@" +
                    hex(instruction.resultAddress) + " " +
                    shapeText(instruction.resultShape,
                              instruction.resultElementType);
  // An empty stride vector is what an opcode with no result carries, and it is
  // not a layout to describe. Printing `s[]` on every `HALT` would be noise.
  if (!instruction.resultStrides.empty() &&
      !isContiguousLayout(instruction.resultShape, instruction.resultStrides))
    out += " s" + vectorText(instruction.resultStrides);
  return out;
}

std::string operandText(const Operand &operand) {
  std::string out = spaceAbbreviation(operand.space) + "@" +
                    hex(operand.address) + " " +
                    shapeText(operand.shape, operand.elementType);
  if (!isContiguousLayout(operand.shape, operand.strides))
    out += " s" + vectorText(operand.strides);
  return out;
}

/// The value of a named field, or nothing when the name is not a field.
std::optional<std::string> fieldText(const Instruction &instruction,
                                     llvm::StringRef name) {
  if (name == "activation")
    return std::string("activation=") + activationName(instruction.activation);
  if (name == "strides")
    return "strides=" + vectorText(instruction.strides);
  if (name == "pads")
    return "pads=" + vectorText(instruction.pads);
  if (name == "dilations")
    return "dilations=" + vectorText(instruction.dilations);
  if (name == "kernel")
    return "kernel=" + vectorText(instruction.kernel);
  if (name == "axes")
    return "axes=" + vectorText(instruction.axes);
  if (name == "group")
    return "group=" + std::to_string(instruction.group);
  if (name == "zeroPoint")
    return "zeroPoint=" + std::to_string(instruction.zeroPoint);
  // Stored in the scale word and printed as a zero point, which is what it is
  // on the opcodes that declare it. Printing the raw float here would make a
  // reader of a disassembly work out the convention for themselves.
  if (name == "outputZeroPoint")
    return "outputZeroPoint=" +
           std::to_string(static_cast<int32_t>(instruction.scale));
  // The rescaling pair, printed as the two numbers it is. `M0` and the shift
  // are what the machine applies, so a disassembly that showed a zero point
  // being added but not the rescale that produced the value it is added to
  // would be one a reader could not check the arithmetic of.
  if (name == "requantize")
    return "requantMultiplier=" +
           std::to_string(instruction.requantMultiplier) +
           " requantShift=" + std::to_string(instruction.requantShift);
  if (name == "scale") {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "scale=%g",
                  static_cast<double>(instruction.scale));
    return std::string(buffer);
  }
  return std::nullopt;
}

/// Renders one token of a format string.
///
/// Returns false when the token names an operand the instruction does not
/// have. What the caller does with that answer differs by context and the
/// difference is the whole of D-0023: inside a group it drops the group, which
/// is how an absent optional bias disappears; at the top level it must not,
/// because an instruction whose mandatory operand is missing is exactly the
/// instruction somebody is running `npu-objdump` to look at.
bool renderToken(const Instruction &instruction, llvm::StringRef token,
                 std::string &out) {
  if (token.starts_with("%")) {
    llvm::StringRef rest = token.drop_front();
    if (rest == "r") {
      out += resultText(instruction);
      return true;
    }
    if (rest == "n") {
      for (size_t index = 0; index < instruction.operands.size(); ++index) {
        if (index)
          out += ", ";
        out += operandText(instruction.operands[index]);
      }
      return true;
    }
    unsigned position = 0;
    if (rest.getAsInteger(10, position))
      return true;
    if (position >= instruction.operands.size())
      return false;
    out += operandText(instruction.operands[position]);
    return true;
  }

  if (std::optional<std::string> text = fieldText(instruction, token)) {
    out += *text;
    return true;
  }
  out += token;
  return true;
}

/// What a missing mandatory operand prints as.
std::string missingOperandText(llvm::StringRef token) {
  return "<missing operand " + token.drop_front().str() + ">";
}

/// Renders a whitespace separated token list, appending to `out`.
///
/// Returns false when any token named a missing operand. `dropOnMissing`
/// decides what that costs: with it set, nothing is appended and the caller
/// drops the whole group, which is how an absent optional bias disappears
/// without leaving an empty slot; without it, the missing operand is rendered
/// as a placeholder and everything else is still printed.
bool renderTokens(const Instruction &instruction, llvm::StringRef text,
                  std::string &out, bool dropOnMissing) {
  std::string rendered;
  bool complete = true;
  bool first = true;
  while (!text.empty()) {
    auto [token, rest] = text.split(' ');
    text = rest;
    if (token.empty())
      continue;
    // A trailing comma belongs to the separator rather than to the token, so
    // `%0,` is an operand reference followed by punctuation.
    llvm::StringRef punctuation;
    while (!token.empty() && (token.back() == ',' || token.back() == ';')) {
      punctuation = token.take_back();
      token = token.drop_back();
    }
    std::string piece;
    if (!renderToken(instruction, token, piece)) {
      if (dropOnMissing)
        return false;
      complete = false;
      piece = missingOperandText(token);
    }
    if (!first)
      rendered += " ";
    rendered += piece;
    rendered += punctuation;
    first = false;
  }
  out += rendered;
  return complete;
}

/// The fields an opcode gives meaning to only at an integer result, appended
/// after its format string has been rendered.
///
/// **They are not in the format string, and that is why they are appended.** A
/// format string is per opcode and this is per instruction: the same `CONV2D`
/// carries them at an i8 result and must not print them at f32, where they hold
/// their neutral values and would be noise on every line of every disassembly
/// this project has produced. Appending leaves every existing line byte
/// identical, which is what the lit tests compare.
void appendIntegerFields(const Instruction &instruction, const OpcodeInfo &info,
                         std::string &out) {
  const bool rescales = (info.fieldMask & kFieldRequantize) != 0;
  if (info.integerFieldMask == 0 && !rescales)
    return;
  uint32_t rawType = static_cast<uint32_t>(instruction.resultElementType);
  if (rawType >= 32 || (kIntegerTypeMask & (1u << rawType)) == 0)
    return;

  // Named one at a time rather than swept out of the mask, for the reason
  // `fieldText` above names its own: a field added to the description and not
  // to this list is a field that silently stops being disassembled, and a table
  // here is the place that becomes visible.
  //
  // The order is **the order the arithmetic applies them**: the input zero
  // point is what a tap outside the input contributes, the pair rescales the
  // accumulator, and the output zero point is added to what the rescale
  // produced. Read down the line and the line is the computation.
  //
  // The rescaling pair is the one entry that reads the ordinary field mask
  // rather than the integer one, because the opcode declares it in `fields`:
  // an f32 instruction carries the neutral pair and the validator's neutrality
  // rule is written against that list. What is true only at an integer result
  // is that the pair *means* something, which is why it is printed here and
  // nowhere else.
  static constexpr struct {
    uint32_t bit;
    const char *name;
    bool integerOnly;
  } kAtAnIntegerResult[] = {
      {kFieldZeroPoint, "zeroPoint", true},
      {kFieldRequantize, "requantize", false},
      {kFieldOutputZeroPoint, "outputZeroPoint", true},
  };

  for (const auto &field : kAtAnIntegerResult) {
    uint32_t mask = field.integerOnly ? info.integerFieldMask : info.fieldMask;
    if ((mask & field.bit) == 0)
      continue;
    if (std::optional<std::string> text = fieldText(instruction, field.name)) {
      if (!out.empty() && out.back() != ' ')
        out += " ";
      out += *text;
    }
  }
}

std::string renderInstruction(const Instruction &instruction) {
  uint32_t raw = static_cast<uint32_t>(instruction.opcode);
  if (!isKnownOpcode(raw))
    return "<opcode " + std::to_string(raw) + ">";

  llvm::StringRef format = opcodeInfo(instruction.opcode).format;
  std::string out;
  while (!format.empty()) {
    size_t open = format.find('{');
    if (open == llvm::StringRef::npos) {
      renderTokens(instruction, format, out, /*dropOnMissing=*/false);
      break;
    }
    renderTokens(instruction, format.substr(0, open), out,
                 /*dropOnMissing=*/false);
    format = format.drop_front(open + 1);
    size_t close = format.find('}');
    llvm::StringRef group =
        close == llvm::StringRef::npos ? format : format.substr(0, close);
    format = close == llvm::StringRef::npos ? llvm::StringRef()
                                            : format.drop_front(close + 1);
    std::string groupText;
    if (renderTokens(instruction, group, groupText, /*dropOnMissing=*/true) &&
        !groupText.empty()) {
      if (!out.empty() && out.back() != ' ')
        out += " ";
      out += groupText;
    }
  }
  appendIntegerFields(instruction, opcodeInfo(instruction.opcode), out);

  // A dropped group can leave a double space behind. Squeezing here rather
  // than threading the state through the renderer keeps the substitution
  // machinery simple, and the output is compared byte for byte by a lit test
  // so it has to be stable.
  std::string squeezed;
  bool previousSpace = false;
  for (char c : out) {
    if (c == ' ' && previousSpace)
      continue;
    previousSpace = c == ' ';
    squeezed += c;
  }
  while (!squeezed.empty() && squeezed.back() == ' ')
    squeezed.pop_back();
  return squeezed;
}

std::string regionText(const MemRegion &region) {
  int64_t bytes = region.byteSize();
  return "dram@" + hex(static_cast<int64_t>(region.offset)) + " " +
         shapeText(region.shape, region.elementType) + " (" +
         (bytes < 0 ? std::string("shape overflows") : std::to_string(bytes)) +
         " bytes)";
}

} // namespace

std::string nbin::disassemble(const Program &program,
                             const std::optional<ProgramError> &failure) {
  std::string out;

  if (failure) {
    out += "; ======================================================="
           "==================\n";
    out += "; WARNING: this file did not validate. It was decoded without\n";
    out += "; WARNING: validation so that it could be shown at all, and\n";
    out += "; WARNING: everything below may be wrong.\n";
    out += "; WARNING: " + failure->toString() + "\n";
    out += "; ======================================================="
           "==================\n";
  }

  out += "; .nbin version " + std::to_string(Program::kVersion) +
         ", host byte order\n";
  out += "; scratchpad " + std::to_string(program.scratchpadBytes) +
         " bytes, dram " + std::to_string(program.dramBytes) + " bytes\n";
  out += "; " + std::to_string(program.inputs.size()) + " inputs, " +
         std::to_string(program.outputs.size()) + " outputs, " +
         std::to_string(program.constants.size()) + " constants, " +
         std::to_string(program.spillSlots.size()) + " spill slots, " +
         std::to_string(program.instructions.size()) + " instructions, " +
         std::to_string(program.debug.size()) + " debug entries\n";

  for (size_t index = 0; index < program.inputs.size(); ++index)
    out += ";   input " + std::to_string(index) + ": " +
           regionText(program.inputs[index]) + "\n";
  for (size_t index = 0; index < program.outputs.size(); ++index)
    out += ";   output " + std::to_string(index) + ": " +
           regionText(program.outputs[index]) + "\n";
  for (size_t index = 0; index < program.constants.size(); ++index)
    out += ";   constant " + std::to_string(index) + ": " +
           regionText(program.constants[index].region) + ", " +
           std::to_string(program.constants[index].data.size()) +
           " bytes of data\n";
  for (size_t index = 0; index < program.spillSlots.size(); ++index)
    out += ";   spill slot " + std::to_string(index) + ": " +
           regionText(program.spillSlots[index]) + "\n";

  for (size_t index = 0; index < program.instructions.size(); ++index) {
    // Wide enough for a 64 bit index written in full, which the four digit
    // minimum never reaches but the compiler cannot know that.
    char pc[24];
    std::snprintf(pc, sizeof(pc), "%04zu", index);
    out += pc;
    out += "  ";
    out += renderInstruction(program.instructions[index]);
    llvm::StringRef name =
        program.debugNameFor(static_cast<uint32_t>(index));
    if (!name.empty()) {
      out += "    ; ";
      out += name;
    }
    out += "\n";
  }

  return out;
}
