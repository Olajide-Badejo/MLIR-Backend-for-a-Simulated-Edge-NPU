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

std::string resultText(const Instruction &instruction) {
  return spaceAbbreviation(instruction.resultSpace) + "@" +
         hex(instruction.resultAddress) + " " +
         shapeText(instruction.resultShape, instruction.resultElementType);
}

std::string operandText(const Operand &operand) {
  std::string out = spaceAbbreviation(operand.space) + "@" +
                    hex(operand.address) + " " +
                    shapeText(operand.shape, operand.elementType);
  // The strides are printed only when they are not the contiguous ones. A
  // stride vector on every operand would triple the width of a listing to
  // repeat what the shape already implies, and the cases that matter, the NHWC
  // permutation and the stride 0 broadcast, are exactly the ones that differ.
  bool contiguous = operand.shape.size() == operand.strides.size();
  if (contiguous) {
    int64_t expected = 1;
    for (size_t index = operand.shape.size(); index-- > 0;) {
      if (operand.strides[index] != expected) {
        contiguous = false;
        break;
      }
      expected *= operand.shape[index];
    }
  }
  if (!contiguous)
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
  if (name == "scale") {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "scale=%g",
                  static_cast<double>(instruction.scale));
    return std::string(buffer);
  }
  return std::nullopt;
}

/// Renders one token of a format string. Returns false when the token names an
/// operand the instruction does not have, which drops the group it is in.
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

/// Renders a whitespace separated token list, appending to `out`. Returns
/// false when any token names a missing operand.
bool renderTokens(const Instruction &instruction, llvm::StringRef text,
                  std::string &out) {
  std::string rendered;
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
    if (!renderToken(instruction, token, piece))
      return false;
    if (!first)
      rendered += " ";
    rendered += piece;
    rendered += punctuation;
    first = false;
  }
  out += rendered;
  return true;
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
      renderTokens(instruction, format, out);
      break;
    }
    renderTokens(instruction, format.substr(0, open), out);
    format = format.drop_front(open + 1);
    size_t close = format.find('}');
    llvm::StringRef group =
        close == llvm::StringRef::npos ? format : format.substr(0, close);
    format = close == llvm::StringRef::npos ? llvm::StringRef()
                                            : format.drop_front(close + 1);
    std::string groupText;
    if (renderTokens(instruction, group, groupText) && !groupText.empty()) {
      if (!out.empty() && out.back() != ' ')
        out += " ";
      out += groupText;
    }
  }
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
