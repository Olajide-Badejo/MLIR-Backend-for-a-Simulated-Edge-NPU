//===- npu-isa-tblgen.cpp - the ISA description generator -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The generator half of Section 9.4. It reads
// include/NPU/Encoding/NPUISADescription.td and writes the layers that would
// otherwise be maintained by hand and would therefore drift: the opcode enum,
// the arity and field presence rules the validator reads, the check name enum,
// the simulator's dispatch skeleton, the manual's generated sections, and the
// opcode list the reachability checker reads.
//
// It is a TableGen driver rather than a script over a hand rolled format for
// one reason worth stating: the project already depends on TableGen, already
// generates its dialect reference that way, and already has a staleness gate
// shaped around that generation. A second mechanism would be a second thing to
// keep working.
//
// Determinism is a requirement, not a nicety. Everything this writes is diffed
// by scripts/check-isa-staleness.sh, so any dependence on hash iteration order
// would turn that gate into a coin toss. Every list below is sorted by an
// explicit key before it is emitted.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// The command line.
//===----------------------------------------------------------------------===//

enum Action {
  GenOpcodeEnum,
  GenOpcodeInfo,
  GenCheckEnum,
  GenDispatchDef,
  GenManual,
  GenJson,
};

cl::opt<Action> action(
    cl::desc("The artifact to generate:"),
    cl::values(
        clEnumValN(GenOpcodeEnum, "gen-opcode-enum",
                   "the Opcode enum, kMaxOpcode and the name table"),
        clEnumValN(GenOpcodeInfo, "gen-opcode-info",
                   "the arity, field presence, space and element type rules"),
        clEnumValN(GenCheckEnum, "gen-check-enum",
                   "the validation check names of Section 9.2"),
        clEnumValN(GenDispatchDef, "gen-dispatch-def",
                   "the simulator's dispatch skeleton, as an X macro"),
        clEnumValN(GenManual, "gen-manual",
                   "the generated sections of docs/ISA_MANUAL.md"),
        clEnumValN(GenJson, "gen-json",
                   "the opcode list scripts/check-reachability.py reads")));

//===----------------------------------------------------------------------===//
// The description, read out of the records.
//===----------------------------------------------------------------------===//

struct FieldDesc {
  StringRef record;
  StringRef name;
  StringRef description;
  uint32_t bit = 0;
};

struct ElemTypeDesc {
  StringRef record;
  StringRef name;
  int64_t value = 0;
};

struct OpcodeDesc {
  StringRef record;
  int64_t value = 0;
  StringRef semantics;
  int64_t minOperands = 0;
  int64_t maxOperands = 0;
  bool hasResult = true;
  std::vector<int64_t> operandSpaces;
  int64_t resultSpace = 0;
  std::vector<StringRef> fields;
  uint32_t fieldMask = 0;
  uint32_t resultTypeMask = 0;
  uint32_t operandTypeMask = 0;
  StringRef shapeRule;
  StringRef format;
  std::vector<StringRef> sources;
  bool needsKernel = true;
  StringRef semanticPhase;
};

struct CheckDesc {
  StringRef record;
  int64_t order = 0;
  StringRef name;
  StringRef description;
};

struct EliminatedDesc {
  StringRef source;
  StringRef explanation;
};

struct Description {
  std::vector<FieldDesc> fields;
  std::vector<ElemTypeDesc> elemTypes;
  std::vector<ElemTypeDesc> memSpaces;
  std::vector<ElemTypeDesc> activations;
  std::vector<OpcodeDesc> opcodes;
  std::vector<CheckDesc> checks;
  std::vector<EliminatedDesc> eliminated;
  std::map<StringRef, uint32_t> fieldBits;
  std::map<StringRef, int64_t> elemTypeValues;
};

/// The C++ enumerator spelling of a check, which is its record name with the
/// `Check` prefix removed. `CheckCountCap` becomes `CountCap`.
std::string enumeratorFor(StringRef record) {
  StringRef stripped = record;
  stripped.consume_front("Check");
  return stripped.str();
}

/// Reads the simple `{ int value; string name; }` classes, sorted by value.
std::vector<ElemTypeDesc> readValueNamePairs(const RecordKeeper &records,
                                             StringRef className) {
  std::vector<ElemTypeDesc> out;
  for (const Record *rec : records.getAllDerivedDefinitions(className))
    out.push_back({rec->getName(), rec->getValueAsString("name"),
                   rec->getValueAsInt("value")});
  llvm::sort(out, [](const ElemTypeDesc &a, const ElemTypeDesc &b) {
    return a.value < b.value;
  });
  return out;
}

Description readDescription(const RecordKeeper &records) {
  Description desc;

  // Fields first: the bit assignment is by sorted record name, which is stable
  // under any edit that does not rename a field.
  for (const Record *rec : records.getAllDerivedDefinitions("ISAField"))
    desc.fields.push_back(
        {rec->getName(), rec->getValueAsString("name"),
         rec->getValueAsString("description"), 0});
  llvm::sort(desc.fields, [](const FieldDesc &a, const FieldDesc &b) {
    return a.record < b.record;
  });
  if (desc.fields.size() > 32)
    PrintFatalError("more than 32 ISAField records: the field presence mask in "
                    "the generated OpcodeInfo is a uint32_t and would silently "
                    "drop the ones past the end");
  for (auto [index, field] : llvm::enumerate(desc.fields)) {
    field.bit = 1u << index;
    desc.fieldBits[field.record] = field.bit;
  }

  desc.elemTypes = readValueNamePairs(records, "ISAElemType");
  desc.memSpaces = readValueNamePairs(records, "ISAMemSpace");
  desc.activations = readValueNamePairs(records, "ISAActivation");
  for (const ElemTypeDesc &type : desc.elemTypes)
    desc.elemTypeValues[type.record] = type.value;

  if (desc.elemTypes.size() > 32)
    PrintFatalError("more than 32 ISAElemType records: the element type mask "
                    "in the generated OpcodeInfo is a uint32_t");

  // The opcodes.
  for (const Record *rec : records.getAllDerivedDefinitions("ISAOpcode")) {
    OpcodeDesc op;
    op.record = rec->getName();
    op.value = rec->getValueAsInt("value");
    op.semantics = rec->getValueAsString("semantics");
    op.minOperands = rec->getValueAsInt("minOperands");
    op.maxOperands = rec->getValueAsInt("maxOperands");
    op.hasResult = rec->getValueAsBit("hasResult");
    op.resultSpace = rec->getValueAsDef("resultSpace")->getValueAsInt("value");
    op.shapeRule = rec->getValueAsString("shapeRule");
    op.format = rec->getValueAsString("format");
    op.needsKernel = rec->getValueAsBit("needsKernel");
    op.semanticPhase = rec->getValueAsString("semanticPhase");

    for (const Record *space : rec->getValueAsListOfDefs("operandSpaces"))
      op.operandSpaces.push_back(space->getValueAsInt("value"));
    for (const Record *field : rec->getValueAsListOfDefs("fields")) {
      op.fields.push_back(field->getValueAsString("name"));
      op.fieldMask |= desc.fieldBits[field->getName()];
    }
    for (const Record *type : rec->getValueAsListOfDefs("elementTypes"))
      op.resultTypeMask |= 1u << desc.elemTypeValues[type->getName()];
    for (const Record *type : rec->getValueAsListOfDefs("operandTypes"))
      op.operandTypeMask |= 1u << desc.elemTypeValues[type->getName()];
    for (StringRef source : rec->getValueAsListOfStrings("sources"))
      op.sources.push_back(source);
    llvm::sort(op.sources);

    // The rules that make a halfway opcode a generator error rather than a
    // runtime surprise, which is the whole claim of Section 9.4.
    if (op.value < 0)
      PrintFatalError("opcode " + op.record.str() + " has a negative value");
    if (op.minOperands < 0)
      PrintFatalError("opcode " + op.record.str() + " has a negative "
                      "minOperands");
    if (op.maxOperands >= 0 && op.maxOperands < op.minOperands)
      PrintFatalError("opcode " + op.record.str() + " has maxOperands below "
                      "minOperands");
    if (op.format.empty())
      PrintFatalError("opcode " + op.record.str() + " has no format string, so "
                      "the disassembler would have nothing to print");
    if (op.semantics.empty())
      PrintFatalError("opcode " + op.record.str() + " has no semantics line, "
                      "so the manual would have a blank row");
    if (op.shapeRule.empty())
      PrintFatalError("opcode " + op.record.str() + " has no shapeRule, so the "
                      "manual would not say how its operands relate to its "
                      "result");
    if (op.resultTypeMask == 0)
      PrintFatalError("opcode " + op.record.str() + " accepts no element type "
                      "at all, so no instruction using it could ever validate");
    {
      size_t needed = op.minOperands == 0 ? 0 : 1;
      needed = std::max<size_t>(
          needed, op.maxOperands < 0 ? static_cast<size_t>(op.minOperands)
                                     : static_cast<size_t>(op.maxOperands));
      if (op.operandSpaces.size() < needed)
        PrintFatalError("opcode " + op.record.str() + " declares " +
                        std::to_string(op.operandSpaces.size()) +
                        " operand spaces but takes up to " +
                        std::to_string(needed) +
                        " operands, so an operand would have no declared "
                        "memory space");
    }
    desc.opcodes.push_back(op);
  }
  llvm::sort(desc.opcodes, [](const OpcodeDesc &a, const OpcodeDesc &b) {
    return a.value < b.value;
  });
  for (size_t i = 1; i < desc.opcodes.size(); ++i)
    if (desc.opcodes[i].value == desc.opcodes[i - 1].value)
      PrintFatalError("opcodes " + desc.opcodes[i - 1].record.str() + " and " +
                      desc.opcodes[i].record.str() +
                      " share the numeric value " +
                      std::to_string(desc.opcodes[i].value) +
                      ". Opcode values are assigned once and never reused");

  // The checks.
  for (const Record *rec : records.getAllDerivedDefinitions("ISACheck"))
    desc.checks.push_back({rec->getName(), rec->getValueAsInt("order"),
                           rec->getValueAsString("name"),
                           rec->getValueAsString("description")});
  llvm::sort(desc.checks, [](const CheckDesc &a, const CheckDesc &b) {
    return a.order < b.order;
  });
  for (size_t i = 1; i < desc.checks.size(); ++i)
    if (desc.checks[i].order == desc.checks[i - 1].order)
      PrintFatalError("checks " + desc.checks[i - 1].record.str() + " and " +
                      desc.checks[i].record.str() + " share the order " +
                      std::to_string(desc.checks[i].order) +
                      ", so the generated enum would not be deterministic");

  // The `npu` operations that reach the encoder by elimination.
  for (const Record *rec :
       records.getAllDerivedDefinitions("ISAEliminatedSource"))
    desc.eliminated.push_back({rec->getValueAsString("source"),
                               rec->getValueAsString("explanation")});
  llvm::sort(desc.eliminated,
             [](const EliminatedDesc &a, const EliminatedDesc &b) {
               return a.source < b.source;
             });

  return desc;
}

//===----------------------------------------------------------------------===//
// The shared banner.
//===----------------------------------------------------------------------===//

void emitBanner(raw_ostream &os, StringRef comment) {
  // The blank line inside the banner is the bare comment marker with no
  // trailing space. The pre-commit trailing whitespace hook would flag one,
  // and a generated file that cannot pass the project's own hooks is a
  // generated file somebody has to remember to exclude.
  StringRef bare = comment.rtrim();
  os << comment << "===----------------------------------------------------"
     << "------------------===\n";
  os << comment << "THIS FILE IS GENERATED. Do not edit it.\n";
  os << bare << "\n";
  os << comment << "It is produced from include/NPU/Encoding/"
     << "NPUISADescription.td by\n";
  os << comment << "npu-isa-tblgen. Change the description, not this file.\n";
  os << comment << "===----------------------------------------------------"
     << "------------------===\n\n";
}

//===----------------------------------------------------------------------===//
// -gen-opcode-enum
//===----------------------------------------------------------------------===//

void emitOpcodeEnum(raw_ostream &os, const Description &desc) {
  emitBanner(os, "// ");

  os << "/// The instruction set of Section 5.4. The numeric values are "
        "assigned\n";
  os << "/// once and are never renumbered; new opcodes are appended.\n";
  os << "enum class Opcode : uint32_t {\n";
  for (const OpcodeDesc &op : desc.opcodes)
    os << "  " << op.record << " = " << op.value << ",\n";
  os << "};\n\n";

  os << "/// The largest numeric opcode value the description names.\n";
  os << "inline constexpr uint32_t kMaxOpcode = "
     << desc.opcodes.back().value << ";\n\n";
  os << "/// How many opcodes the description names.\n";
  os << "inline constexpr uint32_t kNumOpcodes = " << desc.opcodes.size()
     << ";\n\n";

  os << "/// Whether a raw word from a file is an opcode this build knows.\n";
  os << "constexpr bool isKnownOpcode(uint32_t raw) {\n";
  os << "  switch (raw) {\n";
  for (const OpcodeDesc &op : desc.opcodes)
    os << "  case " << op.value << ":\n";
  os << "    return true;\n";
  os << "  default:\n";
  os << "    return false;\n";
  os << "  }\n";
  os << "}\n\n";

  os << "/// The spelling of an opcode. Never null: an unknown value prints as\n";
  os << "/// `<opcode N>` so that a disassembly of a corrupt file still "
        "reads.\n";
  os << "constexpr const char *opcodeName(Opcode op) {\n";
  os << "  switch (op) {\n";
  for (const OpcodeDesc &op : desc.opcodes)
    os << "  case Opcode::" << op.record << ":\n    return \"" << op.record
       << "\";\n";
  os << "  }\n";
  os << "  return \"<unknown opcode>\";\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// -gen-opcode-info
//===----------------------------------------------------------------------===//

void emitStringLiteral(raw_ostream &os, StringRef text) {
  os << '"';
  for (char c : text) {
    if (c == '"' || c == '\\')
      os << '\\';
    os << c;
  }
  os << '"';
}

void emitOpcodeInfo(raw_ostream &os, const Description &desc) {
  emitBanner(os, "// ");

  os << "/// A bit per field of the `Instruction` record that an opcode can "
        "give\n";
  os << "/// meaning to. A field an opcode does not name must hold its neutral "
        "value,\n";
  os << "/// which is what makes `attribute-size` and `attribute-value` "
        "checkable.\n";
  os << "enum : uint32_t {\n";
  for (const FieldDesc &field : desc.fields)
    os << "  kField" << field.record.substr(strlen("Field")) << " = 0x"
       << utohexstr(field.bit) << "u, // " << field.name << "\n";
  os << "};\n\n";

  os << "/// The mechanical half of an opcode's rules. Section 9.4: the arity "
        "and\n";
  os << "/// field presence rules in validate() are generated, leaving only "
        "the\n";
  os << "/// genuinely semantic checks hand written.\n";
  os << "struct OpcodeInfo {\n";
  os << "  const char *name;\n";
  os << "  uint32_t value;\n";
  os << "  int32_t minOperands;\n";
  os << "  /// -1 means variadic, bounded only by kMaxCount.\n";
  os << "  int32_t maxOperands;\n";
  os << "  bool hasResult;\n";
  os << "  /// The memory space the result lives in, as a MemSpace value.\n";
  os << "  uint32_t resultSpace;\n";
  os << "  /// One MemSpace value per operand slot. The last entry repeats "
        "for\n";
  os << "  /// every operand a variadic opcode has beyond the ones listed.\n";
  os << "  const uint32_t *operandSpaces;\n";
  os << "  uint32_t numOperandSpaces;\n";
  os << "  /// The fields this opcode gives meaning to, as a mask of kField*.\n";
  os << "  uint32_t fieldMask;\n";
  os << "  /// A bit per ElemType value the result may take.\n";
  os << "  uint32_t resultTypeMask;\n";
  os << "  /// A bit per ElemType value the operands may take, or 0 when the\n";
  os << "  /// operands take the result's type.\n";
  os << "  uint32_t operandTypeMask;\n";
  os << "  const char *format;\n";
  os << "  const char *semantics;\n";
  os << "  const char *shapeRule;\n";
  os << "  bool needsKernel;\n";
  os << "  const char *semanticPhase;\n";
  os << "};\n\n";

  for (const OpcodeDesc &op : desc.opcodes) {
    if (op.operandSpaces.empty())
      continue;
    os << "inline constexpr uint32_t kOperandSpaces" << op.record << "[] = {";
    for (auto [index, space] : llvm::enumerate(op.operandSpaces))
      os << (index ? ", " : "") << space << "u";
    os << "};\n";
  }
  os << "\n";

  os << "inline constexpr OpcodeInfo kOpcodeTable[] = {\n";
  for (const OpcodeDesc &op : desc.opcodes) {
    os << "    {";
    emitStringLiteral(os, op.record);
    os << ", " << op.value << "u, " << op.minOperands << ", " << op.maxOperands
       << ", " << (op.hasResult ? "true" : "false") << ", " << op.resultSpace
       << "u, ";
    if (op.operandSpaces.empty())
      os << "nullptr, 0u, ";
    else
      os << "kOperandSpaces" << op.record << ", " << op.operandSpaces.size()
         << "u, ";
    os << "0x" << utohexstr(op.fieldMask) << "u, 0x"
       << utohexstr(op.resultTypeMask) << "u, 0x"
       << utohexstr(op.operandTypeMask) << "u,\n     ";
    emitStringLiteral(os, op.format);
    os << ",\n     ";
    emitStringLiteral(os, op.semantics);
    os << ",\n     ";
    emitStringLiteral(os, op.shapeRule);
    os << ",\n     " << (op.needsKernel ? "true" : "false") << ", ";
    emitStringLiteral(os, op.semanticPhase);
    os << "},\n";
  }
  os << "};\n\n";

  os << "static_assert(sizeof(kOpcodeTable) / sizeof(kOpcodeTable[0]) == "
     << desc.opcodes.size() << ",\n";
  os << "              \"the opcode table lost a row\");\n\n";

  os << "/// The rules for one opcode. The argument is a known opcode: callers "
        "reach\n";
  os << "/// this only after `isKnownOpcode` has said so, which is what the\n";
  os << "/// `structure` check does before anything else looks at an "
        "instruction.\n";
  os << "inline const OpcodeInfo &opcodeInfo(Opcode op) {\n";
  os << "  return kOpcodeTable[static_cast<uint32_t>(op)];\n";
  os << "}\n\n";

  os << "// The table is indexed by opcode value, so a gap in the numbering "
        "would\n";
  os << "// misindex it silently. The values are contiguous today and this "
        "asserts\n";
  os << "// it at compile time rather than trusting it.\n";
  for (auto [index, op] : llvm::enumerate(desc.opcodes))
    os << "static_assert(kOpcodeTable[" << index << "].value == " << op.value
       << "u, \"opcode table is not indexed by value\");\n";
}

//===----------------------------------------------------------------------===//
// -gen-check-enum
//===----------------------------------------------------------------------===//

void emitCheckEnum(raw_ostream &os, const Description &desc) {
  emitBanner(os, "// ");

  os << "/// The validation checks of Section 9.2. Every failure returns one "
        "of\n";
  os << "/// these, and the stable name below is what the message carries, so "
        "a\n";
  os << "/// test can assert the check rather than the wording.\n";
  os << "enum class Check : uint32_t {\n";
  for (const CheckDesc &check : desc.checks)
    os << "  " << enumeratorFor(check.record) << " = " << check.order
       << ", // " << check.name << "\n";
  os << "};\n\n";

  os << "inline constexpr uint32_t kNumChecks = " << desc.checks.size()
     << ";\n\n";

  os << "/// The stable name of a check, as it appears in every message.\n";
  os << "constexpr const char *checkName(Check check) {\n";
  os << "  switch (check) {\n";
  for (const CheckDesc &check : desc.checks)
    os << "  case Check::" << enumeratorFor(check.record) << ":\n    return \""
       << check.name << "\";\n";
  os << "  }\n";
  os << "  return \"<unknown check>\";\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// -gen-dispatch-def
//===----------------------------------------------------------------------===//

void emitDispatchDef(raw_ostream &os, const Description &desc) {
  emitBanner(os, "// ");

  os << "// The simulator's dispatch skeleton, as an X macro.\n";
  os << "//\n";
  os << "// Section 9.4 asks for a new opcode with no kernel to be a build "
        "error\n";
  os << "// rather than a runtime surprise. Two mechanisms together give "
        "that. A\n";
  os << "// hand written switch over `Opcode` with no `default` label is a "
        "build\n";
  os << "// error under -Werror=switch the moment the enum grows, which is "
        "how the\n";
  os << "// encoder, the validator and the disassembler notice. This file is "
        "how a\n";
  os << "// table driven consumer notices: expand it, and a row with no case "
        "is a\n";
  os << "// missing symbol at link time rather than a default that quietly "
        "does\n";
  os << "// nothing.\n";
  os << "//\n";
  os << "// Define NPU_ISA_OPCODE(NAME, VALUE, NEEDS_KERNEL) before "
        "including.\n";
  os << "// NEEDS_KERNEL is 0 for the control opcodes, which are not "
        "computation\n";
  os << "// and have no kernel to write.\n\n";

  os << "#ifndef NPU_ISA_OPCODE\n";
  os << "#error \"define NPU_ISA_OPCODE(NAME, VALUE, NEEDS_KERNEL) before "
        "including this file\"\n";
  os << "#endif\n\n";
  for (const OpcodeDesc &op : desc.opcodes)
    os << "NPU_ISA_OPCODE(" << op.record << ", " << op.value << ", "
       << (op.needsKernel ? 1 : 0) << ")\n";
  os << "\n#undef NPU_ISA_OPCODE\n";
}

//===----------------------------------------------------------------------===//
// -gen-manual
//===----------------------------------------------------------------------===//

std::string spaceName(const Description &desc, int64_t value) {
  for (const ElemTypeDesc &space : desc.memSpaces)
    if (space.value == value)
      return space.name.str();
  return "?";
}

std::string typeMaskNames(const Description &desc, uint32_t mask) {
  std::string out;
  for (const ElemTypeDesc &type : desc.elemTypes)
    if (mask & (1u << type.value)) {
      if (!out.empty())
        out += ", ";
      out += type.name.str();
    }
  return out.empty() ? "(none)" : out;
}

std::string arityText(const OpcodeDesc &op) {
  if (op.maxOperands < 0)
    return std::to_string(op.minOperands) + " or more";
  if (op.minOperands == op.maxOperands)
    return std::to_string(op.minOperands);
  return std::to_string(op.minOperands) + " or " +
         std::to_string(op.maxOperands);
}

void emitManual(raw_ostream &os, const Description &desc) {
  os << "<!-- BEGIN GENERATED: opcode table -->\n\n";
  os << "<!--\nTHIS SECTION IS GENERATED from include/NPU/Encoding/"
        "NPUISADescription.td by\n\n    ninja -C build npu-isa-doc\n\n"
        "and scripts/check-isa-staleness.sh regenerates it and diffs, so an "
        "edit here\nis reverted by the next build and reported as staleness by "
        "the next run.\n-->\n\n";

  os << "| Opcode | Value | Operands | Result | Element types | Fields | "
        "Semantics |\n";
  os << "|---|---|---|---|---|---|---|\n";
  for (const OpcodeDesc &op : desc.opcodes) {
    os << "| `" << op.record << "` | " << op.value << " | ";
    if (op.maxOperands == 0)
      os << "none";
    else
      os << arityText(op) << " in "
         << spaceName(desc, op.operandSpaces.front());
    os << " | ";
    if (op.hasResult)
      os << spaceName(desc, op.resultSpace);
    else
      os << "none";
    os << " | ";
    if (!op.hasResult) {
      // An opcode with no result accepts no element type either, and printing
      // the mask's default here would claim it was an f32 instruction.
      os << "n/a";
    } else {
      os << typeMaskNames(desc, op.resultTypeMask);
      if (op.operandTypeMask)
        os << " (operands " << typeMaskNames(desc, op.operandTypeMask) << ")";
    }
    os << " | ";
    if (op.fields.empty()) {
      os << "none";
    } else {
      for (auto [index, field] : llvm::enumerate(op.fields))
        os << (index ? ", " : "") << "`" << field << "`";
    }
    os << " | " << op.semantics << " |\n";
  }
  os << "\n";

  os << "### Shape relations and disassembly\n\n";
  os << "| Opcode | Shape relation | Disassembly | Semantics land at |\n";
  os << "|---|---|---|---|\n";
  for (const OpcodeDesc &op : desc.opcodes)
    os << "| `" << op.record << "` | " << op.shapeRule << " | `" << op.format
       << "` | " << op.semanticPhase << " |\n";
  os << "\n";

  os << "### Which `npu` operation reaches which opcode\n\n";
  os << "| `npu` operation | Reaches |\n";
  os << "|---|---|\n";
  {
    std::map<std::string, std::vector<std::string>> bySource;
    for (const OpcodeDesc &op : desc.opcodes)
      for (StringRef source : op.sources)
        bySource[source.str()].push_back(op.record.str());
    for (const auto &[source, opcodes] : bySource) {
      os << "| `npu." << source << "` | ";
      for (auto [index, name] : llvm::enumerate(opcodes))
        os << (index ? ", " : "") << "`" << name << "`";
      os << " |\n";
    }
    for (const EliminatedDesc &entry : desc.eliminated)
      os << "| `npu." << entry.source << "` | no instruction: "
         << entry.explanation << " |\n";
  }
  os << "\n";

  os << "### Element types\n\n";
  os << "| Name | Value | Bytes per element |\n";
  os << "|---|---|---|\n";
  for (const ElemTypeDesc &type : desc.elemTypes) {
    int bytes = type.name == "i8" ? 1 : 4;
    os << "| `" << type.name << "` | " << type.value << " | " << bytes
       << " |\n";
  }
  os << "\n";

  os << "### Memory spaces\n\n";
  os << "| Name | Value |\n";
  os << "|---|---|\n";
  for (const ElemTypeDesc &space : desc.memSpaces)
    os << "| `" << space.name << "` | " << space.value << " |\n";
  os << "\n";

  os << "### Activations\n\n";
  os << "| Name | Value |\n";
  os << "|---|---|\n";
  for (const ElemTypeDesc &act : desc.activations)
    os << "| `" << act.name << "` | " << act.value << " |\n";
  os << "\n";

  os << "<!-- END GENERATED: opcode table -->\n";

  os << "\n<!-- BEGIN GENERATED: validation checks -->\n\n";
  os << "<!--\nTHIS SECTION IS GENERATED from the ISACheck records in\n"
        "include/NPU/Encoding/NPUISADescription.td. Section 9.2 asks for the "
        "manual's\ncheck names to be taken from the source rather than copied, "
        "and this is\nthat: the C++ enum and this table come out of the same "
        "records.\n-->\n\n";
  os << "| Check | What it asserts |\n";
  os << "|---|---|\n";
  for (const CheckDesc &check : desc.checks)
    os << "| `" << check.name << "` | " << check.description << " |\n";
  os << "\n<!-- END GENERATED: validation checks -->\n";
}

//===----------------------------------------------------------------------===//
// -gen-json
//===----------------------------------------------------------------------===//

void emitJsonString(raw_ostream &os, StringRef text) {
  os << '"';
  for (char c : text) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\n':
      os << "\\n";
      break;
    default:
      os << c;
    }
  }
  os << '"';
}

void emitJson(raw_ostream &os, const Description &desc) {
  os << "{\n";
  os << "  \"generated_by\": \"npu-isa-tblgen -gen-json from "
        "include/NPU/Encoding/NPUISADescription.td\",\n";
  os << "  \"do_not_edit\": \"Regenerate with: ninja -C build npu-isa-doc\",\n";
  os << "  \"max_opcode\": " << desc.opcodes.back().value << ",\n";
  os << "  \"opcodes\": [\n";
  for (auto [index, op] : llvm::enumerate(desc.opcodes)) {
    os << "    {\"name\": ";
    emitJsonString(os, op.record);
    os << ", \"value\": " << op.value << ", \"sources\": [";
    for (auto [sourceIndex, source] : llvm::enumerate(op.sources)) {
      os << (sourceIndex ? ", " : "");
      emitJsonString(os, source);
    }
    // `needs_kernel` is emitted from P8 for scripts/check-reachability.py's
    // simulation layer. It says whether this opcode is computation the
    // simulator must implement, as opposed to control it merely sequences. The
    // guarantee that a computation opcode *has* its kernel is the compiler's
    // rather than the script's: the generated dispatch table expands to a
    // missing identifier and a failed static assertion when one is absent,
    // which Section 9.4 asks for and P7 demonstrated in four places at once.
    os << "], \"needs_kernel\": " << (op.needsKernel ? "true" : "false");
    os << ", \"semantic_phase\": ";
    emitJsonString(os, op.semanticPhase);
    os << "}" << (index + 1 == desc.opcodes.size() ? "" : ",") << "\n";
  }
  os << "  ],\n";
  os << "  \"eliminated_sources\": [\n";
  for (auto [index, entry] : llvm::enumerate(desc.eliminated)) {
    os << "    {\"source\": ";
    emitJsonString(os, entry.source);
    os << ", \"explanation\": ";
    emitJsonString(os, entry.explanation);
    os << "}" << (index + 1 == desc.eliminated.size() ? "" : ",") << "\n";
  }
  os << "  ],\n";
  os << "  \"checks\": [\n";
  for (auto [index, check] : llvm::enumerate(desc.checks)) {
    os << "    ";
    emitJsonString(os, check.name);
    os << (index + 1 == desc.checks.size() ? "" : ",") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// The entry point.
//===----------------------------------------------------------------------===//

bool emit(raw_ostream &os, const RecordKeeper &records) {
  Description desc = readDescription(records);
  if (desc.opcodes.empty())
    PrintFatalError("the description names no opcode at all. Either the file "
                    "is empty or the ISAOpcode class was renamed, and the "
                    "second is worse than the first because it generates an "
                    "empty instruction set that compiles");

  switch (action) {
  case GenOpcodeEnum:
    emitOpcodeEnum(os, desc);
    break;
  case GenOpcodeInfo:
    emitOpcodeInfo(os, desc);
    break;
  case GenCheckEnum:
    emitCheckEnum(os, desc);
    break;
  case GenDispatchDef:
    emitDispatchDef(os, desc);
    break;
  case GenManual:
    emitManual(os, desc);
    break;
  case GenJson:
    emitJson(os, desc);
    break;
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM lifetime(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &emit);
}
