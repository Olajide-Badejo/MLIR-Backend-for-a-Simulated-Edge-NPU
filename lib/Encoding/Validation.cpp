//===- Validation.cpp - Program::validate() -------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 9.2 in full. Every rule has a stable check name that comes from the
// ISA description, so a test asserts the check rather than the wording and a
// reworded message is not a broken test.
//
// This runs twice on the way to execution, by design. `Program::decode` calls
// it, and `npu-sim` calls it again before it runs anything, because the two
// calls guard different moments: a program that arrived through a path which
// skipped the first still meets the second.
//
// Three rules from Section 9.2 are easy to get subtly wrong and each is
// implemented here in the form the specification insists on rather than the
// form that reads more naturally:
//
//   1. Shape arithmetic tests `extent > kShapeLimit / product` **before**
//      multiplying. `checkedElementCount` in Program.cpp owns that and nothing
//      here multiplies extents itself.
//   2. Operand checking records each written buffer's byte span and requires
//      the consumer's span to fit inside it. Membership alone is not enough: a
//      DMA_STORE reading a hundred elements from a four element buffer passes
//      a membership test and then traps.
//   3. Every length is bounded before anything is allocated. The decoder owns
//      that; this file re-checks the container sizes because validate() is
//      also called on a Program that was built in memory rather than decoded.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>

using namespace nbin;

namespace {

/// The byte spans that have been written, per memory space.
///
/// Section 9.2 rule 2. Spans are kept distinct rather than merged, and that is
/// the load bearing decision: merging two adjacent buffers into one range
/// would let an over read that runs off the end of the first and into the
/// second pass validation, which is precisely the case the rule exists to
/// catch.
class WrittenSpans {
public:
  void write(MemSpace space, int64_t begin, int64_t size) {
    if (size <= 0 || begin < 0)
      return;
    // A later write at the same address replaces the earlier one rather than
    // widening it. A program that wrote four elements at an address and then
    // reads a hundred from it is reading stale bytes beyond the write, which
    // is the interior over read the rule names.
    spans(space)[begin] = begin + size;
  }

  /// The end of the span containing `address`, or nothing when no write
  /// covers that byte at all.
  std::optional<int64_t> spanEndAt(MemSpace space, int64_t address) const {
    auto it = maps.find(static_cast<uint32_t>(space));
    if (it == maps.end())
      return std::nullopt;
    const auto &map = it->second;
    auto entry = map.upper_bound(address);
    if (entry == map.begin())
      return std::nullopt;
    --entry;
    if (address >= entry->first && address < entry->second)
      return entry->second;
    return std::nullopt;
  }

private:
  std::map<int64_t, int64_t> &spans(MemSpace space) {
    return maps[static_cast<uint32_t>(space)];
  }
  std::map<uint32_t, std::map<int64_t, int64_t>> maps;
};

/// Collects the first failure. Every check below asks this to record and the
/// caller stops at the first one, because a validator that reported everything
/// would report a hundred consequences of one truncated shape.
struct Validator {
  const Program &program;
  std::optional<ProgramError> failure;

  explicit Validator(const Program &program) : program(program) {}

  bool fail(Check check, std::string detail, int64_t instructionIndex = -1,
            int64_t elementIndex = -1, llvm::StringRef where = {}) {
    if (!failure)
      failure = ProgramError{check, std::move(detail), instructionIndex,
                             elementIndex, where.str()};
    return false;
  }
};

/// The size of the memory a space names.
uint64_t memorySize(const Program &program, MemSpace space) {
  return space == MemSpace::Scratchpad ? program.scratchpadBytes
                                       : program.dramBytes;
}

/// Whether a byte span fits inside a memory of `size` bytes.
bool fitsInMemory(int64_t address, int64_t span, uint64_t size) {
  if (address < 0 || span < 0)
    return false;
  if (static_cast<uint64_t>(address) > size)
    return false;
  return static_cast<uint64_t>(span) <= size - static_cast<uint64_t>(address);
}

std::string shapeText(const std::vector<int64_t> &shape) {
  std::string out = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index)
      out += ", ";
    out += std::to_string(shape[index]);
  }
  out += "]";
  return out;
}

//===----------------------------------------------------------------------===//
// The memory regions.
//===----------------------------------------------------------------------===//

bool checkRegion(Validator &validator, const MemRegion &region,
                 const std::string &what, int64_t index,
                 WrittenSpans *defined) {
  if (!isKnownElemType(static_cast<uint32_t>(region.elementType)))
    return validator.fail(
        Check::ElementType,
        "the element type is " +
            std::to_string(static_cast<uint32_t>(region.elementType)) +
            ", which this build does not know",
        -1, index, what);

  int64_t count = region.elementCount();
  if (count < 0)
    return validator.fail(Check::RegionShape,
                          "the shape " + shapeText(region.shape) +
                              " is empty, holds a non positive extent, or has "
                              "a product above the shape limit",
                          -1, index, what);

  int64_t bytes = region.byteSize();
  if (bytes < 0)
    return validator.fail(Check::RegionShape,
                          "the byte size of shape " + shapeText(region.shape) +
                              " at " + elemTypeName(region.elementType) +
                              " is above the shape limit",
                          -1, index, what);

  if (region.offset > static_cast<uint64_t>(INT64_MAX) - static_cast<uint64_t>(bytes))
    return validator.fail(Check::RegionOffset,
                          "the DRAM offset " + std::to_string(region.offset) +
                              " plus the region's " + std::to_string(bytes) +
                              " bytes overflows",
                          -1, index, what);

  if (!fitsInMemory(static_cast<int64_t>(region.offset), bytes,
                    validator.program.dramBytes))
    return validator.fail(Check::RegionInRange,
                          "the region runs from " +
                              std::to_string(region.offset) + " for " +
                              std::to_string(bytes) +
                              " bytes, past the declared DRAM size of " +
                              std::to_string(validator.program.dramBytes),
                          -1, index, what);

  if (defined)
    defined->write(MemSpace::Dram, static_cast<int64_t>(region.offset), bytes);
  return true;
}

bool checkRegions(Validator &validator, WrittenSpans &defined) {
  const Program &program = validator.program;

  // Inputs and constants hold data before the first instruction runs. Outputs
  // and spill slots do not: they are places to write, and a program that reads
  // one before writing it is reading whatever the loader left there, which is
  // the `operand-defined` failure.
  for (const auto &[index, region] : llvm::enumerate(program.inputs))
    if (!checkRegion(validator, region, "input region",
                     static_cast<int64_t>(index), &defined))
      return false;
  for (const auto &[index, region] : llvm::enumerate(program.outputs))
    if (!checkRegion(validator, region, "output region",
                     static_cast<int64_t>(index), nullptr))
      return false;
  for (const auto &[index, constant] : llvm::enumerate(program.constants)) {
    if (!checkRegion(validator, constant.region, "constant region",
                     static_cast<int64_t>(index), &defined))
      return false;
    int64_t bytes = constant.region.byteSize();
    if (static_cast<int64_t>(constant.data.size()) != bytes)
      return validator.fail(
          Check::ConstantData,
          "the constant carries " + std::to_string(constant.data.size()) +
              " bytes of data and its region is " + std::to_string(bytes) +
              " bytes at " + elemTypeName(constant.region.elementType),
          -1, static_cast<int64_t>(index), "constant");
  }
  for (const auto &[index, region] : llvm::enumerate(program.spillSlots))
    if (!checkRegion(validator, region, "spill slot",
                     static_cast<int64_t>(index), nullptr))
      return false;
  return true;
}

//===----------------------------------------------------------------------===//
// One instruction.
//===----------------------------------------------------------------------===//

/// The vector field rules that are generated: which fields an opcode gives
/// meaning to, and how long each of them is when it does.
struct VectorField {
  const char *name;
  uint32_t bit;
  const std::vector<int64_t> *values;
  /// The length the field has when the opcode declares it. Zero means the
  /// length is decided by a semantic check of its own, which is the case for
  /// `axes`.
  size_t declaredLength;
};

bool checkAttributeSizes(Validator &validator, const Instruction &instruction,
                         const OpcodeInfo &info, int64_t at) {
  const VectorField fields[] = {
      {"pads", kFieldPads, &instruction.pads, 4},
      {"strides", kFieldStrides, &instruction.strides, 2},
      {"dilations", kFieldDilations, &instruction.dilations, 2},
      {"kernel", kFieldKernel, &instruction.kernel, 2},
      {"axes", kFieldAxes, &instruction.axes, 0},
  };

  for (const VectorField &field : fields) {
    bool declared = (info.fieldMask & field.bit) != 0;
    if (!declared) {
      if (!field.values->empty())
        return validator.fail(
            Check::AttributeSize,
            std::string(info.name) + " gives `" + field.name +
                "` no meaning, so it must be empty, and it holds " +
                std::to_string(field.values->size()) + " entries",
            at);
      continue;
    }
    if (field.declaredLength != 0 &&
        field.values->size() != field.declaredLength)
      return validator.fail(
          Check::AttributeSize,
          std::string(info.name) + " needs " +
              std::to_string(field.declaredLength) + " `" + field.name +
              "` entries and has " + std::to_string(field.values->size()),
          at);
  }
  return true;
}

bool checkAttributeValues(Validator &validator, const Instruction &instruction,
                          const OpcodeInfo &info, int64_t at) {
  for (int64_t stride : instruction.strides)
    if (stride <= 0)
      return validator.fail(Check::AttributeValue,
                            "a window stride is " + std::to_string(stride) +
                                " and a stride is positive",
                            at);
  for (int64_t dilation : instruction.dilations)
    if (dilation <= 0)
      return validator.fail(Check::AttributeValue,
                            "a dilation is " + std::to_string(dilation) +
                                " and a dilation is positive",
                            at);
  for (int64_t pad : instruction.pads)
    if (pad < 0)
      return validator.fail(Check::AttributeValue,
                            "a pad is " + std::to_string(pad) +
                                " and a pad is not negative",
                            at);

  bool hasGroup = (info.fieldMask & kFieldGroup) != 0;
  if (hasGroup && instruction.group <= 0)
    return validator.fail(Check::AttributeValue,
                          "the group count is " +
                              std::to_string(instruction.group) +
                              " and a group count is positive",
                          at);
  if (!hasGroup && instruction.group != 0)
    return validator.fail(Check::AttributeValue,
                          std::string(info.name) +
                              " has no channel groups, so its group field is "
                              "zero, and it holds " +
                              std::to_string(instruction.group),
                          at);
  return true;
}

bool checkQuantization(Validator &validator, const Instruction &instruction,
                       const OpcodeInfo &info, int64_t at) {
  bool quantizes = (info.fieldMask & kFieldScale) != 0;

  if (quantizes) {
    if (!std::isfinite(instruction.scale) || instruction.scale <= 0.0f)
      return validator.fail(Check::QuantScale,
                            "the quantization scale is not a finite positive "
                            "number",
                            at);
    // The integer side of both quantization opcodes is i8, so the zero point
    // is an i8 value whichever direction the instruction goes.
    if (instruction.zeroPoint < -128 || instruction.zeroPoint > 127)
      return validator.fail(Check::QuantZeroPoint,
                            "the zero point is " +
                                std::to_string(instruction.zeroPoint) +
                                " and an i8 zero point is within [-128, 127]",
                            at);
  } else {
    if (instruction.scale != 0.0f)
      return validator.fail(Check::QuantScale,
                            std::string(info.name) +
                                " does not quantize, so its scale is zero",
                            at);
    if (instruction.zeroPoint != 0)
      return validator.fail(Check::QuantZeroPoint,
                            std::string(info.name) +
                                " does not quantize, so its zero point is zero",
                            at);
  }

  // The requantization pair is bounded on every instruction, not only on the
  // ones that use it. Section 9.2: the shift is within [0, 31] and the
  // multiplier is a positive int32, which is the range the fixed point
  // decomposition of Section 14 produces. A value outside it is a corrupt
  // file, not an exotic configuration.
  if (instruction.requantMultiplier <= 0)
    return validator.fail(Check::QuantRequantize,
                          "the requantization multiplier is " +
                              std::to_string(instruction.requantMultiplier) +
                              " and it is a positive int32",
                          at);
  if (instruction.requantShift < 0 || instruction.requantShift > 31)
    return validator.fail(Check::QuantRequantize,
                          "the requantization shift is " +
                              std::to_string(instruction.requantShift) +
                              " and it is within [0, 31]",
                          at);
  bool requantizes = (info.fieldMask & kFieldRequantize) != 0;
  if (!requantizes &&
      (instruction.requantMultiplier != 1 || instruction.requantShift != 0))
    return validator.fail(
        Check::QuantRequantize,
        std::string(info.name) +
            " does not requantize, so its multiplier is 1 and its shift is 0",
        at);
  return true;
}

/// TRANSPOSE. The permutation is of exactly the result rank, and every index
/// appears once.
bool checkTranspose(Validator &validator, const Instruction &instruction,
                    int64_t at) {
  {
    size_t rank = instruction.resultShape.size();
    if (instruction.axes.size() != rank)
      return validator.fail(Check::AxesPermutation,
                            "the permutation has " +
                                std::to_string(instruction.axes.size()) +
                                " entries and the result has rank " +
                                std::to_string(rank),
                            at);
    std::vector<bool> seen(rank, false);
    for (int64_t axis : instruction.axes) {
      if (axis < 0 || static_cast<size_t>(axis) >= rank)
        return validator.fail(Check::AxesPermutation,
                              "the permutation names axis " +
                                  std::to_string(axis) +
                                  " and the result has rank " +
                                  std::to_string(rank),
                              at);
      if (seen[static_cast<size_t>(axis)])
        return validator.fail(Check::AxesPermutation,
                              "the permutation names axis " +
                                  std::to_string(axis) + " more than once",
                              at);
      seen[static_cast<size_t>(axis)] = true;
    }
    const Operand &input = instruction.operands.front();
    if (input.shape.size() != rank)
      return validator.fail(Check::AxesPermutation,
                            "the operand has rank " +
                                std::to_string(input.shape.size()) +
                                " and the result has rank " +
                                std::to_string(rank),
                            at);
    for (size_t index = 0; index < rank; ++index) {
      int64_t expected = input.shape[static_cast<size_t>(
          instruction.axes[index])];
      if (instruction.resultShape[index] != expected)
        return validator.fail(
            Check::AxesPermutation,
            "result extent " + std::to_string(index) + " is " +
                std::to_string(instruction.resultShape[index]) +
                " and operand extent " +
                std::to_string(instruction.axes[index]) + " is " +
                std::to_string(expected),
            at);
    }
    return true;
  }
}

/// CONCAT. One axis, in range, and the extents agree everywhere else.
bool checkConcat(Validator &validator, const Instruction &instruction,
                 int64_t at) {
  {
    size_t rank = instruction.resultShape.size();
    if (instruction.axes.size() != 1)
      return validator.fail(Check::ConcatAxis,
                            "the axes vector holds " +
                                std::to_string(instruction.axes.size()) +
                                " entries and a concatenation has one axis",
                            at);
    int64_t axis = instruction.axes.front();
    if (axis < 0 || static_cast<size_t>(axis) >= rank)
      return validator.fail(Check::ConcatAxis,
                            "the axis is " + std::to_string(axis) +
                                " and the result has rank " +
                                std::to_string(rank),
                            at);
    int64_t sum = 0;
    for (const auto &[index, operand] : llvm::enumerate(instruction.operands)) {
      if (operand.shape.size() != rank)
        return validator.fail(Check::ConcatExtents,
                              "operand " + std::to_string(index) +
                                  " has rank " +
                                  std::to_string(operand.shape.size()) +
                                  " and the result has rank " +
                                  std::to_string(rank),
                              at);
      for (size_t dim = 0; dim < rank; ++dim) {
        if (static_cast<int64_t>(dim) == axis)
          continue;
        if (operand.shape[dim] != instruction.resultShape[dim])
          return validator.fail(
              Check::ConcatExtents,
              "operand " + std::to_string(index) + " extent " +
                  std::to_string(dim) + " is " +
                  std::to_string(operand.shape[dim]) +
                  " and the result's is " +
                  std::to_string(instruction.resultShape[dim]),
              at);
      }
      sum += operand.shape[static_cast<size_t>(axis)];
    }
    if (sum != instruction.resultShape[static_cast<size_t>(axis)])
      return validator.fail(
          Check::ConcatExtents,
          "the operands sum to " + std::to_string(sum) + " along axis " +
              std::to_string(axis) + " and the result extent is " +
              std::to_string(instruction.resultShape[static_cast<size_t>(axis)]),
          at);
    return true;
  }
}

/// QUANT and DEQUANT. Same shape in and out.
bool checkQuantShape(Validator &validator, const Instruction &instruction,
                     int64_t at) {
  const Operand &input = instruction.operands.front();
  if (input.shape != instruction.resultShape)
    return validator.fail(Check::QuantShape,
                          "the operand shape " + shapeText(input.shape) +
                              " and the result shape " +
                              shapeText(instruction.resultShape) + " differ",
                          at);
  return true;
}

/// RESHAPE. Same element count in and out; the extents need not agree.
///
/// The manual states this rule and so it is enforced rather than described. It
/// is reported as `result-shape` because that is what is wrong: the operand is
/// whatever it is, and the result shape is the one field a reshape gets to
/// choose.
bool checkReshape(Validator &validator, const Instruction &instruction,
                  int64_t at) {
  const Operand &input = instruction.operands.front();
  int64_t need = checkedElementCount(input.shape);
  int64_t have = checkedElementCount(instruction.resultShape);
  if (need != have)
    return validator.fail(Check::ResultShape,
                          "the operand holds " + std::to_string(need) +
                              " elements and the result holds " +
                              std::to_string(have) +
                              ", and a reshape moves every element exactly "
                              "once",
                          at);
  return true;
}

/// The semantic checks, dispatched per opcode.
///
/// **This switch has no `default` label, on purpose.** Everything mechanical
/// about an opcode is generated from the ISA description and needs no case
/// here; what is left is the handful of rules that need a human to decide
/// them. Under `-Werror=switch`, which lib/Encoding/CMakeLists.txt sets for
/// exactly this reason, adding an opcode to the description and nothing else
/// stops this file compiling until somebody has said what its semantic rule
/// is, even if the answer is that it has none. That is Section 9.4's claim
/// enforced by the toolchain rather than by a script that inspects source
/// files after the fact.
bool checkShapeSemantics(Validator &validator, const Instruction &instruction,
                         int64_t at) {
  switch (instruction.opcode) {
  case Opcode::TRANSPOSE:
    return checkTranspose(validator, instruction, at);
  case Opcode::CONCAT:
    return checkConcat(validator, instruction, at);
  case Opcode::QUANT:
  case Opcode::DEQUANT:
    return checkQuantShape(validator, instruction, at);
  case Opcode::RESHAPE:
    return checkReshape(validator, instruction, at);

  // Nothing semantic beyond the generated arity, field presence, memory space
  // and element type rules, and the operand extent arithmetic every
  // instruction goes through. Each of these is listed rather than swept into a
  // default, which is what makes the switch a gate.
  case Opcode::NOP:
  case Opcode::HALT:
  case Opcode::DMA_LOAD:
  case Opcode::DMA_STORE:
  case Opcode::MATMUL:
  case Opcode::CONV2D:
  case Opcode::ADD:
  case Opcode::MUL:
  case Opcode::RELU:
  case Opcode::POOL_MAX:
  case Opcode::POOL_AVG:
    return true;
  }
  // Unreachable: `isKnownOpcode` has already rejected anything the switch does
  // not name. The return is here because a scoped enum can hold a value no
  // enumerator names, and falling off the end of a value returning function is
  // undefined behaviour even when it cannot happen.
  return true;
}

bool checkInstruction(Validator &validator, const Instruction &instruction,
                      int64_t at, WrittenSpans &defined) {
  const Program &program = validator.program;

  // An opcode this build does not know cannot be interpreted at all: it
  // declares no arity, no fields and no spaces, so nothing after this point
  // has a rule to apply. Section 9.2's check list carries no `opcode` name and
  // inventing one would be inventing a name the specification does not have,
  // so this is reported as `structure`: the instruction is not an instruction.
  uint32_t rawOpcode = static_cast<uint32_t>(instruction.opcode);
  if (!isKnownOpcode(rawOpcode))
    return validator.fail(Check::Structure,
                          "the opcode is " + std::to_string(rawOpcode) +
                              " and the largest this build knows is " +
                              std::to_string(kMaxOpcode),
                          at);
  const OpcodeInfo &info = opcodeInfo(instruction.opcode);

  // ---- Arity, which is generated. ---------------------------------------
  int64_t operandCount = static_cast<int64_t>(instruction.operands.size());
  if (operandCount < info.minOperands ||
      (info.maxOperands >= 0 && operandCount > info.maxOperands))
    return validator.fail(Check::Arity,
                          std::string(info.name) + " takes " +
                              (info.maxOperands < 0
                                   ? std::to_string(info.minOperands) +
                                         " or more operands"
                                   : (info.minOperands == info.maxOperands
                                          ? std::to_string(info.minOperands) +
                                                " operands"
                                          : std::to_string(info.minOperands) +
                                                " or " +
                                                std::to_string(
                                                    info.maxOperands) +
                                                " operands")) +
                              " and has " + std::to_string(operandCount),
                          at);

  // ---- Element types. ----------------------------------------------------
  uint32_t rawResultType = static_cast<uint32_t>(instruction.resultElementType);
  if (!isKnownElemType(rawResultType))
    return validator.fail(Check::ElementType,
                          "the result element type is " +
                              std::to_string(rawResultType) +
                              ", which this build does not know",
                          at);
  for (const auto &[index, operand] : llvm::enumerate(instruction.operands)) {
    uint32_t rawType = static_cast<uint32_t>(operand.elementType);
    if (!isKnownElemType(rawType))
      return validator.fail(Check::ElementType,
                            "operand " + std::to_string(index) +
                                " has element type " + std::to_string(rawType) +
                                ", which this build does not know",
                            at);
    uint32_t rawSpace = static_cast<uint32_t>(operand.space);
    if (!isKnownMemSpace(rawSpace))
      return validator.fail(Check::OperandInRange,
                            "operand " + std::to_string(index) +
                                " names memory space " +
                                std::to_string(rawSpace) +
                                ", which this build does not know",
                            at);
  }
  if (!isKnownMemSpace(static_cast<uint32_t>(instruction.resultSpace)))
    return validator.fail(
        Check::ResultAddress,
        "the result names memory space " +
            std::to_string(static_cast<uint32_t>(instruction.resultSpace)) +
            ", which this build does not know",
        at);

  if (info.hasResult && (info.resultTypeMask & (1u << rawResultType)) == 0)
    return validator.fail(Check::ElementTypeSupported,
                          std::string(info.name) + " does not accept a " +
                              elemTypeName(instruction.resultElementType) +
                              " result",
                          at);

  for (const auto &[index, operand] : llvm::enumerate(instruction.operands)) {
    uint32_t rawType = static_cast<uint32_t>(operand.elementType);
    if (info.operandTypeMask != 0) {
      // A bridging opcode: its operands take a type of their own, and getting
      // the direction wrong is what `quant-types` names.
      if ((info.operandTypeMask & (1u << rawType)) == 0)
        return validator.fail(Check::QuantTypes,
                              std::string(info.name) + " reads " +
                                  elemTypeName(operand.elementType) +
                                  " and it reads the other type",
                              at);
      continue;
    }
    if (info.hasResult && operand.elementType != instruction.resultElementType)
      return validator.fail(Check::ElementTypeSupported,
                            "operand " + std::to_string(index) + " is " +
                                elemTypeName(operand.elementType) +
                                " and the result is " +
                                elemTypeName(instruction.resultElementType),
                            at);
  }

  // ---- The activation field. --------------------------------------------
  uint32_t rawActivation = static_cast<uint32_t>(instruction.activation);
  if (!isKnownActivation(rawActivation))
    return validator.fail(Check::Activation,
                          "the activation is " + std::to_string(rawActivation) +
                              ", which this build does not know",
                          at);
  if ((info.fieldMask & kFieldActivation) == 0 &&
      instruction.activation != Activation::None)
    return validator.fail(Check::Activation,
                          std::string(info.name) +
                              " fuses no activation, so its activation field "
                              "is `none`, and it holds `" +
                              activationName(instruction.activation) + "`",
                          at);

  // ---- The result. -------------------------------------------------------
  if (!info.hasResult) {
    if (!instruction.resultShape.empty())
      return validator.fail(Check::ResultShape,
                            std::string(info.name) +
                                " writes no result, so its result shape is "
                                "empty, and it holds " +
                                std::to_string(instruction.resultShape.size()) +
                                " extents",
                            at);
    if (instruction.resultAddress != 0)
      return validator.fail(Check::ResultAddress,
                            std::string(info.name) +
                                " writes no result, so its result address is "
                                "zero, and it holds " +
                                std::to_string(instruction.resultAddress),
                            at);
  } else {
    int64_t resultBytes = instruction.resultByteSize();
    if (resultBytes < 0)
      return validator.fail(Check::ResultShape,
                            "the result shape " +
                                shapeText(instruction.resultShape) +
                                " is empty, holds a non positive extent, or "
                                "has a product above the shape limit",
                            at);
    if (instruction.resultAddress < 0)
      return validator.fail(Check::ResultAddress,
                            "the result address is " +
                                std::to_string(instruction.resultAddress),
                            at);
    if (static_cast<uint32_t>(instruction.resultSpace) != info.resultSpace)
      return validator.fail(Check::ResultAddress,
                            std::string(info.name) + " writes its result in " +
                                memSpaceName(static_cast<MemSpace>(
                                    info.resultSpace)) +
                                " and this one names " +
                                memSpaceName(instruction.resultSpace),
                            at);

    if (instruction.resultSpace == MemSpace::Dram) {
      if (!fitsInMemory(instruction.resultAddress, resultBytes,
                        program.dramBytes))
        return validator.fail(Check::DramInRange,
                              "the result runs from " +
                                  std::to_string(instruction.resultAddress) +
                                  " for " + std::to_string(resultBytes) +
                                  " bytes, past the declared DRAM size of " +
                                  std::to_string(program.dramBytes),
                              at);
    } else if (!fitsInMemory(instruction.resultAddress, resultBytes,
                             program.scratchpadBytes)) {
      return validator.fail(
          Check::ResultInRange,
          "the result runs from " + std::to_string(instruction.resultAddress) +
              " for " + std::to_string(resultBytes) +
              " bytes, past the declared scratchpad size of " +
              std::to_string(program.scratchpadBytes),
          at);
    }
  }

  // ---- The operands. -----------------------------------------------------
  for (const auto &[index, operand] : llvm::enumerate(instruction.operands)) {
    // The generated space rule. The last declared slot repeats, which is what
    // makes a variadic opcode's operands all live in one space.
    uint32_t expectedSpace =
        info.numOperandSpaces == 0
            ? 0u
            : info.operandSpaces[std::min<size_t>(index,
                                                  info.numOperandSpaces - 1)];
    if (static_cast<uint32_t>(operand.space) != expectedSpace)
      return validator.fail(Check::OperandInRange,
                            "operand " + std::to_string(index) + " of " +
                                info.name + " lives in " +
                                memSpaceName(
                                    static_cast<MemSpace>(expectedSpace)) +
                                " and this one names " +
                                memSpaceName(operand.space),
                            at);

    if (operand.shape.size() != operand.strides.size())
      return validator.fail(Check::OperandExtent,
                            "operand " + std::to_string(index) + " has " +
                                std::to_string(operand.shape.size()) +
                                " extents and " +
                                std::to_string(operand.strides.size()) +
                                " strides",
                            at);
    int64_t span = operand.addressedByteSpan();
    if (span < 0)
      return validator.fail(Check::OperandExtent,
                            "operand " + std::to_string(index) + " with shape " +
                                shapeText(operand.shape) + " and strides " +
                                shapeText(operand.strides) +
                                " addresses an impossible byte range",
                            at);

    if (operand.address < 0)
      return validator.fail(operand.space == MemSpace::Dram
                                ? Check::DramAddress
                                : Check::OperandInRange,
                            "operand " + std::to_string(index) +
                                " has address " +
                                std::to_string(operand.address),
                            at);

    if (!fitsInMemory(operand.address, span,
                      memorySize(program, operand.space)))
      return validator.fail(
          operand.space == MemSpace::Dram ? Check::DramInRange
                                          : Check::OperandInRange,
          "operand " + std::to_string(index) + " runs from " +
              std::to_string(operand.address) + " for " + std::to_string(span) +
              " bytes, past the declared " + memSpaceName(operand.space) +
              " size of " + std::to_string(memorySize(program, operand.space)),
          at);

    std::optional<int64_t> end = defined.spanEndAt(operand.space,
                                                   operand.address);
    if (!end)
      return validator.fail(Check::OperandDefined,
                            "operand " + std::to_string(index) + " reads " +
                                memSpaceName(operand.space) + " address " +
                                std::to_string(operand.address) +
                                ", which nothing has written and no declared "
                                "region covers",
                            at);
    // `*end - operand.address` rather than `operand.address + span`. The
    // subtraction cannot overflow, because `spanEndAt` returns an end only for
    // a span that contains the address, so the difference is at least one. The
    // addition can: a file is free to declare a memory of nearly 2^64 bytes,
    // and an address near the signed limit plus a span is undefined behaviour
    // rather than a large number. UBSan found this one on the corpus, which is
    // recorded as D-0021.
    if (span > *end - operand.address)
      return validator.fail(
          Check::OperandExtent,
          "operand " + std::to_string(index) + " reads " +
              std::to_string(span) + " bytes from " +
              std::to_string(operand.address) +
              " and the buffer written there ends at " + std::to_string(*end),
          at);
  }

  // ---- The fields. -------------------------------------------------------
  if (!checkAttributeSizes(validator, instruction, info, at))
    return false;
  if (!checkAttributeValues(validator, instruction, info, at))
    return false;
  if (!checkQuantization(validator, instruction, info, at))
    return false;
  if (!checkShapeSemantics(validator, instruction, at))
    return false;

  if (info.hasResult)
    defined.write(instruction.resultSpace, instruction.resultAddress,
                  instruction.resultByteSize());
  return true;
}

//===----------------------------------------------------------------------===//
// The debug section.
//===----------------------------------------------------------------------===//

bool checkDebug(Validator &validator) {
  const Program &program = validator.program;
  int64_t previous = -1;
  for (const auto &[index, entry] : llvm::enumerate(program.debug)) {
    int64_t at = static_cast<int64_t>(index);
    if (entry.pc >= program.instructions.size())
      return validator.fail(Check::DebugPc,
                            "the program counter is " +
                                std::to_string(entry.pc) +
                                " and the program has " +
                                std::to_string(program.instructions.size()) +
                                " instructions",
                            -1, at, "debug entry");
    if (static_cast<int64_t>(entry.pc) <= previous)
      return validator.fail(Check::DebugOrder,
                            "the program counter is " +
                                std::to_string(entry.pc) +
                                " and the previous entry's was " +
                                std::to_string(previous) +
                                ", so the entries are not strictly increasing",
                            -1, at, "debug entry");
    previous = entry.pc;

    if (entry.name.size() > Program::kMaxDebugNameBytes)
      return validator.fail(Check::DebugSize,
                            "the name is " + std::to_string(entry.name.size()) +
                                " bytes, above the limit of " +
                                std::to_string(Program::kMaxDebugNameBytes),
                            -1, at, "debug entry");
    for (const auto &[position, character] : llvm::enumerate(entry.name)) {
      auto byte = static_cast<unsigned char>(character);
      if (byte == 0)
        return validator.fail(Check::DebugName,
                              "the name holds a NUL at byte " +
                                  std::to_string(position),
                              -1, at, "debug entry");
      if (byte >= 0x80)
        return validator.fail(Check::DebugName,
                              "the name holds the byte 0x" +
                                  std::string(1, "0123456789abcdef"[byte >> 4]) +
                                  std::string(1,
                                              "0123456789abcdef"[byte & 0xf]) +
                                  " at position " + std::to_string(position) +
                                  ", and an ONNX node name is ASCII",
                              -1, at, "debug entry");
    }
  }
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// The entry point.
//===----------------------------------------------------------------------===//

std::optional<ProgramError> Program::validate() const {
  Validator validator(*this);

  // The order is dependency order, not the order Section 9.2 lists the names
  // in. Regions first, because the instructions read them. Instructions in
  // program order, because `operand-defined` is a property of what ran before.
  // Debug last, because a program counter is an index into the instructions.
  //
  // **`count-cap` is not checked here and that is deliberate.** Section 9.2's
  // third rule is about bounding a length before allocating from it, and the
  // only place a length arrives from untrusted input is the decoder, which
  // checks the cap and then checks the payload against the bytes that remain
  // before reserving anything. A copy of the cap here could only ever fire on
  // a Program somebody built in memory with more than two hundred and sixty
  // eight million of something, which is not a state this process could reach
  // without having already allocated far more than the cap exists to prevent.
  // Writing it anyway would be writing a branch no test can take.

  WrittenSpans defined;
  if (!checkRegions(validator, defined))
    return validator.failure;

  for (const auto &[index, instruction] : llvm::enumerate(instructions))
    if (!checkInstruction(validator, instruction, static_cast<int64_t>(index),
                          defined))
      return validator.failure;

  if (!checkDebug(validator))
    return validator.failure;

  return std::nullopt;
}
