//===- TestPrograms.h - hand building a .nbin for a test ------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The scaffolding every semantics test in this directory is built out of.
//
// **A kernel is tested through a whole program, never in isolation.** A kernel
// called directly is a kernel tested with the validation, the dispatch, the
// memory model and the DMA path all absent, which are four of the five things
// that could be wrong. So every test here declares DRAM regions, loads them
// into the scratchpad with `DMA_LOAD`, computes, stores the result back with
// `DMA_STORE`, and reads the output region: the same shape the compiler emits.
//
// **The scratchpad is sized strictly and every test says the number out loud.**
// Section 9.3 requires the simulator never to grow the scratchpad to cover the
// writes it finds, and requires every hand built test program to set an
// explicit tight `scratchpadBytes` with the arithmetic in a comment.
// `Builder::finish` takes that number as an argument and fails the test if it
// is not exactly what the buffers add up to, so the comment and the code cannot
// drift: a test that declares too much has a scratchpad that is not tight, and
// a test that declares too little would be describing a program the validator
// refuses.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_UNITTESTS_SIMULATOR_TESTPROGRAMS_H
#define NPU_UNITTESTS_SIMULATOR_TESTPROGRAMS_H

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Simulator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "gtest/gtest.h"

#include <cstring>
#include <vector>

namespace npusim {

using namespace nbin;

/// Row major strides for a contiguous buffer, in elements.
inline std::vector<int64_t> contiguous(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty())
    return {};
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t axis = shape.size() - 1; axis-- > 0;)
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  return strides;
}

/// The product of a shape.
inline int64_t elements(llvm::ArrayRef<int64_t> shape) {
  int64_t product = 1;
  for (int64_t extent : shape)
    product *= extent;
  return product;
}

/// An operand over a contiguous buffer.
inline Operand at(MemSpace space, int64_t address, llvm::ArrayRef<int64_t> shape,
                  ElemType type = ElemType::F32) {
  Operand operand;
  operand.space = space;
  operand.elementType = type;
  operand.address = address;
  operand.shape.assign(shape.begin(), shape.end());
  operand.strides = contiguous(shape);
  return operand;
}

/// An operand over a strided view: a broadcast, a permuted layout, or the non
/// unit innermost stride the DMA cost term exists for.
inline Operand strided(MemSpace space, int64_t address,
                       llvm::ArrayRef<int64_t> shape,
                       llvm::ArrayRef<int64_t> strides,
                       ElemType type = ElemType::F32) {
  Operand operand;
  operand.space = space;
  operand.elementType = type;
  operand.address = address;
  operand.shape.assign(shape.begin(), shape.end());
  operand.strides.assign(strides.begin(), strides.end());
  return operand;
}

/// Builds one program, and keeps the DRAM map and the scratchpad layout.
class Builder {
public:
  /// A constant region holding these values, returned as its DRAM offset.
  int64_t constant(llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<float> data) {
    EXPECT_EQ(elements(shape), static_cast<int64_t>(data.size()));
    Constant entry;
    entry.region.offset = static_cast<uint64_t>(dramCursor);
    entry.region.elementType = ElemType::F32;
    entry.region.shape.assign(shape.begin(), shape.end());
    entry.data.resize(data.size() * sizeof(float));
    if (!data.empty())
      std::memcpy(entry.data.data(), data.data(), entry.data.size());
    const int64_t offset = dramCursor;
    advanceDram(static_cast<int64_t>(data.size() * sizeof(float)));
    program.constants.push_back(std::move(entry));
    return offset;
  }

  /// A declared input region, returned as its DRAM offset.
  int64_t input(llvm::ArrayRef<int64_t> shape,
                ElemType type = ElemType::F32) {
    MemRegion region;
    region.offset = static_cast<uint64_t>(dramCursor);
    region.elementType = type;
    region.shape.assign(shape.begin(), shape.end());
    const int64_t offset = dramCursor;
    advanceDram(region.byteSize());
    program.inputs.push_back(std::move(region));
    return offset;
  }

  /// A declared output region, returned as its DRAM offset.
  int64_t output(llvm::ArrayRef<int64_t> shape,
                 ElemType type = ElemType::F32) {
    MemRegion region;
    region.offset = static_cast<uint64_t>(dramCursor);
    region.elementType = type;
    region.shape.assign(shape.begin(), shape.end());
    const int64_t offset = dramCursor;
    advanceDram(region.byteSize());
    program.outputs.push_back(std::move(region));
    return offset;
  }

  /// A DRAM spill slot, which is a place to write rather than a place to read.
  int64_t spillSlot(llvm::ArrayRef<int64_t> shape,
                    ElemType type = ElemType::F32) {
    MemRegion region;
    region.offset = static_cast<uint64_t>(dramCursor);
    region.elementType = type;
    region.shape.assign(shape.begin(), shape.end());
    const int64_t offset = dramCursor;
    advanceDram(region.byteSize());
    program.spillSlots.push_back(std::move(region));
    return offset;
  }

  /// A scratchpad buffer, returned as its byte address.
  ///
  /// Allocation is sequential and unaligned beyond the element size, which is
  /// what makes the total tight: an allocator that rounded every buffer up to a
  /// cache line would produce a scratchpad larger than the program needs, and
  /// the point of these programs is that the declared size is exactly the size.
  int64_t scratch(int64_t count, ElemType type = ElemType::F32) {
    const int64_t address = scratchCursor;
    scratchCursor += count * elementByteSize(type);
    return address;
  }

  void add(Instruction instruction) {
    program.instructions.push_back(std::move(instruction));
  }

  /// Closes the program, asserting the declared scratchpad is the tight one.
  Program finish(uint64_t declaredScratchpadBytes) {
    EXPECT_EQ(static_cast<uint64_t>(scratchCursor), declaredScratchpadBytes)
        << "the scratchpad this program actually needs and the size it "
           "declares disagree. Section 9.3 requires the declared size to be "
           "tight and the arithmetic to be in a comment beside it";
    program.scratchpadBytes = declaredScratchpadBytes;
    program.dramBytes = static_cast<uint64_t>(dramCursor);
    return program;
  }

private:
  void advanceDram(int64_t bytes) {
    // Sixty four byte alignment, which is what the encoder's DRAM map uses. It
    // is deliberately not applied to the scratchpad: the scratchpad total is
    // the number these tests declare and pad bytes would make it a number
    // nobody could compute from the shapes.
    dramCursor += bytes;
    dramCursor = (dramCursor + 63) & ~int64_t{63};
  }

  Program program;
  int64_t dramCursor = 0;
  int64_t scratchCursor = 0;
};

//===----------------------------------------------------------------------===//
// The instructions the scaffolding builds by hand.
//===----------------------------------------------------------------------===//

/// The contiguous, row major strides a shape implies.
///
/// **The neutral value of `Instruction::resultStrides`, which version 2
/// added.** A result that is not a view of anything steps by the product of the
/// extents below each axis, which is the same rule an operand's strides have
/// followed since version 1. Every builder below fills it, because the
/// validator requires a stride per extent on any opcode that has a result and
/// a scaffolding that left it empty would fail every program it built.
inline std::vector<int64_t> resultStridesFor(llvm::ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (size_t index = shape.size(); index-- > 1;)
    strides[index - 1] = strides[index] * shape[index];
  return strides;
}

inline Instruction dmaLoad(int64_t scratchAddress, llvm::ArrayRef<int64_t> shape,
                           Operand source) {
  Instruction instruction;
  instruction.opcode = Opcode::DMA_LOAD;
  instruction.resultSpace = MemSpace::Scratchpad;
  instruction.resultElementType = source.elementType;
  instruction.resultAddress = scratchAddress;
  instruction.resultShape.assign(shape.begin(), shape.end());
  instruction.resultStrides = resultStridesFor(shape);
  instruction.operands.push_back(std::move(source));
  return instruction;
}

inline Instruction dmaStore(int64_t dramAddress, llvm::ArrayRef<int64_t> shape,
                            Operand source) {
  Instruction instruction;
  instruction.opcode = Opcode::DMA_STORE;
  instruction.resultSpace = MemSpace::Dram;
  instruction.resultElementType = source.elementType;
  instruction.resultAddress = dramAddress;
  instruction.resultShape.assign(shape.begin(), shape.end());
  instruction.resultStrides = resultStridesFor(shape);
  instruction.operands.push_back(std::move(source));
  return instruction;
}

inline Instruction compute(Opcode opcode, int64_t resultAddress,
                           llvm::ArrayRef<int64_t> resultShape,
                           std::vector<Operand> operands) {
  Instruction instruction;
  instruction.opcode = opcode;
  instruction.resultSpace = MemSpace::Scratchpad;
  instruction.resultElementType = ElemType::F32;
  instruction.resultAddress = resultAddress;
  instruction.resultShape.assign(resultShape.begin(), resultShape.end());
  instruction.resultStrides = resultStridesFor(resultShape);
  instruction.operands = std::move(operands);
  return instruction;
}

inline Instruction halt() {
  Instruction instruction;
  instruction.opcode = Opcode::HALT;
  return instruction;
}

inline Instruction nop() {
  Instruction instruction;
  instruction.opcode = Opcode::NOP;
  return instruction;
}

//===----------------------------------------------------------------------===//
// Running one.
//===----------------------------------------------------------------------===//

/// A program, its machine, and the outputs it produced.
///
/// The simulator holds a reference to the program, so the harness owns both and
/// keeps them alive together. A test that let the program go out of scope while
/// the simulator lived would be reading freed memory, which is exactly the kind
/// of bug this project's sanitizer build exists to catch and exactly the kind
/// of bug a test helper should make impossible instead.
class Harness {
public:
  explicit Harness(Program built)
      : program(std::move(built)), simulator(program) {}

  SimResult run(SimOptions options = {}) { return simulator.run(options); }
  SimResult runUnvalidated(SimOptions options = {}) {
    return simulator.runUnvalidated(options);
  }

  /// The f32 contents of a declared output region.
  std::vector<float> outputF32(size_t index) {
    llvm::ArrayRef<uint8_t> bytes = simulator.outputBytes(index);
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!values.empty())
      std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
  }

  Simulator &sim() { return simulator; }
  const Program &built() const { return program; }

private:
  Program program;
  Simulator simulator;
};

//===----------------------------------------------------------------------===//
// The canonical one instruction program.
//===----------------------------------------------------------------------===//

/// One operand of a semantics test.
///
/// `shape` and `data` describe the buffer as it arrives in DRAM and is loaded
/// into the scratchpad. `viewShape` and `viewStrides` describe how the compute
/// instruction reads it, and default to the buffer itself. The two are separate
/// so that the stride cases have somewhere to live: a rank 1 channel broadcast
/// is a three element buffer read as a rank 4 view with three zero strides, and
/// nothing about the buffer changes to make that work.
struct OperandSpec {
  std::vector<int64_t> shape;
  std::vector<float> data;
  std::vector<int64_t> viewShape;
  std::vector<int64_t> viewStrides;
};

/// A run and what it produced.
struct Outcome {
  SimResult result;
  std::vector<float> values;
};

/// Builds and runs the shape every semantics test in this directory uses:
/// constants in DRAM, one `DMA_LOAD` per operand, one compute instruction, one
/// `DMA_STORE`, `HALT`.
///
/// `declaredScratchpadBytes` is checked against what the buffers actually need,
/// so every caller states the number and the arithmetic in a comment beside it.
inline Outcome computeOnce(llvm::ArrayRef<OperandSpec> inputs,
                           llvm::ArrayRef<int64_t> resultShape, Opcode opcode,
                           llvm::function_ref<void(Instruction &)> configure,
                           uint64_t declaredScratchpadBytes,
                           SimOptions options = {}) {
  Builder builder;

  std::vector<int64_t> constants;
  std::vector<int64_t> buffers;
  for (const OperandSpec &spec : inputs) {
    constants.push_back(builder.constant(spec.shape, spec.data));
    buffers.push_back(builder.scratch(elements(spec.shape)));
  }
  const int64_t resultBuffer = builder.scratch(elements(resultShape));
  const int64_t outputRegion = builder.output(resultShape);

  for (size_t index = 0; index < inputs.size(); ++index)
    builder.add(dmaLoad(buffers[index], inputs[index].shape,
                        at(MemSpace::Dram, constants[index],
                           inputs[index].shape)));

  std::vector<Operand> operands;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const OperandSpec &spec = inputs[index];
    if (spec.viewShape.empty())
      operands.push_back(at(MemSpace::Scratchpad, buffers[index], spec.shape));
    else
      operands.push_back(strided(MemSpace::Scratchpad, buffers[index],
                                 spec.viewShape, spec.viewStrides));
  }

  Instruction instruction =
      compute(opcode, resultBuffer, resultShape, std::move(operands));
  configure(instruction);
  builder.add(std::move(instruction));

  builder.add(dmaStore(outputRegion, resultShape,
                       at(MemSpace::Scratchpad, resultBuffer, resultShape)));
  builder.add(halt());

  Harness harness(builder.finish(declaredScratchpadBytes));
  Outcome outcome;
  outcome.result = harness.run(options);
  outcome.values = harness.outputF32(0);
  return outcome;
}

/// Asserts two float vectors agree elementwise, printing the index that did
/// not. `EXPECT_FLOAT_EQ` per element inside a loop reports the failure without
/// saying which element it was, which turns a two second fix into a hunt.
inline void expectValues(llvm::ArrayRef<float> actual,
                         llvm::ArrayRef<float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index)
    EXPECT_FLOAT_EQ(actual[index], expected[index])
        << "at element " << index;
}

} // namespace npusim

#endif // NPU_UNITTESTS_SIMULATOR_TESTPROGRAMS_H
