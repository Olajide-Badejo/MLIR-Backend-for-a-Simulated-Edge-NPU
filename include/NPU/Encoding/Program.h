//===- Program.h - Encoded npuisa program model -----------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// In memory model of an encoded npuisa program and the .nbin binary format.
// The format is a fixed header followed by tagged records; it is deliberately
// simple to get right rather than bit packed. See docs/ISA_MANUAL.md.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_PROGRAM_H
#define NPU_ENCODING_PROGRAM_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace npu {

// Instruction opcodes. The numeric values are part of the binary format and must
// not be renumbered without bumping the format version.
enum class Opcode : uint16_t {
  Nop = 0,
  Halt = 1,
  DmaLoad = 2,
  DmaStore = 3,
  Conv2D = 4,
  MatMul = 5,
  Relu = 6,
  Add = 7,
  Mul = 8,
  PoolMax = 9,
  PoolAvg = 10,
  Reshape = 11,
};

const char *opcodeName(Opcode op);

// A single instruction. Fields not relevant to an opcode keep their defaults.
struct Instruction {
  Opcode op = Opcode::Nop;
  int64_t resultAddr = -1;             // scratchpad byte offset of the result
  std::vector<int64_t> resultShape;    // logical shape of the result buffer
  std::vector<int64_t> operandAddrs;   // scratchpad offsets of operand buffers
  int64_t dramAddr = -1;               // dma_load source / dma_store destination
  int32_t activation = 0;              // 0 none, 1 relu (conv, matmul)
  int64_t group = 1;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  std::vector<int64_t> kernelShape;
};

// A DRAM resident region: an offset plus the logical shape of its fp32 tensor.
struct MemRegion {
  int64_t dramOffset = 0;
  std::vector<int64_t> shape;
  int64_t byteSize() const {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n * 4;
  }
};

struct Program {
  uint32_t version = 1;
  int64_t scratchpadBytes = 0;
  int64_t dramBytes = 0;
  std::vector<MemRegion> inputs;
  std::vector<MemRegion> outputs;
  std::vector<MemRegion> constants;
  std::vector<std::vector<float>> constantData; // parallel to constants
  std::vector<Instruction> instructions;

  // Serialize to / from the .nbin byte stream. decode returns nullopt on a
  // malformed or truncated stream.
  std::vector<uint8_t> encode() const;
  static std::optional<Program> decode(const std::vector<uint8_t> &bytes);
};

} // namespace npu

#endif // NPU_ENCODING_PROGRAM_H
