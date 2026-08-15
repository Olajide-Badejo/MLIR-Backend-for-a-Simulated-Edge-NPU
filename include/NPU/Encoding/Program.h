//===- Program.h - Encoded npuisa program model -----------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// In memory model of an encoded npuisa program and the .nbin binary format.
// The format is a fixed header followed by fixed order sections, each repeated
// section prefixed by a u32 count. There are no tags, so a reader cannot skip a
// field it does not recognise; the version field is what carries compatibility
// instead. Byte oriented rather than bit packed, and written in host byte order.
// See docs/ISA_MANUAL.md.
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

// Highest opcode the current format defines. A decoded u16 outside this range
// is not a valid instruction, and letting it reach the simulator's switch is
// undefined behaviour.
constexpr uint16_t kMaxOpcode = static_cast<uint16_t>(Opcode::Reshape);
bool isValidOpcode(Opcode op);

// Upper bounds on the memory a program may declare.
//
// These exist because the simulator allocates both spaces up front, so a
// corrupt size field is otherwise a request for an impossible allocation: a
// single flipped byte in the header turned into a std::bad_alloc rather than a
// diagnostic. Both are absurdly generous for the accelerator being modelled,
// whose scratchpad budget is 1 MB and whose whole LeNet DRAM footprint is under
// 400 KB, so no real program comes close.
constexpr int64_t kMaxScratchpadBytes = int64_t{64} << 20; // 64 MiB
constexpr int64_t kMaxDramBytes = int64_t{64} << 20;       // 64 MiB

// Why a program cannot be executed.
//
// Returning this rather than a bare nullopt matters: the .nbin format is the
// interface between npu-translate and npu-sim, so a rejection is usually a
// compiler bug, and "malformed .nbin" gives whoever has to find it nothing to
// go on.
struct ValidationError {
  // instructionIndex for a problem in the header or the region tables, rather
  // than in a particular instruction.
  static constexpr size_t kProgramLevel = static_cast<size_t>(-1);

  size_t instructionIndex = kProgramLevel;
  std::string check;  // the rule that failed, for example "operand-in-range"
  std::string detail; // what was actually wrong, with numbers

  std::string toString() const;
};

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
  // Bump when the layout or the opcode numbering changes. Written by encode
  // and, since the hardening in phase U3, actually checked by decode.
  static constexpr uint32_t kVersion = 1;

  uint32_t version = kVersion;
  int64_t scratchpadBytes = 0;
  int64_t dramBytes = 0;
  std::vector<MemRegion> inputs;
  std::vector<MemRegion> outputs;
  std::vector<MemRegion> constants;
  std::vector<std::vector<float>> constantData; // parallel to constants
  std::vector<Instruction> instructions;

  std::vector<uint8_t> encode() const;

  // Check every invariant the simulator relies on: the version, opcode range,
  // operand arity, that scratchpad and DRAM accesses fall inside their regions,
  // that an operand address was actually written before it is read, and that
  // shapes are non empty with positive extents whose product matches the region
  // they describe. Returns nothing when the program is executable.
  std::optional<ValidationError> validate() const;

  // Decode and validate. Returns nullopt on a truncated or malformed stream or
  // on any validation failure, filling *error when one is supplied.
  static std::optional<Program> decode(const std::vector<uint8_t> &bytes,
                                       ValidationError *error = nullptr);

  // Structural decode with no validation, for npu-objdump. Disassembling a file
  // you already suspect is broken is the whole point of a disassembler, so it
  // must not refuse the ones validate would reject.
  static std::optional<Program>
  decodeUnvalidated(const std::vector<uint8_t> &bytes);
};

} // namespace npu

#endif // NPU_ENCODING_PROGRAM_H
