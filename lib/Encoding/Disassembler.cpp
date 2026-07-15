//===- Disassembler.cpp - Render an encoded program as text ---------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"

#include <sstream>

namespace npu {

namespace {

std::string shapeStr(const std::vector<int64_t> &shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i)
    os << (i ? "x" : "") << shape[i];
  os << "]";
  return os.str();
}

std::string listStr(const std::vector<int64_t> &v) {
  std::ostringstream os;
  for (size_t i = 0; i < v.size(); ++i)
    os << (i ? "," : "") << v[i];
  return os.str();
}

} // namespace

std::string disassemble(const Program &p) {
  std::ostringstream os;
  os << "; npu-objdump\n";
  os << "; version " << p.version << "\n";
  os << "; scratchpad " << p.scratchpadBytes << " bytes, dram " << p.dramBytes
     << " bytes\n";
  os << "; " << p.inputs.size() << " inputs, " << p.outputs.size()
     << " outputs, " << p.constants.size() << " constants, "
     << p.instructions.size() << " instructions\n\n";

  os << ".dram\n";
  for (size_t i = 0; i < p.inputs.size(); ++i)
    os << "  input" << i << "  @0x" << std::hex << p.inputs[i].dramOffset
       << std::dec << " " << shapeStr(p.inputs[i].shape) << "\n";
  for (size_t i = 0; i < p.constants.size(); ++i)
    os << "  const" << i << "  @0x" << std::hex << p.constants[i].dramOffset
       << std::dec << " " << shapeStr(p.constants[i].shape) << "\n";
  for (size_t i = 0; i < p.outputs.size(); ++i)
    os << "  output" << i << " @0x" << std::hex << p.outputs[i].dramOffset
       << std::dec << " " << shapeStr(p.outputs[i].shape) << "\n";

  os << "\n.text\n";
  int idx = 0;
  for (const Instruction &in : p.instructions) {
    os << "  " << idx++ << ": " << opcodeName(in.op);
    switch (in.op) {
    case Opcode::DmaLoad:
      os << " sp[0x" << std::hex << in.resultAddr << "] <- dram[0x"
         << in.dramAddr << std::dec << "] " << shapeStr(in.resultShape);
      break;
    case Opcode::DmaStore:
      os << " dram[0x" << std::hex << in.dramAddr << "] <- sp[0x"
         << in.operandAddrs.front() << std::dec << "] "
         << shapeStr(in.resultShape);
      break;
    case Opcode::Conv2D:
    case Opcode::MatMul:
      os << " sp[0x" << std::hex << in.resultAddr << "] <- ";
      for (size_t i = 0; i < in.operandAddrs.size(); ++i)
        os << (i ? ", " : "") << "sp[0x" << in.operandAddrs[i] << "]";
      os << std::dec << " " << shapeStr(in.resultShape);
      if (in.activation)
        os << " act=relu";
      break;
    case Opcode::Nop:
    case Opcode::Halt:
      break;
    default:
      os << " sp[0x" << std::hex << in.resultAddr << "] <- ";
      for (size_t i = 0; i < in.operandAddrs.size(); ++i)
        os << (i ? ", " : "") << "sp[0x" << in.operandAddrs[i] << "]";
      os << std::dec << " " << shapeStr(in.resultShape);
      break;
    }
    if (!in.kernelShape.empty())
      os << " kernel=" << listStr(in.kernelShape);
    if (!in.strides.empty() &&
        (in.op == Opcode::Conv2D || in.op == Opcode::PoolMax ||
         in.op == Opcode::PoolAvg))
      os << " stride=" << listStr(in.strides);
    os << "\n";
  }
  return os.str();
}

} // namespace npu
