//===- Program.cpp - .nbin encode and decode ------------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include <cstring>

namespace npu {

const char *opcodeName(Opcode op) {
  switch (op) {
  case Opcode::Nop:
    return "NOP";
  case Opcode::Halt:
    return "HALT";
  case Opcode::DmaLoad:
    return "DMA_LOAD";
  case Opcode::DmaStore:
    return "DMA_STORE";
  case Opcode::Conv2D:
    return "CONV2D";
  case Opcode::MatMul:
    return "MATMUL";
  case Opcode::Relu:
    return "RELU";
  case Opcode::Add:
    return "ADD";
  case Opcode::Mul:
    return "MUL";
  case Opcode::PoolMax:
    return "POOL_MAX";
  case Opcode::PoolAvg:
    return "POOL_AVG";
  case Opcode::Reshape:
    return "RESHAPE";
  }
  return "UNKNOWN";
}

namespace {

// Little endian byte stream helpers.
struct Writer {
  std::vector<uint8_t> bytes;
  template <typename T> void put(T value) {
    auto *p = reinterpret_cast<const uint8_t *>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  void putVec(const std::vector<int64_t> &v) {
    put<uint32_t>(static_cast<uint32_t>(v.size()));
    for (int64_t x : v)
      put<int64_t>(x);
  }
};

struct Reader {
  const uint8_t *p;
  const uint8_t *end;
  bool ok = true;
  explicit Reader(const std::vector<uint8_t> &b) : p(b.data()), end(b.data() + b.size()) {}
  template <typename T> T get() {
    if (p + sizeof(T) > end) {
      ok = false;
      return T{};
    }
    T value;
    std::memcpy(&value, p, sizeof(T));
    p += sizeof(T);
    return value;
  }
  // Bound element counts so a corrupt length cannot request a huge allocation.
  uint32_t getCount() {
    uint32_t n = get<uint32_t>();
    if (n > (1u << 28)) {
      ok = false;
      return 0;
    }
    return n;
  }
  std::vector<int64_t> getVec() {
    uint32_t n = getCount();
    std::vector<int64_t> v(n);
    for (uint32_t i = 0; i < n; ++i)
      v[i] = get<int64_t>();
    return v;
  }
};

constexpr char kMagic[4] = {'N', 'P', 'U', 'B'};

void writeRegion(Writer &w, const MemRegion &r) {
  w.put<int64_t>(r.dramOffset);
  w.putVec(r.shape);
}

MemRegion readRegion(Reader &r) {
  MemRegion m;
  m.dramOffset = r.get<int64_t>();
  m.shape = r.getVec();
  return m;
}

void writeInstr(Writer &w, const Instruction &in) {
  w.put<uint16_t>(static_cast<uint16_t>(in.op));
  w.put<int64_t>(in.resultAddr);
  w.putVec(in.resultShape);
  w.putVec(in.operandAddrs);
  w.put<int64_t>(in.dramAddr);
  w.put<int32_t>(in.activation);
  w.put<int64_t>(in.group);
  w.putVec(in.strides);
  w.putVec(in.pads);
  w.putVec(in.dilations);
  w.putVec(in.kernelShape);
}

Instruction readInstr(Reader &r) {
  Instruction in;
  in.op = static_cast<Opcode>(r.get<uint16_t>());
  in.resultAddr = r.get<int64_t>();
  in.resultShape = r.getVec();
  in.operandAddrs = r.getVec();
  in.dramAddr = r.get<int64_t>();
  in.activation = r.get<int32_t>();
  in.group = r.get<int64_t>();
  in.strides = r.getVec();
  in.pads = r.getVec();
  in.dilations = r.getVec();
  in.kernelShape = r.getVec();
  return in;
}

} // namespace

std::vector<uint8_t> Program::encode() const {
  Writer w;
  w.bytes.insert(w.bytes.end(), kMagic, kMagic + 4);
  w.put<uint32_t>(version);
  w.put<int64_t>(scratchpadBytes);
  w.put<int64_t>(dramBytes);

  w.put<uint32_t>(static_cast<uint32_t>(inputs.size()));
  for (const auto &r : inputs)
    writeRegion(w, r);
  w.put<uint32_t>(static_cast<uint32_t>(outputs.size()));
  for (const auto &r : outputs)
    writeRegion(w, r);

  w.put<uint32_t>(static_cast<uint32_t>(constants.size()));
  for (size_t i = 0; i < constants.size(); ++i) {
    writeRegion(w, constants[i]);
    w.put<uint32_t>(static_cast<uint32_t>(constantData[i].size()));
    for (float f : constantData[i])
      w.put<float>(f);
  }

  w.put<uint32_t>(static_cast<uint32_t>(instructions.size()));
  for (const auto &in : instructions)
    writeInstr(w, in);
  return w.bytes;
}

std::optional<Program> Program::decode(const std::vector<uint8_t> &bytes) {
  Reader r(bytes);
  char magic[4];
  for (char &c : magic)
    c = static_cast<char>(r.get<uint8_t>());
  if (!r.ok || std::memcmp(magic, kMagic, 4) != 0)
    return std::nullopt;

  Program p;
  p.version = r.get<uint32_t>();
  p.scratchpadBytes = r.get<int64_t>();
  p.dramBytes = r.get<int64_t>();

  uint32_t nIn = r.getCount();
  for (uint32_t i = 0; i < nIn && r.ok; ++i)
    p.inputs.push_back(readRegion(r));
  uint32_t nOut = r.getCount();
  for (uint32_t i = 0; i < nOut && r.ok; ++i)
    p.outputs.push_back(readRegion(r));

  uint32_t nConst = r.getCount();
  for (uint32_t i = 0; i < nConst && r.ok; ++i) {
    p.constants.push_back(readRegion(r));
    uint32_t n = r.getCount();
    std::vector<float> data(n);
    for (uint32_t j = 0; j < n; ++j)
      data[j] = r.get<float>();
    p.constantData.push_back(std::move(data));
  }

  uint32_t nInstr = r.getCount();
  for (uint32_t i = 0; i < nInstr && r.ok; ++i)
    p.instructions.push_back(readInstr(r));

  if (!r.ok)
    return std::nullopt;
  return p;
}

} // namespace npu
