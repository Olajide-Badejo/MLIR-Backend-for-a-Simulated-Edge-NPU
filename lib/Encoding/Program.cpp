//===- Program.cpp - .nbin encode and decode ------------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"

#include <cstring>
#include <map>
#include <sstream>

namespace npu {

bool isValidOpcode(Opcode op) {
  return static_cast<uint16_t>(op) <= kMaxOpcode;
}

std::string ValidationError::toString() const {
  std::ostringstream os;
  if (instructionIndex == kProgramLevel)
    os << "program: ";
  else
    os << "instruction " << instructionIndex << ": ";
  os << check << ": " << detail;
  return os.str();
}

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

// Byte stream helpers. These copy the object representation straight in and out
// of the stream, so the encoding is host byte order, not a fixed endianness.
// On every machine this project targets that is little endian, but a .nbin is
// not portable across byte orders and nothing here swaps. See the byte order
// section of docs/ISA_MANUAL.md.
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
  size_t remaining() const { return static_cast<size_t>(end - p); }

  // Bound element counts so a corrupt length cannot request a huge allocation.
  //
  // The 2^28 cap alone was not a bound on work. At the cap a single shape
  // vector is 2 GiB, and nothing stopped a thirty byte file from naming one,
  // because the count was believed before the bytes behind it were known to
  // exist. minBytesPerElement is the smallest space one element can occupy in
  // the stream, so a count the remaining bytes cannot possibly back is a
  // truncated file, and it is refused here rather than after something has been
  // sized from it. A well formed file always carries those bytes, so this
  // rejects nothing the format permits.
  uint32_t getCount(size_t minBytesPerElement) {
    uint32_t n = get<uint32_t>();
    if (!ok)
      return 0;
    if (n > (1u << 28) || n > remaining() / minBytesPerElement) {
      ok = false;
      return 0;
    }
    return n;
  }
  std::vector<int64_t> getVec() {
    uint32_t n = getCount(sizeof(int64_t));
    std::vector<int64_t> v(n);
    for (uint32_t i = 0; i < n; ++i)
      v[i] = get<int64_t>();
    return v;
  }
};

constexpr char kMagic[4] = {'N', 'P', 'U', 'B'};

// The least space one element of each repeated section can occupy in the
// stream, used above to refuse a count the file cannot back. A region is an
// int64 offset plus a shape count; a constant is a region plus its data count;
// an instruction is its opcode, four scalars, and the counts of its six
// vectors. These are lower bounds, so they stay correct if a field grows.
constexpr size_t kMinRegionBytes = sizeof(int64_t) + sizeof(uint32_t);
constexpr size_t kMinConstantBytes = kMinRegionBytes + sizeof(uint32_t);
constexpr size_t kMinInstructionBytes = sizeof(uint16_t) + sizeof(int64_t) +
                                        sizeof(int64_t) + sizeof(int32_t) +
                                        sizeof(int64_t) + 6 * sizeof(uint32_t);

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

namespace {

// Element count of a shape, or nullopt if the shape is unusable: empty, holding
// a non positive extent, or holding a product above the 2^40 element cap. The
// cap matters because the byte size is compared against a region bound, and a
// product that wrapped int64 compares as small. Each extent is tested against
// the headroom that is left before it is multiplied in, so the guard never
// performs the overflow it exists to catch. The cap is inclusive: a product of
// exactly 2^40 is returned, and the caller's own size check is what refuses it.
std::optional<int64_t> shapeElements(const std::vector<int64_t> &shape) {
  if (shape.empty())
    return std::nullopt;
  constexpr int64_t kLimit = int64_t{1} << 40;
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0 || d > kLimit)
      return std::nullopt;
    if (n > kLimit / d)
      return std::nullopt;
    n *= d;
  }
  return n;
}

// What each opcode requires. Written out rather than inferred so that adding an
// opcode forces a decision about its arity and whether it touches memory.
struct OpcodeShape {
  int minOperands;
  int maxOperands;
  bool writesScratchpad; // needs a resultAddr inside the scratchpad
  bool touchesDram;      // needs a dramAddr inside DRAM
};

std::optional<OpcodeShape> opcodeShape(Opcode op) {
  switch (op) {
  case Opcode::Nop:
  case Opcode::Halt:
    return OpcodeShape{0, 0, false, false};
  case Opcode::DmaLoad:
    return OpcodeShape{0, 0, true, true};
  case Opcode::DmaStore:
    return OpcodeShape{1, 1, false, true};
  case Opcode::Conv2D:
  case Opcode::MatMul:
    // Two operands, or three when a bias is present. Conv2D used to index
    // operandAddrs[1] unconditionally, so a one operand CONV2D read out of
    // bounds rather than being rejected.
    return OpcodeShape{2, 3, true, false};
  case Opcode::Relu:
  case Opcode::Reshape:
  case Opcode::PoolMax:
  case Opcode::PoolAvg:
    return OpcodeShape{1, 1, true, false};
  case Opcode::Add:
  case Opcode::Mul:
    return OpcodeShape{2, 2, true, false};
  }
  return std::nullopt;
}

} // namespace

std::optional<ValidationError> Program::validate() const {
  constexpr size_t kProgram = ValidationError::kProgramLevel;
  auto fail = [](size_t index, const char *check,
                 const std::string &detail) -> ValidationError {
    return ValidationError{index, check, detail};
  };
  auto num = [](int64_t v) { return std::to_string(v); };

  if (version != kVersion)
    return fail(kProgram, "version",
                "file declares version " + std::to_string(version) +
                    " but this build understands version " +
                    std::to_string(kVersion));
  if (scratchpadBytes < 0)
    return fail(kProgram, "scratchpad-size",
                "negative scratchpad size " + num(scratchpadBytes));
  if (scratchpadBytes > kMaxScratchpadBytes)
    return fail(kProgram, "scratchpad-size",
                "declared scratchpad of " + num(scratchpadBytes) +
                    " bytes exceeds the format limit of " +
                    num(kMaxScratchpadBytes));
  if (dramBytes < 0)
    return fail(kProgram, "dram-size", "negative DRAM size " + num(dramBytes));
  if (dramBytes > kMaxDramBytes)
    return fail(kProgram, "dram-size",
                "declared DRAM of " + num(dramBytes) +
                    " bytes exceeds the format limit of " + num(kMaxDramBytes));

  // Every declared region has to describe a real, in bounds piece of DRAM.
  auto checkRegion = [&](const MemRegion &r, const char *what,
                         size_t i) -> std::optional<ValidationError> {
    std::string where = std::string(what) + " " + std::to_string(i);
    std::optional<int64_t> n = shapeElements(r.shape);
    if (!n)
      return fail(kProgram, "region-shape",
                  where + " has an empty shape or a non positive extent");
    if (r.dramOffset < 0)
      return fail(kProgram, "region-offset",
                  where + " has negative DRAM offset " + num(r.dramOffset));
    // Everything is fp32 and every access indexes as addr / 4, so a misaligned
    // offset would silently read across element boundaries.
    if (r.dramOffset % 4 != 0)
      return fail(kProgram, "region-offset",
                  where + " has DRAM offset " + num(r.dramOffset) +
                      ", which is not 4 byte aligned");
    if (*n * 4 > dramBytes - r.dramOffset)
      return fail(kProgram, "region-in-range",
                  where + " spans DRAM [" + num(r.dramOffset) + ", " +
                      num(r.dramOffset + *n * 4) + ") but DRAM is " +
                      num(dramBytes) + " bytes");
    return std::nullopt;
  };
  for (size_t i = 0; i < inputs.size(); ++i)
    if (auto bad = checkRegion(inputs[i], "input", i))
      return bad;
  for (size_t i = 0; i < outputs.size(); ++i)
    if (auto bad = checkRegion(outputs[i], "output", i))
      return bad;
  for (size_t i = 0; i < constants.size(); ++i)
    if (auto bad = checkRegion(constants[i], "constant", i))
      return bad;

  if (constantData.size() != constants.size())
    return fail(kProgram, "constant-data",
                "there are " + std::to_string(constants.size()) +
                    " constant regions but " +
                    std::to_string(constantData.size()) + " data blocks");
  for (size_t i = 0; i < constants.size(); ++i) {
    int64_t n = *shapeElements(constants[i].shape);
    if (static_cast<int64_t>(constantData[i].size()) != n)
      return fail(kProgram, "constant-data",
                  "constant " + std::to_string(i) + " has shape holding " +
                      num(n) + " elements but carries " +
                      std::to_string(constantData[i].size()) + " floats");
  }

  // Walk the instructions, tracking what each scratchpad address holds. This
  // mirrors what the simulator does at run time, so reading an address nothing
  // has written is caught here rather than becoming a silent zero read.
  std::map<int64_t, int64_t> writtenElements;
  for (size_t i = 0; i < instructions.size(); ++i) {
    const Instruction &in = instructions[i];

    if (!isValidOpcode(in.op))
      return fail(i, "opcode",
                  "opcode " + std::to_string(static_cast<uint16_t>(in.op)) +
                      " is outside the range 0 to " +
                      std::to_string(kMaxOpcode));
    OpcodeShape want = *opcodeShape(in.op);
    const char *name = opcodeName(in.op);

    int operands = static_cast<int>(in.operandAddrs.size());
    if (operands < want.minOperands || operands > want.maxOperands)
      return fail(i, "arity",
                  std::string(name) + " takes " +
                      std::to_string(want.minOperands) +
                      (want.maxOperands != want.minOperands
                           ? " to " + std::to_string(want.maxOperands)
                           : "") +
                      " operands but has " + std::to_string(operands));

    int64_t resultElements = 0;
    if (want.writesScratchpad || in.op == Opcode::DmaStore) {
      std::optional<int64_t> n = shapeElements(in.resultShape);
      if (!n)
        return fail(i, "result-shape",
                    std::string(name) +
                        " has an empty result shape or a non positive extent");
      resultElements = *n;
    }

    if (want.writesScratchpad) {
      if (in.resultAddr < 0)
        return fail(i, "result-address",
                    std::string(name) + " has negative scratchpad address " +
                        num(in.resultAddr));
      if (in.resultAddr % 4 != 0)
        return fail(i, "result-address",
                    std::string(name) + " writes scratchpad address " +
                        num(in.resultAddr) + ", which is not 4 byte aligned");
      if (resultElements * 4 > scratchpadBytes - in.resultAddr)
        return fail(i, "result-in-range",
                    std::string(name) + " writes scratchpad [" +
                        num(in.resultAddr) + ", " +
                        num(in.resultAddr + resultElements * 4) +
                        ") but the scratchpad is " + num(scratchpadBytes) +
                        " bytes");
    }

    // How many elements this consumer reads from each operand address. The
    // written before read walk above only establishes that something wrote the
    // address; it says nothing about how much. Reading more than was written is
    // a read off the end of a live buffer when the address is near the top of
    // the scratchpad, and a silent read of stale data when it is not. Only the
    // first of those would ever trap, which is why membership alone was not
    // enough.
    //
    // CONV2D and MATMUL are deliberately not given a count. Their operand
    // extents follow from the recorded tensor shapes, and this walk tracks
    // element counts rather than shapes, so deriving them would mean
    // reproducing convolution shape inference here. The rule implemented for
    // them is the weaker one: every operand the kernel indexes must carry a non
    // zero recorded count. Full extent checking for those two needs shape
    // tracking and is not attempted here.
    //
    // The pooling bound is a lower bound. A pool reads its input window, which
    // is at least as large as its output for every configuration the backend
    // emits. A heavily padded pool whose output is larger than its input would
    // be rejected by this, and nothing in the compiler produces one.
    int64_t operandNeeds = 0;
    switch (in.op) {
    case Opcode::DmaStore:
    case Opcode::Relu:
    case Opcode::Add:
    case Opcode::Mul:
    case Opcode::Reshape:
    case Opcode::PoolMax:
    case Opcode::PoolAvg:
      operandNeeds = resultElements;
      break;
    case Opcode::Conv2D:
    case Opcode::MatMul:
      operandNeeds = 1;
      break;
    case Opcode::Nop:
    case Opcode::Halt:
    case Opcode::DmaLoad:
      break;
    }

    for (int k = 0; k < operands; ++k) {
      int64_t addr = in.operandAddrs[k];
      if (addr < 0 || addr >= scratchpadBytes)
        return fail(i, "operand-in-range",
                    std::string(name) + " operand " + std::to_string(k) +
                        " is at scratchpad address " + num(addr) +
                        ", outside [0, " + num(scratchpadBytes) + ")");
      auto known = writtenElements.find(addr);
      if (known == writtenElements.end())
        return fail(i, "operand-defined",
                    std::string(name) + " operand " + std::to_string(k) +
                        " reads scratchpad address " + num(addr) +
                        ", which no earlier instruction wrote, so its shape is "
                        "unknown");
      if (known->second < operandNeeds)
        return fail(i, "operand-extent",
                    std::string(name) + " operand " + std::to_string(k) +
                        " reads " + num(operandNeeds) +
                        " element(s) from scratchpad address " + num(addr) +
                        ", but only " + num(known->second) +
                        " element(s) were written there");
    }

    if (want.touchesDram) {
      if (in.dramAddr < 0)
        return fail(i, "dram-address",
                    std::string(name) + " has negative DRAM address " +
                        num(in.dramAddr));
      if (resultElements * 4 > dramBytes - in.dramAddr)
        return fail(i, "dram-in-range",
                    std::string(name) + " touches DRAM [" + num(in.dramAddr) +
                        ", " + num(in.dramAddr + resultElements * 4) +
                        ") but DRAM is " + num(dramBytes) + " bytes");
    }

    // Attribute vectors the kernels index without checking.
    auto needVector = [&](const std::vector<int64_t> &v, size_t want_,
                          const char *what) -> std::optional<ValidationError> {
      if (v.size() != want_)
        return fail(i, "attribute-size",
                    std::string(name) + " needs a " + std::to_string(want_) +
                        " element " + what + " but has " +
                        std::to_string(v.size()));
      for (int64_t d : v)
        if (d < 0)
          return fail(i, "attribute-value",
                      std::string(name) + " has a negative " + what + " entry");
      return std::nullopt;
    };
    if (in.op == Opcode::Conv2D) {
      if (auto bad = needVector(in.strides, 2, "strides"))
        return bad;
      if (auto bad = needVector(in.pads, 4, "pads"))
        return bad;
      if (auto bad = needVector(in.dilations, 2, "dilations"))
        return bad;
      if (in.strides[0] == 0 || in.strides[1] == 0)
        return fail(i, "attribute-value", "CONV2D stride is zero");
      if (in.dilations[0] == 0 || in.dilations[1] == 0)
        return fail(i, "attribute-value", "CONV2D dilation is zero");
      if (in.group < 1)
        return fail(i, "group",
                    "CONV2D group is " + num(in.group) + ", must be at least 1");
    }
    if (in.op == Opcode::PoolMax || in.op == Opcode::PoolAvg) {
      if (auto bad = needVector(in.kernelShape, 2, "kernel_shape"))
        return bad;
      if (auto bad = needVector(in.strides, 2, "strides"))
        return bad;
      if (auto bad = needVector(in.pads, 4, "pads"))
        return bad;
      if (in.kernelShape[0] == 0 || in.kernelShape[1] == 0)
        return fail(i, "attribute-value", "pool kernel_shape is zero");
      if (in.strides[0] == 0 || in.strides[1] == 0)
        return fail(i, "attribute-value", "pool stride is zero");
    }
    if (in.op == Opcode::Conv2D || in.op == Opcode::MatMul) {
      if (in.activation != 0 && in.activation != 1)
        return fail(i, "activation",
                    "activation " + std::to_string(in.activation) +
                        " is not none (0) or relu (1)");
    }

    if (want.writesScratchpad)
      writtenElements[in.resultAddr] = resultElements;
  }

  return std::nullopt;
}

std::optional<Program> Program::decode(const std::vector<uint8_t> &bytes,
                                       ValidationError *error) {
  std::optional<Program> raw = decodeUnvalidated(bytes);
  if (!raw) {
    if (error)
      *error = ValidationError{ValidationError::kProgramLevel, "structure",
                               "byte stream is truncated, or does not begin "
                               "with the NPUB magic"};
    return std::nullopt;
  }
  if (std::optional<ValidationError> bad = raw->validate()) {
    if (error)
      *error = *bad;
    return std::nullopt;
  }
  return raw;
}

std::optional<Program>
Program::decodeUnvalidated(const std::vector<uint8_t> &bytes) {
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

  uint32_t nIn = r.getCount(kMinRegionBytes);
  for (uint32_t i = 0; i < nIn && r.ok; ++i)
    p.inputs.push_back(readRegion(r));
  uint32_t nOut = r.getCount(kMinRegionBytes);
  for (uint32_t i = 0; i < nOut && r.ok; ++i)
    p.outputs.push_back(readRegion(r));

  uint32_t nConst = r.getCount(kMinConstantBytes);
  for (uint32_t i = 0; i < nConst && r.ok; ++i) {
    p.constants.push_back(readRegion(r));
    uint32_t n = r.getCount(sizeof(float));
    std::vector<float> data(n);
    for (uint32_t j = 0; j < n; ++j)
      data[j] = r.get<float>();
    p.constantData.push_back(std::move(data));
  }

  uint32_t nInstr = r.getCount(kMinInstructionBytes);
  for (uint32_t i = 0; i < nInstr && r.ok; ++i)
    p.instructions.push_back(readInstr(r));

  if (!r.ok)
    return std::nullopt;
  return p;
}

} // namespace npu
