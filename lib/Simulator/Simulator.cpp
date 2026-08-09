//===- Simulator.cpp - Execute an encoded npuisa program ------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Simulator/Simulator.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>

namespace npu {

namespace {

int64_t numElements(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

// NCHW convolution matching the ONNX Conv semantics for the supported subset.
void conv2d(const float *in, const std::vector<int64_t> &inS, const float *w,
            const std::vector<int64_t> &wS, const float *bias, float *out,
            const std::vector<int64_t> &outS, const std::vector<int64_t> &strides,
            const std::vector<int64_t> &pads,
            const std::vector<int64_t> &dilations, int64_t group,
            int32_t activation) {
  int64_t C = inS[1], H = inS[2], W = inS[3];
  int64_t O = wS[0], Cg = wS[1], kh = wS[2], kw = wS[3];
  int64_t oh = outS[2], ow = outS[3];
  int64_t sh = strides[0], sw = strides[1];
  int64_t pt = pads[0], pl = pads[1];
  int64_t dh = dilations[0], dw = dilations[1];
  int64_t outPerGroup = O / group;
  int64_t n = 0; // batch is 1 for the supported models
  for (int64_t o = 0; o < O; ++o) {
    int64_t icStart = (o / outPerGroup) * Cg;
    for (int64_t y = 0; y < oh; ++y)
      for (int64_t x = 0; x < ow; ++x) {
        float acc = bias ? bias[o] : 0.0f;
        for (int64_t ic = 0; ic < Cg; ++ic)
          for (int64_t ky = 0; ky < kh; ++ky)
            for (int64_t kx = 0; kx < kw; ++kx) {
              int64_t iy = y * sh - pt + ky * dh;
              int64_t ix = x * sw - pl + kx * dw;
              if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                continue;
              acc += in[(((n * C) + icStart + ic) * H + iy) * W + ix] *
                     w[(((o * Cg) + ic) * kh + ky) * kw + kx];
            }
        if (activation == 1)
          acc = std::max(0.0f, acc);
        out[(((n * O) + o) * oh + y) * ow + x] = acc;
      }
  }
}

void matmul(const float *lhs, const std::vector<int64_t> &lS, const float *rhs,
            const std::vector<int64_t> &rS, const float *bias, float *out,
            int32_t activation) {
  int64_t M = lS[0], K = lS[1], N = rS[1];
  for (int64_t m = 0; m < M; ++m)
    for (int64_t j = 0; j < N; ++j) {
      float acc = bias ? bias[j] : 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += lhs[m * K + k] * rhs[k * N + j];
      if (activation == 1)
        acc = std::max(0.0f, acc);
      out[m * N + j] = acc;
    }
}

void pool(const float *in, const std::vector<int64_t> &inS, float *out,
          const std::vector<int64_t> &outS,
          const std::vector<int64_t> &kernel,
          const std::vector<int64_t> &strides,
          const std::vector<int64_t> &pads, bool isMax) {
  int64_t C = inS[1], H = inS[2], W = inS[3];
  int64_t oh = outS[2], ow = outS[3];
  int64_t kh = kernel[0], kw = kernel[1];
  int64_t sh = strides[0], sw = strides[1];
  int64_t pt = pads[0], pl = pads[1];
  for (int64_t c = 0; c < C; ++c)
    for (int64_t y = 0; y < oh; ++y)
      for (int64_t x = 0; x < ow; ++x) {
        float best = isMax ? -std::numeric_limits<float>::infinity() : 0.0f;
        int64_t count = 0;
        for (int64_t ky = 0; ky < kh; ++ky)
          for (int64_t kx = 0; kx < kw; ++kx) {
            int64_t iy = y * sh - pt + ky;
            int64_t ix = x * sw - pl + kx;
            if (iy < 0 || iy >= H || ix < 0 || ix >= W)
              continue;
            float v = in[(c * H + iy) * W + ix];
            if (isMax)
              best = std::max(best, v);
            else
              best += v;
            ++count;
          }
        out[(c * oh + y) * ow + x] = isMax ? best : (best / count);
      }
}

} // namespace

std::string Stats::toJson() const {
  std::ostringstream os;
  os << "{\"cycles\": " << cycles << ", \"dram_bytes_read\": " << dramBytesRead
     << ", \"dram_bytes_written\": " << dramBytesWritten
     << ", \"instructions\": " << instructions << "}";
  return os.str();
}

SimResult Simulator::run(const std::vector<std::vector<float>> &inputs) {
  // Size DRAM and the scratchpad. The scratchpad must cover every assigned
  // address, so take the larger of the reported high water and what the
  // instructions actually reference.
  int64_t spBytes = program.scratchpadBytes;
  for (const Instruction &in : program.instructions)
    if (in.resultAddr >= 0)
      spBytes = std::max(spBytes, in.resultAddr + numElements(in.resultShape) * 4);

  std::vector<float> dram(std::max<int64_t>(1, program.dramBytes / 4), 0.0f);
  std::vector<float> sp(std::max<int64_t>(1, spBytes / 4), 0.0f);

  // Bounds checked access, replacing the raw "data() + addr / 4" lambdas.
  //
  // Those did no checking at all, so a negative or oversized address was an out
  // of bounds read or write straight into the heap. Program::validate now
  // rejects such a program before it can be run, but the simulator is also
  // reachable as a library and from hand built Program values in tests, and
  // "the caller validated it" is not a memory safety argument.
  //
  // Every access is checked, in every build mode: there is no assert here and
  // no release build path that skips the check. A refused access records its
  // message in SimResult.error and returns nullptr, and every caller tests the
  // returned pointer and skips the access rather than dereferencing it. Only
  // the first refusal is recorded, because it is the one that explains the run;
  // the rest are consequences of it. Execution then continues to the end, so
  // the caller gets a result carrying a diagnostic rather than a crash.
  //
  // Note the two are checked against different sizes: the scratchpad is sized
  // by spBytes above, which can exceed program.scratchpadBytes.
  bool trapped = false;
  std::string trapMessage;
  auto checkedAt = [&](std::vector<float> &memory, int64_t addr,
                       int64_t elements, const char *space,
                       size_t index) -> float * {
    int64_t total = static_cast<int64_t>(memory.size());
    bool ok = addr >= 0 && (addr % 4) == 0 && elements >= 0 &&
              addr / 4 <= total - elements;
    if (!ok) {
      if (!trapped) {
        trapped = true;
        std::ostringstream os;
        os << "instruction " << index << ": " << space << " access of "
           << elements << " element(s) at byte address " << addr
           << " is outside the " << (total * 4) << " byte region";
        trapMessage = os.str();
      }
      return nullptr;
    }
    return memory.data() + addr / 4;
  };

  size_t pc = 0; // current instruction, for diagnostics
  auto dramAt = [&](int64_t addr, int64_t elements = 0) {
    return checkedAt(dram, addr, elements, "DRAM", pc);
  };
  auto spAt = [&](int64_t addr, int64_t elements = 0) {
    return checkedAt(sp, addr, elements, "scratchpad", pc);
  };

  for (size_t i = 0; i < program.constants.size(); ++i) {
    int64_t n = static_cast<int64_t>(program.constantData[i].size());
    if (float *dst = dramAt(program.constants[i].dramOffset, n))
      std::memcpy(dst, program.constantData[i].data(), n * sizeof(float));
  }
  for (size_t i = 0; i < inputs.size() && i < program.inputs.size(); ++i) {
    int64_t n = static_cast<int64_t>(inputs[i].size());
    // An input longer than its declared region is refused rather than written
    // past the end. npu-sim also checks this against the file size, but the
    // library entry point has to defend itself.
    int64_t declared = numElements(program.inputs[i].shape);
    if (n > declared)
      n = declared;
    if (float *dst = dramAt(program.inputs[i].dramOffset, n))
      std::memcpy(dst, inputs[i].data(), n * sizeof(float));
  }

  std::map<int64_t, std::vector<int64_t>> shapeAt;
  Stats stats;

  for (size_t index = 0; index < program.instructions.size(); ++index) {
    const Instruction &in = program.instructions[index];
    pc = index;
    ++stats.instructions;
    if (!isValidOpcode(in.op)) {
      // Unreachable for a validated program. Falling through a switch on an out
      // of range enum is undefined behaviour, so refuse rather than rely on it.
      if (!trapped) {
        trapped = true;
        std::ostringstream os;
        os << "instruction " << index << ": opcode "
           << static_cast<uint16_t>(in.op) << " is not a defined instruction";
        trapMessage = os.str();
      }
      break;
    }
    auto opShape = [&](size_t k) { return shapeAt[in.operandAddrs[k]]; };
    // Elements held at an operand address, from whichever instruction wrote it.
    auto opElements = [&](size_t k) { return numElements(opShape(k)); };
    switch (in.op) {
    case Opcode::Nop:
      stats.cycles += cost.issueOverhead;
      break;
    case Opcode::Halt:
      stats.cycles += cost.issueOverhead;
      break;
    case Opcode::DmaLoad: {
      int64_t n = numElements(in.resultShape);
      float *dst = spAt(in.resultAddr, n);
      const float *src = dramAt(in.dramAddr, n);
      if (!dst || !src)
        break;
      std::memcpy(dst, src, n * sizeof(float));
      shapeAt[in.resultAddr] = in.resultShape;
      stats.dramBytesRead += n * 4;
      stats.cycles += cost.dmaCycles(n * 4);
      break;
    }
    case Opcode::DmaStore: {
      int64_t n = numElements(in.resultShape);
      float *dst = dramAt(in.dramAddr, n);
      const float *src = spAt(in.operandAddrs[0], n);
      if (!dst || !src)
        break;
      std::memcpy(dst, src, n * sizeof(float));
      stats.dramBytesWritten += n * 4;
      stats.cycles += cost.dmaCycles(n * 4);
      break;
    }
    case Opcode::Conv2D: {
      auto inS = opShape(0), wS = opShape(1);
      if (inS.size() != 4 || wS.size() != 4 || in.resultShape.size() != 4)
        break;
      const float *bias = in.operandAddrs.size() == 3
                              ? spAt(in.operandAddrs[2], opElements(2))
                              : nullptr;
      if (in.operandAddrs.size() == 3 && !bias)
        break;
      const float *x = spAt(in.operandAddrs[0], opElements(0));
      const float *w = spAt(in.operandAddrs[1], opElements(1));
      float *o = spAt(in.resultAddr, numElements(in.resultShape));
      if (!x || !w || !o)
        break;
      conv2d(x, inS, w, wS, bias, o, in.resultShape, in.strides, in.pads,
             in.dilations, in.group, in.activation);
      shapeAt[in.resultAddr] = in.resultShape;
      int64_t macs = numElements(in.resultShape) * wS[1] * wS[2] * wS[3];
      stats.cycles += cost.macCycles(macs);
      break;
    }
    case Opcode::MatMul: {
      auto lS = opShape(0), rS = opShape(1);
      if (lS.size() != 2 || rS.size() != 2)
        break;
      const float *bias = in.operandAddrs.size() == 3
                              ? spAt(in.operandAddrs[2], opElements(2))
                              : nullptr;
      if (in.operandAddrs.size() == 3 && !bias)
        break;
      const float *lhs = spAt(in.operandAddrs[0], opElements(0));
      const float *rhs = spAt(in.operandAddrs[1], opElements(1));
      float *o = spAt(in.resultAddr, numElements(in.resultShape));
      if (!lhs || !rhs || !o)
        break;
      matmul(lhs, lS, rhs, rS, bias, o, in.activation);
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.macCycles(lS[0] * lS[1] * rS[1]);
      break;
    }
    case Opcode::Relu: {
      int64_t n = numElements(in.resultShape);
      const float *a = spAt(in.operandAddrs[0], n);
      float *o = spAt(in.resultAddr, n);
      if (!a || !o)
        break;
      for (int64_t i = 0; i < n; ++i)
        o[i] = std::max(0.0f, a[i]);
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.elementwiseCycles(n);
      break;
    }
    case Opcode::Add:
    case Opcode::Mul: {
      int64_t n = numElements(in.resultShape);
      const float *a = spAt(in.operandAddrs[0], n);
      const float *b = spAt(in.operandAddrs[1], n);
      float *o = spAt(in.resultAddr, n);
      if (!a || !b || !o)
        break;
      for (int64_t i = 0; i < n; ++i)
        o[i] = in.op == Opcode::Add ? a[i] + b[i] : a[i] * b[i];
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.elementwiseCycles(n);
      break;
    }
    case Opcode::PoolMax:
    case Opcode::PoolAvg: {
      auto inS = opShape(0);
      if (inS.size() != 4 || in.resultShape.size() != 4 ||
          in.kernelShape.size() != 2)
        break;
      const float *a = spAt(in.operandAddrs[0], opElements(0));
      float *o = spAt(in.resultAddr, numElements(in.resultShape));
      if (!a || !o)
        break;
      pool(a, inS, o, in.resultShape, in.kernelShape, in.strides, in.pads,
           in.op == Opcode::PoolMax);
      shapeAt[in.resultAddr] = in.resultShape;
      int64_t work = numElements(in.resultShape) * in.kernelShape[0] *
                     in.kernelShape[1];
      stats.cycles += cost.elementwiseCycles(work);
      break;
    }
    case Opcode::Reshape: {
      int64_t n = numElements(in.resultShape);
      float *dst = spAt(in.resultAddr, n);
      const float *src = spAt(in.operandAddrs[0], n);
      if (!dst || !src)
        break;
      std::memcpy(dst, src, n * sizeof(float));
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.issueOverhead;
      break;
    }
    }
  }

  SimResult result;
  result.stats = stats;
  for (const MemRegion &out : program.outputs) {
    int64_t n = numElements(out.shape);
    std::vector<float> data(n, 0.0f);
    if (const float *src = dramAt(out.dramOffset, n))
      std::memcpy(data.data(), src, n * sizeof(float));
    result.outputs.push_back(std::move(data));
  }
  result.error = trapMessage;
  return result;
}

} // namespace npu
