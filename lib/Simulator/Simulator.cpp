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

  auto dramAt = [&](int64_t addr) { return dram.data() + addr / 4; };
  auto spAt = [&](int64_t addr) { return sp.data() + addr / 4; };

  for (size_t i = 0; i < program.constants.size(); ++i)
    std::memcpy(dramAt(program.constants[i].dramOffset),
                program.constantData[i].data(),
                program.constantData[i].size() * sizeof(float));
  for (size_t i = 0; i < inputs.size() && i < program.inputs.size(); ++i)
    std::memcpy(dramAt(program.inputs[i].dramOffset), inputs[i].data(),
                inputs[i].size() * sizeof(float));

  std::map<int64_t, std::vector<int64_t>> shapeAt;
  Stats stats;

  for (const Instruction &in : program.instructions) {
    ++stats.instructions;
    auto opShape = [&](size_t k) { return shapeAt[in.operandAddrs[k]]; };
    switch (in.op) {
    case Opcode::Nop:
      stats.cycles += cost.issueOverhead;
      break;
    case Opcode::Halt:
      stats.cycles += cost.issueOverhead;
      break;
    case Opcode::DmaLoad: {
      int64_t n = numElements(in.resultShape);
      std::memcpy(spAt(in.resultAddr), dramAt(in.dramAddr), n * sizeof(float));
      shapeAt[in.resultAddr] = in.resultShape;
      stats.dramBytesRead += n * 4;
      stats.cycles += cost.dmaCycles(n * 4);
      break;
    }
    case Opcode::DmaStore: {
      int64_t n = numElements(in.resultShape);
      std::memcpy(dramAt(in.dramAddr), spAt(in.operandAddrs[0]),
                  n * sizeof(float));
      stats.dramBytesWritten += n * 4;
      stats.cycles += cost.dmaCycles(n * 4);
      break;
    }
    case Opcode::Conv2D: {
      const float *bias =
          in.operandAddrs.size() == 3 ? spAt(in.operandAddrs[2]) : nullptr;
      conv2d(spAt(in.operandAddrs[0]), opShape(0), spAt(in.operandAddrs[1]),
             opShape(1), bias, spAt(in.resultAddr), in.resultShape, in.strides,
             in.pads, in.dilations, in.group, in.activation);
      shapeAt[in.resultAddr] = in.resultShape;
      auto wS = opShape(1);
      int64_t macs = numElements(in.resultShape) * wS[1] * wS[2] * wS[3];
      stats.cycles += cost.macCycles(macs);
      break;
    }
    case Opcode::MatMul: {
      const float *bias =
          in.operandAddrs.size() == 3 ? spAt(in.operandAddrs[2]) : nullptr;
      auto lS = opShape(0), rS = opShape(1);
      matmul(spAt(in.operandAddrs[0]), lS, spAt(in.operandAddrs[1]), rS, bias,
             spAt(in.resultAddr), in.activation);
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.macCycles(lS[0] * lS[1] * rS[1]);
      break;
    }
    case Opcode::Relu: {
      int64_t n = numElements(in.resultShape);
      const float *a = spAt(in.operandAddrs[0]);
      float *o = spAt(in.resultAddr);
      for (int64_t i = 0; i < n; ++i)
        o[i] = std::max(0.0f, a[i]);
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.elementwiseCycles(n);
      break;
    }
    case Opcode::Add:
    case Opcode::Mul: {
      int64_t n = numElements(in.resultShape);
      const float *a = spAt(in.operandAddrs[0]);
      const float *b = spAt(in.operandAddrs[1]);
      float *o = spAt(in.resultAddr);
      for (int64_t i = 0; i < n; ++i)
        o[i] = in.op == Opcode::Add ? a[i] + b[i] : a[i] * b[i];
      shapeAt[in.resultAddr] = in.resultShape;
      stats.cycles += cost.elementwiseCycles(n);
      break;
    }
    case Opcode::PoolMax:
    case Opcode::PoolAvg: {
      pool(spAt(in.operandAddrs[0]), opShape(0), spAt(in.resultAddr),
           in.resultShape, in.kernelShape, in.strides, in.pads,
           in.op == Opcode::PoolMax);
      shapeAt[in.resultAddr] = in.resultShape;
      int64_t work = numElements(in.resultShape) * in.kernelShape[0] *
                     in.kernelShape[1];
      stats.cycles += cost.elementwiseCycles(work);
      break;
    }
    case Opcode::Reshape: {
      int64_t n = numElements(in.resultShape);
      std::memcpy(spAt(in.resultAddr), spAt(in.operandAddrs[0]),
                  n * sizeof(float));
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
    std::vector<float> data(n);
    std::memcpy(data.data(), dramAt(out.dramOffset), n * sizeof(float));
    result.outputs.push_back(std::move(data));
  }
  return result;
}

} // namespace npu
