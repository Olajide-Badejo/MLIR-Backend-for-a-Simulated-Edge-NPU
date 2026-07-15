//===- SimulatorTest.cpp - Simulator semantics and cost tests -------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/CostModel.h"
#include "NPU/Simulator/Simulator.h"

#include "gtest/gtest.h"

using namespace npu;

namespace {

Instruction halt() { return Instruction{Opcode::Halt}; }

Instruction load(int64_t sp, std::vector<int64_t> shape, int64_t dram) {
  Instruction in;
  in.op = Opcode::DmaLoad;
  in.resultAddr = sp;
  in.resultShape = std::move(shape);
  in.dramAddr = dram;
  return in;
}

Instruction store(int64_t sp, std::vector<int64_t> shape, int64_t dram) {
  Instruction in;
  in.op = Opcode::DmaStore;
  in.resultShape = std::move(shape);
  in.operandAddrs = {sp};
  in.dramAddr = dram;
  return in;
}

} // namespace

TEST(Simulator, Relu) {
  Program p;
  p.dramBytes = 32;
  p.inputs.push_back({0, {4}});
  p.outputs.push_back({16, {4}});
  Instruction relu;
  relu.op = Opcode::Relu;
  relu.resultAddr = 16;
  relu.resultShape = {4};
  relu.operandAddrs = {0};
  p.instructions = {load(0, {4}, 0), relu, store(16, {4}, 16), halt()};

  SimResult r = Simulator(p).run({{-1.0f, 2.0f, -3.0f, 4.0f}});
  ASSERT_EQ(r.outputs.size(), 1u);
  EXPECT_EQ(r.outputs[0], (std::vector<float>{0.0f, 2.0f, 0.0f, 4.0f}));
  EXPECT_EQ(r.stats.dramBytesRead, 16);
  EXPECT_EQ(r.stats.dramBytesWritten, 16);
}

TEST(Simulator, MatMulIdentity) {
  Program p;
  p.dramBytes = 48;
  p.inputs.push_back({0, {2, 2}});
  p.constants.push_back({16, {2, 2}});
  p.constantData.push_back({1.0f, 0.0f, 0.0f, 1.0f}); // identity
  p.outputs.push_back({32, {2, 2}});
  Instruction mm;
  mm.op = Opcode::MatMul;
  mm.resultAddr = 32;
  mm.resultShape = {2, 2};
  mm.operandAddrs = {0, 16};
  p.instructions = {load(0, {2, 2}, 0), load(16, {2, 2}, 16), mm,
                    store(32, {2, 2}, 32), halt()};

  SimResult r = Simulator(p).run({{1.0f, 2.0f, 3.0f, 4.0f}});
  EXPECT_EQ(r.outputs[0], (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(Simulator, Conv2DKnown) {
  // 1x1x3x3 input, 1x1x2x2 weight [[1,0],[0,1]], valid, stride 1 -> 1x1x2x2.
  Program p;
  p.dramBytes = 3 * 36 + 16;
  p.inputs.push_back({0, {1, 1, 3, 3}});   // 36 bytes
  p.constants.push_back({36, {1, 1, 2, 2}}); // 16 bytes
  p.constantData.push_back({1.0f, 0.0f, 0.0f, 1.0f});
  p.outputs.push_back({52, {1, 1, 2, 2}});
  Instruction conv;
  conv.op = Opcode::Conv2D;
  conv.resultAddr = 64;
  conv.resultShape = {1, 1, 2, 2};
  conv.operandAddrs = {0, 36};
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  conv.group = 1;
  p.instructions = {load(0, {1, 1, 3, 3}, 0), load(36, {1, 1, 2, 2}, 36), conv,
                    store(64, {1, 1, 2, 2}, 52), halt()};

  SimResult r = Simulator(p).run(
      {{1, 2, 3, 4, 5, 6, 7, 8, 9}});
  EXPECT_EQ(r.outputs[0], (std::vector<float>{6, 8, 12, 14}));
}

TEST(Simulator, ElementwiseAddMul) {
  Program p;
  p.dramBytes = 64;
  p.inputs.push_back({0, {4}});
  p.inputs.push_back({16, {4}});
  p.outputs.push_back({48, {4}});
  Instruction add;
  add.op = Opcode::Add;
  add.resultAddr = 32;
  add.resultShape = {4};
  add.operandAddrs = {0, 16};
  Instruction mul;
  mul.op = Opcode::Mul;
  mul.resultAddr = 48;
  mul.resultShape = {4};
  mul.operandAddrs = {32, 0};
  p.instructions = {load(0, {4}, 0), load(16, {4}, 16), add, mul,
                    store(48, {4}, 48), halt()};

  SimResult r = Simulator(p).run({{1, 2, 3, 4}, {10, 20, 30, 40}});
  // add -> [11, 22, 33, 44], then mul by the first input [1, 2, 3, 4].
  EXPECT_EQ(r.outputs[0], (std::vector<float>{11, 44, 99, 176}));
}

TEST(Simulator, AvgPool) {
  Program p;
  p.dramBytes = 32;
  p.inputs.push_back({0, {1, 1, 2, 2}});
  p.outputs.push_back({16, {1, 1, 1, 1}});
  Instruction pool;
  pool.op = Opcode::PoolAvg;
  pool.resultAddr = 16;
  pool.resultShape = {1, 1, 1, 1};
  pool.operandAddrs = {0};
  pool.kernelShape = {2, 2};
  pool.strides = {2, 2};
  pool.pads = {0, 0, 0, 0};
  p.instructions = {load(0, {1, 1, 2, 2}, 0), pool, store(16, {1, 1, 1, 1}, 16),
                    halt()};

  SimResult r = Simulator(p).run({{1, 2, 3, 4}});
  EXPECT_EQ(r.outputs[0], (std::vector<float>{2.5f})); // mean of 1,2,3,4
}

TEST(Simulator, Reshape) {
  Program p;
  p.dramBytes = 32;
  p.inputs.push_back({0, {2, 2}});
  p.outputs.push_back({16, {4}});
  Instruction rs;
  rs.op = Opcode::Reshape;
  rs.resultAddr = 16;
  rs.resultShape = {4};
  rs.operandAddrs = {0};
  p.instructions = {load(0, {2, 2}, 0), rs, store(16, {4}, 16), halt()};

  SimResult r = Simulator(p).run({{1, 2, 3, 4}});
  EXPECT_EQ(r.outputs[0], (std::vector<float>{1, 2, 3, 4}));
}

TEST(CostModelArithmetic, MatchesFormulas) {
  CostModel c;
  EXPECT_EQ(c.dmaCycles(64), 64 / 16 + 1);
  EXPECT_EQ(c.macCycles(256), 256 / 256 + 1);
  EXPECT_EQ(c.macCycles(257), (257 + 255) / 256 + 1);
  EXPECT_EQ(c.elementwiseCycles(16), 16 / 16 + 1);
}
