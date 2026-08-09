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
  p.scratchpadBytes = 32; // input 4 fp32 at 0, result 4 fp32 at 16
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
  p.scratchpadBytes = 48; // lhs 4 fp32 at 0, rhs at 16, result at 32
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
  p.scratchpadBytes = 80; // input 9 fp32 at 0, weight 4 at 36, result 4 at 64
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
  p.scratchpadBytes = 64; // two inputs at 0 and 16, add at 32, mul at 48
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
  p.scratchpadBytes = 20; // input 4 fp32 at 0, the single result fp32 at 16
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
  p.scratchpadBytes = 32; // source 4 fp32 at 0, reshaped copy 4 fp32 at 16
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

TEST(Simulator, RefusesAWriteJustPastTheScratchpadEnd) {
  // The result ends one fp32 past the declared scratchpad, and nothing else
  // about the program is wrong. The simulator used to grow the scratchpad to
  // cover whatever the instructions referenced, which made this run clean and
  // left the bounds check with nothing to catch on a scratchpad write.
  auto programWith = [](int64_t scratchpadBytes) {
    Program p;
    p.dramBytes = 32;
    p.scratchpadBytes = scratchpadBytes;
    p.inputs.push_back({0, {4}});
    p.outputs.push_back({16, {4}});
    Instruction relu;
    relu.op = Opcode::Relu;
    relu.resultAddr = 20; // spans [20, 36), 4 bytes past a 32 byte scratchpad
    relu.resultShape = {4};
    relu.operandAddrs = {0};
    p.instructions = {load(0, {4}, 0), relu, store(20, {4}, 16), halt()};
    return p;
  };

  Program tooSmall = programWith(32);
  SimResult refused = Simulator(tooSmall).run({{-1.0f, 2.0f, -3.0f, 4.0f}});
  EXPECT_FALSE(refused.error.empty());
  EXPECT_NE(refused.error.find("instruction 1"), std::string::npos);
  EXPECT_NE(refused.error.find("scratchpad"), std::string::npos);
  // The size in the diagnostic is the declared one, not one grown to fit.
  EXPECT_NE(refused.error.find("32 byte region"), std::string::npos);

  // Control: the same instructions with a scratchpad that genuinely holds them
  // run clean, so it is the declared size doing the refusing above and not
  // something else about the program.
  Program bigEnough = programWith(36);
  SimResult ok = Simulator(bigEnough).run({{-1.0f, 2.0f, -3.0f, 4.0f}});
  EXPECT_TRUE(ok.error.empty()) << ok.error;
  EXPECT_EQ(ok.outputs[0], (std::vector<float>{0.0f, 2.0f, 0.0f, 4.0f}));
}

TEST(Simulator, ScratchpadIsSizedFromTheDeclaredFieldOnly) {
  // The declaration is what provides the room. These instructions work in the
  // last four cells of a 4096 byte declaration while referencing nothing in
  // between, so the run can only succeed if the scratchpad was sized from the
  // declared field. This pins the other direction of the change: sizing is not
  // narrowed to what the instructions happen to reference either.
  Program p;
  p.dramBytes = 32;
  p.scratchpadBytes = 4096;
  p.inputs.push_back({0, {4}});
  p.outputs.push_back({16, {4}});
  Instruction relu;
  relu.op = Opcode::Relu;
  relu.resultAddr = 4080; // spans [4080, 4096), exactly filling the declaration
  relu.resultShape = {4};
  relu.operandAddrs = {0};
  p.instructions = {load(0, {4}, 0), relu, store(4080, {4}, 16), halt()};

  SimResult r = Simulator(p).run({{-1.0f, 2.0f, -3.0f, 4.0f}});
  EXPECT_TRUE(r.error.empty()) << r.error;
  EXPECT_EQ(r.outputs[0], (std::vector<float>{0.0f, 2.0f, 0.0f, 4.0f}));
}

TEST(CostModelArithmetic, MatchesFormulas) {
  CostModel c;
  EXPECT_EQ(c.dmaCycles(64), 64 / 16 + 1);
  EXPECT_EQ(c.macCycles(256), 256 / 256 + 1);
  EXPECT_EQ(c.macCycles(257), (257 + 255) / 256 + 1);
  EXPECT_EQ(c.elementwiseCycles(16), 16 / 16 + 1);
}
