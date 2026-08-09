//===- EncodingTest.cpp - Encoder and format unit tests -------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/InstructionEncoder.h"
#include "NPU/Encoding/Program.h"

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace npu;

// A small but genuinely valid program: load the input, load the weights,
// convolve, halt. Shared by the round trip and validation tests so there is one
// place that knows what a well formed program looks like.
//
// This used to be written inline with a scratchpad of 4096 bytes holding a
// 13824 byte convolution result, and a weight operand at an address no
// instruction ever wrote. Program::decode accepted it because it validated
// nothing, so the round trip test was round tripping a program the simulator
// could not have run.
static Program validProgram() {
  Program p;
  p.scratchpadBytes = 32768;
  p.dramBytes = 8192;
  p.inputs.push_back({0, {1, 1, 28, 28}});    // 3136 bytes at 0
  p.constants.push_back({4096, {6, 1, 5, 5}}); // 600 bytes at 4096
  p.constantData.push_back(std::vector<float>(150, 0.5f));
  p.outputs.push_back({8000, {1, 10}}); // 40 bytes, ending at 8040

  Instruction loadInput;
  loadInput.op = Opcode::DmaLoad;
  loadInput.resultAddr = 0;
  loadInput.resultShape = {1, 1, 28, 28};
  loadInput.dramAddr = 0;
  p.instructions.push_back(loadInput);

  Instruction loadWeight;
  loadWeight.op = Opcode::DmaLoad;
  loadWeight.resultAddr = 3136;
  loadWeight.resultShape = {6, 1, 5, 5};
  loadWeight.dramAddr = 4096;
  p.instructions.push_back(loadWeight);

  Instruction conv;
  conv.op = Opcode::Conv2D;
  conv.resultAddr = 3736;
  conv.resultShape = {1, 6, 24, 24}; // 13824 bytes, ending at 17560
  conv.operandAddrs = {0, 3136};
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  conv.activation = 1;
  p.instructions.push_back(conv);

  p.instructions.push_back(Instruction{Opcode::Halt});
  return p;
}

TEST(EncodingFormat, RoundTrip) {
  Program p = validProgram();
  ASSERT_FALSE(p.validate().has_value()) << p.validate()->toString();

  auto decoded = Program::decode(p.encode());
  ASSERT_TRUE(decoded.has_value());
  Program q = *decoded;

  EXPECT_EQ(q.version, Program::kVersion);
  EXPECT_EQ(q.scratchpadBytes, 32768);
  EXPECT_EQ(q.dramBytes, 8192);
  ASSERT_EQ(q.inputs.size(), 1u);
  EXPECT_EQ(q.inputs[0].shape, (std::vector<int64_t>{1, 1, 28, 28}));
  ASSERT_EQ(q.constantData.size(), 1u);
  EXPECT_EQ(q.constantData[0].size(), 150u);
  EXPECT_FLOAT_EQ(q.constantData[0][0], 0.5f);
  ASSERT_EQ(q.instructions.size(), 4u);
  EXPECT_EQ(q.instructions[2].op, Opcode::Conv2D);
  EXPECT_EQ(q.instructions[2].operandAddrs, (std::vector<int64_t>{0, 3136}));
  EXPECT_EQ(q.instructions[2].activation, 1);
  EXPECT_EQ(q.instructions[3].op, Opcode::Halt);
}

TEST(EncodingFormat, BadMagicRejected) {
  std::vector<uint8_t> junk = {'X', 'X', 'X', 'X', 0, 0, 0, 0};
  EXPECT_FALSE(Program::decode(junk).has_value());
}

TEST(EncodingFormat, DisassembleMentionsOpcodes) {
  Program p;
  p.inputs.push_back({0, {1, 2}});
  Instruction relu;
  relu.op = Opcode::Relu;
  relu.resultAddr = 8;
  relu.resultShape = {1, 2};
  relu.operandAddrs = {0};
  p.instructions.push_back(relu);
  std::string text = disassemble(p);
  EXPECT_NE(text.find("RELU"), std::string::npos);
}

TEST(EncodeFunction, LowersSmallProgram) {
  const char *ir = R"mlir(
    func.func @main(%x: tensor<1x2xf32>) -> tensor<1x2xf32>
        attributes {npuisa.scratchpad_bytes = 16 : i64} {
      %0 = npuisa.dma_load %x {address = 0 : i64}
        : (tensor<1x2xf32>) -> !npuisa.buffer<tensor<1x2xf32>>
      %1 = npuisa.relu %0 {address = 8 : i64}
        : (!npuisa.buffer<tensor<1x2xf32>>) -> !npuisa.buffer<tensor<1x2xf32>>
      %2 = npuisa.dma_store %1
        : (!npuisa.buffer<tensor<1x2xf32>>) -> tensor<1x2xf32>
      return %2 : tensor<1x2xf32>
    }
  )mlir";

  mlir::MLIRContext ctx;
  ctx.loadDialect<mlir::npuisa::NPUISADialect, mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
  ASSERT_TRUE(module);

  mlir::func::FuncOp func;
  module->walk([&](mlir::func::FuncOp f) { func = f; });
  auto program = encodeFunction(func);
  ASSERT_TRUE(mlir::succeeded(program));

  EXPECT_EQ(program->inputs.size(), 1u);
  EXPECT_EQ(program->outputs.size(), 1u);
  EXPECT_EQ(program->scratchpadBytes, 16);
  ASSERT_EQ(program->instructions.size(), 4u); // load, relu, store, halt
  EXPECT_EQ(program->instructions[0].op, Opcode::DmaLoad);
  EXPECT_EQ(program->instructions[1].op, Opcode::Relu);
  EXPECT_EQ(program->instructions[1].resultAddr, 8);
  EXPECT_EQ(program->instructions[2].op, Opcode::DmaStore);
  EXPECT_EQ(program->instructions[3].op, Opcode::Halt);
}
