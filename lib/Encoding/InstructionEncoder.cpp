//===- InstructionEncoder.cpp - npuisa MLIR to Program --------------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/InstructionEncoder.h"

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace npu {

namespace {

std::vector<int64_t> toVec(ArrayAttr attr) {
  std::vector<int64_t> v;
  for (Attribute a : attr)
    v.push_back(llvm::cast<IntegerAttr>(a).getInt());
  return v;
}

std::vector<int64_t> shapeOf(Type t) {
  if (auto buffer = llvm::dyn_cast<npuisa::BufferType>(t))
    t = buffer.getTensorType();
  auto ranked = llvm::cast<RankedTensorType>(t);
  return std::vector<int64_t>(ranked.getShape().begin(),
                              ranked.getShape().end());
}

// Scratchpad byte offset recorded on a buffer producing instruction.
int64_t scratchpadAddr(Value buffer) {
  Operation *def = buffer.getDefiningOp();
  std::optional<int64_t> addr;
  llvm::TypeSwitch<Operation *>(def)
      .Case<npuisa::DmaLoadOp, npuisa::Conv2DOp, npuisa::MatMulOp,
            npuisa::ReluOp, npuisa::ReshapeOp, npuisa::AddOp, npuisa::MulOp,
            npuisa::PoolMaxOp, npuisa::PoolAvgOp>(
          [&](auto op) { addr = op.getAddress(); });
  return addr.value_or(-1);
}

} // namespace

FailureOr<Program> encodeFunction(func::FuncOp func) {
  if (func.getBody().empty())
    return func.emitError("cannot encode an empty function");
  Block &block = func.getBody().front();

  Program program;
  DenseMap<Value, int64_t> dramOffset; // DRAM tensor value -> byte offset
  int64_t dram = 0;

  // Inputs: the function arguments, in order.
  for (BlockArgument arg : block.getArguments()) {
    MemRegion region{dram, shapeOf(arg.getType())};
    dramOffset[arg] = dram;
    dram += region.byteSize();
    program.inputs.push_back(region);
  }

  // Constants: each npuisa.const gets a DRAM region and its data.
  for (Operation &op : block) {
    if (auto c = dyn_cast<npuisa::ConstOp>(op)) {
      MemRegion region{dram, shapeOf(c.getType())};
      std::vector<float> data;
      auto elements = llvm::cast<DenseElementsAttr>(c.getValue());
      for (APFloat f : elements.getValues<APFloat>())
        data.push_back(f.convertToFloat());
      dramOffset[c.getResult()] = dram;
      dram += region.byteSize();
      program.constants.push_back(region);
      program.constantData.push_back(std::move(data));
    }
  }

  // Outputs: the values returned, each produced by a dma_store.
  auto returnOp = cast<func::ReturnOp>(block.getTerminator());
  DenseMap<Value, int64_t> outputOffset;
  for (Value ret : returnOp.getOperands()) {
    MemRegion region{dram, shapeOf(ret.getType())};
    outputOffset[ret] = dram;
    dram += region.byteSize();
    program.outputs.push_back(region);
  }
  program.dramBytes = dram;

  if (auto attr = func->getAttrOfType<IntegerAttr>("npuisa.scratchpad_bytes"))
    program.scratchpadBytes = attr.getInt();

  // Instructions, in program order. Constants are data, not instructions.
  for (Operation &op : block) {
    Instruction in;
    bool emit = true;
    llvm::TypeSwitch<Operation *>(&op)
        .Case<npuisa::ConstOp, func::ReturnOp>([&](auto) { emit = false; })
        .Case<npuisa::NopOp>([&](auto) { in.op = Opcode::Nop; })
        .Case<npuisa::HaltOp>([&](auto) { in.op = Opcode::Halt; })
        .Case<npuisa::DmaLoadOp>([&](npuisa::DmaLoadOp o) {
          in.op = Opcode::DmaLoad;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.dramAddr = dramOffset.lookup(o.getSource());
        })
        .Case<npuisa::DmaStoreOp>([&](npuisa::DmaStoreOp o) {
          in.op = Opcode::DmaStore;
          in.resultShape = shapeOf(o.getSource().getType());
          in.operandAddrs = {scratchpadAddr(o.getSource())};
          in.dramAddr = outputOffset.lookup(o.getResult());
        })
        .Case<npuisa::Conv2DOp>([&](npuisa::Conv2DOp o) {
          in.op = Opcode::Conv2D;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getInput()),
                             scratchpadAddr(o.getWeight())};
          if (o.getBias())
            in.operandAddrs.push_back(scratchpadAddr(o.getBias()));
          in.strides = toVec(o.getStridesAttr());
          in.pads = toVec(o.getPadsAttr());
          in.dilations = toVec(o.getDilationsAttr());
          in.group = o.getGroup();
          in.activation = o.getActivation();
        })
        .Case<npuisa::MatMulOp>([&](npuisa::MatMulOp o) {
          in.op = Opcode::MatMul;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getLhs()),
                             scratchpadAddr(o.getRhs())};
          if (o.getBias())
            in.operandAddrs.push_back(scratchpadAddr(o.getBias()));
          in.activation = o.getActivation();
        })
        .Case<npuisa::ReluOp>([&](npuisa::ReluOp o) {
          in.op = Opcode::Relu;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getInput())};
        })
        .Case<npuisa::ReshapeOp>([&](npuisa::ReshapeOp o) {
          in.op = Opcode::Reshape;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getInput())};
        })
        .Case<npuisa::AddOp>([&](npuisa::AddOp o) {
          in.op = Opcode::Add;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getLhs()),
                             scratchpadAddr(o.getRhs())};
        })
        .Case<npuisa::MulOp>([&](npuisa::MulOp o) {
          in.op = Opcode::Mul;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getLhs()),
                             scratchpadAddr(o.getRhs())};
        })
        .Case<npuisa::PoolMaxOp>([&](npuisa::PoolMaxOp o) {
          in.op = Opcode::PoolMax;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getInput())};
          in.kernelShape = toVec(o.getKernelShapeAttr());
          in.strides = toVec(o.getStridesAttr());
          in.pads = toVec(o.getPadsAttr());
        })
        .Case<npuisa::PoolAvgOp>([&](npuisa::PoolAvgOp o) {
          in.op = Opcode::PoolAvg;
          in.resultAddr = o.getAddress().value_or(-1);
          in.resultShape = shapeOf(o.getType());
          in.operandAddrs = {scratchpadAddr(o.getInput())};
          in.kernelShape = toVec(o.getKernelShapeAttr());
          in.strides = toVec(o.getStridesAttr());
          in.pads = toVec(o.getPadsAttr());
        })
        .Default([&](Operation *bad) {
          emit = false;
          bad->emitError("cannot encode unexpected op");
        });
    if (emit)
      program.instructions.push_back(std::move(in));
  }

  program.instructions.push_back(Instruction{Opcode::Halt});
  return program;
}

} // namespace npu
