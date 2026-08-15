//===- InstructionEncoder.cpp - npuisa MLIR to Program --------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/InstructionEncoder.h"

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
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

  auto returnOp = cast<func::ReturnOp>(block.getTerminator());
  llvm::SmallPtrSet<Value, 4> returned(returnOp.getOperands().begin(),
                                       returnOp.getOperands().end());

  // Spill temporaries: a dma_store whose result is not returned spills a buffer
  // to DRAM to be reloaded later, so it needs its own DRAM region.
  for (Operation &op : block)
    if (auto store = dyn_cast<npuisa::DmaStoreOp>(op)) {
      Value res = store.getResult();
      if (!returned.count(res)) {
        MemRegion region{dram, shapeOf(res.getType())};
        dramOffset[res] = dram;
        dram += region.byteSize();
      }
    }

  // Outputs: the values returned, each produced by a dma_store.
  for (Value ret : returnOp.getOperands()) {
    MemRegion region{dram, shapeOf(ret.getType())};
    dramOffset[ret] = dram;
    dram += region.byteSize();
    program.outputs.push_back(region);
  }
  program.dramBytes = dram;

  if (auto attr = func->getAttrOfType<IntegerAttr>("npuisa.scratchpad_bytes"))
    program.scratchpadBytes = attr.getInt();

  // Instructions, in program order. Constants are data, not instructions.
  //
  // `emit` is false for the two ops that legitimately produce no instruction, a
  // constant and the terminator. An op with no case at all is a different thing
  // and is tracked separately, because it means the encoder was handed IR it
  // does not understand and the resulting program would be missing work.
  bool unencodable = false;
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
          in.dramAddr = dramOffset.lookup(o.getResult());
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
          unencodable = true;
          bad->emitError("cannot encode unexpected op");
        });
    if (emit)
      program.instructions.push_back(std::move(in));
  }

  // The diagnostic above used to be the whole response: the op was skipped and
  // the function returned the program anyway, so npu-translate printed an error,
  // wrote the .nbin, and exited 0. The file it wrote was a program with the work
  // silently removed. Fail instead, after the loop rather than inside it, so one
  // run names every op it cannot encode instead of only the first.
  if (unencodable)
    return failure();

  program.instructions.push_back(Instruction{Opcode::Halt});
  return program;
}

} // namespace npu
