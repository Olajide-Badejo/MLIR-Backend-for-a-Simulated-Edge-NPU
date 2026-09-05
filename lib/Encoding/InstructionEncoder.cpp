//===- InstructionEncoder.cpp - npuisa IR to .nbin ----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Allocated `npuisa` IR in, a `Program` out.
//
// Three things about the shape of this file are worth stating before the code,
// because each of them is a decision rather than an implementation detail.
//
// **Addresses come from one place.** A scratchpad address is
// `computeBufferRange(value)->offset`, which is the same function the
// asynchronous overlap rule uses. There is deliberately no second
// implementation of "where does this buffer live", because two of them would
// eventually disagree and the disagreement would be a wrong answer rather than
// a crash.
//
// **The DRAM map is laid out here and nowhere else.** Inputs, then outputs,
// then constants, then the allocator's spill slots, each aligned. The P5
// extension of `docs/ARCHITECTURE.md` obliges the encoder to give every
// `npuisa.spill_slot` allocation an address the way it already does for the
// other three, and marking them in the IR is what makes finding them a
// predicate rather than an analysis.
//
// **The asynchronous forms encode to the synchronous opcodes.** There is no
// asynchronous opcode and there is not going to be one: the machine's DMA
// engine is asynchronous and the synchronous instruction is the engine start
// followed by the wait, so the dialect level distinction is about where the
// wait goes and it disappears here. `npuisa.await` encodes to nothing at all.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/InstructionEncoder.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstring>

using namespace mlir;
using namespace nbin;

namespace {

/// Every scratchpad offset the allocator assigns is a multiple of 64 bytes,
/// and the DRAM map uses the same figure so that a DMA between the two never
/// starts on a different alignment at each end.
constexpr uint64_t kDramAlignment = 64;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
  uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

/// The ONNX node name a location carries, or an empty string.
///
/// The importer gives every operation it creates a `NameLoc` holding the node
/// name, so this is where the debug section's data comes from. A `FusedLoc` is
/// searched because the lowering fuses locations when it decomposes one
/// operation into several, and the first name found is the originating node,
/// which is the mapping Section 9.1 asks for.
llvm::StringRef nameFromLocation(Location loc) {
  if (auto named = dyn_cast<NameLoc>(loc))
    return named.getName().strref();
  if (auto fused = dyn_cast<FusedLoc>(loc))
    for (Location nested : fused.getLocations())
      if (llvm::StringRef name = nameFromLocation(nested); !name.empty())
        return name;
  return {};
}

class FunctionEncoder {
public:
  FunctionEncoder(func::FuncOp function, Program &program, bool stripDebug)
      : function(function), program(program), stripDebug(stripDebug) {}

  LogicalResult run();

private:
  LogicalResult readScratchpadSize();
  LogicalResult layOutDram();
  LogicalResult findArena();
  LogicalResult encodeBody();

  /// The DRAM region a value denotes, laid out by `layOutDram`.
  FailureOr<int64_t> dramAddressOf(Value value, Operation *at);
  /// The scratchpad byte offset of a view over the arena.
  FailureOr<int64_t> scratchpadAddressOf(Value value, Operation *at);

  FailureOr<Operand> makeOperand(Value value, Operation *at);
  /// Fills the result fields of `instruction` from a destination memref.
  LogicalResult setResult(Instruction &instruction, Value value,
                          Operation *at);

  FailureOr<ElemType> elemTypeOf(Type type, Operation *at);
  FailureOr<MemSpace> spaceOf(MemRefType type, Operation *at);
  /// The strides of a memref, refusing a layout this encoder cannot represent.
  ///
  /// `layoutOffset` receives the layout's own element offset. It is an output
  /// rather than a refusal from P13: a `memref.subview` carries the offset of
  /// the sub region it names, and the caller has to reconcile it against the
  /// address the view chain walk produced rather than add the two.
  FailureOr<SmallVector<int64_t>> stridesOf(MemRefType type, Operation *at,
                                            int64_t &layoutOffset);
  FailureOr<MemRegion> regionFor(Value value, Operation *at);
  LogicalResult appendConstantData(npuisa::ConstOp op, Constant &constant);

  void record(Operation *op) {
    if (stripDebug)
      return;
    llvm::StringRef name = nameFromLocation(op->getLoc());
    if (name.empty())
      return;
    program.debug.push_back(
        DebugEntry{static_cast<uint32_t>(program.instructions.size() - 1),
                   name.str()});
  }

  func::FuncOp function;
  Program &program;
  bool stripDebug;

  Value arena;
  llvm::DenseMap<Value, int64_t> dramAddresses;
};

//===----------------------------------------------------------------------===//
// Types, spaces and strides.
//===----------------------------------------------------------------------===//

FailureOr<ElemType> FunctionEncoder::elemTypeOf(Type type, Operation *at) {
  if (type.isF32())
    return ElemType::F32;
  if (type.isInteger(8))
    return ElemType::I8;
  if (type.isInteger(32))
    return ElemType::I32;
  at->emitError() << "cannot encode the element type " << type
                  << ": the format carries f32, i8 and i32";
  return failure();
}

FailureOr<MemSpace> FunctionEncoder::spaceOf(MemRefType type, Operation *at) {
  Attribute space = type.getMemorySpace();
  if (isa_and_present<mlir::npu::ScratchpadAttr>(space))
    return MemSpace::Scratchpad;
  if (isa_and_present<mlir::npu::DramAttr>(space))
    return MemSpace::Dram;
  at->emitError() << "cannot encode a buffer with memory space " << type
                  << ": every buffer below the tensor level lives in "
                     "#npu.scratchpad or #npu.dram";
  return failure();
}

FailureOr<SmallVector<int64_t>>
FunctionEncoder::stridesOf(MemRefType type, Operation *at,
                           int64_t &layoutOffset) {
  if (!type.hasStaticShape()) {
    at->emitError() << "cannot encode the dynamically shaped buffer " << type;
    return failure();
  }
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset))) {
    at->emitError() << "cannot encode the layout of " << type
                    << ": the format carries a stride per dimension and this "
                       "layout is not strided";
    return failure();
  }
  if (ShapedType::isDynamic(offset) || offset < 0) {
    at->emitError() << "cannot encode " << type
                    << ": its layout offset must be a non negative constant";
    return failure();
  }
  // **This used to be a refusal and P13 turned it into an output.** The old
  // reasoning was sound and its last sentence was what expired: the address of
  // a buffer comes from the view chain that `computeBufferRange` walks, a
  // layout offset would be a second place the address came from, adding the two
  // would be guessing, and *nothing in this pipeline emitted one*. A tiled
  // program emits one per tile. The two are not two addresses, they are the
  // same number written twice, because the walk computes a subview's offset as
  // the dot product of its offsets with the source's strides and that is
  // exactly what the layout records. The caller reconciles them and uses the
  // walk, which is the one that also works for `memref.view` over the
  // allocator's arena, where the layout offset is zero and the address is not.
  layoutOffset = offset;
  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride) || stride < 0) {
      at->emitError() << "cannot encode " << type
                      << ": every stride must be a non negative constant";
      return failure();
    }
  return strides;
}

//===----------------------------------------------------------------------===//
// Addresses.
//===----------------------------------------------------------------------===//

FailureOr<int64_t> FunctionEncoder::dramAddressOf(Value value, Operation *at) {
  // **The DRAM side walks the view chain from P13, which is what the scratchpad
  // side has done since P5.** Before tiling, nothing produced a view of a DRAM
  // buffer, so a lookup was the whole answer and the asymmetry cost nothing. A
  // tiled program produces one view per tile, and a sub region of an argument
  // is a DRAM value this map will never hold: the map holds whole regions,
  // because a region is what the header declares and what the loader places.
  //
  // So the address of a DRAM buffer is the address of the region it is a view
  // of, plus the bytes the walk accumulates. `computeBufferRange` is the same
  // analysis `scratchpadAddressOf` uses and it already understood
  // `memref.subview`; only this side had not asked it.
  //
  // A value the walk cannot analyse falls back to being looked up directly,
  // which is exactly the old behaviour for every value that is its own base.
  std::optional<npuisa::BufferRange> range = npuisa::computeBufferRange(value);
  Value base = range ? range->base : value;
  int64_t offset = range ? range->offset : 0;

  auto entry = dramAddresses.find(base);
  if (entry == dramAddresses.end()) {
    at->emitError() << "this DRAM buffer has no address in the DRAM map. The "
                       "map holds the function's arguments, the npuisa.const "
                       "results, and the allocator's npuisa.spill_slot "
                       "allocations, and nothing else may live off chip. A "
                       "view of one of those is resolved to the region it "
                       "views, so this is a buffer that is not derived from "
                       "any of them";
    return failure();
  }
  return entry->second + offset;
}

FailureOr<int64_t> FunctionEncoder::scratchpadAddressOf(Value value,
                                                        Operation *at) {
  std::optional<npuisa::BufferRange> range = npuisa::computeBufferRange(value);
  if (!range) {
    at->emitError() << "cannot compute the scratchpad address of this buffer: "
                    << npuisa::describeWhyNotAnalysable(value);
    return failure();
  }
  if (!arena || range->base != arena) {
    at->emitError() << "this scratchpad buffer is not a view over the "
                       "allocator's arena. Run -npu-allocate-scratchpad before "
                       "encoding: the encoder reads offsets out of the IR and "
                       "does not assign them";
    return failure();
  }
  return range->offset;
}

FailureOr<Operand> FunctionEncoder::makeOperand(Value value, Operation *at) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type) {
    at->emitError() << "cannot encode an operand of type " << value.getType();
    return failure();
  }

  FailureOr<MemSpace> space = spaceOf(type, at);
  if (failed(space))
    return failure();
  FailureOr<ElemType> elementType = elemTypeOf(type.getElementType(), at);
  if (failed(elementType))
    return failure();
  int64_t layoutOffset = 0;
  FailureOr<SmallVector<int64_t>> strides = stridesOf(type, at, layoutOffset);
  if (failed(strides))
    return failure();

  FailureOr<int64_t> address = *space == MemSpace::Dram
                                   ? dramAddressOf(value, at)
                                   : scratchpadAddressOf(value, at);
  if (failed(address))
    return failure();

  // **The layout offset is reconciled against the address rather than added to
  // it.** Both address paths walk the view chain, and for a `memref.subview`
  // the walk's contribution and the layout's offset are the same number written
  // twice; adding them would double the offset and address the wrong bytes.
  // What has to be checked is the case the walk did not see: a non zero layout
  // offset on a value whose address did **not** come from a walk is an offset
  // nothing accounted for, and encoding it would be encoding the base as though
  // it were the sub region. That is the refusal this used to make
  // unconditionally, kept for exactly the case it was right about.
  if (layoutOffset != 0 && !npuisa::computeBufferRange(value)) {
    at->emitError() << "cannot encode " << type
                    << ": its layout carries a non zero offset and its view "
                       "chain could not be analysed, so nothing accounts for "
                       "that offset. "
                    << npuisa::describeWhyNotAnalysable(value);
    return failure();
  }

  Operand operand;
  operand.space = *space;
  operand.elementType = *elementType;
  operand.address = *address;
  operand.shape.assign(type.getShape().begin(), type.getShape().end());
  operand.strides.assign(strides->begin(), strides->end());
  return operand;
}

LogicalResult FunctionEncoder::setResult(Instruction &instruction, Value value,
                                         Operation *at) {
  FailureOr<Operand> destination = makeOperand(value, at);
  if (failed(destination))
    return failure();
  // **All five fields, from version 2.** Until then this copied four and
  // dropped the strides, and the drop was invisible: a destination that was a
  // view of a larger buffer encoded as though it were the whole of a smaller
  // one, so a `DMA_STORE` into a sub region wrote a contiguous block instead of
  // scattering. Nothing caught it, because an output region is written and
  // never read, and the only reason it never produced a wrong answer is that
  // nothing emitted a strided destination until tiling did. D-0050 carries the
  // reproduction: 160 of 512 elements, into the wrong channels.
  instruction.resultSpace = destination->space;
  instruction.resultElementType = destination->elementType;
  instruction.resultAddress = destination->address;
  instruction.resultShape = destination->shape;
  instruction.resultStrides = destination->strides;
  return success();
}

//===----------------------------------------------------------------------===//
// The DRAM map.
//===----------------------------------------------------------------------===//

FailureOr<MemRegion> FunctionEncoder::regionFor(Value value, Operation *at) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type) {
    at->emitError() << "expected a memref and found " << value.getType();
    return failure();
  }
  if (!type.getLayout().isIdentity()) {
    at->emitError() << "cannot place " << type
                    << " in the DRAM map: a region's extents describe a "
                       "contiguous buffer, and this one carries a layout map";
    return failure();
  }
  FailureOr<ElemType> elementType = elemTypeOf(type.getElementType(), at);
  if (failed(elementType))
    return failure();

  MemRegion region;
  region.elementType = *elementType;
  region.shape.assign(type.getShape().begin(), type.getShape().end());
  if (region.byteSize() < 0) {
    at->emitError() << "cannot place " << type
                    << " in the DRAM map: its element count is zero or above "
                       "the shape limit";
    return failure();
  }
  return region;
}

LogicalResult FunctionEncoder::appendConstantData(npuisa::ConstOp op,
                                                  Constant &constant) {
  auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
  if (!dense) {
    op.emitError() << "cannot encode a constant whose value is not a dense "
                      "elements attribute";
    return failure();
  }

  // The value iterator is used rather than the raw data on purpose. A splat
  // stores one element and has to be expanded, and a raw copy of a splat would
  // write four bytes where the region needs four megabytes, which the
  // `constant-data` check would catch but only after the file had been
  // written.
  switch (constant.region.elementType) {
  case ElemType::F32: {
    for (const APFloat &value : dense.getValues<APFloat>()) {
      float raw = value.convertToFloat();
      const auto *bytes = reinterpret_cast<const uint8_t *>(&raw);
      constant.data.insert(constant.data.end(), bytes, bytes + sizeof(float));
    }
    break;
  }
  case ElemType::I8: {
    for (const APInt &value : dense.getValues<APInt>())
      constant.data.push_back(static_cast<uint8_t>(value.getSExtValue()));
    break;
  }
  case ElemType::I32: {
    for (const APInt &value : dense.getValues<APInt>()) {
      auto raw = static_cast<int32_t>(value.getSExtValue());
      const auto *bytes = reinterpret_cast<const uint8_t *>(&raw);
      constant.data.insert(constant.data.end(), bytes, bytes + sizeof(int32_t));
    }
    break;
  }
  }
  return success();
}

LogicalResult FunctionEncoder::layOutDram() {
  uint64_t cursor = 0;
  auto place = [&](MemRegion &region) {
    cursor = alignUp(cursor, kDramAlignment);
    region.offset = cursor;
    cursor += static_cast<uint64_t>(region.byteSize());
  };

  // The arguments, in order, with the kind read from the attribute rather than
  // inferred from the position. The positional rule still holds and is checked
  // below rather than assumed.
  bool seenOutput = false;
  for (BlockArgument argument : function.getArguments()) {
    unsigned index = argument.getArgNumber();
    auto kind = function.getArgAttrOfType<StringAttr>(index, npuisa::kArgKindAttrName);
    if (!kind) {
      function.emitError()
          << "argument " << index << " carries no `" << npuisa::kArgKindAttrName
          << "` attribute. Every argument of an encodable function says "
             "whether it is an input region or an output region, because the "
             "binary declares the two separately and a positional convention "
             "is a convention the encoder cannot check";
      return failure();
    }
    bool isOutput = kind.getValue() == npuisa::kArgKindOut;
    if (!isOutput && kind.getValue() != npuisa::kArgKindIn) {
      function.emitError() << "argument " << index << " has `"
                           << npuisa::kArgKindAttrName << " = \"" << kind.getValue()
                           << "\"`, and the two values are \"" << npuisa::kArgKindIn
                           << "\" and \"" << npuisa::kArgKindOut << "\"";
      return failure();
    }
    if (isOutput)
      seenOutput = true;
    else if (seenOutput) {
      function.emitError()
          << "argument " << index << " is an input and follows an output. A "
             "lowered function takes its outputs as trailing arguments, so "
             "argument N of the lowered function is argument N of the model";
      return failure();
    }

    FailureOr<MemRegion> region = regionFor(argument, function);
    if (failed(region))
      return failure();
    if (isOutput)
      program.outputs.push_back(*region);
    else
      program.inputs.push_back(*region);
  }

  for (MemRegion &region : program.inputs)
    place(region);
  for (MemRegion &region : program.outputs)
    place(region);
  // The addresses, mapped back onto the values. The two loops above assigned
  // offsets in argument order within each list, so walking the arguments again
  // in the same order recovers the pairing without a second data structure.
  {
    size_t nextInput = 0;
    size_t nextOutput = 0;
    for (BlockArgument argument : function.getArguments()) {
      auto kind = function.getArgAttrOfType<StringAttr>(argument.getArgNumber(),
                                                        npuisa::kArgKindAttrName);
      if (kind.getValue() == npuisa::kArgKindOut)
        dramAddresses[argument] =
            static_cast<int64_t>(program.outputs[nextOutput++].offset);
      else
        dramAddresses[argument] =
            static_cast<int64_t>(program.inputs[nextInput++].offset);
    }
  }

  // The constants, in the order they appear.
  WalkResult walked = function.walk([&](npuisa::ConstOp op) {
    FailureOr<MemRegion> region = regionFor(op.getResult(), op);
    if (failed(region))
      return WalkResult::interrupt();
    Constant constant;
    constant.region = *region;
    place(constant.region);
    if (failed(appendConstantData(op, constant)))
      return WalkResult::interrupt();
    dramAddresses[op.getResult()] =
        static_cast<int64_t>(constant.region.offset);
    program.constants.push_back(std::move(constant));
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();

  // The allocator's spill slots. They are marked in the IR rather than
  // inferred, per the P5 extension of docs/ARCHITECTURE.md, so finding them is
  // a predicate and not an analysis.
  walked = function.walk([&](memref::AllocOp op) {
    if (!op->hasAttr("npuisa.spill_slot"))
      return WalkResult::advance();
    FailureOr<MemRegion> region = regionFor(op.getResult(), op);
    if (failed(region))
      return WalkResult::interrupt();
    place(*region);
    dramAddresses[op.getResult()] = static_cast<int64_t>(region->offset);
    program.spillSlots.push_back(*region);
    return WalkResult::advance();
  });
  if (walked.wasInterrupted())
    return failure();

  program.dramBytes = alignUp(cursor, kDramAlignment);
  return success();
}

//===----------------------------------------------------------------------===//
// The body.
//===----------------------------------------------------------------------===//

LogicalResult FunctionEncoder::readScratchpadSize() {
  auto attr = function->getAttrOfType<IntegerAttr>("npuisa.scratchpad_bytes");
  if (!attr) {
    function.emitError()
        << "the function carries no `npuisa.scratchpad_bytes` attribute. It is "
           "written by -npu-allocate-scratchpad and it is what Section 9.3 "
           "calls scratchpadBytes: the simulator sizes its scratchpad strictly "
           "from it and never grows it to cover the writes it finds";
    return failure();
  }
  int64_t bytes = attr.getInt();
  if (bytes < 0) {
    function.emitError() << "`npuisa.scratchpad_bytes` is " << bytes;
    return failure();
  }
  program.scratchpadBytes = static_cast<uint64_t>(bytes);
  return success();
}

LogicalResult FunctionEncoder::findArena() {
  WalkResult walked = function.walk([&](memref::AllocOp op) {
    if (!op->hasAttr("npuisa.scratchpad_arena"))
      return WalkResult::advance();
    if (arena) {
      op.emitError() << "a function has one scratchpad arena and this is the "
                        "second";
      return WalkResult::interrupt();
    }
    arena = op.getResult();
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

LogicalResult FunctionEncoder::encodeBody() {
  Block &body = function.getBody().front();

  for (Operation &op : body) {
    Instruction instruction;

    // The operations that carry no instruction. Each is listed by name rather
    // than skipped by a default, so an operation this encoder has never seen
    // is a diagnostic instead of a silently dropped computation.
    if (isa<memref::AllocOp, memref::ViewOp, memref::ReinterpretCastOp,
            memref::SubViewOp, arith::ConstantOp, npuisa::ConstOp,
            npuisa::AwaitOp, func::ReturnOp>(&op))
      continue;

    if (auto load = dyn_cast<npuisa::DmaLoadOp>(&op)) {
      instruction.opcode = Opcode::DMA_LOAD;
      FailureOr<Operand> source = makeOperand(load.getSource(), &op);
      if (failed(source) || failed(setResult(instruction, load.getDest(), &op)))
        return failure();
      instruction.operands.push_back(*source);
    } else if (auto load = dyn_cast<npuisa::DmaLoadAsyncOp>(&op)) {
      // The asynchronous form encodes to the synchronous opcode. There is no
      // asynchronous opcode: the wait is where the two forms differ and the
      // wait is a property of the dialect level, not of the machine.
      instruction.opcode = Opcode::DMA_LOAD;
      FailureOr<Operand> source = makeOperand(load.getSource(), &op);
      if (failed(source) || failed(setResult(instruction, load.getDest(), &op)))
        return failure();
      instruction.operands.push_back(*source);
    } else if (auto store = dyn_cast<npuisa::DmaStoreOp>(&op)) {
      instruction.opcode = Opcode::DMA_STORE;
      FailureOr<Operand> source = makeOperand(store.getSource(), &op);
      if (failed(source) ||
          failed(setResult(instruction, store.getDest(), &op)))
        return failure();
      instruction.operands.push_back(*source);
    } else if (auto store = dyn_cast<npuisa::DmaStoreAsyncOp>(&op)) {
      instruction.opcode = Opcode::DMA_STORE;
      FailureOr<Operand> source = makeOperand(store.getSource(), &op);
      if (failed(source) ||
          failed(setResult(instruction, store.getDest(), &op)))
        return failure();
      instruction.operands.push_back(*source);
    } else if (auto matmul = dyn_cast<npuisa::MatMulOp>(&op)) {
      instruction.opcode = Opcode::MATMUL;
      for (Value input : {matmul.getLhs(), matmul.getRhs()}) {
        FailureOr<Operand> operand = makeOperand(input, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (Value bias = matmul.getBias()) {
        FailureOr<Operand> operand = makeOperand(bias, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (failed(setResult(instruction, matmul.getDestination(), &op)))
        return failure();
    } else if (auto conv = dyn_cast<npuisa::Conv2DOp>(&op)) {
      instruction.opcode = Opcode::CONV2D;
      for (Value input : {conv.getInput(), conv.getFilter()}) {
        FailureOr<Operand> operand = makeOperand(input, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (Value bias = conv.getBias()) {
        FailureOr<Operand> operand = makeOperand(bias, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      instruction.strides.assign(conv.getStrides().begin(),
                                 conv.getStrides().end());
      instruction.pads.assign(conv.getPads().begin(), conv.getPads().end());
      instruction.dilations.assign(conv.getDilations().begin(),
                                   conv.getDilations().end());
      instruction.group = conv.getGroup();
      if (failed(setResult(instruction, conv.getDestination(), &op)))
        return failure();
    } else if (auto add = dyn_cast<npuisa::AddOp>(&op)) {
      instruction.opcode = Opcode::ADD;
      for (Value input : {add.getLhs(), add.getRhs()}) {
        FailureOr<Operand> operand = makeOperand(input, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (failed(setResult(instruction, add.getDestination(), &op)))
        return failure();
    } else if (auto mul = dyn_cast<npuisa::MulOp>(&op)) {
      instruction.opcode = Opcode::MUL;
      for (Value input : {mul.getLhs(), mul.getRhs()}) {
        FailureOr<Operand> operand = makeOperand(input, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (failed(setResult(instruction, mul.getDestination(), &op)))
        return failure();
    } else if (auto relu = dyn_cast<npuisa::ReluOp>(&op)) {
      instruction.opcode = Opcode::RELU;
      FailureOr<Operand> operand = makeOperand(relu.getInput(), &op);
      if (failed(operand) ||
          failed(setResult(instruction, relu.getDestination(), &op)))
        return failure();
      instruction.operands.push_back(*operand);
    } else if (auto pool = dyn_cast<npuisa::PoolMaxOp>(&op)) {
      instruction.opcode = Opcode::POOL_MAX;
      FailureOr<Operand> operand = makeOperand(pool.getInput(), &op);
      if (failed(operand) ||
          failed(setResult(instruction, pool.getDestination(), &op)))
        return failure();
      instruction.operands.push_back(*operand);
      instruction.kernel.assign(pool.getKernel().begin(),
                                pool.getKernel().end());
      instruction.strides.assign(pool.getStrides().begin(),
                                 pool.getStrides().end());
      instruction.pads.assign(pool.getPads().begin(), pool.getPads().end());
      instruction.dilations.assign(pool.getDilations().begin(),
                                   pool.getDilations().end());
    } else if (auto pool = dyn_cast<npuisa::PoolAvgOp>(&op)) {
      instruction.opcode = Opcode::POOL_AVG;
      FailureOr<Operand> operand = makeOperand(pool.getInput(), &op);
      if (failed(operand) ||
          failed(setResult(instruction, pool.getDestination(), &op)))
        return failure();
      instruction.operands.push_back(*operand);
      instruction.kernel.assign(pool.getKernel().begin(),
                                pool.getKernel().end());
      instruction.strides.assign(pool.getStrides().begin(),
                                 pool.getStrides().end());
      instruction.pads.assign(pool.getPads().begin(), pool.getPads().end());
      instruction.dilations.assign(pool.getDilations().begin(),
                                   pool.getDilations().end());
    } else if (auto reshape = dyn_cast<npuisa::ReshapeOp>(&op)) {
      instruction.opcode = Opcode::RESHAPE;
      FailureOr<Operand> operand = makeOperand(reshape.getInput(), &op);
      if (failed(operand) ||
          failed(setResult(instruction, reshape.getDestination(), &op)))
        return failure();
      instruction.operands.push_back(*operand);
    } else if (auto transpose = dyn_cast<npuisa::TransposeOp>(&op)) {
      instruction.opcode = Opcode::TRANSPOSE;
      FailureOr<Operand> operand = makeOperand(transpose.getInput(), &op);
      if (failed(operand) ||
          failed(setResult(instruction, transpose.getDestination(), &op)))
        return failure();
      instruction.operands.push_back(*operand);
      instruction.axes.assign(transpose.getPermutation().begin(),
                              transpose.getPermutation().end());
    } else if (auto concat = dyn_cast<npuisa::ConcatOp>(&op)) {
      instruction.opcode = Opcode::CONCAT;
      for (Value input : concat.getInputs()) {
        FailureOr<Operand> operand = makeOperand(input, &op);
        if (failed(operand))
          return failure();
        instruction.operands.push_back(*operand);
      }
      if (failed(setResult(instruction, concat.getDestination(), &op)))
        return failure();
      instruction.axes.push_back(concat.getAxis());
    } else {
      op.emitError() << "cannot encode this operation. The encoder knows the "
                        "npuisa instructions, the memref view operations the "
                        "allocator emits, and func.return, and refuses "
                        "everything else rather than dropping it";
      return failure();
    }

    program.instructions.push_back(std::move(instruction));
    record(&op);
  }

  // Every program ends with HALT. It carries no location, so it gets no debug
  // entry, which is why `record` is not called for it.
  program.instructions.push_back(Instruction{});
  program.instructions.back().opcode = Opcode::HALT;
  return success();
}

LogicalResult FunctionEncoder::run() {
  if (function.isExternal()) {
    function.emitError() << "cannot encode a declaration: it has no body";
    return failure();
  }
  if (!function.getBody().hasOneBlock()) {
    function.emitError()
        << "cannot encode a function with more than one block. The instruction "
           "set of Section 5.4 has no branch, so a program is one straight "
           "line";
    return failure();
  }
  if (function.getNumResults() != 0) {
    function.emitError()
        << "cannot encode a function that returns a value. A lowered function "
           "takes its outputs as trailing DRAM arguments and returns nothing";
    return failure();
  }

  if (failed(readScratchpadSize()) || failed(findArena()) ||
      failed(layOutDram()) || failed(encodeBody()))
    return failure();
  return success();
}

} // namespace

LogicalResult nbin::encodeModule(ModuleOp module, Program &program,
                                bool stripDebug) {
  program = Program();

  SmallVector<func::FuncOp> functions;
  module.walk([&](func::FuncOp function) { functions.push_back(function); });

  if (functions.empty()) {
    module.emitError() << "the module holds no function, so there is nothing "
                          "to encode";
    return failure();
  }
  if (functions.size() != 1) {
    // Diagnosed rather than truncated. A `.nbin` holds one program, and
    // quietly encoding the first of two functions produces a file that runs
    // and computes the wrong model.
    InFlightDiagnostic error = module.emitError()
                               << "the module holds " << functions.size()
                               << " functions and a .nbin holds one program";
    for (func::FuncOp function : functions)
      error.attachNote(function.getLoc()) << "function @" << function.getName();
    return failure();
  }

  FunctionEncoder encoder(functions.front(), program, stripDebug);
  return encoder.run();
}
