//===- InterfaceTest.cpp - npuisa interface unit tests ----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The two interfaces Section 8 puts on the `npuisa` compute operations, and the
// overlap arithmetic the asynchronous rule is decided on.
//
// These are unit tests rather than lit tests for a reason that is specific to
// what is being asserted. A lit test sees what an operation prints. None of the
// three claims here is printed: which operands `DestinationStyleOpInterface`
// calls inputs and which it calls inits, what `MemoryEffectOpInterface` reports,
// and what byte range a chain of views denotes. A lit test can only observe
// those through a diagnostic that happens to mention them, which tests the
// wording of the message and not the answer.
//
// The P2 gate names four things this file has to prove, and each has its own
// test suite below:
//
//   1. `ins` and `outs` partition the operands exactly once, on every compute
//      operation, including the ones with an optional bias.
//   2. No compute operation reports itself free of effects.
//   3. The destination is written and the inputs are read, per operand, so that
//      the overlap rule has something specific to ask about.
//   4. The overlap rule is decided on effects plus view offsets and extents,
//      and in particular it catches the partially overlapping `memref.view`
//      case that an SSA identity check misses.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <string>

using namespace mlir;
using namespace mlir::npuisa;

namespace {

//===----------------------------------------------------------------------===//
// The fixture.
//===----------------------------------------------------------------------===//

/// A context with everything these tests parse loaded into it.
///
/// The IR is written as text and parsed rather than built with an OpBuilder.
/// That is deliberate: it means every module in this file is IR a user could
/// write and a lit test could contain, so a test that passes here is a claim
/// about the dialect and not about a builder call sequence that happens to
/// produce something the dialect would reject.
class NPUISAInterfaceTest : public ::testing::Test {
protected:
  NPUISAInterfaceTest() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, npu::NPUDialect, NPUISADialect>();
  }

  /// Parses a module, requiring it to parse and verify.
  ///
  /// The module is kept alive in the fixture because every Value and Operation
  /// the tests hold points into it. Returning operations out of a function that
  /// let the module die is the one memory error these tests could plausibly
  /// have, and holding the owning reference here is what rules it out.
  ModuleOp parse(StringRef moduleText) {
    module = parseSourceString<ModuleOp>(moduleText, &context);
    EXPECT_TRUE(module) << "failed to parse or verify:\n" << moduleText.str();
    return module.get();
  }

  /// The first operation of the given kind in the parsed module.
  template <typename OpTy>
  OpTy first(StringRef moduleText) {
    ModuleOp parsed = parse(moduleText);
    if (!parsed)
      return nullptr;
    OpTy found = nullptr;
    parsed.walk([&](OpTy op) {
      if (!found)
        found = op;
    });
    EXPECT_TRUE(found) << "no operation of the expected kind in:\n"
                       << moduleText.str();
    return found;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

//===----------------------------------------------------------------------===//
// The shared assertions.
//===----------------------------------------------------------------------===//

/// Every operand is covered exactly once by the union of `ins` and `outs`.
///
/// This is the P2 gate's "ins and outs partition the operands exactly once",
/// checked through the interface's own accessors. Counting coverage rather than
/// comparing two lists is what makes the assertion catch both halves of a
/// partition failure at once: an operand in neither list scores 0 and an operand
/// in both scores 2, and a plain "the counts add up" check would let a swap of
/// one for the other through.
void expectOperandsPartitionedExactlyOnce(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  ASSERT_TRUE(dps) << op->getName().getStringRef().str()
                   << " does not implement DestinationStyleOpInterface";

  const unsigned numOperands = op->getNumOperands();
  llvm::SmallVector<unsigned> coverage(numOperands, 0);

  for (int64_t i = 0, e = dps.getNumDpsInits(); i < e; ++i)
    ++coverage[dps.getDpsInitOperand(i)->getOperandNumber()];
  for (OpOperand *input : dps.getDpsInputOperands())
    ++coverage[input->getOperandNumber()];

  for (unsigned i = 0; i < numOperands; ++i)
    EXPECT_EQ(coverage[i], 1u)
        << op->getName().getStringRef().str() << " operand " << i
        << " is covered " << coverage[i]
        << " times by ins and outs, and a partition covers each exactly once";

  // Exactly one destination on every compute instruction, which is the shape
  // Section 8 specifies: `ins` operands are read, one `outs` destination memref
  // is written, no results. A second init would be a second written buffer that
  // the overlap rule and the encoder would each have to learn about.
  EXPECT_EQ(dps.getNumDpsInits(), 1)
      << op->getName().getStringRef().str() << " must have exactly one outs";

  // And the destination is the last operand, which is what
  // `getDpsInitsMutable` claims. An operation whose destination drifted to the
  // middle would still partition correctly and would still be wrong.
  EXPECT_EQ(dps.getDpsInitOperand(0)->getOperandNumber(), numOperands - 1)
      << op->getName().getStringRef().str()
      << " must carry its destination as its last operand";

  // No results. This is the difference from the npu tensor level and it is the
  // reason the memory effects matter: an operation with no results and no
  // declared effects is one the canonicalizer may delete.
  EXPECT_EQ(op->getNumResults(), 0u)
      << op->getName().getStringRef().str()
      << " is destination passing and must have no results";
}

/// The operation declares memory effects, and the list is not empty.
///
/// This is the P2 gate's "no compute operation reports itself free of effects",
/// and it is checked two ways because the two failures are different. An
/// operation that does not implement the interface at all is entitled to be
/// treated as free of effects by every pass in MLIR; an operation that
/// implements it and returns nothing says outright that it touches no memory.
/// Both would let the canonicalizer delete a write.
void expectDeclaresEffects(Operation *op) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  ASSERT_TRUE(effects) << op->getName().getStringRef().str()
                       << " does not implement MemoryEffectOpInterface, so "
                          "every pass may assume it is free of effects";

  llvm::SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  EXPECT_FALSE(instances.empty())
      << op->getName().getStringRef().str()
      << " reports an empty effect list, which says it touches no memory";

  // The strongest form of the same claim, and the one MLIR's own passes ask.
  // `isMemoryEffectFree` is what `isOpTriviallyDead` consults, so an operation
  // for which it returns true is an operation dead code elimination will delete,
  // results or no results.
  EXPECT_FALSE(isMemoryEffectFree(op))
      << op->getName().getStringRef().str()
      << " reports itself free of memory effects and would be deleted as dead";
}

/// The destination operand is written and every input operand is read.
///
/// The overlap rule of Section 8 is decided on which value each effect names,
/// so a blanket "this operation touches memory somewhere" would not be enough:
/// the scan has to be able to ask whether *this buffer* is the one written.
void expectDestinationWrittenAndInputsRead(Operation *op) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  ASSERT_TRUE(effects);
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  ASSERT_TRUE(dps);

  llvm::SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);

  Value destination = dps.getDpsInitOperand(0)->get();

  bool destinationWritten = false;
  for (const MemoryEffects::EffectInstance &instance : instances)
    if (isa<MemoryEffects::Write>(instance.getEffect()) &&
        instance.getValue() == destination)
      destinationWritten = true;

  EXPECT_TRUE(destinationWritten)
      << op->getName().getStringRef().str()
      << " does not declare a write on its destination, so the overlap rule "
         "cannot tell which buffer it fills";

  for (OpOperand *input : dps.getDpsInputOperands()) {
    // An in place operation names the same buffer as an input and as the
    // destination, and then the write above is the effect on it. Only the
    // operands that are not also the destination need a read of their own.
    if (input->get() == destination)
      continue;
    bool read = false;
    for (const MemoryEffects::EffectInstance &instance : instances)
      if (isa<MemoryEffects::Read>(instance.getEffect()) &&
          instance.getValue() == input->get())
        read = true;
    EXPECT_TRUE(read) << op->getName().getStringRef().str() << " operand "
                      << input->getOperandNumber()
                      << " is an ins operand with no declared read";
  }
}

//===----------------------------------------------------------------------===//
// The compute operations, one module each.
//
// The modules are held as text next to the operation they exercise rather than
// generated from a table, because a table of shapes is unreadable and a wrong
// shape in it produces a parse failure whose message points at the table.
//===----------------------------------------------------------------------===//

constexpr StringRef kMatMul = R"mlir(
  func.func @f(%a: memref<4x16xf32, #npu.scratchpad>,
               %b: memref<16x10xf32, #npu.scratchpad>,
               %d: memref<4x10xf32, #npu.scratchpad>) {
    npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.scratchpad>,
                               memref<16x10xf32, #npu.scratchpad>)
                  outs(%d : memref<4x10xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kMatMulWithBias = R"mlir(
  func.func @f(%a: memref<4x16xf32, #npu.scratchpad>,
               %b: memref<16x10xf32, #npu.scratchpad>,
               %c: memref<10xf32, #npu.scratchpad>,
               %d: memref<4x10xf32, #npu.scratchpad>) {
    npuisa.matmul ins(%a, %b, %c : memref<4x16xf32, #npu.scratchpad>,
                                   memref<16x10xf32, #npu.scratchpad>,
                                   memref<10xf32, #npu.scratchpad>)
                  outs(%d : memref<4x10xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kConv2D = R"mlir(
  func.func @f(%x: memref<1x3x8x8xf32, #npu.scratchpad>,
               %w: memref<8x3x3x3xf32, #npu.scratchpad>,
               %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
    npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                               memref<8x3x3x3xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
    return
  }
)mlir";

constexpr StringRef kConv2DWithBias = R"mlir(
  func.func @f(%x: memref<1x3x8x8xf32, #npu.scratchpad>,
               %w: memref<8x3x3x3xf32, #npu.scratchpad>,
               %b: memref<8xf32, #npu.scratchpad>,
               %d: memref<1x8x8x8xf32, #npu.scratchpad>) {
    npuisa.conv2d ins(%x, %w, %b : memref<1x3x8x8xf32, #npu.scratchpad>,
                                   memref<8x3x3x3xf32, #npu.scratchpad>,
                                   memref<8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
    return
  }
)mlir";

constexpr StringRef kAdd = R"mlir(
  func.func @f(%a: memref<1x8x4x4xf32, #npu.scratchpad>,
               %b: memref<1x8x4x4xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.add ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                            memref<1x8x4x4xf32, #npu.scratchpad>)
               outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kMul = R"mlir(
  func.func @f(%a: memref<1x8x4x4xf32, #npu.scratchpad>,
               %b: memref<1x8x4x4xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.mul ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                            memref<1x8x4x4xf32, #npu.scratchpad>)
               outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kRelu = R"mlir(
  func.func @f(%x: memref<1x8x4x4xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kReluInPlace = R"mlir(
  func.func @f(%x: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
                outs(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kPoolMax = R"mlir(
  func.func @f(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.pool_max ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                    outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                    {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                     pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
    return
  }
)mlir";

constexpr StringRef kPoolAvg = R"mlir(
  func.func @f(%x: memref<1x8x8x8xf32, #npu.scratchpad>,
               %d: memref<1x8x4x4xf32, #npu.scratchpad>) {
    npuisa.pool_avg ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                    outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                    {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                     pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
    return
  }
)mlir";

constexpr StringRef kReshape = R"mlir(
  func.func @f(%x: memref<4x8x2x2xf32, #npu.scratchpad>,
               %d: memref<4x32xf32, #npu.scratchpad>) {
    npuisa.reshape ins(%x : memref<4x8x2x2xf32, #npu.scratchpad>)
                   outs(%d : memref<4x32xf32, #npu.scratchpad>)
    return
  }
)mlir";

constexpr StringRef kTranspose = R"mlir(
  func.func @f(%x: memref<1x3x8x8xf32, #npu.scratchpad>,
               %d: memref<1x8x8x3xf32, #npu.scratchpad>) {
    npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                     outs(%d : memref<1x8x8x3xf32, #npu.scratchpad>)
                     {permutation = array<i64: 0, 2, 3, 1>}
    return
  }
)mlir";

constexpr StringRef kConcat = R"mlir(
  func.func @f(%a: memref<1x4x8x8xf32, #npu.scratchpad>,
               %b: memref<1x6x8x8xf32, #npu.scratchpad>,
               %d: memref<1x10x8x8xf32, #npu.scratchpad>) {
    npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                               memref<1x6x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                  {axis = 1 : i64}
    return
  }
)mlir";

/// Every compute instruction in the dialect, with the module that builds one.
///
/// The list is written out rather than discovered by walking the dialect's
/// registered operations, and that is the point: a new compute instruction added
/// without a row here is a new instruction with no interface coverage, and a
/// test that discovered its subjects automatically would report success for a
/// dialect it had never looked at. `NoNewComputeOpIsUncovered` below closes the
/// loop by counting.
struct ComputeCase {
  StringRef name;
  StringRef ir;
};

const ComputeCase kComputeCases[] = {
    {"matmul", kMatMul},
    {"matmul_with_bias", kMatMulWithBias},
    {"conv2d", kConv2D},
    {"conv2d_with_bias", kConv2DWithBias},
    {"add", kAdd},
    {"mul", kMul},
    {"relu", kRelu},
    {"relu_in_place", kReluInPlace},
    {"pool_max", kPoolMax},
    {"pool_avg", kPoolAvg},
    {"reshape", kReshape},
    {"transpose", kTranspose},
    {"concat", kConcat},
};

/// The compute instruction inside one of the modules above.
Operation *computeOpIn(ModuleOp module) {
  Operation *found = nullptr;
  module.walk([&](Operation *op) {
    if (isa<DestinationStyleOpInterface>(op) && !found)
      found = op;
  });
  return found;
}

//===----------------------------------------------------------------------===//
// 1. `ins` and `outs` partition the operands exactly once.
//===----------------------------------------------------------------------===//

TEST_F(NPUISAInterfaceTest, InsAndOutsPartitionOperandsExactlyOnce) {
  for (const ComputeCase &testCase : kComputeCases) {
    SCOPED_TRACE(testCase.name.str());
    ModuleOp parsed = parse(testCase.ir);
    ASSERT_TRUE(parsed);
    Operation *op = computeOpIn(parsed);
    ASSERT_TRUE(op) << "no destination style operation in the module";
    expectOperandsPartitionedExactlyOnce(op);
  }
}

// The optional bias is the case a partition rule gets wrong. When it is absent
// the destination is operand 2 and when it is present it is operand 3, so a
// destination found by a fixed index rather than by "the last one" would put the
// bias in `outs` and the destination in `ins` on exactly one of these two.
TEST_F(NPUISAInterfaceTest, TheOptionalBiasDoesNotMoveTheDestination) {
  ModuleOp withoutBias = parse(kMatMul);
  ASSERT_TRUE(withoutBias);
  auto plain = cast<MatMulOp>(computeOpIn(withoutBias));
  auto plainDps = cast<DestinationStyleOpInterface>(plain.getOperation());
  EXPECT_EQ(plain.getNumOperands(), 3u);
  EXPECT_EQ(plainDps.getDpsInitOperand(0)->get(), plain.getDestination());
  EXPECT_EQ(plainDps.getDpsInputOperands().size(), 2u);

  ModuleOp withBias = parse(kMatMulWithBias);
  ASSERT_TRUE(withBias);
  auto biased = cast<MatMulOp>(computeOpIn(withBias));
  auto biasedDps = cast<DestinationStyleOpInterface>(biased.getOperation());
  EXPECT_EQ(biased.getNumOperands(), 4u);
  EXPECT_EQ(biasedDps.getDpsInitOperand(0)->get(), biased.getDestination());
  // The bias joined `ins`, not `outs`. It is read, and reading it into the
  // destination would be a second written buffer.
  EXPECT_EQ(biasedDps.getDpsInputOperands().size(), 3u);
  EXPECT_EQ(biasedDps.getNumDpsInits(), 1);
}

// The variadic case, for the same reason: `npuisa.concat` takes any number of
// inputs, so the operand index of its destination is not a constant of the
// operation at all, and a partition that assumed one would be wrong for every
// arity but the one it was written against.
TEST_F(NPUISAInterfaceTest, TheVariadicConcatStillPartitions) {
  constexpr StringRef kThreeInputs = R"mlir(
    func.func @f(%a: memref<1x2x8x8xf32, #npu.scratchpad>,
                 %b: memref<1x3x8x8xf32, #npu.scratchpad>,
                 %c: memref<1x5x8x8xf32, #npu.scratchpad>,
                 %d: memref<1x10x8x8xf32, #npu.scratchpad>) {
      npuisa.concat ins(%a, %b, %c : memref<1x2x8x8xf32, #npu.scratchpad>,
                                     memref<1x3x8x8xf32, #npu.scratchpad>,
                                     memref<1x5x8x8xf32, #npu.scratchpad>)
                    outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                    {axis = 1 : i64}
      return
    }
  )mlir";

  ModuleOp parsed = parse(kThreeInputs);
  ASSERT_TRUE(parsed);
  Operation *op = computeOpIn(parsed);
  ASSERT_TRUE(op);
  expectOperandsPartitionedExactlyOnce(op);

  auto dps = cast<DestinationStyleOpInterface>(op);
  EXPECT_EQ(dps.getDpsInputOperands().size(), 3u);
  EXPECT_EQ(dps.getNumDpsInits(), 1);
}

//===----------------------------------------------------------------------===//
// 2. No compute operation reports itself free of effects.
//===----------------------------------------------------------------------===//

TEST_F(NPUISAInterfaceTest, NoComputeOperationIsFreeOfEffects) {
  for (const ComputeCase &testCase : kComputeCases) {
    SCOPED_TRACE(testCase.name.str());
    ModuleOp parsed = parse(testCase.ir);
    ASSERT_TRUE(parsed);
    Operation *op = computeOpIn(parsed);
    ASSERT_TRUE(op);
    expectDeclaresEffects(op);
  }
}

// The DMA operations are not destination passing, so they are not in the table
// above, and they are the operations whose whole purpose is to move memory. An
// effect free `dma_load` would be deleted by the canonicalizer and the program
// would read an uninitialised scratchpad.
TEST_F(NPUISAInterfaceTest, TheTransferOperationsAreNotFreeOfEffects) {
  constexpr StringRef kTransfers = R"mlir(
    func.func @f(%dram: memref<4x4xf32, #npu.dram>,
                 %pad: memref<4x4xf32, #npu.scratchpad>) {
      npuisa.dma_load %dram, %pad
        : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
      npuisa.dma_store %pad, %dram
        : memref<4x4xf32, #npu.scratchpad> to memref<4x4xf32, #npu.dram>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kTransfers);
  ASSERT_TRUE(parsed);

  bool sawLoad = false;
  parsed.walk([&](DmaLoadOp load) {
    sawLoad = true;
    EXPECT_FALSE(isMemoryEffectFree(load))
        << "npuisa.dma_load reports itself free of effects";
  });
  EXPECT_TRUE(sawLoad);

  bool sawStore = false;
  parsed.walk([&](DmaStoreOp store) {
    sawStore = true;
    EXPECT_FALSE(isMemoryEffectFree(store))
        << "npuisa.dma_store reports itself free of effects";
  });
  EXPECT_TRUE(sawStore);
}

// The asynchronous forms too. These have a result, the token, so the "no result
// means deletable" argument does not apply to them; what does apply is that a
// pass reordering an effect free operation across a compute instruction would
// move the transfer past the work that reads its destination.
TEST_F(NPUISAInterfaceTest, TheAsynchronousTransfersAreNotFreeOfEffects) {
  constexpr StringRef kAsync = R"mlir(
    func.func @f(%dram: memref<4x4xf32, #npu.dram>,
                 %pad: memref<4x4xf32, #npu.scratchpad>,
                 %other: memref<2x2xf32, #npu.scratchpad>) {
      %t = npuisa.dma_load_async %dram, %pad
         : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
      npuisa.relu ins(%other : memref<2x2xf32, #npu.scratchpad>)
                  outs(%other : memref<2x2xf32, #npu.scratchpad>)
      npuisa.await %t
      return
    }
  )mlir";

  ModuleOp parsed = parse(kAsync);
  ASSERT_TRUE(parsed);

  bool sawAsync = false;
  parsed.walk([&](DmaLoadAsyncOp async) {
    sawAsync = true;
    EXPECT_FALSE(isMemoryEffectFree(async))
        << "npuisa.dma_load_async reports itself free of effects";

    auto effects = dyn_cast<MemoryEffectOpInterface>(async.getOperation());
    ASSERT_TRUE(effects);
    llvm::SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);

    bool destWritten = false;
    for (const MemoryEffects::EffectInstance &instance : instances)
      if (isa<MemoryEffects::Write>(instance.getEffect()) &&
          instance.getValue() == async.getDest())
        destWritten = true;
    EXPECT_TRUE(destWritten)
        << "npuisa.dma_load_async must declare the write on its destination, "
           "because that write is what the overlap rule scans for";
  });
  EXPECT_TRUE(sawAsync);
}

//===----------------------------------------------------------------------===//
// 3. The destination is written and the inputs are read, per operand.
//===----------------------------------------------------------------------===//

TEST_F(NPUISAInterfaceTest, EffectsNameTheDestinationAndTheInputs) {
  for (const ComputeCase &testCase : kComputeCases) {
    SCOPED_TRACE(testCase.name.str());
    ModuleOp parsed = parse(testCase.ir);
    ASSERT_TRUE(parsed);
    Operation *op = computeOpIn(parsed);
    ASSERT_TRUE(op);
    expectDestinationWrittenAndInputsRead(op);
  }
}

// `npuisa.await` declares no memory effect, and Section 8 says so deliberately:
// what it does is order an effect the asynchronous operation already declared.
// The test is here rather than absent so that giving it one later is a decision
// somebody makes against a failing test, not a change nobody notices.
TEST_F(NPUISAInterfaceTest, TheAwaitDeclaresNoEffectOfItsOwn) {
  constexpr StringRef kAsync = R"mlir(
    func.func @f(%dram: memref<4x4xf32, #npu.dram>,
                 %pad: memref<4x4xf32, #npu.scratchpad>) {
      %t = npuisa.dma_load_async %dram, %pad
         : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
      npuisa.await %t
      return
    }
  )mlir";

  ModuleOp parsed = parse(kAsync);
  ASSERT_TRUE(parsed);

  bool sawAwait = false;
  parsed.walk([&](AwaitOp await) {
    sawAwait = true;
    EXPECT_FALSE(isa<MemoryEffectOpInterface>(await.getOperation()))
        << "npuisa.await now declares memory effects. That may be right, but "
           "the overlap scan skips awaits by name on the strength of it not "
           "declaring any, so the two have to change together";
  });
  EXPECT_TRUE(sawAwait);
}

//===----------------------------------------------------------------------===//
// 4. The overlap rule: effects plus view offsets and extents, never SSA
//    identity.
//===----------------------------------------------------------------------===//

/// The named value in a function, found by the operation that defines it.
/// Values are unnamed after parsing, so the tests below pick them out by walking
/// to the nth view or allocation rather than by name.
template <typename OpTy>
llvm::SmallVector<Value> resultsOf(ModuleOp module) {
  llvm::SmallVector<Value> results;
  module.walk([&](OpTy op) { results.push_back(op->getResult(0)); });
  return results;
}

// The case the P2 gate names outright: two `memref.view` results over one flat
// buffer whose byte ranges partially intersect. They are different SSA values,
// so an identity check reports no race; they share 32 bytes, so the arithmetic
// reports one.
TEST_F(NPUISAInterfaceTest, PartiallyOverlappingViewsOverlap) {
  // 4 by 4 f32 is 64 bytes. The first view is bytes [0, 64) and the second is
  // [32, 96), so they share the 32 bytes in [32, 64).
  constexpr StringRef kViews = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %a = memref.view %flat[%c0][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %b = memref.view %flat[%c32][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kViews);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 2u);

  // The claim an SSA identity check would get wrong: the values differ.
  EXPECT_NE(views[0], views[1])
      << "the two views must be different SSA values for this test to be "
         "about anything";

  EXPECT_EQ(overlaps(views[0], views[1]), OverlapResult::Overlaps);
  EXPECT_EQ(overlaps(views[1], views[0]), OverlapResult::Overlaps)
      << "the overlap relation must be symmetric";

  // And the arithmetic underneath, so a failure says which half is wrong.
  std::optional<BufferRange> first = computeBufferRange(views[0]);
  std::optional<BufferRange> second = computeBufferRange(views[1]);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first->offset, 0);
  EXPECT_EQ(first->size, 64);
  EXPECT_EQ(second->offset, 32);
  EXPECT_EQ(second->size, 64);
  EXPECT_EQ(first->base, second->base)
      << "both views are over one allocation, so their ranges are comparable";
}

// The other half of the same claim. Adjacent views do not overlap, and an
// analysis that answered Overlaps for everything sharing a base would pass the
// test above and fail here.
TEST_F(NPUISAInterfaceTest, AdjacentViewsAreDisjoint) {
  constexpr StringRef kViews = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %a = memref.view %flat[%c0][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %b = memref.view %flat[%c64][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kViews);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 2u);

  // [0, 64) and [64, 128) touch at the boundary and share no byte. The half open
  // interval convention is what makes that the answer, and an implementation
  // using closed intervals would report an overlap of exactly one byte here.
  EXPECT_EQ(overlaps(views[0], views[1]), OverlapResult::Disjoint);
}

// A value always overlaps itself, and it does so through the arithmetic rather
// than through a special case for identity. The distinction matters because a
// special case is a branch that could be removed without any other test
// noticing.
TEST_F(NPUISAInterfaceTest, AValueOverlapsItself) {
  ModuleOp parsed = parse(kRelu);
  ASSERT_TRUE(parsed);
  auto func = *parsed.getOps<func::FuncOp>().begin();
  Value argument = func.getArgument(0);
  EXPECT_EQ(overlaps(argument, argument), OverlapResult::Overlaps);
}

// Two distinct allocations are two distinct memories, whatever their offsets.
TEST_F(NPUISAInterfaceTest, DistinctAllocationsAreDisjoint) {
  constexpr StringRef kTwoAllocations = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %one = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %two = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %a = memref.view %one[%c0][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %b = memref.view %two[%c0][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      memref.dealloc %one : memref<256xi8, #npu.scratchpad>
      memref.dealloc %two : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kTwoAllocations);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 2u);

  // Both are at byte offset 0 of their own buffer, so an analysis that compared
  // offsets without comparing bases would call these a race.
  EXPECT_EQ(overlaps(views[0], views[1]), OverlapResult::Disjoint);
}

// Two distinct function arguments are two distinct buffers whose addresses this
// analysis cannot see, and treating them as distinct is the right answer: a
// caller that aliased two arguments would be the caller's defect, and the
// alternative, refusing every function that takes two memrefs, would make the
// asynchronous form unusable before allocation ever runs.
TEST_F(NPUISAInterfaceTest, DistinctBlockArgumentsAreDisjoint) {
  ModuleOp parsed = parse(kRelu);
  ASSERT_TRUE(parsed);
  auto func = *parsed.getOps<func::FuncOp>().begin();
  EXPECT_EQ(overlaps(func.getArgument(0), func.getArgument(1)),
            OverlapResult::Disjoint);
}

// A non static offset is Unknown rather than Disjoint, which is the distinction
// Section 8 insists on: "I cannot prove these overlap" and "these are disjoint"
// are different answers and only one of them is safe.
TEST_F(NPUISAInterfaceTest, ANonStaticOffsetIsUnknownAndNotDisjoint) {
  constexpr StringRef kDynamic = R"mlir(
    func.func @f(%offset: index) {
      %c0 = arith.constant 0 : index
      %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %a = memref.view %flat[%c0][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %b = memref.view %flat[%offset][]
         : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kDynamic);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 2u);

  EXPECT_EQ(overlaps(views[0], views[1]), OverlapResult::Unknown);
  EXPECT_NE(overlaps(views[0], views[1]), OverlapResult::Disjoint)
      << "an unprovable offset must never be reported as disjoint";

  // The dynamic one is the unanalysable half, and the static one is still fine.
  EXPECT_FALSE(computeBufferRange(views[1]).has_value());
  EXPECT_TRUE(computeBufferRange(views[0]).has_value());
  EXPECT_FALSE(describeWhyNotAnalysable(views[1]).empty())
      << "an unanalysable value must come with a reason for the diagnostic";
  EXPECT_TRUE(describeWhyNotAnalysable(views[0]).empty())
      << "an analysable value has no reason to give";
}

// Nested views, which is what a spill reload over a sub buffer produces. The
// offsets accumulate down the chain, so a walk that stopped at the first view
// would compute the wrong base offset and place the range in the wrong part of
// the buffer.
TEST_F(NPUISAInterfaceTest, NestedViewOffsetsAccumulate) {
  constexpr StringRef kNested = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %flat = memref.alloc() : memref<512xi8, #npu.scratchpad>
      %mid = memref.view %flat[%c128][]
           : memref<512xi8, #npu.scratchpad> to memref<256xi8, #npu.scratchpad>
      %inner = memref.view %mid[%c64][]
             : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %sibling = memref.view %flat[%c0][]
               : memref<512xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      memref.dealloc %flat : memref<512xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kNested);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 3u);

  // 128 from the outer view plus 64 from the inner one is 192, not 64.
  std::optional<BufferRange> inner = computeBufferRange(views[1]);
  ASSERT_TRUE(inner);
  EXPECT_EQ(inner->offset, 192)
      << "the offsets of a nested view chain must accumulate";
  EXPECT_EQ(inner->size, 64);

  // The sibling at byte 0 is 64 bytes, so [0, 64) against [192, 256).
  EXPECT_EQ(overlaps(views[1], views[2]), OverlapResult::Disjoint);
}

//===----------------------------------------------------------------------===//
// 4a. `memref.reinterpret_cast`, added at P5.
//
// P4's handoff left this analysis a question rather than a bug: the broadcast
// view the lowering emits for ADR 0005 is a `reinterpret_cast`, this walk had
// never been shown one, and somebody had to decide **whether a stride 0 view
// has a byte range at all**. It does, and the answer is the range of the bytes
// it actually addresses.
//
// The four tests below are the decision written as assertions rather than as
// prose in a design document, because prose cannot fail.
//===----------------------------------------------------------------------===//

// A stride 0 broadcast spans the rank 1 buffer it is over, not the tensor it is
// shaped like. `sizes [1, 8, 4, 4]` over `strides [0, 1, 0, 0]` is 32 bytes of
// f32, not 512, because the only indices it can reach are the eight channels.
TEST_F(NPUISAInterfaceTest, AStrideZeroBroadcastSpansTheBufferItIsOver) {
  constexpr StringRef kBroadcast = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %flat = memref.alloc() : memref<1024xi8, #npu.scratchpad>
      %scale = memref.view %flat[%c0][]
             : memref<1024xi8, #npu.scratchpad> to memref<8xf32, #npu.scratchpad>
      %bcast = memref.reinterpret_cast %scale to offset: [0],
                   sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
             : memref<8xf32, #npu.scratchpad>
            to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
      memref.dealloc %flat : memref<1024xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kBroadcast);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> casts = resultsOf<memref::ReinterpretCastOp>(parsed);
  ASSERT_EQ(casts.size(), 1u);

  std::optional<BufferRange> range = computeBufferRange(casts[0]);
  ASSERT_TRUE(range) << "a broadcast view must be analysable, not Unknown: it "
                        "is what every per channel scale in the pipeline is";
  EXPECT_EQ(range->offset, 0);
  EXPECT_EQ(range->size, 32)
      << "the span comes from the strides. Taking it from the extents would "
         "claim 512 bytes for a view that reads 8 floats, and every buffer the "
         "allocator packed within 480 bytes of it would then look like a race";
}

// The consequence that makes the decision matter. Two broadcasts over two
// adjacent 32 byte scale buffers are disjoint. Under a span taken from the
// extents both would claim 512 bytes from their own offset, they would overlap
// each other and everything else in the arena, and a double buffering pass
// would be refused a transfer it is entitled to.
TEST_F(NPUISAInterfaceTest, TwoAdjacentBroadcastsAreDisjoint) {
  constexpr StringRef kTwo = R"mlir(
    func.func @f() {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %flat = memref.alloc() : memref<1024xi8, #npu.scratchpad>
      %first = memref.view %flat[%c0][]
             : memref<1024xi8, #npu.scratchpad> to memref<8xf32, #npu.scratchpad>
      %second = memref.view %flat[%c32][]
              : memref<1024xi8, #npu.scratchpad> to memref<8xf32, #npu.scratchpad>
      %a = memref.reinterpret_cast %first to offset: [0],
               sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
         : memref<8xf32, #npu.scratchpad>
        to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
      %b = memref.reinterpret_cast %second to offset: [0],
               sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
         : memref<8xf32, #npu.scratchpad>
        to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
      memref.dealloc %flat : memref<1024xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kTwo);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> casts = resultsOf<memref::ReinterpretCastOp>(parsed);
  ASSERT_EQ(casts.size(), 2u);

  EXPECT_EQ(overlaps(casts[0], casts[1]), OverlapResult::Disjoint);
  // And a broadcast does overlap the buffer it is a view of, which is the
  // assertion that stops the test above from passing for the wrong reason.
  llvm::SmallVector<Value> views = resultsOf<memref::ViewOp>(parsed);
  ASSERT_EQ(views.size(), 2u);
  EXPECT_EQ(overlaps(casts[0], views[0]), OverlapResult::Overlaps);
  EXPECT_EQ(overlaps(casts[0], views[1]), OverlapResult::Disjoint);
}

// A permutation layout, which is what an NHWC buffer gets, spans exactly the
// contiguous block its extents describe. The stride rule has to agree with the
// extent rule here or every layout assigned buffer would change size.
TEST_F(NPUISAInterfaceTest, APermutationLayoutSpansItsWholeExtent) {
  constexpr StringRef kPermuted = R"mlir(
    func.func @f() {
      %c256 = arith.constant 256 : index
      %flat = memref.alloc() : memref<4096xi8, #npu.scratchpad>
      %v = memref.view %flat[%c256][]
         : memref<4096xi8, #npu.scratchpad>
        to memref<1x3x8x8xf32, #npu.scratchpad>
      %p = memref.reinterpret_cast %v to offset: [0],
               sizes: [1, 3, 8, 8], strides: [192, 1, 24, 3]
         : memref<1x3x8x8xf32, #npu.scratchpad>
        to memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
      memref.dealloc %flat : memref<4096xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kPermuted);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> casts = resultsOf<memref::ReinterpretCastOp>(parsed);
  ASSERT_EQ(casts.size(), 1u);

  std::optional<BufferRange> range = computeBufferRange(casts[0]);
  ASSERT_TRUE(range);
  // 192 elements at 4 bytes, reached as (1-1)*192 + (3-1)*1 + (8-1)*24 +
  // (8-1)*3 + 1, which is 192 exactly.
  EXPECT_EQ(range->size, 768);
  EXPECT_EQ(range->offset, 256)
      << "the view underneath the cast still contributes its byte shift";
}

// A cast at a non zero offset. Neither pass in this project writes one today,
// and it is handled rather than refused because the operation permits it and an
// analysis that silently answered Unknown for a legal form would push the
// refusal into whichever pass wrote one first.
TEST_F(NPUISAInterfaceTest, AReinterpretCastOffsetIsCountedInBytes) {
  constexpr StringRef kOffset = R"mlir(
    func.func @f() {
      %c64 = arith.constant 64 : index
      %flat = memref.alloc() : memref<1024xi8, #npu.scratchpad>
      %v = memref.view %flat[%c64][]
         : memref<1024xi8, #npu.scratchpad> to memref<16xf32, #npu.scratchpad>
      %c = memref.reinterpret_cast %v to offset: [4], sizes: [4], strides: [1]
         : memref<16xf32, #npu.scratchpad>
        to memref<4xf32, strided<[1], offset: 4>, #npu.scratchpad>
      memref.dealloc %flat : memref<1024xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kOffset);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> casts = resultsOf<memref::ReinterpretCastOp>(parsed);
  ASSERT_EQ(casts.size(), 1u);

  std::optional<BufferRange> range = computeBufferRange(casts[0]);
  ASSERT_TRUE(range);
  // 64 bytes from the view plus 4 f32 from the cast is 80, and the cast is 4
  // f32 wide. The offset is counted once: the type's own layout offset is not
  // read, because the walk gets it from the operation.
  EXPECT_EQ(range->offset, 80);
  EXPECT_EQ(range->size, 16);
}

// D-0018, found while adding the case above. A `memref.subview` result was
// measured by the product of its extents, which is the number of elements it
// holds and not the number of bytes it reaches across. Two subviews of one
// buffer that genuinely share elements were then reported Disjoint, which is the
// unsafe direction: the async rule would have allowed a real race.
//
// The two here are the top left 2 by 2 of a 4 by 4 and the 2 by 2 starting one
// row down. They share the whole of row 1, elements 4 and 5.
TEST_F(NPUISAInterfaceTest, OverlappingSubviewsAreNotReportedDisjoint) {
  constexpr StringRef kSubviews = R"mlir(
    func.func @f() {
      %buffer = memref.alloc() : memref<4x4xf32, #npu.scratchpad>
      %top = memref.subview %buffer[0, 0] [2, 2] [1, 1]
           : memref<4x4xf32, #npu.scratchpad>
          to memref<2x2xf32, strided<[4, 1]>, #npu.scratchpad>
      %next = memref.subview %buffer[1, 0] [2, 2] [1, 1]
            : memref<4x4xf32, #npu.scratchpad>
           to memref<2x2xf32, strided<[4, 1], offset: 4>, #npu.scratchpad>
      memref.dealloc %buffer : memref<4x4xf32, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kSubviews);
  ASSERT_TRUE(parsed);
  llvm::SmallVector<Value> subviews = resultsOf<memref::SubViewOp>(parsed);
  ASSERT_EQ(subviews.size(), 2u);

  // Elements 0, 1, 4 and 5 against 4, 5, 8 and 9. The hull of the first is
  // elements [0, 6), which is bytes [0, 24), and of the second [4, 10), which
  // is bytes [16, 40).
  std::optional<BufferRange> top = computeBufferRange(subviews[0]);
  std::optional<BufferRange> next = computeBufferRange(subviews[1]);
  ASSERT_TRUE(top);
  ASSERT_TRUE(next);
  EXPECT_EQ(top->size, 24)
      << "a non contiguous subview reaches across more bytes than it holds";
  EXPECT_EQ(next->offset, 16);
  EXPECT_EQ(next->size, 24);
  EXPECT_EQ(overlaps(subviews[0], subviews[1]), OverlapResult::Overlaps);
}

// The rule end to end, through the verifier, on IR whose only race is between
// two partially overlapping views. This is the same claim as
// PartiallyOverlappingViewsOverlap made one level up, and it is here because the
// arithmetic being right and the verifier calling it are two separate things.
TEST_F(NPUISAInterfaceTest, TheVerifierRejectsAnOverlappingInterveningWrite) {
  constexpr StringRef kRacing = R"mlir(
    func.func @f(%src: memref<4x4xf32, #npu.dram>) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %dst = memref.view %flat[%c0][]
           : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %over = memref.view %flat[%c32][]
            : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %t = npuisa.dma_load_async %src, %dst
         : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
      npuisa.relu ins(%over : memref<4x4xf32, #npu.scratchpad>)
                  outs(%over : memref<4x4xf32, #npu.scratchpad>)
      npuisa.await %t
      memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  // Parsing runs the verifier, so a module that verifies is a module the rule
  // let through. The diagnostic is suppressed because the failure being tested
  // is the parse returning nothing, not the wording.
  context.getDiagEngine().registerHandler([](Diagnostic &) {});
  OwningOpRef<ModuleOp> racing = parseSourceString<ModuleOp>(kRacing, &context);
  EXPECT_FALSE(racing)
      << "the verifier accepted an intervening write to a partially "
         "overlapping view, which is the race the token exists to prevent";
}

// And the disjoint sibling of it verifies, so the rule is not simply rejecting
// everything with a view in it.
TEST_F(NPUISAInterfaceTest, TheVerifierAcceptsADisjointInterveningWrite) {
  constexpr StringRef kDisjoint = R"mlir(
    func.func @f(%src: memref<4x4xf32, #npu.dram>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %flat = memref.alloc() : memref<256xi8, #npu.scratchpad>
      %dst = memref.view %flat[%c0][]
           : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %other = memref.view %flat[%c64][]
             : memref<256xi8, #npu.scratchpad> to memref<4x4xf32, #npu.scratchpad>
      %t = npuisa.dma_load_async %src, %dst
         : memref<4x4xf32, #npu.dram> to memref<4x4xf32, #npu.scratchpad>
      npuisa.relu ins(%other : memref<4x4xf32, #npu.scratchpad>)
                  outs(%other : memref<4x4xf32, #npu.scratchpad>)
      npuisa.await %t
      memref.dealloc %flat : memref<256xi8, #npu.scratchpad>
      return
    }
  )mlir";

  ModuleOp parsed = parse(kDisjoint);
  EXPECT_TRUE(parsed)
      << "a write to a provably disjoint view between a transfer and its await "
         "is not a race and must verify";
}

//===----------------------------------------------------------------------===//
// The coverage guard.
//===----------------------------------------------------------------------===//

// The table above is written by hand, so a compute instruction added to the
// dialect without a row in it would be a compute instruction with no interface
// coverage and no test would say so. This one does: it counts the operations in
// the dialect that are destination passing and compares that against the number
// of distinct operation names the table exercises.
//
// It fails loudly on a new operation, which is the intent. The fix is to add the
// operation to kComputeCases and bump the number here, in the same commit that
// adds the operation.
TEST_F(NPUISAInterfaceTest, EveryComputeOperationHasARow) {
  llvm::SmallVector<StringRef> covered;
  for (const ComputeCase &testCase : kComputeCases) {
    ModuleOp parsed = parse(testCase.ir);
    ASSERT_TRUE(parsed);
    Operation *op = computeOpIn(parsed);
    ASSERT_TRUE(op);
    StringRef name = op->getName().getStringRef();
    if (!llvm::is_contained(covered, name))
      covered.push_back(name);
  }

  // matmul, conv2d, add, mul, relu, pool_max, pool_avg, reshape, transpose,
  // concat. Ten of the sixteen opcodes of Section 5.4: NOP and HALT are
  // properties of the encoding rather than of the instruction stream, QUANT and
  // DEQUANT arrive with their integer kernels at P14, and DMA_LOAD and DMA_STORE
  // are transfers rather than compute and are covered by their own tests above.
  EXPECT_EQ(covered.size(), 10u)
      << "the number of distinct compute operations covered by kComputeCases "
         "changed. If an operation was added to the dialect, add a module for "
         "it to the table and update this count in the same commit";
}

} // namespace
