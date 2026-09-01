//===- FuseOps.cpp - forming npu.fused_op regions ---------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 12's `-npu-fuse-ops`, at `-O2`. A `npu.relu` whose input comes from
// an `npu.conv2d` or an `npu.matmul` with exactly one use is moved, together
// with that producer, into one `npu.fused_op` region.
//
// **Why a region.** Section 7.2 settles it and the reason survives the summary:
// separate fused operation names produce a combinatorial explosion as the
// fusible set grows, and an enum cannot express what fusion actually does,
// which is to keep an intermediate value in the scratchpad instead of writing
// it to DRAM. A region expresses exactly that and grows without a migration.
//
// **Every value the region reads is an operand, destinations included.**
// `npu.fused_op` is `IsolatedFromAbove`, so there is no other route in, and its
// verifier admits only `npu` operations inside, so a `tensor.empty` cannot be
// cloned into the body even if it were tempting. The operand list is therefore
// the producer's operands followed by the consumer's destination, and the block
// argument list is the same types in the same order.
//
// **It is numerically inert, and that is a property of the memory model rather
// than a weakness.** `-npu-lower-to-npuisa` flattens the region into its
// parent, so the instruction stream is exactly the one the unfused chain
// produced, and an unfused chain already keeps its intermediate in the
// scratchpad because the only DMA producers are the boundary, the spiller and
// the double buffering pass. What the region adds is that the fusion is
// *stated* in the IR, which is what P13's tiling and double buffering read, and
// it is what gives `npu.fused_op` and `npu.yield` the model layer of law 2 that
// `docs/EXEMPTIONS.md` had been carrying an exemption for since P8.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::npu {
#define GEN_PASS_DEF_NPUFUSEOPS
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

// `impl` is qualified in full: `mlir` and `mlir::npu` both have one and both
// are in scope through the using directives above.
struct FuseOpsPass : public mlir::npu::impl::NPUFuseOpsBase<FuseOpsPass> {
  using mlir::npu::impl::NPUFuseOpsBase<FuseOpsPass>::NPUFuseOpsBase;

  void runOnOperation() override;
};

bool fuse(ReluOp relu) {
  // Section 12's "already fused" guard, seen from the consumer's side. A relu
  // inside a region is one this pass put there on an earlier operation of the
  // same run, and fusing it again would nest a region inside itself until the
  // pass ran out of things to wrap.
  if (relu->getParentOfType<FusedOp>())
    return false;

  Operation *producer = relu.getInput().getDefiningOp();
  if (!producer || !isa<Conv2DOp, MatMulOp>(producer))
    return false;
  if (producer->getBlock() != relu->getBlock())
    return false;

  // Section 12's "exactly one use" guard. A second reader of the intermediate
  // would have to read a value that now lives inside a region it is not in, and
  // the only ways out of that are to duplicate the producer or to yield two
  // results, which are a compute cost and a dialect change respectively.
  if (!producer->getResult(0).hasOneUse())
    return false;

  OpBuilder builder(relu);
  Location loc = relu.getLoc();

  SmallVector<Value> operands(producer->getOperands());
  operands.push_back(relu.getDestination());

  SmallVector<Type> argumentTypes;
  SmallVector<Location> argumentLocations;
  argumentTypes.reserve(operands.size());
  argumentLocations.reserve(operands.size());
  for (Value operand : operands) {
    argumentTypes.push_back(operand.getType());
    argumentLocations.push_back(operand.getLoc());
  }

  auto fused =
      FusedOp::create(builder, loc, relu.getResult().getType(), operands);

  Block *body = builder.createBlock(&fused.getBody(), fused.getBody().end(),
                                    argumentTypes, argumentLocations);

  IRMapping mapping;
  for (auto [operand, argument] :
       llvm::zip_equal(operands, body->getArguments()))
    mapping.map(operand, argument);

  builder.setInsertionPointToStart(body);
  builder.clone(*producer, mapping);
  Operation *clonedRelu = builder.clone(*relu.getOperation(), mapping);
  YieldOp::create(builder, loc, clonedRelu->getResult(0));

  relu.getResult().replaceAllUsesWith(fused.getResult());
  relu.erase();
  // After the consumer, because until it is gone the producer still has a use
  // and erasing an operation with users is an assertion failure rather than a
  // diagnostic.
  producer->erase();
  return true;
}

void FuseOpsPass::runOnOperation() {
  func::FuncOp function = getOperation();

  SmallVector<ReluOp> candidates;
  function.walk([&](ReluOp op) { candidates.push_back(op); });

  int64_t formed = 0;
  for (ReluOp op : candidates)
    if (fuse(op))
      ++formed;

  fusedRegions += formed;
}

} // namespace
