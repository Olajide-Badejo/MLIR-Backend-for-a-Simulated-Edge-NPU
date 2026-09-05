//===- DoubleBuffer.cpp - overlapping a transfer with a computation -*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 5.1's `-npu-double-buffer`, running **before allocation**.
//
// The position in the pipeline is the design and Section 5.1 states it in
// parentheses because it is the part somebody would otherwise get wrong:
// double buffering doubles the working set of the prefetched operand, and the
// allocator has to see the doubled set or it places a program that cannot fit
// and spills the wrong buffers. A pass that ran after allocation would produce
// a program whose live ranges the allocator had never seen.
//
// **What it does is one rewrite.** A `npuisa.dma_load` is moved above the
// computation that precedes it and becomes a `npuisa.dma_load_async`, with an
// `npuisa.await` left where the load was. Nothing else changes: same
// instructions, same arithmetic, one token added, and the transfer now runs
// underneath the computation rather than after it.
//
// **What makes it safe is not an identity check**, and that is the reason this
// file depends on the overlap analysis rather than on comparing SSA values.
// Section 8's rule 4 says no operation between the asynchronous operation and
// its await may access memory overlapping the destination, and after the
// allocator has materialised offsets as views over one flat buffer, two
// different values are routinely two halves of the same memory. This pass runs
// before that, where distinct allocations really are distinct, and it still
// asks `npuisa::overlaps`, for two reasons: the answer is the same one the
// verifier will check, and a pass that was correct only because of where it sat
// in the pipeline would be one pipeline edit away from being wrong.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::npuisa {
#define GEN_PASS_DEF_NPUDOUBLEBUFFER
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"
} // namespace mlir::npuisa

using namespace mlir;
using namespace mlir::npuisa;

namespace {

/// Whether this operation is worth overlapping a transfer with.
///
/// The compute instructions of Section 5.4 and nothing else. A transfer hoisted
/// above another transfer has gained nothing, because both are charged to the
/// same DMA port and the two port model of Section 5.5 serializes them anyway;
/// a transfer hoisted above a view or a constant has gained nothing because
/// neither costs a cycle. **The point of the rewrite is to put a transfer under
/// something on the other port**, so the thing it is hoisted over has to be on
/// the other port.
bool isComputation(Operation *op) {
  return isa<npuisa::Conv2DOp, npuisa::MatMulOp, npuisa::AddOp, npuisa::MulOp,
             npuisa::ReluOp, npuisa::PoolMaxOp, npuisa::PoolAvgOp,
             npuisa::ReshapeOp, npuisa::TransposeOp, npuisa::ConcatOp>(op);
}

/// Whether `op` might touch bytes that `buffer` denotes.
///
/// **Unknown counts as yes**, which is Section 8's rule stated as code: "I
/// cannot prove these overlap" and "these are disjoint" are different answers
/// and only one of them makes a program safe. Every memref operand is asked,
/// because an operation's effect on memory reaches it through its operands and
/// this dialect has no operation that names a buffer any other way.
bool mightTouch(Operation *op, Value buffer) {
  for (Value operand : op->getOperands()) {
    if (!isa<MemRefType>(operand.getType()))
      continue;
    if (npuisa::overlaps(operand, buffer) != npuisa::OverlapResult::Disjoint)
      return true;
  }
  return false;
}

struct DoubleBufferPass
    : public npuisa::impl::NPUDoubleBufferBase<DoubleBufferPass> {
  using npuisa::impl::NPUDoubleBufferBase<
      DoubleBufferPass>::NPUDoubleBufferBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();

    // The loads are collected before anything moves, because hoisting one
    // changes the order the walk would see.
    SmallVector<npuisa::DmaLoadOp> loads;
    function.walk([&](npuisa::DmaLoadOp load) { loads.push_back(load); });

    for (npuisa::DmaLoadOp load : loads)
      if (!hoist(load))
        ++notHoisted;
  }

private:
  /// The pure view and allocation operations a load's operands are built from.
  ///
  /// **A transfer cannot be hoisted without them and that is the whole reason
  /// this exists.** The lowering emits a tile's destination allocation and its
  /// source subview immediately before the load that uses them, so a backward
  /// walk from the load stops on its own operands after one step and never
  /// reaches the computation there was something to overlap with. Moving them
  /// up with the transfer is not a liberty: an allocation and a view have no
  /// side effects, so their position carries no meaning except the live range
  /// it implies, and **extending that live range is exactly what double
  /// buffering is.** The allocator sees the longer range because this pass runs
  /// before it, which is the ordering Section 5.1 fixes.
  ///
  /// Only operands defined in the same block are collected, and the use count
  /// is deliberately **not** consulted. A tile's destination buffer is read by
  /// the computation that consumes it as well as written by the transfer, so it
  /// always has more than one use, and a rule that required exactly one would
  /// decline every transfer this pass exists for. Moving a pure operation
  /// **earlier** cannot break a later use, because every use it had is still
  /// after it; what has to be checked is the other direction, that its own
  /// operands still reach it, and `hoistIsDominanceSafe` below is that check.
  SmallVector<Operation *> prologueOf(npuisa::DmaLoadOp load) {
    SmallVector<Operation *> prologue;
    SmallVector<Value> worklist{load.getSource(), load.getDest()};
    llvm::SmallPtrSet<Operation *, 8> seen;

    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != load->getBlock())
        continue;
      if (!isa<memref::AllocOp, memref::SubViewOp>(definition))
        continue;
      if (!seen.insert(definition).second)
        continue;
      prologue.push_back(definition);
      for (Value operand : definition->getOperands())
        worklist.push_back(operand);
    }
    return prologue;
  }

  /// Whether every operand of every operation that is about to move is still
  /// defined before the point it is moving to.
  ///
  /// A `memref.subview` over a buffer defined after `earliest` cannot move
  /// above it, and moving it anyway would produce IR that does not verify. The
  /// check is here rather than trusted because the prologue is collected before
  /// `earliest` is known and the two are computed by different walks.
  static bool hoistIsDominanceSafe(ArrayRef<Operation *> prologue,
                                   Operation *earliest,
                                   const llvm::SmallPtrSetImpl<Operation *> &moving) {
    for (Operation *op : prologue) {
      for (Value operand : op->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        if (!definition)
          continue; // a block argument dominates everything in the block
        if (moving.contains(definition))
          continue; // it is moving too, and the order among them is kept
        if (!definition->isBeforeInBlock(earliest))
          return false;
      }
    }
    return true;
  }

  /// Moves one load above the computation before it, or leaves it alone.
  bool hoist(npuisa::DmaLoadOp load) {
    Value source = load.getSource();
    Value destination = load.getDest();

    SmallVector<Operation *> prologue = prologueOf(load);
    llvm::SmallPtrSet<Operation *, 8> movable(prologue.begin(), prologue.end());

    // Walk backwards from the load, stepping over the operations that will move
    // with it, and stopping at the first thing that makes the hoist unsafe.
    Operation *earliest = load;
    bool passedComputation = false;
    for (Operation *op = load->getPrevNode(); op; op = op->getPrevNode()) {
      if (movable.contains(op)) {
        earliest = op;
        continue;
      }

      // The load cannot move above what defines what it reads or writes, unless
      // that definition is moving with it.
      bool definesAnOperand = false;
      for (Value result : op->getResults())
        if (result == source || result == destination)
          definesAnOperand = true;
      if (definesAnOperand)
        break;

      // Nor above anything that touches the buffer the hardware will own for
      // the whole window between the two halves.
      if (mightTouch(op, destination))
        break;

      // **The walk stops at another transfer, and stopping is the right answer
      // rather than a limitation.** Both transfers are charged to the same DMA
      // port, so hoisting one above the other moves work from one end of a
      // saturated timeline to the other and hides nothing; what it does do is
      // extend a buffer's live range, which is the cost double buffering pays
      // for a benefit it would not be getting. One computation deep is the
      // whole of what this rewrite is for.
      //
      // An `await` stops it for a related reason: it is the barrier of some
      // other transfer, and crossing it would reorder two transfers against
      // each other.
      if (isa<npuisa::DmaLoadOp, npuisa::DmaStoreOp, npuisa::DmaLoadAsyncOp,
              npuisa::DmaStoreAsyncOp, npuisa::AwaitOp>(op))
        break;

      if (isComputation(op))
        passedComputation = true;
      earliest = op;
    }

    // A load hoisted past nothing becomes an asynchronous operation whose
    // `await` is the next operation, which canonicalizes straight back to the
    // synchronous form. Declining is the same answer without the residue.
    if (!passedComputation || earliest == load)
      return false;
    if (!hoistIsDominanceSafe(prologue, earliest, movable))
      return false;

    // The prologue moves first and in its own order, so that a view still comes
    // after the allocation it views.
    llvm::sort(prologue, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    for (Operation *op : prologue)
      op->moveBefore(earliest);

    OpBuilder builder(load);
    builder.setInsertionPoint(earliest);
    auto async = npuisa::DmaLoadAsyncOp::create(
        builder, load.getLoc(), npuisa::TokenType::get(&getContext()), source,
        destination);
    builder.setInsertionPoint(load);
    npuisa::AwaitOp::create(builder, load.getLoc(), async.getToken());
    load.erase();
    ++prefetched;
    return true;
  }
};

} // namespace
