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
// **What decides whether it is worth it is the budget**, and that is the reason
// this file depends on the allocator's own liveness. Double buffering doubles
// the prefetched operand's residency: the destination is live from the issue to
// the await instead of from the load to its reader. On five of the seven models
// of this suite that doubling puts the function's peak over its ADR 0008 tight
// budget, `lenet` at 234880 bytes against 194624 and `resnet_block` at 8736
// against 6464, and **a prefetch that cannot be placed is not a prefetch**.
// Those budgets are frozen, so the pass asks what the doubling costs before it
// commits to it and declines the hoist when the answer is over budget. D-0054
// is the measurement that made this a decision rather than a set membership.
//
// **The estimate is the allocator's own offset assignment, and the sweep line
// peak was tried first and is not enough.** Section 13.1 says the peak is a
// lower bound on any placement and that the spill trigger is therefore "offset
// assignment failed" and never "peak exceeded budget". A rule built on the peak
// obeys the first half of that and ignores the second, and the measurement is
// what settled it: on `dilated_stack` at its tight budget of 8064 the peak with
// the prefetch is 8028, which fits, and the arena then needs 8084 and the
// program does not compile. **56 bytes of alignment, and a cell that stops
// existing.** So the question the pass asks is the question it means: run the
// allocator's `assignOffsets` over the intervals the hoist would produce, and
// decline when they do not place.
//
// That is deliberately stronger than the peak in a second way as well. A
// program whose intervals do not place is one the allocator is about to spill,
// and a spill is a `dma_store` and a `dma_load` on the **same port** the
// prefetch exists to unload. A prefetch bought with two transfers on that port
// is not a prefetch either, so declining there is the same sentence rather than
// a second rule.
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
#include "NPU/Dialect/NPUISA/Transforms/ScratchpadAllocation.h"
#include "NPU/Dialect/NPUISA/Transforms/ScratchpadLiveness.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

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

/// What this pass did with one transfer.
///
/// **Three answers rather than two, because a pass that answered no reads
/// differently from one that never asked.** Section 19.0 asks for exactly that
/// separation and D-0054 is why it is worth the enumeration here: for one
/// commit the whole suite reported `not-hoisted` on every transfer, and the
/// reason was neither of the two this pass can now tell apart.
enum class Answer {
  /// Hoisted above a computation and made asynchronous.
  Prefetched,
  /// Nothing safe to overlap it with, so it stays where it is.
  NotHoisted,
  /// There was something to hide it under, and the prefetched destination's
  /// residency would put the function's peak over the budget.
  WouldNotFit,
};

struct DoubleBufferPass
    : public npuisa::impl::NPUDoubleBufferBase<DoubleBufferPass> {
  using npuisa::impl::NPUDoubleBufferBase<
      DoubleBufferPass>::NPUDoubleBufferBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;

    // **One budget and one placement rule on this machine**, read the way the
    // allocator reads them, so that what this pass prices a prefetch against is
    // what the allocator will place it against.
    if (failed(readPlacementOptions(function)))
      return signalPassFailure();
    FailureOr<int64_t> resolved =
        npuisa::readScratchpadBudget(function, static_cast<int64_t>(budget));
    if (failed(resolved))
      return signalPassFailure();
    resolvedBudget = *resolved;

    // The loads are collected before anything moves, because hoisting one
    // changes the order the walk would see.
    SmallVector<npuisa::DmaLoadOp> loads;
    function.walk([&](npuisa::DmaLoadOp load) { loads.push_back(load); });

    for (npuisa::DmaLoadOp load : loads) {
      switch (hoist(load)) {
      case Answer::Prefetched:
        break;
      case Answer::NotHoisted:
        ++notHoisted;
        break;
      case Answer::WouldNotFit:
        ++wouldNotFit;
        break;
      }
    }
  }

private:
  /// The budget every prefetch is priced against, filled by `runOnOperation`.
  int64_t resolvedBudget = npuisa::kDefaultScratchpadBudget;
  /// The allocator's placement rule, so that the estimate is its question.
  npuisa::Strategy resolvedStrategy = npuisa::Strategy::Pack;
  int64_t resolvedAlignment = npuisa::kDefaultAlignment;

  /// Parses the two options that decide what "will fit" means.
  ///
  /// Both are taken as the allocator takes them and refused the way it refuses
  /// them: a bad value is a diagnostic naming the string and listing what is
  /// accepted, not a silent fallback to a default nobody asked for. The
  /// `static_cast<int64_t>` on the alignment is load bearing rather than
  /// defensive, and D-0017 is why: streaming an `llvm::cl::opt` into a
  /// `Diagnostic` selects the `char` overload and prints an alignment of 48 as
  /// the character '0'.
  LogicalResult readPlacementOptions(func::FuncOp function) {
    bool ok = true;
    if (std::optional<npuisa::Strategy> parsed = npuisa::parseStrategy(strategy))
      resolvedStrategy = *parsed;
    else {
      function.emitError() << "unknown strategy '" << strategy
                           << "'. The accepted values are: "
                           << npuisa::strategyOptions();
      ok = false;
    }

    const int64_t requested = alignment;
    if (requested > 0 && (requested & (requested - 1)) == 0)
      resolvedAlignment = requested;
    else {
      function.emitError()
          << "the alignment must be a positive power of two, but it is "
          << requested;
      ok = false;
    }
    return success(ok);
  }

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
  ///
  /// **`npuisa.const` is admitted and that is D-0054's other half.** In the
  /// programs this compiler emits every argument load sits in the entry block
  /// beside the other argument loads, where the walk correctly stops at another
  /// transfer, so the one transfer with a computation before it is the load of a
  /// weight, whose source is an `npuisa.const`. With the constant left behind,
  /// `hoistIsDominanceSafe` refused every one of them and the pass fired on
  /// nothing at all. A constant is a pure definition whose position carries no
  /// meaning beyond the live range it starts, exactly as an allocation's does,
  /// and its result is a **DRAM** buffer, so moving it changes no scratchpad
  /// pressure at all: the residency the estimate prices is the destination's.
  SmallVector<Operation *> prologueOf(npuisa::DmaLoadOp load) {
    SmallVector<Operation *> prologue;
    SmallVector<Value> worklist{load.getSource(), load.getDest()};
    llvm::SmallPtrSet<Operation *, 8> seen;

    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *definition = value.getDefiningOp();
      if (!definition || definition->getBlock() != load->getBlock())
        continue;
      if (!isa<memref::AllocOp, memref::SubViewOp, npuisa::ConstOp>(definition))
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

  /// Whether the function still places inside its budget with this prefetch.
  ///
  /// The allocator's own offset assignment over the intervals the allocator
  /// itself collects, with every allocation that would move with the transfer
  /// starting at the hoist's destination instead of where it is defined. That
  /// extension **is** double buffering: the buffer belongs to the DMA engine
  /// from the issue to the await, and the liveness this shares with the
  /// allocator already ends an asynchronous range at the await rather than at
  /// the issue.
  ///
  /// **The sweep line peak is not this question**, and the file comment carries
  /// the 56 bytes that proved it. What this asks is what the allocator will
  /// ask, with the same strategy, the same alignment and the same budget.
  ///
  /// A destination whose residency this cannot price is declined rather than
  /// taken on trust. The alternative is committing to a cost nobody measured,
  /// which is the thing this whole function exists to stop.
  bool prefetchFits(Operation *earliest, ArrayRef<Operation *> prologue) {
    func::FuncOp function = getOperation();
    if (!function.getBody().hasOneBlock())
      return false;
    Block &block = function.getBody().front();
    if (earliest->getBlock() != &block)
      return false;

    FailureOr<SmallVector<npuisa::ScratchpadBuffer>> buffers =
        npuisa::collectScratchpadBuffers(function);
    if (failed(buffers))
      return false;

    llvm::DenseMap<Operation *, int64_t> indices;
    for (auto [index, op] : llvm::enumerate(block))
      indices[&op] = static_cast<int64_t>(index);
    const int64_t target = indices[earliest];

    llvm::SmallPtrSet<Operation *, 8> moving(prologue.begin(), prologue.end());
    bool priced = false;
    SmallVector<npuisa::LiveInterval> intervals;
    intervals.reserve(buffers->size());
    for (const npuisa::ScratchpadBuffer &buffer : *buffers) {
      // The handle is copied out rather than the struct being taken by value:
      // an operation handle is a value type and its accessors are not const.
      // That is the MLIR idiom and it costs a pointer copy.
      memref::AllocOp alloc = buffer.alloc;
      npuisa::LiveInterval interval = buffer.interval;
      if (moving.contains(alloc.getOperation())) {
        interval.definition = std::min(interval.definition, target);
        priced = true;
      }
      intervals.push_back(interval);
    }
    if (!priced)
      return false;

    return npuisa::placesWithin(intervals, resolvedStrategy, resolvedAlignment,
                                resolvedBudget);
  }

  /// Moves one load above the computation before it, or leaves it alone.
  Answer hoist(npuisa::DmaLoadOp load) {
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
      return Answer::NotHoisted;
    if (!hoistIsDominanceSafe(prologue, earliest, movable))
      return Answer::NotHoisted;

    // **The last question is what it costs**, and it is asked last because it
    // is the only one that needs a walk of the whole function. Everything above
    // decides whether the rewrite is legal; this decides whether the program
    // that comes out can be placed.
    if (!prefetchFits(earliest, prologue))
      return Answer::WouldNotFit;

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
    return Answer::Prefetched;
  }
};

} // namespace
