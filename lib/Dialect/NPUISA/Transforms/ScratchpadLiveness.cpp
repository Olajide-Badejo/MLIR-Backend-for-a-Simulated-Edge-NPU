//===- ScratchpadLiveness.cpp - The scratchpad buffers of a function ------===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The walk `AllocateScratchpad.cpp` used to own, moved here whole so that
// `-npu-double-buffer` can ask the same question rather than a second version
// of it. The header says why. Nothing about the walk changed in the move, and
// the allocator's tests are what say so.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/ScratchpadLiveness.h"

#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

using namespace mlir;

namespace {

/// Whether an operation is one of the view forms this project produces.
///
/// `memref.reinterpret_cast` is here because the lowering emits one for the
/// rank 1 channel broadcast of ADR 0005, and this project's views end up
/// underneath those casts. `memref.view` and `memref.subview` are here because
/// the allocator and the tiling pass produce them.
bool isViewLike(Operation *op) {
  return isa<memref::ViewOp, memref::SubViewOp, memref::ReinterpretCastOp,
             memref::CastOp>(op);
}

/// Whether a memref lives in the scratchpad.
bool isScratchpad(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && isa_and_present<npu::ScratchpadAttr>(memref.getMemorySpace());
}

/// The effects an operation declares on one specific value.
///
/// An operation that does not implement `MemoryEffectOpInterface` at all has
/// not said it touches nothing, so it is reported as both a read and a write.
/// That is the same conservatism the overlap rule of Section 8 uses, applied
/// here: an unknown operation makes a buffer unspillable rather than making the
/// spill wrong.
void effectsOnValue(Operation *op, Value value, bool &reads, bool &writes) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) {
    reads = true;
    writes = true;
    return;
  }

  llvm::SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffectsOnValue(value, effects);
  reads = false;
  writes = false;
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      reads = true;
    if (isa<MemoryEffects::Write>(effect.getEffect()))
      writes = true;
  }
}

} // namespace

/// A token with anything other than one `npuisa.await` use is a program the
/// verifier has already refused, so returning null there is not a policy: it is
/// the answer for IR that cannot reach these passes.
Operation *npuisa::asynchronousCompletionOf(Operation *op) {
  Value token;
  if (auto load = dyn_cast<npuisa::DmaLoadAsyncOp>(op))
    token = load.getToken();
  else if (auto store = dyn_cast<npuisa::DmaStoreAsyncOp>(op))
    token = store.getToken();
  else
    return nullptr;
  if (!token || !token.hasOneUse())
    return nullptr;
  return dyn_cast<npuisa::AwaitOp>(*token.user_begin());
}

FailureOr<llvm::SmallVector<npuisa::ScratchpadBuffer>>
npuisa::collectScratchpadBuffers(func::FuncOp function) {
  Block &block = function.getBody().front();

  llvm::DenseMap<Operation *, int64_t> indices;
  for (auto [index, op] : llvm::enumerate(block))
    indices[&op] = static_cast<int64_t>(index);

  llvm::SmallVector<npuisa::ScratchpadBuffer> buffers;
  for (Operation &op : block) {
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc || !isScratchpad(alloc.getType()))
      continue;
    if (alloc->hasAttr(npuisa::kScratchpadArenaMark))
      continue;

    std::optional<npuisa::BufferRange> range =
        npuisa::computeBufferRange(alloc.getResult());
    if (!range)
      return alloc.emitError()
             << "this scratchpad allocation has no byte size the allocator can "
                "compute, because "
             << npuisa::describeWhyNotAnalysable(alloc.getResult())
             << ". Section 13.1 takes sizes from the memref type and its "
                "element type, so a buffer it cannot measure is refused rather "
                "than guessed at";

    npuisa::ScratchpadBuffer buffer;
    buffer.alloc = alloc;
    buffer.interval.definition = indices[&op];
    buffer.interval.lastUse = buffer.interval.definition;
    buffer.interval.bytes = range->size;
    buffer.isReload = alloc->hasAttr(npuisa::kSpillReloadMark);
    buffer.alreadySpilled = alloc->hasAttr(npuisa::kSpilledMark);

    // The worklist walks views as well as direct users, so a buffer read only
    // through a broadcast cast is still live at that read.
    int64_t writers = 0;
    llvm::SmallVector<Value> worklist{alloc.getResult()};
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      const bool isTheAllocation = value == alloc.getResult();
      for (Operation *user : value.getUsers()) {
        Operation *ancestor = block.findAncestorOpInBlock(*user);
        if (!ancestor)
          continue;
        buffer.interval.lastUse =
            std::max(buffer.interval.lastUse, indices[ancestor]);

        // **An asynchronous transfer holds this buffer until its await**, so
        // the interval reaches the await rather than the issue. Without this
        // the sweep line ends the range at the issue and hands the same bytes
        // to a buffer defined inside the window, which is the race Section 8's
        // rule 4 exists to prevent and which the verifier catches only where
        // the two happen to be compared. D-0054 is the measurement: a
        // prefetched weight was placed at offset 0 and the reload of a
        // different buffer was placed at offset 0 inside its window.
        if (Operation *finish = npuisa::asynchronousCompletionOf(ancestor))
          buffer.interval.lastUse =
              std::max(buffer.interval.lastUse, indices[finish]);

        if (isViewLike(user)) {
          for (Value result : user->getResults())
            worklist.push_back(result);
          // **A view taken directly of the allocation is a reader**, and that
          // is what lets the spill rewrite re-base it. `spill` replaces uses of
          // the allocation in every reader after the write, and a view's use of
          // the allocation is its source operand, so the same call moves the
          // view onto the reload without knowing it is a view. A view of a view
          // is not listed, because it names the inner view's result and follows
          // it for free. D-0056 is why this is here.
          if (isTheAllocation && !llvm::is_contained(buffer.readers, user))
            buffer.readers.push_back(user);
          continue;
        }

        // A use reached **through** a view still decides one thing: whether the
        // view is written through. That is the case a reload cannot serve, and
        // it is the only case the view rule now refuses.
        if (!isTheAllocation) {
          bool viewReads = false;
          bool viewWrites = false;
          effectsOnValue(user, value, viewReads, viewWrites);
          if (viewWrites)
            buffer.viewWrittenThrough = true;
          continue;
        }

        bool reads = false;
        bool writes = false;
        effectsOnValue(user, value, reads, writes);
        if (writes) {
          ++writers;
          buffer.writer = user;
        }
        if (reads && !llvm::is_contained(buffer.readers, user))
          buffer.readers.push_back(user);
      }
    }

    if (writers != 1)
      buffer.writer = nullptr;
    llvm::sort(buffer.readers, [&](Operation *left, Operation *right) {
      return indices[block.findAncestorOpInBlock(*left)] <
             indices[block.findAncestorOpInBlock(*right)];
    });
    buffers.push_back(std::move(buffer));
  }

  return buffers;
}

bool npuisa::placesWithin(llvm::ArrayRef<npuisa::LiveInterval> intervals,
                          npuisa::Strategy strategy, int64_t alignment,
                          int64_t budget) {
  return npuisa::assignOffsets(intervals, strategy, alignment, budget)
      .has_value();
}

npuisa::PeakPressure
npuisa::peakOf(llvm::ArrayRef<npuisa::ScratchpadBuffer> buffers) {
  llvm::SmallVector<npuisa::LiveInterval> intervals;
  intervals.reserve(buffers.size());
  for (const npuisa::ScratchpadBuffer &buffer : buffers)
    intervals.push_back(buffer.interval);
  return npuisa::sweepLinePeak(intervals);
}

FailureOr<int64_t> npuisa::readScratchpadBudget(func::FuncOp function,
                                                int64_t option) {
  if (option > 0)
    return option;
  if (option != -1)
    return function.emitError()
           << "the budget option must be positive, but it is " << option;

  Attribute existing = function->getAttr(npuisa::kScratchpadBudgetAttr);
  if (!existing)
    return npuisa::kDefaultScratchpadBudget;

  auto integer = dyn_cast<IntegerAttr>(existing);
  if (!integer || integer.getInt() <= 0)
    return function.emitError()
           << "the " << npuisa::kScratchpadBudgetAttr << " attribute of @"
           << function.getName()
           << " must be a positive integer, but it is " << existing;
  return integer.getInt();
}
