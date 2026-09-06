//===- ScratchpadLiveness.h - The scratchpad buffers of a function -*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The IR reading half of Section 13.1's liveness, in one place because two
// passes now need the same answer.
//
// `AllocateScratchpad.cpp` owned this walk and still owns everything about
// spilling. What moved here is the part `-npu-double-buffer` also has to ask:
// which buffers a function has, when each is live, and which budget the function
// is allocated against. The double buffering pass needs it because a prefetch
// that cannot be placed is not a prefetch, so it has to know what extending a
// destination's live range would do to the function's peak **before** it commits
// to the hoist, and the peak is `npuisa::sweepLinePeak` over exactly these
// intervals.
//
// **A second walk beside the first would be two definitions of what a live range
// is**, which is D-0054's own shape one level up: the allocator believed an
// asynchronous transfer finishes at its issue and the pass that emits them
// believed otherwise, and the two agreed until they did not. One walk cannot
// disagree with itself.
//
// The arithmetic these intervals feed has no IR in it and lives in
// `ScratchpadAllocation.h`. This header is the other side of that split.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADLIVENESS_H
#define NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADLIVENESS_H

#include "NPU/Dialect/NPUISA/Transforms/ScratchpadAllocation.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace npuisa {

//===----------------------------------------------------------------------===//
// The attributes this liveness reads and the allocator writes.
//===----------------------------------------------------------------------===//

/// The budget the allocator was given, in bytes.
constexpr llvm::StringLiteral kScratchpadBudgetAttr = "npuisa.scratchpad_budget";
/// Marks the one flat arena allocation, which is not a buffer to be placed.
constexpr llvm::StringLiteral kScratchpadArenaMark = "npuisa.scratchpad_arena";
/// Marks a scratchpad buffer that exists only to hold a reloaded value.
constexpr llvm::StringLiteral kSpillReloadMark = "npuisa.spill_reload";
/// Marks a buffer that has already been spilled.
constexpr llvm::StringLiteral kSpilledMark = "npuisa.spilled";

//===----------------------------------------------------------------------===//
// One buffer.
//===----------------------------------------------------------------------===//

/// One scratchpad allocation, with everything either pass needs to know about
/// it.
///
/// The live interval is the part `-npu-double-buffer` reads. The rest is what
/// the spill rules of Section 13.1 are decided from, and it is collected here
/// rather than in a second pass over the same users because it is the same walk:
/// the worklist that finds the last use is the worklist that finds the writer,
/// the readers and the views.
struct ScratchpadBuffer {
  memref::AllocOp alloc;
  /// The live interval, in operation indices within the entry block.
  LiveInterval interval;
  /// The single operation that writes it, or null when there is not exactly
  /// one. Zero writers means an uninitialised buffer and more than one means a
  /// value no definition point can be named for; both make it unspillable
  /// rather than making them an error, because both are legal IR.
  Operation *writer = nullptr;
  /// The operations that read it, in index order, deduplicated: an operation
  /// that reads the same buffer through two operands is one reader, and gets
  /// one reload.
  llvm::SmallVector<Operation *> readers;
  /// Whether any view of this buffer is **written through**.
  ///
  /// **Not whether a view is taken, and the difference is D-0056.** A view that
  /// is only read through can be re-based onto the reload a spill inserts,
  /// because the reload holds the same bytes at that point and
  /// `replaceUsesOfWith` rewrites the view's source exactly as it rewrites a
  /// reader's operand. A view that is **written** through cannot: the write
  /// would land in the reload and be lost.
  bool viewWrittenThrough = false;
  /// Whether it is a reload buffer, and whether it has already been spilled.
  /// Each is excluded from the spill candidate set for its own reason and both
  /// are recorded in the IR rather than in a side table, so a reader of the
  /// module can see them.
  bool isReload = false;
  bool alreadySpilled = false;
};

//===----------------------------------------------------------------------===//
// The walk.
//===----------------------------------------------------------------------===//

/// Every scratchpad allocation in the function's entry block, with its live
/// range.
///
/// Section 13.1: each range runs from the allocation to the last operation that
/// reads or writes that memref, and sizes come from the memref type and element
/// type, never from a hardcoded factor of four. The size comes from
/// `computeBufferRange`, which is the same function the overlap rule of Section
/// 8 measures byte ranges with, so the allocator and the aliasing analysis can
/// never disagree about how large a buffer is.
///
/// **A use through a view is a use of the underlying buffer**, which matters
/// from the day the lowering started emitting the rank 1 broadcast of ADR 0005:
/// the `memref.reinterpret_cast` is what the instruction names as an operand,
/// and a liveness that only looked at the allocation's direct users would
/// compute a last use before the reads it is actually alive for.
///
/// **An asynchronous transfer holds its destination until its await**, so a
/// range reaches the await rather than the issue. Without that the sweep line
/// ends the range at the issue and hands the same bytes to a buffer defined
/// inside the window, which is the race Section 8's rule 4 exists to prevent.
/// D-0054 is the measurement.
///
/// A failure is a buffer whose byte size cannot be computed, reported on the
/// allocation with the reason.
FailureOr<llvm::SmallVector<ScratchpadBuffer>>
collectScratchpadBuffers(func::FuncOp function);

/// The `npuisa.await` that finishes an asynchronous transfer, or null when `op`
/// is not one or its token is not used exactly once by an await.
///
/// **An asynchronous transfer is not done at its issue.** Everything that asks
/// when a buffer stops being written has to ask this rather than look at the
/// operation that names the buffer: the destination belongs to the DMA engine
/// from the issue until the await, which is Section 8's rule 4 stated from the
/// other side. The liveness below uses it and so does the allocator's spill,
/// which has to put its store after the transfer has finished rather than
/// between the two halves. D-0054 is what believing otherwise cost.
Operation *asynchronousCompletionOf(Operation *op);

/// Whether a set of live intervals can be placed inside the budget, by the
/// allocator's own offset assignment.
///
/// **This is a placement test and the sweep line peak is not**, which is the
/// distinction `ScratchpadAllocation.h` draws and Section 13.1 states: the peak
/// is the minimum arena any placement could need, and a program whose peak fits
/// can still fail to place because offsets are aligned and gaps are not
/// fungible. The allocator's own spill trigger is therefore "offset assignment
/// failed" and never "peak exceeded budget", and a caller asking whether
/// something will fit has to ask the same question the same way.
///
/// `-npu-double-buffer` is the caller and D-0054 is why it asks. The peak was
/// tried first and measured insufficient: on `dilated_stack` at its tight
/// budget of 8064 a prefetch left a peak of 8028, which fits, and the program
/// then needed 8084 to place and did not compile. **56 bytes of alignment is
/// the difference between a lower bound and an answer.**
bool placesWithin(llvm::ArrayRef<LiveInterval> intervals, Strategy strategy,
                  int64_t alignment, int64_t budget);

/// The peak simultaneous live bytes of a collected set, by the sweep line.
///
/// A convenience over `sweepLinePeak` so that a caller holding buffers does not
/// have to copy the intervals out by hand, and so that both callers do it the
/// same way.
PeakPressure peakOf(llvm::ArrayRef<ScratchpadBuffer> buffers);

/// The budget, in bytes, from the pass option, the function attribute, or the
/// default, in that order of precedence.
///
/// **There is one budget on this machine and this is the function that says
/// what it is.** The tiling search, the allocator and the double buffering pass
/// all size against it, and three passes with three ideas of how much
/// scratchpad there is would tile against one number, prefetch against another
/// and spill against a third.
///
/// The option wins over the attribute because the option is a command line
/// override and the attribute is data the driver wrote: an experiment sweeping
/// budgets sets the option, and a module that already carries a budget from an
/// earlier run must not silently ignore it. `option` is the pass option's value,
/// where minus one means "not given"; any other non positive value is a
/// diagnostic rather than a fallback, because a budget of zero is a mistake and
/// not a request.
FailureOr<int64_t> readScratchpadBudget(func::FuncOp function, int64_t option);

} // namespace npuisa
} // namespace mlir

#endif // NPU_DIALECT_NPUISA_TRANSFORMS_SCRATCHPADLIVENESS_H
