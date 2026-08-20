//===- AllocateScratchpad.cpp - Scratchpad allocation -----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 13.1: liveness over a straight line function, offset assignment into
// one flat arena, and spilling when the offsets do not fit.
//
// **What this pass produces.** Every `memref.alloc` in `#npu.scratchpad`
// disappears and is replaced by a `memref.view` at a constant byte offset over
// one flat `memref<Nxi8, #npu.scratchpad>`. Section 8 requires exactly that
// shape and gives the reason: the offset is then an SSA operand every verifier
// and consumer can see, and a pass cannot silently drop it the way it can drop
// a discardable attribute it does not recognise. Spill reloads are views over
// the same arena, so aliasing is visible in the IR rather than conventional.
//
// **The three questions this pass answers, in order, because getting the order
// wrong is the classic bug.**
//
//   1. How much is live at once. That is the sweep line of
//      `ScratchpadAllocation.h`, and it is a *lower bound* on the arena any
//      placement could need.
//   2. Where does each buffer go. That is offset assignment, and it can fail
//      on a program whose peak fits, because of fragmentation.
//   3. What to spill. The trigger is **"offset assignment failed"**, never
//      "peak exceeded budget". Getting this backwards spills when it need not
//      and fails to spill when it must, and
//      `test/Dialect/NPUISA/scratchpad-alloc.mlir` carries the fragmentation
//      case that tells the two questions apart.
//
// **The arithmetic lives in ScratchpadAllocation.cpp and has no IR in it.**
// This file is the half that reads and writes MLIR: it finds the buffers,
// computes their live ranges, materialises the views, rewrites the spills, and
// turns a failure into a diagnostic with numbers in it. The property test of
// Section 17.2 tests the other half directly.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAMemoryOverlap.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/Transforms/ScratchpadAllocation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <optional>

namespace mlir::npuisa {
#define GEN_PASS_DEF_NPUALLOCATESCRATCHPAD
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"
} // namespace mlir::npuisa

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// The attribute names this pass writes and reads.
//===----------------------------------------------------------------------===//

/// The budget the allocator was given, in bytes. Read from the function when it
/// carries one, written back so that the encoder and the simulator read the
/// number the allocation was actually made against.
constexpr llvm::StringLiteral kBudgetAttr = "npuisa.scratchpad_budget";
/// The arena that was actually used, in bytes: the assigned high water mark.
constexpr llvm::StringLiteral kBytesAttr = "npuisa.scratchpad_bytes";
/// The sweep line peak, in bytes. The lower bound the high water mark is
/// measured against, carried so the ratio below can be checked rather than
/// trusted.
constexpr llvm::StringLiteral kPeakAttr = "npuisa.scratchpad_peak_bytes";
/// The headline allocator metric of Section 13.1: assigned high water mark
/// divided by the sweep line peak.
constexpr llvm::StringLiteral kFragmentationAttr = "npuisa.fragmentation_ratio";
/// How many buffers were spilled.
constexpr llvm::StringLiteral kSpillCountAttr = "npuisa.spill_count";
/// How many DMA operations this pass inserted. Section 8 asks the allocator and
/// the tiling pass each to count the DMA they insert, so that the sum over the
/// three permitted producers is checkable rather than asserted.
constexpr llvm::StringLiteral kSpillDmaAttr = "npuisa.spill_dma_count";

/// Marks the one flat arena allocation.
constexpr llvm::StringLiteral kArenaMark = "npuisa.scratchpad_arena";
/// Marks a DRAM buffer that exists only to hold a spilled value.
constexpr llvm::StringLiteral kSpillSlotMark = "npuisa.spill_slot";
/// Marks a scratchpad buffer that exists only to hold a reloaded value.
constexpr llvm::StringLiteral kSpillReloadMark = "npuisa.spill_reload";
/// Marks a buffer that has already been spilled, so that the spill loop cannot
/// choose it a second time and spill its own `dma_store`.
constexpr llvm::StringLiteral kSpilledMark = "npuisa.spilled";

//===----------------------------------------------------------------------===//
// Buffers.
//===----------------------------------------------------------------------===//

/// One scratchpad allocation, with everything the allocator needs to know about
/// it.
struct Buffer {
  memref::AllocOp alloc;
  /// The live interval, in operation indices within the entry block.
  npuisa::LiveInterval interval;
  /// The single operation that writes it, or null when there is not exactly
  /// one. Zero writers means an uninitialised buffer and more than one means a
  /// value this pass cannot name a definition point for; both make it
  /// unspillable rather than making them an error, because both are legal IR.
  Operation *writer = nullptr;
  /// The operations that read it, in index order, deduplicated: an operation
  /// that reads the same buffer through two operands is one reader, and gets
  /// one reload.
  llvm::SmallVector<Operation *> readers;
  /// Whether anything took a view of it. A buffer with a view has more than one
  /// SSA name for the same bytes, and a spill that rewrote only the direct uses
  /// would leave the view reading a buffer whose contents had moved.
  bool hasViewUser = false;
  /// Whether it is a reload buffer, a spill slot's landing site, or a buffer
  /// that has already been spilled. Each of these is excluded from the
  /// candidate set for a different reason and all three are recorded in the IR
  /// rather than in a side table, so a reader of the module can see them.
  bool isReload = false;
  bool alreadySpilled = false;
};

/// Whether an operation is one of the view forms this project produces.
///
/// `memref.reinterpret_cast` is here because the lowering emits one for the
/// rank 1 channel broadcast of ADR 0005, and this pass's views end up
/// underneath those casts. `memref.view` and `memref.subview` are here because
/// this pass and any later tiling pass produce them.
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

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

struct AllocateScratchpadPass
    : public npuisa::impl::NPUAllocateScratchpadBase<AllocateScratchpadPass> {
  using npuisa::impl::NPUAllocateScratchpadBase<
      AllocateScratchpadPass>::NPUAllocateScratchpadBase;

  void runOnOperation() override;

private:
  /// Resolved options, filled by `readOptions`.
  npuisa::Strategy resolvedStrategy = npuisa::Strategy::Pack;
  npuisa::SpillHeuristic resolvedHeuristic =
      npuisa::SpillHeuristic::LongestRange;
  int64_t resolvedAlignment = npuisa::kDefaultAlignment;

  LogicalResult readOptions(func::FuncOp function);
  FailureOr<int64_t> readBudget(func::FuncOp function);
  FailureOr<llvm::SmallVector<Buffer>> collect(func::FuncOp function);
  void spill(Buffer &buffer, int64_t &spilledBuffers, int64_t &insertedDma);
  void materialise(func::FuncOp function, llvm::ArrayRef<Buffer> buffers,
                   const npuisa::Placement &placement);
};

//===----------------------------------------------------------------------===//
// Options.
//===----------------------------------------------------------------------===//

/// Both enumerated options are taken as strings and parsed here rather than
/// declared as ODS enum options, and that is deliberate. Section 13.1: an
/// unknown option value is a diagnostic naming the offending string and listing
/// the accepted values, then a pass failure. An ODS enum option rejects the
/// string inside the pass manager's own option parser, before the pass runs,
/// with a message that names neither the pass nor the reason and that no
/// `-verify-diagnostics` test can capture. A typo must not silently select a
/// heuristic nobody asked for, and it must not produce a message that leaves
/// the reader guessing which of two options was misspelt.
/// Every bad option is reported, not just the first. Somebody who mistyped two
/// of them should not have to fix one, rerun, and discover the second.
LogicalResult AllocateScratchpadPass::readOptions(func::FuncOp function) {
  bool ok = true;

  if (std::optional<npuisa::Strategy> parsed = npuisa::parseStrategy(strategy))
    resolvedStrategy = *parsed;
  else {
    function.emitError() << "unknown strategy '" << strategy
                         << "'. The accepted values are: "
                         << npuisa::strategyOptions();
    ok = false;
  }

  if (std::optional<npuisa::SpillHeuristic> parsed =
          npuisa::parseSpillHeuristic(spillHeuristic))
    resolvedHeuristic = *parsed;
  else {
    function.emitError() << "unknown spill heuristic '" << spillHeuristic
                         << "'. The accepted values are: "
                         << npuisa::spillHeuristicOptions();
    ok = false;
  }

  // `static_cast<int64_t>` on the option rather than streaming it directly, and
  // it is load bearing rather than defensive. A pass option is an
  // `llvm::cl::opt` with an implicit conversion to its data type, and streaming
  // one into a `Diagnostic` selects the `char` overload: an alignment of 48
  // printed as the character '0', which is a diagnostic that lies about the
  // number it was given. Defect D-0017.
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

/// The budget, in bytes, from the pass option, the function attribute, or the
/// default, in that order of precedence.
///
/// The option wins over the attribute because the option is a command line
/// override and the attribute is data the driver wrote: an experiment sweeping
/// budgets sets the option, and a module that already carries a budget from an
/// earlier run must not silently ignore it. The attribute is written back
/// either way, so after this pass the function always says which budget it was
/// allocated against.
FailureOr<int64_t> AllocateScratchpadPass::readBudget(func::FuncOp function) {
  // The same `static_cast` as the alignment check above, for the same reason:
  // D-0017 is about how a pass option prints, not about one option.
  const int64_t requested = budget;
  if (requested > 0)
    return requested;
  if (requested != -1)
    return function.emitError()
           << "the budget option must be positive, but it is " << requested;

  Attribute existing = function->getAttr(kBudgetAttr);
  if (!existing)
    return npuisa::kDefaultScratchpadBudget;

  auto integer = dyn_cast<IntegerAttr>(existing);
  if (!integer || integer.getInt() <= 0)
    return function.emitError()
           << "the " << kBudgetAttr << " attribute of @" << function.getName()
           << " must be a positive integer, but it is " << existing;
  return integer.getInt();
}

//===----------------------------------------------------------------------===//
// Liveness.
//===----------------------------------------------------------------------===//

/// Finds every scratchpad allocation and computes its live range.
///
/// Section 13.1: each range runs from the allocation to the last operation that
/// reads or writes that memref, and sizes come from the memref type and element
/// type, never from a hardcoded factor of four. The size comes from
/// `computeBufferRange`, which is the same function the overlap rule of
/// Section 8 measures byte ranges with, so the allocator and the aliasing
/// analysis can never disagree about how large a buffer is.
///
/// A use *through a view* is a use of the underlying buffer. That matters from
/// the day the lowering started emitting the rank 1 broadcast of ADR 0005: the
/// `memref.reinterpret_cast` is what the instruction names as an operand, and a
/// liveness that only looked at the allocation's direct users would compute a
/// last use before the reads it is actually alive for.
FailureOr<llvm::SmallVector<Buffer>>
AllocateScratchpadPass::collect(func::FuncOp function) {
  Block &block = function.getBody().front();

  llvm::DenseMap<Operation *, int64_t> indices;
  for (auto [index, op] : llvm::enumerate(block))
    indices[&op] = static_cast<int64_t>(index);

  llvm::SmallVector<Buffer> buffers;
  for (Operation &op : block) {
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc || !isScratchpad(alloc.getType()))
      continue;
    if (alloc->hasAttr(kArenaMark))
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

    Buffer buffer;
    buffer.alloc = alloc;
    buffer.interval.definition = indices[&op];
    buffer.interval.lastUse = buffer.interval.definition;
    buffer.interval.bytes = range->size;
    buffer.isReload = alloc->hasAttr(kSpillReloadMark);
    buffer.alreadySpilled = alloc->hasAttr(kSpilledMark);

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

        if (isViewLike(user)) {
          buffer.hasViewUser = true;
          for (Value result : user->getResults())
            worklist.push_back(result);
          continue;
        }

        // Only the allocation's own uses tell us where the value is defined and
        // read. A use through a view is counted for liveness above and makes
        // the buffer unspillable, so it never reaches the rewrite that would
        // need to know which of the two names to replace.
        if (!isTheAllocation)
          continue;

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

/// Whether a buffer can be spilled, and why not when it cannot.
///
/// The rules, each with a reason rather than a convention:
///
/// - **Exactly one writer.** The spill semantics of Section 13.1 are a
///   `dma_store` after *the* definition, and a buffer written twice has two.
/// - **No view users.** A view is a second SSA name for the same bytes;
///   rewriting the direct uses and leaving the view behind would make the view
///   read a buffer whose contents had moved to DRAM.
/// - **Not a reload and not already spilled.** Both would let the loop spill
///   its own output, which is how a spill loop fails to terminate.
/// - **At least one read after the write.** Spilling a buffer nothing reads
///   later adds a `dma_store` and shortens nothing, so it costs DRAM traffic
///   and buys no space.
/// - **An identity layout.** A `dma_store` requires its two operands to agree
///   on shape and element type, and a buffer whose layout map permutes its
///   extents has no DRAM counterpart this pass can name without deciding what
///   order to write it in. That decision belongs to a relayouting transfer,
///   which Section 12 marks as a named future extension and which nothing may
///   cite as available today.
bool isSpillable(const Buffer &buffer,
                 const llvm::DenseMap<Operation *, int64_t> &indices) {
  if (!buffer.writer || buffer.hasViewUser || buffer.isReload ||
      buffer.alreadySpilled)
    return false;

  // An operation handle is a value type and its accessors are not const, so the
  // handle is copied out rather than the struct being taken by value. That is
  // the MLIR idiom and it costs a pointer copy.
  memref::AllocOp alloc = buffer.alloc;
  if (!alloc.getType().getLayout().isIdentity())
    return false;

  auto writerIndex = indices.find(buffer.writer);
  if (writerIndex == indices.end())
    return false;
  return llvm::any_of(buffer.readers, [&](Operation *reader) {
    auto found = indices.find(reader);
    return found != indices.end() && found->second > writerIndex->second;
  });
}

//===----------------------------------------------------------------------===//
// Spilling.
//===----------------------------------------------------------------------===//

/// Rewrites one buffer into a DRAM slot plus a reload per later use.
///
/// Section 13.1's spill semantics, exactly: a `dma_store` after the definition
/// and a `dma_load` before each later use, with the reloaded value replacing
/// that use. Under memrefs the reload needs its own scratchpad allocation, and
/// that allocation participates in liveness like any other, which is why this
/// pass recollects the buffers from the IR after every spill instead of
/// patching a side table. A second spill round that did not see the reloads
/// would mis-size the peak.
///
/// **This is the second of the three permitted DMA producers of Section 8**,
/// and the count it inserts is recorded on the function so the sum over the
/// three is checkable.
///
/// The DRAM slot is a `memref.alloc` in `#npu.dram`. That is the one place in
/// this compiler that allocates DRAM, and it is marked in the IR so the encoder
/// can find it; see the P5 extension in `docs/ARCHITECTURE.md`, which amends
/// the P4 sentence that nothing below the tensor level allocates DRAM.
void AllocateScratchpadPass::spill(Buffer &buffer, int64_t &spilledBuffers,
                                   int64_t &insertedDma) {
  OpBuilder builder(buffer.alloc);
  Location loc = buffer.alloc.getLoc();
  MemRefType scratchType = buffer.alloc.getType();
  auto dramType = MemRefType::get(scratchType.getShape(),
                                  scratchType.getElementType(),
                                  MemRefLayoutAttrInterface{},
                                  npu::DramAttr::get(&getContext()));

  Block *block = buffer.alloc->getBlock();
  Operation *writer = buffer.alloc->getBlock()->findAncestorOpInBlock(
      *buffer.writer);

  builder.setInsertionPointAfter(writer);
  auto slot = memref::AllocOp::create(builder, loc, dramType);
  slot->setAttr(kSpillSlotMark, builder.getUnitAttr());
  npuisa::DmaStoreOp::create(builder, loc, buffer.alloc.getResult(),
                             slot.getResult());
  ++insertedDma;

  // The readers are walked in index order and only the ones after the write are
  // reloaded. A read before the write reads the buffer itself, which is still
  // correct: the store happens after the write, so nothing before it has moved.
  llvm::DenseMap<Operation *, int64_t> indices;
  for (auto [index, op] : llvm::enumerate(*block))
    indices[&op] = static_cast<int64_t>(index);
  const int64_t writerIndex = indices[writer];

  for (Operation *reader : buffer.readers) {
    Operation *ancestor = block->findAncestorOpInBlock(*reader);
    if (!ancestor || indices[ancestor] <= writerIndex)
      continue;

    // The reload carries the *reader's* location, not the spilled buffer's. A
    // diagnostic about a reload is a diagnostic about the instruction that
    // needed the value back, and pointing it at the allocation the value came
    // from sends the reader to the wrong line.
    builder.setInsertionPoint(ancestor);
    Location reloadLoc = ancestor->getLoc();
    auto reload = memref::AllocOp::create(builder, reloadLoc, scratchType);
    reload->setAttr(kSpillReloadMark, builder.getUnitAttr());
    npuisa::DmaLoadOp::create(builder, reloadLoc, slot.getResult(),
                              reload.getResult());
    ++insertedDma;
    reader->replaceUsesOfWith(buffer.alloc.getResult(), reload.getResult());
  }

  buffer.alloc->setAttr(kSpilledMark, builder.getUnitAttr());
  ++spilledBuffers;
}

//===----------------------------------------------------------------------===//
// Materialisation.
//===----------------------------------------------------------------------===//

/// Replaces every allocation with a view at its assigned offset.
///
/// One `memref.alloc` of `memref<Nxi8, #npu.scratchpad>` at the top of the
/// block is the arena, and every buffer becomes a `memref.view` into it at a
/// constant byte offset. `memref.view` requires an identity layout on its
/// result, so a buffer whose type carries a strided layout map, which is what
/// an NHWC tensor lowers to, gets the view at its NCHW extents followed by a
/// `memref.reinterpret_cast` that puts the layout back. The bytes are the same
/// bytes either way: a permutation layout spans exactly the contiguous extent
/// its shape does.
void AllocateScratchpadPass::materialise(func::FuncOp function,
                                         llvm::ArrayRef<Buffer> buffers,
                                         const npuisa::Placement &placement) {
  if (buffers.empty())
    return;

  Block &block = function.getBody().front();
  OpBuilder builder(&block, block.begin());
  Location loc = function.getLoc();

  auto arenaType =
      MemRefType::get({placement.highWaterMark}, builder.getI8Type(),
                      MemRefLayoutAttrInterface{},
                      npu::ScratchpadAttr::get(&getContext()));
  auto arena = memref::AllocOp::create(builder, loc, arenaType);
  arena.setAlignment(resolvedAlignment);
  arena->setAttr(kArenaMark, builder.getUnitAttr());

  for (auto [index, buffer] : llvm::enumerate(buffers)) {
    const int64_t offset = placement.offsets[index];
    memref::AllocOp alloc = buffer.alloc;
    MemRefType type = alloc.getType();
    Location allocLoc = alloc.getLoc();
    builder.setInsertionPoint(alloc);

    auto shift = arith::ConstantIndexOp::create(builder, allocLoc, offset);
    auto flat = MemRefType::get(type.getShape(), type.getElementType(),
                                MemRefLayoutAttrInterface{},
                                type.getMemorySpace());
    Value view = memref::ViewOp::create(builder, allocLoc, flat,
                                        arena.getResult(), shift, ValueRange{})
                     .getResult();

    if (!type.getLayout().isIdentity()) {
      llvm::SmallVector<int64_t> strides;
      int64_t layoutOffset = 0;
      if (succeeded(type.getStridesAndOffset(strides, layoutOffset))) {
        llvm::SmallVector<OpFoldResult> sizes;
        for (int64_t extent : type.getShape())
          sizes.push_back(builder.getIndexAttr(extent));
        llvm::SmallVector<OpFoldResult> stridesAsFold;
        for (int64_t stride : strides)
          stridesAsFold.push_back(builder.getIndexAttr(stride));
        view = memref::ReinterpretCastOp::create(
                   builder, allocLoc, type, view,
                   builder.getIndexAttr(layoutOffset), sizes, stridesAsFold)
                   .getResult();
      }
    }

    alloc.getResult().replaceAllUsesWith(view);
    alloc.erase();
  }
}

//===----------------------------------------------------------------------===//
// runOnOperation.
//===----------------------------------------------------------------------===//

void AllocateScratchpadPass::runOnOperation() {
  func::FuncOp function = getOperation();
  if (function.isExternal())
    return;

  if (failed(readOptions(function)))
    return signalPassFailure();

  // Section 8: multiple blocks are diagnosed, not ignored. The liveness of
  // Section 13.1 is over a straight line function, and an operation index in a
  // second block is not comparable with one in the first, so a live range
  // spanning them is not a range at all.
  if (!function.getBody().hasOneBlock()) {
    function.emitError()
        << "the allocator requires a single block function body, but @"
        << function.getName() << " has "
        << std::distance(function.getBody().begin(), function.getBody().end())
        << " blocks. Liveness here is an ordering of one straight line "
           "instruction stream, and this instruction set has no branch "
           "instructions for a second block to be reached by";
    return signalPassFailure();
  }

  // Idempotence. After this pass there are no scratchpad allocations left
  // except the arena, and the arena is one, so a second run would allocate an
  // arena for the arena. The function attribute is the guard rather than the
  // arena's own mark, because the attribute is what a reader and a driver can
  // both see, and because a function with no buffers at all still carries it.
  if (function->hasAttr(kBytesAttr))
    return;

  FailureOr<int64_t> resolvedBudget = readBudget(function);
  if (failed(resolvedBudget))
    return signalPassFailure();

  int64_t spilledBuffers = 0;
  int64_t insertedDma = 0;

  // The spill loop. Every round recomputes liveness and the peak from the IR,
  // because a spill inserts allocations that participate in liveness like any
  // other. The sweep line is what makes recomputing affordable: the naive
  // nested peak would make this loop cubic, which is the growth
  // `experiments/compile_time_benchmark.py` exists to make visible.
  llvm::SmallVector<Buffer> buffers;
  npuisa::Placement placement;
  npuisa::PeakPressure peak;
  while (true) {
    FailureOr<llvm::SmallVector<Buffer>> collected = collect(function);
    if (failed(collected))
      return signalPassFailure();
    buffers = std::move(*collected);

    llvm::SmallVector<npuisa::LiveInterval> intervals;
    intervals.reserve(buffers.size());
    for (const Buffer &buffer : buffers)
      intervals.push_back(buffer.interval);

    peak = npuisa::sweepLinePeak(intervals);

    npuisa::PlacementFailure failure;
    if (std::optional<npuisa::Placement> assigned = npuisa::assignOffsets(
            intervals, resolvedStrategy, resolvedAlignment, *resolvedBudget,
            &failure)) {
      placement = std::move(*assigned);
      break;
    }

    // Offset assignment failed, which is the spill trigger and the only spill
    // trigger. The candidate set is the buffers live across the pressure peak,
    // per Section 13.1, and the heuristic picks among them.
    Block &block = function.getBody().front();
    llvm::DenseMap<Operation *, int64_t> indices;
    for (auto [index, op] : llvm::enumerate(block))
      indices[&op] = static_cast<int64_t>(index);

    llvm::SmallVector<npuisa::SpillCandidate> candidates;
    llvm::SmallVector<int64_t> candidateBuffers;
    for (auto [index, buffer] : llvm::enumerate(buffers)) {
      if (buffer.interval.definition > peak.index ||
          peak.index > buffer.interval.lastUse)
        continue;
      npuisa::SpillCandidate candidate;
      candidate.interval = buffer.interval;
      candidate.spillable = isSpillable(buffer, indices);
      candidate.usesAfterPeak = llvm::count_if(
          buffer.readers, [&](Operation *reader) {
            Operation *ancestor = block.findAncestorOpInBlock(*reader);
            return ancestor && indices[ancestor] > peak.index;
          });
      candidates.push_back(candidate);
      candidateBuffers.push_back(static_cast<int64_t>(index));
    }

    std::optional<int64_t> victim =
        npuisa::chooseSpillVictim(candidates, resolvedHeuristic);
    if (!victim) {
      const Buffer &stuck = buffers[failure.interval];
      memref::AllocOp stuckAlloc = stuck.alloc;
      stuckAlloc.emitError()
          << "the scratchpad budget of " << *resolvedBudget
          << " bytes is too small: this buffer of " << stuck.interval.bytes
          << " bytes could not be placed below offset " << failure.wantedOffset
          << " in @" << function.getName()
          << ", and no buffer live across the pressure peak can be spilled. "
             "The sweep line peak is "
          << peak.bytes
          << " bytes, which is a lower bound on any placement; the "
             "requirement is therefore at least "
          << std::max(peak.bytes, failure.wantedOffset + stuck.interval.bytes)
          << " bytes against a budget of " << *resolvedBudget;
      return signalPassFailure();
    }

    spill(buffers[candidateBuffers[*victim]], spilledBuffers, insertedDma);
  }

  materialise(function, buffers, placement);

  // The attributes Section 8 puts on the function, plus the two numbers the
  // headline metric of Section 13.1 is computed from. The ratio is written as
  // well as its two operands so that a consumer reads one field, and the two
  // operands are written as well as the ratio so that a reader can check it.
  OpBuilder builder(function);
  function->setAttr(kBudgetAttr, builder.getI64IntegerAttr(*resolvedBudget));
  function->setAttr(kBytesAttr,
                    builder.getI64IntegerAttr(placement.highWaterMark));
  function->setAttr(kPeakAttr, builder.getI64IntegerAttr(peak.bytes));
  function->setAttr(kFragmentationAttr,
                    builder.getF64FloatAttr(
                        peak.bytes == 0
                            ? 1.0
                            : static_cast<double>(placement.highWaterMark) /
                                  static_cast<double>(peak.bytes)));
  function->setAttr(kSpillCountAttr, builder.getI64IntegerAttr(spilledBuffers));
  function->setAttr(kSpillDmaAttr, builder.getI64IntegerAttr(insertedDma));

  allocatedBuffers += buffers.size();
  allocatedBytes += placement.highWaterMark;
  peakBytes += peak.bytes;
  spilledBufferCount += spilledBuffers;
  insertedDmaCount += insertedDma;
}

} // namespace
