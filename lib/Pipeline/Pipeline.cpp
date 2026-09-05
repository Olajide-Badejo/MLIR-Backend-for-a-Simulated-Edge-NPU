//===- Pipeline.cpp - the compiler's optimization levels --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// One table per level, and one function that walks it.
//
// The tables below are the whole of Section 12 that exists at Phase P9. `-O0`
// is "import and verify" plus the two passes the table marks as running at
// every level, and that pairing is the part worth stating: **verification is not
// a pass here**. MLIR verifies every operation when it is parsed and again after
// every pass, so `-O0` gets its verification from the pass manager rather than
// from a row in this file, and adding a row that ran a verifier would be adding
// a second, weaker one beside the one that already runs.
//
// **`-O1` and `-O2` arrived here at P9** with the four `npu` level passes and
// the four upstream ones Section 12's table names, and **P13 completes the
// table** by adding `-npu-assign-layout`, `-npu-tile-to-scratchpad` and
// `-npu-double-buffer` to `-O2`. The one row still absent is `-npu-calibrate`,
// which Section 12 marks quantized mode only and never in a default `-O` level.
// The ablatable set is therefore **eleven**, which is the number Section 2's
// cell count arithmetic uses, and the three arrived in one commit because
// wiring them one at a time would have moved that arithmetic three times
// through two states nothing would ever run again.
//
// **Two passes take an option from the pipeline rather than their own default,
// and both are Section 13.2's doing.** `-npu-tile-to-scratchpad` is given the
// allocator's budget, because there is one budget on this machine and two
// passes with different ideas of how much scratchpad there is would tile
// against one number and spill against another. It is also told whether
// `-npu-double-buffer` is in this pipeline, because Section 13.2 makes the
// doubled working set the tiling search's problem: a tiling that fits only
// without the prefetch silently defeats the pass that adds it. That second one
// couples two ablation rows, which is stated where the rows are.
//
// **The order in the `-O2` table is not Section 5.1's listing order, and there
// are exactly two deviations. Both were measured rather than chosen.**
//
// *First*, Section 5.1 lists canonicalization ahead of constant folding;
// `-npu-constant-fold` runs before it here, because the folder is what *creates*
// the dead operand constants and a canonicalization that ran before it would
// have nothing to clean up. At `-O2` there are two canonicalizations and the
// question would be academic; at `-O1` there is one, and if it ran first every
// folded operand would survive as an `npuisa.const` and a `dma_load` in the
// instruction stream.
//
// *Second*, Section 5.1 lists the batch norm fold ahead of the bias fusion, and
// they are the other way round here. `-npu-fold-batchnorm` matches on a
// convolution as the batch norm's producer, and in
// `conv -> add(bias) -> batch_norm` the producer is the add until
// `-npu-fuse-bias` has moved it. Folding first leaves *both* passes declining
// on a shape both would otherwise handle, and then `-npu-fuse-ops` declines too
// because the activation's producer is a batch norm rather than a convolution:
// one ordering choice turning three passes off. `test/Pipeline/opt-levels.mlir`
// carries the function that shows it, and it went red the first time this table
// was written in the listing order.
//
// Section 12's own note on `-npu-fold-batchnorm` says "before fusion, so the
// convolution still has no fused **activation**", and that is `-npu-fuse-ops`,
// which still runs after it. Everything else is Section 12's order exactly,
// including the duplicate canonicalization around fusion, which that table
// marks as deliberate.
//
// **One table, two consumers, and no second copy.** `build()` switches on
// `PassEntry::kind` and the switch has no `default`, so a pass added to a level
// and not to the builder is a build error. The description JSON is generated
// from the same rows. `test/Pipeline/opt-levels.mlir` runs each level against
// the explicit list of pass arguments and diffs the two outputs, which is what
// catches a `kind` and an `argument` that disagree.
//
//===----------------------------------------------------------------------===//

#include "NPU/Pipeline/Pipeline.h"

#include "NPU/Dialect/NPU/Transforms/Passes.h"
#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/JSON.h"

#include <cassert>

using namespace mlir;
using namespace mlir::npu::pipeline;

namespace {

//===----------------------------------------------------------------------===//
// The tables.
//===----------------------------------------------------------------------===//

/// The two passes Section 12's table marks as running at every level and as
/// **not ablatable**, in the words that table uses: removing either produces no
/// program at all, so an ablation of one of them measures nothing and fails the
/// run for a reason that has nothing to do with the pass.
///
/// Written out per level rather than shared, because a level's row list is the
/// thing a reader compares against Section 12's table, and a table that ended
/// in "and then the usual two" would be a table they had to assemble first.

/// `-O0`. Import and verify, then lowering and allocation.
const PassEntry kO0[] = {
    PassEntry(PassKind::NPULowerToNPUISA, "npu-lower-to-npuisa",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "the dialect conversion to npuisa on memrefs; removing it "
              "produces no program at all"),
    PassEntry(PassKind::NPUAllocateScratchpad, "npu-allocate-scratchpad",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "assigns every scratchpad buffer an offset; removing it produces "
              "no program at all"),
};

/// `-O1`. Constant folding and canonicalization, on top of `-O0`.
const PassEntry kO1[] = {
    PassEntry(PassKind::NPUConstantFold, "npu-constant-fold",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "evaluates elementwise npu operations over constant operands; "
              "exact, because none of them has a reduction in it"),
    PassEntry(PassKind::Canonicalize, "canonicalize", /*ablatable=*/true,
              /*eliminatesDeadCode=*/true,
              "removes the operands the folder left dead, and every other npu "
              "operation nothing reads, because they all carry Pure"),
    PassEntry(PassKind::NPULowerToNPUISA, "npu-lower-to-npuisa",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "the dialect conversion to npuisa on memrefs; removing it "
              "produces no program at all"),
    PassEntry(PassKind::NPUAllocateScratchpad, "npu-allocate-scratchpad",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "assigns every scratchpad buffer an offset; removing it produces "
              "no program at all"),
};

/// `-O2`. Section 12's whole table except the four rows P9 excludes by name.
const PassEntry kO2[] = {
    PassEntry(PassKind::NPUConstantFold, "npu-constant-fold",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "evaluates elementwise npu operations over constant operands; "
              "exact, because none of them has a reduction in it"),
    PassEntry(PassKind::Canonicalize, "canonicalize", /*ablatable=*/true,
              /*eliminatesDeadCode=*/true,
              "the first of the two Section 12 puts around fusion; removes the "
              "operands the folder left dead"),
    PassEntry(PassKind::NPUFuseBias, "npu-fuse-bias", /*ablatable=*/true,
              /*eliminatesDeadCode=*/false,
              "before the batch norm fold, because a separate bias add sits "
              "between a convolution and its batch norm until this has run; "
              "exact"),
    PassEntry(PassKind::NPUFoldBatchNorm, "npu-fold-batchnorm",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "before op fusion, so the convolution it rewrites still has no "
              "fused activation; the one pass at this phase that moves numbers"),
    PassEntry(PassKind::NPUFuseOps, "npu-fuse-ops", /*ablatable=*/true,
              /*eliminatesDeadCode=*/false,
              "forms npu.fused_op regions; numerically inert, because the "
              "lowering flattens them back into the same instructions"),
    PassEntry(PassKind::Canonicalize, "canonicalize", /*ablatable=*/true,
              /*eliminatesDeadCode=*/true,
              "the second of the two, after fusion, to remove the dead "
              "parameter constants the batch norm fold left behind"),
    PassEntry(PassKind::CSE, "cse", /*ablatable=*/true,
              /*eliminatesDeadCode=*/false,
              "merges identical operations over identical operands"),
    PassEntry(PassKind::SCCP, "sccp", /*ablatable=*/true,
              /*eliminatesDeadCode=*/false,
              "propagates constants through the call graph; needs the "
              "dialect's constant materialiser, which is D-0033"),
    PassEntry(PassKind::SymbolDCE, "symbol-dce", /*ablatable=*/true,
              /*eliminatesDeadCode=*/true,
              "removes a private symbol nobody calls"),
    PassEntry(PassKind::NPUAssignLayout, "npu-assign-layout",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "scores the rank 4 layout question on Section 5.5 and cancels "
              "the permutations the answer makes redundant; exact, and the "
              "answer is NCHW at every extent on this machine"),
    PassEntry(PassKind::NPUTileToScratchpad, "npu-tile-to-scratchpad",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "splits an operation whose working set exceeds the budget over "
              "its parallel dimensions; exact, because no reduction is split"),
    PassEntry(PassKind::NPULowerToNPUISA, "npu-lower-to-npuisa",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "the dialect conversion to npuisa on memrefs; removing it "
              "produces no program at all"),
    PassEntry(PassKind::NPUDoubleBuffer, "npu-double-buffer",
              /*ablatable=*/true, /*eliminatesDeadCode=*/false,
              "overlaps a transfer with the computation before it; before "
              "allocation, since the doubled working set has to be visible to "
              "the allocator"),
    PassEntry(PassKind::NPUAllocateScratchpad, "npu-allocate-scratchpad",
              /*ablatable=*/false, /*eliminatesDeadCode=*/false,
              "assigns every scratchpad buffer an offset; removing it produces "
              "no program at all"),
};

struct LevelInfo {
  OptLevel level;
  llvm::StringRef name;
  llvm::StringRef pipelineName;
  llvm::ArrayRef<PassEntry> passes;
  bool implemented;
  /// The phase at which an unimplemented level arrives. Empty when it is here.
  llvm::StringRef phase;
  llvm::StringRef summary;
};

const LevelInfo kLevels[] = {
    {OptLevel::O0, "-O0", "npu-O0", llvm::ArrayRef<PassEntry>(kO0), true, "",
     "import and verify, then lowering and allocation"},
    {OptLevel::O1, "-O1", "npu-O1", llvm::ArrayRef<PassEntry>(kO1), true, "",
     "constant folding and canonicalization, on top of -O0"},
    {OptLevel::O2, "-O2", "npu-O2", llvm::ArrayRef<PassEntry>(kO2), true, "",
     "batch norm folding, bias fusion, operation fusion, CSE, SCCP, symbol "
     "DCE, layout assignment, tiling and double buffering, on top of -O1"},
};

const LevelInfo &infoFor(OptLevel level) {
  for (const LevelInfo &info : kLevels)
    if (info.level == level)
      return info;
  llvm_unreachable("every OptLevel enumerator has a row in kLevels");
}

const OptLevel kAllLevels[] = {OptLevel::O0, OptLevel::O1, OptLevel::O2};

//===----------------------------------------------------------------------===//
// The pipeline options `npu-opt` parses.
//===----------------------------------------------------------------------===//

/// The command line form of `PipelineOptions`.
///
/// It is a separate type from `PipelineOptions` on purpose. This one exists to
/// be parsed out of a string by MLIR's option machinery; the other one exists to
/// be passed around in C++ and is what the tests and the driver construct
/// directly. Collapsing them would put an `llvm::cl` dependency into every
/// caller that only wanted to build a pipeline.
struct PipelineCLOptions : public PassPipelineOptions<PipelineCLOptions> {
  Option<int64_t> budget{
      *this, "budget",
      llvm::cl::desc("The scratchpad budget in bytes. Minus one means the "
                     "allocator's own default."),
      llvm::cl::init(-1)};
  Option<std::string> strategy{
      *this, "strategy",
      llvm::cl::desc("The allocator's offset assignment strategy."),
      llvm::cl::init("pack")};
  Option<std::string> spillHeuristic{
      *this, "spill-heuristic",
      llvm::cl::desc("Which buffer the allocator spills."),
      llvm::cl::init("longest-range")};
  Option<int64_t> alignment{
      *this, "alignment",
      llvm::cl::desc("The byte alignment of every assigned offset."),
      llvm::cl::init(64)};
  Option<std::string> ablate{
      *this, "ablate",
      llvm::cl::desc("Leave this pass out, by its argument, for Section 16.2's "
                     "leave one out ablation. Only a pass the table marks "
                     "ablatable is removed; naming another is ignored here and "
                     "caught by the pass statistics, which record what actually "
                     "ran."),
      llvm::cl::init("")};
  Option<std::string> stopAfter{
      *this, "stop-after",
      llvm::cl::desc("Where to stop: 'npuisa', the whole level, or 'npu', the "
                     "tensor level half of it. The second is what "
                     "npu-compile --emit npu runs, and it runs it through this "
                     "pipeline rather than through a pass list assembled in "
                     "Python, which Section 17.4 says would enforce nothing."),
      llvm::cl::init("npuisa")};

  PipelineOptions toPipelineOptions() const {
    PipelineOptions options;
    options.scratchpadBudget = budget;
    options.allocationStrategy = strategy;
    options.spillHeuristic = spillHeuristic;
    options.allocationAlignment = alignment;
    options.stopAfter = stopAfter == "npu" ? PipelineStage::Npu
                                           : PipelineStage::NpuIsa;
    options.ablatedPass = ablate;
    return options;
  }
};

/// Adds one table row to `pm`.
///
/// The switch has no `default`, so a `PassKind` added to the enumerator and not
/// handled here is a `-Werror=switch` build error rather than a pipeline that
/// silently skips a pass. That is the same mechanism the simulator's opcode
/// dispatch uses and it is here for the same reason.
///
/// The nesting is per pass and not per level. `-sccp` and `-symbol-dce` reason
/// about the module: one is interprocedural and the other is about symbols, and
/// nesting either inside a function would give it a view in which every
/// question it asks has the wrong answer.
void addPass(OpPassManager &pm, const PassEntry &entry,
             const PipelineOptions &options, bool doubleBufferInPipeline) {
  switch (entry.kind) {
  case PassKind::Canonicalize:
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    return;
  case PassKind::CSE:
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    return;
  case PassKind::SCCP:
    pm.addPass(createSCCPPass());
    return;
  case PassKind::SymbolDCE:
    pm.addPass(createSymbolDCEPass());
    return;
  case PassKind::NPUConstantFold:
    pm.addNestedPass<func::FuncOp>(npu::createNPUConstantFold());
    return;
  case PassKind::NPUFoldBatchNorm:
    pm.addNestedPass<func::FuncOp>(npu::createNPUFoldBatchNorm());
    return;
  case PassKind::NPUFuseBias:
    pm.addNestedPass<func::FuncOp>(npu::createNPUFuseBias());
    return;
  case PassKind::NPUFuseOps:
    pm.addNestedPass<func::FuncOp>(npu::createNPUFuseOps());
    return;
  case PassKind::NPUAssignLayout:
    pm.addNestedPass<func::FuncOp>(npu::createNPUAssignLayout());
    return;
  case PassKind::NPUTileToScratchpad: {
    // **The tiling pass is told the allocator's budget, and that is not a
    // convenience.** Section 13.2 has the pass split an operation whose working
    // set exceeds *the budget*, and there is exactly one budget on this
    // machine: two passes with different ideas of how much scratchpad there is
    // would tile against one number and spill against another. A budget of
    // minus one means the allocator's own default, which is the tiling pass's
    // default too, so leaving both alone still leaves them agreeing.
    npu::NPUTileToScratchpadOptions tiling;
    if (options.scratchpadBudget >= 0)
      tiling.budget = options.scratchpadBudget;
    // **And it is told whether double buffering is in this pipeline.**
    // Section 13.2 makes the doubled working set the search's problem rather
    // than the allocator's, because a tiling that fits only without the
    // prefetch silently defeats the pass that adds it. That couples the two:
    // ablating `-npu-double-buffer` also relaxes the tiling search, so its
    // ablation row measures the pass together with the sizing it forces. The
    // coupling is real rather than an artifact of this wiring, it is what
    // Section 13.2 asks for, and `docs/PASSES.md` says so beside the row.
    tiling.doubleBuffer = doubleBufferInPipeline;
    pm.addNestedPass<func::FuncOp>(npu::createNPUTileToScratchpad(tiling));
    return;
  }
  case PassKind::NPULowerToNPUISA:
    pm.addPass(npuisa::createNPULowerToNPUISA());
    return;
  case PassKind::NPUDoubleBuffer:
    pm.addNestedPass<func::FuncOp>(npuisa::createNPUDoubleBuffer());
    return;
  case PassKind::NPUAllocateScratchpad: {
    npuisa::NPUAllocateScratchpadOptions allocation;
    allocation.budget = options.scratchpadBudget;
    allocation.strategy = options.allocationStrategy;
    allocation.spillHeuristic = options.spillHeuristic;
    allocation.alignment = options.allocationAlignment;
    pm.addNestedPass<func::FuncOp>(
        npuisa::createNPUAllocateScratchpad(allocation));
    return;
  }
  }
  llvm_unreachable("unhandled pass kind");
}

} // namespace

//===----------------------------------------------------------------------===//
// The queries.
//===----------------------------------------------------------------------===//

bool mlir::npu::pipeline::isTensorLevel(PassKind kind) {
  switch (kind) {
  case PassKind::Canonicalize:
  case PassKind::CSE:
  case PassKind::SCCP:
  case PassKind::SymbolDCE:
  case PassKind::NPUConstantFold:
  case PassKind::NPUFoldBatchNorm:
  case PassKind::NPUFuseBias:
  case PassKind::NPUFuseOps:
  case PassKind::NPUAssignLayout:
  case PassKind::NPUTileToScratchpad:
    return true;
  case PassKind::NPULowerToNPUISA:
  // `-npu-double-buffer` rewrites the asynchronous transfer tokens, and those
  // exist only below the tensor level, so it is on the far side of the
  // conversion even though Section 12 lists it before the allocator.
  case PassKind::NPUDoubleBuffer:
  case PassKind::NPUAllocateScratchpad:
    return false;
  }
  llvm_unreachable("unhandled pass kind");
}

llvm::ArrayRef<OptLevel> mlir::npu::pipeline::allOptLevels() {
  return llvm::ArrayRef<OptLevel>(kAllLevels);
}

llvm::StringRef mlir::npu::pipeline::optLevelName(OptLevel level) {
  return infoFor(level).name;
}

llvm::StringRef mlir::npu::pipeline::optLevelPipelineName(OptLevel level) {
  return infoFor(level).pipelineName;
}

std::optional<OptLevel> mlir::npu::pipeline::optLevelFromNumber(int64_t number) {
  for (const LevelInfo &info : kLevels)
    if (static_cast<int64_t>(info.level) == number)
      return info.level;
  return std::nullopt;
}

bool mlir::npu::pipeline::isImplemented(OptLevel level) {
  return infoFor(level).implemented;
}

llvm::StringRef mlir::npu::pipeline::arrivingPhase(OptLevel level) {
  return infoFor(level).phase;
}

llvm::ArrayRef<PassEntry> mlir::npu::pipeline::describe(OptLevel level) {
  return infoFor(level).passes;
}

//===----------------------------------------------------------------------===//
// Building.
//===----------------------------------------------------------------------===//

void mlir::npu::pipeline::build(OpPassManager &pm, OptLevel level,
                                const PipelineOptions &options) {
  assert(isImplemented(level) &&
         "build() was asked for a level this compiler cannot build; the caller "
         "checks isImplemented and says which phase it arrives at");

  // Whether this level runs `-npu-double-buffer`, which the tiling search has
  // to know before it chooses a tile: Section 13.2 sizes the working set for
  // the prefetch. It is read from the same rows the loop below walks, with the
  // **ablation** filter applied, so an ablation row measures the pass together
  // with the sizing it forces and the answer cannot drift from what is built.
  //
  // **The stage filter is deliberately not applied**, and that asymmetry is the
  // point. `--stop-after npu` is `npu-compile --emit npu`, which is a view of
  // the tensor level half of *this* compilation rather than a different one. If
  // stopping early relaxed the tiling search, the IR a reader was shown would
  // be tiled differently from the IR that gets compiled, and the two would
  // disagree without either being wrong on its own terms. The tensor level
  // output stays a prefix of the whole pipeline's, which is what makes it worth
  // printing.
  bool doubleBufferInPipeline = false;
  for (const PassEntry &entry : infoFor(level).passes) {
    if (entry.kind != PassKind::NPUDoubleBuffer)
      continue;
    if (entry.ablatable && !options.ablatedPass.empty() &&
        entry.argument == options.ablatedPass)
      continue;
    doubleBufferInPipeline = true;
  }

  for (const PassEntry &entry : infoFor(level).passes) {
    if (options.stopAfter == PipelineStage::Npu && !isTensorLevel(entry.kind))
      continue;
    // Section 16.2's leave one out ablation. `-canonicalize` has two entries at
    // `-O2` and this removes **both**, which is the right reading and not an
    // accident of matching on the argument: an ablation removes the pass, and a
    // row that removed one of two positions would be measuring an ordering
    // change rather than the absence of canonicalization.
    if (entry.ablatable && !options.ablatedPass.empty() &&
        entry.argument == options.ablatedPass)
      continue;
    addPass(pm, entry, options, doubleBufferInPipeline);
  }
}

//===----------------------------------------------------------------------===//
// The description.
//===----------------------------------------------------------------------===//

void mlir::npu::pipeline::printDescriptionAsJson(llvm::raw_ostream &out) {
  llvm::json::Array levels;
  for (const LevelInfo &info : kLevels) {
    llvm::json::Array passes;
    for (const PassEntry &entry : info.passes) {
      passes.push_back(llvm::json::Object{
          {"pass", entry.argument},
          {"ablatable", entry.ablatable},
          {"eliminates_dead_code", entry.eliminatesDeadCode},
          {"stage", isTensorLevel(entry.kind) ? "npu" : "npuisa"},
          {"note", entry.note},
      });
    }
    levels.push_back(llvm::json::Object{
        {"level", static_cast<int64_t>(info.level)},
        {"name", info.name},
        {"pipeline", info.pipelineName},
        {"implemented", info.implemented},
        {"arrives_at", info.phase},
        {"summary", info.summary},
        {"passes", std::move(passes)},
    });
  }

  llvm::json::Object root{
      {"generated_by", "npu-opt --npu-describe-pipeline"},
      {"source", "lib/Pipeline/Pipeline.cpp"},
      {"levels", std::move(levels)},
  };
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
}

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void mlir::npu::pipeline::registerNPUPipelines() {
  // One registration per implemented level, written out rather than looped,
  // because `PassPipelineRegistration` is a constructor with a side effect and
  // a list of them reads as a list of what exists.
  //
  // The objects outlive the call deliberately: the registry keeps the callback
  // and the process keeps the registry. This is called once from a tool's main.
  static PassPipelineRegistration<PipelineCLOptions> registerO0(
      "npu-O0",
      "The -O0 pipeline of Section 12: import and verify, then lowering and "
      "scratchpad allocation.",
      [](OpPassManager &pm, const PipelineCLOptions &options) {
        build(pm, OptLevel::O0, options.toPipelineOptions());
      });
  static PassPipelineRegistration<PipelineCLOptions> registerO1(
      "npu-O1",
      "The -O1 pipeline of Section 12: constant folding and canonicalization, "
      "then lowering and scratchpad allocation.",
      [](OpPassManager &pm, const PipelineCLOptions &options) {
        build(pm, OptLevel::O1, options.toPipelineOptions());
      });
  static PassPipelineRegistration<PipelineCLOptions> registerO2(
      "npu-O2",
      "The -O2 pipeline of Section 12: batch norm folding, bias fusion, "
      "operation fusion, CSE, SCCP, symbol DCE, layout assignment, tiling and "
      "double buffering on top of -O1.",
      [](OpPassManager &pm, const PipelineCLOptions &options) {
        build(pm, OptLevel::O2, options.toPipelineOptions());
      });
  (void)registerO0;
  (void)registerO1;
  (void)registerO2;
}
