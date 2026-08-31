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
// The tables below are the whole of Section 12 that exists at Phase P8. `-O0`
// is "import and verify" plus the two passes the table marks as running at
// every level, and that pairing is the part worth stating: **verification is not
// a pass here**. MLIR verifies every operation when it is parsed and again after
// every pass, so `-O0` gets its verification from the pass manager rather than
// from a row in this file, and adding a row that ran a verifier would be adding
// a second, weaker one beside the one that already runs.
//
// `-O1` and `-O2` are named and not built. Their rows arrive at P9 with the
// passes that fill them, and until then `isImplemented` says so and the driver
// refuses them by name. A level registered with an empty pipeline would be
// worse than one that does not exist: `-O2` would run, produce `-O0`'s answer,
// and every ablation cell measured against it would be measuring nothing.
//
//===----------------------------------------------------------------------===//

#include "NPU/Pipeline/Pipeline.h"

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/Support/JSON.h"

#include <cassert>

using namespace mlir;
using namespace mlir::npu::pipeline;

namespace {

//===----------------------------------------------------------------------===//
// The tables.
//===----------------------------------------------------------------------===//

/// `-O0`. Import and verify, then the two passes Section 12's table marks as
/// running at every level and as **not ablatable**.
///
/// Neither is ablatable and the note says why in the words Section 12 uses:
/// removing either produces no program at all, so an ablation of one of them
/// measures nothing and fails the run for a reason that has nothing to do with
/// the pass.
const PassEntry kO0[] = {
    PassEntry("npu-lower-to-npuisa", /*ablatable=*/false,
              /*eliminatesDeadCode=*/false,
              "the dialect conversion to npuisa on memrefs; removing it "
              "produces no program at all"),
    PassEntry("npu-allocate-scratchpad", /*ablatable=*/false,
              /*eliminatesDeadCode=*/false,
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
    {OptLevel::O1, "-O1", "npu-O1", {}, false, "P9",
     "canonicalize and constant fold, on top of -O0"},
    {OptLevel::O2, "-O2", "npu-O2", {}, false, "P9",
     "the full set: batch norm folding, fusion, bias fusion, CSE, SCCP, symbol "
     "DCE, layout assignment, tiling and double buffering"},
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

  PipelineOptions toPipelineOptions() const {
    PipelineOptions options;
    options.scratchpadBudget = budget;
    options.allocationStrategy = strategy;
    options.spillHeuristic = spillHeuristic;
    options.allocationAlignment = alignment;
    return options;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The queries.
//===----------------------------------------------------------------------===//

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

  // The switch has no `default`, so a level added to the enumerator and not to
  // this function is a `-Werror=switch` build error rather than a pipeline that
  // silently builds nothing. That is the same mechanism the simulator's opcode
  // dispatch uses and it is here for the same reason.
  switch (level) {
  case OptLevel::O0: {
    pm.addPass(npuisa::createNPULowerToNPUISA());

    npuisa::NPUAllocateScratchpadOptions allocation;
    allocation.budget = options.scratchpadBudget;
    allocation.strategy = options.allocationStrategy;
    allocation.spillHeuristic = options.spillHeuristic;
    allocation.alignment = options.allocationAlignment;
    pm.addNestedPass<func::FuncOp>(
        npuisa::createNPUAllocateScratchpad(allocation));
    return;
  }
  case OptLevel::O1:
  case OptLevel::O2:
    // Unreachable through the assertion above. Written out rather than folded
    // into a default so that P9 gets a build error here when it adds the rows
    // and forgets the builder.
    llvm_unreachable("-O1 and -O2 are named and not implemented until P9");
  }
  llvm_unreachable("unhandled optimization level");
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
  // a list of them reads as a list of what exists. P9 adds two lines here in
  // the same commit that fills their tables.
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
  (void)registerO0;
}
