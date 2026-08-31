//===- Pipeline.h - the compiler's optimization levels ----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The `-O` levels of Section 12, described once, in C++.
//
// Section 6 settles where this lives and why, and the reason is worth keeping
// beside the code rather than only in the specification. `npu-compile` is a
// Python driver, because the import step is Python by design. The pass pipeline
// is not, because the `PassInstrumentation` of Section 16.2 has to sit on the
// `PassManager` that actually runs the passes, and a pipeline assembled in
// Python out of one `npu-opt` invocation per pass would be a different pipeline
// from the one under test. So the driver names a level and this library builds
// it.
//
// **The description is data, not a comment.** `describe()` returns the passes a
// level runs, in order, each carrying the `ablatable` property Section 12
// requires. `--npu-describe-pipeline` prints the whole table as JSON, and that
// is how the driver reads the ablatable set at run time rather than keeping a
// second copy of it that quietly stops covering a pass the day one is added.
//
// **A pass entry with no `ablatable` property is a build error**, which Section
// 12 asks for in those words. `PassEntry` has one constructor, it takes all
// three fields, and none of them has a default, so a row written as
// `{"npu-fuse-ops"}` does not compile.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_PIPELINE_PIPELINE_H
#define NPU_PIPELINE_PIPELINE_H

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::npu::pipeline {

/// The optimization levels of Section 12.
///
/// All three are named here from the phase the first one lands, because the
/// driver has to be able to say "that level arrives at P9" rather than "unknown
/// argument". `isImplemented` is what distinguishes a level this compiler can
/// build from one it can only name.
enum class OptLevel : int { O0 = 0, O1 = 1, O2 = 2 };

/// Every level, in order. The array a caller iterates rather than a range it
/// reconstructs from the enumerators.
llvm::ArrayRef<OptLevel> allOptLevels();

/// `-O0`, `-O1`, `-O2`.
llvm::StringRef optLevelName(OptLevel level);

/// The registered pipeline name for a level: `npu-O0` and so on.
llvm::StringRef optLevelPipelineName(OptLevel level);

/// The level a number names, or nothing when the number is not a level.
std::optional<OptLevel> optLevelFromNumber(int64_t number);

/// Whether this compiler can build the level, as opposed to name it.
bool isImplemented(OptLevel level);

/// The phase at which an unimplemented level arrives, for the diagnostic.
llvm::StringRef arrivingPhase(OptLevel level);

/// One pass in one level's pipeline.
///
/// The constructor takes all three fields and none of them has a default. That
/// is Section 12's rule that a pass registered with no `ablatable` property is a
/// build error rather than a silent default, expressed as something the compiler
/// enforces: a default of `false` would quietly shrink the ablation table by
/// exactly the passes somebody forgot to think about.
struct PassEntry {
  PassEntry(llvm::StringRef argument, bool ablatable, llvm::StringRef note)
      : argument(argument), ablatable(ablatable), note(note) {}

  /// The name `npu-opt` knows the pass by, without a leading dash.
  llvm::StringRef argument;
  /// Whether Section 16.2's leave one out ablation may remove it.
  bool ablatable;
  /// One line saying what it does here, or why it is not ablatable.
  llvm::StringRef note;
};

/// The passes a level runs, in the order it runs them.
llvm::ArrayRef<PassEntry> describe(OptLevel level);

/// What the caller gets to choose about a pipeline.
///
/// Everything here belongs to the allocator, which is the only pass in the
/// `-O0` set that takes options at all. The fields carry the pass's own
/// defaults so that a caller who sets nothing gets the pipeline the pass
/// description documents.
struct PipelineOptions {
  /// The scratchpad budget in bytes. Minus one means the allocator's default.
  int64_t scratchpadBudget = -1;
  std::string allocationStrategy = "pack";
  std::string spillHeuristic = "longest-range";
  int64_t allocationAlignment = 64;
};

/// Builds `level` onto `pm`.
///
/// The level must be implemented; `isImplemented` is the caller's check and
/// this function asserts it rather than inventing a pipeline for a level that
/// has none.
void build(OpPassManager &pm, OptLevel level, const PipelineOptions &options);

/// The whole table as JSON, on one line per level.
///
/// This is the channel Section 16.2 asks for: the ablatable set is read from
/// the driver at run time instead of being written down a second time in
/// Python. It prints every level, implemented or not, because a driver that
/// could not see `-O2` at all could not tell a level that is missing from one
/// that is empty.
void printDescriptionAsJson(llvm::raw_ostream &out);

/// Registers the implemented levels as `npu-opt` pass pipelines.
///
/// Called by the tool rather than done by a namespace scope initializer, so
/// that a tool which does not want the pipelines does not get them, which is
/// the same arrangement `registerNPUISAPasses` already has.
void registerNPUPipelines();

} // namespace mlir::npu::pipeline

#endif // NPU_PIPELINE_PIPELINE_H
