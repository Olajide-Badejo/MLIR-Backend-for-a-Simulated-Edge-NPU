//===- PassStats.h - the Section 16.2 instrumentation -----------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Operation counts and wall clock, per pass, from inside the pipeline that
// actually runs.
//
// **The instrumentation computes the counts. No flag does.** Section 16.2 is
// explicit about this and says so as a correction of an earlier draft that
// named `--print-op-stats`. That flag prints one summary for one invocation; it
// cannot produce a before and after pair per pass inside a running pipeline, so
// a gate written against it would be unmeetable. What produces the pair is a
// `PassInstrumentation` that walks the operation being acted on in
// `runBeforePass`, walks it again in `runAfterPass`, and records both.
//
// **The real pipeline, once.** The other way to get per pass numbers is to run
// one pass at a time and feed each invocation the previous output. That is
// quadratic in the pass count and, worse, measures a pipeline that is not the
// one under test, which contradicts Section 17.4's rule that tests drive the
// real `-O` pipelines. So this sits on the `PassManager` that `npu-compile`
// already runs and adds no invocation at all.
//
// **Where it lives follows from Section 6's driver decision.** `npu-compile` is
// a Python driver and this is C++, because a `PassInstrumentation` has to sit on
// a `PassManager` and there is no `PassManager` in Python. The driver surfaces
// the two halves: `--pass-stats-json` names the path this writes, and
// `--mlir-timing` passes through to MLIR's own timing output, which is the
// independent cross check that the two clocks agree.
//
// **It emits JSON directly**, so nothing is scraped out of human readable text.
// A missing timing or a missing op count is an error rather than a zero: the
// file records what it measured and the reader raises on a pass that is in the
// pipeline and not in the file. That reader is
// `python/npu_frontend/pass_stats.py`, because the pipeline the file is checked
// against is the one the driver asked for and the driver is the half that knows
// which level it asked for.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_PIPELINE_PASSSTATS_H
#define NPU_PIPELINE_PASSSTATS_H

#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::npu::pipeline {

/// Installs the Section 16.2 instrumentation on `pm`.
///
/// The JSON is written when `pm` is destroyed, which is after `pm.run()` has
/// returned and after the output has been printed. That is deliberate rather
/// than convenient: the instrumentation cannot know when the last pass has run,
/// because "the last pass" is a property of the pipeline and not of any
/// callback it receives, and a flush driven by a pass count would be a second
/// copy of the pipeline description.
///
/// A run that failed still writes what it recorded, with `run_completed` false.
/// The alternative, writing nothing, would leave a reader of a red run with no
/// record of which pass it got to.
void installPassStatistics(PassManager &pm, llvm::StringRef path);

} // namespace mlir::npu::pipeline

#endif // NPU_PIPELINE_PASSSTATS_H
