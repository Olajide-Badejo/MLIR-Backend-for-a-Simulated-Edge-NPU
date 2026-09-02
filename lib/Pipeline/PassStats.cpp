//===- PassStats.cpp - the Section 16.2 instrumentation ---------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Three decisions in here are not obvious and each is stated where it is made.
//
// **Adaptors are filtered out on `getArgument()` being empty**, not on a type
// test. MLIR instruments the `OpToOpPassAdaptor` that wraps a nested pipeline
// exactly the way it instruments a real pass, so without a filter every nested
// run would be recorded twice: once as the adaptor and once as the pass inside
// it. The type is only declared in `mlir/lib/Pass/PassDetail.h`, which an out of
// tree project does not get, so `isa<OpToOpPassAdaptor>` is not available here.
// What is available is that an adaptor is not registered and therefore has no
// command line argument, while every pass in `lib/Pipeline/Pipeline.cpp` has
// one and is keyed on it in that table. Filtering on the argument is the same
// question asked through the field the pipeline description already uses, which
// makes the two agree by construction rather than by inspection.
//
// **The op walk is outside the timed span.** `runBeforePass` walks first and
// starts the clock afterwards; `runAfterPass` stops the clock first and walks
// afterwards. Counting is a full traversal of the operation, which on the larger
// models costs as much as a cheap pass does, and a wall clock that included it
// would be reporting the instrumentation rather than the pass. MLIR's own
// `--mlir-timing` cannot do this, because its timer is started and stopped
// around every instrumentation including this one, which is why the cross check
// between the two has a direction: MLIR's figure is always the larger, and the
// difference is this file's own cost.
//
// **Every invocation is a row.** A pass that runs on two operations produces two
// rows rather than one summed row. This project's modules hold one function, so
// the sequence is the pipeline's sequence, and the reader in
// `python/npu_frontend/pass_stats.py` compares it against the level's
// description and raises when they differ. Summing here would turn a module
// that grew a second function into a file that still looked right.
//
//===----------------------------------------------------------------------===//

#include "NPU/Pipeline/PassStats.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace {

using Clock = std::chrono::steady_clock;

/// What one pass invocation cost and what it did to the operation count.
struct Record {
  int64_t position = 0;
  std::string argument;
  std::string passName;
  std::string anchor;
  llvm::StringMap<int64_t> opsBefore;
  llvm::StringMap<int64_t> opsAfter;
  int64_t opsBeforeTotal = 0;
  int64_t opsAfterTotal = 0;
  double wallMs = 0.0;
};

/// What `runBeforePass` recorded and `runAfterPass` completes.
struct Open {
  llvm::StringMap<int64_t> opsBefore;
  int64_t opsBeforeTotal = 0;
  std::string argument;
  std::string passName;
  std::string anchor;
  Clock::time_point started;
};

llvm::json::Object countsAsJson(const llvm::StringMap<int64_t> &counts) {
  // Sorted, because the file is compared for byte equality by the determinism
  // test of Section 16.1 and a `StringMap` iterates in hash order.
  std::vector<llvm::StringRef> names;
  names.reserve(counts.size());
  for (const auto &entry : counts)
    names.push_back(entry.getKey());
  llvm::sort(names);

  llvm::json::Object object;
  for (llvm::StringRef name : names)
    object[name] = counts.lookup(name);
  return object;
}

class PassStatsInstrumentation : public PassInstrumentation {
public:
  explicit PassStatsInstrumentation(std::string path)
      : path(std::move(path)) {}

  ~PassStatsInstrumentation() override { writeFile(); }

  void runBeforePass(Pass *pass, Operation *op) override {
    if (pass->getArgument().empty())
      return;

    Open opened;
    opened.argument = pass->getArgument().str();
    opened.passName = pass->getName().str();
    opened.anchor = op->getName().getStringRef().str();
    opened.opsBeforeTotal = countInto(op, opened.opsBefore);
    opened.started = Clock::now();

    llvm::sys::SmartScopedLock<true> guard(mutex);
    active[{pass, op}] = std::move(opened);
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    Clock::time_point finished = Clock::now();
    if (pass->getArgument().empty())
      return;

    Open opened;
    {
      llvm::sys::SmartScopedLock<true> guard(mutex);
      auto found = active.find({pass, op});
      if (found == active.end()) {
        // runAfterPass without a runBeforePass is not something the pass
        // manager does, so this is unreachable rather than tolerated. Recording
        // it as a note keeps the file honest instead of dropping a pass and
        // leaving the reader to find a gap.
        unmatched.push_back(pass->getArgument().str());
        return;
      }
      opened = std::move(found->second);
      active.erase(found);
    }

    Record record;
    record.argument = std::move(opened.argument);
    record.passName = std::move(opened.passName);
    record.anchor = std::move(opened.anchor);
    record.opsBefore = std::move(opened.opsBefore);
    record.opsBeforeTotal = opened.opsBeforeTotal;
    record.opsAfterTotal = countInto(op, record.opsAfter);
    record.wallMs =
        std::chrono::duration<double, std::milli>(finished - opened.started)
            .count();

    llvm::sys::SmartScopedLock<true> guard(mutex);
    record.position = static_cast<int64_t>(records.size());
    records.push_back(std::move(record));
  }

  void runAfterPassFailed(Pass *pass, Operation *op) override {
    {
      llvm::sys::SmartScopedLock<true> guard(mutex);
      completed = false;
      failedPass = pass->getArgument().empty() ? pass->getName().str()
                                               : pass->getArgument().str();
    }
    // The record is still wanted: a reader of a red run needs to know which
    // pass the pipeline reached. The op counts after a failure describe an
    // operation the pass may have left half rewritten, which is why the file
    // says `run_completed` false rather than leaving the reader to infer it.
    runAfterPass(pass, op);
  }

private:
  /// Fills `counts` with a per operation name census of `op` and everything
  /// nested inside it, and returns the total.
  ///
  /// `Operation::walk` visits `op` itself, so the anchor is counted. That is
  /// the honest reading of Section 16.2's "walks the operation being acted on":
  /// the operation the pass was handed is part of what the pass may rewrite.
  static int64_t countInto(Operation *op, llvm::StringMap<int64_t> &counts) {
    int64_t total = 0;
    op->walk([&](Operation *nested) {
      ++counts[nested->getName().getStringRef()];
      ++total;
    });
    return total;
  }

  void writeFile() {
    llvm::json::Array passes;
    double totalMs = 0.0;
    for (const Record &record : records) {
      totalMs += record.wallMs;
      passes.push_back(llvm::json::Object{
          {"position", record.position},
          {"name", record.argument},
          {"pass_name", record.passName},
          {"anchor_op", record.anchor},
          {"ops_before", countsAsJson(record.opsBefore)},
          {"ops_after", countsAsJson(record.opsAfter)},
          {"ops_before_total", record.opsBeforeTotal},
          {"ops_after_total", record.opsAfterTotal},
          {"wall_ms", record.wallMs},
          {"pass_timing_source",
           "measured: PassInstrumentation runBeforePass and runAfterPass, "
           "std::chrono::steady_clock, with the operation walk outside the "
           "timed span"},
      });
    }

    llvm::json::Array left;
    for (const auto &entry : active)
      left.push_back(entry.second.argument);
    llvm::json::Array unmatchedNames;
    for (const std::string &name : unmatched)
      unmatchedNames.push_back(name);

    llvm::json::Object root{
        {"generated_by", "lib/Pipeline/PassStats.cpp"},
        {"source", "PassInstrumentation::runBeforePass and runAfterPass"},
        {"clock", "std::chrono::steady_clock"},
        {"run_completed", completed},
        {"failed_pass", failedPass.empty() ? llvm::json::Value(nullptr)
                                           : llvm::json::Value(failedPass)},
        {"passes_total_wall_ms", totalMs},
        {"passes", std::move(passes)},
        // Both of these are empty on every run this project has ever recorded.
        // They are written anyway, because a diagnostic that only appears when
        // something is wrong is a diagnostic nobody has seen work.
        {"still_running_at_exit", std::move(left)},
        {"unmatched_after_pass", std::move(unmatchedNames)},
    };

    // Atomically, through a `.tmp` rename, which is the rule Section 16.1 sets
    // for a result file and applies just as well to the input to one: a reader
    // that opened this path must see either the previous file or a whole new
    // one, never a partial write from a run that was interrupted.
    std::string temporary = path + ".tmp";
    std::error_code failure;
    {
      llvm::raw_fd_ostream out(temporary, failure, llvm::sys::fs::OF_Text);
      if (failure) {
        llvm::errs() << "npu pass statistics: cannot write " << temporary
                     << ": " << failure.message() << "\n";
        return;
      }
      out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
    }
    if (std::error_code renamed = llvm::sys::fs::rename(temporary, path)) {
      llvm::errs() << "npu pass statistics: cannot rename " << temporary
                   << " onto " << path << ": " << renamed.message() << "\n";
    }
  }

  std::string path;
  llvm::sys::SmartMutex<true> mutex;
  llvm::DenseMap<std::pair<Pass *, Operation *>, Open> active;
  std::vector<Record> records;
  std::vector<std::string> unmatched;
  std::string failedPass;
  bool completed = true;
};

} // namespace

void mlir::npu::pipeline::installPassStatistics(PassManager &pm,
                                                llvm::StringRef path) {
  pm.addInstrumentation(
      std::make_unique<PassStatsInstrumentation>(path.str()));
}
