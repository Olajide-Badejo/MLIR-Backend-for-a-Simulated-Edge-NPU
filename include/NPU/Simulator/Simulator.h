//===- Simulator.h - executing a .nbin ------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 10: the simulator, its statistics, and the two timelines it keeps.
//
// **The statistics are the product.** A cycle count nobody can reconstruct is a
// number, not a measurement, so `Stats` carries the raw MAC count beside the
// occupancy terms that turned it into cycles, and the two timeline totals
// beside the total they produced. Section 16.1 consumes all of it and the
// energy path of Phase P11 consumes `macs` raw, never scaled.
//
// **`stats.instructions` is the only instruction count anywhere in this
// project.** A regex over an IR dump is not an instruction count: it matches
// inside type strings and counts constants that the encoder treats as data. The
// benchmark harness raises if this field is missing rather than falling back.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_SIMULATOR_SIMULATOR_H
#define NPU_SIMULATOR_SIMULATOR_H

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/CostModel.h"
#include "NPU/Simulator/Memory.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace nbin {

//===----------------------------------------------------------------------===//
// What this build of the kernels can do.
//===----------------------------------------------------------------------===//

/// Whether `lib/Simulator/Kernels.cpp` was compiled with OpenMP.
///
/// **This asks the kernels, not the caller.** It is defined in `Kernels.cpp`
/// and returns that translation unit's `_OPENMP`, so a caller cannot answer it
/// from its own preprocessor state. That distinction is the whole reason the
/// function exists, and D-0047 is why it is not paranoia: between P7 and P12
/// the OpenMP usage requirement was attached to `NPUSimulator` and not to the
/// object library its sources compile in, so every consumer of this header was
/// compiled with `-fopenmp` and the kernels were not. `omp_get_max_threads()`
/// answered 28 in a test whose kernel had no parallel region in it, and the
/// bitwise determinism assertion of Section 10.3 was comparing two serial runs.
///
/// A caller that has OpenMP itself and gets `false` from this is looking at
/// that fault, and `unittests/Simulator/DeterminismTest.cpp` fails on exactly
/// that comparison so it can never be silent again.
bool kernelsUseOpenMP();

/// The thread count the convolution kernel would use for its parallel region,
/// read inside `Kernels.cpp`, or 1 when this build has no OpenMP.
///
/// Reported rather than asserted. It is a property of the host and of whatever
/// `OMP_NUM_THREADS` says, so nothing gates on its value; it is here so that a
/// test and a tool can print what the run actually had rather than what the
/// machine could have offered.
int kernelThreadCount();

//===----------------------------------------------------------------------===//
// Statistics.
//===----------------------------------------------------------------------===//

/// What a run measured. Section 10.2's list, plus the four fields Section 5.5
/// requires beside `macs` so that a reader can reconstruct the cycle charge.
struct Stats {
  /// Instructions **executed**, `HALT` included.
  ///
  /// Executed rather than present: a program that runs off the end of its
  /// instruction stream has executed fewer instructions than it holds, and the
  /// count that matters for a cycles per instruction figure is the one that
  /// ran.
  uint64_t instructions = 0;

  /// The total, which is the later of the two timelines at `HALT`.
  ///
  /// It is a `double` and that is deliberate. The charges of Section 5.5 are
  /// analytic and fractional, and rounding each instruction to an integer would
  /// accumulate a bias of half a cycle per instruction in a fixed direction,
  /// which on a program of ten thousand instructions is a five thousand cycle
  /// error introduced by the reporting rather than by the model.
  double cycles = 0.0;

  uint64_t dramBytesRead = 0;
  uint64_t dramBytesWritten = 0;

  /// Elements read from and written to the scratchpad.
  ///
  /// Counted as the logical size of each buffer an instruction names: the
  /// element count of every scratchpad operand it reads, and the element count
  /// of its result when the result lives in the scratchpad. It is the traffic
  /// at the scratchpad port rather than the number of times a kernel's inner
  /// loop touched a word, and the two differ for the windowed kernels, where
  /// one input element is read by several windows. The definition is stated
  /// here because it is exactly the kind of convention two tools disagree about
  /// silently.
  uint64_t scratchpadElementsRead = 0;
  uint64_t scratchpadElementsWritten = 0;

  /// Multiply accumulate operations, **raw**.
  ///
  /// Counted explicitly where the cost is charged, never inferred from a cycle
  /// figure, because Accelergy consumes action counts and a derived count is a
  /// fiction. Padded positions are counted: a weight stationary array is fed
  /// the padding as zeros and the multiplies happen.
  uint64_t macs = 0;
  /// The int8 MAC count, which is zero until the integer kernels of Phase P14.
  uint64_t int8Macs = 0;

  /// `cycles * peak` summed over the compute instructions: the MAC count a
  /// fully occupied array would have retired in the time they actually took.
  /// Always at least `macs`. **Nothing in the energy path ever sees it.**
  double effectiveMacs = 0.0;
  /// The MAC weighted mean spatial occupancy of the array, in (0, 1].
  double utilization = 1.0;
  /// The MAC weighted mean temporal factor for the weight preload, in (0, 1].
  double delta = 1.0;

  /// The two timelines, each the sum of what was placed on that port.
  double dmaCycles = 0.0;
  double computeCycles = 0.0;

  /// The fraction of the shorter timeline hidden underneath the longer one.
  /// 0 is fully serialized and 1 is perfect overlap. See `overlapFraction`.
  double overlapFraction = 0.0;
};

//===----------------------------------------------------------------------===//
// Options and results.
//===----------------------------------------------------------------------===//

/// How to run.
struct SimOptions {
  /// Put every instruction on one port.
  ///
  /// Section 5.5 keeps this behind a flag so that any number published under
  /// the simpler model stays reproducible. Under it the total is exactly the
  /// sum of the two timelines and `overlap_fraction` is zero, which
  /// `SimulatorTest` asserts rather than assumes.
  bool singlePort = false;
};

/// What a run produced.
struct SimResult {
  Stats stats;

  /// The first refusal, or nothing.
  ///
  /// It carries a validation failure, a trap from the bounds checked accessors,
  /// or a kernel that this phase does not implement. One field, because a
  /// caller that had to check three would eventually check two.
  std::optional<std::string> error;

  /// Whether the run stopped at a `HALT` rather than by running out of
  /// instructions. See the class comment on `Simulator` for what the second
  /// case means and why it is not an error.
  bool reachedHalt = false;

  bool ok() const { return !error.has_value(); }
};

//===----------------------------------------------------------------------===//
// The simulator.
//===----------------------------------------------------------------------===//

/// One program, one machine, one run.
///
/// **Running out of instructions stops the machine.** This was the open
/// question Phase P6 left here: a file without a trailing `HALT` decodes and
/// validates, because Section 9.2's check list carries no name for its absence
/// and inventing one would be inventing a rule the specification does not have.
/// So the decision is P7's, and it is that the machine stops exactly as a
/// `HALT` would have stopped it, `SimResult::reachedHalt` says which of the two
/// happened, and `npu-sim` prints a line naming it. It is not an error, for two
/// reasons that point the same way: `docs/ISA_MANUAL.md` already says a machine
/// with no branches stops when it runs out of straight line code, and refusing
/// here would be enforcing at run time a rule the validator was deliberately
/// not given, which is the worst place to put a rule because it is the last
/// place anybody looks.
class Simulator {
public:
  /// Sizes the machine from the program and loads its constants into DRAM.
  ///
  /// The scratchpad comes **strictly** from `program.scratchpadBytes`, per
  /// Section 9.3. Nothing here inspects the instruction stream to decide how
  /// much memory to allocate, because that would be arithmetic on unvalidated
  /// input at the exact entry point the validation exists to defend.
  explicit Simulator(const Program &program);
  ~Simulator();

  Simulator(const Simulator &) = delete;
  Simulator &operator=(const Simulator &) = delete;

  /// Validates and then runs. **This is what `npu-sim` uses.**
  ///
  /// The validation is deliberate repetition: `Program::decode` already
  /// validated, and Section 9.3 asks for it again here because the two calls
  /// guard different moments and a program that arrived through a path that
  /// skipped one still meets the other. It is cheap; the whole malformed input
  /// corpus of Phase P6 validates in single digit milliseconds.
  SimResult run(const SimOptions &options);

  /// Runs **without** validating first.
  ///
  /// It exists for the same reason `Program::decodeUnvalidated` does, and the
  /// reason is worth stating rather than leaving as an apparent hole. Section
  /// 9.3 requires the bounds checked accessors to refuse gracefully **in every
  /// build mode**, and requires a test to prove it. A program that has passed
  /// `validate()` cannot reach that path, which is the whole point of the
  /// validator, so a test that could only submit validated programs could never
  /// exercise the last line of defence and would be asserting that a mechanism
  /// exists rather than that it works.
  ///
  /// **No tool calls this.** Its only callers are the trap tests.
  SimResult runUnvalidated(const SimOptions &options);

  /// Copies bytes into a declared input region, refusing a size mismatch.
  ///
  /// Section 9.3 requires the size comparison, and it is here rather than in
  /// `npu-sim` so that every caller gets it, including the differential test
  /// harness which never goes through the tool.
  bool loadInput(size_t index, llvm::ArrayRef<uint8_t> bytes,
                 std::string &error);

  /// The bytes of a declared output region, or an empty range when the index
  /// or the region is out of range.
  ///
  /// It is not `const`, and the reason is the contract rather than an
  /// oversight: reading DRAM goes through the same checked accessor everything
  /// else does, and that accessor can record a trap.
  llvm::ArrayRef<uint8_t> outputBytes(size_t index);

  /// The machine, for a test that wants to read a scratchpad address directly.
  Machine &machine();
  const Machine &machine() const;

private:
  SimResult execute(const SimOptions &options);

  const Program &program;
  std::unique_ptr<Machine> memory;
  /// The refusal the constructor met, if any: a memory too large to allocate,
  /// or a constant that does not fit where the file says it goes.
  std::optional<std::string> constructionError;
};

} // namespace nbin

#endif // NPU_SIMULATOR_SIMULATOR_H
