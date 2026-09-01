<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Phase state

*Diataxis type: reference.*

Ground rule 17: this file is updated at the end of **every** session, including
a session that achieved nothing, and it carries four things: the current phase,
the status of its gate, the open questions, and the exact next command. This
build spans dozens of sessions, and reconstructing where it stood from `git log`
costs more than writing these lines did.

**Last updated:** 2026-09-01.

## Current phase

**Interphase P9b, the accumulated open questions, decided.** Branch
`phase/p9b-debt`, cut from `main` at `1be5b12`, which is the P9 merge. Nine
commits, none pushed.

**This is not a phase and it does not have a gate.** P9 is merged and P10 has
not started. What this branch is: four items the P9 handoff carried as open
questions, each of which had reached the point where leaving it open cost more
than deciding it, taken deliberately between the two phases rather than folded
into either. Folding them into P10 would have meant P10's gate arriving mixed
with four decisions that have nothing to do with measurement, and one of the
four, the model suite change, moves numbers that P10 is going to report.

| Commit | Subject | Item |
|---|---|---|
| `cb74831` | `docs(breaking): declare what the dilated_stack bias add moves, before it moves` | 3 |
| `34cd163` | `feat(models): dilated_stack carries the separate bias add -npu-fuse-bias exists for` | 3 |
| `d0a239b` | `chore(baseline): re-record after the suite change, in its own commit` | 3 |
| `0bf704a` | `docs(adr): re-measure the dilated_stack tight budget, which did not move` | 3 |
| `e802e30` | `build(docker): libomp-18-dev, so the determinism claim is the same in every job` | 1 |
| `f65378b` | `ci: an ndebug job, because the one CI thought it had was a comment` | 2 |
| `ff6991d` | `ci: regression-baseline --check, with the tolerance left at zero on purpose` | 4 |
| `ed63b0f` | `fix(coverage): clear the previous run's counters, so the number is about this run` | D-0037 |
| tip | `docs: hand off the interphase P9b debt branch` | all |

**The first four commits are ground rule 7's ordering and they are the part a
reviewer should check first.** The declaration at `cb74831` is strictly before
the movement at `34cd163`, the re-record at `d0a239b` is its own commit strictly
after it and touches `test/baseline/` and nothing else, and the ADR
re-measurement at `0bf704a` follows. P9 was the first phase to exercise that
mechanism; this is the second time, and the second time is when a procedure
either holds or turns out to have been written for one case.

**Item 3 is deliberately first in the history.** Nothing after `d0a239b` moves a
recorded number, so the re-record stands as the last word on the baseline and
`regression-baseline --check` is green at every commit from `d0a239b` to the tip.
Putting the CI work first would have meant re-recording over four commits of
unrelated change.

**One commit was not planned and is the branch's best result.** `ed63b0f` fixes
D-0037, which the closing verification matrix found by dying: `coverage.sh`
aborted at collection because gcov counters accumulated across every run since
P8 until a hot line passed 2^32. The crash is the harmless half; see the entry.

## Item by item

### 1. `libomp-dev` in the CI image

**Done in the tree, not yet live.** The image is built and published by CI only,
so this item's deliverable is a Dockerfile that will produce the right image and
a procedure for the orchestrator to run. Both are here; the image itself is an
hour of runner time away and the runbook below is how to spend it.

- **What changed.** `docker/Dockerfile.llvm` installs `libomp-18-dev` in the
  final stage, beside `libclang-rt-18-dev` and for the same reason: clang's
  OpenMP wants a runtime the archive packages separately, and the version is
  pinned to the clang this base ships rather than left to a metapackage that
  follows the archive's default.
- **The smoke test compiles, links and runs an OpenMP program** rather than
  testing for `omp.h`. A present header with no runtime behind it is D-0025 one
  library along, and `test -f` cannot tell the two apart.
- **`llvm-image.yml` gains no trigger.** It gains the procedure, written into
  the file where the design decision lives, and a report step that prints the
  ref, the commit and the published digest. Dispatch takes a **ref**, so a
  Dockerfile on a phase branch is republished by dispatching this workflow
  against that branch with no merge first. The push trigger stays retired; P1
  retired it on purpose and nothing here argues with that.
- **Proof, and it is a rehearsal against the pinned base digest rather than an
  argument.** Before, in a container from
  `ubuntu:24.04@sha256:1e0a86e5...`: `clang -fopenmp` fails at `'omp.h' file not
  found` and cmake prints `Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS
  OpenMP_CXX_LIB_NAMES)`, which is the CI line verbatim. After
  `apt-get install libomp-18-dev`: the program compiles, links, runs at exit 0
  and under `OMP_NUM_THREADS=4`, and cmake prints
  `Found OpenMP: TRUE (found version "5.1")`.
- **Predicted, and this is the thing to check after the republish.** The
  `build-and-test` and `sanitizers` jobs' configure output changes from
  `OpenMP: not found. The convolution kernel runs single threaded.` to
  `OpenMP: found 5.1. The convolution kernel parallelises over the batch and
  output channel dimensions`, and `NPUSimulatorTests` prints
  `[          ] OpenMP is on and reports N threads available.` where it printed
  `[          ] OpenMP is off in this build, so both runs below are single
  threaded and this test asserts less than it does where OpenMP is present.`
  N should be 4 on a hosted runner.
- **Expect the coverage job to keep saying 4.5 while the other two say 5.1**,
  and that is not a discrepancy. gcc's `libgomp` and clang's `libomp` implement
  different revisions of the specification. Both numbers are right and the
  determinism assertion is the same assertion in all three.

**A prediction that was wrong, recorded because it changed the deliverable.** I
predicted the `build-and-test` job's restored build cache would keep reporting
`OpenMP: not found` after the republish, because `find_package` writes
`OpenMP_CXX_FLAGS:STRING=NOTFOUND` into `CMakeCache.txt` and the cache key in
`ci.yml` is keyed on the LLVM tag, which does not move. The planned deliverable
included an image revision component in that key. Measured in the container:
configure without `libomp`, install it, re-run cmake on the **same** build
directory, and the answer is `OpenMP: found 5.1`. `FindOpenMP` re-runs its
`try_compile` when the cached flag variable is falsy, so a `NOTFOUND` does not
stick. **No cache key change is needed and none was made.** A key bumped for a
reason that turned out not to exist would have invalidated every build cache in
the project to fix nothing.

### 2. The NDEBUG CI configuration

**Done and live on the next push.** Decision: **no second LLVM tree**, recorded
in `docs/adr/0009-ndebug-coverage-without-a-second-llvm-tree.md` and pointed at
from D-0031, which stays open as the limit it is.

- **What changed.** A new `ndebug` job in `.github/workflows/ci.yml`. It
  configures `-DNPU_FORCE_NDEBUG=ON`, greps its own configure log for the line
  beginning `NDEBUG:`, builds `NPUSimulatorTests` and `NPUEncodingTests` by name
  and runs both in full.
- **A job and not a step**, because it is a third build configuration in a third
  directory, which is what the sanitizers job is. One job name over two builds
  names neither when it goes red, and it runs in parallel where a step would be
  serial inside the longest job in the file.
- **Two targets by name and not a sweep**, which is D-0031 and a correctness
  requirement rather than an optimization: this directory may build the binaries
  that link no MLIR, and an `npu-opt` built here aborts inside MLIR's own
  context construction.
- **It found a defect, D-0036, and that is why the job was earned rather than
  added.** `ci.yml` claimed since P7 that the sanitizers job was already the
  NDEBUG half of Section 9.3's clause, "which configures RelWithDebInfo and
  therefore compiles with `-DNDEBUG`". Measured on 2026-09-01, that
  configuration against the assertions LLVM ends its flags
  `-DNDEBUG -D_DEBUG -D_GLIBCXX_ASSERTIONS -UNDEBUG`, and the last one wins. So
  **CI has had no NDEBUG coverage at all**, in three jobs that all ran the trap
  tests with assertions on, while a comment said otherwise. That is D-0028, which
  this project found and fixed at P7, surviving one phase later in prose because
  a comment is not a mechanism.
- **Both activation proofs are rehearsed and restored**, under the step script,
  in the recipes section below.

### 3. The `-npu-fuse-bias` suite gap

**Done, and the governance ordering is the deliverable as much as the change
is.**

- **What changed.** `dilated_stack`'s `conv1` was biasless and followed directly
  by a `Relu`, and it now carries an `Add` of a `(1, C, 1, 1)` initializer
  between them. `GENERATOR_VERSION` moves from `1.0.0` to `1.1.0`.
- **Why `(1, C, 1, 1)` and not `(C,)`.** ONNX broadcasting aligns from the
  trailing axis, so a rank 1 initializer of length 5 broadcasts against the width
  of 6 and `onnx.checker` refuses the graph. The four dimensional spelling is
  also what an exported graph carries, and the importer normalises it to the rank
  1 constant `docs/adr/0005` describes, which is what the pass guards on.
- **The order in git history, which is the point.** Declaration `cb74831`,
  movement `34cd163`, re-record `d0a239b`, ADR re-measurement `0bf704a`.
- **The P9 test is deleted and replaced by its inverse.**
  `test_no_suite_model_gives_the_bias_fusion_anything_to_do` has done its job;
  `test_the_suite_gives_the_bias_fusion_exactly_one_target` asserts
  `fired_on == ["dilated_stack"]`, naming the model rather than counting, so a
  shorter list is the gap reopening and a longer one is a model that gained the
  shape without anybody saying so.
- **Proof that the pass does something and not merely that it matches.**
  `test_the_bias_fusion_is_a_saving_and_not_a_rearrangement` asserts the `-O2`
  program is exactly one instruction shorter than the `-O0` one on this model and
  that the two answers are bit identical. Measured: 13 instructions at `-O0` and
  12 at `-O2`, `max_abs_movement_vs_o0` still 0.0.
- **The cycle count is the pretty part.** `dilated_stack` at `-O2` comes back to
  1234.0625 cycles and 710.8125 compute cycles, which are **exactly** the numbers
  it had before the node existed. The twenty extra DRAM bytes are the bias and
  they are read at every level, because the bytes have to arrive whichever
  operation consumes them.
- **The tight budget did not move.** Re-measured by ADR 0008's own 64 byte sweep
  at all three levels: allocated peak 8036, smallest allocatable 8064, which are
  the P8 numbers. The two new buffers are live late in the program where the
  working set is a little over five kilobytes, and this model's peak is set by
  `conv0`, which the change does not touch. Recorded in the ADR as a dated
  re-measurement rather than as an amendment, because the decision did not
  change.
- **The `-sccp` zero row stays and the distinction is recorded in three
  places**: `docs/PASSES.md` under both passes, the module docstring of
  `test/Python/test_transform_passes.py`, and
  `test_sccp_has_nothing_to_do_on_a_single_function`, which asserts the **one
  function** property rather than only the outcome, so a compiler that grew calls
  makes it red instead of leaving a stale sentence.

**What P10's ablation machinery should now expect**, which is the note the P9
handoff asked for:

- **`-npu-fuse-bias`'s row is no longer zero.** It is one instruction on
  `dilated_stack` and zero on the other six models, at `-O2` only. A harness that
  reported the pass's total saving over the suite will report 1.
- **`-sccp`'s row is still zero on all seven**, and the report should say why in
  the sentence beside the table rather than leaving a reader to infer that
  constant propagation is worthless on this workload. It is not worthless; there
  is nothing for it to cross.
- **Six of the eight ablatable passes are unchanged by this branch.**
- **`-O1` is still exactly `-O0` on every model including this one**, so the
  observation P9 asked P10's report to state out loud still holds.

### 4. `regression-baseline --check` cross host

**The step is written and rehearsed. The question is not answered and cannot be
until it runs in the container.** That is the honest status and it is by design.

- **What changed.** A step at the end of `build-and-test`, with
  `NPU_BASELINE_JOBS: "4"`, and `GOLDEN_TOLERANCE` left at **zero**.
- **Why the tolerance was not widened first.** The unknown is whether a baseline
  recorded under gcc on this machine reproduces bit for bit under clang against a
  different libc. A tolerance chosen in advance to make the first run green
  throws away the only measurement the question is about. If the container
  reproduces exactly, that is a property this project did not know it had; if it
  does not, the drift report is the number.
- **The drift report was rewritten so a red first run is readable from the log
  alone.** It said `largest movement 4.7e-08`, which is the same sentence for one
  element moving in its last bit and for every element moving, and those are
  opposite findings. It now names how many elements differ, the index of the
  worst, both values there, and the movement in units in the last place at that
  scale.
- **Rehearsed both ways**, in the recipes below.
- **The watch plan for the first run is in the runbook below** and it is the one
  thing on this branch that needs somebody to read a log and decide.

## Verification output

Every command below was run at the tip of this branch, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 25 discovered, 25 passed |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed |
| `build/bin/NPUAllocatorTests` | 29 tests, 29 passed |
| `build/bin/NPUEncodingTests` | 77 tests, 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 55 tests, 54 passed, 1 skipped |
| `build-ndebug/bin/NPUSimulatorTests` | 54 passed, 1 skipped. See D-0031 for what this directory may and may not build |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build-fuzz/bin/NPUSimulatorTests` under ASan and UBSan | 54 passed, 1 skipped, exit 0 |
| `build-fuzz/bin/NPUEncodingTests` under ASan and UBSan | 75 passed, 2 skipped, exit 0 |
| `python -m pytest test/Python -q` | 860 passed, 18 skipped, 7 deselected, exit 0 |
| `python -m pytest test/Python -q -m 'slow or not slow'` | 867 passed, 18 skipped, exit 0. 864 at P9, plus this branch's three |
| `mypy` | no issues found in 19 source files |
| `black --check .` | 42 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 270 of 270 files. 269 at P9, plus ADR 0009 |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `python scripts/check-reachability.py --skip-models` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `bash scripts/regression-baseline.sh --check` | no drift, exit 0, 1 minute 42 seconds |
| `bash scripts/coverage.sh 85 90` | C++ 86.5 PASS, Python 90.50 PASS, exit 0 |
| `git status --short` | empty |

**The suite grew by three pytest tests**, which is one deleted and four added.
No lit file changed and no C++ test changed.

### The baseline, recorded and checked

`test/baseline/baseline.json`, `schema_version` 2 and unmoved: the schema did not
change, only what it records. `generator_version` is `1.1.0`.

```
regression-baseline --check

  recorded at 34cd1639bc90, checked at f65378bff9ca
  42 cells, 21 golden tensors, levels -O0, -O1, -O2
  largest movement against -O0: 4.470e-08
  suite NPUAllocatorTests      29 passed    0 failed    0 skipped
  suite NPUEncodingTests       76 passed    0 failed    1 skipped
  suite NPUInterfaceTests      23 passed    0 failed    0 skipped
  suite NPUSimulatorTests      54 passed    0 failed    1 skipped
  suite NPUTilingTests         12 passed    0 failed    0 skipped
  suite check-npu              25 passed    0 failed    0 skipped
  suite dash-lint               2 passed    0 failed    0 skipped
  suite pytest                867 passed    0 failed   18 skipped
regression-baseline: no drift.
```

**The diff of the re-record is exactly what the declaration predicted**, checked
line by line rather than glanced at: six `dilated_stack` cells, three
`dilated_stack` goldens, `generator_version`, `git_sha`, the pytest count and the
four test names. Sixty three changed lines in `baseline.json` and nothing else.
`max_abs_movement_vs_o0` is 0.0 on all six cells and
`max_abs_error_vs_onnxruntime` is unchanged, which is the half a reader needs:
the goldens moved because the model computes a different function, not because
the compiler's arithmetic did.

## Activation proofs and rehearsal recipes

**This branch activates two CI steps and one CI job**, which is more CI change
than any phase since P8. Every one was rehearsed under its own step script with
the prediction written first.

### 1. The `ndebug` job, three steps, verbatim

Substitutions for this machine and nothing else: `MLIR_DIR`, `LLVM_DIR`,
`LLVM_EXTERNAL_LIT` and the build directory, so the committed `build-ndebug/` is
left alone.

**Prediction.** The configure prints a line beginning `-- NDEBUG: forced on` and
the grep succeeds. Both binaries build. `NPUSimulatorTests` 54 passed 1 skipped,
`NPUEncodingTests` 76 passed 1 skipped, exit 0.

**Result.** Exactly that. The line is `-- NDEBUG: forced on. NDEBUG is defined,
_DEBUG and _GLIBCXX_ASSERTIONS are not, and no -UNDEBUG is appended.` Twenty
eight ninja edges, both binaries green.

**Which trigger this needs:** none of its own. It runs under `push` to `phase/**`
and under `pull_request`, like every other job in the file.

### 2. Fault, product side: the range trap's diagnostic compiled out under NDEBUG

The fault the new job exists to catch, and it is chosen so that **only** the new
job can catch it.

```cpp
// in readBytes, include/NPU/Simulator/Memory.h
    if (!inRange(memory.size(), byteAddress, bytes))
#ifdef NDEBUG
      return nullptr;
#else
      return outOfRange(space, byteAddress, bytes, what);
#endif
```

The check still runs and the accessor still returns null, so every caller
behaves; what is gone is the recorded trap message. This is the shape a release
build acquires when somebody decides a diagnostic is a debug convenience.

**Prediction.** The assertions build green, unchanged. The NDEBUG build red at
the graceful trap tests, and I predicted three of them.

**Result.** The assertions build reported 55 tests, 54 passed, 1 skipped, exit 0.
The NDEBUG build reported **one** failure,
`Trap.AnOutOfRangeOperandAddressTrapsGracefully`, exit 1. **The count was wrong
and the reason is worth having**: the fault is in `readBytes` alone, and the two
result address tests go through `writeBytes`, so exactly the one test whose fault
this is caught it. A prediction of three would have been satisfied by a net that
was firing for the wrong reason.

**Restore:** `git checkout -- include/NPU/Simulator/Memory.h`, rebuild both. Back
to 54 passed 1 skipped exit 0 in both directories.

**Which trigger this needs:** the `ndebug` job is new, so proving it red under
`pull_request` needs a pull request per Section 19.1. Under `push` to `phase/**`
it proves the job catches the fault, which is the weaker of the two claims.

### 3. Fault, test side: one expectation moved

`EXPECT_EQ(harness.sim().machine().scratchpadBytes(), 32u)` becomes `33u` in
`unittests/Simulator/TrapTest.cpp`.

**Prediction.** Red in both the assertions build and the NDEBUG build, at
`Trap.AnOutOfRangeResultAddressTrapsGracefully`, because a wrong expectation is
wrong in every build mode.

**Result.** Exactly that, one failure and exit 1 in both.

**Restore:** `git checkout -- unittests/Simulator/TrapTest.cpp`, rebuild both.
Both green.

**Which trigger this needs:** none of its own for the existing net; the new job's
half wants the same pull request as fault 2, and one pull request carrying fault
2 covers both nets in one run.

### 4. The `regression-baseline --check` step, verbatim

The step's own guard and its two failure branches, with
`MLIR_PYTHON_PACKAGES_DIR` pointed at the local bindings and
`NPU_BASELINE_JOBS=4` as the workflow sets it.

**Prediction.** Exit 0, no drift, 42 cells, 21 goldens, largest movement against
`-O0` 4.470e-08, pytest 867 passed 18 skipped.

**Result.** Exactly that, in 1 minute 42 seconds.

### 5. Fault: one golden element moved by one ulp

The shape the cross host answer will take if there is one, and the proof that a
red first run is diagnosable from the log alone.

```python
# push the largest element of test/baseline/golden/lenet-O0-out0.npy
# to the next representable float32
```

**Prediction.** Exit 1, one drift line, naming the golden and the movement.

**Result.** Exit 1 and this line, which is what the rewritten report buys:

```
golden lenet-O0-out0: 1 of 10 elements differ, largest movement 1.490116e-08
at index (0, 3), where the baseline records 0.22920392453670502 and this run
produced 0.22920390963554382, which is 1.0 ulps at that scale, against a
tolerance of 0.000000e+00
```

**Restore:** `git checkout -- test/baseline/golden/lenet-O0-out0.npy`. Back to no
drift.

**Which trigger this needs:** the step is new, so the `pull_request` proof wants
a pull request, and it can be the same one as fault 2.

### 6. The image, rehearsed against the pinned base digest

Not a fault, a measurement, and it is the only rehearsal on this branch that
could not be done in the repository at all.

**Prediction.** `clang -fopenmp` fails before and succeeds after, and
`find_package(OpenMP)` follows it.

**Result.** Before: `fatal error: 'omp.h' file not found`, and
`Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)`,
which is the CI line verbatim. After `libomp-18-dev`: compiles, links, runs at
exit 0 and under `OMP_NUM_THREADS=4`, and
`Found OpenMP: TRUE (found version "5.1")`.

**And the prediction that was wrong**, which is written up under item 1: the
build cache does **not** keep a `NOTFOUND` for OpenMP, `FindOpenMP` re-runs its
`try_compile`, and the cache key change I had planned was deleted rather than
committed.

## Defects

Two, both found by this branch and both fixed in it.

- **D-0036**, the NDEBUG build CI named and did not have. Found while writing the
  job that replaces the claim. Not by a test, and no test could have found it:
  the fault was a sentence.
- **D-0037**, the local coverage number as the union of every run the directory
  had ever seen. Found by this branch's own closing verification matrix, which
  died at collection. The crash is the harmless half; a line covered by a test
  that was later deleted stayed covered, and this branch deletes a test.

**The two are the same shape from opposite sides**, and it is worth saying so
once. D-0030 and D-0032 were results that depended on what else was lying around
in CI and were invisible locally. D-0037 is a result that depends on what is
lying around locally and is invisible in CI. D-0036 is neither: it is a claim
that was never measured in any environment, which is the failure class D-0028
named and which this project has now met twice.

## The orchestrator's runbook

Four things to run, in this order. The first is the only one that costs an hour.

### 1. Push the branch

```
git push -u origin phase/p9b-debt
```

That is the first CI run of the `ndebug` job and of the
`regression-baseline --check` step. **Expect `build-and-test` to be several
minutes longer than at P9**, because the baseline check is a second full pass
over the suite: 1 minute 42 seconds here on many cores, so four to eight minutes
on four vCPUs is the range to expect and anything near thirty is worth reading.
The `ndebug` job is a container pull, a configure, twenty eight ninja edges and
two binaries, so six to ten minutes end to end.

**This run happens against the old image**, which has no `libomp`. That is
correct and it is the useful order: it separates "the new job and the new step
work" from "the new image changes what they report".

### 2. Republish the image

```
gh workflow run llvm-image.yml --ref phase/p9b-debt
```

Dispatch offers only workflows on the default branch and separately offers a ref
to run them on. `llvm-image.yml` is on `main`, so it is offered; the ref decides
which `docker/Dockerfile.llvm` is built, because the checkout step takes
`github.ref`. **So the branch must be pushed first, and no merge is needed.**

- About an hour of runner time. The last step prints the ref, the commit and the
  published digest; record all three in `docs/ENGINEERING_LOG.md`, because the
  tag `llvmorg-22.1.8` is reused and the digest is the only thing that
  distinguishes one republish from another.
- Nothing repins. `LLVM_IMAGE` in `ci.yml` names the tag and a republish moves
  what the tag resolves to.
- **If the build fails, it will fail at the smoke test**, which compiles and runs
  an OpenMP program in the final stage, and the message will be a compile or link
  error naming `omp.h` or `libomp`. That would mean the package name is wrong for
  the base, which the rehearsal against the pinned digest makes unlikely.

### 3. Re-run `ci.yml` and read four lines

Re-run the workflow on the branch after the image is live.

- **The configure output of `build-and-test`, `sanitizers` and `ndebug`** should
  read `OpenMP: found 5.1`. The coverage job should still read `4.5`, because it
  uses gcc, and that is right rather than a discrepancy.
- **`NPUSimulatorTests` in `build-and-test`, `sanitizers` and `ndebug`** should
  print `OpenMP is on and reports 4 threads available.` If it still prints the
  off line, the image did not move: check the digest the republish printed
  against what the job pulled.
- **`Determinism.OneThreadAndMaxThreadsAgreeBitwise` should still pass**, and
  now it is asserting something. It is the one test on this branch whose
  **meaning** changes without its code changing, which makes it the one to watch
  rather than the one to assume.
- **`regression-baseline --check`.** See the watch plan below.

### 4. The `regression-baseline --check` first run watch plan

This is the item that is not decided and it is deliberate. Three outcomes.

- **Green.** Then a baseline recorded under gcc on WSL2 reproduces bit for bit
  under clang in the container, which is a stronger reproducibility property
  than this project had any evidence for, and it should be recorded in
  `ENGINEERING_LOG.md` in those words. `GOLDEN_TOLERANCE` stays at zero and the
  question P8 and P9 both deferred is answered.
- **Red on goldens only, by a few ulps.** Then the drift report **is** the data.
  Record the exact lines, which carry the element count, the index, both values
  and the ulp count. **Do not widen `GOLDEN_TOLERANCE` to whatever makes it
  green.** The decision to take then is between two options and it wants the
  measurement in hand: either the tolerance becomes the measured delta with the
  measurement and its date beside the constant, or the baseline gains a recorded
  host and the check becomes a same host comparison. Both are decisions above a
  CI step's pay grade and neither should be taken in the same hour the red
  arrives.
- **Red on a cell or a suite count.** Then it is not the cross host question at
  all. A moved cell is a change somebody made, declared in
  `BREAKING_CHANGES.md` and re-recorded, or a defect. A moved suite count in the
  container most likely means a test skipped there that runs here, which is
  D-0032's family and would be a real find.

The step prints all three readings in its own failure output, so the log says
this too.

## Open questions

Five, down from nine. Four of P9's are closed by this branch and are not
repeated here.

**`GOLDEN_TOLERANCE` is zero across hosts and nobody has run the experiment
yet.** This is item 4's whole point and it is open by construction: the step
exists, the tolerance is unwidened, and the first container run is the
measurement. It closes on the next CI run either way. See the watch plan above.

**The NDEBUG and sanitizer directories still cannot build anything that links
MLIR**, D-0031. What P9b decided is that the second LLVM tree is not worth its
cost against the requirement as it stands, ADR 0009. What is still true is the
limit itself, and a later phase that writes a clause about MLIR linking tools in
a non-assertions build reopens the decision rather than working around it.

**`-O1` is exactly `-O0` on every model in the suite**, unchanged by this branch
and still worth stating in P10's report rather than leaving a reader to infer
that `-O1` does nothing.

**The Python coverage headroom is 0.50 points**, measured 90.50 against a
threshold of 90. It was 0.49 at P9 and 0.61 at P8. The threshold stays at 90.
Still worth watching on the CI host, which has measured a tenth of a point above
this machine both times it has been compared.

**Section 17.3a's fifth metamorphic relation still cannot be written.** `Pad` is
refused by name and `Slice` has no converter, and a test asserts the reason is
still true.

## Next command

Push the branch and open the merge pull request for `phase/p9b-debt`, then
dispatch the image republish against the branch.

```
git push -u origin phase/p9b-debt
gh workflow run llvm-image.yml --ref phase/p9b-debt
```

**Nothing has been pushed.** The order matters: the push is what makes the ref
dispatchable, and the first `ci.yml` run against the old image is what separates
"the new job works" from "the new image changed what it says".

## Next phase

**P10, measurement**, unchanged. Sections 16.1 and 16.2 in full, the full end to
end matrix of Section 17.4, and the prediction mechanism of Section 17.8.

The six things P9 left on P10's desk still stand, with one of them now smaller:
the ablatable set is still eight and **one** of its rows will be zero rather than
two, `-sccp`'s, for the structural reason recorded above and in `docs/PASSES.md`.
`-npu-fuse-bias` now has a target and a saving of one instruction on
`dilated_stack`. The other five items are unchanged: `-canonicalize` appears
twice at `-O2` and is one pass to ablate, the `PassInstrumentation` has a
pipeline to sit on, the matrix is 420 cells and Section 2's 90 minute budget
becomes a gate, `max_abs_movement_vs_o0` is the field a report quotes from, and
the tight budgets are P13's to re-measure.

**One thing P9b adds to that desk.** The `regression-baseline --check` step's
first container run either answers the cross host question or produces the
number that answers it, and P10's report is the first document in this project
that will want to say what reproducibility means here. Whichever way it goes,
the answer belongs in that report rather than only in this file.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both, and repeated at every phase because a fact that stops being
repeated is a fact somebody eventually does not know.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, no agent, not once. It may be read, and only through a command
that cannot write. **Only the owner may retire it.**

The reason it exists on top of git history is that the two protect against
different failures. History protects against a bad commit. A second directory
protects against everything else, because if this rebuild goes wrong at any
point, deleting `~/npu-mlir-v2` returns the machine exactly to its pre build
state with no reasoning about reflogs required. That guarantee holds only while
the frozen copy is untouched.
