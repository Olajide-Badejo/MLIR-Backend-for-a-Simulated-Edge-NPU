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

**Last updated:** 2026-09-04.

## Current phase

**P12, performance.** Branch `phase/p12-performance`, cut from `main` at
`d67bb3c`, which is the P11 merge. Six commits. **Not pushed.**

| Commit | Subject |
|---|---|
| `a182724` | `fix(simulator): the convolution kernel was never compiled with OpenMP` |
| `4ebdfe3` | `perf(experiments): the allocator's fitted growth exponent, at the sizes 13.1 names` |
| `dcf913f` | `perf(experiments): the kernel's thread scaling, and its bitwise check on real models` |
| `d93d48d` | `record: the suite and the baseline with the kernel actually parallel, one run each` |
| tip | `docs: hand off P12, with the phase that was a measurement and turned out to be a repair` |

**The commit order carries meaning and the first commit is the phase.** P12 was
briefed as a measurement: prove the sweep line's growth, parallelise the
convolution kernel, confirm the suite runtime, and keep the whole thing
numerically inert. The kernel already had its `#pragma omp parallel for
collapse(2)`, written at P7. It had never been compiled. `a182724` is that
repair, and everything after it measures a machine that only started doing the
thing this phase was supposed to measure once `a182724` landed.

**No intermediate commit is red.** That is different from P11 and it is not a
virtue, it is a consequence: nothing here declares a numeric movement, because
nothing here moves a number.

## Gate status

**Met.** Clause by clause, with the proof beside each.

| Clause | Status |
|---|---|
| Allocator runtime measured at 500, 1000, 2000 and 5000 **buffers** | met, and the unit is P12's correction. P5 measured the same four numbers as **operations**, and the chain allocates one buffer per two operations, so the committed P5 curve was taken at 249, 499, 999 and 2499. `--size-unit operations` keeps that table reproducible |
| The **fitted** growth exponent reported | met. **1.1038**, least squares of log time on log size over all four points, r squared 0.9987, worst residual +0.0377 in log space, with the residual printed per point and the consecutive steps beside it |
| Consistent with O(n log n) | met, and it is a comparison rather than an opinion. The n log n reference **computed at these exact sizes** is 1.1365 and the fit is **below** it, which is the strongest form the clause admits because O is an upper bound. Linear is 1.0000 and quadratic is 2.0000 at the same sizes; `--check` fails at their midpoint, 1.5683 |
| Bitwise equal outputs at one thread and at full thread count | met, twice over. `Determinism.OneThreadAndMaxThreadsAgreeBitwise` in process on a synthetic convolution, **for the first time meaningfully**, and `experiments/kernel_threads.py` on all seven models at 1, 2, 4, 8 and 28 threads, byte identical on every row |
| Goldens byte identical, exactly | met. All 21, `git status` on `test/baseline/golden` empty after a re-record. `GOLDEN_TOLERANCE` is zero |
| The 90 minute suite budget re-measured and still met | met. **175 cells, 3.43 minutes, 1.17 seconds per cell**, one run, serially, on a quiet machine. `run_benchmarks.py` exits 1 over budget and the branch is rehearsed at P10 |
| The run fails if it is not | met, unchanged from P10. `--budget-minutes 0` drives it and the nightly step turns the nonzero exit into an error naming both readings a red run can have |
| **No naive baseline ratio** | met by not doing it. No naive allocator was written, none is compared against, and no benchmark only flag was added to build one. The report sections this phase touches quote the fitted exponent and the references, which is a comparison against arithmetic rather than against a strawman |
| Numerically inert: goldens byte identical | met, above |
| Cycles, DRAM bytes and instruction counts unmoved | met. 175 cells, **95614 leaf fields**, seven field names moved and every one is a wall clock or provenance: `median_ms`, `iqr_ms`, `ci95_low_ms`, `ci95_high_ms`, `timestamp`, `git_sha`, `content_hash`. Zero forbidden movers |
| Only host wall clock changes | met, and `content_hash` moving is the evidence rather than the exception: it is a sha256 over the compiler sources and the cost model constants, so a phase that changed `Kernels.cpp` must move it, and it moving while every number under it holds still is the claim |
| `CHANGELOG.md` says so explicitly | met. The P12 section opens with it, in the words "a speedup reported below is a statement about how long a developer waits, and it must not be read as the modelled accelerator having become faster" |
| The cost model untouched, D-0045 left to P13 | met. `git diff main..HEAD` touches no file under `lib/Simulator/CostModel.cpp`, `include/NPU/Simulator/CostModel.h` or `python/npu_frontend/cost_model.py`. `content_hash` includes `cost_model.VALUES` and the constants inside it did not move |
| The roofline, the regression bound most likely to notice | met and re-run. 175 cells, 550 layers, every cell at or above its bound, tightest layer unchanged at 0.0006 headroom |

## D-0047, which is the phase

**The convolution kernel was never compiled with OpenMP, in any build, in any
environment, from P7 to P12.** `lib/Simulator/CMakeLists.txt` had one line of
OpenMP wiring, `target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)`.
`add_mlir_library` compiles that library's sources in an object library called
`obj.NPUSimulator`, so a usage requirement attached to `NPUSimulator` reaches
everything that **links** it and never reaches what it is **made of**.
`-fopenmp` landed on `npu-sim`, on `NPUSimulatorTests`, on every consumer, and on
none of the kernels.

**The determinism test is what it cost.** `DeterminismTest.cpp` links
NPUSimulator, so it did receive `-fopenmp`, so its `_OPENMP` was defined, so it
printed a thread count of 28, called `omp_set_num_threads(1)` and then
`omp_set_num_threads(28)`, and compared two single threaded runs. It passed for
three phases while asserting nothing, under a comment header saying in as many
words that a test which silently becomes vacuous is worse than no test. **This
file said at P11 that the determinism assertion "asserts at full strength
everywhere now".** It did not, and it never had.

**How it was found:** by measuring before changing anything. Seven models, 1
thread against 28, and a table of speedups between 0.98 and 1.03. That is exactly
what small models look like and these models are small, so the reading was
available and it was wrong. `/usr/bin/time -v` reporting **98 percent of one CPU**
at `OMP_NUM_THREADS=28` is what separated the two, and `nm` on the object file
settled it.

**The fix is three parts and the third is the one that matters.**
`add_compile_options` at directory scope, where `-Werror=switch` already lives
for the same reason; `nbin::kernelsUseOpenMP()` and `nbin::kernelThreadCount()`
defined in `Kernels.cpp` and reachable from a command line as `npu-sim
--kernel-info`; and `Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`, which
compares this test's `_OPENMP` against the kernels' own answer.

**A second fault the fix exposed.** With the region finally compiled, an uncapped
team made five of the seven models **slower than serial** at 28 threads, by as
much as seven times on `depthwise_separable`, whose depthwise convolutions have
eight and sixteen output channels and were being handed twenty eight threads
each. The kernel caps its team at `batch * outputChannels`, the number of
independent output tiles the instruction has. That carries no tuned constant and
cannot move a bit, because neither the cap nor the `if` clause changes which
iterations exist, what one computes, or the order of the reductions inside it.

## What P12 measured, in one place

`docs/NUMBERS.md` is the ledger. Five things worth repeating here.

- **The suite is 1.17 seconds per cell, 3.43 minutes for 175 cells**, against 90.
  It was 1.27 at P11 with the same tools inside the same suite, and the
  difference is the kernel. The factor in hand went from twenty four to twenty
  six. **This is a host wall clock and nothing else.**
- **Thread scaling is 0.86 to 3.17 times, geometric mean 2.10**, byte identical
  at every thread count on every model. `depthwise_separable` is the row below
  one and it stays reported: 12800 multiply accumulates inside a process that
  takes longer than that to start has no arithmetic left for a thread to win.
- **The fitted exponent is 1.1038 against an n log n reference of 1.1365 at the
  same sizes**, so the curve is **below** n log n rather than merely near it.
- **The inertness proof is a diff and not a claim.** 95614 leaf fields over 175
  cells, seven movers, all wall clock or provenance, zero forbidden. 21 golden
  tensors byte identical. That claim would have been true at P11 for the
  uninteresting reason that the kernel was serial at both ends; it is worth
  something here because the kernel really went parallel in between.
- **The worst `--mlir-timing` gap was 0.1856 ms against D-0043's 0.2000 bound**,
  measured with nothing else on the machine. P11's two quiet runs measured 0.1577
  and 0.1177. Inside the bound, and closer to it than either. Recorded as an
  observation and not as a trend, because one run is not one.

## Verification output

Every command run at the tip of this branch, from `/home/elijah/npu-mlir-v2`, in
`~/npu-venv`. **Every measurement run was taken serially with nothing else on the
machine**, which D-0043's load sensitive bound requires and which this phase in
particular has to say out loud, because this phase is the one that put twenty
eight threads into that machine.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build-ndebug -j6` | clean, no warnings |
| `ninja -C build check-npu` | 26 discovered, 26 passed |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | 12 passed |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | **55 passed**, 1 skipped. 54 at P11, plus the D-0047 assertion |
| `build-ndebug/bin/NPUSimulatorTests` | 55 passed, 1 skipped, and the kernels report 28 threads there too |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **1076 passed, 18 skipped**. 1029 at P11, plus 47 |
| `mypy` | no issues found in 26 source files |
| `black --check .` | 66 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 478 of 478 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date |
| `npu-sim --kernel-info` | `kernel openmp: yes`, `kernel threads: 28` |
| `nm` on `obj.NPUSimulator.dir/Kernels.cpp.o` | **5** OpenMP symbols, 0 before this branch |
| `ldd build/bin/npu-sim` | links `libgomp.so.1`, which it did not before this branch |
| `python experiments/compile_time_benchmark.py --check` | **fitted exponent 1.1038** against a ceiling of 1.5683, exit 0 |
| `python experiments/compile_time_benchmark.py --size-unit operations --check` | **1.0605**, reproducing the P5 curve to the millisecond, exit 0 |
| `python experiments/kernel_threads.py --check` | **byte identical at every thread count**, 0.86 to 3.17 times, exit 0 |
| `python experiments/roofline.py` | 175 cells, 550 layers, every cell at or above its bound, tightest 0.0006, exit 0 |
| `python experiments/scalesim_export.py` | 175 cells, worst whole model divergence **-87.14%**, unchanged, exit 0 |
| `python experiments/accelergy_energy.py` | 175 cells, per MAC **49.286 pJ**, unchanged, exit 0 |
| `python scripts/patch-scalesim.py --check` | every edit in place, exit 0 |
| `python experiments/run_benchmarks.py --force` | **175 cells, 3.43 minutes, 1.17 s per cell**, worst clock gap 0.1856 ms, inside the budget, exit 0 |
| `bash scripts/regression-baseline.sh --check` | **no drift**, 42 cells, 21 golden tensors, exit 0 |
| the field by field inertness diff, 175 cells before against after | **95614 leaves, 7 movers, all wall clock or provenance, 0 forbidden** |
| `bash scripts/coverage.sh 85 93 16 58` | C++ **86.4** PASS, branch 76.8; per tree **93.4313 / 16.1191 / 74.5156** PASS, exit 0 |
| the whole suite in the CI shape, all three differences modelled | **1063 passed, 31 skipped, 0 failed**, mypy clean, `coverage.sh` PASS at **93.4313 / 16.1191 / 62.0669**. The skip count is CI's exactly |
| `regression-baseline --check` in the CI shape | **no drift**, with both environments named and the count difference printed |
| the same environment with `NPU_EXTERNAL_TOOLS=1` | the guards **fail** naming the variable rather than skipping, which is the third branch of `tools.py`'s policy |
| `git status --short` | empty |
| `git log -p main..HEAD` grepped for tooling and authorship traces | 0 matches, case sensitive with word boundaries |
| the same diff grepped for em and en dashes | 0 matches |

**The suite grew from 1029 pytest tests at P11 to 1076**, in two new files:
`test_compile_time_benchmark.py` and `test_kernel_threads.py`. One C++ test was
added, `Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`, and one lit RUN line
for `npu-sim --kernel-info`. **No existing test changed its result**, and the
baseline re-record touches `git_sha`, the two suite counts and the added test
names, and nothing else.

**Python coverage on `experiments/` widened from 0.45 points of headroom to
4.07.** The threshold stays at 58. The two new test files cover the two
experiment modules that had none, which is why a phase that added roughly 230
statements to a tree gated on a ratio raised the ratio.

| Tree | CI shape | Developer machine | Threshold |
|---|---|---|---|
| `python/npu_frontend` | 93.4313 | 93.4313 | **93** |
| `scripts` | 16.1191 | 16.1191 | **16** |
| `experiments` | **62.0669** | **74.5156** | **58** |

**C++ coverage moved from 86.5 to 86.4 and branch from 76.9 to 76.8**, both above
their thresholds. The 0.1 is `npu-sim --kernel-info`, whose two lines are covered
by a lit test rather than by the gcov instrumented unit test binaries. It is
named here rather than left as an unexplained dip.

## The Section 2 carve out, for the owner

**Section 2 says the per cell cost is "budgeted at 15 seconds per cell as a
planning figure" and that "the 15 second per cell figure is itself replaced by
the measured value at P10".** The specification file lives outside this
repository and the standing rule is that it is not edited from here, so the
replacement is recorded rather than applied. **This is the one item of P10's gate
that needs a hand other than this branch's.**

- **The measured figure is 1.17 seconds per cell**, over 175 cells in 3.43
  minutes, serially, on the 14700K under WSL2. Recorded in
  `experiments/results-runtime.json` and quoted in `docs/NUMBERS.md`. It was 1.27
  at P11 and the difference is D-0047: the convolution kernel is genuinely
  parallel from P12 and was not before. **That is a host wall clock and no
  simulated number moved with it.**
- **It was 0.60 seconds at P10 and that difference is P11's external tools**,
  which now run inside the same suite: a SCALE-Sim invocation and, on the dilated
  cells, two, plus one Accelergy invocation per distinct scratchpad budget. The
  suite finishes in under four minutes against ninety, so the factor in hand has
  gone fifty, twenty four, twenty six. The figure below is the one to quote.
- **Section 2's arithmetic needs two further corrections**, and they change the
  cell count rather than the cost:
  - it multiplies **11** ablatable passes; there are **8**, because
    `-npu-assign-layout`, `-npu-tile-to-scratchpad` and `-npu-double-buffer`
    arrive at P13 and no `-O` level names them yet;
  - it takes budget and batch as a free cross product; they are not, per
    `docs/adr/0010`, so the benchmark cells are 63 rather than 84.
- **Suggested replacement text**, for the paragraph beginning "That 90 minutes is
  derived rather than guessed":

  > **Benchmark cells:** 7 models times 3 levels times 3 budget and batch
  > combinations equals **63**; the budget and the batch are not independent
  > axes, because a tight budget is the smallest budget at which one program
  > allocates and the same model at a larger batch is a different program.
  > **Ablation cells:** 8 ablatable `-O2` passes times 7 models times 2 budgets
  > equals **112**. Total **175 cells**. Measured at Phase P12 at **1.17 seconds
  > per cell** serially, which is 3.43 minutes for the whole suite including the
  > external cross validation tools, so **the stated budget of 90 minutes stands
  > with a factor of twenty six in hand**. The headroom is what pays for running
  > the cells serially, which the per cell timing objects of Section 16.1
  > require. It was 1.27 seconds at P11, before the convolution kernel was
  > compiled with OpenMP, and 0.60 at P10, before SCALE-Sim and Accelergy ran
  > inside the suite. Both differences are host wall clock and neither moved a
  > simulated number.

**The ablatable set goes from 8 to 11 at P13**, when `-npu-assign-layout`,
`-npu-tile-to-scratchpad` and `-npu-double-buffer` arrive and an `-O` level names
them. That takes the ablation cells from 112 to 154 and the total from 175 to
217, so the paragraph above needs re-deriving in the P13 commit that adds them
rather than after it.

## What P11 measured, and still holds

`docs/NUMBERS.md` is the ledger and is the file to read. Six things worth
repeating here. **The suite runtime figures in this section are P11's and are
superseded**; every number that is a property of the design rather than of the
host is unchanged, which is P12's whole claim.

- **The roofline cannot fail against this cost model**, and that is the phase's
  most useful negative result. `effective_macs` is defined as `cycles * peak`, so
  the compute branch is the kernel's own cycle count; a transfer costs bytes over
  bandwidth **plus** a descriptor, so it always exceeds the memory branch its own
  bytes produce. All 175 cells and all 550 layers are at or above their bound and
  none of that is evidence. It is a regression bound waiting for P13, and both
  halves of the tautology are asserted so the day either stops holding a test
  says so.
- **The divergence prediction was mostly wrong and is answered as written.**
  Direction wrong on five of seven models, all three magnitude bands wrong, both
  rank fidelity figures wrong, the coverage floor on `lenet` wrong. Right about
  the mechanism behind the widest positive gaps, about pooling, and about there
  being a fragmentation disagreement. 340 of 550 layers exceed 25 percent where
  it predicted none would.
- **The root cause of that band is D-0045 and it is a real finding about this
  cost model.** The array's weight preload is charged once per instruction here
  and per fold by SCALE-Sim, worth about a factor of three on a narrow deep
  convolution. Not fixed at P11, because retuning a model against an external
  tool invalidates every ablation already recorded.
- **The suite wide divergence headline is the small remainder of two large
  opposite terms**: double buffering at plus 442289 cycles and array
  fragmentation at minus 435825, summing to plus 109756. Quoting the headline
  without them would be quoting an accident.
- **Energy runs from 1.7 uJ per inference on `depthwise_separable` to 54.4 uJ on
  `lenet`, on an 8.5 mm2 design at 45 nm.** The fp32 MAC coefficient **fails**
  Section 16.4's order of magnitude sanity check at a factor of 10.71, for an
  identified reason that is not this project, and `docs/NUMBERS.md` records that
  at the published coefficient the scratchpad would be the largest consumer on
  every model. **No conclusion here rests on the array being dominant.**
- **Fusion moves exactly zero picojoules**, on all seven models, which is the
  same zero the P10 ablation table records for instructions and cycles arriving
  in the currency fusion is usually argued in. What it would be worth where the
  intermediate spilled is quantified beside it.

### What P10 measured, and still holds

- **175 cells, 1.76 minutes, 0.60 seconds per cell**, against a 90 minute budget.
- **Two of the eight ablatable passes have a nonzero row.**
  `-npu-fold-batchnorm` saves 8 instructions and 212 cycles on
  `conv_bn_relu_stack`; `-npu-fuse-bias` saves 1 instruction and 9.625 cycles on
  `dilated_stack`. Both agree with the P9 measurements taken one pass at a time,
  arriving at the same numbers from the opposite direction.
- **`-O1` is exactly `-O0` on all seven models**, which P9 asked P10's report to
  state out loud. The reason is in the ledger: no model has an elementwise
  operation with two constant operands, so the folder has nothing to fold and the
  canonicalization after it has nothing dead to clean up.
- **`-sccp`'s row is zero for a structural reason**, one function and no calls,
  and that is a different kind of zero from `-canonicalize`'s. The next section is
  the phase's most interesting result.

### The finding: `-canonicalize`'s zero row is a limit of the method

The registered prediction expected `-canonicalize` to be one of three nonzero
rows. It is zero on all seven models, and the instrumentation says why:

```
with canonicalize            without canonicalize
  npu-fuse-ops    34 -> 38     npu-fuse-ops    34 -> 38
  canonicalize    38 -> 24
  cse             24 -> 21     cse             38 -> 21
```

The canonicalization removes fourteen operations, so it is not idle. `-cse`
reaches the same twenty one without it, because MLIR's CSE erases trivially dead
operations as it walks. **A leave one out ablation cannot see a pass whose work
another pass would have done.** Recorded in `docs/PASSES.md` and
`docs/NUMBERS.md`, because a table of deltas with no prose beside it would report
this identically to `-sccp`'s zero and the two have nothing in common.

## What reproducibility means in this project

P9b's handoff asked P10's report to be the first document that says this. It has
two halves and they have different answers.

**The compiler and the simulator are bit stable, at a tolerance of zero.** A
baseline recorded under gcc on WSL2 reproduces bit for bit under clang in the CI
container: every cell field, all 21 golden tensors, every suite count, over four
CI runs across at least two runner hardware generations. `GOLDEN_TOLERANCE` is
zero on evidence. P10 adds to that from a different direction: this phase put an
instrumentation on the pass manager, added a schema, a harness and ninety tests,
and the re-recorded baseline moved not one cell field and not one golden byte.
The diff of `5401d39` is that claim's proof.

**The distance to the oracle is a property of the measuring host, and is bounded
rather than fixed.** `max_abs_error_vs_onnxruntime` has two ends and only one
belongs to this project; `onnxruntime` dispatches its CPU kernels on what the
host supports, and eighteen cells moved between 1e-8 and 1e-7 in both directions
across runner hardware. It is compared against Section 17.4's band, imported from
`npu_frontend.tolerances`, and never for equality. That is D-0039.

**The two halves are why the result schema separates counted metrics from timed
ones.** `instruction_count` is exact and reproduces anywhere; a wall clock is a
measurement of one host and carries an interval saying so. Section 16.1's
determinism test asserts a re-run is byte identical **apart from the timestamp
and the timing object**, which is the same distinction made executable.

## Activation proofs and rehearsal recipes

### 0. Reproducing the CI image locally, which is now a standing recipe

*Added at P11 after D-0046.* This project has to stay green in **two**
environments, and the second cannot be reached by running the suite here. The
recipe below is what CI's image looks like from this machine, and run
33707070166 is the evidence that getting it slightly wrong is worth catching:
the first version modelled two of the three differences and predicted a suite
row two tests off.

Three things, and all three are needed:

1. **`import scalesim` and `import accelergy` must fail.** A `sitecustomize.py`
   on `PYTHONPATH` installing a meta path finder that raises
   `ModuleNotFoundError` for those two roots.
2. **The `accelergy` binary must not be on `PATH`**, and everything else must
   be. The `PATH` is the venv's `bin` **minus** those entry points, not a `PATH`
   without the venv: the image has `gcovr`, `pytest` and `python3`, and the
   first shim dropped them and died on a missing `gcovr` rather than on anything
   real. Symlink each entry except `accelergy*` and `scalesim*`, and write
   `python`, `python3` and `python3.14` as `exec` wrappers rather than symlinks,
   because a symlinked venv interpreter resolves its prefix from the link's own
   directory and then finds no site-packages.
3. **The pinned SCALE-Sim source clone must not exist.** Point
   `NPU_SCALESIM_SOURCE` at a path that is not there. `~/npu-external/` is a
   developer machine artefact, and the two tests that read the example CSVs out
   of it never import `scalesim`, so 1 and 2 leave them running. **Those are the
   two tests run 33707070166 found.**

`NPU_EXTERNAL_TOOLS` must also be unset, so the guards take their skip branch
rather than their fail branch.

**What it does not model, stated so nobody assumes otherwise**: a different libc,
CPU, compiler or container uid. Those move golden tensors and timings rather than
which tests run, and `--check` has its own bands for them.

**mypy needs a separate reproduction**, because it resolves imports statically
rather than at run time and no meta path finder reaches it.
`mypy --python-executable /usr/bin/python3` is what makes it see what CI sees.

The recipe predicts CI's suite row exactly: **996 passed, 31 skipped** at
`1e77083`, which is what run 33707070166 reported. **At P12's tip it predicts
1063 passed, 31 skipped**, and the shim was re-run at that tip to get it.

### What P12 activates, which is nothing, and the two triggers that go with it

**This branch activates no CI step and no CI job.** That is a decision rather
than an omission and it is recorded with the trigger for reversing it, per
Section 19.1.

**`experiments/compile_time_benchmark.py --check` is not wired in.** It is a wall
clock measurement and the runner pool is heterogeneous. A fitted exponent is a
slope taken within one run on one host, so it is not the forbidden comparison of
a wall clock across hosts and gating it would be defensible. It is still not
being switched on, because a four vCPU shared runner measuring a five millisecond
pass at the smallest size has a noise floor this machine does not, and Section
19.0's rule cuts both ways: silence and success must not look alike, and a red
nobody believes is a red nobody reads. The P12 gate asks for the exponent to be
**reported** and consistent, and it is, in the matrix above.

> **Trigger: P13.** Tiling makes functions longer, which moves the crossover with
> the genuinely quadratic offset assignment scan toward the measured range, and
> that is the phase where this curve starts being able to catch something. The P5
> prediction entry already says so in those words. Switch it on under
> `pull_request` and `push` to `phase/**` like every other step in `ci.yml`, and
> rehearse it red first with `--sizes 500`, which is the branch that has no fit
> and exits 1 naming what to do about it.

**`experiments/kernel_threads.py` is not wired in either, and for a different
reason.** Its gate is the byte comparison, and the byte comparison already runs
in CI as `Determinism.OneThreadAndMaxThreadsAgreeBitwise`, in process, on a
synthetic convolution, **at full strength for the first time** now that the
kernels compile with OpenMP. What the script adds over that is the seven real
models, which is a nightly's worth of value rather than a per push step's.

> **Trigger: the first phase that changes the convolution kernel's loop nest**,
> which is P13's tiling or P14's integer kernels. Add it to `nightly.yml` beside
> `full-matrix`, not to `ci.yml`.

### 0b. The two faults this branch injected, with predictions written first

Both new checks were driven to their failure branches. Neither is a CI
activation, so neither needs a trigger; both are here because a gate nobody has
seen fail is a gate nobody knows works.

**Fault A: D-0047's own cause, reintroduced.** *Predicted:* removing the two new
lines from `lib/Simulator/CMakeLists.txt` turns
`Determinism.TheKernelsAgreeWithThisTestAboutOpenMP` red naming the object
library, and leaves the two beneath it **green**, because green is what they were
in exactly this state for three phases. *Result:* exactly that.
`npu-sim --kernel-info` printed `kernel openmp: no` in the same tree, and the
surviving test printed `OpenMP is on and reports 28 threads available, and the
kernels report 1`, which is the entire defect in one line, printed by the test
that could not see it. Restored, tree clean.

**Fault B: the reduction moved into the parallel region**, which is the mistake
Section 10.3 forbids by name: the outer pragma removed and
`#pragma omp parallel for reduction(+ : accumulator)` put on the input channel
loop. *Predicted:* the bytes move, `kernel_threads.py` prints DIFFER and exits 1,
and `Determinism.OneThreadAndMaxThreadsAgreeBitwise` goes red in the same tree.
*Result:* **DIFFER on all seven models**, exit code **1** confirmed directly
rather than through a pipe, and the C++ test red. Both gates see it, which is
what a second gate is for. Restored, tree clean.

**The ceiling branch of `--check` is driven in pytest rather than on the command
line**, and the reason is worth stating: making the real allocator quadratic is
not a fault injection, it is a different program.
`test_the_ceiling_separates_the_two_hypotheses_13_1_names` hands the same
function a synthetic quadratic curve and asserts it fails, and a synthetic
n log n curve and asserts it passes. That test runs in every CI job that runs
pytest, so the **discrimination** is checked everywhere even though the
**measurement** is not.

### The activations P11 carried

**That branch activated one CI step and one CI job.** Both were rehearsed under
their own step scripts with the prediction written first.

### 1. `pytest slow cells` in `ci.yml`

**Prediction.** Two tests carry the `slow` marker, the two P10 adds, and the full
run is green.

**Result.** **Nine.** The prediction was wrong and finding out why is D-0040:
seven `test_every_model_imports_at_a_second_seed` cases have been marked `slow`
since the model suite landed, and CI has never run one of them, because the only
step that runs slow tests is this one and this one was off. The suite runs green
at 957 passed, 18 skipped.

**Fault, and it is one only this step can catch.** Every `@pytest.mark.slow`
removed. The step's count guard reports zero and exits 1 with the message saying
a zero count means the step is a second copy of the one above it. Restored, tree
clean.

**Which trigger this needs:** none of its own. It runs under `push` to `phase/**`
and under `pull_request` like every other step in the file.

### 2. The `full-matrix` job in `nightly.yml`

**Prediction.** 175 cells, roughly 0.6 seconds each, inside the budget, exit 0,
and `experiments/results/` untouched because the job writes to `RUNNER_TEMP`.

**Result.** Exactly that: 175 cells, 1.84 minutes, 0.63 seconds per cell, exit 0,
`git status` on `experiments/results` empty.

**Fault: the budget gate driven to its failure branch.** `--budget-minutes 0` is
exceeded by any run at all. The harness prints the measured runtime against the
budget and exits 1; the step turns that into an error naming the two readings a
red run can have, a suite over budget or an ablation that moved the numerics.

**Which trigger this needs:** the job is new and runs on a schedule, so proving
it wants `gh workflow run nightly.yml --ref phase/p10-measurement` rather than a
wait until 03:30 UTC.

### 3. The shallow checkout, which is D-0041 and was not rehearsed in advance

**This one was found by CI rather than by a rehearsal, and that is the honest
label.** The two activations above were rehearsed under their own step scripts
with predictions written first, and neither rehearsal could have caught this,
because both ran in this repository with its history present. The variable was
the checkout itself, and nothing local varies it.

Rehearsed now, as four real fetches from a bare mirror rather than as mocks,
because what is under test is what git does. `push` and the `pull_request` merge
ref, each at depth 1 and depth 0. At depth 0 both are green and the harness exits
0; at depth 1 both refuse by name. The merge ref matters separately because its
`HEAD~1` is the base branch, so the ancestor assertions could plausibly have
broken there and do not.

**Which trigger this needs:** none of its own, and that is the point. The fix is
in the checkout step of three jobs, so the next push exercises it, and a green
`pytest` arm is the proof.

### 4. The dubious ownership refusal, D-0042, in the real image

Also found by CI rather than by a rehearsal, for the same reason as D-0041: the
variable was who owns the workspace, and nothing local varies it.

Rehearsed now in the pinned image, workspace chowned to uid 1001 and the
container running as root, which is the runner's shape. Every helper, before and
after the `safe.directory` step:

| | without `safe.directory` | with it |
|---|---|---|
| `repository_is_shallow` | refuses, quoting git's fatal | `False` |
| `commit_exists(present)` | refuses | `True` |
| `commit_exists(absent)` | refuses | `False` |
| `is_ancestor` | refuses | `True` |
| `head_sha` | refuses | the sha |
| `landing_sha` | refuses | `f92de427d1f3` |

**The left column is the point.** Before the fix those same six calls returned
`False`, `False`, `False`, "genuinely absent", `""` and `None`, which is six
wrong answers and is what CI reported. The code half of the fix stands on its
own: even with the environment still broken, nothing is answered wrongly.

**Which trigger this needs:** none. The `safe.directory` step runs in every job
that has it, so the next push exercises it.

### 5. The two clocks' quantum, D-0043, under the coverage build

Found by CI's coverage job, which is the one build where the fault is frequent
enough to be seen: gcov makes each pass slower and noisier, so the sum of eleven
rounded figures crosses zero often.

Rehearsed against `build-coverage` locally, ten runs of one cell:

```
run 0: mlir  4.1000  instr 3.978771  shortfall -0.121229
run 2: mlir  5.5000  instr 5.297579  shortfall -0.202421
run 3: mlir  3.2000  instr 3.262634  shortfall +0.062634   <-- fails a strict >=
run 7: mlir  3.0000  instr 2.859018  shortfall -0.140982
```

**One run in ten reproduces it**, and run 3 is CI's failure in a larger margin:
CI's was 1.7 microseconds, this is 63. Against the derived bounds, over the same
ten runs: worst per pass deficit **0.039971 ms** against 0.05, worst total
shortfall **+0.062634 ms** against an allowance of 0.55.

**The bound got tighter rather than wider**, from 0.15 ms per pass to 0.05, which
is the point: 0.05 is not a tolerance chosen against data, it is the largest
error a figure printed to four decimals of seconds can carry, and the containment
argument makes it exact.

**Which trigger this needs:** none. It is a pytest test and runs in every job that
runs pytest, including the coverage job that found it.

### 6. The two traceability tests, both faults

**Fault A**, a hand typed number: `resnet_block`'s cycle count changed from 1626
to 1499 in the README table. Red, naming the number and the row it sits in.

**Fault B**, stale provenance: `npuResultsSha` pointed at a commit that is not
the results'. Red, saying the results were re-recorded and the macros were not,
with the staleness check red beside it. Both restored.

**Which trigger these need:** none. They are pytest tests and run in every job
that runs pytest.

## Defects

**One new at P12, and it is in the build, which is a first.** Every defect this
project has found before this one was in code, in a test, or in a claim about
one. This one was in a CMake usage requirement, which is the layer everything
else takes on trust.

- **D-0047**, the convolution kernel was never compiled with OpenMP, and the test
  that proves it is bitwise stable was comparing two serial runs. The OpenMP
  usage requirement was attached to `NPUSimulator` and `add_mlir_library`
  compiles that library's sources in `obj.NPUSimulator`, so `-fopenmp` reached
  every consumer of the library and never reached the library. From P7 to P12 the
  `parallel for` of Section 10.3 was preprocessed away while the configure log
  said it was there, and `DeterminismTest.cpp`, which links the simulator and
  therefore did have OpenMP, printed a thread count of 28 and compared two single
  threaded runs. **Resolved**, in three parts, with a rehearsal that turns the
  fault back on and shows which test goes red. `docs/DEFECT_LOG.md` carries the
  reproduction and the second, smaller fault the fix exposed.

  **It is the fourth appearance of P10's shape and the first inside the build.**
  D-0040 through D-0043 were each a value that arrived through a channel which
  loses information, treated as though it had not. Here the value is "was this
  compiled with OpenMP", the channel is CMake's distinction between a target and
  the object library it is built from, and the two readers of that value never
  compared notes. **What is new is that nothing about it was observable from
  inside the process that cared.** `_OPENMP` is a property of a translation unit
  and every unit that asked was answering correctly about itself.

  **It was found by measuring before changing anything**, which is the practice
  worth carrying rather than the fault worth remembering. A table of speedups
  between 0.98 and 1.03 on seven small models is what small models look like, and
  the reading that made the difference was `/usr/bin/time -v` reporting 98 percent
  of one CPU at `OMP_NUM_THREADS=28`.

### The two from P11, one open and one resolved

**Neither is in this project's own code.** One is in an external tool and one is
in this project's cost model, found by the external tool, which is what cross
validation is for.

- **D-0044**, SCALE-Sim v3 does not run under numpy 2, and reports a missing
  input file by exiting **zero**. Three `int(max(...))` casts over numpy arrays,
  failing on the tool's own shipped example, on every upstream branch at the
  pinned sha. `scripts/patch-scalesim.py` is the three expression fix, applied by
  hand and never as a side effect of running a benchmark, and every manifest
  records `scalesim_installed_tree_sha256` beside the upstream sha so the record
  says the tool was modified. **The second half is the one to remember**:
  `scale_sim.set_params` calls the builtin `exit()` on an input error, which is
  status zero, and Accelergy's own shipped example crashes and exits zero too.
  Neither wrapper in this project reads an exit status; both check the outputs.
  That is D-0040 through D-0043's shape arriving from outside the repository.
- **D-0045**, the cost model charges the array's weight preload once per
  instruction and SCALE-Sim charges it per fold. **Found by the cross
  validation**, not by reading the code, on `resnet_block`'s 3 by 3 convolution:
  478 cycles here against 1465 there for the same 36864 multiply accumulates,
  reconciled by `check_the_same_arithmetic` before being compared. It is the
  dominant term in the fragmentation column and the root cause of every layer in
  the above 25 percent band. **Open, deliberately.** Retuning a cost model
  against an external tool invalidates every ablation already recorded, which is
  the rule Section 16.5 states for ZigZag; P13 gets the reproduction.

### The three from P10, which are fixed and stay recorded

Three, all found by that branch and fixed in it. **Two were found by CI, on the
two pushes it had, and neither could have been found locally**: one varied the
checkout depth and the other varied who owns the workspace, and nothing on a
developer machine varies either.

- **D-0040**, seven tests marked `slow` at P3 that CI has never run. **Found by a
  prediction that was wrong**, while rehearsing the step that turns them on. It
  is D-0037's mirror inverted: D-0037 was a result that depends on what is lying
  around locally and is invisible in CI; this is a *test* that only ever ran
  locally, so any environment dependent failure in it had exactly one place to
  hide and that was the only place nobody looked.
- **D-0041**, the first CI run of this phase's tests asked questions a shallow
  checkout cannot answer. **Found by CI**, run 33559636835: eight failures, all
  in P10 test files, all green locally. `actions/checkout` fetches one commit by
  default and P10 is the first phase whose tests resolve historical shas. The fix
  is `fetch-depth: 0` on the three jobs that run the suite **and** a refusal in
  this project's own code, which is the half that mattered more: `landing_sha`
  was returning the graft commit, a sha that looks like an answer and is not, and
  a result naming it would have satisfied the ancestor test while recording the
  wrong provenance. No checkout option prevents that one.
- **D-0042**, a git fatal read as an answer, and a probe that could not have told
  either way. **Found by CI**, run 33571635111, on the commit that fixed D-0041:
  same tests red, evidence moved, the history now fetched and the commits now
  present, and the tests still calling them absent. The workspace is owned by the
  runner user and the job runs as root in a container, so git refused the
  repository as dubiously owned and exited 128, which this project read as "no".
  `rev-parse --is-shallow-repository` prints its fatal to **stdout**, so D-0041's
  day old guard passed and let everything downstream run. Two parts to the fix,
  and the second is the one that made the first possible: every git call goes
  through one runner that raises unless the exit code is genuinely an answer, and
  `commit_exists` moved off `cat-file -e`, which returns 128 for an absent object
  **and** for an unreadable repository and cannot separate them at all.
- **D-0043**, a bound between two clocks that ignored the quantum of the coarser
  one. **Found by CI**, run 33575891610, which is also the run where
  `build-and-test` went green including every D-0041 and D-0042 test, so those
  two are proven in CI. `--mlir-timing` prints seconds to four decimals, the
  instrumentation records microseconds, and the comparison treated both as exact;
  it failed on a 1.7 microsecond margin between a one decimal figure and a six
  decimal one. Every bound now derives its quantum from the text actually parsed,
  and **the per pass bound got tighter**, from a loosely chosen 0.15 ms to a
  derived 0.05 ms. Reproduced locally against `build-coverage` at one run in ten.

**No defect was found in the compiler.** Every failure this phase produced was in
the measuring apparatus or in a claim about it, which is what a measurement phase
should expect and is worth saying rather than leaving as an absence.

**The four are one lesson told four ways, and it is worth stating because the
project already knew it.** D-0040 was a test that only ever ran in one
environment. D-0041 was a helper that answered a question it could not observe.
D-0042 was the same helper answering a different unobservable question, one day
later, through a probe that could not have distinguished the cases even had the
exit codes been read correctly. D-0043 is a comparison that ignored the precision
of one of its two operands.

All four are the same shape: **a value arrived through a channel that loses
information, and the code treated it as though it had not.** Section 16.1 spends
a paragraph forbidding exactly this for result fields, which is why
`instruction_count` is an integer and every wall clock is an object carrying an
interval that says how uncertain it is, and why `values_of` refuses to average a
`null`. **The schema had the discipline and the code around it kept not having
it.** That is the part worth carrying into P11, which shells out to two more
tools and parses their output: whatever reads an external tool's numbers has to
carry that tool's precision alongside them.

**D-0047 makes it five, and it moves the lesson down a layer.** The four above
are all in code somebody could read. D-0047 is in the build, where the lost
information is which target a compile option reached, and the two files that
disagreed about it were both correct about themselves. **The practice that found
it is the transferable part**: measure the thing before changing it, and when the
measurement is the number you would have predicted anyway, ask the machine a
question with a yes or no answer. `nm` and `ldd` settled in one line what a table
of speedups could not settle at all.

## The prediction, adjudicated

`experiments/predictions/p11-scalesim-divergence.md`, committed at `f92de42`
before `experiments/scalesim_export.py` existed at all. **Most of it is wrong**,
and the full clause by clause answer with the entry quoted verbatim is in
`docs/NUMBERS.md`. The file is not edited.

| Claim | Verdict |
|---|---|
| Direction: this project reads **above** SCALE-Sim on the covered layers | **wrong on five of seven models.** Above on `lenet` and `lenet_batched`, below on the rest, by as much as 87 percent. The entry's own falsification list names this |
| The gap is dominated by named modelling differences rather than by error | **met.** The decomposition is a partition of named terms with a residual of exactly zero |
| Under 10 percent per layer on dense compute bound layers, 10 to 25 on the 1 by 1s, 5 to 20 whole model | **wrong on all three.** 340 of 550 layers exceed 25 percent; the 1 by 1s reach +334; whole model reaches -87 |
| No layer above 25 percent at the default budget with both ports on | **wrong**, and D-0045 is the root cause the clause demands |
| Mechanism 1: SCALE-Sim reads low on 1 by 1 convolutions, widest gaps there, this project reading higher | **confirmed**, and it is the one mechanism the entry got fully right |
| Mechanism 2: pooling leaves the comparison as coverage rather than as divergence | **confirmed.** 53114 cycles, its own term, never inside a divergence |
| Mechanism 3: the two models disagree about array fragmentation | **confirmed**, and larger than expected at minus 435825 cycles |
| Coverage 0.5 to 0.85 on the convolutional models | **met.** 0.591 to 0.797 |
| Coverage below 0.3 on `lenet` | **wrong.** 0.946, for a reason the entry's falsifier misdiagnoses. `scalesim_covered_op_fraction` is 0.208 and is the quantity the clause was reaching for |
| Kendall tau above 0.8, pairwise above 0.85 | **wrong on tau**, 0.6337 over cells and 0.7460 over layers. Pairwise 0.8211 over cells and 0.8783 over layers, so below on one and above on the other |

**The entry called rank fidelity "a stronger finding than any absolute error
here" and it was right about that.** The cost model is not merely imprecise; it
orders one pair in five differently from the reference.

### The P10 prediction, still adjudicated

`experiments/predictions/p10-ablation-deltas.md`, committed at `f92de42` before
the harness had been run once. **Two of its four claims are wrong.** The file is
not edited; the adjudication is in `docs/ENGINEERING_LOG.md`.

| Claim | Verdict |
|---|---|
| 1. Exactly three ablation rows are not zero | **wrong.** Two are. `-canonicalize` is zero, and the mechanism above is why |
| 2. No ablation moves numerics beyond tolerance, and ablating the batch norm fold returns the movement to 0.0 | **met**, with a wording defect recorded: "no other ablation moves that field at all" has two readings and the strict one is false |
| 3. The rows are identical at both budgets | **wrong.** 16 of 56 differ, because `resnet_block` and `inception_block` spill at their tight budget, which ADR 0008's own table already recorded and the prediction misread |
| 4. Suite total between 100 and 130, and no ablation raises it by more than 30 | **met.** 117, and the largest single raise is 8 |

Claim 3 being wrong is the more useful of the two: it is Section 16.2's stated
reason for requiring ablation rows at every budget, arriving as evidence rather
than as a rule this project took on trust.

## Open questions

Eleven, and three are new at P12.

**How many other compile options do not reach the sources they were written
for.** D-0047 was one line of CMake that had been wrong for five phases with no
observable consequence except a missing speedup and a vacuous test. `lib/Encoding`
and `lib/Dialect` are built by the same `add_mlir_library`, and this branch
checked only the one it was already looking at. **Nothing suggests another is
wrong**, and that is exactly what was true of this one. A cheap answer exists and
was not taken here because it is not P12's phase: read `build.ninja` for the
options each object library actually receives and compare against what its
`CMakeLists.txt` intends. The one thing that would make it unnecessary is a
project wide habit of asserting build properties from inside the artefact, which
is what `kernelsUseOpenMP()` is for one file.

**The `--mlir-timing` gap margin narrowed and one run is not a trend.** P11's
quiet runs measured 0.1577 and 0.1177 ms against D-0043's 0.2000 bound; P12's
measured 0.1856 on a machine with nothing else on it. There is no mechanism that
ought to couple a parallel simulation to a compile timing, because they are
separate processes run in sequence within a cell. That is an argument rather than
a measurement, and P13 has the next data point. **Do not respond to a red run at
that bound by widening it.**

**Whether the thread scaling table should be recorded per host or left as one
machine's number.** `experiments/kernel_threads.py` writes JSON carrying the host
context, and nothing consumes it: the numbers live in `docs/NUMBERS.md` labelled
as the 14700K's. The nightly runner would give a second point on a four vCPU
machine and the honest expectation is that it is close to 1.0, which is worth
recording precisely because it is unflattering. Left open rather than guessed at,
and the trigger for wiring it is in the activation section.

**D-0045 is open and P13 inherits it, and P12 deliberately did not touch it.**
The weight preload accounting is optimistic for narrow deep GEMMs, which is most
of this suite. Every performance claim taken from `simulated_cycles`,
`effective_macs` or `utilization` carries that, and `docs/DEFECT_LOG.md` says so
rather than a report footnote saying it. **A performance phase that changed a
charge would have made its own inertness claim unfalsifiable**, which is why the
cost model is untouched on this branch and why `git diff main..HEAD` is the
proof.

**SCALE-Sim runs from a patched install and there is no upstream fix.** The three
expression numpy 2 patch is applied by hand and recorded by tree hash. The
question for a later phase is whether to carry the patch upstream, pin a
container with an older interpreter, or wait: none is obviously right, and the
tree hash makes the current arrangement honest rather than tidy.

**The fp32 MAC coefficient fails Section 16.4's sanity check at 10.71 times.**
The cause is identified and is not this project, and the measured value is pinned
so that it moving is a failure. What is still open is whether an fp32 array is
the right thing to model at all: at P14 the integer kernels arrive and the
published int8 multiply is 0.2 pJ against 3.7, so the energy story changes shape
rather than scale.

**The clock is 1 ns and this project has no other reason to have one.** It was
needed because Accelergy's tables are indexed by cycle time, it was chosen before
looking at what it did to the sanity check, and it is recorded as an assumption.
A later phase that pins a clock for another reason should reconcile the two
rather than discover them separately.

**The external tools do not run in CI and the committed numbers are the only
record of them.** The step is off, says so in the run log, and asserts the tools
are absent so that it cannot quietly become a second copy of a step that is on.
The trigger to switch it on is the CI image gaining the tools, which is a P0
shaped decision.

**The measured per cell cost has not reached Section 2.** This is the carve out
above and it is the only part of the P10 and P11 gates this branch cannot close
by itself. Until the owner applies it, Section 2 says 15 seconds and 238 cells
while the repository says 1.27 seconds and 175, and a reader comparing them finds
a disagreement rather than a correction.

**`lenet_batched`'s tight budget cell at batch 1 does not exist, and would not
have been tight if it did.** Its recorded tight budget is 200832 against a batch
1 peak of 194592, so the cell would be a second copy of its default budget cell
wearing the tight label. ADR 0010 excludes it under the same rule that excludes
the six failures, and this is the one case where the rule does something other
than avoid a crash. Worth a look when P13 re-measures.

**The declared Python floor and the checkable one still differ by a minor
version.** `requires-python` says 3.11, mypy cannot go below 3.12. Unchanged by
this branch and unchanged in its consequences.

**The NDEBUG and sanitizer directories still cannot build anything that links
MLIR**, D-0031, ADR 0009. Unchanged. Note that P10's instrumentation is in
`lib/Pipeline`, which links MLIR, so it is not covered by the `ndebug` job; that
is the existing limit rather than a new one.

**Section 17.3a's fifth metamorphic relation still cannot be written.** `Pad` is
refused by name and `Slice` has no converter, and a test asserts the reason is
still true.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both, and repeated at every phase because a fact that stops being repeated
is a fact somebody eventually does not know.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, not once. It may be read, and only through a command that cannot
write. **Only the owner may retire it.**

The reason it exists on top of git history is that the two protect against
different failures. History protects against a bad commit. A second directory
protects against everything else, because if this rebuild goes wrong at any
point, deleting `~/npu-mlir-v2` returns the machine exactly to its pre build
state with no reasoning about reflogs required. That guarantee holds only while
the frozen copy is untouched.

## The external tool install, recorded

Started before anything else in the phase, because Section 23 says P11 blocks on
it entirely. **Total wall clock under two minutes**, and then it cost most of a
session anyway for reasons unrelated to download time.

| Step | Seconds |
|---|---|
| clone and install SCALE-Sim | 35 |
| clone and install Accelergy | 11 |
| the four plug ins, first pass | 44, two of them failing |
| `accelergy-table-based-plug-ins` with `--no-build-isolation` | 1 |
| CACTI submodule, `make -j4`, and the plug in install | 9 |

Every install used a constraints file built from `requirements-lock.txt`, so no
external tool could move a pin the 175 committed results were measured under.
**Nothing in the lock file moved.**

Three deviations, each recorded where it belongs rather than absorbed:

- `accelergy-table-based-plug-ins` imports `yaml` in its `setup.py`, which pip's
  isolated build environment does not have. Installed with
  `--no-build-isolation`. `docs/adr/0003-resolved-tool-matrix.md`.
- `accelergy-cacti-plug-in` copies a built CACTI binary its clone does not
  contain. Submodule initialised and `make` run first. Same record.
- **SCALE-Sim does not run under numpy 2**, so its install is patched.
  D-0044, and `scalesim_installed_tree_sha256` in every manifest.

And one deviation from the specification itself: Section 16.3 says to read the
example topologies from the **installed path** of the pinned version, and the
pinned version's wheel ships the package without its `topologies/` or `layouts/`
directories. They are read from the pinned source clone instead, and
`test_the_column_order_is_the_pinned_versions_own` reads the real file and
asserts the exporter's header still matches it, so the rule that matters is
enforced from the pinned source rather than dropped.

## Next phase

**P13, tiling, double buffering and layout.** Section 13.2 consuming the
`TilingInterface` that P1 implemented, `-npu-double-buffer` over the tokens
running **before** allocation per Section 5.1, `-npu-assign-layout` with the
inverse transpose fold, and the ZigZag cross check.

**What P12 leaves on P13's desk.**

- **D-0045 is now yours and P12 did not touch it.** The array's weight preload is
  charged once per instruction here and per fold by SCALE-Sim, worth about a
  factor of three on a narrow deep convolution, and it is the dominant term in
  the fragmentation column. P12 was required not to touch the cost model, because
  a performance phase that changed a charge would have made its own inertness
  claim unfalsifiable. `git diff main..HEAD` touches no cost model file.
- **The roofline still cannot fail against this cost model, and P13 is the phase
  that changes that.** `effective_macs` is defined as `cycles * peak`, so the
  compute branch is the kernel's own cycle count. Tiling is the first change that
  produces a charge no longer built from the traffic it moves.
- **The suite has more headroom than it did, and none of it is yours to spend on
  a slower measurement.** 1.17 seconds per cell against 1.27, a factor of twenty
  six against ninety minutes. The gain is a host wall clock and P13 adds cells if
  it adds an ablatable pass: `-npu-assign-layout`, `-npu-tile-to-scratchpad` and
  `-npu-double-buffer` all arrive at P13, so the ablatable set goes from 8 to 11
  and the ablation cells from 112 to 154, which is 217 cells rather than 175.
  Section 2's arithmetic and the carve out above both need re-deriving in that
  commit rather than after it.
- **`--mlir-timing` gap watch.** D-0043's bound is 0.2000 ms, principled and load
  sensitive. P11's quiet runs measured 0.1577 and 0.1177; P12's measured 0.1856
  on a machine with nothing else on it. The margin narrowed while this phase put
  twenty eight threads into the machine. There is no mechanism that ought to
  couple them, because the compile and the simulation are separate processes run
  in sequence within a cell, and that is an argument rather than a measurement.
  **Serialise your measurement runs and say so**, and if a run goes red at that
  bound, read it as the third data point rather than as noise.
- **The allocator growth benchmark is committed and P13 is the phase it starts
  being able to catch something.** P5 predicted it: "if P13's tiling pass makes
  functions an order of magnitude longer, this benchmark is already committed and
  will say so". The offset assignment scan is genuinely quadratic and its
  crossover is beyond 5000 buffers today. Longer functions move it toward the
  measured range. `--check` is written, rehearsed red, and **not wired into CI**;
  the trigger and the recipe are in the activation section above.
- **The kernel's team cap is `batch * outputChannels` and tiling changes both.**
  A tiled convolution has a smaller output tile per instruction and more
  instructions, so the collapsed loop shortens and the number of parallel regions
  grows. That is a wall clock question and not a numeric one, but it is the
  quantity `experiments/kernel_threads.py` measures, so re-run it and record the
  table beside P12's rather than assuming the 2.10 geometric mean survives.
- **Every P12 field is filled in all 175 cells from one run and a P13 run must
  keep them filled.** Unchanged from P11: the suite writes them from one run, a
  cell reused from disk raises, and `results_to_tex.py` refuses a table whose
  rows come from more than one commit.
- **Tiling is an exact transformation and its gate says goldens byte identical,
  exactly.** P12's inertness diff is the shape of proof that clause wants, and
  the throwaway that produced it is worth writing again rather than reaching for
  `git diff`: 95614 leaf fields classified, seven permitted movers named in
  advance, zero forbidden. Any movement from layout or double buffering is inside
  1e-6 and goes in `docs/BREAKING_CHANGES.md` **before** the commit that causes
  it.

## Next command

Push the branch, then dispatch the nightly.

```
git push -u origin phase/p12-performance
gh workflow run nightly.yml --ref phase/p12-performance
```

**The green condition for the push** is `pytest slow cells` running **1063
passed, 31 skipped** in the CI image, the `external cross validation` step
printing `external: confirmed absent`, and `regression-baseline: no drift.`

**Three lines to check before anything else, and the first is new.**

**The `build-and-test` job's `NPUSimulatorTests` should report 55 passed rather
than 54**, and the new one is `Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`.
**If it is red in CI, that is the most interesting result this branch can
produce** and it is not a regression: it would mean the CI image's clang finds
OpenMP and the same object library gap exists there in a form the directory scope
fix does not close. The gtest message names the mechanism and the file. Do not
relax the assertion; the whole point of it is that it is the only thing that can
tell a serial kernel from a parallel one from inside the process.

**The determinism assertion now means something in CI for the first time.** The
image has OpenMP 5.1 for the clang jobs and gcc with libgomp in coverage, and
until this branch the kernel it exercised had no parallel region in any of them.
A green `Determinism.OneThreadAndMaxThreadsAgreeBitwise` in CI is the first
evidence this project has that the convolution is bitwise stable across thread
counts **on a second toolchain**, which is worth more than the local run.

**The `pytest` arm's skip count.** 18 locally and 31 in the CI shape, because the
tests that need SCALE-Sim or Accelergy skip through `pytest.importorskip` there
and run here. A skip count of 18 in CI would mean those tests are being collected
and passed without a tool, which is not possible and would mean something else is
wrong.

**The green condition for the nightly** is the benchmark suite printing a per
cell cost and `run-benchmarks: inside the budget`. Record that cost in the
engineering log beside the 1.17 seconds this machine measured. **Expect it to be
higher and expect the gap to have narrowed**: a hosted runner is four vCPU, so
the kernel parallelism is worth much less there than the 2.10 geometric mean this
machine measured, and the external tools still run inside the suite. A nightly
figure close to P11's is the expected result and not a regression.
