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

**Last updated:** 2026-09-03.

## Current phase

**P11, external cross validation and the roofline.** Branch
`phase/p11-cross-validation`, cut from `main` at `d4e19a2`, which is the P10
merge. Seven commits. **Not pushed.**

| Commit | Subject |
|---|---|
| `53c50e5` | `feat(roofline): the physical bound of Section 16.6, and what it is worth` |
| `087ad3a` | `feat(scalesim): the cross validation of Section 16.3, with its gap named` |
| `6c54e26` | `docs(breaking): declare the P11 schema movement before it happens` |
| `17d63ed` | `feat(energy): Accelergy energy and area, and the schema movement declared` |
| `6afe2d8` | `fix(p11): a partial DRAM access is paid in full, and four tests that encoded P10` |
| `9bdb317` | `record: the suite and the baseline at the P11 schema, in one run each` |
| tip | `docs: hand off P11, with the prediction answered and two tools that exit zero on failure` |

**The commit order carries meaning and is the part to check first.** The roofline
lands first because Section 16.6 says to build the frame before the pictures. The
SCALE-Sim exporter lands second, which is what turns the P10 test asserting it
does not exist into the ordering proof P11's gate asks for. `6c54e26` is the
`docs/BREAKING_CHANGES.md` declaration and touches that file and nothing else;
`17d63ed` is the movement it declared; `9bdb317` is the re-record, touching only
recorded numbers. That is Section 17.6's declare then re-record in three commits
rather than two, because `git log` is the only thing that can tell a decision
from an explanation.

**Two intermediate commits are red, and that is what the governance costs.** At
`17d63ed` and `6afe2d8` the committed cells are still at `schema_version` 1 while
the reader writes and reads 2, so every test that loads a cell refuses. `9bdb317`
re-records them and the tip is green. The alternative was to fold the re-record
into the movement, which would have made the declaration unfalsifiable.

## Gate status

**Met.** Clause by clause, with the proof beside each.

| Clause | Status |
|---|---|
| The roofline bound computed for every model and every cell | met. 175 cells, 550 MAC bearing layers, `experiments/roofline.py`, exit 0 |
| `roofline_verdict` recorded | met. `at_or_above_bound` in all 175, with the branch that bound each layer beside it: 375 compute, 175 memory |
| Any cell above its bound investigated and written up as a finding rather than noted | met, and the finding is that **the check cannot fail against this cost model**. `effective_macs` is defined as `cycles * peak` and a transfer costs bytes over bandwidth plus a descriptor, so neither branch can bind. Both halves are asserted in `test/Python/test_roofline.py` and `docs/NUMBERS.md` carries the account. No cell is below its bound; the tightest layer is 0.000635 above, which is the issue overhead and nothing else |
| SCALE-Sim cycles and per layer breakdown on every cell | met. `scalesim_cycles` and `scalesim_cycles_per_layer` in all 175 |
| The skipped and approximated layers recorded | met. `scalesim_skipped` carries every operation with no systolic representation **and the analytical cycles it cost**, which is what lets the divergence read as a sum; `scalesim_approximations` carries the dilated layers with their effective extents |
| The divergence decomposed into named terms | met. Pooling, elementwise, uncovered DMA, dilation approximation, array fragmentation, double buffering, residual. The residual is zero on every cell **by construction**, because the terms are a partition rather than a fit, and the module says so rather than presenting it as a result |
| Accelergy energy and area per component on every cell | met. Three components, `energy_pj_per_component` and `area_mm2_per_component` in all 175, with the estimation plug in that answered recorded per component |
| Computed from **raw** `macs`, never from `effective_macs` | met. `counts_for` reads `simulation.macs` by name. `test_the_energy_path_never_sees_the_scaled_count` builds a cell whose `effective_macs` is four times its `macs` and checks which one the answer followed |
| Energy fields added to the baseline with a `schema_version` bump | met. Baseline 2 to 3, results 1 to 2, both declared in `docs/BREAKING_CHANGES.md` in a commit that touches only that file, moved in the next commit, re-recorded in one after that |
| The prediction quoted verbatim with its sha and answered | met. `f92de42`, quoted clause by clause in `docs/NUMBERS.md`, answered including where it is wrong, and not edited |
| The `git log` ordering proven in a test | met. `test_the_divergence_prediction_landed_before_the_exporter` asserts the prediction's landing commit is a **strict** ancestor of the exporter's, and refuses the same commit for both |
| Fusion re-argued in energy terms with numbers | met. Exactly 0.000 pJ on all seven models, with the counterfactual quantified beside it: up to 46.11 percent of `depthwise_separable`'s energy where the intermediate spilled. `docs/PASSES.md` and `docs/NUMBERS.md` |
| The declare then re-record step run for any intended baseline movement | met, in three commits, and the two red intermediates are named above rather than hidden |
| Dependencies pinned at the versions the manifests record | met. Six repositories by git sha in `docs/adr/0003-resolved-tool-matrix.md` and in every result manifest, plus `scalesim_installed_tree_sha256` because the sha does not describe the patched install |

## The Section 2 carve out, for the owner

**Section 2 says the per cell cost is "budgeted at 15 seconds per cell as a
planning figure" and that "the 15 second per cell figure is itself replaced by
the measured value at P10".** The specification file lives outside this
repository and the standing rule is that it is not edited from here, so the
replacement is recorded rather than applied. **This is the one item of P10's gate
that needs a hand other than this branch's.**

- **The measured figure is 1.27 seconds per cell**, over 175 cells in 3.70
  minutes, serially, on the 14700K under WSL2. Recorded in
  `experiments/results-runtime.json` and quoted in `docs/NUMBERS.md`.
- **It was 0.60 seconds at P10 and the difference is P11's external tools**,
  which now run inside the same suite: a SCALE-Sim invocation and, on the dilated
  cells, two, plus one Accelergy invocation per distinct scratchpad budget. The
  suite still finishes in under four minutes against ninety, so the factor in
  hand went from fifty to twenty four. The figure below is the one to quote.
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
  > equals **112**. Total **175 cells**. Measured at Phase P11 at **1.27 seconds
  > per cell** serially, which is 3.70 minutes for the whole suite including the
  > external cross validation tools, so **the stated budget of 90 minutes stands
  > with a factor of twenty four in hand**. The headroom is what pays for running
  > the cells serially, which the per cell timing objects of Section 16.1
  > require. It was 0.60 seconds at P10, before SCALE-Sim and Accelergy ran
  > inside the suite.

## What P11 measured, in one place

`docs/NUMBERS.md` is the ledger and is the file to read. Six things worth
repeating here.

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

## Verification output

Every command run at the tip of this branch, from `/home/elijah/npu-mlir-v2`, in
`~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 26 discovered, 26 passed |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | 12 passed |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 54 passed, 1 skipped |
| `build-ndebug/bin/NPUSimulatorTests` | 54 passed, 1 skipped |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `python -m pytest test/Python -q` | 988 passed, 18 skipped, 18 deselected |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **1006 passed, 18 skipped**. 957 at P10, plus 49 |
| `mypy` | no issues found in 25 source files |
| `black --check .` | 61 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 473 of 473 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `python scripts/check-reachability.py --skip-models` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date |
| `python experiments/roofline.py` | 175 cells, 550 layers, 175 memory bound and 375 compute bound, **every cell at or above its bound**, exit 0 |
| `python experiments/scalesim_export.py` | 175 cells, worst whole model divergence **-87.14%** on `dilated_stack-O0-tight-n1-fp32-normal` at coverage 0.711, exit 0 |
| `python experiments/accelergy_energy.py` | 175 cells at 45 nm, per MAC **49.286 pJ** against a published 4.60, a factor of **10.71**, exit 0 |
| `python scripts/patch-scalesim.py --check` | every edit in place, exit 0 |
| `python experiments/run_benchmarks.py --force` | **175 cells, 3.70 minutes, 1.27 s per cell**, inside the budget, exit 0 |
| `bash scripts/regression-baseline.sh --check` | **no drift, exit 0** |
| `bash scripts/coverage.sh 85 93 14 58` | C++ **86.5** PASS, branch 76.9; per tree **93.2337 / 14.6597 / 73.1302** PASS, exit 0 |
| the whole suite in an environment with neither external tool | **998 passed, 29 skipped, 0 failed**, mypy clean, `coverage.sh` PASS at **93.2337 / 14.6597 / 58.4488** |
| the same environment with `NPU_EXTERNAL_TOOLS=1` | the guard **fails** naming the variable rather than skipping, which is the third branch of `tools.py`'s policy |
| `git status --short` | empty |
| `git log -p main..HEAD` grepped for tooling and authorship traces | 0 matches, case sensitive with word boundaries |
| the same diff grepped for em and en dashes | 0 matches |

**The suite grew by 49 pytest tests**, in three new files: `test_roofline.py`,
`test_scalesim_export.py` and `test_accelergy_energy.py`. Four existing tests
changed, all of them because they encoded the P10 state, and each change is
recorded in `6afe2d8`'s message rather than folded into a larger commit.

**All 21 golden tensors are byte identical to P10's**, which is what the
`docs/BREAKING_CHANGES.md` entry said would happen. The baseline moved in shape
and not in value.

**The external tool steps are off in CI and say so in the run log**, per Section
19.0's rule that silence and success must not look alike. The step asserts the
tools are **absent**, so it cannot quietly become a second copy of a step that is
on. The trigger to switch it on is the CI image gaining them.

**Python coverage is measured over three trees now and gated per tree**, which
is the P8 rationale for one package root being superseded rather than
contradicted: `scripts/` is 955 statements and `experiments/` 1439, and both
carry real logic at P11 where `scripts/` carried entry points at P8.

| Tree | CI shape | Developer machine | Threshold |
|---|---|---|---|
| `python/npu_frontend` | 93.2337 | 93.2337 | **93** |
| `scripts` | 14.6597 | 14.6597 | **14** |
| `experiments` | 58.4488 | 73.1302 | **58** |

Three things about that table are load bearing and are recorded where the
thresholds are defined as well as here.

- **Subprocess coverage is wired**, through `COVERAGE_PROCESS_START` and
  `parallel = true`. It is worth 1.8 points on the frontend alone, 91.5561
  before and 93.3687 after, which were lines being executed and not counted.
  Stale `.coverage.*` files are erased first, which is D-0037 on the Python side.
- **`experiments/` differs by 14.7 points between the two environments**, because
  `scalesim_export.py` and `accelergy_energy.py` only execute where the tools do.
  The gate is set from what CI can execute; setting it from the developer figure
  would make CI red for having less installed.
- **`scripts` at 14 is a real number and a weak gate**, and the prose says so
  where the threshold lives. Five of its seven files measure exactly 0.0, because
  they are driven by shell scripts and CI steps rather than by pytest, so the
  figure is close to a statement about `regression_baseline.py` alone. It is
  gated as a ratchet on the part pytest can see.

**The coverage job got slower and by how much is recorded**: its pytest phase
went from 248.45 to 327.57 seconds with the tools present, plus 32 percent, which
is the tracer over two more source trees. The whole script is 351 seconds locally
and 246 in the CI shape. C++ is unmoved at 86.5, which is right: this phase added
no C++.

**The authorship grep is worth one sentence, because it caught itself once.** The
row above it originally named the thing it searches for, which made the row a
match for its own pattern and turned a clean result into a count of one. It is
worded to describe the check rather than to quote it. A case insensitive form of
the same grep also reports four matches on this branch, all of them the substring
inside `wallMs`, which is why the recorded run is the case sensitive one with
word boundaries.

**The suite grew by 86 pytest tests and one lit test.** No existing test changed
its result and no recorded number moved: both baseline re-records on this branch
touch `git_sha`, the suite counts and the test names, and nothing else. All 21
golden tensors are byte identical to P9b's.

**Python coverage headroom widened from 0.54 to 0.97 points**, 90.97 against a
threshold of 90. It was 0.49 at P9 and 0.61 at P8. The threshold stays at 90.

## Activation proofs and rehearsal recipes

**This branch activates one CI step and one CI job.** Both were rehearsed under
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

**Two new at P11, and neither is in this project's own code.** One is in an
external tool and one is in this project's cost model, found by the external
tool, which is what cross validation is for.

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

Eight, and five are new.

**D-0045 is open and P13 inherits it.** The weight preload accounting is
optimistic for narrow deep GEMMs, which is most of this suite. Every performance
claim taken from `simulated_cycles`, `effective_macs` or `utilization` carries
that, and `docs/DEFECT_LOG.md` says so rather than a report footnote saying it.

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

## The orchestrator's runbook

Three things, in this order.

### 1. Push the branch

```
git push -u origin phase/p10-measurement
```

**This has happened three times and all three were red, on three different
defects. Each run got further than the last.**

Run 33559636835 was D-0041: `fetch-depth: 1` meeting the first phase whose tests
resolve historical shas. Run 33571635111, on the commit that fixed it, was
D-0042: the history was fetched and the commits were present, and git was
refusing the repository as dubiously owned while this project read its exit 128
as "the commit is absent". Run 33575891610, on the commit that fixed that, turned
**`build-and-test` green including every D-0041 and D-0042 test**, and was red in
the coverage job alone on D-0043.

**All three are fixed.** D-0041 and D-0042 are additionally **proven in CI** by
that third run rather than only rehearsed. D-0043 is rehearsed against
`build-coverage`, which is the build it appears in. The fourth push is the one
that confirms it.

**What to expect.** `lint`, `sanitizers`, `coverage` and `ndebug` green.
`build-and-test` green with two things to read: the new `pytest slow cells` step
should report **nine** tests carrying the marker and then run 957 passed 18
skipped, and `regression-baseline --check` should report no drift with the suite
table showing `check-npu 26` and `pytest 957`.

**The one line that is genuinely new information** is the slow cells step's
count. It is the first time in this project's history that CI has executed
`test_every_model_imports_at_a_second_seed` on any model. If one of those seven
goes red, that is D-0040 having been worth finding rather than a regression this
branch caused, and the failure is a first measurement rather than a break.

**Expect the oracle notes** from `--check` if the runner hardware differs from
this machine's, and none if it matches. Either is green.

### 2. Dispatch the nightly rather than waiting for 03:30 UTC

```
gh workflow run nightly.yml --ref phase/p10-measurement
```

The `full-matrix` job is new. **A dispatch of it was in flight against the tree
before D-0041 was fixed and will have gone red the same way**, because its
checkout was at the same default and the harness exits before the first cell
without the prediction's landing commit. Re-dispatch after this branch is pushed;
that run is the first meaningful one.

It configures, builds, runs the whole pytest suite including the slow cells, then
runs the 175 cell benchmark suite against its 90 minute budget.

- **Read the per cell cost off it and record it in `docs/ENGINEERING_LOG.md`
  beside the local one.** A hosted runner is four vCPU and the suite is serial,
  so expect several times 0.60 seconds and still far inside the budget. That
  second figure is what makes the budget claim about more than one machine.
- The job uploads its cells as an artifact. It writes to `RUNNER_TEMP` and never
  over `experiments/results/`, so a green run leaves the committed numbers alone.
- If it goes red, the step prints the two readings a red run can have and they
  want different responses. Do not raise the budget.

### 3. The `pull_request` proof for the new step

Section 19.1 wants an activation proven red under the trigger it will run under.
The `pytest slow cells` fault is above and is cheap: remove every
`@pytest.mark.slow` in one commit on a scratch branch, open a draft pull request,
show the count guard red, close it unmerged and delete the branch. One pull
request covers it.

## Next phase

**P12, performance.** Prove the sweep line's growth against the committed
benchmark, parallelise the convolution kernel, and confirm the suite runtime.
**The phase must be numerically inert**: goldens byte identical, cycles and DRAM
bytes and instruction counts unmoved, only host wall clock changing, and
`CHANGELOG.md` saying so explicitly so nobody reads the speedup as a simulated
result.

**What P11 leaves on P12's desk.**

- **The suite budget has less headroom than it did and the figure to quote moved.**
  1.27 seconds per cell against P10's 0.60, because SCALE-Sim and Accelergy now
  run inside the suite. 3.70 minutes against ninety. P12's gate re-measures the
  budget and this is the number it starts from.
- **Do not run the benchmark suite beside the external tools.** The first P11
  re-record died at cell 74 on the `--mlir-timing` cross check with a gap of
  0.2157 ms against a bound of 0.2000, while SCALE-Sim was running in another
  process. Two quiet runs measured 0.1577 and 0.1177. The bound is D-0043's and
  is principled; it is also load sensitive, and P12 parallelises a kernel, which
  is exactly the change that makes the machine busier.
- **`--skip-external` exists and is not a way to make a red run quiet.** It
  records the P11 fields as null with a reason naming the flag, which Section
  16.4 distinguishes from a tool that could not be found. A P12 run that only
  cares about wall clock may use it; a recorded suite may not.
- **Every P11 field is now filled in all 175 cells and a P12 run must keep them
  filled.** The suite writes them from one run, and a cell reused from disk
  raises rather than being written with the fields it no longer has. That is
  deliberate: `results_to_tex.py` refuses a table whose rows come from more than
  one commit.
- **The roofline is the regression bound P12 has to stay above**, and it is worth
  saying which way round that is. P12 is numerically inert, so nothing it does
  should move a cycle count at all. If one moves, the roofline is the check most
  likely to notice.
- **D-0045 is open and belongs to P13, not P12.** P12 must not touch the cost
  model. A performance phase that changed a charge would make its own inertness
  claim unfalsifiable.

**The frozen tools are installed and pinned** in `~/npu-external/`, recorded by
git sha in `docs/adr/0003-resolved-tool-matrix.md`. `scripts/patch-scalesim.py
--check` reports whether the numpy 2 patch is still in place, and the verification
matrix runs it.

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

## Next command

Push the branch, then dispatch the nightly.

```
git push -u origin phase/p11-cross-validation
gh workflow run nightly.yml --ref phase/p11-cross-validation
```

**The green condition for the push** is the `pytest slow cells` step running
1006 passed and 18 skipped, the new `external cross validation` step printing
`external: confirmed absent`, and `regression-baseline: no drift.`

**Three lines to check before anything else.**

**The `external cross validation` step must print `confirmed absent`.** If it
errors instead, the CI image has gained SCALE-Sim, and the right response is to
switch the step on and run the tools rather than to relax the assertion. The
recipe is in the step's own comment.

**The `pytest` arm's skip count.** 18 locally and it should be more in CI,
because the tests that need SCALE-Sim or Accelergy skip through
`pytest.importorskip` there and run here. A skip count of 18 in CI would mean
those tests are being collected and passed without a tool, which is not possible
and would mean something else is wrong.

**And `regression-baseline --check`, which recomputes energy against the
recorded coefficients rather than against Accelergy.** If it reports energy
drift, read the manifest difference first: a coefficient that moved shows up
there, and a hundred cell differences with no manifest change means the counts
moved, which is the thing this field exists to catch.

**The green condition for the nightly** is the benchmark suite printing a per
cell cost and `run-benchmarks: inside the budget`. Record that cost in the
engineering log beside the 1.27 seconds this machine measured, and expect it to
be higher: the nightly runner is slower and the external tools now run inside the
suite.
