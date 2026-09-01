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

**P10, measurement.** Branch `phase/p10-measurement`, cut from `main` at
`fe43ee4`, which is the P9b merge. Nine commits, **none pushed**.

| Commit | Subject |
|---|---|
| `847837c` | `feat(pipeline): the PassInstrumentation of Section 16.2, on the manager that runs` |
| `f92de42` | `feat(experiments): the prediction mechanism of Section 17.8, with two real entries` |
| `d4210f3` | `feat(experiments): the result schema of Section 16.1 and the harness that fills it` |
| `a203995` | `chore(results): record the 175 cells, in their own commit after the code` |
| `1889089` | `docs: the README table, the numbers ledger, and the two traceability tests` |
| `16cf7aa` | `docs(passes): the measured ablation delta Section 12 has asked every entry for` |
| `d318a49` | `ci: the slow cells step and the nightly full matrix, both at the phase 19.0 names` |
| `5401d39` | `chore(baseline): re-record the suite counts, in its own commit and nothing else` |
| tip | `docs: hand off P10, with the number Section 2 asks for and cannot be given here` |

**The tip carries this table's own last row**, so it is named by subject rather
than by a sha it cannot know, which is what P9 and P9b both did for the same
reason.

**The ordering carries two of this project's rules and is the part to check
first.** `f92de42` lands the prediction **before** `a203995` records the cells
that name it, which is the whole of Section 17.8 and is what the ancestor test
asserts. `d4210f3` lands the harness before `a203995` records what it measured,
because a result is committed after the code it measures, and the suite was run a
second time at `d4210f3` so that every `manifest.git_sha` names the commit that
contains the harness rather than the one before it.

## Gate status

**Met.** Clause by clause, with the proof beside each.

| Clause | Status |
|---|---|
| Before and after operation counts per pass computed by the `PassInstrumentation` in `runBeforePass` and `runAfterPass`, not by any flag | met. `lib/Pipeline/PassStats.cpp`. `test_the_counts_chain_across_the_pipeline` asserts one pass's after equals the next pass's before across the anchor change, which only a running pipeline produces |
| Wall clock from the same instrumentation cross checked against `--mlir-timing` | met. Both flags on one invocation, so the two clocks describe the same execution. Worst gap over 175 cells and 1750 trials: **0.1968 ms** |
| A pass present in the pipeline but absent from the JSON raises, shown | met. `test_a_pass_absent_from_the_json_raises`, on a doctored file, plus ten more that each drop one field |
| Every ablatable `-O2` pass has an ablation row at every budget, set read from the driver at run time | met. 8 passes times 7 models times 2 budgets equals 112 rows. `ablatable_passes(2)` is the only source of the set |
| Every schema field present in every result file, `null` plus a reason where a later phase fills it | met. `test_every_committed_result_validates` over all 175, and three refusals shown on doctored input |
| The full matrix passes | met. 426 tests, 24.8 seconds |
| Suite runtime measured against the 90 minute budget, run fails if exceeded | met. **175 cells, 1.76 minutes, 0.60 s per cell.** Failure branch rehearsed at `--budget-minutes 0`, exit 1 |
| The measured per cell cost replaces the 15 second planning figure | **carved out, see below.** The number is 0.60 seconds; the specification is outside this repository and is not edited from here |
| At least one prediction through the full path with its ancestor test green | met. `p10-ablation-deltas`, named by 126 of 175 cells, ancestor test green, and the check shown refusing a non ancestor |
| The README table and both traceability tests green | met, and both proven red on a deliberate fault and restored |

## The Section 2 carve out, for the owner

**Section 2 says the per cell cost is "budgeted at 15 seconds per cell as a
planning figure" and that "the 15 second per cell figure is itself replaced by
the measured value at P10".** The specification file lives outside this
repository and the standing rule is that it is not edited from here, so the
replacement is recorded rather than applied. **This is the one item of P10's gate
that needs a hand other than this branch's.**

- **The measured figure is 0.60 seconds per cell**, over 175 cells in 105.7
  seconds, serially, on the 14700K under WSL2. Recorded in
  `experiments/results-runtime.json` and quoted in `docs/NUMBERS.md`.
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
  > equals **112**. Total **175 cells**. Measured at Phase P10 at **0.60 seconds
  > per cell** serially, which is 105.7 seconds for the whole suite, so **the
  > stated budget of 90 minutes stands with a factor of fifty in hand**. The
  > headroom is what pays for running the cells serially, which the per cell
  > timing objects of Section 16.1 require.

## What P10 measured, in one place

`docs/NUMBERS.md` is the ledger and is the file to read. Four things worth
repeating here.

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

Every command run at the tip of this branch, `5401d39`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 26 discovered, 26 passed. 25 at P9b, plus `test/Pipeline/pass-stats.mlir` |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | 12 passed |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 54 passed, 1 skipped |
| `build-ndebug/bin/NPUSimulatorTests` | 54 passed, 1 skipped |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `python -m pytest test/Python -q` | 945 passed, 18 skipped, 9 deselected |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **954 passed, 18 skipped**. 871 at P9b, plus 83 |
| `mypy` | no issues found in 23 source files |
| `black --check .` | 53 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 465 of 465 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `python scripts/check-reachability.py --skip-models` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date |
| `python experiments/run_benchmarks.py` | 175 cells, 1.76 minutes, 0.60 s per cell, inside the budget, exit 0 |
| `bash scripts/regression-baseline.sh --check` | **no drift, exit 0**, 2 minutes 17 seconds |
| `bash scripts/coverage.sh 85 90` | C++ **86.5** PASS, Python **90.97** PASS, exit 0 |
| `git status --short` | empty |
| `git log -p main..HEAD` grepped for tooling and authorship traces | 0 matches, case sensitive with word boundaries |
| the same diff grepped for em and en dashes | 0 matches |

**The authorship grep is worth one sentence, because it caught itself once.** The
row above it originally named the thing it searches for, which made the row a
match for its own pattern and turned a clean result into a count of one. It is
worded to describe the check rather than to quote it. A case insensitive form of
the same grep also reports four matches on this branch, all of them the substring
inside `wallMs`, which is why the recorded run is the case sensitive one with
word boundaries.

**The suite grew by 83 pytest tests and one lit test.** No existing test changed
its result and no recorded number moved.

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
at 954 passed, 18 skipped.

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

### 3. The two traceability tests, both faults

**Fault A**, a hand typed number: `resnet_block`'s cycle count changed from 1626
to 1499 in the README table. Red, naming the number and the row it sits in.

**Fault B**, stale provenance: `npuResultsSha` pointed at a commit that is not
the results'. Red, saying the results were re-recorded and the macros were not,
with the staleness check red beside it. Both restored.

**Which trigger these need:** none. They are pytest tests and run in every job
that runs pytest.

## Defects

Two, both found by this branch and fixed in it.

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

**No defect was found in the compiler.** Every failure this phase produced was in
the measuring apparatus or in a claim about it, which is what a measurement phase
should expect and is worth saying rather than leaving as an absence.

## The prediction, adjudicated

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

Five, and two are new.

**The measured per cell cost has not reached Section 2.** This is the carve out
above and it is the only part of P10's gate this branch cannot close by itself.
Until the owner applies it, Section 2 says 15 seconds and 238 cells while the
repository says 0.60 seconds and 175, and a reader comparing them finds a
disagreement rather than a correction.

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

**This has happened once and it went red**, run 33559636835, on D-0041: eight
failures, all in P10 test files, all of them the `actions/checkout` default of
`fetch-depth: 1` meeting the first phase whose tests resolve historical shas.
Fixed on this branch, in the three jobs that run the suite and in the code that
was answering the question wrongly rather than refusing it. **The next push is
the one that confirms it.**

**What to expect.** `lint`, `sanitizers`, `coverage` and `ndebug` green.
`build-and-test` green with two things to read: the new `pytest slow cells` step
should report **nine** tests carrying the marker and then run 954 passed 18
skipped, and `regression-baseline --check` should report no drift with the suite
table showing `check-npu 26` and `pytest 954`.

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

**P11, external cross validation and the roofline.** The roofline of Section 16.6
first, then Sections 16.3 and 16.4.

**What P10 leaves on P11's desk, and most of it is already built.**

- **The prediction P11 needs already exists and is already an ancestor.**
  `experiments/predictions/p11-scalesim-divergence.md` landed at `f92de42`,
  before `experiments/scalesim_export.py` exists at all, which is the strongest
  form of that claim available. P11's gate asks for the divergence prediction to
  predate the first SCALE-Sim number; it predates the exporter. A test asserts
  the exporter does not exist yet, so the day it does, that test is the prompt to
  check the ordering rather than assume it.
- **Every field P11 fills is already in the schema, carrying `null` and a reason
  naming P11.** `roofline_bound_cycles`, `operational_intensity`,
  `roofline_verdict`, `scalesim_cycles` and both coverage fractions, `energy_pj`,
  `area_mm2`, `technology_node`, `tool_shas` and `registered_estimators`. Filling
  one means replacing a null and **deleting its `_null_reason` sibling**; the
  validator refuses a field carrying both, so a half done fill is red.
- **`macs` is raw and is the only one Accelergy may see.** `effective_macs` sits
  beside it with `utilization` and `delta` so the charge can be reconstructed.
  Section 5.5 forbids the energy path from seeing the scaled figure, the schema
  keeps them apart, and `docs/NUMBERS.md` says so where the two would first get
  confused.
- **`experiments/results_to_tex.py` refuses to generate if the committed cells
  were measured at more than one commit.** P11 re-records the whole suite when it
  adds fields, in one run, or that refusal fires.
- **The roofline check fails a cell below its bound**, per Section 16.1, and
  `roofline_verdict` records that the check ran. Leaving it null rather than
  writing `below_bound` is the difference between a check that has not run and a
  check that passed, which the null reason says in those words.

**Accelergy and its plug ins install from source at pinned git shas and are the
longest lead dependency in the project.** Section 23 says P11 blocks on them
entirely and to start that install before the phase does. Nothing in P10 started
it.

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

## Next command

Push the branch, then dispatch the nightly, which is the job that has never run.

```
git push -u origin phase/p10-measurement
gh workflow run nightly.yml --ref phase/p10-measurement
```

**The green condition for the push** is the `pytest slow cells` step reporting
nine tests and running 954 passed, followed by `regression-baseline: no drift.`

**The green condition for the nightly** is the benchmark suite printing a per
cell cost and `run-benchmarks: inside the budget`. Record that cost in the
engineering log beside the 0.60 seconds this machine measured.
