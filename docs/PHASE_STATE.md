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

**Last updated:** 2026-08-31.

## Current phase

**P8, the walking skeleton and the safety net.** Branch
`phase/p8-walking-skeleton`, cut from `main` at `7a4d060`, which is the P7
merge. Twelve commits. The branch has been pushed once and its first CI run is
recorded under "Next command"; the twelfth commit is the fix that run produced
and is not pushed yet. The twelfth is also the one that carries this table, so
it is the branch tip and is named by subject rather than by a sha it cannot
know.

| Commit | Subject |
|---|---|
| `b1ce3c9` | `feat(pipeline): the -O0 level, described once, in lib/Pipeline` |
| `af73bd7` | `feat(simulator): the statistics leave npu-sim as data, not as text` |
| `5225b5b` | `feat(driver): npu-compile, from ONNX to a .nbin at -O0, in four stages` |
| `d19fbb1` | `test(e2e): the walking skeleton walks, checked against two oracles` |
| `2906a25` | `test(oracles): the metamorphic relations, and a subgraph nothing reads` |
| `4cae9ab` | `docs(adr): the per model tight scratchpad budgets, measured and frozen` |
| `8c9fa4a` | `feat(scripts): the regression baseline of Section 17.6, and its --check` |
| `852b3f5` | `feat(scripts): the reachability check in full, with a mechanical simulation layer` |
| `68e8984` | `feat(scripts): coverage measured, and the real thresholds set` |
| `93679c3` | `fix(build): say out loud what the NDEBUG and sanitizer directories are for` |
| `6cf6c1a` | `docs: record the P8 defects and hand off the phase` |
| tip | `fix(coverage): one rule for finding a built binary, and one policy about not` |

**The commit order is the roadmap's.** The `-O0` pipeline comes first because
nothing can name a level until one exists. The simulator's JSON statistics come
second because the driver reads them and Section 10.2 forbids reading them any
other way. The driver comes third, then the end to end matrix that drives it,
then the oracles that need a pipeline to act on. The tight budgets come before
the baseline because the baseline's tight budget cells use them. The baseline
comes before the reachability and coverage work because the roadmap puts the net
after the run it freezes.

**Every commit was checked out and built on its own.** A worktree at
`/tmp/p8check`, since removed, walked the first ten in order, ran
`cmake -S . -B build` and `ninja -C build -j6` at each, and then
`ninja -C build check-npu`. All ten built clean and all ten reported 20 of 20.
The eleventh and twelfth touch documentation and the Python suite and build
nothing new. A commit sequence nobody has bisected is a sequence that only reads
as one.

**The end to end pipeline worked the first time it was run**, matching
onnxruntime to 2.98e-8 on LeNet with no debugging. That is recorded because it is
the phase's headline and because everything else in this file is about something
other than the pipeline. The four stages P4 through P7 landed with their own
tests fitted together.

## Gate status

The P8 gate is the roadmap entry's, parsed clause by clause. Every item is **met
locally**; the CI runs that Section 19.1 also requires are the orchestrator's and
are listed under "Next command". Item by item, with the proof.

| # | Gate clause | Status | Proof |
|---|---|---|---|
| 1 | every model matches onnxruntime at `-O0` | PASS | `test_the_simulated_answer_matches_onnxruntime`, 70 cells, all seven models. Worst absolute 4.77e-06, worst relative 8.08e-07 |
| 2 | both tolerances asserted separately | PASS | `assert_both_bounds` makes two assertions, not one `assert_allclose`. Section 17.4 says the combined predicate lets either bound hide the other's failure and this is that rule obeyed |
| 3 | all five input classes | PASS | `npu_frontend.input_classes`, seeded per cell from `crc32(model:batch:class)` so a failing cell's input reconstructs from its own name. `test_input_classes.py` asserts no two classes produce the same array |
| 4 | both batch sizes | PASS | `generate_model(..., batch=)`, with a test that the weights and node counts do not move with the batch. 1 and 4, over every model |
| 5 | the baseline records `-O0` only | PASS | `test_the_committed_baseline_records_minus_o_zero_only`. 14 cells, seven models times two budgets, one level |
| 6 | `schema_version` set | PASS | 1. `--check` refuses a version it does not recognise rather than reading a later field as a regression from zero, asserted by `test_an_unknown_schema_version_is_refused_rather_than_guessed` |
| 7 | energy and per level fields explicitly absent | PASS | `absent_fields` names `energy` as P11 and `per_level` as P9, and a test asserts no cell carries an energy key |
| 8 | `--check` reports zero drift on unchanged code | PASS | Output below. `regression-baseline: no drift`, exit 0 |
| 9 | and demonstrably fails on a deliberate cost model perturbation, both outputs shown | PASS | Output below. 32 differences across two suites and every cell's cycles, exit 1 |
| 10 | the reachability check passes with an empty exemption block, or every gap is a dated exemption naming its phase | PASS, second form | Full mode, five layers. Every **imported computation** operation meets all five with no exemption of any kind. Two dated exemptions naming P9, both structural, both the same fact |
| 11 | `yield` and `fused_op` classified as structural per law 2 | PASS | Read from the ODS description, not from a table in the script. `note: structural: npu.fused_op, npu.yield` |
| 12 | the metamorphic relations pass on every model | PASS | Four of Section 17.3a's five, exact agreement on every applicable cell. The fifth cannot be written and the reason is asserted rather than asserted once |
| 13 | a dead subgraph changes neither the outputs | PASS | Bit identical, every model, every input class |
| 14 | nor the `-O2` instruction count | PASS, resolved | See "How the `-O2` clause was read" below |
| 15 | the per model tight budgets recorded | PASS | `docs/adr/0008-per-model-tight-scratchpad-budgets.md`, dated, with the measurement and two recorded deviations |
| 16 | CI proven red for all four fault classes with URLs recorded | PASS | Pull requests 9 through 12, one fault each under the `pull_request` trigger, all red at the predicted job and step; four red URLs and the green restore URL in `docs/ENGINEERING_LOG.md` |
| 17 | coverage measured and the real thresholds set | PASS | C++ 86.1 percent, threshold 85. Python 90.6 percent, threshold 90. Branch coverage reported for the allocator and the decoder. Proven in CI's own environment after D-0032 |

### How the `-O2` clause was read, and why

The gate asks that a dead subgraph change neither the outputs nor the **-O2**
instruction count. `-O2` arrives at P9. Worse, at `-O0` the second half is not
merely unmeasurable, it is **false by construction**: Section 12 puts every pass
that removes anything at `-O1` and above, so a dead subgraph at `-O0` must change
the count.

The resolution, recorded rather than chosen quietly:

- **The outputs half is asserted now**, bit identical, on every model and every
  input class. It is the stronger half and it needs no level above `-O0`.
- **The count half is asserted at every level whose pipeline eliminates dead
  code**, and which levels those are is read out of the compiler.
  `PassEntry` carries an `eliminatesDeadCode` property under the same rule as
  `ablatable`: a missing one is a build error.
  `compile.levels_that_eliminate_dead_code()` reads it through
  `npu-opt --npu-describe-pipeline`. The set is empty at P8 and fills itself at
  P9 with no edit to the test.
- **`test_no_level_this_compiler_builds_eliminates_dead_code` asserts the
  emptiness**, so the vacuous parametrization is declared rather than left to be
  noticed. It goes red at P9, which is correct: the phase that makes the claim
  true is the phase that deletes the assertion.
- **Beside it, the P8 form of the same check**, which is just as falsifiable in
  the other direction: at `-O0` the count grows by exactly the three instructions
  the injection brought, and by no more. Four would mean the injection cost
  something it did not declare; two would mean a pass nobody registered removed
  something.

### The two exemptions, and why they are not a weakened gate

`npu.fused_op` and `npu.yield` are missing the model layer. They are created by
`-npu-fuse-ops`, which Section 12 puts at `-O2` and which lands at P9, so no
model's IR can hold one at P8. The gate's own wording allows this: "with an
**empty** exemption block ... **or** every gap is a dated exemption in
`docs/EXEMPTIONS.md` naming its phase".

Nothing about either operation is unfinished. `-npu-lower-to-npuisa` flattens
the region, the lit suite has a case for it, the ISA description records both as
reaching the encoder by elimination, and the simulator needs no kernel for either
because neither survives to the instruction stream. What is missing is a
**producer**. The commit that lands `-npu-fuse-ops` deletes both entries, or the
check goes red at the next run.

### Verification output

Every command below was run on this branch at `93679c3`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 20 discovered, 20 passed. Nineteen at P7, plus this phase's `test/Pipeline/opt-levels.mlir` |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed |
| `build/bin/NPUAllocatorTests` | 29 tests, 29 passed |
| `build/bin/NPUEncodingTests` | 77 tests, 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 55 tests, 54 passed, 1 skipped |
| `build-ndebug/bin/NPUSimulatorTests` | 54 passed, 1 skipped. See D-0031 for what this directory may and may not build |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build-fuzz/bin/NPUSimulatorTests` under ASan and UBSan | 54 passed, 1 skipped, exit 0 |
| `build-fuzz/bin/NPUEncodingTests` under ASan and UBSan | 75 passed, 2 skipped, exit 0 |
| `python -m pytest test/Python -q` | 488 passed, 7 skipped, 7 deselected, exit 0 |
| `python -m pytest test/Python -q -m 'slow or not slow'` | 495 passed, 7 skipped, exit 0. 180 at P7, plus this phase's 322 |
| `mypy` | no issues found in 19 source files |
| `black --check .` | 39 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 241 of 241 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 28 IR files written |
| `python scripts/check-reachability.py` | pass, `layers checked: import, lowering, encoding, simulation, model`, 2 exemptions in force |
| `python scripts/check-reachability.py --skip-models` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `bash scripts/regression-baseline.sh --check` | no drift, exit 0 |
| `bash scripts/coverage.sh 85 90` | C++ 86.1 PASS, Python 90.6 PASS, exit 0. Also run in a worktree with no `build/`, which is the coverage job's own shape |
| `git status --short` | empty |

**Five gtest binaries and one lit suite, unchanged in number.** What grew is
pytest, from 180 to 495, which is where the end to end matrix, the two oracles,
the driver's contract, the tight budgets and the tool discovery rule all live.

### The baseline, recorded and checked

`test/baseline/baseline.json`, `schema_version` 1: 8 suites with their pass, fail
and skip counts and their full test name lists, the git sha, the tool versions,
`absent_fields`, and 14 cells. `test/baseline/golden/` holds seven `.npy`
tensors, one per model at `-O0`.

**The recorded `git_sha` is `6cf6c1a` and the branch tip is later**, which is
correct rather than stale. Section 16.1 states the same fact for result files: a
result is committed **after** the code it measures, so it can never carry the sha
of the commit that contains it. `--check` at the tip reports no drift, which is
the claim that matters.

**It was re-recorded once, in the commit that fixed D-0032**, because that
commit adds six tests and the baseline records the test names of every suite.
The diff is the point rather than a nuisance: `--check` reported seven
differences, the pass count and the six added names, and **no cell and no golden
moved**. The suite's composition changed and none of its numbers did, which is
exactly the distinction the two halves of this file exist to draw. Section
17.6's rule that a re-record goes in its own commit after a
`docs/BREAKING_CHANGES.md` declaration is about an intended movement in the
**numbers**; there was none here, so there was nothing to declare.

Zero drift on unchanged code:

```
regression-baseline --check

  recorded at 6cf6c1a143c9, checked at 6cf6c1a143c9
  14 cells, 7 golden tensors
  suite NPUAllocatorTests      29 passed    0 failed    0 skipped
  suite NPUEncodingTests       76 passed    0 failed    1 skipped
  suite NPUInterfaceTests      23 passed    0 failed    0 skipped
  suite NPUSimulatorTests      54 passed    0 failed    1 skipped
  suite NPUTilingTests         12 passed    0 failed    0 skipped
  suite check-npu              20 passed    0 failed    0 skipped
  suite dash-lint               2 passed    0 failed    0 skipped
  suite pytest                495 passed    0 failed    7 skipped

regression-baseline: no drift.
```

### The deliberate cost model perturbation, and what the net caught

The fault, one constant, `kDramBandwidthBytesPerCycle` from 16 to 15:

```bash
sed -i 's/kDramBandwidthBytesPerCycle = 16.0;/kDramBandwidthBytesPerCycle = 15.0;/' \
    include/NPU/Simulator/CostModel.h
```

`--check` output with the fault in place, exit 1, abridged only in the cell list:

```
  suite NPUSimulatorTests      52 passed    2 failed    1 skipped
  suite pytest                472 passed    2 failed    7 skipped

  32 differences:

    suite NPUSimulatorTests: passed 54 -> 52
    suite NPUSimulatorTests: failed 0 -> 2
    suite pytest: passed 474 -> 472
    suite pytest: failed 0 -> 2
    cell conv_bn_relu_stack-O0-default: cycles 1372.5 -> 1389.5
    cell conv_bn_relu_stack-O0-default: dma_cycles 1015 -> 1032.8
    ...
    cell lenet-O0-default: cycles 17766.25 -> 18624.350000000002
    cell lenet-O0-default: dma_cycles 16441 -> 17482.66666666667
    ...
    cell resnet_block-O0-tight: dma_cycles 1614 -> 1676.2666666666667

regression-baseline: FAIL. An optimization that moves a cycle count is not
necessarily wrong, but it must never move silently.
```

Restoring with `git checkout -- include/NPU/Simulator/CostModel.h` returns
`--check` to `no drift`, exit 0, and `git status --short` to empty.

**Three independent catches, and one deliberate non catch.** The frozen constants
test in `NPUSimulatorTests`, the Python cost model mirror in pytest, and every
cell's `cycles` and `dma_cycles`. The **golden tensors did not move**, and that is
correct rather than a gap: a cost model constant changes what the machine is
charged and not what it computes. The two halves of the baseline measure
different things and this run shows it.

## Activation proofs

**Two steps activate at P8** and each was broken deliberately, shown red, and
restored. Both rehearsals ran under **the exact CI invocation**, inside
`set -euo pipefail`, with the step's own script. **The CI runs are done.** The
reachability activation went red on `phase/p8-proof-rehearsal` at exactly the
`check-reachability full` step, with the lint job's `--skip-models` variant
green beside it
(<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33432288385>),
and the restore returned green
(<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33432842831>).
The CI fault was the model side one, not the product side one, and the reason
is in the engineering log: a committed perturbation of `ISA_OPCODES.json` is
caught by the staleness step first, which runs earlier and would stop the job
at the wrong net. The coverage arm's environment was proven by its first real
run instead of by a rehearsal, which is D-0032's story, also in the log.

**A rehearsal branch must be named under `phase/`.** `ci.yml` triggers on
`phase/**` and `main` only, and P7 lost a push to that. The branch to use is
`phase/p8-activation-rehearsal`, deleted after the restore run.

### 1. `check-reachability` full, product side: the mechanical simulation layer

```bash
sed -i 's/{"name": "POOL_MAX", "value": 9, "sources": \["max_pool2d"\], "needs_kernel": true/{"name": "POOL_MAX", "value": 9, "sources": ["max_pool2d"], "needs_kernel": false/' \
    docs/ISA_OPCODES.json
```

**Prediction, written before the run.** The reachability step red, exit 1,
naming `npu.max_pool2d missing: simulation`, because the simulation layer is
decided from this field now rather than by searching a comment. The ISA staleness
step also red, because a committed generated artifact no longer matches its
description.

**Result.** The first happened exactly. **The second did not**, and the
disagreement is the useful part. `check-isa-staleness.sh` **regenerates** the
artifacts before it diffs them, so a hand edit in the working tree is overwritten
by the regeneration and the diff finds nothing. That is correct behaviour: an
edit somebody **commits** is still caught, because then the regeneration
overwrites the working tree and the diff against `HEAD` shows it. Only an
uncommitted edit is silently repaired, and silently repairing a generated file is
what a generator is for. It also means the two steps interact through the
filesystem, and the rehearsal ran them in the order that shows the fault by luck
rather than by design.

**Restore:** `git checkout -- docs/ISA_OPCODES.json docs/ISA_MANUAL.md`. Both
steps green afterwards.

### 2. `check-reachability` full, model side: the layer that is new

```bash
sed -i '/^npu\.yield  /d' docs/EXEMPTIONS.md
```

**Prediction.** The reachability step red, exit 1, naming
`npu.yield missing: model`, and `--skip-models` still green, because the model
layer is exactly the one that mode leaves out.

**Result.** Exactly that. Full: exit 1, `npu.yield missing: model`,
`1 exemptions in force`. `--skip-models`: exit 0. Restore returns both to pass
with `2 exemptions in force`.

### 3. The coverage thresholds, C++ arm

Set `NPU_COVERAGE_THRESHOLD` to `99` in `.github/workflows/ci.yml`, or run
`bash scripts/coverage.sh 99 90`.

**Prediction.** Exit 1 at the C++ arm, printing the measured 86.0 against 99, and
the Python phase never runs because the script exits first.

**Result.** Exactly that.

### 4. The coverage thresholds, Python arm

`bash scripts/coverage.sh 85 99`.

**Prediction.** The C++ arm passes, the Python phase runs, exit 1 naming 90.27
below 99. Both arms are separately reachable, which is what a two threshold gate
has to prove.

**Result.** Exactly that.

**And this prediction was about the wrong thing, which the first CI run
showed.** It tested the threshold arithmetic and was right about it. It was
never a test of the arm's **environment**, and the environment is what broke:
the job configures `build-coverage/` and nothing else, this machine has a
`build/` sitting beside it, and the Python suite found its binaries through the
second one on every local run. That is D-0032. It is fixed, and the fix is
rehearsed below under conditions stricter than the job's own.

**The general lesson, recorded because it will recur.** The first run of a new
CI step is the first time anybody learns what that step's environment actually
is. A local rehearsal proves the step's logic; it cannot prove the step's
surroundings, because the surroundings are exactly what differs. That is the
argument for switching steps on one at a time, and it is what the activation
table is for.

### 5. D-0032, rehearsed after the fact, under the job's own conditions

The one rehearsal in this phase that was run **after** a CI failure rather than
before one, and the only one whose conditions were reproduced rather than
assumed. A worktree with no `build/`, which is what the job's checkout is, with
`MLIR_DIR` and `LLVM_DIR` from the environment the way the image supplies them
and `NPU_BUILD_DIR`, `NPU_OPT` and `PYTHONPATH` unset the way the job leaves
them:

```bash
git worktree add --detach /tmp/p8cov HEAD
cd /tmp/p8cov
export MLIR_DIR=... LLVM_DIR=...
unset NPU_BUILD_DIR NPU_OPT PYTHONPATH MLIR_PYTHON_PACKAGES_DIR
bash scripts/coverage.sh 85 90
```

`MLIR_PYTHON_PACKAGES_DIR` is unset too, which is **stricter than CI**: the job
does set it, and after the fix the script no longer needs it to.

**Before the fix**, reproducing job 99449908485 exactly, including its message:

```
coverage: PASS. C++ 86.0 percent is at or above the threshold of 85 percent.
ERROR collecting test/Python/test_metamorphic.py
E   VerificationError: npu-opt was not found ... Looked at $NPU_OPT (unset or
E   not a file), /tmp/p8cov/build/bin/npu-opt , and PATH.
Interrupted: 1 error during collection
STEP EXIT=2
```

**The five silent skips behind it**, exposed by naming `npu-opt` directly so
collection survived and changing nothing else: `484 passed, 12 skipped`, the
five being four tests in `test_refexec_differential.py` and one in
`test_cost_model_mirror.py`. One of them is `test_every_case_agrees`, the P7
gate item. A green run with a coverage number attached, describing a suite that
had not run its differential oracle.

**After the fix**, same worktree, same command:

```
coverage: PASS. C++ 86.1 percent is at or above the threshold of 85 percent.
coverage: MLIR bindings from /tmp/p8cov/build-coverage/CMakeCache.txt
495 passed, 7 skipped
coverage: PASS. Python 90.60 percent is at or above the threshold of 90.
STEP EXIT=0
```

495 where the broken state gave 484, and the Python headroom widened from 0.27
points to 0.61 because the tests that had been skipping now run.

## The proof of failure gate, Section 19.1

Four fault classes, each rehearsed locally under CI's own invocation, with the
prediction written before the run. **All four matched, all four restored, and
the CI record is complete**: pull requests 9 through 12, one fault each, opened
as drafts so the `pull_request` trigger fires and closed unmerged with their
branches deleted, every one red at the predicted job and step, with the four
red URLs and the green restore URL pasted into `docs/ENGINEERING_LOG.md`. The
first CI pass ran the faults as pushes to one rehearsal branch, which proved
the steps and not the trigger; it is recorded in the log as the finding it is.

The baseline before any fault: dash-lint exit 0, `check-npu` 20 of 20,
`NPUSimulatorTests` 54 passed, reachability pass, ISA staleness up to date.

### Fault class 1, an em dash in a markdown file

```bash
printf 'A sentence with an em dash \u2014 in it.\n' >> docs/BREAKING_CHANGES.md
```

The character is written as an escape rather than typed, because this file is
one of the files `scripts/dash-lint.sh` reads and a recipe that contained the
literal would fail the check it is a recipe for. `printf` in bash expands
`\uHHHH` in its format string, so the byte that reaches the markdown file is the
real U+2014.

**Predicted:** the `lint` job, at the `dash lint` step. **Result:** exit 1,
`docs/BREAKING_CHANGES.md:47:28: unicode-dash: em dash U+2014 is banned
everywhere`. `check-npu` untouched at 20 of 20. Restore: `git checkout --
docs/BREAKING_CHANGES.md`, dash-lint clean.

### Fault class 2, a failing lit test

```bash
sed -i 's|// LOWERED: npuisa.dma_store|// LOWERED: npuisa.dma_never|' \
    test/Pipeline/opt-levels.mlir
```

**Predicted:** `build-and-test`, at the `check-npu` step, one test.
**Result:** exit 1, `FAIL: NPU :: Pipeline/opt-levels.mlir`, 19 of 20 passed.
dash-lint untouched. Restore returns 20 of 20.

### Fault class 3, a failing GoogleTest

```bash
sed -i 's/EXPECT_EQ(kPeakMacsPerCycleF32, 256);/EXPECT_EQ(kPeakMacsPerCycleF32, 257);/' \
    unittests/Simulator/CostModelTest.cpp
ninja -C build NPUSimulatorTests -j6
```

**Predicted:** `build-and-test` at the `NPUSimulatorTests` step, one test,
`FrozenConstants.TheCostModelsNumbers`; also the `sanitizers` job's GoogleTest
step and the `coverage` job, which both run the same binary. `check-npu` green,
because lit never compiles the unit tests.
**Result:** exit 1, exactly that one test, 53 passed. `check-npu` 20 of 20.
Restore and rebuild returns 54 passed.

### Fault class 4, a failing pytest

```bash
sed -i 's/assert TIGHT_BUDGETS\["lenet"\] == 194624/assert TIGHT_BUDGETS["lenet"] == 194625/' \
    test/Python/test_tight_budgets.py
```

**Predicted:** `build-and-test` at the `pytest` step, one test,
`test_the_anchor_models_tight_budget_is_the_recorded_one`; and now also the
`coverage` job's Python arm, which is new at P8 and runs the whole matrix.
`check-npu` green.
**Result:** exit 1, exactly that one test, 481 passed. `check-npu` 20 of 20.
Restore returns 482 passed.

**A note the orchestrator will want when reading the CI runs.** A job stops at
its first red step, so CI will show less of each fault's net than these local
runs did. P7 recorded the same thing and the reason is mechanical rather than
interesting.

## Open questions

Eight. None blocks the gate.

**The NDEBUG and sanitizer directories cannot build anything that links MLIR.**
D-0031, found this phase. `NPU_FORCE_NDEBUG` turns `_GLIBCXX_ASSERTIONS` off in
this project's translation units while the LLVM they link keeps it on, which is
an ODR violation across the link; `build-fuzz` has the mirror with
AddressSanitizer's container annotations. Benign for the two targets these
directories exist for and fatal for `npu-opt`, which aborts on an empty module
inside MLIR's own context construction. **The real fix is a second LLVM tree
built without assertions**, an hour of build time and a decision with a cost. It
is documented loudly, warned about at configure time, and left for the
orchestrator, exactly like the third CI build below.

**The NDEBUG third CI build is still an open decision owned by the
orchestrator**, and this phase did not add one. D-0031 changes the calculus in
one direction: a third CI build would need an LLVM image without assertions, so
the two decisions are now one decision about an image rather than two.

**The OpenMP split is resolved and the fix is not taken.** The mechanism is the
compiler, reproduced locally: `build-and-test` and `sanitizers` configure with
clang, which needs `libomp` and `omp.h`; the coverage job passes no compiler and
gets gcc, which brings `libgomp`. The image installs `clang` and not
`libomp-dev`. The fix is one package in `docker/Dockerfile.llvm` plus an image
republish, which costs an hour of CI. Section 10.3's determinism assertion runs
at full strength wherever OpenMP is found, which is the coverage job and every
developer machine, so nothing is unproven meanwhile.

**No cell of the P8 matrix is marked `slow`, and that is a measurement.** Section
17.4 asks for a fast subset so an edit and rerun loop stays usable. The whole
`-O0` matrix is 140 cells and takes twelve seconds including the exports, so
there is nothing to carve out; marking a cell that costs a tenth of a second as
slow would be a label rather than a measurement. The marker and the P10 CI step
stay in place and start doing work when three levels and two budgets multiply
this matrix by six.

**The Python coverage threshold now has 0.61 points of headroom, up from 0.27.**
The measurement moved from 90.273 to 90.606 when D-0032 was fixed, for the
plainest of reasons: five tests that had been skipping now run. The threshold
stays at 90, which is Section 17.7's own rule of the measured value rounded down
to a whole percent, and raising it to 90.6 would be gating on a number measured
on one host. Still worth watching on the next run, since the container is a
different platform, and if it lands below 90 the honest response is to record the
CI host's measurement and set the threshold from it rather than widen it to a
round number nobody measured.

**`scripts/regression-baseline.sh --check` is not a CI step**, and Section 19.0's
activation table has no row for it. Two reasons, and the second is the real one:
it takes about a minute on top of a build, and its golden comparison is byte
identical, which is a bound between two runs of the *same* build. A different
compiler or host may contract a multiply and an add differently. Making it a CI
step means first deciding what tolerance a cross host golden comparison has, and
that is a decision, not an omission.

**The tight budget fraction is recorded and inoperative.** Section 15 asks for a
fixed fraction of the measured peak and the floor binds on all seven models, so
the fraction does nothing at P8. It becomes a live knob at **P13**, when tiling
gives the compiler a way to fit an instruction whose operands exceed the budget,
and the phase that lands tiling re-measures these constants once, in its own
commit, with `docs/BREAKING_CHANGES.md` written first per ground rule 7.

**Section 17.3a's fifth metamorphic relation cannot be written.** Pad then slice
back needs an ONNX `Pad`, which this importer refuses by name, and an ONNX
`Slice`, which has no converter. It returns when Section 11's operator set does,
and a test asserts the reason is still true so the relation is written rather
than forgotten if either gains a converter.

## Next command

Open the merge pull request for `phase/p8-walking-skeleton`.

The D-0032 fix ran and every job is green, Python coverage 90.61 on the CI
host:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33428771811>.
The activation proofs and the Section 19.1 record are complete, with all URLs
in `docs/ENGINEERING_LOG.md`.

### What the first push already established

Run 33367169622. **`lint`, `build-and-test` and `sanitizers` all green**,
including both of this phase's newly activated steps: the full reachability
check with its model IR build, and everything else the build and test job gained.

**The coverage job went red**, job 99449908485, and not at a threshold. The C++
arm passed at 86.1, one tenth of a point above the local measurement, and then
pytest died at **collection**: the Python suite could not find `npu-opt`, because
the job configures `build-coverage/` and nothing else and the suite was looking
in `build/`. That is D-0032, it is fixed in the tip commit, and the fix is
rehearsed in a worktree with no `build/` under conditions stricter than the
job's own. The defect log carries the before and after; the short version is
`STEP EXIT=2` becoming `STEP EXIT=0` with `495 passed, 7 skipped`.

**The five silent skips are the part worth carrying forward.** Behind the loud
failure, two test modules with their own hand written binary lookup would have
**skipped** rather than failed, and one of them carries P7's gate item that the
reference interpreter and the simulator agree on every operation. A green
coverage run would have attached a number to a suite that had not run its
differential oracle. The lookup is now one rule with one policy, and
`test_tool_discovery.py` makes a second copy a red test.

### What to watch on the next run

**The coverage job, again and specifically.** It is the only job whose new
behaviour is still unproven in CI. Expect the C++ arm at about 86.1 and the
Python arm at about 90.6, and expect the log to carry the line
`coverage: MLIR bindings from .../build-coverage/CMakeCache.txt`, which is the
fix reporting that it resolved the path itself rather than relying on the job's
environment.

**The other three jobs are re-runs of a green configuration** plus the tip
commit, which touches the Python suite and documentation only.

### Then, still outstanding

**The four proof of failure pull requests of Section 19.1**, each on its own,
each from a branch named under `phase/`, with the four red URLs and one green
URL pasted into `docs/ENGINEERING_LOG.md`. The recipes and the local results are
above; if a fault is caught by a different job than the prediction says, Section
19.1 is explicit that this is a finding to record rather than a fault to adjust.

**And the two activation proofs above**, on `phase/p8-activation-rehearsal`,
deleted after the restore run.

**Then the merge pull request.**

## Next phase

**P9, optimization passes and opt levels.** Every pass in Section 12 except
tiling, layout, double buffering and calibration, wired into `-O1` and `-O2`,
each carrying its `ablatable` property.

**P8 is the first legitimate stop and ship point**, and the project is at it. The
compiler computes correct answers end to end and can prove it.

Six things P8 leaves on P9's desk.

1. **`-O1` and `-O2` are named and not registered**, in
   `lib/Pipeline/Pipeline.cpp`. Adding a level is two lines in `kLevels`, a table
   of `PassEntry` rows, one `PassPipelineRegistration`, and the `switch` in
   `build()` which has no `default` and will be a build error until it is
   handled. Three tests go red on purpose the day the level lands and each is the
   assertion that should move with it:
   `test_only_minus_o_zero_is_implemented_at_this_phase`,
   `test_import_and_npu_are_the_same_text_at_minus_o_zero`, and
   `test_no_level_this_compiler_builds_eliminates_dead_code`.
2. **The dead subgraph's instruction count check is waiting for a pass that
   removes something.** Mark `-canonicalize` and `-symbol-dce` with
   `eliminatesDeadCode = true` and
   `test_a_dead_subgraph_does_not_change_the_instruction_count` starts running
   with no edit. If it is marked and the test does not go green, that is the
   finding the check exists for.
3. **The two exemptions in `docs/EXEMPTIONS.md` are `-npu-fuse-ops`'s to
   delete.** The commit that lands the pass deletes them, or the reachability
   check goes red at the next run.
4. **The baseline gains per level fields and a `schema_version` bump**, per
   Section 17.6 and P9's own gate. `absent_fields` names `per_level` today;
   removing that name and adding the fields is the same commit.
5. **`docs/BREAKING_CHANGES.md` is written before the commit that moves a
   number**, then the baseline is re-recorded in its own commit. Ground rule 7,
   and P9 is the first phase that moves numbers on purpose.
6. **The metamorphic relations are the cheapest test P9 has.** Fusion changes
   accumulation order, so the four relations stop being exactly equal and become
   a tolerance question, and the answer to that question is P9's to record rather
   than to inherit.

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
