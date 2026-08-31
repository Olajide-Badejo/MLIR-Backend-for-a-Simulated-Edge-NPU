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

**P9, optimization passes and opt levels.** Branch `phase/p9-opt-passes`, cut
from `main` at `5929af8`, which is the P8 merge. Six commits, none pushed. The
sixth carries this table, so it is named by subject rather than by a sha it
cannot know.

| Commit | Subject |
|---|---|
| `e5ac4e2` | `feat(npu): the tensor level passes of Section 12, and three defects they found` |
| `b4834fd` | `docs(breaking): declare the numbers -O2 moves, before the commit that moves them` |
| `2d2563b` | `feat(pipeline): -O1 and -O2, and the two exemptions -npu-fuse-ops closes` |
| `ae72489` | `feat(scripts): the baseline's per level fields, at schema_version 2` |
| `dc3beca` | `chore(baseline): re-record at -O0, -O1 and -O2, after the declaration` |
| tip | `docs: hand off phase P9` |

**The commit order is Section 17.6's and ground rule 7's, and it is the part of
this phase a reviewer should check first.** The declaration at `b4834fd` is
strictly before the movement at `2d2563b`, and the re-record at `dc3beca` is its
own commit strictly after it, touching `test/baseline/` and nothing else. An
entry written after a number moved is an explanation; one written before it
moved is a decision, and git commit order is what tells the two apart. This is
the first phase in the project that moves a number on purpose, so it is the
first time the mechanism has been exercised rather than described.

**Every commit was checked out and built on its own.** A worktree at
`/tmp/p9check`, since removed, walked all five and ran a fresh configure, a
build and `check-npu` at each. All five configured, built clean and reported 25
of 25, which is 20 from P8 plus the five files of `test/Transforms/` that arrive
in the first commit.

**A fresh worktree needs three arguments the developer build's cache already
holds**, which P8's note omitted and which cost a run to rediscover:

```bash
git worktree add --detach /tmp/p9check HEAD
cmake -S . -B build -G Ninja \
      -DMLIR_DIR=/home/elijah/llvm-project/build/lib/cmake/mlir \
      -DLLVM_DIR=/home/elijah/llvm-project/build/lib/cmake/llvm
ninja -C build -j6 && ninja -C build check-npu
```

Without `-G Ninja` cmake writes Makefiles and `ninja -C build` fails with "no
such file or directory" for `build.ninja`, which reads like a configure that did
not run rather than one that ran with a different generator.

**One commit has a red pytest suite and it is the ordering, not an oversight.**
At `ae72489` the script writes `schema_version` 2 and the committed baseline is
still the P8 file at version 1, so the five assertions about the committed
baseline fail. `dc3beca` is the commit that makes them pass. Ground rule 7
requires exactly that interval: declare, move, record, and the gap between the
second and the third is what makes the ordering checkable at all. Every other
commit is green on the full suite.

**Why the four passes are one commit.** They share a TableGen file, a CMake
target and a library that cannot be built with fewer than one pass in it, so a
commit per pass would have meant four rounds of splitting `Passes.td` and the
source list to produce a history that bisects no better. Each pass has its own
source file, its own lit file and its own section of `docs/PASSES.md`, which is
the granularity a reviewer reads at.

## Gate status

The P9 gate is the roadmap entry's, parsed clause by clause. Every item is **met
locally**. No CI step is activated or changed by this phase, so there is nothing
for the orchestrator to prove red beyond the merge run itself; see "Activation
proofs" below for why.

| # | Gate clause | Status | Proof |
|---|---|---|---|
| 1 | a positive lit test per pass | PASS | `test/Transforms/{constant-fold,fuse-bias,fold-batchnorm,fuse-ops,level-passes}.mlir`. The last covers `-canonicalize`, `-cse`, `-sccp` and `-symbol-dce`, because Section 12's negative test rule applies to every pass in its table and not only to the ones written here |
| 2 | at least one negative lit test per pass | PASS | Every file above carries between one and five. `-npu-constant-fold` has three, `-npu-fuse-bias` five, `-npu-fold-batchnorm` five, `-npu-fuse-ops` three, and the four upstream passes one each |
| 3 | the fusion negative cases: already fused, multiple uses, non channel constant addend | PASS | `no_fuse_already_biased`, `no_fuse_multiple_uses`, and the addend case in both its halves: `no_fuse_same_shaped_addend` and `no_fuse_non_constant_addend`. All in `test/Transforms/fuse-bias.mlir` |
| 4 | `-npu-fuse-bias` fires on a real imported model | PASS, with a finding | `test_fuse_bias_fires_on_a_real_imported_model` builds the ONNX graph, runs `import_model`, and runs the pass on what came out. The finding beside it is item 5 of "Open questions": **no model of Section 15's suite has the shape**, and a test asserts that so the claim cannot go stale |
| 5 | numerics within 1e-6 at every level | PASS | Every one of the 42 baseline cells carries `max_abs_movement_vs_o0`, the largest is 4.470e-08, and `test_the_committed_baseline_records_this_phases_numerics_movement` asserts the bound. Separately the end to end matrix runs 210 cells per oracle at the P8 tolerances, unmoved |
| 6 | the largest observed movement recorded and its mechanism named | PASS | 4.470e-08 on `conv_bn_relu_stack` at `-O2`, at both budgets. The mechanism is `-npu-fold-batchnorm` and it is named by measurement rather than by attribution: `test_fold_batchnorm_is_the_only_pass_that_moves_a_number` runs each of the eight ablatable passes alone and asserts the other seven move exactly zero |
| 7 | the intended movement declared in `docs/BREAKING_CHANGES.md` **before** the commit that causes it | PASS | `b4834fd` precedes `2d2563b`. The entry names the cells, the fields, the direction, the magnitude and the pass |
| 8 | then the baseline re-recorded in its own commit | PASS | `dc3beca`, whose diff is `test/baseline/baseline.json` and 21 `.npy` files and nothing else |
| 9 | per level golden and baseline fields added | PASS | 42 cells over three levels where there were 14 over one, 21 goldens where there were 7, a top level `levels` field, and `max_abs_movement_vs_o0` per cell. `per_level` has left `absent_fields`; `energy` is still there and still names P11 |
| 10 | with a `schema_version` bump | PASS | 1 to 2. `--check` refuses a version it does not recognise, which is what stops a P8 file being read as 28 regressions from nothing |
| 11 | every user visible movement named in `CHANGELOG.md` | PASS | A P9 section with the levels, the four passes, the emptied exemption block, the numerics movement with its magnitude, the driver's `npu` stage, the matrix, the baseline schema, and a `Fixed at P9` block naming D-0033, D-0034 and D-0035 |

### The three assertions P8 left to move, and what each did

P8's handoff named three tests that would go red the day a level landed and said
each was the assertion that should move with it. Two moved. The third did not
need to, and that is recorded rather than quietly left alone.

- **`test_only_minus_o_zero_is_implemented_at_this_phase`** is now
  `test_every_level_this_compiler_names_is_one_it_builds`, asserting
  `implemented_levels() == [0, 1, 2]`. It still reads the set from the compiler.
- **`test_no_level_this_compiler_builds_eliminates_dead_code`** is now
  `test_the_levels_that_eliminate_dead_code_are_the_ones_with_a_pass_that_does`,
  asserting `[1, 2]` and, beside it, which pass in each level carries the mark.
  The parametrized check it was guarding filled itself from two
  `eliminatesDeadCode = true` marks with no edit, which is what P8 designed it to
  do.
- **`test_import_and_npu_are_the_same_text_at_minus_o_zero`** kept its name and
  its assertion. P8 predicted the two stages would stop being equal, and they
  have not: `-O0`'s tensor level half is empty, and although the stage now goes
  through `npu-opt` at every level, the importer already prints locations and its
  output round trips byte for byte. That is a stronger property than the test was
  written to assert. Its docstring now says so, and a new pair asserts that at
  `-O1` and `-O2` the two stages do differ, which is the half that had to be
  true or the levels would be levels in name only.

### The dead subgraph clause, and how P8's reading came out

P8 answered the gate's "a dead subgraph changes neither the outputs nor the
`-O2` instruction count" in two halves, asserted the outputs half, and left the
count half to fill itself from the compiler's description. It filled itself.
`-canonicalize` and `-symbol-dce` are marked `eliminatesDeadCode = true`,
`levels_that_eliminate_dead_code()` returns `[1, 2]`, and
`test_a_dead_subgraph_does_not_change_the_instruction_count` runs on 14 cells,
seven models at each of two levels, with no edit to the test. Its `-O0`
counterpart is kept as the control: without it, a `-O2` count that did not move
would be consistent with an injection that brought nothing.

### The metamorphic tolerance question, answered

P8's sixth desk item said fusion changes accumulation order, so the four
relations would stop being exactly equal and the answer was P9's to record
rather than to inherit. Measured over four relations times seven models times
three levels, of which 29 cells apply:

- at `-O0` and `-O1`, **every applicable cell agrees to exactly zero**;
- at `-O2`, every cell agrees to exactly zero **except** `convolution_split` on
  `conv_bn_relu_stack`, which moves by **2.98e-08** absolute and 1.30e-07
  against the answer's largest magnitude.

**The mechanism is not fusion in general, and that is the part worth carrying
forward.** A relation compiles the original and the variant *at the same level*,
so a reassociating pass runs on both sides and cancels. What does not cancel is
a rewrite that changes which passes can **match**: `convolution_split` replaces
one convolution with two over channel groups and a concatenation, so the batch
norm's producer becomes an `npu.concat` and `-npu-fold-batchnorm` declines on
the variant while folding the original.

So the band is 1e-6, which is Section 17.6's class for this phase and the same
band `docs/BREAKING_CHANGES.md` declares, and it applies only at a level whose
pipeline runs a pass a rewrite can unmatch. Which levels those are is read out
of the compiler through `REASSOCIATING_PASSES`, so `-O0` and `-O1` keep byte
equality and a level that stopped folding would get it back with no edit.
`test_the_one_relation_that_moves_moves_by_what_was_recorded` asserts the
measured value rather than only the bound, so a cell that crept to 9e-07 would
fail while still inside the band.

### The exemption block is empty

`scripts/check-reachability.py` passes in full with **no exemptions in force**,
which is the first form the P8 gate offered and the one P8 could not reach.
`npu.fused_op` and `npu.yield` were exempt from the model layer because nothing
created one; `-npu-fuse-ops` creates them, `scripts/build-model-ir.py` sweeps
every level the compiler builds rather than `-O0` alone, and both appear in the
IR of five of the seven models.

The entries were deleted by `2d2563b`, which is the commit that put the pass in
a level and swept at that level, and not by the commit that landed the pass.
That is the correct commit and the fault rehearsal below shows why: with the
pass out of `-O2` and the block empty, the check goes red naming both operations
at the model layer.

## Verification output

Every command below was run on this branch at `dc3beca`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 25 discovered, 25 passed. Twenty at P8, plus this phase's five in `test/Transforms/` |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed |
| `build/bin/NPUAllocatorTests` | 29 tests, 29 passed |
| `build/bin/NPUEncodingTests` | 77 tests, 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 55 tests, 54 passed, 1 skipped |
| `build-ndebug/bin/NPUSimulatorTests` | 54 passed, 1 skipped. See D-0031 for what this directory may and may not build |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build-fuzz/bin/NPUSimulatorTests` under ASan and UBSan | 54 passed, 1 skipped, exit 0 |
| `build-fuzz/bin/NPUEncodingTests` under ASan and UBSan | 75 passed, 2 skipped, exit 0 |
| `python -m pytest test/Python -q` | 857 passed, 18 skipped, 7 deselected, exit 0 |
| `python -m pytest test/Python -q -m 'slow or not slow'` | 864 passed, 18 skipped, exit 0. 495 at P8, plus this phase's 369 |
| `mypy` | no issues found in 19 source files |
| `black --check .` | 42 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 269 of 269 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/build-model-ir.py` | 84 IR files written, up from 28 |
| `python scripts/check-reachability.py` | pass, all five layers, **no exemptions in force** |
| `python scripts/check-reachability.py --skip-models` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `bash scripts/regression-baseline.sh --check` | no drift, exit 0 |
| `bash scripts/coverage.sh 85 90` | C++ 86.5 PASS, Python 90.49 PASS, exit 0 |
| `git status --short` | empty |

**The suite grew by 369 pytest tests and five lit files.** Where it grew is the
level axis on two matrices, the per pass measurements in
`test/Python/test_transform_passes.py`, and the driver's contract at three
levels rather than one.

### The baseline, recorded and checked

`test/baseline/baseline.json`, `schema_version` 2: 8 suites, the git sha, the
tool versions, `absent_fields` naming only `energy`, a `levels` list, and 42
cells. `test/baseline/golden/` holds 21 `.npy` tensors, seven models at three
levels.

```
regression-baseline --check

  recorded at ae72489c33ac, checked at ae72489c33ac
  42 cells, 21 golden tensors, levels -O0, -O1, -O2
  largest movement against -O0: 4.470e-08
  suite NPUAllocatorTests      29 passed    0 failed    0 skipped
  suite NPUEncodingTests       76 passed    0 failed    1 skipped
  suite NPUInterfaceTests      23 passed    0 failed    0 skipped
  suite NPUSimulatorTests      54 passed    0 failed    1 skipped
  suite NPUTilingTests         12 passed    0 failed    0 skipped
  suite check-npu              25 passed    0 failed    0 skipped
  suite dash-lint               2 passed    0 failed    0 skipped
  suite pytest                864 passed    0 failed   18 skipped

regression-baseline: no drift.
```

**The `-O0` half is unchanged bit for bit against P8**, and that is checked
rather than asserted: the previous file was read out of git and diffed field by
field over all fourteen `-O0` cells and all seven `-O0` goldens. No field moved
and the largest golden difference is exactly 0.0. It matters because both of
this phase's lowering changes touch the pass every level runs, and the argument
that they are no operations at `-O0` is only as good as that measurement.

**The record was taken twice and the first take discarded.** A baseline records
the pass and fail counts of every suite, and at the first take the five
assertions about the committed baseline were still describing the P8 file, so
the record captured a red pytest suite. The script warned in its own words: "a
baseline recorded from a red tree records what is broken as if it were correct".
The second take ran against the file the first had written. That is a wrinkle in
the declare then re-record procedure worth knowing about in advance at P13 and
P14, which carry the same step.

## Activation proofs

**This phase activates no CI step and changes no CI configuration.**
`.github/workflows/` is untouched, and `git diff main..HEAD --stat` shows it.
Section 19.0's activation table has no row that flips at P9: `check-npu` picks
up the new lit directory by discovery, `pytest` picks up the new files by
`testpaths`, and `check-reachability full` was switched on at P8 and now has
three times as much model IR to look at.

What did change is the **workload** of three steps, so each was rehearsed under
its exact step script inside the shell options the workflow uses, with the
prediction written before the run.

### 1. The `pytest` step, verbatim

The step's own guard and its exit code 5 branch, with
`MLIR_PYTHON_PACKAGES_DIR` pointed at the local bindings.

**Prediction.** Exit 0. Around 857 passed with 7 deselected, because the step
runs the default marker expression rather than `slow or not slow`. The exit code
5 branch not taken.

**Result.** Exactly that: `857 passed, 18 skipped, 7 deselected`, `STEP EXIT=0`.
The 7 deselected are the `slow` marked cells that have been deselected since P7
and are still nothing to do with this phase's matrix, which carries no `slow`
marks.

### 2. The `check-reachability full` step, verbatim

**Prediction.** Exit 0, `build-model-ir` writing 84 files rather than 28, and
the check reporting **no exemptions in force**, which is a sentence that has
never appeared in this project's output before.

**Result.** Exactly that. `build-model-ir` took 8 seconds locally for the 84
files, up from about 3 for the 28, which is the one CI runtime number this phase
moves noticeably and is well inside the step's budget.

### 3. Fault: `-npu-fuse-ops` removed from `-O2`, with the exemption block empty

The proof that deleting the two exemptions was earned rather than convenient.

```python
# remove the PassEntry(PassKind::NPUFuseOps, ...) row from kO2 in
# lib/Pipeline/Pipeline.cpp, rebuild npu-opt, rebuild the model IR
```

**Prediction.** `check-reachability` full red, exit 1, naming `npu.fused_op` and
`npu.yield` as missing the model layer, with no exemption to cover them.

**Result.** Exactly that:

```
  note: no exemptions in force.
  Operations missing a required layer:
    npu.fused_op  missing: model
    npu.yield     missing: model
check-reachability: FAIL
```

**Restore:** `git checkout -- lib/Pipeline/Pipeline.cpp`, rebuild, rebuild the
model IR. Back to `check-reachability: pass` with no exemptions in force.

**Which trigger this needs if the orchestrator wants it in CI:** none of its
own. It is a fault in a step that has been on since P8, so it runs under both
`push` to `phase/**` and `pull_request`.

### 4. Fault: `-canonicalize` stops declaring `eliminatesDeadCode`

The proof that the mark is what fills the parametrization, and that emptying it
is loud rather than quiet.

```python
# flip /*eliminatesDeadCode=*/true to false on the -O1 canonicalize row
```

**Prediction.** `test_the_levels_that_eliminate_dead_code_are_the_ones_with_a_pass_that_does`
red, because the set becomes `[2]`. The dead subgraph count check silently loses
its seven `-O1` cells, which is the vacuous parametrization failure mode that
assertion exists to make loud.

**Result.** Exactly that, and the arithmetic shows the silent half:
`1 failed, 112 passed` against `120 passed` restored. One loud failure and seven
cells that would otherwise have disappeared without a word.

**Restore:** `git checkout -- lib/Pipeline/Pipeline.cpp` and rebuild. 120
passed, 18 skipped.

**Which trigger this needs:** none of its own; it is a `pytest` failure and that
step has been on since P3.

## Open questions

Nine. None blocks the gate.

**`-npu-fuse-bias` fires on no model in Section 15's suite.** Every convolution
in the seven models carries its bias inline as a third `Conv` input, which is
what `torch.onnx.export` and this project's ONNX built models both emit, so
`add(conv(x, w), b)` appears nowhere. The pass fires on a real imported model
built for it, which is the gate's clause, and
`test_no_suite_model_gives_the_bias_fusion_anything_to_do` asserts the suite gap
so the claim cannot go stale. **The consequence is P10's**: the pass's ablation
row will be a row of zeros across the whole suite. The cheapest fix is one node:
`dilated_stack`'s `conv1` is already biasless and is followed by a `Relu`, so an
`Add` of a rank 1 initializer between them gives the pass a target in the suite.
That is a model suite change, so it moves `GENERATOR_VERSION`, every
`dilated_stack` cell, and possibly that model's tight budget, which
`docs/adr/0008` froze. It is a decision above this phase's pay grade and it is
recorded here for P10 to take.

**`-sccp` fires on no model in the suite either, and for a structural reason.**
It needs a call graph and an imported model is one function. D-0033 made it able
to write an answer down; it still has nothing to write. Its ablation row will
also be zero, and unlike the bias fusion there is no model change that would
alter that: this compiler has no calls. Section 12 puts it at `-O2` and it stays
there, because a row of zero is the measurement that section asked for.

**`-O1` is exactly `-O0` on every model in the suite**, in every field including
the goldens. Constant folding has nothing to fold and canonicalization has
nothing to remove in an exported graph. Both passes are proven on a graph built
to have both, which is the dead subgraph injection, so this is a property of the
suite rather than of the passes. Worth stating at P10 in the report rather than
leaving a reader to infer that `-O1` does nothing.

**The Python coverage headroom narrowed from 0.61 points to 0.49.** Measured
90.49 against a threshold of 90. It moved because this phase added
`python/npu_frontend/refgraph.py`'s fused region path and the driver's second
`npu-opt` invocation, both of which have error branches no test reaches. The
threshold stays at 90, which is Section 17.7's rule of the measured value
rounded down to a whole percent. Still worth watching on the CI host, which
measured 90.61 where this machine measured 90.60 at P8.

**The end to end matrix is now 420 cells and takes about 30 seconds**, so no
cell is marked `slow` and the fast subset is still the whole matrix. That
remains a measurement rather than a decision, and P10 is where it changes: the
ablation cells arrive beside these and Section 2's 90 minute budget becomes a
gate.

**`scripts/regression-baseline.sh --check` is still not a CI step**, for the
reason P8 gave: its golden comparison is byte identical, which is a bound
between two runs of the same build, and making it a CI step means first deciding
what tolerance a cross host golden comparison has. P9 sharpens the question
rather than answering it, because there are now two bands in the file and only
one of them is about reproducibility.

**The NDEBUG and sanitizer directories still cannot build anything that links
MLIR**, D-0031, and the NDEBUG third CI build is still an open decision owned by
the orchestrator. Unchanged by this phase.

**The OpenMP split is unchanged and the fix is still not taken.** One package in
`docker/Dockerfile.llvm` plus an image republish.

**Section 17.3a's fifth metamorphic relation still cannot be written.** `Pad` is
refused by name and `Slice` has no converter, and a test asserts the reason is
still true.

## Next command

Push the branch and open the merge pull request for `phase/p9-opt-passes`.

```
git push -u origin phase/p9-opt-passes
```

**Nothing has been pushed.** There are no CI runs to record for this phase yet,
and no activation proof needs one: `.github/workflows/` is untouched, so the
merge run is the first and only CI this phase requires.

### What to watch on that run

**The `build-and-test` job's `check-reachability full` step**, which is the one
whose workload changed most: `build-model-ir.py` now compiles every model at
three levels and writes 84 files instead of 28. Eight seconds locally, so
expect tens of seconds there, and expect the line `no exemptions in force`,
which this project's CI has never printed.

**The `pytest` step**, at 857 passed and 7 deselected rather than 488 and 7. The
matrix tripled and the wall clock roughly did too.

**The `coverage` job's Python arm**, at about 90.5 against a threshold of 90.
It measured 90.61 on the CI host at P8 and 90.49 here now, so the margin is real
but thinner than it was.

**The `lint` job and the `sanitizers` job are re-runs of a green
configuration.** Neither reads anything this phase changed except the source
files themselves.

## Next phase

**P10, measurement.** Sections 16.1 and 16.2 in full, the full end to end matrix
of Section 17.4, and the prediction mechanism of Section 17.8.

Six things P9 leaves on P10's desk.

1. **The ablatable set is eight and two of its rows will be zero.**
   `-npu-fuse-bias` fires on no model of the suite and `-sccp` cannot fire on a
   single function program at all. Both are recorded above with their reasons.
   The first has a one node fix that costs a `GENERATOR_VERSION` bump and a
   possible tight budget re-measurement; the second has none. Section 16.2 wants
   the ablatable set read from the driver at run time and `ablatable_passes()`
   already does that, deduplicated, so the harness needs no list.
2. **`-canonicalize` appears twice at `-O2` and is one pass to ablate.**
   `ablatable_passes()` returns the set rather than the sequence, which is what
   makes Section 12's eleven reachable. A harness that read the sequence would
   give one pass two rows.
3. **The `PassInstrumentation` of Section 16.2 has a pipeline to sit on now.**
   Section 6's whole argument for putting the levels in C++ was that the
   instrumentation must sit on the `PassManager` that actually runs the passes,
   and `mlir::npu::pipeline::build` is that manager. A pass present in the
   pipeline but absent from the JSON must raise, which the description makes
   checkable: `describe()` is the list to compare against.
4. **The full matrix is 420 cells and the budget question is now live.** No cell
   is marked `slow` because the whole thing takes half a minute, and the
   ablation cells multiply it again. Section 2's 90 minute budget becomes a gate
   at P10 and the marker is already in place and inert.
5. **`max_abs_movement_vs_o0` is per cell and is the field a report should quote
   from.** Nothing hand copies a number in this project, and the largest
   movement of a phase is now a field rather than a sentence.
6. **The tight budgets survived this phase and nearly did not.** D-0035's second
   half raised LeNet's `-O2` peak past the budget `docs/adr/0008` froze at P8,
   and the fix was in the lowering rather than in the budgets. P13 is the phase
   that re-measures them, with `docs/BREAKING_CHANGES.md` written first; until
   then a pass that raises a peak is a defect rather than a reason to move a
   number.

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
