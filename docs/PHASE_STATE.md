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

**Last updated:** 2026-08-19.

## Current phase

**P2, the `npuisa` dialect and the memory model.** Branch
`phase/p2-npuisa-dialect`. Seven commits, not merged, not pushed.

**Read this before merging anything.** The branch was cut from
`phase/p1-npu-dialect`, **not** from `main`, and P1 has never been merged. The
merge base of `main` and this branch is `9bf5d5e`, the P0 merge, so
`git log main..HEAD` shows twelve commits: P1's five and P2's seven. `main` is
still at P0. That is not a mistake to correct by rebasing, since P2 genuinely
depends on P1's dialect and its memory space attributes, but it does change the
merge plan: **P1 merges first, then P2**, or the two merge together as one pull
request that closes both gates. The P1 handoff assumed its own merge would have
happened by now and this file recorded the branch as cut from `main`; it was not,
and correcting that here is more useful than leaving a reader to discover it at
the merge.

| Commit | Subject |
|---|---|
| `00cce3b` | `feat(dialect): add the npuisa dialect with the memory model and the token` |
| `784cc68` | `fix(dialect): reject a null memory space and let two transfers be in flight` |
| `cc640ba` | `test(dialect): add the npuisa round trip, verifier and canonicalization suites` |
| `8e175af` | `test(dialect): add NPUInterfaceTests and find gtest from either source` |
| `aefd7fa` | `build(coverage): add scripts/coverage.sh per Section 17.7` |
| `487745d` | `ci: switch on NPUInterfaceTests and the coverage job, and rebuild the image` |
| `8dd32a5` | `docs: record the memory model design and hand off P2` |

The first commit is inherited from an interrupted session. Everything from
`784cc68` onward is this session.

Below those, and already on this branch because of the branch point above, are
P1's five: `50b27f1`, `a7fdbba`, `cc8a889`, `ec45f99`, `9e341f8`.

## Gate status

The P2 gate is Section 23's: round trip and verifier failure coverage;
`NPUInterfaceTests` green, including the overlap rule decided on effects plus
view offsets rather than on SSA identity, and a test asserting `ins` and `outs`
partition the operands exactly once; no compute operation reports itself free of
effects; the design entry written in `docs/ARCHITECTURE.md` and treated as
binding by later phases.

Item by item, with what proves it. Every item is **met locally**; the CI half of
two of them is **pending an image republish**, which is the orchestrator's next
action and is not a gap in the work.

### Met, proved locally

- **Round trip coverage.** `test/Dialect/NPUISA/ops.mlir` pipes `npu-opt` output
  back through `npu-opt` before checking, so an operation that prints something
  it cannot read back fails rather than passing a parse. Every operation is
  covered, plus the token in a function signature, both element types, the
  optional bias on `matmul` and `conv2d`, a grouped convolution, a `ceil_mode = 1`
  pool with its arithmetic written into the test, an in place relu, and the two
  function level scratchpad attributes. `test/Dialect/NPUISA/ops-memref.mlir`
  covers the memory model separately: the two spaces, `memref.alloc` in the
  scratchpad, the flat buffer with views over it, and the boundary shape of a
  lowered function.
- **A verifier failure case per rule.** `test/Dialect/NPUISA/invalid.mlir`, run
  under `-split-input-file -verify-diagnostics`, so an unexpected diagnostic
  fails as loudly as a missing one. Every `expected-error` quotes the substring
  the verifier actually emits rather than matching generically.
- **The overlap rule decided on effects plus view offsets, never SSA identity.**
  Three places, at three levels. `@intervening_write_to_a_partially_overlapping_view`
  in `invalid.mlir` is the lit case: two `memref.view` results over one flat
  buffer, bytes [0, 64) and [32, 96), different SSA values sharing 32 bytes.
  `NPUISAInterfaceTest.PartiallyOverlappingViewsOverlap` asserts the same at the
  arithmetic level and explicitly asserts the two values differ, so the test is
  about something. `NPUISAInterfaceTest.TheVerifierRejectsAnOverlappingInterveningWrite`
  and its disjoint sibling assert that the verifier calls the arithmetic, which
  is a separate claim from the arithmetic being right.
- **A non static offset is refused rather than assumed disjoint.**
  `@async_destination_with_a_dynamic_offset` and
  `@intervening_buffer_with_a_dynamic_offset` in `invalid.mlir`, and
  `NPUISAInterfaceTest.ANonStaticOffsetIsUnknownAndNotDisjoint`, which asserts
  `Unknown` and separately asserts *not* `Disjoint`.
- **`ins` and `outs` partition the operands exactly once.**
  `NPUISAInterfaceTest.InsAndOutsPartitionOperandsExactlyOnce` over all thirteen
  table entries, counting coverage per operand so an operand in neither list and
  an operand in both both fail. Plus `TheOptionalBiasDoesNotMoveTheDestination`,
  because the destination is operand 2 without a bias and operand 3 with one, and
  `TheVariadicConcatStillPartitions`, because `concat`'s destination index is not
  a constant of the operation at all.
- **No compute operation reports itself free of effects.**
  `NPUISAInterfaceTest.NoComputeOperationIsFreeOfEffects`, checked two ways: an
  empty effect list, and `isMemoryEffectFree`, which is what `isOpTriviallyDead`
  consults. An operation that does not implement the interface at all would pass
  one and fail the other. The transfers and the asynchronous forms have their own
  tests. `@compute_instructions_survive_canonicalization` in `canonicalize.mlir`
  asserts the same thing at the level a user would notice: the function does not
  canonicalize to a bare `return`.
- **The async with immediate await canonicalization.**
  `test/Dialect/NPUISA/canonicalize.mlir`, both directions and both transfer
  directions, plus the two negative cases that matter: an await two operations
  later does not fold, and two transfers in flight do not fold.
- **`docs/ARCHITECTURE.md` written**, Diataxis explanation type, covering the two
  spaces, the scoped DMA boundary invariant with its three permitted producers,
  offsets as SSA operands, the token and the overlap rule, and why
  `TilingInterface` lives on `npu` and not here. It closes with a list of what it
  binds on later phases.

### Verification output

Every command below was run on this branch at `487745d`, from
`/home/elijah/npu-mlir-v2`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 7 discovered, 7 passed, 0 failed |
| `./build/bin/NPUInterfaceTests` | 18 tests, 18 passed |
| `./build/bin/NPUTilingTests` | 12 tests, 12 passed |
| `bash scripts/coverage.sh` | 75.7 percent line, 68.1 percent branch, exit 0 |
| `bash scripts/coverage.sh 99` | exit 1, names the number and the threshold |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 102 of 102 files |
| `pre-commit run --all-files` | all hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, exit 0 |
| `git status --short` | empty |

Both branches of the gtest search were verified, not just the one this machine
takes. The build tree branch is what `build/` uses. The system package fallback
was forced by configuring with `-DNPU_LLVM_THIRD_PARTY=/nonexistent-on-purpose`,
which produced `GoogleTest: using the system GoogleTest package` and the same
eighteen passing tests.

### Met locally, CI pending an image republish

Two gate items are green locally and cannot be green in CI until the image is
rebuilt. This is a sequencing fact, not an unmet gate: the commit that requires
the new image is on the branch, and the rebuild is the orchestrator's next
action.

- **`NPUInterfaceTests` in CI.** The step is switched on and now fails rather
  than printing an off line if the binary is missing. The binary needs a
  GoogleTest, and the current published image has none. `487745d` adds
  `libgtest-dev` to the image for the CMake fallback to find. **Pushing before
  the republish will turn `build-and-test` red at this step, and that is the
  expected sequence rather than a defect.**
- **The coverage job in CI.** Same cause: the job needs `gcovr`, which the same
  Dockerfile revision adds.

## What the orchestrator does next

In this order. Steps 2 and 3 are the reason step 1 comes first.

1. **Dispatch `llvm-image.yml` and let it publish.** This is the blocking step
   and it is roughly an hour. The workflow is `workflow_dispatch` only since the
   push trigger was retired at P1, so it has to be started by hand. The rebuilt
   image must satisfy the smoke test at the end of `docker/Dockerfile.llvm`,
   which now also checks for the GTest CMake config, the two gtest static
   libraries, `gcovr`, `gcov`, the bindings directory, and an actual
   `import mlir.ir` with a `Context` constructed.

2. **Push the branch** and let CI run it green. Pushing before step 1 completes
   will be red at `NPUInterfaceTests` and at the `coverage` job, for the reason
   above.

   ```bash
   cd ~/npu-mlir-v2 && git push -u origin phase/p2-npuisa-dialect
   ```

3. **Prove the `NPUInterfaceTests` activation red, then green.** Section 19.0
   requires this of every step on the day it switches on. The perturbation is
   named here rather than left to be invented, and it is chosen to break the
   *assertion* rather than the build, because a test binary that fails to compile
   is caught by the build step and would prove the wrong thing:

   ```bash
   # In unittests/Dialect/NPUISA/InterfaceTest.cpp, in the test
   # EveryComputeOperationHasARow, change the expected count from 10 to 9.
   sed -i 's/EXPECT_EQ(covered.size(), 10u)/EXPECT_EQ(covered.size(), 9u)/' \
     unittests/Dialect/NPUISA/InterfaceTest.cpp
   ```

   Commit that alone on a scratch branch, open a pull request so the
   `pull_request` trigger fires, and show `build-and-test` failing at the step
   named `NPUInterfaceTests (activation table: P2, on)` with the gtest failure
   naming `EveryComputeOperationHasARow`. Then revert, show green, record both
   run URLs in `docs/ENGINEERING_LOG.md`, and delete the scratch branch.

   A second perturbation is available if a stronger proof is wanted, and it
   exercises the product rather than the test: change `isa_and_present` back to
   `isa` in one predicate in `NPUISATypes.td`. That turns `check-npu` red as well
   as the unit tests, through a segmentation fault, which is a truthful
   demonstration of D-0008 but a noisier run log. The count perturbation is the
   cleaner proof of *this step*.

4. **The coverage job, and why it gets a different treatment.** At a threshold of
   0 the threshold comparison **cannot** be made to fail, because no measured
   percentage is below zero. Perturbing the threshold to force a red would be
   perturbing the gate's configuration rather than the thing it guards, which
   proves nothing about the gate as configured. So the honest statement is:

   - What **is** provable now, and should be the red proof recorded for this
     job: break a test and show the coverage job red *before it ever reports a
     number*. That is rule 2 of Section 17.7 working, coverage is only counted
     from a run where every test passed, and it is the part of this job that
     gates on something today. The same one line perturbation from step 3 does
     it, since `scripts/coverage.sh` runs `NPUInterfaceTests`; the finding to
     record is that one fault turns two jobs red, `build-and-test` and
     `coverage`, which is the two nets agreeing rather than a surprise.
   - What is **not** provable until P8: that the threshold gate rejects a real
     regression. **P8 is where that proof belongs**, because P8 is where the
     thresholds stop being 0 and are set from measured values. The perturbation
     at P8 is to lower coverage by deleting a test and show the job red against
     a real floor. This is written here so P8 inherits the obligation rather than
     discovering it.

   Locally the threshold arm was proved breakable at a non zero threshold,
   `bash scripts/coverage.sh 99` exits 1 and names both numbers, and a
   non numeric threshold exits 2. That is evidence the arm works; it is not the
   CI proof, and it is labelled as such.

5. **Merge, remembering that P1 is still unmerged.** `main` is at P0 and this
   branch carries both phases. Either merge `phase/p1-npu-dialect` into `main`
   first and then `phase/p2-npuisa-dialect`, which keeps one pull request per
   gate and is the shape ground rule 11 implies, or merge this branch once and
   say in the pull request body that it closes both gates. The first is tidier
   and the second is fewer runs; either is defensible, but doing it without
   noticing produces a P1 merge that appears to contain P2's work.

## Open questions

Four. None blocks the gate.

**`test/Dialect/NPUISA/dma-boundaries.mlir` does not exist yet, and Section 8
names it.** The scoped DMA boundary invariant is asserted "immediately after
lowering and before allocation", and there is no lowering until P4. A file
asserting the invariant at P2 would have nothing to run it against: it could only
hand write IR that already satisfies the shape, which asserts that I can write a
correct example rather than that the pipeline produces one. The file therefore
belongs to P4, with the lowering pass, and `docs/ARCHITECTURE.md` records the
invariant and its three permitted producers now so that P4 implements against a
written rule. Flagging it because a reader who greps Section 8 for that filename
will not find it and should know why.

**`npuisa.concat` cannot express an empty operand list in its custom assembly
syntax.** `ins()` and `ins( : )` are both parse errors, because the format prints
the variadic operands and their types around a literal `:` and with no operands
there is nothing on either side for the parser to latch onto. The verifier rule
that rejects an empty concatenation is still correct and still needed, because a
pass building the operation programmatically can produce one, and the test spells
it in the generic form. The question I am leaving open rather than deciding
unilaterally is whether the assembly format should grow an optional group so the
empty case is writable. My inclination is no: it would add syntax for a form
nobody should write, and the generic form already covers the test. Recorded so
that it is a decision rather than an omission.

**`docs/DIALECT_REFERENCE.md` covers the `npu` dialect only, and the `npuisa`
dialect has no generated reference.** That is deliberate and matches the
specification rather than being an omission: Section 5.4 puts the opcode table,
operand encoding, semantics, memory model, validation rules, byte order policy
and version policy in `docs/ISA_MANUAL.md`, written like a small processor
manual, and the activation table schedules its staleness gate at P6 with the
binary format it documents. So the `npuisa` operations are documented today by
their ODS descriptions, which are unusually full, and by
`docs/ARCHITECTURE.md`; the manual and its gate arrive at P6. The `npu-dialect-doc`
target and its CI staleness step are unchanged and still green. Recorded because
a reader who sees a P1 dialect with a generated reference and a P2 dialect
without one will reasonably wonder which of the two is the mistake.

**The gtest fallback depends on `libgtest-dev` shipping prebuilt libraries.** On
this base image it ships `libgtest.a`, `libgtest_main.a` and a CMake package
config, which is what `find_package(GTest)` needs. Older Debian derivatives
shipped headers and sources only, with no library and no config, and on one of
those the fallback would find nothing and the unit tests would silently not
build. The Dockerfile smoke test checks for the config file and both libraries
explicitly rather than trusting the package name, so a base image change that
regressed this fails the image build rather than producing an image whose CI
quietly stops running the unit tests. That is the mitigation; the dependency
itself remains.

## Next command

```bash
gh workflow run llvm-image.yml --ref phase/p2-npuisa-dialect
```

Wait for it to publish, then push the branch, then perform the proofs in the
order given, then open the merge pull request.

## Next phase

**P3, the ONNX frontend and the model suite.** The importer per Section 11 and
the seven model generator per Section 15, at the opset P0 resolved.

P3 inherits two things from P2 that it would otherwise have had to do itself.
The image already carries the MLIR Python bindings at
`/opt/llvm/python_packages/mlir_core`, already on `PYTHONPATH`, so the pytest
activation at P3 needs the importer and its tests and no image republish. And
`onnxscript` must be installed before that phase starts: torch's dynamo exporter
imports it and Section 15 forbids the `dynamo=False` escape, so a missing
`onnxscript` is the single most likely blocker there.

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
