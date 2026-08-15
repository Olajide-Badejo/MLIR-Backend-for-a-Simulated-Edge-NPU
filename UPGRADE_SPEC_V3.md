# MLIR Backend for a Simulated Edge NPU: Upgrade Specification v3

## 0. How to use this document

Paste this entire file as the opening message of a fresh Claude Code session on the
target machine (Section 3). **This is not a build specification. It is an upgrade
specification.** A working v1.0.0 repository already exists at
`/home/elijah/npu-mlir` and it passes all of its tests. Your job is to raise it to
a substantially higher standard **without breaking what already works**.

Before writing a single line of code you must complete Phase U0, which captures a
machine checkable baseline of the current behaviour. Every later phase is gated on
that baseline staying green. Work the phases in order. Do not start a phase until
the previous phase's gate passes with output shown.

If genuinely blocked, ask one specific question rather than drifting from the spec.

### 0.1 Read these first, in this order

1. `docs/ASSESSMENT.md` in the repository. This is the audit that produced this
   specification. It contains the evidence, the reproductions, and the reasoning
   for every change requested here. **Every phase below cites its ASSESSMENT item
   numbers. When this document and the assessment disagree, this document wins,
   but say so and explain why.**
2. `README.md`, `docs/ARCHITECTURE.md`, `docs/ISA_MANUAL.md`, `docs/PASSES.md`.
3. The source, in this order: `include/NPU/Dialect/NPU/IR/NPUOps.td`,
   `lib/Dialect/NPU/IR/NPUOps.cpp`, `lib/Dialect/NPUISA/Transforms/`,
   `lib/Encoding/`, `lib/Simulator/Simulator.cpp`, `python/npu_frontend/`.
4. `npu_mlir_compiler_spec.md` (the v2 build specification) for historical context
   only. **It is superseded by this document.** Several of its pinned assumptions
   turned out to be wrong for this machine; see Section 6.

### 0.2 The prime directive

> **Do not break working behaviour. If you must, say so explicitly, in writing,
> before you do it, and record it as a deliberate breaking change.**

The repository currently passes 13 lit tests, 11 GoogleTests, and 12 pytest tests,
and produces a LeNet result that matches onnxruntime to 3e-8. Those are the crown
jewels. Every phase gate re-runs all of them. A phase is not complete if the
baseline regressed and you have not documented why the regression is correct.

### 0.3 Pinned assumptions (change deliberately, not accidentally)

Carried over from v2 and still true:

- Dialect namespaces: `npu` (tensor level), `npuisa` (instruction level).
- LLVM/MLIR pinned at tag `llvmorg-22.1.8`, prebuilt at
  `/home/elijah/llvm-project/build`. **Never rebuild it.**
- Python venv at `/home/elijah/npu-venv`.
- Conventional Commits, one logical change each.
- No em dashes, no en dashes anywhere, enforced by `scripts/dash-lint.sh`.

Corrected or newly pinned by this document:

- **License is MIT**, not Apache-2.0 with LLVM exceptions. The v2 spec said
  Apache-2.0; commit `48a0be3` deliberately relicensed. Keep MIT and record the
  reason in `docs/DESIGN_DECISIONS.md` (ASSESSMENT 3.7).
- **C++20**, set in `CMakeLists.txt` at commit `d797ab5`. The v2 spec said C++17.
- **Python is 3.14** in the venv, not 3.10+.
- **The build environment is capped at 12 GB of RAM.** See Section 3. The v2 spec's
  suggestion of `memory=20GB, processors=24` is wrong for this machine and must not
  be applied.
- **Core scope is no longer fp32 CNN inference at batch 1.** See Section 5. This
  single sentence in the v2 spec is the root cause of most defects found in the
  audit, and widening it is the most important change in this document.
- **Simulator strategy is decided:** keep the in house simulator as the correctness
  oracle, add SCALE-Sim for cycles and Accelergy for energy. See Section 5.4 and
  ASSESSMENT sections 10.3 through 10.7. Do not replace the in house simulator.

### 0.4 Status as of 2026-08-09 (second audit)

The repository was re-audited on 2026-08-09; the full evidence is
`docs/ASSESSMENT.md` section 13. Where this spec and that section disagree
about the current state of the tree, section 13 is newer and wins. Standing:

- **Phases U0, U1, U2 are implemented and merged locally** (`main` at
  `6d62406`). Their code gates pass on this machine.
- **Phase U2 is not finished by its own gate:** nothing since 2026-08-07 has
  been pushed, so the new CI has never run on GitHub, the README badge still
  reports the old lint only workflow, the GHCR LLVM image does not exist yet
  (anonymous manifest probe returns 403), and the four fault class proof runs
  of Section 10.2 are outstanding. Completion requirements are in Section
  10.4.
- **Phase U3 is in flight, uncommitted, on the `main` checkout.** 39 of its 42
  encoding tests pass. The re-audit found five defects that must be fixed
  before it lands; they are itemized in Section 11.4 and in ASSESSMENT 13.2.
- **The committed PDFs are stale** and still embed numbers traced to the
  nonexistent commit `8095dbec`; regenerating results and rebuilding both PDFs
  after U3 lands is a required step (ASSESSMENT 13.3 and 13.4 item 3).
- **The remaining work is serialized into 65 consecutive, independently
  executable parts in Section 25**, published as
  `upgrade_parts/UPGRADE_SPEC_V3_Part_1.md` through `_Part_65.md`. One agent
  session implements one part; each part restates its own context, so a fresh
  agent needs only its part file and this repository.

---

## 1. Aim and objectives

**Aim.** Take a working but narrowly scoped MLIR compiler backend and raise it to a
standard where every claim it makes is independently verified, every number it
reports is cross checked against an external model, and every operator it defines
is reachable, tested, and exercised by a real model. Do this incrementally, on
branches, with the existing test suite as a continuous regression net, leaving a
documented trail of what changed and why.

**Objectives.**

1. **Eliminate silent wrong answers.** No input may produce an incorrect numerical
   result without a diagnostic. Today, batch size greater than one does exactly
   that (ASSESSMENT 2.1).
2. **Close the reachability gaps.** Every op defined in the `npu` dialect must be
   importable from ONNX, lowerable to `npuisa`, encodable, simulatable, and
   exercised by at least one model in the benchmark suite. Today three ops cannot
   be lowered at all and the flagship BatchNorm pass is unreachable (ASSESSMENT
   2.2, 3.2).
3. **Make the evaluation honest and broad.** A model suite rather than one model,
   per pass ablations, cycle counts cross checked against SCALE-Sim, energy
   numbers from Accelergy, and results committed and traceable to the commit that
   produced them (ASSESSMENT 3.4, 3.5, 4.2, 4.3).
4. **Make CI mean something.** It must build the project and run every suite, and
   it must be proven capable of failing (ASSESSMENT 4.1).
5. **Harden the binary interface.** The `.nbin` format must be validated on decode
   and the simulator must not perform unchecked memory access (ASSESSMENT 2.3).
6. **Then, and only then, go deep:** cost aware spilling, a memref based memory
   model, quantization, tiling, and optionally a real hardware target.

---

## 2. Estimated effort

**ESTIMATED EFFORT PER PHASE IS GIVEN IN EACH PHASE HEADING. TOTAL FOR PHASES U0
THROUGH U6 (THE MANDATORY CORE): 10 TO 13 WORKING SESSIONS.**

**PHASES U7 THROUGH U12 ARE A MENU, NOT A SEQUENCE, EXCEPT WHERE DEPENDENCIES ARE
STATED. TOTAL IF ALL ARE TAKEN: A FURTHER 22 TO 33 SESSIONS.**

**PHASE U13 (GEMMINI) IS OPTIONAL AND ADDS 11 TO 18 SESSIONS ON TOP.**

**THE INCREMENTAL PROJECT BUILD IS SECONDS. THE FULL TEST SUITE IS UNDER 30
SECONDS. THERE IS NO LONG BUILD IN THIS PROJECT: LLVM IS ALREADY BUILT AND MUST NOT
BE REBUILT.**

Estimates, not measurements. Replace them with measured wall clock in
`docs/ENGINEERING_LOG.md` as they become real.

These totals are higher than the 25 to 35 sessions quoted in `docs/ASSESSMENT.md`
section 11.5, and deliberately so. That figure costed the roadmap items alone. This
document additionally requires the Phase U0 baseline, the test layer matrix in
Section 9, the fuzz corpus, the reachability check, and the proof of failure gate,
none of which are roadmap items. The extra sessions buy the regression safety that
makes the rest of the work safe to do at all. If effort is constrained, cut phases
from the U7 to U12 menu rather than cutting the testing requirements.

**Stop points.** After U1 the project has no silent failures. After U2 its CI is
real. After U6 every claim in the README is true. Each of those is a legitimate
place to stop and ship. Do not treat the full list as all or nothing.

---

## 3. Target machine and environment

| Component | Actual, verified 2026-08-08 |
|---|---|
| CPU | Intel Core i7-14700K, 8 P plus 12 E cores, 28 threads |
| Host RAM | 31.7 GB |
| WSL2 RAM | **12 GB, plus 8 GB swap. This is a hard ceiling.** |
| WSL2 processors | 28 |
| WSL2 disk | 897 GB free on `/dev/sdd`. Not a constraint. |
| OS | Windows 11 Pro, WSL2 Ubuntu 26.04 |
| Toolchain | gcc 15.2.0, cmake 4.2.3, ninja 1.13.2, python 3.14.4 |

### 3.1 The memory ceiling is not negotiable

`~/.wslconfig` sets `memory=12GB` deliberately. Its absence crashed the machine on
2026-07-14. **Do not edit it. Do not suggest editing it.** The v2 specification's
recommendation of `memory=20GB, processors=24` predates that crash and is wrong.

Practical consequences:

- Keep memory hungry build parallelism at or below 6 jobs. Test parallelism can use
  all 28.
- The project itself builds in seconds and is nowhere near the ceiling. The ceiling
  only matters if Phase U13 is attempted.

### 3.2 Build and test commands

```bash
cd ~/npu-mlir
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld
ninja -C build -j6
ninja -C build check-npu          # lit and FileCheck
./build/bin/NPUEncodingTests      # GoogleTest
./build/bin/NPUSimulatorTests     # GoogleTest
source ~/npu-venv/bin/activate && python -m pytest test/Python -q
bash scripts/dash-lint.sh
```

Multi line commands invoked from the Windows side must be written to a `.sh` file
and run as `wsl -d Ubuntu -- bash <path>`; nested quoting through PowerShell
mangles otherwise.

---

## 4. Ground rules

Rules 1 through 6 are inherited from the v2 specification and still bind. Rules 7
through 14 are new and specific to upgrade work.

1. **Never fabricate** a benchmark number, test result, citation, or API detail.
   Unverifiable means unstated.
2. **Never mark a phase complete without running its tests and showing output.**
3. **No em dashes, no en dashes anywhere** (U+2014, U+2013): code, comments,
   commits, markdown, LaTeX. Enforced by `scripts/dash-lint.sh` in pre-commit and
   CI.
4. **Simulated numbers are labeled simulated estimates, never measurements.**
5. **Write as the repo owner.** First person, varied sentence length, concrete
   numbers over vague qualifiers, prose in paragraphs not bullet walls, no AI stock
   phrasing.
6. Conventional Commits, one logical change each, incremental. No TODOs in shipped
   code.
7. **The baseline is sacred.** Every phase gate re-runs the full baseline from
   Phase U0. A phase with a regressed baseline is not complete. If a regression is
   intentional, it must be recorded in `docs/BREAKING_CHANGES.md` with the reason,
   before the commit that causes it.
8. **No silent behaviour changes.** If a change alters any output that a user or a
   test could observe (numerics, instruction counts, cycle counts, file formats,
   CLI flags, error messages), it must be named in the commit body and in
   `CHANGELOG.md`.
9. **Every bug fixed gets a regression test that fails before the fix and passes
   after.** Show both runs in the phase report. A fix without a failing-first test
   does not count as fixed.
10. **Every new feature ships with tests at every layer it touches.** See Section 9
    for the layer matrix. A pass needs a lit test. An op needs a round trip test
    and a verifier failure test. A simulator kernel needs a GoogleTest. An importer
    op needs a pytest. Anything end to end needs an e2e test.
11. **Work on branches.** One branch per phase, named `upgrade/uN-short-name`.
    Merge only when the gate passes. Never commit directly to `main`.
12. **Update the documentation in the same commit as the code.** A pass that
    changes behaviour updates `docs/PASSES.md` in the same commit, not later.
13. **Maintain `docs/ENGINEERING_LOG.md` continuously**, in the existing style:
    dated entries with symptom, root cause, options considered, chosen fix and why,
    commit, verification. This is a first class deliverable, not a chore. The debug
    report is built from it.
14. **Check items off `docs/ASSESSMENT.md` as they land**, by appending a status
    marker to the roadmap item. Do not delete the item text. The assessment
    becomes the project's audit trail.

---

## 5. Scope changes: what this specification widens, and why

**This section is the most important in the document.** The audit found that most
defects were not implementation mistakes. They were faithful implementations of a
specification whose scope was too narrow. Widening the scope statements is
therefore the fix that prevents recurrence.

### 5.1 Core scope

The v2 specification said:

> Core scope: fp32 CNN inference graphs. INT8 quantization is a stretch goal.

and

> a LeNet style CNN for the core suite; a depthwise separable block as stretch.

That produced a compiler correct for exactly one model at exactly one batch size.
Replace it with:

> **Core scope: fp32 CNN inference graphs at arbitrary static batch size, over a
> suite of at least six structurally distinct models that between them exercise
> every operator defined in the `npu` dialect. INT8 quantization is a defined
> phase, not a stretch goal.**

### 5.2 The reachability rule (new, and binding)

> **No operator may exist in the `npu` dialect unless it is reachable end to end:
> importable from ONNX, lowerable to `npuisa`, encodable, simulatable, and
> exercised by at least one model in the benchmark suite. An operator that cannot
> satisfy this must be deleted from the dialect or must have a tracked, dated
> exemption in `docs/DESIGN_DECISIONS.md` stating when it will be completed.**

A CI check enforces this; see Section 9.7. This single rule would have prevented
ASSESSMENT 2.2 and 3.2.

### 5.3 The no silent failure rule (new, and binding)

> **Any input the compiler accepts must produce either a correct result or a
> diagnostic. There is no third option. Where a limitation exists, it is enforced
> at the earliest possible layer, with a message naming the offending construct.**

### 5.4 Measurement strategy (decided, do not relitigate)

- **The in house simulator remains the correctness oracle.** Twelve established
  simulators were evaluated (ASSESSMENT 10). None validates whole program numerics
  against onnxruntime. Do not replace it.
- **SCALE-Sim provides cross checked cycle counts.** The analytical model stays;
  the comparison between the two becomes an evaluation subsection.
- **Accelergy provides energy and area**, which nothing currently produces and
  which is the metric that actually justifies an edge NPU's design.
- **ZigZag is available for tiling design space exploration** in Phase U12.

### 5.5 Explicitly out of scope

Do not do these unless a later phase names them:

- Rebuilding LLVM. It is built. Link against it.
- Migrating the frontend to TOSA or Linalg as the primary path. The hand written
  importer stays (ASSESSMENT 9.3). A TOSA path may be added later as an
  alternative, never as a replacement.
- Graph coloring register allocation. The ISA is straight line by design, so
  interval based allocation is already optimal in colours used (ASSESSMENT 9.3).
- Training, dynamic shapes, or control flow.
- Replacing the in house simulator.

---

## 6. What was wrong with v2, so it is not repeated

Recorded so the next revision of this document does not undo the fix.

| v2 statement | What it produced | v3 replacement |
|---|---|---|
| "LeNet style CNN for the core suite" | One model. Every generalization in the report is from a sample of one. | Section 5.1, six model minimum |
| "fp32 CNN inference graphs" with no batch statement | Batch hardcoded to 1 in the simulator, silently wrong for N > 1 | Section 5.1, arbitrary static batch |
| Op set listed in Section 5, with no reachability requirement | Three ops defined but unlowerable; BatchNorm pass unreachable | Section 5.2, the reachability rule |
| "CI on a prebuilt LLVM Docker image" | A lint only workflow wearing a CI badge | Section 10, plus the proof of failure gate in 10.2 |
| "every report number traces to `experiments/results/`" | Results untracked, and stale relative to HEAD | Section 8.1 items 3 and 4, committed and manifest checked |
| "at least 80 percent line coverage" | A coverage script that ignores test failures | Section 9.8 |
| `memory=20GB, processors=24` | Wrong for this machine; the crash it caused is documented | Section 3.1 |

---

## 7. Phase U0: baseline and safety net (1 session, mandatory, first)

**No other phase may begin until this one's gate passes.** You are building the net
before you start walking the wire.

### 7.1 Changes

1. Create `scripts/regression-baseline.sh`. It must:
   - Build the project and run all four suites (lit, both GoogleTest binaries,
     pytest) plus `dash-lint.sh`.
   - Record, in a machine readable JSON at `test/baseline/baseline.json`: the pass
     and fail count of each suite, the list of test names per suite, the current
     git sha, and the tool versions.
   - Compile LeNet at `-O0`, `-O1`, `-O2` at both the default 1 MB and the tight
     140 KB budget, and record for each: the encoded instruction count read from
     the simulator's own `stats.instructions` (**not** the regex count, ASSESSMENT
     4.3), simulated cycles, DRAM bytes read and written, and the max absolute
     error versus onnxruntime.
   - Support `--check`, which re-runs everything and diffs against the recorded
     baseline, exiting nonzero on any drift, with a readable report of what moved.
2. Create `test/baseline/golden/` holding the fp32 output tensors of every model in
   the suite at every optimization level, as `.npy`. These are numerical golden
   files. A change that moves any of them by more than 1e-6 must be justified.
3. Create `docs/BREAKING_CHANGES.md`, empty apart from a header explaining its
   purpose (rule 7).
4. Add a `make baseline` or `ninja baseline-check` convenience target.

### 7.2 Tests and verification

- Run `scripts/regression-baseline.sh` to record.
- Run `scripts/regression-baseline.sh --check` immediately afterwards on unchanged
  code. **It must report zero drift.** Show the output.
- Deliberately perturb one cost model constant, re-run `--check`, and show that it
  **fails** and names the metric that moved. Then revert. This proves the net
  catches things (the same discipline as the proof of failure gate in Section 10.2).

### 7.3 Gate

- [ ] `baseline.json` committed, with the current sha recorded
- [ ] `--check` reports zero drift on unchanged code, output shown
- [ ] `--check` demonstrably fails on a deliberate perturbation, output shown
- [ ] Golden output tensors committed
- [ ] `docs/BREAKING_CHANGES.md` exists
- [ ] `dash-lint.sh` clean

Commit: `test: record regression baseline and golden outputs before upgrade work`

---

## 8. Phase U1: stop being wrong (1 to 2 sessions, mandatory)

ASSESSMENT roadmap items 1 through 6. This phase closes every defect that causes a
wrong answer or a false claim, and does nothing else.

### 8.1 Changes

1. **Reject batch size other than 1, loudly** (ASSESSMENT 2.1). Until Phase U6 adds
   real batch support, the compiler must refuse rather than lie. Enforce at two
   layers:
   - `python/npu_frontend/op_mapping.py`: `tensor_type` or a new check raises
     `UnsupportedOpError` naming the tensor and its shape.
   - `NPUOps.cpp` verifiers for `conv2d`, the pools, and `batch_norm`: emit an op
     error if `getDimSize(0) != 1`.

   The message must state that this is a current limitation with a tracked plan,
   not a permanent design choice.
2. **Correct `op_mapping.py`'s module docstring** to list the seven operators
   actually implemented, and to describe the remaining six as not yet implemented
   with a pointer to Phase U7 (ASSESSMENT 3.1).
3. **Commit `experiments/results/*.json`** and regenerate them at the current sha
   (ASSESSMENT 4.2). `.gitignore` currently contains the line
   `/experiments/results/*.json`, which is why they were never tracked. Delete that
   line. Keep `/experiments/results/*.tmp` ignored, since those are the atomic
   write temporaries. Consider whether `/experiments/models/*.onnx` should stay
   ignored: models are regenerated from a seed, so ignoring them is correct, but
   the seed and the generator version must then be in the result manifest.
4. **Make `valid()` in `run_benchmarks.py` manifest aware** (ASSESSMENT 4.2): a
   result is stale, and must be regenerated, if its `manifest.git_sha` differs
   from the current sha or its `manifest.cost_model` differs from the current
   constants. Add `--allow-stale` for deliberate reuse.
5. **Delete `|| true` from `scripts/coverage.sh`** (ASSESSMENT 4.6) and fix the
   `build-coverage` versus `build-cov` directory inconsistency.
6. **Tighten the end to end tolerance** (ASSESSMENT 4.4). The observed error is
   3e-8; the assertion is 1e-3. Set it to a value close to observed, for example
   `rtol=1e-5, atol=1e-6`, so a real numerics regression fails. Record the observed
   value in the test as a comment.

### 8.2 Tests

| Test | Layer | Asserts |
|---|---|---|
| `test/Python/test_batch_guard.py` | pytest | importing a batch 2 ONNX model raises `UnsupportedOpError` naming the shape |
| `test/Dialect/NPU/invalid.mlir` (extend) | lit | a batch 2 `npu.conv2d` fails the verifier with the expected message |
| `test/Python/test_benchmarks.py` (new) | pytest | `valid()` returns False for a result whose manifest sha differs |
| existing suites | all | unchanged |

**Failing first is required** (rule 9). Show the batch guard test failing before
the guard is added.

### 8.3 Gate

- [ ] All new tests pass; all baseline tests pass
- [ ] `regression-baseline.sh --check` clean, **except** the deliberately tightened
      tolerance, which is recorded in `CHANGELOG.md`
- [ ] `experiments/results/*.json` committed and regenerated at HEAD
- [ ] `coverage.sh` fails when a test fails, demonstrated
- [ ] `docs/ASSESSMENT.md` items 1 to 6 marked done
- [ ] `dash-lint.sh` clean

---

## 9. Testing requirements: the layer matrix

This section is normative for every phase after U1. **More is required here than in
the v2 specification, deliberately.**

### 9.1 Required layers by change type

| If you change | You must add or update |
|---|---|
| An ODS op definition | lit round trip test in `test/Dialect/*/ops.mlir`, at least one verifier failure case in `invalid.mlir`, and a `DIALECT_REFERENCE.md` regeneration |
| A pass | lit before/after test with `CHECK` and `CHECK-NOT`, including a negative case proving it does **not** fire when it should not |
| A lowering pattern | lit test in `test/Dialect/NPUISA/lowering.mlir`, plus an e2e test if it is reachable from ONNX |
| A simulator kernel | GoogleTest asserting numerics against a hand computed expected value, including a padding case and an edge case |
| The binary format | encoder/decoder round trip GoogleTest, **plus** a malformed input test (Section 9.5) |
| An importer operator | pytest for the op in isolation, plus an ONNX model that uses it in the suite |
| The allocator | lit tests for fits, spill, fragmentation, and the budget-too-small error path |
| Anything numerical | golden file comparison against `test/baseline/golden/` |
| Anything user visible | `CHANGELOG.md` entry in the same commit |

### 9.2 The negative test rule

**Every pass must have at least one test proving it does not fire when it should
not.** The existing `fuse-conv-bias-relu.mlir` does this correctly with
`@no_double_fuse`. Follow that pattern. A pass with only positive tests is not
adequately tested, because a pass that fires unconditionally would pass them all.

### 9.3 Simulator semantics coverage (currently incomplete)

ASSESSMENT 4.5 found the simulator GoogleTests cover Relu, MatMul, Conv2D, Add,
Mul, AvgPool, and Reshape. **Every opcode in `docs/ISA_MANUAL.md` must have at
least one semantics test.** Currently missing and required:

- `POOL_MAX`
- `DMA_LOAD` and `DMA_STORE` (including a spill round trip)
- grouped convolution (`group > 1`), which is implemented but never tested
- depthwise convolution (`group == C`), which is the `group` path's important case
- dilated convolution (`dilation > 1`), implemented but never tested
- asymmetric padding
- `NOP` and `HALT`

### 9.4 Property and round trip tests (new requirement)

Add `unittests/Encoding/PropertyTest.cpp`: generate randomized `Program` structures
(bounded sizes, valid opcodes, plausible shapes), encode, decode, and assert
structural equality. At least 1000 iterations with a fixed seed so failures
reproduce. This covers the encoder far better than the four hand written cases.

### 9.5 Malformed input tests (new requirement, security relevant)

Per ASSESSMENT 2.3, `Program::decode` currently trusts its input. After Phase U3
adds validation, add `unittests/Encoding/FuzzTest.cpp`:

- A corpus of at least 200 deliberately malformed `.nbin` byte strings: truncated
  at every section boundary, bit flipped, out of range opcodes, negative addresses,
  addresses beyond `scratchpadBytes`, operand count mismatches, oversized shape
  vectors, and a valid file with one byte changed at each of 100 random offsets.
- **Assert that every one is rejected cleanly by `Program::validate()` and that
  none causes a crash, hang, or out of bounds access.**
- Run the same corpus under ASan and UBSan in the CI `sanitizers` job (Section 10.1).

### 9.6 End to end matrix (expanded)

The v2 spec asked for "all models, all levels" and the implementation tested one
model at one fixed pass pipeline. The e2e suite must be a full cross product:

```
for model in MODELS:              # at least 6 after Phase U8
  for level in [0, 1, 2]:
    for budget in [default, tight]:
      for batch in [1, 4]:        # after Phase U6
        assert simulated ~= onnxruntime within stated rtol and atol
```

Mark the slow cells with a pytest marker so a fast subset runs by default and the
full matrix runs in CI and on demand.

Each cell asserts **both** absolute and relative tolerance, not just absolute
(ASSESSMENT 4.4), and validates against **multiple inputs**: a standard normal
draw, all zeros, all large positive, all large negative, and a draw designed to
straddle the ReLU knee. One random input is not evidence.

### 9.7 The reachability check (new, enforces Section 5.2)

Add `scripts/check-reachability.py`, run in CI. It must:

1. Parse the op list from `NPUOps.td`.
2. For each op, confirm there is a lowering pattern in `LowerNPUToNPUISA.cpp`, a
   case in `InstructionEncoder.cpp`, a case in `Simulator.cpp`, and a converter in
   `op_mapping.py`.
3. Confirm each op appears in at least one generated benchmark model's IR.
4. Fail with a per op table showing which layer is missing, unless the op is listed
   in an `EXEMPT` block in `docs/DESIGN_DECISIONS.md` with a dated plan.

This check failing is what would have caught ASSESSMENT 2.2 and 3.2 on the day they
were introduced.

### 9.8 Coverage

- `llvm-cov` or `gcovr` for C++, `pytest-cov` for Python.
- **At least 85 percent line coverage on `lib/Dialect/`, `lib/Encoding/`, and
  `lib/Simulator/`**, raised from the v2 target of 80.
- **Coverage is only counted from a run where every test passed.** The `|| true`
  removed in Phase U1 must not return.
- Branch coverage reported alongside line coverage for the allocator and the
  decoder, where the error paths matter most.

### 9.9 Performance regression guard

`regression-baseline.sh --check` doubles as a performance guard: simulated cycles,
DRAM bytes, and instruction counts are part of the baseline. Any phase that moves
them must say so and explain. An optimization pass that makes cycles go **up** is
not necessarily wrong (see the tight budget result in ASSESSMENT 5.1), but it must
never move silently.

---

## 10. Phase U2: CI that actually tests (1 to 2 sessions, mandatory)

ASSESSMENT roadmap item 7, and ASSESSMENT 4.1.

### 10.1 Changes

The remote is already configured as
`origin https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU.git`,
so GHCR and Actions are available without further setup. Verify push access before
starting this phase; everything in it depends on CI actually running.

1. Build `docker/Dockerfile.llvm` at tag `llvmorg-22.1.8` and publish it to GHCR.
   If a local build is too slow, build it in a GitHub Actions job and push from
   there. This image already exists in the repository and is used by nothing.
   **Building LLVM inside a 12 GB WSL guest will be slow and may thrash; prefer
   building the image in Actions, where the runner has more memory.**
2. Rewrite `.github/workflows/ci.yml` with these jobs:
   - `lint`: the existing dash-lint and ruff job, plus `mypy` (item 12), plus
     `scripts/check-reachability.py` (Section 9.7).
   - `build-and-test`: pull the LLVM image, configure, build with `-j6`, run
     `check-npu`, both GoogleTest binaries, and the full pytest matrix.
   - `sanitizers`: the same build with `-fsanitize=address,undefined`, running the
     GoogleTests and the fuzz corpus from Section 9.5.
   - `coverage`: run `scripts/coverage.sh` and fail if below the Section 9.8
     thresholds.
3. Add `.github/workflows/report.yml` building both PDFs and uploading them as
   artifacts (ASSESSMENT 3.7).
4. Cache the build directory and ccache between runs, keyed on the LLVM tag.

### 10.2 The proof of failure gate (this is the important part)

A CI pipeline that has never failed is not known to work. Before merging:

1. On a scratch branch, introduce **four** deliberate faults, one at a time:
   an em dash in a markdown file; a failing lit test; a failing GoogleTest; a
   failing pytest.
2. Show CI going red for each, and show **which job** caught it.
3. Revert all four.
4. Paste the four red run URLs into the phase report.

### 10.3 Gate

- [ ] CI green on a clean run, all jobs, URL shown
- [ ] CI proven red for all four fault classes, URLs shown
- [ ] `report.yml` produces both PDFs as artifacts
- [ ] The README CI badge points at the real workflow
- [ ] Baseline check clean

### 10.4 Addendum 2026-08-09: completion requirements

The workflows exist and pass locally, but the 10.3 gate is unmet because
nothing has been pushed. To finish the phase (serialized as Part 6 in Section
25):

1. Push `main` and the `upgrade/*` branches to origin.
2. Dispatch `llvm-image.yml` and wait for
   `ghcr.io/olajide-badejo/npu-mlir-llvm:22.1.8` to exist; it currently does
   not (anonymous probe returns 403), so any earlier push shows a red badge at
   container pull. If the Actions build of the image exceeds the public
   runner's 6 hour job limit, build the same Dockerfile locally in WSL and
   `docker push` it once by hand; the workflow's manual dispatch design
   tolerates that.
3. Re-run CI, record the green URL, then run the four fault class proofs of
   Section 10.2 and record the four red URLs in `docs/ENGINEERING_LOG.md`.

---

## 11. Phase U3: harden the binary interface (1 session, mandatory)

ASSESSMENT roadmap item 8, and ASSESSMENT 2.3.

### 11.1 Changes

1. Add `Program::validate()` in `lib/Encoding/Program.cpp`, called by
   `Program::decode` and again by `npu-sim` before execution. It must check:
   - the `version` field matches the expected version (currently written and never
     read),
   - every opcode is within the valid enum range,
   - for each instruction, `resultAddr >= 0` and `resultAddr + resultBytes <=
     scratchpadBytes`,
   - every operand address is in range and its shape is known,
   - `operandAddrs.size()` matches the arity the opcode requires (`Conv2D` with
     fewer than two operands must be rejected, not indexed),
   - `dramAddr + regionBytes <= dramBytes` for DMA instructions,
   - shape vectors are non empty, have positive extents, and their product matches
     the region size.
2. Return a structured error naming the instruction index and the failed check, not
   a bare `nullopt`.
3. Add bounds checked accessors in `Simulator.cpp` replacing the raw `spAt`/`dramAt`
   lambdas. Assert in debug builds; return a diagnostic in release.
4. In `npu-sim.cpp`, compare the input file size against the declared input region
   and refuse a mismatch (currently a heap overflow, ASSESSMENT 2.3).
5. Make `npu-sim` write **all** outputs, not just `outputs.front()` (ASSESSMENT
   4.7).
6. Diagnose rather than silently ignore: `npu-translate` encoding only the first
   function, and the allocator ignoring blocks after the first (ASSESSMENT 4.7,
   5.6).

### 11.2 Tests

- The fuzz corpus of Section 9.5, at least 200 cases, all rejected cleanly.
- The property round trip of Section 9.4.
- A GoogleTest per validation rule, asserting the specific error message.
- ASan and UBSan clean over the whole corpus.

### 11.3 Gate

- [ ] 200+ malformed inputs, none crash, all rejected with a specific message
- [ ] ASan and UBSan clean
- [ ] Property test passes 1000 iterations
- [ ] Baseline check clean
- [ ] `docs/ISA_MANUAL.md` documents the validation rules and the version policy

### 11.4 Addendum 2026-08-09: defects in the in flight implementation

The working tree already holds most of this phase, uncommitted: `validate()`,
`decodeUnvalidated()`, the checked simulator accessor, the `npu-sim` and
`npu-translate` and allocator diagnostics, 32 validation tests, the property
test, and a 322 case fuzz corpus. 39 of 42 encoding tests pass. The re-audit
(ASSESSMENT 13.2, evidence in WSL `/tmp/npuverify/`) found what remains, and
these are now gate items:

1. `shapeElements()` checks its running product against the 2^40 cap only
   after multiplying, so the guard itself overflows (UBSan confirms at
   `Program.cpp:197`) and a shape like `{2^40, 2^24}` wraps to 0 and is
   accepted. Test `d > kLimit / n` before multiplying. This is why
   `Validation.RejectsShapeThatWouldOverflow` fails.
2. The simulator grows the scratchpad to cover every result write before
   execution (`Simulator.cpp:127`), which absorbs out of range result
   addresses and neutralizes the new result bounds check; it also computes
   `resultAddr + elements * 4` on unvalidated input, which can itself overflow
   or demand an absurd allocation. Gate the expansion behind validation or
   size strictly from the declared `scratchpadBytes`. This is why
   `Validation.SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory`
   fails.
3. `Validation.RejectsRegionPastTheEndOfDram` uses `dramOffset = 8190`, which
   is rejected by the alignment rule before the range rule it names. Use an
   aligned offset such as 8160.
4. `validate()` never compares operand read extents to what was written; the
   written before read walk checks membership only, so a validated program can
   still trap in the simulator, and interior over reads silently read stale
   data. Record each written buffer's element count and require the consumer's
   need to fit it.
5. The trap path contains `assert(false)`, so an assert enabled build aborts
   instead of returning `SimResult.error`, and the comment above it describes
   clamping the code does not do. Remove the assert and make graceful refusal
   the behavior in every build mode.

Also fold in while the files are open: probe the decoder's just under the
2^28 count cap allocation path in the corpus; make `encodeFunction`'s
`.Default` case return failure instead of emitting a diagnostic and
succeeding; have `npu-sim` either support multiple `--input` flags or refuse
multi input programs; correct the false "tagged records" phrase in both
`docs/ISA_MANUAL.md` and the `Program.h` header comment, and state the real
byte order policy (the helpers are host endian despite their comment). Move
the work to `upgrade/u3-harden-the-binary-interface` before committing (rule
11).

---

## 12. Phase U4: measurement integrity (2 sessions, mandatory)

ASSESSMENT roadmap items 9 through 14.

### 12.1 Changes

1. **Per pass ablations** (item 9, ASSESSMENT 3.5). This is the v2 spec's stated
   "backbone of the evaluation" and it was never built. `run_benchmarks.py` must
   record, per model:
   - IR op counts **before and after each individual pass**, obtained by running
     `npu-opt` one pass at a time, not by a regex over the final dump;
   - wall clock per pass, from `--mlir-timing` (which `npu-opt` already supports,
     ASSESSMENT 9.1);
   - a one-pass-at-a-time ablation: for each pass P in the `-O2` set, run `-O2`
     without P and record the delta.
2. **Replace the regex instruction count** with the simulator's
   `stats.instructions` (item 10, ASSESSMENT 4.3). The regex counts
   `npuisa.const`, which is data rather than an instruction, and matches inside
   type strings. **This changes the headline numbers. Record the correction in
   `CHANGELOG.md` and update the README table.**
3. **e2e at every optimization level** (item 11), per the Section 9.6 matrix.
4. **Add `mypy`** to `pyproject.toml`, `.pre-commit-config.yaml`, and CI (item 12).
   Type the `python/npu_frontend/` package fully.
5. **TTY aware progress bar in `npu-sim`** (item 13), per the v2 spec Section 9
   which required it and never got it. Single line, percent by instruction index,
   suppressed when stdout is not a TTY.
6. **Write up the tight budget regression** (item 14, ASSESSMENT 5.1). At the
   140 KB budget, `-O1` is 45 percent slower than `-O0` and moves 50 percent more
   DRAM. This is the most interesting empirical result in the repository and it is
   currently undiscussed. It needs a subsection in the evaluation with the numbers,
   a hypothesis, and the plan to address it in Phase U9.

### 12.2 Gate

- [ ] Every pass has before/after op counts and a wall clock number
- [ ] Every `-O2` pass has an ablation row
- [ ] Instruction counts come from the simulator; the change is documented
- [ ] Full e2e matrix passes
- [ ] `mypy` clean and in CI
- [ ] Progress bar works and is TTY aware, verified both ways
- [ ] Tight budget regression written up

### 12.3 Addendum 2026-08-09

Three additions from the re-audit (ASSESSMENT 13.4 items 4, 5, 9):

1. When replacing the regex count, note that the README's objdump excerpt
   ("21 instructions") already matches the simulator's true `-O2` count; it is
   the headline table that is wrong. Regenerate both from a real run so they
   agree.
2. mypy landed early in U2 but is scoped to `python/npu_frontend` and targets
   Python 3.12 while the venv runs 3.14. Widen the scope to `experiments/`,
   `scripts/`, and `test/Python`, and raise the target to the newest version
   the pinned tools support, recording why if it cannot reach 3.14.
3. Replace the CI coverage step's grep of the first `lines:` figure with a
   `gcovr --json-summary` parse, and add a Python coverage threshold to match
   the C++ one.

---

## 13. Phase U5: external cost model cross validation (2 to 3 sessions, mandatory)

ASSESSMENT roadmap items 15 through 18, decisions D1a and D2.

### 13.1 SCALE-Sim

1. Add `experiments/scalesim_export.py`: walk the allocated `npuisa` IR and emit a
   SCALE-Sim topology CSV plus an architecture config. The conv and matmul rows are
   a direct read of the shapes already on each op; the architecture config restates
   the existing `CostModel` constants (16x16 array, the scratchpad budget as SRAM
   size).
2. Record `scalesim_cycles` next to `simulated_cycles` in every result JSON, and
   add the SCALE-Sim version to the manifest alongside the LLVM tag.
3. Add a `cost_model_agreement` subsection to the evaluation: per layer analytical
   versus SCALE-Sim cycles, the ratio, and an honest account of the divergence.

**Expect disagreement, and predict its direction before measuring.** The analytical
model has no DMA and compute overlap; SCALE-Sim models double buffering. The
analytical model should therefore look pessimistic on compute bound layers and
optimistic on memory bound ones. State that prediction in the report, then show
whether it held. A predicted and confirmed divergence is far stronger evidence of
understanding than two numbers that happen to agree.

### 13.2 Accelergy

1. Add an explicit MAC count field to `Stats` (currently only implicit in
   `macCycles`).
2. Write an Accelergy estimation plug in adapter mapping `Stats` to action counts.
3. Record `energy_pj` and `area_mm2` in every result JSON.
4. Add an energy column to the README headline table.
5. **Re-argue fusion in energy terms** (item 17). Fusion keeps an intermediate in
   the scratchpad. In cycles that is a 2 percent win; in energy it is a large one,
   because a DRAM access costs orders of magnitude more than a scratchpad access.
   The evaluation currently makes the case with the metric that flatters it least.

### 13.3 Gate

- [ ] SCALE-Sim cycles recorded for every benchmark cell
- [ ] Divergence direction predicted in writing **before** measuring, then compared
- [ ] Accelergy energy and area recorded for every cell
- [ ] Fusion re-argued in energy terms with numbers
- [ ] `scale-sim` and `accelergy` pinned in dependencies and documented in
      `docs/BUILD.md`
- [ ] Baseline check clean apart from newly added fields

---

## 14. Phase U6: real batch support (2 sessions, mandatory)

ASSESSMENT roadmap item 19, closing ASSESSMENT 2.1 properly.

### 14.1 Changes

1. Add the batch loop to `conv2d` in `lib/Simulator/Simulator.cpp`, replacing
   `int64_t n = 0;`.
2. Add the batch dimension to `pool`, which currently iterates channels and never
   sees batch.
3. Add batched `matmul` (rank 3 operands) if the model suite needs it; otherwise
   document that matmul remains rank 2 by design.
4. Update the cost model so MAC and element counts scale with batch.
5. **Remove the Phase U1 guard** and replace it with a test that batch 4 works.

### 14.2 Tests

- GoogleTest: conv, pool, and matmul at batch 4, against hand computed values.
- The regression test from ASSESSMENT 2.1: a 2 batch conv plus relu compiled and
  simulated, asserting **both** batches match onnxruntime. This test must fail on
  the pre-fix code; show that.
- e2e matrix extended with `batch in [1, 4]`.
- A batched model added to `MODELS`.

### 14.3 Gate

- [ ] Batch 2 and batch 4 match onnxruntime on every model
- [ ] The pre-fix failing test is shown failing, then passing
- [ ] Phase U1's guard removed, its test replaced
- [ ] Baseline check clean; batch 1 numbers unchanged

---

## 15. Phase U7: operator coverage and lowering completeness (3 to 4 sessions)

ASSESSMENT roadmap items 20, 21, 24, 25, 26. This phase makes Section 5.2's
reachability rule true.

### 15.1 Changes

1. **Lower `transpose` and `concat`** (item 20): new `npuisa` opcodes, encoder
   cases, simulator kernels, `ISA_MANUAL.md` entries, opcode numbers appended (not
   renumbered, per the manual's own version policy).
2. **Handle `batch_norm` that did not fold**: decompose into mul plus add during
   lowering, so an unfolded batch norm is legal rather than a hard error.
3. **Import the missing operators** (item 21): `MatMul`, `Add`, `Identity`,
   `GlobalAveragePool`, `Concat`, `Clip`, `BatchNormalization`. `Identity` matters
   more than it looks: it is what blocks importing any model exported without
   constant folding, which is what blocks the BatchNorm pass (ASSESSMENT 3.2).
4. **Bias fusion** (item 24): a pattern folding `add(conv(x, w), b)` into the
   conv's bias operand. Rename or fix `fuse-conv-bias-relu.mlir`, whose name
   promises bias fusion that its input IR does not exercise (ASSESSMENT 3.3).
5. **`inferReturnTypes` and arithmetic shape verification** (item 25): implement
   `InferTypeOpInterface` for conv, matmul, and the pools. Verify that output
   spatial dimensions match what input, weight, strides, pads, and dilations imply;
   that bias length equals output channels; that `group` divides both channel
   counts; that strides and dilations are positive (ASSESSMENT 2.6).
6. **Enable `-cse` and `-sccp`** in `_passes_for_level` (item 26). They are already
   linked into `npu-opt`; this is two lines and yields two ablation rows.
7. Fix `_attr` in `op_mapping.py` to switch on `AttributeProto.type` rather than
   truthiness (ASSESSMENT 2.5), and handle `count_include_pad` on `AveragePool`
   or reject it (ASSESSMENT 2.4).
8. Guard the average pooling divide by zero on all-padding windows (ASSESSMENT
   2.4).

### 15.2 Gate

- [ ] `scripts/check-reachability.py` passes with no exemptions
- [ ] Every new op has a round trip test, a verifier failure test, a lowering test,
      a simulator semantics test, and an importer test
- [ ] Bias fusion has a positive and a negative lit test
- [ ] Shape verification rejects an arithmetically impossible conv, with a test
- [ ] `DIALECT_REFERENCE.md` regenerated; CI fails if it is stale
- [ ] Baseline check clean

---

## 16. Phase U8: the model suite and the BatchNorm pass (2 to 3 sessions)

ASSESSMENT roadmap items 22 and 23. This is the highest leverage phase in the
document: it converts an evaluation with a sample of one into a real one.

### 16.1 The suite

Add to `model_generator.MODELS`, all seeded, all generated, none downloaded:

| Model | Forces into the pipeline |
|---|---|
| LeNet (existing) | baseline, regression anchor |
| Depthwise separable block | grouped conv, pointwise conv, the `group` path |
| Small ResNet block | residual `Add`, identity shortcut |
| Small Inception block | `Concat`, parallel branches |
| Conv plus BatchNorm plus ReLU stack | the BatchNorm folding pass |
| Dilated conv stack | dilation, asymmetric pads |
| Batched LeNet (N=4) | the Phase U6 batch path |

Add `INPUT_SHAPES` entries for each. Every model must have a structural pytest
asserting its exported ONNX contains the ops it is supposed to.

### 16.2 Make the BatchNorm pass real

Per ASSESSMENT 3.2 the flagship pass is currently unreachable, because
`torch.onnx.export` constant folds BatchNorm into the Conv before the importer sees
it, and disabling that folding produces `Identity` nodes the importer rejects.

1. Export the batch norm model with `do_constant_folding=False` (needs `Identity`
   from Phase U7).
2. Wire `-npu-fold-batchnorm` into the `-O2` pipeline in `_passes_for_level`. It is
   currently in no optimization level at all.
3. Produce a real ablation row: `-O2` with and without the pass, on a model that
   actually contains batch norm.
4. Add an e2e numerical test proving the fold does not change results beyond
   tolerance.

### 16.3 Gate

- [ ] At least six models in the suite, all passing the full e2e matrix
- [ ] Every `npu` op exercised by at least one model, proven by the reachability
      check
- [ ] BatchNorm folding fires on a real imported model, with an ablation row
      showing a measured delta
- [ ] Full benchmark suite completes in under 30 minutes, measured
- [ ] README headline table regenerated from real results across the suite

---

## 17. Phase U9: allocator and performance (3 to 4 sessions)

ASSESSMENT roadmap items 27 through 30.

1. **Cost aware spill heuristic** (item 27). The current heuristic spills the
   longest live range at the peak, ignoring size and reload count. Replace with a
   Belady style cost: bytes moved times number of reloads. **Benchmark against the
   current heuristic as an ablation and report both**, and check whether it fixes
   the tight budget regression from ASSESSMENT 5.1. Keep the old heuristic behind a
   pass option so the comparison is reproducible.
2. **O(n log n) liveness and allocation** (item 28). `peakPressure` is O(ops x
   buffers) and is recomputed inside the spill loop. Use a sweep line over live
   range endpoints. Add a synthetic 5000 op function as a compile time benchmark.
3. **Two port cost model** (item 29): reassess after Phase U5. SCALE-Sim already
   models double buffering, so decide with divergence data in hand whether to
   complicate the analytical model or keep it deliberately simple.
4. **Parallel or im2col convolution** (item 30) so the suite can grow without the
   benchmark budget blowing out.

Gate: allocator tests for fits, spill, fragmentation, and budget-too-small; the
heuristic comparison recorded as an ablation; compile time benchmark showing the
complexity improvement.

---

## 18. Phase U10: memref and memory spaces (4 to 6 sessions, architectural)

ASSESSMENT roadmap items 31 and 32. **Own long lived branch. Must land before
Phase U11, since both change the type system.**

Replace `!npuisa.buffer` plus the `i64 address` attribute with
`memref<..., #npu.scratchpad>` and `#npu.dram`. This puts the project on MLIR's
bufferization and memory space infrastructure rather than beside it, and it is the
prerequisite for both quantization and any real hardware target.

Then implement `TilingInterface` and `DestinationStyleOpInterface`.

Gate: every existing lit test either passes unchanged or has a documented,
justified update; the full e2e matrix passes; numerics unchanged against the golden
files; `docs/ARCHITECTURE.md` and `docs/DESIGN_DECISIONS.md` updated with the
rationale.

---

## 19. Phase U11: INT8 quantization (4 to 6 sessions)

ASSESSMENT roadmap item 33. **Depends on Phase U10.**

`quantize` and `dequantize` ops, a calibration pass, `QUANT` and `DEQUANT` opcodes,
integer kernels in the simulator, and an accuracy versus cycles versus energy
evaluation. This is the single most domain relevant addition available, and with
Accelergy in place from Phase U5 the energy story becomes quantitative.

Gate: accuracy degradation measured and reported per model; the cycles and energy
win measured; a documented calibration methodology.

---

## 20. Phase U12: advanced optimization (menu, 6 to 10 sessions)

ASSESSMENT roadmap items 34 through 38. Take these individually, in any order,
except where noted.

- Asynchronous DMA with tokens plus double buffering (item 34).
- Region based `npu.fused_op` replacing the two case `Activation` enum (item 35).
- Layout assignment, NCHW versus NHWC (item 36).
- Tiling and loop fusion for convolutions exceeding the scratchpad (item 37,
  **depends on Phase U10**). This is the real edge NPU problem, currently sidestepped
  by spilling whole tensors.
- Debug section in `.nbin` mapping program counter to ONNX node name, plus
  `npu-sim --trace` (item 38). Cheap, and it turns the simulator into a debugging
  tool.
- ZigZag for tiling design space exploration.

---

## 21. Phase U13: Gemmini (optional, 11 to 18 sessions)

ASSESSMENT section 12 and roadmap item 39. **Read ASSESSMENT section 12 in full
before starting.** Gemmini's ISA forces Phases U10, U11, and the tiling item of
U12, because its array is fixed at 16x16, its scratchpad is row addressed, it has a
separate accumulator address space, and its default types are INT8.

Staged plan, gates, and machine requirements are in ASSESSMENT 12.5 and 12.6. Stage
0 is a toolchain smoke test with no compiler work; if it fails on 12 GB, stop.

The Gemmini backend lives in a sibling repository or behind a CMake flag; the
compiler work it depends on lives here.

---

## 22. Documentation requirements

Every phase updates documentation in the same commit as the code (rule 12).

| Document | Update trigger |
|---|---|
| `CHANGELOG.md` | every user visible change, Keep a Changelog format |
| `docs/ENGINEERING_LOG.md` | every non trivial problem encountered, continuously |
| `docs/BREAKING_CHANGES.md` | every deliberate baseline regression |
| `docs/ASSESSMENT.md` | mark items done as they land; never delete item text |
| `docs/PASSES.md` | any pass added or changed, with before/after IR |
| `docs/ISA_MANUAL.md` | any opcode, encoding, or validation rule change |
| `docs/ONNX_FRONTEND.md` | any importer operator change |
| `docs/DIALECT_REFERENCE.md` | regenerated by `ninja npu-dialect-doc`; CI fails if stale |
| `docs/DESIGN_DECISIONS.md` | any pinned assumption changed, any reachability exemption |
| `docs/BUILD.md` | any new dependency |
| `README.md` | any headline number, at the end of each phase |

**The README must never contain a number that is not reproducible from
`experiments/results/`.** ASSESSMENT 6 found an objdump excerpt that appears
illustrative rather than real; either regenerate it from an actual run or label it
explicitly as illustrative.

---

## 23. Definition of done

### 23.1 Mandatory core, Phases U0 through U6

- [ ] `regression-baseline.sh --check` green, with every intentional change
      recorded in `CHANGELOG.md` and `docs/BREAKING_CHANGES.md`
- [ ] No input produces a wrong numerical answer without a diagnostic
- [ ] Batch sizes 1 and 4 correct on every model, verified against onnxruntime
- [ ] CI builds the project and runs every suite, and is **proven** capable of
      failing for all four fault classes
- [ ] `Program::validate()` rejects 200+ malformed inputs cleanly; ASan and UBSan
      clean
- [ ] Per pass ablations and per pass timings recorded for every model
- [ ] SCALE-Sim cycles and Accelergy energy recorded alongside analytical estimates,
      with the divergence analysed
- [ ] At least six models in the suite; every `npu` op reachable end to end, proven
      by `check-reachability.py`
- [ ] BatchNorm folding fires on a real imported model with a measured ablation delta
- [ ] Coverage at or above 85 percent on `lib/Dialect/`, `lib/Encoding/`,
      `lib/Simulator/`, from a run where every test passed
- [ ] `mypy` clean; `dash-lint.sh` clean; `DIALECT_REFERENCE.md` fresh
- [ ] Every number in the README and both PDFs traces to a committed result file
      whose manifest sha matches the commit that produced it
- [ ] Both PDFs build with zero errors in CI
- [ ] `v2.0.0` tagged

### 23.2 Per phase, without exception

- [ ] All baseline tests pass
- [ ] New tests exist at every layer the change touches (Section 9.1)
- [ ] Every bug fix has a test shown failing before and passing after
- [ ] Every pass has a negative test
- [ ] Documentation updated in the same commit
- [ ] `docs/ASSESSMENT.md` items marked done
- [ ] Conventional Commit messages, one logical change each
- [ ] Phase report posted showing the gate checklist with real command output

---

## 24. What to do when this document is wrong

It will be, somewhere. The audit that produced it was thorough but it is not
omniscient, and the codebase will move underneath it.

- If a phase's premise turns out to be false (the bug does not reproduce, the fix
  is unnecessary, the design is already correct), **say so with evidence and skip
  it.** Do not implement a fix for a problem that does not exist.
- If you find a defect not in `docs/ASSESSMENT.md`, add it to the assessment with a
  reproduction, then fix it in the phase where it belongs.
- If a phase is much larger than estimated, stop at a working point, commit, and
  report the revised estimate rather than half landing a large change.
- If two requirements conflict, the prime directive (Section 0.2) wins, then the no
  silent failure rule (Section 5.3), then everything else.

---

## 25. The 65 part execution plan (added 2026-08-09)

The remaining work, from finishing the in flight Phase U3 through the full
U13 Gemmini backend, is serialized here into 65 consecutive parts. Each part
is sized for one agent session and is published as a self contained work
order at `upgrade_parts/UPGRADE_SPEC_V3_Part_N.md`. The intended workflow: a
fresh agent receives exactly one part file, implements it, passes its gate,
lands it, and stops; the next agent receives the next part.

### 25.1 The execution contract, binding for every part

1. Part N assumes parts 1 through N-1 are merged. Every part begins with a
   preflight that verifies this (suites green, expected artifacts present).
   If preflight fails, stop and report; do not repair earlier parts silently.
2. Every part obeys this spec's ground rules (Section 4), the prime directive
   (0.2), the no silent failure rule (5.3), the testing matrix (Section 9),
   and the machine constraints (Section 3: 12 GB ceiling, build at `-j6`,
   never rebuild LLVM).
3. Every part works on the phase branch its file names, lands via merge to
   `main` only when its gate passes with output shown, and updates
   `CHANGELOG.md`, `docs/ENGINEERING_LOG.md`, and the relevant docs in the
   same commits. Parts marked GATE also walk their phase's gate checklist
   from this spec.
4. A part that discovers its premise is false follows Section 24: evidence,
   skip, report. A part that discovers a new defect adds it to
   `docs/ASSESSMENT.md` and fixes it in the part where it belongs.
5. No part may leave the tree red. If a part cannot finish, it stops at a
   working point on its branch and reports what remains.

### 25.2 Part manifest

**Finishing Phase U3 (Section 11, addendum 11.4):**

- **Part 1 (U3).** Overflow safe `shapeElements()` in `Program.cpp` (test
  `d > kLimit / n` before multiplying) and the aligned `dramOffset` fix in
  `Validation.RejectsRegionPastTheEndOfDram`. Gate: those two tests green,
  UBSan clean on the validation suite, no other test moves.
- **Part 2 (U3).** Remove the simulator's pre validation scratchpad auto
  expansion (size strictly from declared `scratchpadBytes`; existing unit
  test programs get explicit sizes), and replace the `assert(false)` trap
  with graceful refusal in every build mode; fix the stale clamping comment.
  Gate: `Validation.SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory`
  green in both NDEBUG and asserts on builds, simulator suite green.
- **Part 3 (U3).** Close the remaining validate gaps: compare operand read
  extents against written element counts; probe the decoder's just under cap
  allocation path with new corpus cases; make `encodeFunction`'s `.Default`
  return failure; `npu-sim` multi input support or a counted refusal. Gate:
  new tests for each, 42 of 42 encoding tests green, fuzz corpus grown and
  clean.
- **Part 4 (U3, GATE).** U3 documentation and landing: ISA manual gains the
  validation rules and version policy, "tagged records" corrected in both the
  manual and `Program.h`, byte order policy stated; migrate the work to
  `upgrade/u3-harden-the-binary-interface`; full 11.3 gate including ASan,
  UBSan, 1000 iteration property test, baseline check; merge.

**Republishing truthful numbers (ASSESSMENT 13.3, 13.4):**

- **Part 5.** Regenerate the six benchmark results on the clean post U3 tree,
  rebuild both PDFs from them, and commit results, generated tex, and PDFs as
  separate commits. Gate: `test_committed_results_are_current` green, the
  local `macros.tex` sha matches the results manifests, no PDF in the tree
  traces to `8095dbec`.
- **Part 6 (closes U2 per 10.4).** Push `main` and branches, dispatch
  `llvm-image.yml` (local docker build and push fallback per 10.4), CI green
  URL recorded, then the four fault class proof runs of 10.2 with their four
  red URLs recorded in `docs/ENGINEERING_LOG.md`. Gate: 10.3 checklist fully
  satisfied.

**Phase U4, measurement integrity (Section 12):**

- **Part 7 (U4).** Replace the regex instruction count with the simulator's
  `stats.instructions` everywhere (harness, plots, README table, report),
  regenerate results, record the headline change in `CHANGELOG.md`, and
  regenerate the README objdump excerpt from a real run (12.3 item 1).
- **Part 8 (U4).** Per pass instrumentation: before and after op counts per
  individual pass and wall clock per pass from `--mlir-timing`, recorded per
  cell in the results schema.
- **Part 9 (U4).** Leave one out ablations for every `-O2` pass, results
  schema and plots extended, evaluation table updated.
- **Part 10 (U4).** The full e2e matrix of Section 9.6 at every level and
  both budgets with both tolerances and the five input classes; slow cells
  markered.
- **Part 11 (U4, GATE).** TTY aware progress bar in `npu-sim` (plus the
  projected wall clock line), the tight budget regression write up in the
  evaluation, mypy scope and version widening (12.3 item 2), coverage step
  moved to a JSON summary parse with a Python threshold (12.3 item 3); walk
  the 12.2 gate.

**Phase U5, external cross validation (Section 13):**

- **Part 12 (U5).** `experiments/scalesim_export.py` emitting the topology
  CSV and architecture config from allocated `npuisa` IR, with pytest
  coverage on shapes and edge cases.
- **Part 13 (U5).** Harness integration: `scalesim_cycles` per cell, SCALE-Sim
  version in the manifest, and the divergence direction prediction written
  into the evaluation **before** any number is recorded.
- **Part 14 (U5).** Accelergy: explicit MAC count in `Stats`, the estimation
  plug in adapter, `energy_pj` and `area_mm2` per cell.
- **Part 15 (U5, GATE).** The `cost_model_agreement` subsection with the
  prediction compared against measurement, the energy column in the README,
  fusion re-argued in energy terms, dependencies pinned and documented; walk
  the 13.3 gate.

**Phase U6, real batch support (Section 14):**

- **Part 16 (U6).** Batch loop in the conv2d kernel, with the failing first
  batch 2 regression test from ASSESSMENT 2.1 shown failing then passing, and
  batch 4 GoogleTests against hand computed values.
- **Part 17 (U6).** Batch in `pool`, the batched matmul decision (rank 3
  support or documented rank 2), elementwise ops verified batched, cost model
  scaling with batch.
- **Part 18 (U6).** Remove the U1 guards at both layers, add the batched
  LeNet model, extend the e2e matrix with `batch in [1, 4]`.
- **Part 19 (U6, GATE).** Walk the 14.3 gate: batch 1 numbers unchanged
  against the baseline, results regenerated (schema now includes batch),
  docs, merge.

**Phase U7, operator coverage (Section 15):**

- **Part 20 (U7).** `TRANSPOSE` end to end: `npuisa` opcode appended, ONNX
  `Transpose` converter, lowering, encoder, simulator kernel, `validate()`
  arity, tests at every layer, ISA manual entry.
- **Part 21 (U7).** `CONCAT` end to end, same layers as Part 20, including
  the ONNX `Concat` converter.
- **Part 22 (U7).** Unfolded `batch_norm` decomposed to mul plus add in the
  lowering, with lit and e2e numerics tests.
- **Part 23 (U7).** Importer wave 1: `Add`, `Mul`, `Identity` converters with
  per op pytests.
- **Part 24 (U7).** Importer wave 2: `MatMul`, `GlobalAveragePool`, `Clip`,
  `BatchNormalization` converters with per op pytests.
- **Part 25 (U7).** Bias fusion pattern (`add(conv(x, w), b)` into the bias
  operand) with positive and negative lit tests, the misleading lit test
  renamed, `-cse` and `-sccp` enabled in `_passes_for_level` with ablation
  rows.
- **Part 26 (U7).** `InferTypeOpInterface` plus arithmetic shape verification
  for conv, matmul, pools (output arithmetic, bias length, group
  divisibility, positivity); `_attr` rewritten to switch on
  `AttributeProto.type`; `count_include_pad` handled or rejected; the all
  padding average pool divide guarded.
- **Part 27 (U7, GATE).** Reachability passes with zero exemptions, the
  exemption table emptied, `DIALECT_REFERENCE.md` regenerated with a CI
  staleness gate added, walk the 15.2 gate, merge.

**Phase U8, the model suite (Section 16):**

- **Part 28 (U8).** Suite wave 1: depthwise separable block and small ResNet
  block in `MODELS` with structural pytests and e2e cells.
- **Part 29 (U8).** Suite wave 2: small Inception block, dilated conv stack,
  batched LeNet row, same requirements.
- **Part 30 (U8).** The Conv plus BatchNorm plus ReLU model exported with
  `do_constant_folding=False`, `-npu-fold-batchnorm` wired into `-O2`, the
  fold numerics e2e test.
- **Part 31 (U8).** Full suite benchmarks across the complete matrix, the
  BatchNorm ablation row with a measured delta, suite runtime measured under
  30 minutes.
- **Part 32 (U8, GATE).** README and report regenerated from the full suite
  results, every number tracing to a committed result, walk the 16.3 gate,
  then walk the 23.1 mandatory core checklist and tag `v2.0.0`.

**Phase U9, allocator and performance (Section 17):**

- **Part 33 (U9).** Belady style cost aware spill heuristic behind a pass
  option, old heuristic preserved, lit tests for both.
- **Part 34 (U9).** Heuristic comparison benchmarked as an ablation; verify
  whether it fixes the tight budget regression and update the evaluation
  subsection either way.
- **Part 35 (U9).** O(n log n) liveness and allocation via a sweep line, with
  the synthetic 5000 op compile time benchmark recorded before and after.
- **Part 36 (U9).** Parallel or im2col convolution kernel, suite runtime drop
  measured, numerics unchanged against goldens.
- **Part 37 (U9, GATE).** Allocator test quartet (fits, spill, fragmentation,
  budget too small), the two port cost model decision made with U5 divergence
  data and recorded in `docs/DESIGN_DECISIONS.md`, walk the Section 17 gate,
  merge.

**Phase U10, memref and memory spaces (Section 18, own branch):**

- **Part 38 (U10).** Design and types: `#npu.scratchpad` and `#npu.dram`
  memory space attributes, memref based `npuisa` op signatures, parser and
  printer round trip tests, `docs/ARCHITECTURE.md` design entry.
- **Part 39 (U10).** The npu to npuisa lowering re-expressed on memrefs with
  bufferization integration.
- **Part 40 (U10).** `AllocateScratchpad` re-expressed as allocation on the
  scratchpad memory space, spill semantics preserved, lit tests updated with
  justification per test.
- **Part 41 (U10).** Encoder consumes the memref form; binary format
  unchanged; round trip and golden tests prove numerics identical.
- **Part 42 (U10).** `TilingInterface` and `DestinationStyleOpInterface`
  implemented for the compute ops, with interface unit tests.
- **Part 43 (U10, GATE).** Full matrix on the branch, goldens unchanged,
  documentation rationale, merge the long lived branch.

**Phase U11, INT8 quantization (Section 19, depends U10):**

- **Part 44 (U11).** `quantize` and `dequantize` ops with verifiers, round
  trip and invalid tests, documentation.
- **Part 45 (U11).** The calibration pass and its Python driver harness
  (min max observation over calibration inputs), with unit tests.
- **Part 46 (U11).** `QUANT` and `DEQUANT` opcodes: encoder, `validate()`
  rules, ISA manual entries, opcodes appended not renumbered.
- **Part 47 (U11).** Integer simulator kernels (int8 by int8 into int32
  accumulate) with hand computed GoogleTests.
- **Part 48 (U11).** Quantized e2e across the suite: accuracy deltas
  recorded per model in the results schema.
- **Part 49 (U11, GATE).** The accuracy versus cycles versus energy
  evaluation section, calibration methodology documented, walk the Section
  19 gate, merge.

**Phase U12, advanced optimization (Section 20):**

- **Part 50 (U12).** Asynchronous DMA: token producing `dma_load_async` and
  `dma_store_async` plus `await`, verifiers, lowering support, lit tests.
- **Part 51 (U12).** Double buffering pass built on the tokens, plus the two
  port overlap treatment in the cost model per the Part 37 decision, with
  ablation rows.
- **Part 52 (U12).** Region based `npu.fused_op` replacing the two case
  `Activation` enum, generalizing fusion to elementwise chains; migration of
  the existing fusion patterns; negative tests.
- **Part 53 (U12).** Layout assignment (NCHW versus NHWC) as an optimization
  with the layout encoding attribute, and its ablation row.
- **Part 54 (U12).** Tiling part 1 (depends U10): tile convolutions and
  matmuls that exceed the scratchpad via `TilingInterface`, replacing whole
  tensor spilling as the only response to a tight budget.
- **Part 55 (U12).** Tiling part 2: tile size selection heuristic, the tight
  budget cells re-benchmarked against the U9 spilling story, evaluation
  updated.
- **Part 56 (U12).** The `.nbin` debug section mapping program counter to
  ONNX node name, `npu-sim --trace`, objdump support, format version bump
  with validation.
- **Part 57 (U12, GATE).** ZigZag design space exploration cross check on the
  tiling choices, U12 evaluation subsection consolidated, merge.

**Phase U13, Gemmini (Section 21, ASSESSMENT section 12):**

- **Part 58 (U13).** Stage 0 toolchain: Chipyard, riscv-tools, Spike; build
  and run Gemmini's own bundled tests inside the 12 GB budget. Hard stop
  gate: if this does not fit the machine, stop U13 and record why.
- **Part 59 (U13).** Stage 0 wrap: the sibling backend skeleton (separate
  repo or CMake flag per ASSESSMENT 12.7), environment pinning documented,
  smoke test scripted.
- **Part 60 (U13).** Stage 1: a single 16x16 matmul emitted as `config_ex`,
  `mvin`, `matmul.preload`, `matmul.compute.preloaded`, `mvout`, validated
  against numpy on Spike.
- **Part 61 (U13).** Stage 2 cheap path: emit `loop_ws`, run LeNet's fully
  connected layers end to end on Spike.
- **Part 62 (U13).** Stage 3 groundwork: the allocator re-expressed for
  Gemmini's row addressed scratchpad and separate accumulator address space,
  building on the U10 memory spaces.
- **Part 63 (U13).** Stage 3 real path: compiler directed tiling emitting
  explicit `matmul.preload` plus `matmul.compute` sequences, building on the
  U12 tiling; whole matmul path validated on Spike.
- **Part 64 (U13).** Stage 4: convolution via im2col or `loop_conv_ws`,
  whole LeNet on Spike, the INT8 versus FP config decision made and
  recorded (ASSESSMENT 12.2 item 4).
- **Part 65 (U13, FINAL GATE).** Stage 5: Verilator cycle counts within the
  memory ceiling plan of ASSESSMENT 12.6, the headline comparison of
  compiler directed tiling versus the hardware `loop_ws` on the same RTL,
  the report's Gemmini section, the full definition of done walked one last
  time, `v3.0.0` tagged.

### 25.3 Regenerating the part files

The part files are derived from this section plus the phase sections they
cite. If this spec changes, regenerate the affected part files rather than
editing them by hand, and keep part numbering stable: a part that becomes
unnecessary is replaced by a file stating so and why, never renumbered.
