# Assessment: shortcomings and upgrade roadmap

Status of the audit: repository at `d797ab5`, LLVM/MLIR `llvmorg-22.1.8`, built and
tested in WSL2 Ubuntu on 2026-08-07. Everything below was verified by running the
code, not by reading it alone. Where I claim something is broken, the reproduction
is included.

**What currently passes:** the project builds clean, 13 lit tests pass, 11
GoogleTests pass, 12 pytest tests pass, `dash-lint.sh` is clean, and the LeNet end
to end path matches onnxruntime to 3e-8. That is real and it works. This document
is about everything that sits outside that one path.

**Re-audit 2026-08-09:** the repository was re-audited at `6d62406` (phases U0,
U1, and U2 merged) plus the uncommitted phase U3 working tree, with every suite
re-run. Findings below now carry inline **Status 2026-08-09** notes recording
what each has become; section 13 holds the re-run evidence, the findings that
are new since the first audit, and the updated next actions. Headline of the
re-run: 13/13 lit, 39/42 encoding GoogleTests (the 3 red are phase U3
validation checks whose tests exist but whose implementation is unfinished),
7/7 simulator GoogleTests, 24/25 pytest (the 1 red is the staleness guard
correctly refusing a dirty tree), reachability green with 6 exemptions dated
2026-12-31, and all six benchmark cells still at 2.98e-8 against onnxruntime.

---

## 1. Executive summary

The compiler is correct and well engineered along a single narrow corridor: a
1x1x28x28 LeNet, batch size 1, seven ONNX operators, one scratchpad budget that
fits. Step outside that corridor in any direction and the project either fails
loudly (acceptable) or produces silently wrong numbers (not acceptable).

The five things that matter most, ranked:

| # | Problem | Severity | Where |
|---|---|---|---|
| 1 | Batch size greater than 1 silently computes garbage | **Critical, silent wrong answers** | `lib/Simulator/Simulator.cpp` |
| 2 | `transpose`, `concat`, `batch_norm` have no lowering, so any graph containing them cannot compile | High | `lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp` |
| 3 | The BatchNorm folding pass, the headline optimization, is unreachable from ONNX and contributes to zero benchmark numbers | High | `python/npu_frontend/op_mapping.py` |
| 4 | CI builds nothing and runs no tests; it is a lint job wearing a CI badge | High | `.github/workflows/ci.yml` |
| 5 | The entire evaluation is one model, with no per pass ablations, and the recorded results are untracked and stale | High | `experiments/` |

Everything else is below.

**Status 2026-08-09, row by row:** (1) closed as a loud failure in phase U1: the
importer and the verifiers now reject a batch greater than 1 with a message
naming the tensor and the tracking phase; the real batched kernels remain phase
U6 and `Simulator.cpp` still hardcodes `n = 0`. (2) unchanged in code, now
tracked as dated reachability exemptions expiring 2026-12-31, scheduled phase
U7. (3) unchanged: `-npu-fold-batchnorm` is still in no optimization level and
`npu.batch_norm` is still unimportable, scheduled phases U7 and U8. (4)
`ci.yml` now builds and runs every suite in four jobs (phase U2), but none of
it has executed on GitHub: phases U0 through U2 are unpushed, the badge still
shows the old lint only workflow's green, and the LLVM container image the new
CI pulls does not exist yet (section 13.4). (5) results are now tracked and
staleness guarded (phase U1); the suite is still one model and the per pass
ablations are still missing (phase U4).

### Decisions taken

Recorded here so they are not relitigated. Reasoning is in the sections named.

**D1. Keep the in house simulator. Do not replace it.** Twelve established
simulators were evaluated (section 10) and none does the job it does: whole program
numerical validation against onnxruntime. The ecosystem splits into simulators that
bring their own compiler (ONNXim, VTA, PyTorchSim, NVDLA), which would replace this
project rather than serve it, and analytical models that never execute code
(SCALE-Sim, Timeloop, ZigZag, Accelergy). Nothing free accepts an arbitrary
external instruction stream except a real hardware ISA such as Gemmini, and that is
structural rather than accidental: a custom ISA implies a custom simulator. Fix the
in house simulator (sections 2.1 and 2.3) and keep it as the correctness oracle.
See sections 10.4 and 10.7.

**D1a. Add Accelergy for energy and area.** Nothing else in the stack produces
either, and for an **edge** NPU energy is the metric that justifies the whole
design. The simulator already counts what Accelergy consumes. Roadmap item 16.
See section 10.5.

**D2. Add SCALE-Sim alongside the analytical cost model, not instead of it.** Emit
a SCALE-Sim topology CSV from the `npuisa` IR and report its cycle counts next to
the analytical estimate, then make the agreement or disagreement between the two a
subsection of the evaluation. That comparison is worth more than either number
alone: it converts the report's weakest claim ("cycles from constants I chose")
into its strongest ("cycles cross checked against an independently developed cycle
accurate model"). Scheduled as Tier 1 item 15 in section 7. See section 10.3.

**D3. Upgrade in place; revise the spec as the contract for that work, not as a
trigger for a rewrite.** See section 11.

---

## 2. Correctness defects

### 2.1 Batch size greater than 1 is silently wrong

This is the most serious defect in the project. The convolution kernel hardcodes
the batch index:

```cpp
// lib/Simulator/Simulator.cpp:41
int64_t n = 0; // batch is 1 for the supported models
```

`pool()` has the same problem in a different form: it iterates `C = inS[1]` and
never sees the batch dimension at all, so it processes only the first image.
`matmul()` handles `M` correctly, which makes the failure mode inconsistent across
ops rather than uniformly wrong.

Nothing in the pipeline rejects `N > 1`. The importer accepts it, the verifiers
accept it, the lowering accepts it, the allocator sizes buffers for the full batch,
and the simulator writes correct data for image 0 and leaves the rest of the output
buffer at whatever the scratchpad happened to hold.

Reproduction, a 2 batch conv plus relu:

```
ref shape (2, 4, 6, 6)
BATCH=2 max abs err: 1.8541955947875977
  batch 0 err: 1.19e-07     <- correct
  batch 1 err: 1.85e+00     <- garbage
```

A compiler that returns a wrong number without complaint is worse than one that
crashes. Two fixes are needed, and both should land:

1. **Immediate guard.** Reject `N != 1` in the ONNX importer and in the `npu` op
   verifiers, with a message naming the offending tensor. This closes the silent
   failure in an afternoon.
2. **Real fix.** Add the batch loop to `conv2d` and `pool`, and add a batched
   `matmul`. Then add a batch 4 model to the test suite so it stays fixed.

**Status 2026-08-09:** fix 1 landed in phase U1 at both layers:
`check_unbatched_activation` in the importer and `verifyUnbatchedActivation`
called from the `conv2d`, both pool, and `batch_norm` verifiers, pinned by four
pytest and four lit cases including the two controls. Fix 2 has not started:
`Simulator.cpp:43` still reads `int64_t n = 0` and `pool()` still never sees
the batch dimension. That is phase U6, and until then the guard makes it loud
instead of wrong.

### 2.2 Three dialect ops cannot be lowered

`npu.transpose`, `npu.concat`, and `npu.batch_norm` are defined in ODS, have
verifiers, have round trip tests, and have no conversion pattern in
`LowerNPUToNPUISA.cpp`. The conversion target marks the whole `npu` dialect
illegal, so any of them reaching the lowering aborts compilation:

```
error: failed to legalize operation 'npu.transpose' that was explicitly marked illegal
error: failed to legalize operation 'npu.concat' that was explicitly marked illegal
error: failed to legalize operation 'npu.batch_norm' that was explicitly marked illegal
```

Today this is invisible because LeNet contains none of them, and because the
transposed weight constants the Gemm importer creates are always dead by the time
lowering runs. It stops being invisible the moment a second model is added.

The fix is either to implement the three lowerings (`TRANSPOSE` and `CONCAT` as new
`npuisa` opcodes with simulator kernels; `batch_norm` decomposed into mul plus add
when folding did not fire) or, at minimum, to make the failure a diagnosed
compiler error rather than a generic legalization failure. Implementing them is the
right answer, because `concat` is what unlocks any branching topology such as an
inception block or a residual connection.

**Status 2026-08-09:** unchanged. The registered patterns are still exactly
Constant, Conv, MatMul, Relu, Reshape, Add, Mul, and the two pools; there is
still no pre flight diagnostic, so the three ops still die with MLIR's generic
legalization error, and no `TRANSPOSE` or `CONCAT` opcode exists in the ISA
yet. The gap is at least visible debt now: the reachability gate lists all
three ops as exempt until 2026-12-31, scheduled phase U7.

### 2.3 The `.nbin` decoder trusts its input, and the simulator has no bounds checks

`Program::decode` validates the magic and caps vector lengths, and stops there. It
does not validate:

- that the opcode is in range (a bogus `u16` becomes an out of range enum and falls
  through the simulator's `switch` as undefined behaviour),
- that `resultAddr` plus the result size fits `scratchpadBytes`,
- that `dramAddr` plus the region size fits `dramBytes`,
- that `operandAddrs` has the arity the opcode requires (`Conv2D` indexes
  `operandAddrs[1]` unconditionally; a one operand CONV2D reads out of bounds),
- the `version` field, which is written but never checked on read.

The simulator then does raw pointer arithmetic with no checking at all:

```cpp
auto spAt = [&](int64_t addr) { return sp.data() + addr / 4; };
```

A negative or oversized address is an out of bounds read or write. Separately,
`npu-sim` memcpys the entire input file into DRAM without comparing its size to the
declared input region, so an oversized `input.bin` is a heap overflow.

This matters even without an attacker: the format is the interface between
`npu-translate` and `npu-sim`, and a compiler bug that emits a bad address should
produce a diagnostic, not memory corruption. Add a `Program::validate()` that runs
after decode and before simulation, and make every `spAt`/`dramAt` access bounds
checked in a debug build.

**Status 2026-08-09: this is the in flight phase U3, mid implementation and not
yet landable.** The uncommitted working tree adds `Program::validate()` with
structured errors naming the instruction and the violated rule, checking the
version field, opcode range, per opcode operand arity from an explicit table,
result and operand addresses against the scratchpad, DMA regions against DRAM,
attribute arities and signs, and a written before read walk; the simulator's
raw `spAt` lambda was replaced by an always on `checkedAt` that refuses the
access and reports it in `SimResult.error`; `npu-sim` now rejects an input file
whose size differs from the declared region and validates before running. 39
of the 42 encoding tests pass. The remaining 3 failures, and the further
defects found in the new code by adversarial testing, are itemized in section
13.2 and must be fixed before the phase lands.

**Status 2026-08-09, closed.** Phase U3 landed. All five defects of 13.2 are
fixed (upgrade parts 1 to 3), the documentation is corrected and the validation
rules and version policy are written down (part 4), and the phase merged to
`main` after the full 11.3 gate: 51 of 51 encoding tests, 9 of 9 simulator, 14
of 14 lit, a 358 case fuzz corpus, 1000 iteration property test, and ASan and
UBSan clean over both GoogleTest binaries. `decode()` now means decode plus
validate, so the decoder no longer trusts its input and the simulator's bounds
checks can actually fire.

### 2.4 Average pooling divides by zero on all pad windows

`pool()` counts contributing elements and divides by `count`. A window entirely
inside the padding gives `count == 0`. Unreachable with the current pad sizes,
reachable with large pads. It also silently disagrees with ONNX, whose
`AveragePool` has a `count_include_pad` attribute the importer ignores entirely,
so any model setting it imports to a kernel with different semantics.

**Status 2026-08-09:** unchanged, and still reachable: the in flight
`validate()` requires pads to be non negative but does not bound them below
the kernel size, so an all padding window still divides 0.0f by zero and
returns NaN. `count_include_pad` is still ignored. Scheduled phase U7.

### 2.5 `_attr` cannot distinguish an absent attribute from a falsy one

```python
def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:              # empty list is falsy, falls through
                return list(a.ints)
```

An attribute legitimately equal to an empty list returns `default` instead. The
function also ignores `AttributeProto.type`, which is the field that actually says
what kind of attribute it is. Rewrite it to switch on `a.type`.

**Status 2026-08-09:** unchanged, scheduled phase U7 (`UPGRADE_SPEC_V3.md`
section 15.1 item 7).

### 2.6 Shape verification is structural, not arithmetic

`Conv2DOp::verify` checks ranks and attribute lengths. It never checks that the
output spatial dimensions are what the input, weight, strides, pads, and dilations
imply. It never checks that the bias length equals the output channel count, that
`group` divides both channel counts, or that strides and dilations are positive.
The same is true of the pooling verifiers.

This means the importer can emit an arithmetically impossible conv, `npu-opt` will
verify it happily, and the error surfaces as a wrong number out of the simulator.
The `InferTypeOpInterface` header is already included in `NPUOps.td` but no op
implements it. Implementing `inferReturnTypes` for conv, matmul, and the pools
would give shape checking, better error messages, and result type inference for
free.

**Status 2026-08-09: partially fixed.** The verifiers are no longer purely
structural for five ops: `matmul` checks the contraction dimension, `reshape`
the element count, `transpose` the permutation, `concat` the axis sums, and
`batch_norm` the parameter length against the channel count. Still missing:
any arithmetic output shape check for `conv2d` and the pools, the bias length
checks, the group divisibility checks, and positivity of strides and
dilations; `InferTypeOpInterface` is still included and still implemented by no
op. The remainder is phase U7.

---

## 3. Scope gaps against the specification

The build specification (`npu_mlir_compiler_spec.md`) is the contract. These are
the places the implementation is narrower than it.

### 3.1 The ONNX frontend implements 7 of the 13 documented ops

`op_mapping.py` supports `Conv`, `Gemm`, `Relu`, `MaxPool`, `AveragePool`,
`Reshape`, `Flatten`. Missing: `MatMul`, `Add`, `BatchNormalization`,
`GlobalAveragePool`, `Concat`, `Clip`.

`docs/ONNX_FRONTEND.md` is honest about this. The docstring at the top of
`op_mapping.py` is not: it lists all 13 as "supported ops (opset 17)" and claims
the unimplemented ones "raise a clear 'not yet implemented' until their phase",
which they do not. Fix the docstring today, implement the ops after.

The importer also never checks the model's opset version despite opset 17 being a
pinned assumption, and it rejects any dynamic dimension because `_collect_shapes`
reads `dim_value` and ignores `dim_param`.

**Status 2026-08-09:** the docstring was corrected in phase U1 and now lists
seven implemented and six missing ops. One wart: it discusses `Identity` as
the op that blocks any `do_constant_folding=False` import, but leaves it out
of the missing list, while `UPGRADE_SPEC_V3.md` section 15.1 counts seven
missing importer ops including `Identity`. The opset version is still never
checked and `dim_param` is still rejected; both remain phase U7.

### 3.2 The BatchNorm folding pass can never fire on a real model

This is the pass the README leads with under "Things I am proud of", and it is
genuinely good code. It is also completely unreachable from the ONNX entry point,
for two independent reasons:

1. `torch.onnx.export` constant folds BatchNorm into the preceding Conv before the
   file is even written. Exporting a Conv plus BatchNorm module yields
   `nodes: ['Conv']`. There is no BatchNormalization node left to import.
2. Export with `do_constant_folding=False` yields
   `['Identity', 'Identity', 'Conv', 'BatchNormalization']`, and the importer
   rejects it on the first `Identity`, before it would have rejected the
   `BatchNormalization` that is not in `CONVERTERS` either.

So the pass is exercised by exactly one hand written 27 line lit test, is not in
any optimization level (`_passes_for_level` never adds `-npu-fold-batchnorm`), and
contributes to zero numbers in the evaluation. The report presents it as a pillar
of the work; the evidence for it is one FileCheck file.

To make it real: add `BatchNormalization` and `Identity` to the importer, export
the benchmark models with `do_constant_folding=False`, wire
`-npu-fold-batchnorm` into `-O2`, and add a model with batch norm to the suite so
the pass shows up as an ablation row with an actual delta.

**Status 2026-08-09:** unchanged, and now enforced as visible debt: the
reachability gate lists `npu.batch_norm` as unreachable at every layer, exempt
until 2026-12-31, and `-npu-fold-batchnorm` is still absent from every
optimization level. The plan in the paragraph above is now phases U7 and U8 of
`UPGRADE_SPEC_V3.md`.

### 3.3 Fusion covers activation only, not bias

The spec asks for "conv/matmul + bias + activation" fusion. `FuseOps.cpp` fuses
only the activation. There is no pattern matching `npu.add(conv(x, w), bias)` and
folding the addend into the conv's bias operand. It is invisible today because the
ONNX `Conv` and `Gemm` importers attach the bias directly as an operand, so the
add form never appears. Any frontend that emits a separate bias add, or a hand
written test, would not fuse. The lit test is named
`fuse-conv-bias-relu.mlir` but its bias is already an operand in the input IR, so
the name overstates what is tested.

Missing fusion opportunities beyond bias: conv plus conv chains, pool folding, and
`Clip` as a `relu6` style bounded activation (the `Activation` enum has exactly two
cases, `none` and `relu`).

**Status 2026-08-09:** unchanged. `FuseOps.cpp` still contains exactly
`FuseReluIntoConv` and `FuseReluIntoMatMul`, the enum still has two cases, and
bias fusion is still absent. Scheduled phase U7, which also renames the
misleading lit test.

### 3.4 Only one test model exists

`model_generator.MODELS` contains `lenet` and nothing else. `INPUT_SHAPES` in
`run_benchmarks.py` likewise. The spec calls for a core suite plus a depthwise
separable block as a stretch goal.

One model means: one topology, one data type, one input size, one batch size, no
grouped convolution ever executed outside a unit test, no dilated convolution ever
executed, no `concat`, no residual add, no global pooling. The evaluation section
of the report generalizes from a sample of one.

This is the single highest leverage upgrade in the document. A suite of five or six
models would immediately exercise most of the gaps listed above:

| Proposed model | What it forces into the pipeline |
|---|---|
| LeNet (existing) | baseline |
| MobileNet style depthwise separable block | grouped conv, pointwise conv |
| Small ResNet block | residual `Add`, identity shortcut |
| Small Inception block | `Concat`, parallel branches |
| Conv plus BatchNorm plus ReLU stack | the BatchNorm folding pass, finally |
| Dilated conv stack | dilation, `pads` asymmetry |
| Batched LeNet, N=4 | the batch bug from section 2.1 |

**Status 2026-08-09:** unchanged. `MODELS` still contains exactly `lenet`. The
proposed suite became the seven row model table of `UPGRADE_SPEC_V3.md`
section 16, phase U8.

### 3.5 No per pass ablations

The spec calls the per pass ablation deltas "the evaluation's backbone" and section
8 requires "one pass at a time ablations", "IR op counts before/after each pass",
and "wall clock compile time per pass". `run_benchmarks.py` iterates
`product(models, levels, budgets)` and records none of the three. It records one
total `compile_ms` and one post lowering `npuisa` op count.

So the report can say what `-O2` buys over `-O0` in aggregate, but not what
canonicalization buys, or fusion, or DCE, individually. That is precisely the
question the project was built to answer.

**Status 2026-08-09:** unchanged. `run_benchmarks.py` still records one total
`compile_ms` and one post lowering op count per cell. The ablations are phase
U4 (`UPGRADE_SPEC_V3.md` section 12), now the largest remaining gap between
what the report claims to evaluate and what is measured.

### 3.6 The simulator has no progress reporting

Spec section 9 requires a TTY aware single line progress bar over the instruction
stream. `npu-sim.cpp` has none. `npu-compile --verbose` does print per stage
timings, and `run_benchmarks.py` does use `tqdm`, so two of the three progress
requirements are met; this one is not. The "projected wall clock printed up front"
requirement is also unmet.

**Status 2026-08-09:** unchanged; no progress reporting of any kind in
`npu-sim` (verified by grep for progress, isatty, and carriage returns across
the tools). Scheduled phase U4.

### 3.7 Missing from the repository layout

- `report.yml` (spec section 15) does not exist. The PDFs are built by hand.
- `NPUTypes.td` for the `npu` dialect does not exist (the dialect uses builtin
  `tensor` throughout, which is a defensible design choice, but the layout in the
  spec lists the file).
- `lib/Simulator/Memory` does not exist as a component; DRAM and scratchpad are
  two local `std::vector<float>` in `Simulator::run`.
- `mypy` is required by spec section 11 and is configured nowhere: not in
  `pyproject.toml`, not in `.pre-commit-config.yaml`, not in CI. No type checking
  runs on the Python at all.
- The license is MIT; the spec pins Apache-2.0 with LLVM exceptions. This looks
  like a deliberate change (commit `48a0be3` says "relicense to MIT") but it is a
  deviation from a pinned assumption and is not recorded in `DESIGN_DECISIONS.md`.

**Status 2026-08-09: mostly closed.** `report.yml` now exists and builds both
PDFs in CI (phase U2). `mypy` is configured in `pyproject.toml`, in pre-commit,
and in the CI lint job (phase U2), though scoped to `python/npu_frontend` only
and targeting Python 3.12 while the venv runs 3.14 (section 13.4). The MIT
decision is now recorded in `docs/DESIGN_DECISIONS.md`, and `UPGRADE_SPEC_V3.md`
adopts MIT as a corrected pin. `NPUTypes.td` and a separate `Memory` component
remain absent, by design.

---

## 4. Engineering and infrastructure

### 4.1 CI is a lint job with a CI badge

`.github/workflows/ci.yml` has exactly one job. It runs `dash-lint.sh` and `ruff`.
It does not configure CMake, does not compile a single translation unit, does not
run lit, does not run GoogleTest, does not run pytest. The workflow file's own
comment admits it: "Building and running the compiler needs a prebuilt LLVM/MLIR
and is done locally."

Meanwhile `docker/Dockerfile.llvm` exists for exactly this purpose and is used by
nothing, and the README displays a green CI badge next to a coverage badge, which
together read as "the test suite passes on every push". It does not run on any
push.

This is the highest value infrastructure fix in the document, and the pieces are
already in the repo. Build `Dockerfile.llvm` once, push it to GHCR, and make
`ci.yml` a second job that pulls the image, configures, builds, and runs all three
test suites. Then the badge means something.

**Status 2026-08-09:** the workflow side is fixed (phase U2): `ci.yml` now has
lint, build and test, sanitizer, and coverage jobs, and `llvm-image.yml`
builds the prebuilt LLVM image. What is not fixed is everything that requires
GitHub: the phases are unpushed, `origin/main` is still the pre upgrade
commit, the green badge on the README is the old lint only workflow's run from
2026-08-07, and `ghcr.io/olajide-badejo/npu-mlir-llvm:22.1.8` answers an
anonymous manifest probe with 403, meaning the image has never been published
and the new CI would fail at container pull on first push. The four fault
class proof runs required by `UPGRADE_SPEC_V3.md` section 10.2 are likewise
outstanding. Sequencing in section 13.5.

### 4.2 Recorded results are untracked and stale

`git ls-files experiments/results/` returns one file: `.gitkeep`. The six JSON
result files that every number in the report traces back to are not in version
control. A fresh clone has no results, and `results_to_tex.py` has nothing to read.

The results that do exist locally carry `git_sha: 8095dbec` in their manifest.
`HEAD` is `d797ab5`. They were generated three commits ago, before the C++20 switch.
`report/main.pdf` was last committed at `a8cd65f`, also behind `HEAD`.

The resume logic makes this permanent: `valid()` returns `True` for any file that
parses as JSON, so a stale result is never regenerated without `--force`. It should
compare the manifest's `git_sha` and cost model constants against the current ones
and invalidate on mismatch. And the results should be committed, since
reproducibility from a clean clone is a stated requirement.

**Status 2026-08-09: closed in phase U1**, with the correction recorded at
roadmap item 3: the stale results' `git_sha: 8095dbec` was never a commit in
this repository at all. The results are now tracked, regenerated at `66e5af5`,
and guarded by `staleness()` plus `test_committed_results_are_current`. The
committed PDFs, however, still embed the July 15 numbers that trace to that
nonexistent commit; see section 13.4.

### 4.3 `instruction_count` is measured by regex and counts non instructions

```python
def count_ops(mlir_text, dialect):
    return dict(Counter(re.findall(rf"{dialect}\.[a-z_0-9]+", mlir_text)))
```

This counts every textual occurrence of `npuisa.something` in the IR dump,
including inside comments, inside type strings such as `!npuisa.buffer`, and
including `npuisa.const`, which the encoder explicitly treats as data and does not
emit as an instruction. `instruction_count` is then the sum of those counts.

So the headline table's "Instructions: 91 / 82 / 70" is not the number of
instructions in the `.nbin`. The simulator already reports a true instruction count
in its stats JSON (`stats.instructions`), and `run_benchmarks.py` ignores it in
favour of the regex. Use the simulator's number.

**Status 2026-08-09:** unchanged in `run_benchmarks.py`, and the contradiction
is now committed in duplicate: `experiments/results/*.json` says 91 / 82 / 70
(regex) while `test/baseline/baseline.json` records the simulator's true
28 / 25 / 21 for the same cells. The README table, the report, and the plots
inherit the regex numbers; the CHANGELOG defers the correction to phase U4
explicitly. See also section 13.4 item 4 for what this does to the README's
objdump excerpt.

**Status 2026-08-09, done** (upgrade part 7, commits `4cc7fc8`, `71feacb`,
`f1c4de7`). `instruction_count` is `int(stats["instructions"])`, and a missing
field raises rather than falling back to the regex, since a silent fallback
would reinstate the defect the next time the stats format shifted. All six cells
regenerated: 28 / 25 / 21 at the default budget and 28 / 31 / 29 at the tight
one, matching `test/baseline/baseline.json` exactly, so the committed
contradiction is gone. The README table, the generated macros, the plot, and the
report PDF all follow. `count_ops` and `npuisa_op_counts` are kept as a
histogram, with a docstring naming both inflation modes. Four tests pin it,
including `test_readme_table_matches_the_results` so the hand written table
cannot drift again, and `test_count_ops_is_not_an_instruction_count`, which
demonstrates both faults on a three line dump rather than only describing them.

### 4.4 Tolerances are asserted loosely and reported tightly

`test_end_to_end.py` asserts `rtol=1e-3, atol=1e-3`. The README reports a max error
of 3e-8. There is a five order of magnitude gap between what is tested and what is
claimed, which means a regression that degraded accuracy by four orders of magnitude
would pass CI (if CI ran the tests, which it does not) while silently invalidating
the README.

`run_benchmarks.py` records only `max_abs_error_vs_onnxruntime`. The spec asks for
"stated absolute and relative tolerances". Add relative error, add a regression
threshold to the test that is close to the observed value, and validate at every
optimization level rather than at one fixed pipeline. The current e2e tests run a
single hardcoded pass list, not `-O0`, `-O1`, `-O2`, so the spec's "every model at
every opt level" gate is not actually enforced anywhere.

Numerical validation also uses a single random input per model. One draw from a
standard normal is not evidence about a ReLU network's edge cases. Add a handful of
inputs including zeros, large magnitudes, and values that straddle the ReLU knee.

**Status 2026-08-09: partially closed.** Phase U1 tightened the e2e tolerance
to `rtol=1e-5, atol=1e-6` with the observed 2.98e-8 recorded alongside. Still
open, all phase U4 (`UPGRADE_SPEC_V3.md` sections 9.6 and 12): the e2e test
runs one hardcoded pass pipeline that matches no `-O` level exactly, relative
error is still not recorded in the benchmark results, and validation still
uses a single standard normal draw rather than the spec's five input classes.

### 4.5 Test suite is thin in specific places

13 lit tests, 11 GoogleTests, 12 pytest tests. The gaps, by name:

- **Simulator GoogleTests** cover Relu, MatMul, Conv2D, Add/Mul, AvgPool, Reshape.
  There is no test for `POOL_MAX`, none for `DMA_LOAD` or `DMA_STORE`, none for
  grouped convolution, none for dilated convolution, none for asymmetric padding.
  Spec section 10 asks for "simulator semantics per instruction".
- **Encoding GoogleTests** cover round trip, bad magic, disassembly text, and one
  function encode. There is no test that feeds truncated or corrupted bytes, which
  is exactly the class of input section 2.3 is about.
- **Scratchpad allocator** has one lit test for the fits case and one for the spill
  case. There is no test for fragmentation, none for the "single buffer exceeds the
  budget" error path, none for the interaction between spilling and address reuse
  (which is where the bug described in the debug report actually lived).
- **No fuzz or property test anywhere.** A round trip property test over randomly
  generated `Program` structures would be twenty lines and would cover the encoder
  better than the four tests that exist.

**Status 2026-08-09: transformed for the encoder, unchanged elsewhere.** The
in flight phase U3 adds exactly what the last two bullets asked for: a 1000
iteration seeded round trip property test, a 322 case malformed `.nbin` fuzz
corpus behind four fuzz tests, and 32 validation tests that each pin one
`validate()` rule by name. Three of those are red until the defects in section
13.2 are fixed. The simulator gaps are untouched: still no `POOL_MAX`,
`DMA_LOAD`/`DMA_STORE`, grouped, dilated, or asymmetric padding test
(`UPGRADE_SPEC_V3.md` section 9.3). The allocator still has no fragmentation,
oversized buffer, or spill address reuse test.

**Status 2026-08-09, encoder half closed with U3.** The encoding suite is 51
tests: 36 validation tests pinning every one of the 22 `validate()` check names
by name, 4 fuzz tests over a 358 case corpus, 2 property tests at 1000
iterations, and the round trip and encode tests that were there before. Writing
the manual's validation table from the `fail()` call sites found the last gap,
`dram-size`, which had no test at all; both of its branches now do, as does the
untested over limit branch of `scratchpad-size`. The simulator suite gained two
tests for scratchpad sizing but the gaps this section names are still open:
still no `POOL_MAX`, `DMA_LOAD`/`DMA_STORE`, grouped, dilated, or asymmetric
padding test, and the allocator still has no fragmentation, oversized buffer, or
spill address reuse test.

### 4.6 `scripts/coverage.sh` swallows failures

```bash
ninja -C "$build" check-npu || true
```

Coverage is collected whether or not the lit suite passed. A build where every lit
test fails still produces a coverage number, and that number is what the README
badge reports. Drop the `|| true`. The script also writes to `build-coverage` while
the repo contains a stale `build-cov` directory from an earlier run.

**Status 2026-08-09: closed in phase U1.** The script runs under
`set -euo pipefail`, the tree name is unified to `build-cov`, `npu-sim` is
instrumented, and a failing suite aborts before any percentage prints. The
badge itself is a different problem; see section 13.4 item 6.

### 4.7 Smaller items

- `npu-translate` walks the module and encodes the **first** function it finds,
  silently ignoring the rest. Multi function modules are not diagnosed.
- `npu-sim` writes only `result.outputs.front()`, so a multi output model loses
  every output but the first, with no warning.
- `NOP` and `HALT` are defined in the dialect and handled by the encoder, but no
  pass ever emits them. `HALT` is appended manually by `encodeFunction`. The `NOP`
  opcode is entirely dead surface area.
- Both `npu-sim` and `npu-translate` hand roll `argv` parsing. LLVM's `cl::opt` is
  already linked in and would give consistent `--help` output across all five
  tools.
- The importer materializes an `npu.constant` for **every** initializer whether
  used or not, then relies on DCE to remove them. That is the source of the "dead
  transposed weight constants" the README credits `-O1` with removing, meaning a
  chunk of the advertised `-O1` win is the compiler cleaning up its own mess.
  Worth stating plainly in the report.
- `Simulator::run` recovers operand shapes from a `std::map<int64_t, shape>` keyed
  by scratchpad address, populated by whatever wrote that address last. It works
  because operands are live, but it is fragile by construction. Encode operand
  shapes per instruction instead.

**Status 2026-08-09:** the first two items are fixed in the in flight tree:
`npu-translate` now refuses a multi function module with a counted diagnostic
and refuses to write a program that fails `validate()`, and `npu-sim` now
writes every output, numbering the files past the first. Still true: hand
rolled argv parsing in both tools (an unrecognized token silently becomes the
input path), the dead `NOP` opcode, the importer materializing every
initializer, and the address keyed shape map, which `validate()` now has to
mirror with a written before read walk of its own.

---

## 5. Design and performance headroom

These are not defects. They are the places where the project stops short of what it
could demonstrate.

### 5.1 The allocator makes tight budgets worse, and nobody looked into why

The recorded results contain a striking result that the report does not discuss:

| LeNet | `-O0` | `-O1` | `-O2` |
|---|---:|---:|---:|
| cycles, 1 MB budget | 23,421 | 13,008 | 12,710 |
| cycles, 140 KB budget | 23,421 | **33,834** | **33,930** |
| DRAM bytes, 140 KB budget | 347,440 | **514,000** | **520,272** |

At a 140 KB budget, optimizing makes the program 45 percent slower and moves 50
percent more data than not optimizing at all. `-O0` does not spill; `-O1` and `-O2`
do. The optimizer shortens live ranges of some values while lengthening others, and
the spill heuristic then picks badly.

This is the most interesting empirical finding in the repository and it is
currently sitting in an untracked JSON file. It deserves a subsection in the
evaluation, and it points at a real upgrade: the spill victim heuristic is
"longest live range at the peak", which ignores size and ignores how many times the
value is reloaded. A cost aware heuristic (spill cost = bytes moved times number of
reloads, minimized) is a well understood improvement and would make a genuinely
good ablation.

**Status 2026-08-09:** the results are tracked now, and the regeneration at
`66e5af5` reproduces these numbers exactly, so the finding is stable across
the U1 and U2 changes; at the tight budget `-O2` is also slower than `-O1`
(33,930 versus 33,834 cycles). The spill heuristic is unchanged. The write up
is roadmap item 14 (phase U4); the heuristic fix is phase U9.

### 5.2 The allocator is O(n^3) in the worst case

`peakPressure` is O(instructions x buffers), it is recomputed inside the spill
loop, and the spill loop can run once per buffer. LeNet has fewer than 100
instructions so this is invisible. A model with a few thousand ops would not be.
An interval tree or a sweep line over live range endpoints makes this O(n log n).

### 5.3 The cost model has no parallelism

Every cost is added to a single running total. There is no DMA and compute overlap,
no double buffering, no pipelining, no bank conflicts, no memory level parallelism.
That is a legitimate simplification and it is documented as one, but it means the
model cannot show the one thing an edge NPU cost model is most useful for: that
prefetching the next layer's weights while computing the current layer hides the
DMA latency. Adding a two port model (one DMA engine, one compute engine, cycles =
max rather than sum on overlapping instructions) would be maybe 60 lines and would
make the fusion result much more interesting, because fusion's benefit is exactly
that it removes DMA from the critical path.

### 5.4 The simulator kernels are naive

`conv2d` is a six deep loop nest with no tiling, no im2col, no vectorization, no
threading. Fine for LeNet. It is the reason the model suite cannot grow to anything
resembling a real network without the benchmark suite blowing past its 30 minute
budget. An im2col plus GEMM path, or simply `#pragma omp parallel for` over output
channels, would buy an order of magnitude for a few lines.

### 5.5 The binary format is larger than it needs to be

Every instruction serializes five length prefixed vectors plus five scalars,
whether or not the opcode uses them. A `NOP` costs roughly 50 bytes. The ISA manual
calls the format "a fixed header followed by tagged records", but there are no tags;
every instruction is the same fixed shape. Either implement the tagged operand list
the spec and the manual describe, or correct the manual to say fixed record. The
format also has no checksum, no section offsets, no explicit endianness marker, and
does not check the version field it writes.

**Status 2026-08-09, with two corrections:** the record is six length prefixed
vectors plus five scalars per instruction, not five and five, and the in
flight phase U3 now checks the version field on read. Still true: no tags, no
checksum, no section offsets, and the byte order is actually host native
despite a comment claiming little endian helpers. The "tagged records" phrase
survives in both `docs/ISA_MANUAL.md` and the `Program.h` header comment; fix
both together in U3's documentation pass.

### 5.6 Single block, single function, no control flow

`AllocateScratchpad` operates on `func.getBody().front()` and ignores every other
block. `encodeFunction` does the same. This is consistent with the "no branches"
design decision and is fine, but it should be enforced with a diagnostic rather
than silently ignoring blocks two and up.

**Status 2026-08-09:** the in flight phase U3 adds exactly that diagnostic to
`AllocateScratchpad` (single block enforced with a counted error), and
`npu-translate` now diagnoses multi function modules. The encoder itself is
unchanged and still assumes a single block, but the allocator's diagnostic
runs earlier in every real pipeline.

### 5.7 Stretch goals never started

INT8 quantization is explicitly a stretch goal in the spec and is entirely absent:
no `quantize`/`dequantize` ops, no `QUANT`/`DEQUANT` opcodes, no integer path in
the simulator. Given that this is the single most characteristic feature of edge
NPU compilers, it is the most valuable large upgrade available if the goal is to
make the project representative of the domain.

---

## 6. Documentation accuracy

The docs are generally good and generally honest. Four corrections:

1. `op_mapping.py`'s module docstring claims 13 supported ops. Seven are
   implemented. `docs/ONNX_FRONTEND.md` has it right; the docstring does not.
2. The README's fusion example shows `CONV2D ... act=relu` at instruction index 10
   with a stated 21 instructions, while the recorded `-O2` result says 70. The
   objdump excerpt appears to be illustrative rather than copied from a real run.
   If it is illustrative it should say so; the project's own ground rule 1 is
   "never fabricate".
3. The README's coverage badge (90 percent C++, 89 percent Python) is produced by a
   script that ignores test failures (section 4.6).
4. `DIALECT_REFERENCE.md` is generated by the `npu-dialect-doc` target, which is
   good, but nothing verifies it is current. Add a CI step that regenerates it and
   fails on a diff, which is what "docs cannot drift" in the spec actually requires.

**Status 2026-08-09:** item 1 was fixed in phase U1 (with the `Identity`
undercount noted at section 3.1). Item 2 inverted rather than resolved: the
excerpt's "21 instructions" matches the simulator's true `-O2` instruction
count exactly (28 / 25 / 21 per `test/baseline/baseline.json`), so the excerpt
now looks genuine and it is the table's regex derived 70 that is wrong; phase
U4 reconciles them. Item 3: the script is fixed, but the badge is still a hand
written shields.io URL claiming 90 / 89 while CI enforces only C++ at 85
percent and no Python threshold at all (section 13.4 item 6). Item 4: the
target still exists and no CI step regenerates or diffs it; that gate is
scheduled with phase U7.

---

## 7. Prioritized roadmap

Ordered by value divided by effort. Tiers are cumulative.

### Tier 0: stop being wrong (days)

1. Reject `N != 1` in the importer and verifiers, with a clear message. Closes the
   silent wrong answer in section 2.1.
   **DONE (phase U1).** Guarded in `op_mapping.check_unbatched_activation` and in
   the `conv2d`, both pool, and `batch_norm` verifiers. Activations only, since a
   conv weight is OIHW and its leading dimension is output channels, not batch.
2. Fix `op_mapping.py`'s docstring to match reality.
   **DONE (phase U1).** It now lists the seven implemented ops and the six that
   are not, with a pointer to phase U7.
3. Commit `experiments/results/*.json` and regenerate them at `HEAD`.
   **DONE (phase U1).** The `.gitignore` line that excluded them is gone and they
   are regenerated and tracked. Correction to this document: the results recorded
   `git_sha: 8095dbec`, and that is **not a commit in this repository**. They were
   not three commits stale, they were produced from a tree that was never
   committed, so they were reproducible from no point in the history at all.
4. Make `valid()` invalidate results whose manifest `git_sha` or cost model
   constants differ from the current ones.
   **DONE (phase U1)**, as `staleness()`, which returns the reason rather than a
   bool. The sha rule is "same sha, or nothing that can move a number has changed
   since it", not literal sha equality, which would make every result stale on
   the very commit that adds it. Rationale in `docs/DESIGN_DECISIONS.md`.
5. Delete `|| true` from `coverage.sh`.
   **DONE (phase U1).** Also unified the `build-coverage` and `build-cov` names
   and added `npu-sim` to the instrumented build.
6. Tighten the e2e tolerance to something near the observed 3e-8 so a numerics
   regression actually fails.
   **DONE (phase U1).** `rtol=1e-5, atol=1e-6`, with the observed 2.98e-8
   recorded in the test as a comment.

### Tier 1: make the claims true (one to two weeks)

7. Real CI: build `Dockerfile.llvm`, publish to GHCR, add a job that configures,
   builds, and runs lit plus GoogleTest plus pytest. Add `report.yml`.
   **DONE (phase U2), pending a push.** `ci.yml` now has four jobs: `lint`
   (dash-lint, ruff, black, mypy, reachability), `build-and-test` (configure,
   build, lit, both GoogleTests, pytest, reachability with the model layer, and
   the regression baseline), `sanitizers` (ASan and UBSan), and `coverage` (with
   the 85 percent threshold enforced). `report.yml` builds both PDFs and uploads
   them. `llvm-image.yml` builds and publishes the LLVM base image, manually and
   only when the pinned tag changes. `Dockerfile.llvm` was rewritten as two
   stages: as written it kept the full build tree *and* a second installed copy,
   which is tens of gigabytes and not pushable.
   Every job was run locally and passes. What cannot be verified without push
   access is CI itself going green and red, so the four fault class proof runs
   required by `UPGRADE_SPEC_V3.md` section 10.2 are outstanding. Until they are
   done the badge still claims more than has been demonstrated.
   **Blocker confirmed 2026-08-09:** none of it has reached GitHub.
   `origin/main` is still `7de39c6`, the GHCR image answers 403 anonymously
   (never published), and the badge's green comes from the old lint only
   workflow. Push order in section 13.5.
8. `Program::validate()` plus bounds checked memory access in the simulator, plus
   an input size check in `npu-sim`. Add a corrupted input GoogleTest.
   **IN FLIGHT (phase U3, uncommitted, 2026-08-09).** All the pieces exist in
   the working tree, plus a property test and a 322 case fuzz corpus, and 39
   of 42 encoding tests pass. Not landable yet: the three red tests and the
   further defects in section 13.2 (the overflow guard that itself overflows,
   the scratchpad auto expansion that swallows the result bounds check, an
   unaligned constant in one test, operand read extents that `validate()`
   never compares, and an `assert(false)` on the trap path). Land it on an
   `upgrade/u3-...` branch per ground rule 11.
9. Per pass ablations in `run_benchmarks.py`: op counts before and after each pass,
   per pass wall clock, one pass at a time deltas. This is the spec's stated
   backbone and it is missing.
10. Use the simulator's `stats.instructions` instead of the regex count.
    **Note 2026-08-09:** both counts are now committed side by side, 91 / 82 /
    70 in the results JSONs versus 28 / 25 / 21 in `test/baseline/baseline.json`.
    Phase U4, and the README table moves with it.
11. e2e validation at every optimization level, not one hardcoded pipeline.
12. Add `mypy` to `pyproject.toml`, pre-commit, and CI.
    **DONE (phase U2), early.** Configured in all three places, scoped to
    `python/npu_frontend` with a Python 3.12 target; widening the scope and
    aligning the target version with the 3.14 venv is noted in section 13.4
    item 5.
13. TTY aware progress bar in `npu-sim`.
14. Write up the tight budget regression (section 5.1) in the evaluation. It is the
    best empirical result in the repo.
    **Note 2026-08-09:** re-verified in the regenerated results; unchanged and
    still unwritten.
15. **SCALE-Sim cross validation (decision D2).** Three pieces, in order:
    - `experiments/scalesim_export.py`: walk the allocated `npuisa` IR and emit a
      SCALE-Sim topology CSV plus an architecture config. The conv and matmul rows
      are a direct read of the shapes already on each op, and the architecture
      config is the existing `CostModel` constants restated (16x16 array, the
      scratchpad budget as SRAM size). Well under a hundred lines.
    - Record `scalesim_cycles` next to `simulated_cycles` in every result JSON,
      and add the SCALE-Sim version to the manifest block alongside the LLVM tag.
    - A `cost_model_agreement` subsection in the evaluation: per layer analytical
      versus SCALE-Sim cycles, the ratio, and an honest account of where they
      diverge. The divergences are the interesting part, since the analytical model
      has no DMA and compute overlap (section 5.3) and SCALE-Sim does model
      double buffering. Expect the analytical model to be pessimistic on compute
      bound layers and optimistic on memory bound ones; say so and show it.

    This lands in Tier 1 rather than later because it is pure Python against an
    MIT licensed dependency, needs no new toolchain, and does not touch a single
    line of the compiler. It is the cheapest large credibility gain available.
16. **Accelergy energy and area numbers (section 10.5).** The simulator already
    counts everything Accelergy needs. Add MAC counts to `Stats` as an explicit
    field (they are currently only implicit in `macCycles`), write an estimation
    plug in adapter mapping `Stats` to Accelergy action counts, and record
    `energy_pj` and `area_mm2` in every result JSON. Then add an energy column to
    the headline table. This is the metric that actually justifies an edge NPU's
    design, and the project currently cannot speak to it at all.
17. Re-argue fusion in energy terms once item 16 lands. Fusion keeps an
    intermediate in the scratchpad; in cycles that is a modest win, in energy it is
    a large one. The evaluation currently makes the case using the metric that
    flatters it least.
18. Add `scale-sim` and `accelergy` to the pinned Python dependencies and to
    `docs/BUILD.md`.

### Tier 2: real batch support and a real model suite (two to four weeks)

19. Batch loops in `conv2d`, `pool`, and `matmul`. Batched test model. Then remove
    the Tier 0 guard.
20. Lower `transpose` and `concat`: new `npuisa` opcodes, simulator kernels, lit
    tests, ISA manual entries.
21. Import `MatMul`, `Add`, `Identity`, `GlobalAveragePool`, `Concat`, `Clip`,
    `BatchNormalization`.
22. Grow `MODELS` to the six model suite in section 3.4.
23. Wire `-npu-fold-batchnorm` into `-O2`, export benchmark models without torch's
    constant folding, and produce a real ablation row for it.
24. Bias fusion pattern (`add(conv(x,w), b)` into the bias operand), and rename or
    fix the misleading lit test.
25. `inferReturnTypes` plus arithmetic shape verification for conv, matmul, pools.
26. Enable `-cse` and `-sccp` in `_passes_for_level` (section 9.2 item 4). They are
    already linked into `npu-opt`; this is two lines and gives two ablation rows.

### Tier 3: depth (one to two months)

27. Cost aware spill heuristic, benchmarked against the current one as an ablation.
    This is where the tight budget regression in section 5.1 gets fixed.
28. O(n log n) liveness and allocation.
29. Two port cost model with DMA and compute overlap. **Reassess after item 15:**
    SCALE-Sim already models double buffering, so once its numbers are recorded the
    question becomes whether the analytical model still needs to be improved or
    whether it is more honest to keep it deliberately simple and let SCALE-Sim carry
    the overlap story. Decide with the divergence data in hand, not before.
30. Parallel or im2col convolution kernel so the model suite can grow.
31. The memref plus memory space migration (section 9.2 item 1): replace
    `!npuisa.buffer` plus the `i64 address` attribute with
    `memref<..., #npu.scratchpad>` and `#npu.dram`. Own long lived branch. Must
    land **before** item 33, since both change the type system.
32. `TilingInterface` and `DestinationStyleOpInterface` (section 9.2 item 2), which
    the memref migration makes tractable.
33. INT8 quantization end to end: `quantize`/`dequantize` ops, a calibration pass,
    `QUANT`/`DEQUANT` opcodes, integer kernels, accuracy versus cycles evaluation.
    This is the single largest and most domain relevant addition available.
34. Asynchronous DMA with tokens plus a double buffering pass (section 9.2 item 3).
35. Region based `npu.fused_op` replacing the two case `Activation` enum
    (section 9.2 item 5).
36. Layout assignment as an optimization (NCHW versus NHWC), with the layout
    encoding attribute the spec's design section already anticipates.
37. Tiling and loop fusion for convolutions that exceed the scratchpad, which is
    the real reason edge NPU compilers are hard and which the current design sides
    steps by spilling whole tensors. Depends on item 32.
38. Debug section in `.nbin` mapping program counter to ONNX node name, plus
    `npu-sim --trace` (section 9.2 item 6).
39. Optional, and only after item 31: retarget to Gemmini (section 10.3 item 4).

---

## 8. What not to change

For balance, the things that are right and should be left alone:

- The two dialect split, and the decision to use MLIR's dialect conversion
  framework with a real `TypeConverter` for the representation change. That is the
  correct infrastructure and it is used correctly.
- Bias as an operand plus activation as an attribute, rather than a combinatorial
  explosion of fused op names.
- Builtin `tensor` over a custom type.
- The straight line no branch ISA, given the static DAG scope.
- `Pure` traits driving generic DCE rather than a hand written pass.
- The engineering log and debug report as first class deliverables.
- The dash lint rule, which is unusual and is enforced consistently.

---

## 9. Triage of the external review

A second review of this project was supplied as an annotated issue list
(`ISSUE_001` through `ISSUE_027` plus a five phase upgrade roadmap). It is a good
document and most of it should be adopted. It was clearly written by reading the
repository rather than running it, which shows in two ways: it gets three facts
about the code wrong, and it misses every defect that only appears when you execute
something. Both lists are worth having, because they fail in opposite directions.
This one knows what a production NPU compiler looks like; sections 2 through 5
above know what this compiler actually does.

### 9.1 Wrong on the facts

| Claim | Reality |
|---|---|
| `ISSUE_018`: ".nbin no magic/version" | The format has both. `kMagic = "NPUB"` is written and checked; a `u32 version` is written. The real defect is narrower and is in section 2.3: the version is **never read back or validated**. The rest of `ISSUE_018` (no debug section, no relocation, no section table) is correct. |
| `ISSUE_027`: "no `-dump-ir-after-all`, `-time-passes`" | `npu-opt` is built on `MlirOptMain`, so it already exposes `--mlir-print-ir-after-all`, `--mlir-print-ir-after`, `--mlir-print-ir-after-change`, `--mlir-print-ir-after-failure`, `--mlir-timing`, and `--dump-pass-pipeline`. They exist under MLIR's standard names. Verified against `npu-opt --help`. |
| `ISSUE_001`: "no GroupConv, no DepthwiseConv" | Grouped convolution **is** supported end to end: the importer reads the ONNX `group` attribute, it survives lowering, it is encoded, and `Simulator.cpp` implements it (`outPerGroup = O / group`). Depthwise convolution is the `group == C` case and therefore also works. The correct criticism is that neither is **tested** or exercised by any model, which is section 3.4's problem, not a missing feature. |
| `ISSUE_025`: "90 percent coverage but small graphs" | The graphs are indeed small, but the deeper problem is that the 90 percent figure comes from a script that runs `ninja check-npu \|\| true` and reports coverage whether or not the tests passed. See section 4.6. |
| `ISSUE_012-017`: "OOM on ResNet-50" | It would not run out of memory. The allocator spills to DRAM, so it would complete and be pathologically slow, both to compile (the O(n^3) liveness recomputation in section 5.2) and to simulate (the naive six deep loop nest in section 5.4). Slow, not OOM. The recommended fix is still right. |

### 9.2 What it found that this document missed

These are genuine additions and they are folded into the revised roadmap in section
11. Ranked by value:

1. **`ISSUE_005`, memory spaces instead of raw addresses.** The strongest idea in
   the list. Replacing `!npuisa.buffer` plus an `i64 address` attribute with
   `memref<1x6x24x24xf32, #npu.scratchpad>` would put the project on MLIR's own
   bufferization and memory space infrastructure instead of beside it. The address
   assignment becomes a real allocation on a typed address space, `one-shot-bufferize`
   becomes available, and the lowering stops being a bespoke `TypeConverter` for a
   bespoke type. This is the single change most likely to make a reviewer say "this
   person understands MLIR" rather than "this person used MLIR".

2. **`ISSUE_004`, `TilingInterface`.** Implementing it is the supported path to
   tiling, and tiling is what section 5.7 and item 28 identify as the actual edge
   NPU problem. `DestinationStyleOpInterface` and `MemoryEffectOpInterface` come
   along with the memref migration.

3. **Phase 2.2, asynchronous DMA with tokens plus double buffering.** This pairs
   exactly with the two port cost model in section 5.3, and together they are the
   pair that makes the fusion and tiling results quantitative. A `%token =
   npuisa.dma_load_async` plus `npuisa.await %token` formulation is the right shape.

4. **`ISSUE_007-011`, the missing standard passes.** Note that `-cse` and `-sccp`
   are already linked into `npu-opt` from MLIR core and cost nothing to enable;
   they are simply absent from `_passes_for_level`. Constant deduplication,
   transpose elimination, and layout optimization are real work. This is the
   cheapest item on either list: two existing passes, one line each.

5. **`ISSUE_006`, region based fusion.** `npu.fused_op { ... yield }` is the
   principled fix for the two case `Activation` enum, and it generalizes to the
   elementwise chains a two case enum can never express.

6. **Phase 5.3, debug information in the binary.** A `.nbin` debug section mapping
   program counter to originating ONNX node name, plus `npu-sim --trace`, is cheap
   and turns the simulator into a debugging tool rather than a black box. It would
   also have made the spill bug in the debug report far easier to find.

7. **Phase 5.1, an autotuner over tile sizes and layouts.** Correct as a research
   track item, correctly placed last.

### 9.3 Where I would push back

- **Phase 1.5, "ONNX to TOSA to Linalg to npu".** This is a real architectural
  option and it would deliver op coverage almost for free, but it is not obviously
  the right call here. The hand written importer is one of the few parts of this
  project that demonstrates understanding of ONNX semantics rather than
  understanding of how to call a conversion pass. Routing through TOSA replaces
  that with configuration. My recommendation is to keep the direct importer as the
  primary path and add a TOSA ingest as an **alternative** front end, which gives
  the op coverage, keeps the original work visible, and makes a genuinely
  interesting comparison section in the report.

- **`ISSUE_012-017`, "graph coloring".** Graph coloring is the right answer when
  you have control flow and interference is a genuine graph. This ISA is straight
  line, single block, by explicit design decision. On straight line code, live
  ranges are intervals, interference is an interval graph, and linear scan over
  intervals is optimal in colours used. Swapping to graph coloring would be
  ceremony without benefit. The parts of that recommendation that **do** matter are
  the ones about spilling and tiling: a Belady style spill cost, buffer reuse, and
  tiling so that spilling is not the only response to a tight budget. That is where
  the 45 percent tight budget regression in section 5.1 lives.

- **`ISSUE_002`, quantization first.** Agreed that it is the highest value large
  feature, and it is already item 26. But it should come after the memref migration,
  not before it as Phase 1.1 proposes. Quantization changes the type system, the
  memory model change also changes the type system, and doing them in the wrong
  order means doing the second one twice.

### 9.4 What neither list would have caught without the other

For the record, the defects in section 2 do not appear anywhere in the external
review: the batch greater than 1 silent wrong answer, the three unlowerable ops,
the unreachable BatchNorm pass, the CI that runs no tests, the untracked and stale
results, the regex based instruction count, and the tight budget performance
regression. All seven required executing the code. Conversely, the memory space
and tiling interface arguments in section 9.2 do not appear in sections 2 through
5, because they are about what the design is missing rather than what it gets
wrong. Use both.

---

## 10. Choosing a simulator

The question raised was whether to replace the in house simulator with an
established one. I looked at all five candidates. The short answer is that this is
the wrong framing, because the five tools are not five versions of the same thing:
they split cleanly into tools that would **replace** this project and tools that
would **strengthen** it.

### 10.1 The decisive question

The in house simulator does two separate jobs:

- **Job A, correctness oracle.** It executes fp32 data and produces numbers that
  are compared against onnxruntime. This is what proves the compiler is correct.
- **Job B, performance model.** It estimates cycles and DRAM traffic from an
  analytical cost model. This is what makes the evaluation interesting.

Job B is weak (section 5.3: no parallelism, no overlap, hand picked constants) and
is worth outsourcing. Job A is the entire correctness argument of the project and
is currently the only thing that works properly.

So the first thing to check for each candidate is: **what does it consume?** A
simulator that consumes an ONNX graph does its own scheduling and mapping, which
means it replaces your compiler. A simulator that consumes an instruction stream or
a mapping is downstream of your compiler and complements it.

### 10.2 The candidates

| Simulator | Consumes | Executes real data | Fit |
|---|---|---|---|
| **ONNXim** | ONNX graph | No (perf only) | **Do not use.** It ingests the model directly and does its own mapping, NoC, and DRAM simulation. Adopting it means your compiler is bypassed entirely and the project becomes "I configured a simulator". It is a competitor to this project, not a component of it. |
| **STONNE** | Per layer description plus a tile configuration | Yes, per layer | Poor fit. Cycle level and genuinely good for microarchitecture exploration (MAERI, SIGMA, Eyeriss-v2), but it is layer at a time. You would lose whole program scratchpad allocation, spilling, and DMA scheduling, which is the most interesting part of the compiler. |
| **Timeloop / Sparseloop** | Architecture plus problem plus mapping YAML | No, purely analytical | Good **secondary** fit. It cannot validate numerics at all, so it can never replace Job A. But a Timeloop mapping is essentially what a tiling and allocation pass produces, so it is an excellent cross check for Phase 2 tiling decisions and for energy numbers, which nothing currently models. BSD-3-Clause, very actively maintained. |
| **SCALE-Sim v2/v3** | Architecture config plus a topology CSV | No, models data movement only | **Best primary fit for Job B.** Cycle accurate for exactly the systolic array you already assume, MIT licensed, pure Python so it drops into the existing Python driver with no new toolchain, and v3 adds Ramulator DRAM modelling with bank conflicts plus Accelergy energy. The compiler can emit the topology CSV from the `npuisa` IR in well under a hundred lines. |
| **Gemmini** | Real RISC-V custom instructions (`mvin`, `mvout`, `matmul.preload`, `matmul.compute`) | Yes, via Spike, Verilator, or FireSim | **Best ceiling, highest cost.** The only candidate that is a real hardware ISA with a real scratchpad and accumulator, and the only one that gives numerics **and** cycle accuracy. Retargeting `npuisa` to Gemmini changes the project's claim from "I designed an ISA and simulated it" to "I wrote an MLIR backend for a real open source accelerator". The cost is a full Chipyard, Chisel, RISC-V GCC, Spike, and Verilator toolchain, which is heavy on a machine capped at 12 GB and 8 processors. |

### 10.3 Recommendation

Do not replace the in house simulator. Do this instead, in order:

1. **Keep and fix the in house simulator as the correctness oracle.** It is the
   only tool of the five that validates whole program numerics against onnxruntime,
   which is the project's central correctness claim. Fix the batch bug (section
   2.1) and add bounds checking (section 2.3). This is not sentimentality: none of
   the five alternatives does this job.

2. **Adopt SCALE-Sim as the performance model.** Emit a topology CSV from the
   `npuisa` IR and report SCALE-Sim cycles alongside the analytical estimate. Cheap
   (Python, MIT, no new toolchain), and it converts the weakest claim in the report
   ("cycles from constants I chose") into the strongest ("cycles from an
   independently developed cycle accurate systolic array model, cross checked
   against my analytical model"). Keep the analytical model too, and make the
   agreement or disagreement between them a subsection of the evaluation. That
   comparison is worth more than either number alone.

3. **Use ZigZag (preferred) or Timeloop for tiling design space exploration in
   Phase 2.** See section 10.5: ZigZag does the same job with a more flexible
   architecture template and is more actively maintained. Secondary, not primary.

4. **Treat Gemmini as the stretch goal**, and only after the memref migration.
   `npuisa` already has the two shapes Gemmini needs, an explicit scratchpad and
   explicit DMA, so `dma_load`/`dma_store` map onto `mvin`/`mvout` and the matmul
   maps onto `matmul.preload` plus `matmul.compute` fairly directly. There is
   published precedent for generating Gemmini backends. If this happens, it is the
   single most impressive thing the project could contain.

5. **Skip STONNE and ONNXim.** STONNE solves a problem this project does not have
   (microarchitecture exploration). ONNXim solves the problem this project *is*.

### 10.4 A second survey, beyond the original five

The five candidates in section 10.2 were supplied. I searched for others to make
sure nothing better existed. Seven more were examined: Accelergy, ZigZag, Stream,
PyTorchSim, VTA, NVDLA, and NPUsim. One is a clear addition, one is a better
version of something already recommended, and the rest confirm a structural point
worth stating explicitly.

**The structural point.** Essentially no free simulator accepts an arbitrary
external instruction stream. The ecosystem splits cleanly in two:

- **Simulators that come with their own compiler** (ONNXim, VTA, PyTorchSim,
  NVDLA). Each ingests a model and does its own lowering, so adopting one means
  discarding the compiler. PyTorchSim's own documentation is explicit that it does
  not accept externally generated instruction streams.
- **Analytical mapping models that do not execute code at all** (Timeloop,
  ZigZag, Stream, MAESTRO, Accelergy). These consume an architecture plus a mapping
  or a set of action counts and return performance, energy, or area. They can never
  validate numerics.

The only thing in either survey that takes an instruction stream from an outside
compiler is Gemmini, and that is because Gemmini is a real hardware ISA rather than
a simulator. This is not an accident: a custom ISA implies a custom simulator.
**It strengthens decision D1.** Writing the simulator was not a shortcut around
using a real one; for a project that invents its own ISA, it is the only option.
That is worth one honest paragraph in the report, because a reader will otherwise
assume the in house simulator exists because established ones were not considered.

### 10.5 The two worth adopting

**Accelergy, and it should be added.** This is the most valuable result of the
second survey, because it fills a gap nothing else in the stack covers.

- It estimates **energy and area**, which nothing currently in or recommended for
  this project produces. For an **edge** NPU this is arguably the headline metric:
  the entire reason edge accelerators have a small scratchpad and explicit DMA is
  energy, and the report currently cannot say anything quantitative about it.
- It consumes **action counts**, which the simulator already produces. `Stats`
  already tracks MAC counts implicitly through `macCycles`, plus `dramBytesRead`,
  `dramBytesWritten`, and `instructions`. Feeding Accelergy means mapping those to
  its action count format through its documented estimation plug in API, not
  instrumenting anything new.
- MIT licensed, actively maintained (v0.4 in 2024), and it is the standard energy
  companion to Timeloop, so it composes with the section 10.3 item 3 work.
- It also makes the fusion argument land properly. Fusion's benefit is that an
  intermediate never leaves the scratchpad. Measured in cycles that is a modest
  win; measured in **energy** it is a large one, because a DRAM access costs
  orders of magnitude more energy than a scratchpad access. Right now the project
  argues for fusion using the metric that flatters it least.

**ZigZag, preferred over Timeloop for the design space exploration role.** Same
job, and its own paper positions it as extending Timeloop and MAESTRO with a more
flexible architecture template and a larger space of valid mappings. Actively
maintained by KU Leuven MICAS, Python, and easier to drive than Timeloop's YAML
triples. Its sibling **Stream** handles multi core layer fused scheduling, which is
out of scope here but is the natural continuation if the project ever grows past a
single core. Substitute ZigZag for Timeloop in section 10.3 item 3; keep Timeloop
in mind only if a specific Timeloop-only feature is needed.

### 10.6 The rest, and why not

| Tool | Verdict |
|---|---|
| **PyTorchSim** (MICRO 2025) | **Not a component, but read it.** Its flow is PyTorch to FX to **MLIR** to LLVM to a RISC-V based NPU ISA, executed on Spike for functional correctness and gem5 plus Ramulator2 plus BookSim2 for timing. That is, very nearly, a research grade version of this project. It cannot be adopted (it does not take external instruction streams and is bound to its own stack), but it is the single best reference for what this project looks like at maturity, and it belongs in the related work section of the report. MIT, very active. |
| **VTA** (Apache TVM) | Skip as a component. Apache-2.0 and genuinely instruction stream based, but the stream is produced by TVM and the accelerator is inseparable from the TVM backend. Same failure mode as ONNXim. Worth one line in related work as the other well known open ISA plus simulator plus compiler triple. |
| **NVDLA virtual platform** | Skip. A SystemC model of a real industrial accelerator, which sounds ideal, but the virtual platform repository is effectively dormant (single digit commits on master, hardware index pinned to an old revision) and NVDLA has not seen meaningful upstream activity in years. Adopting a dead SystemC toolchain is a worse version of the Gemmini cost with none of the upside. |
| **NPUsim** (Yonsei) | Skip. Full model, cycle level, value aware, which is a good combination on paper, but it is a 2021 artifact with limited activity since, and it takes layer descriptions rather than an instruction stream. |
| **MAESTRO** | Skip. Superseded for this purpose by ZigZag, which its own authors position as the more flexible successor. |
| **Stream** (KU Leuven) | Not now. Multi core layer fused scheduling is out of scope for a single core design, but it is the right tool if the project ever adds multiple cores. |

### 10.7 Revised simulator stack

Putting sections 10.3 and 10.5 together, the recommended final stack is four tools,
each doing one job nothing else does:

| Job | Tool | Status |
|---|---|---|
| Numerical correctness, whole program | **the in house simulator** | exists; fix per sections 2.1 and 2.3 |
| Cycle counts, cross checked | **SCALE-Sim** plus the analytical model | decision D2, roadmap item 15 |
| **Energy and area** | **Accelergy** | new, roadmap item 16 |
| Tiling design space exploration | **ZigZag** | Phase 2, secondary |
| Real hardware validation | **Gemmini** | optional stretch, roadmap item 39 |

---

## 11. Rewrite or upgrade

The question is whether the amount of change here justifies revising
`npu_mlir_compiler_spec.md` and having an agent rebuild from scratch, rather than
upgrading in place.

**Upgrade in place. It is both cheaper and faster, and it produces a better
artifact.** The reasoning:

### 11.1 Why a rewrite is more expensive than it looks

- **The codebase is small; the scaffolding is not.** The whole project is about
  5,200 lines. But the expensive part of an out of tree MLIR project is never the
  lines, it is getting LLVM built at a pinned tag, the CMake and TableGen
  dependency order right, `lit` and `FileCheck` discovering tests, the Python
  bindings importable, and a `TypeConverter` based partial conversion that actually
  legalizes. All of that works today and represents the bulk of the original effort.
  A rewrite pays for it again and gains nothing.

- **A rewrite reproduces the same class of bug.** The batch size hardcode in
  section 2.1 exists because LeNet is batch 1 and LeNet is the only model. Hand a
  fresh agent a spec whose stated scope is still "fp32 CNN inference" with "a LeNet
  style CNN for the core suite", and it will take the same shortcut for the same
  reason. **The spec is what produced these gaps.** Section 3 is largely a list of
  places where the implementation matched the spec and the spec was too narrow.

- **You lose the git history, which is a deliverable.** Fifteen conventional commits
  across twelve phases document a real build. A fresh repository whose history is
  one enormous "implement the compiler" commit reads as machine generated, because
  it is. For a portfolio piece this is a real cost.

- **You lose the engineering log and the debug report.** `docs/ENGINEERING_LOG.md`
  and `report_debug/` are built from problems actually encountered, including the
  spill encoding bug the README highlights. These cannot be regenerated from a
  specification; they are a record of events. The spec itself ranks the debug
  report equal to the main report.

### 11.2 Why the upgrades are less invasive than the list suggests

Sorting the combined roadmap by how much existing code it disturbs:

- **Purely additive, touches nothing (roughly 60 percent):** every new ONNX op,
  every new model, the new passes, CSE and SCCP, CI, tests, SCALE-Sim integration,
  progress bars, debug sections, ablations, results tracking. All of this is new
  files and new lines.
- **Localized changes (roughly 25 percent):** the batch fix is three loop nests in
  one file. The lowering gaps are three new patterns in one file. The spill
  heuristic is one function. Bounds checking is one file.
- **Genuinely architectural (roughly 15 percent):** the memref plus memory space
  migration, and quantization. These two rewrite the `npuisa` type system.

Only that last 15 percent resembles a rewrite, and even it is a branch: one type
file, one lowering pass, one allocator, one encoder. It is a large refactor of four
files, not a new project. Do it on a branch, keep the lit tests green, merge when it
lands.

### 11.3 What to actually do

**Revise the spec and upgrade in place.** These are not alternatives; the revised
spec is the contract for the upgrade work, not the trigger for a restart.

1. Write `npu_mlir_compiler_spec_v3.md` incorporating sections 7, 9, and 10 of this
   document. The critical edits are to the **scope and the phase gates**, not the
   task list. Specifically: core scope becomes multi model and multi batch;
   "validated end to end" becomes "on every model in the suite at every optimization
   level and at least two batch sizes"; the definition of done gains a row for CI
   actually building and testing; and the model suite from section 3.4 becomes a
   named requirement rather than a stretch goal. Those four edits alone would have
   prevented most of section 2 and 3.
2. Work the tiers in section 7 in order, on branches, with the existing tests as the
   regression net. Tier 0 is days and removes the embarrassing failures. Do it first
   regardless of what else you decide.
3. Treat the two architectural items (memref migration, then quantization, in that
   order) as their own long lived branches.
4. Keep committing in the existing style. The history continuing from `v1.0.0`
   through a `v2.0.0` that adds real hardware modelling is a better story than a
   second repository that starts from nothing.

### 11.4 The one case for a separate repository

If the Gemmini retargeting in section 10.3 item 4 happens, that is a reasonable
place to fork, because it drags in Chipyard, Chisel, and a RISC-V toolchain that
have nothing to do with this repository's build. Either a sibling repository or an
optional subdirectory behind a CMake flag works. Everything else belongs here.

### 11.5 Rough cost comparison

Estimates, in the same units the spec uses, and no more reliable than the spec's
own were:

| Path | Effort | First useful output |
|---|---|---|
| Rewrite from an expanded spec | Original 8 to 12 sessions to reach parity, plus the new scope on top. Call it 15 to 20. | Nothing until it reaches parity with what already works. |
| Tier 0 upgrade | About 1 session | Immediately: no more silent wrong answers. |
| Tiers 0 and 1 | 5 to 7 sessions | A project whose claims are all true, whose CI proves it, and whose cycle numbers are cross checked against SCALE-Sim. |
| Tiers 0 through 2 | 9 to 13 sessions | Multi model, multi batch, real op coverage, BatchNorm pass actually firing. |
| Tiers 0 through 3 | 25 to 35 sessions | Quantization, tiling, async DMA, memref migration, optionally Gemmini. |

Tier 1 grew by one to two sessions to absorb the SCALE-Sim work (decision D2), and
Tier 3 grew substantially after folding in the adopted items from section 9.2. Tier
3 is deliberately a menu rather than a sequence: items 29 through 31 are the
architectural spine and should be taken in order, but the rest can be picked off
individually as interest and time allow.

The upgrade path reaches "everything the README claims is now true" in five to seven
sessions. The rewrite path is still building scaffolding at that point.

---

## 12. Targeting Gemmini: feasibility

Written because the question was asked directly. Short answer: yes, it is
achievable on this machine, and it would be the most valuable thing in the project.
But it is not a retarget that bolts onto the current design, and the reason is
worth understanding before committing.

### 12.1 First, a correction on what "real hardware" means here

Gemmini is a **generator**, written in Chisel, that emits Verilog for a systolic
array attached to a RISC-V Rocket core over the RoCC interface. Berkeley has taped
it out, so the RTL is silicon proven. But unless you buy FPGA time (FireSim on AWS
F1) or a board, you will still be **simulating** it. There are three levels:

| Level | What runs | Gives you | Cost on this machine |
|---|---|---|---|
| **Spike** | A functional ISA simulator with the Gemmini extension | Numerical correctness. No timing. | Comfortable in 12 GB |
| **Verilator** | The actual generated RTL | Cycle counts from the real hardware description | Tight; see 12.4 |
| **FireSim** | The RTL on an FPGA | Fast cycle accurate | AWS spend, out of scope |

Even Spike-only is a real step up from the status quo, because the instructions
are a real ISA rather than one you invented. Verilator is the honest version of
"real hardware numbers" without silicon.

### 12.2 The blocker: Gemmini's ISA forces three roadmap items

This is the part that changes the plan. Four concrete mismatches between `npuisa`
and Gemmini, all verified against the Gemmini ISA documentation:

1. **The systolic array can only execute matmuls up to `DIM x DIM`**, where `DIM`
   is 16 by default. LeNet's first fully connected layer is 256x120. There is no
   valid instruction sequence for that without splitting it into 16x16 tiles.
   **Tiling (item 37) is a hard prerequisite, not an optimization.**

2. **The scratchpad is row addressed, not byte addressed.** Each row is `DIM`
   elements wide. `AllocateScratchpad` assigns byte offsets. The allocator has to
   be re-expressed in rows.

3. **There is a separate accumulator memory with semantics encoded in the address.**
   Bit 31 selects scratchpad versus accumulator, bit 30 selects overwrite versus
   accumulate, bit 29 selects the read data type. The current model has one flat
   scratchpad and no concept of a second address space. **This is exactly what the
   memref plus memory space migration (item 31) exists to express**, which promotes
   item 31 from "good MLIR practice" to "required".

4. **The default data types are INT8 in, INT32 accumulate.** The project is fp32
   only. Gemmini has floating point configurations, but INT8 is the well trodden
   path and the one the bundled tests and documentation assume. **Either
   quantization (item 33) becomes a prerequisite, or you commit to the less
   exercised FP config.** Verify the current state of `GemminiFPConfigs` before
   deciding; I could not confirm how well maintained that path is.

So Gemmini requires items 31, 33, and 37, which are the three hardest things on
the roadmap.

### 12.3 Why that is good news, not bad

Reframe it. Right now those three items are on a list, competing with easier work,
with no forcing function. Gemmini turns "I should probably add tiling" into "I
cannot emit a single valid instruction without tiling". An external, non-negotiable
hardware constraint is the best possible reason to build the three features that
would most improve the compiler anyway.

**Recommendation: stop treating Gemmini as item 39 at the end of the queue and
start treating it as the north star that justifies items 31, 33, and 37.** Same
work, better motivation, and a far better story: the project stops being "I
designed an ISA and simulated it" and becomes "I wrote an MLIR backend for a real
silicon proven accelerator, and here is my tiling compared against the hardware's
own."

### 12.4 The escape hatch, and why it is also an experiment

Gemmini provides `gemmini_loop_ws`, a CISC instruction that performs a larger
matmul by tiling **in hardware**, subject to using at most half the scratchpad so
it can double buffer. There is a matching `gemmini_loop_conv_ws` for convolution.

This means there are two viable backends:

- **The cheap path.** Emit `loop_ws` and `loop_conv_ws` and let the hardware tile.
  You get a working LeNet on real Gemmini much sooner, and you skip item 37. The
  honest caveat is that the interesting work is then being done by the accelerator,
  not by your compiler.
- **The real path.** Implement tiling in the compiler and emit explicit
  `matmul.preload` plus `matmul.compute` sequences.

Do both, in that order, and **the comparison between them is the most interesting
evaluation the project could contain**: compiler-directed tiling versus the
hardware's built in loop, measured in cycles on the same RTL. That is a genuine
research-shaped question with a real answer, and almost nothing else in this
roadmap can say that.

### 12.5 A staged plan

Each stage is independently valuable and independently abandonable. Stop whenever
the return stops justifying the cost.

| Stage | Work | Sessions (estimate) | Gate |
|---|---|---|---|
| **0. Toolchain** | Install Chipyard, `riscv-tools`, Spike. Build and run Gemmini's own bundled C tests. **No compiler work.** | 1 | If this fails on 12 GB, stop here. Everything downstream depends on it. |
| **1. One tile** | Generate a single 16x16 matmul: `config_ex`, `mvin`, `matmul.preload`, `matmul.compute.preloaded`, `mvout`. Validate against numpy on Spike. | 1 to 2 | Proves the encoder emits a valid instruction stream. |
| **2. Cheap path** | Emit `loop_ws`. Run LeNet's fully connected layers end to end on Spike. | 2 to 3 | First real "it runs on Gemmini" result. |
| **3. Real path** | Items 31 and 37. Compiler-directed tiling, explicit matmul sequences. | 4 to 6 | The comparison in 12.4 becomes possible. |
| **4. Convolution** | im2col in the compiler, or `loop_conv_ws`. Whole LeNet. | 2 to 3 | End to end on real hardware. |
| **5. Cycle counts** | Verilator. This is where the memory ceiling bites. | 1 to 3 | Real RTL cycle numbers. |

Realistically 11 to 18 sessions on top of the base roadmap. That is a large
commitment and should be made deliberately, not drifted into.

### 12.6 Machine requirements

Verified against this machine on 2026-08-08.

- **Disk: not a constraint.** 897 GB free on the WSL rootfs. Chipyard is large but
  nowhere near that.
- **CPU: not a constraint.** WSL sees all 28 logical processors.
- **RAM: the only constraint, and it only bites at Stage 5.** Chipyard documents
  that elaborating a plain `RocketConfig` needs about 6.5 GB and that other
  configurations need more; a Gemmini config adds the array generator on top.
  Chisel and FIRRTL elaboration is a **single JVM process**, so unlike the LLVM
  build its peak cannot be reduced by lowering the job count. The 8 GB swap gives
  roughly 20 GB of address space, so expect thrashing rather than an OOM kill, and
  budget 20-plus minutes for elaboration instead of a few.
- **Do not raise `memory=12GB`.** That ceiling exists because removing it crashed
  the machine. Stages 0 through 4 all fit inside it comfortably. If Stage 5 proves
  impossible, the only upgrade that would change the answer is more host RAM (64 GB
  would allow a 24 GB WSL ceiling while leaving Windows its headroom). A faster CPU
  or more disk would not help.

One correction to most tutorials found online: **Gemmini now uses the standard
`riscv-tools`, not `esp-tools`.** Chipyard changed this; `esp-tools` is for Hwacha
now. Most Gemmini blog posts predate the change.

### 12.7 Where it should live

Section 11.4 argued for a separate repository. Refined, now that the dependency on
items 31, 33, and 37 is clear:

- Items 31, 33, and 37 are **compiler** work and belong in `npu-mlir`. They improve
  the project whether or not Gemmini ever happens, which also means Stages 0 to 2
  can be abandoned without waste.
- Only the Gemmini **backend** (the new lowering, the instruction encoder, and the
  Chipyard integration) should be split out, either as a sibling repository or as a
  subdirectory behind a CMake flag, because Chipyard, Chisel, and a RISC-V
  toolchain have nothing to do with this repository's build.

---

## 13. Re-audit, 2026-08-09

The repository was re-audited two days after the first audit, at `6d62406`
(phases U0, U1, U2 merged) plus the uncommitted phase U3 working tree. Method
as before: everything below was established by building the tree and running
every suite, not by reading alone. The inline **Status 2026-08-09** notes in
sections 2 through 7 record what each original finding has become; this
section holds the evidence, the defects that are new since the first audit,
and the updated order of work.

### 13.1 What the three merged phases delivered, verified by execution

- **U0, baseline and safety net.** `scripts/regression_baseline.py` records
  five suites test by test and six benchmark cells with golden fp32 outputs,
  `--check` classifies drift against notes, `ninja baseline-check` exists, and
  `docs/BREAKING_CHANGES.md` establishes the protocol (still reading "None
  yet"). Notably the baseline records the simulator's true instruction
  counts, 28 / 25 / 21 at the default budget, which is what exposed the regex
  inflation as a committed contradiction (section 4.3 status).
- **U1, stop being wrong.** The batch guard at two layers, the corrected
  docstring, tracked and regenerated results, `staleness()` with nine pinning
  tests, the tightened e2e tolerance, and the fixed `coverage.sh`. Roadmap
  items 1 through 6 are genuinely done as annotated.
- **U2, CI that actually tests.** Four real jobs in `ci.yml` (lint with black,
  mypy, and reachability; build and test running every suite plus the
  baseline check; ASan and UBSan; coverage with an enforced 85 percent C++
  threshold), `report.yml` building both PDFs, `llvm-image.yml` for the
  prebuilt LLVM image, a two stage rewrite of `Dockerfile.llvm`, and
  `scripts/check-reachability.py` enforcing spec section 5.2 with dated
  exemptions (its first run found six unreachable ops, three more than the
  first audit had). All of it runs and passes locally. None of it has run on
  GitHub; see 13.4.

### 13.2 The in flight phase U3, and why it cannot land yet

The working tree holds phase U3 (spec section 11) uncommitted: ten modified
files plus three new test files. What already works: `Program::validate()`
with structured errors, `decode()` now meaning decode plus validate,
`decodeUnvalidated()` for the disassembler (which deliberately dumps suspect
files, prefixed with a warning), an always on bounds checked accessor in the
simulator, input size checking and multi output writing in `npu-sim`, a multi
function diagnostic in `npu-translate`, a multi block diagnostic in
`AllocateScratchpad`, 32 validation tests, a 1000 iteration round trip
property test, and a 322 case fuzz corpus. 39 of 42 encoding tests pass. The
three failures, root caused:

1. `Validation.RejectsShapeThatWouldOverflow` fails because `shapeElements()`
   checks its running product against the 2^40 cap only after multiplying, so
   the multiply itself is the overflow (UBSan confirms signed overflow at
   `Program.cpp:197`). A shape like `{2^40, 2^24}` wraps to 0 elements and the
   guard accepts exactly what it exists to reject. Fix shape: test
   `d > kLimit / n` before multiplying.

   **Status 2026-08-09, done** (upgrade part 1, commit `fb3d4cf` on
   `upgrade/u3-harden-the-binary-interface`). Fixed in the equivalent form
   `n > kLimit / d` tested before the multiply, with `kLimit` and the signature
   unchanged so all three callers keep their contract.
   `Validation.RejectsShapeThatWouldOverflow` is green, and
   `Validation.RejectsShapeAtTheOverflowBoundary` was added to pin both sides
   of the cap by asserting which rule fires. UBSan over `Validation.*` now
   reports nothing at `Program.cpp`.
2. `Validation.SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory`
   fails because a pre existing convenience, the simulator growing the
   scratchpad to cover every result write before execution
   (`Simulator.cpp:127`), absorbs the out of range `resultAddr` the test
   plants. The auto expansion neutralizes the new result bounds check
   entirely for scratchpad writes: only a negative or misaligned address can
   trap. The sizing loop also runs before any validation, so a hostile
   program can still drive `resultAddr + elements * 4` into int64 overflow or
   an absurd allocation at exactly the entry point the hardening targets.
   Either gate the expansion behind validation or size strictly from the
   declared `scratchpadBytes`.

   **Status 2026-08-09, done** (upgrade part 2, commit `304df3c` on
   `upgrade/u3-harden-the-binary-interface`). Sized strictly from the declared
   field, which is the second of the two options: it removes the arithmetic on
   unvalidated input rather than reordering it behind a validation step. The
   test is green. Nothing real regressed, because
   `-npu-allocate-scratchpad` writes its high water mark to
   `npuisa.scratchpad_bytes` and the encoder emits that as `scratchpadBytes`,
   so a compiled program declares exactly the extent it uses; lit and the end
   to end LeNet pytest are unchanged and no `docs/BREAKING_CHANGES.md` entry
   was needed. The six hand built programs in `SimulatorTest.cpp` did depend on
   the expansion and now declare explicit sizes.
3. `Validation.RejectsRegionPastTheEndOfDram` fails for a test authoring
   reason: its `dramOffset = 8190` is not 4 byte aligned, so the alignment
   rule fires first and the test never reaches the rule it names. Use an
   aligned offset such as 8160.

   **Status 2026-08-09, done** (upgrade part 1, commit `42edd39` on
   `upgrade/u3-harden-the-binary-interface`). 8160 it is: aligned at 4 * 2040,
   and the 10 element fp32 output still spans [8160, 8200), ending 8 bytes past
   an 8192 byte DRAM, so the alignment rule passes and `region-in-range` fires.
   No decoder code changed; the rule was correct and simply had no test that
   reached it.

Two further defects found in the new code by adversarial testing (the harness
and logs are kept in WSL `/tmp/npuverify/`):

4. `validate()` never compares operand read extents to what was written: the
   written before read walk checks membership only. A validated program can
   therefore still trap in the simulator (verified: a `DMA_STORE` reading 100
   elements from a 4 element buffer at the scratchpad top passes `validate()`
   and then traps at run time), which contradicts the contract comment on
   `Program.h` that it checks every invariant the simulator relies on.
   Interior over reads that stay inside the scratchpad are silently permitted
   and read stale data.

   **Status 2026-08-09, done with a stated gap** (upgrade part 3, commit
   `a33f9dc` on `upgrade/u3-harden-the-binary-interface`). A new
   `operand-extent` rule requires the recorded count to cover what the consumer
   reads, naming instruction, operand, needed and written. It is exact for
   `DMA_STORE`, `RELU`, `ADD`, `MUL` and `RESHAPE`, a lower bound of
   `resultElements` for the two pools, and for `CONV2D` and `MATMUL` only a non
   zero recorded count. That last is the gap and is deliberate: their extents
   follow from the recorded tensor shapes and the walk tracks element counts,
   so a real check needs shape tracking, which is not attempted here. Both the
   trapping case and the interior over read are now rejected, each with its own
   test, and an exact extent read is kept passing by a third.
5. The trap path contains `assert(false)` (`Simulator.cpp:164`), so any
   assert enabled build aborts the process on the first refused access
   instead of returning `SimResult.error`, and the comment above it describes
   a clamping behavior the code does not have. The graceful refusal contract
   currently exists only under NDEBUG.

   **Status 2026-08-09, done** (upgrade part 2, commit `7e3ed6b` on
   `upgrade/u3-harden-the-binary-interface`). The assert and the now unused
   `<cassert>` include are deleted, and the comment rewritten to describe the
   code: every access checked in every build mode, the first refusal recorded
   in `SimResult.error`, `nullptr` returned, the caller skipping the access.
   Verified by running the target test in a `-DCMAKE_BUILD_TYPE=Debug` tree and
   a `-DCMAKE_BUILD_TYPE=Release` tree, which is the pair that diverged before.
   All eleven `spAt` and `dramAt` call sites were audited for a null test,
   since a missing one becomes a null dereference once the assert stops
   aborting first; all eleven already had one.

Smaller items in the same change set, worth fixing while the files are open:
the decoder's 2^28 element cap still permits roughly 2 GiB single vector
allocations from a tiny crafted file, and the corpus never probes just under
the cap; `encodeFunction`'s `.Default` case emits an error diagnostic for an
unencodable op but still returns success, so `npu-translate` can print an
error yet write the `.nbin` and exit 0 when the op's result is unused;
`npu-sim` accepts exactly one `--input`, so a multi input program runs its
other declared inputs as zeros with no warning.

**Status 2026-08-09, all three done** (upgrade part 3, on
`upgrade/u3-harden-the-binary-interface`).

- The allocation cap, commit `1a21f73`. The probes turned the request into a
  defect report: the 36 near cap cases were already refused, but refusing them
  cost 2.0 GiB peak RSS and 43 seconds for under four kilobytes of input, so
  the cap bounded the number and not the work. `getCount` now also takes the
  least space one element can occupy and refuses a count the remaining bytes
  cannot back. The same 36 cases now cost 6 MB and no measurable time. Corpus
  322 to 358, now pinned exactly rather than only floored at 200.
- `encodeFunction`'s `.Default`, commit `b326979`. Measured before the fix on
  an unlowered `npu.relu`: exit code 0 and a 150 byte `.nbin` holding the
  program with the relu deleted from it. It returns failure now, after the loop
  so one run names every unencodable op. Covered by a unit test and by
  `test/Encoding/unencodable.mlir`, which runs the real tool and checks nonzero
  exit, the diagnostic, and the absence of an output file. Note the finding
  underneath it: `not` was not a substituted tool in `test/lit.cfg.py`, so no
  negative tool test was writable for this suite before now.
- `npu-sim` multi input, commit `2e05b7f`. Implemented rather than refused.
  One `--input` per declared region, in declaration order, each size checked
  against its own region, and a count mismatch refused with both numbers named.
  `docs/ISA_MANUAL.md` now documents `npu-sim` at all, including the numbered
  multi output files.

One further defect surfaced by this part's gate, which is the first to run the
whole fuzz corpus under UBSan rather than `Validation.*` only: the corpus test
itself accumulated touched fields into a signed `int64_t`, and the corpus holds
a program declaring `INT64_MAX` bytes of scratchpad, so the test asserting that
hostile input cannot cause undefined behaviour was itself undefined behaviour.
Pre-existing, unchanged since before this part, fixed in commit `0e9c923`.

**Status 2026-08-09, section closed.** All five defects fixed, the two further
defects found by adversarial testing fixed, and all three smaller items fixed,
across upgrade parts 1 to 4. The phase merged to `main` after the full 11.3
gate with output shown. The process note below is answered: the work moved to
`upgrade/u3-harden-the-binary-interface` in part 1 and `main` was never
committed to directly.

Process note: this work sits uncommitted on the `main` checkout. Ground rule
11 of `UPGRADE_SPEC_V3.md` gives every phase an `upgrade/uN-name` branch; move
it there before committing.

### 13.3 One red pytest, and why that is correct

`test_committed_results_are_current` fails on this tree because `staleness()`
counts uncommitted edits to compiler inputs, and the U3 edits touch `lib/` and
`tools/`. That is the guard doing its job on a dirty tree, not a defect.
Consequence for the order of work: when U3 lands, regenerate the six results
on the clean tree (the numbers should not move; U3 adds no optimization), and
rebuild the report from them in the same sitting, as separate commits.

**Status 2026-08-09, done and the prediction held** (upgrade part 5, commits
`8c41cc3`, `89517a5`, `212cec3`). Regenerated on the clean post U3 tree and no
measured number moved: `instruction_count`, `simulated_cycles`, both DRAM
counters and `max_abs_error_vs_onnxruntime` are identical across all six cells,
with only `manifest.git_sha`, `manifest.timestamp` and the wall clock
`compile_ms` differing. `test_committed_results_are_current` is green, pytest is
28 of 28 with no exceptions, and the regression baseline reports no drift for
the first time.

### 13.4 New findings, not in the first audit

1. **Nothing since 2026-08-07 has been pushed.** `origin/main` is `7de39c6`;
   all twelve commits of U0 through U2 exist only locally. The README's green
   CI badge is the old lint only workflow's run at `7de39c6`. The U2 gate in
   spec section 10.3 (one green URL, four deliberate fault red URLs) is not
   just outstanding but impossible until a push happens.
2. **The LLVM container image does not exist.** An anonymous manifest probe
   for `ghcr.io/olajide-badejo/npu-mlir-llvm:22.1.8` returns 403 (never
   published; `llvm-image.yml` is dispatch only and unpushed). Pushing main
   today would turn the badge red at container pull for three of the four
   jobs. Push order that avoids advertising a red badge is in 13.5. One risk
   to plan for: the image build compiles LLVM on a 2 core public runner under
   a 6 hour job limit; if it times out, build the image locally in WSL with
   the same Dockerfile and push it once by hand, which the workflow's manual
   dispatch design already tolerates.
3. **The committed PDFs still embed numbers from a commit that does not
   exist.** Both PDFs in the tree are byte identical to the 2026-07-15
   builds, whose results carried `git_sha: 8095dbec`, the sha
   `docs/DESIGN_DECISIONS.md` itself calls the failure mode the staleness rule
   exists to prevent. Commit `cc99973` deliberately reverted the rebuilt PDFs
   out of the U2 merge to keep that commit clean, which was reasonable, but
   the follow up rebuild never happened. `report/generated/` is gitignored
   and its local `macros.tex` still says `GitSha 38af13633388`, one
   regeneration behind the committed results. Nothing enforces the spec
   section 23.1 requirement that every PDF number traces to a committed
   result; until something does, treat "results changed, rebuild both PDFs"
   as a standing manual step.

   **Status 2026-08-09, done, and the root cause was not the one assumed**
   (upgrade part 5, commit `212cec3`). The follow up rebuild had not merely been
   forgotten: it was unbuildable by the documented command. `report/Makefile`
   declared `main.pdf` as depending on `main.tex`, `references.bib` and
   `sections/*.tex`, none of which change when the results do, so
   `make -C report` printed "Nothing to be done for 'all'" and anyone running it
   would conclude the PDF was current. The results are a dependency of the rule
   now, so a stale PDF cannot recur silently, and this stops being a standing
   manual step. `report/generated/` is tracked, and
   `test_macros_match_the_committed_results` enforces spec 23.1 rather than
   leaving it to discipline. Both PDFs rebuilt from the current results.

   One caveat: `\GitSha` is defined in `macros.tex` but cited by no `.tex` file,
   so no sha ever reaches either PDF and the `strings ... | grep -c 8095dbec`
   check is satisfied trivially. Citing it in the report prose would make the
   artifact state its own provenance; that is a change to the document and was
   left out of this part's scope.
4. **Two contradictory instruction counts are committed side by side.**
   `experiments/results/*.json` records the regex count (91 / 82 / 70);
   `test/baseline/baseline.json` records the simulator's count (28 / 25 / 21)
   for the same cells. The README table, the report, and the plots inherit
   the regex numbers; the CHANGELOG openly defers the correction to phase U4.
   This also inverts first audit section 6 item 2: the README's objdump
   excerpt says "21 instructions", which matches the simulator's true `-O2`
   count exactly, so the excerpt was probably genuine all along and it is the
   headline table that is wrong.

   **Status 2026-08-09, done, and the suspicion was right** (upgrade part 7).
   The excerpt was genuine: regenerating it from a fresh
   `model_generator.export(lenet, seed=0)` compiled at `-O2` reproduces "1
   inputs, 1 outputs, 10 constants, 21 instructions" exactly, and the
   disassembly below it matches instruction for instruction. The table was the
   thing that disagreed with it. Both now come from the same run, all six
   results equal the recorded baseline, and
   `test_results_agree_with_the_recorded_baseline` makes committing two answers
   to the same question a test failure rather than something to notice a month
   later.
5. **The type checker validates a different Python than the one that runs.**
   mypy, ruff, black, and the CI lint job all target Python 3.12; the venv,
   the baseline manifest, and the README badge say 3.14. mypy's scope is also
   only `python/npu_frontend`: `experiments/`, `scripts/`, and `test/Python`
   are unchecked. Cheap to widen; fold into U4 with the other measurement
   integrity work.
6. **The README still advertises what does not run.** The pipeline diagram
   and the passes table both show the BatchNorm fold, which no `-O` level
   contains and no importable model can reach (exempt until 2026-12-31), and
   the coverage badge is a hand written shields.io URL claiming 90 / 89 while
   CI enforces only C++ at 85 percent and no Python threshold at all.
7. **The false "tagged records" phrase lives in two places**,
   `docs/ISA_MANUAL.md` and the `Program.h` header comment, and the format is
   additionally host endian despite a comment claiming little endian helpers.
   Fix all three in U3's documentation pass; the U3 gate requires the ISA
   manual to document validation and the version policy anyway.

   **Status 2026-08-09, done, and it was three places not two** (upgrade part 4,
   commit `0e3de30`). `docs/DESIGN_DECISIONS.md` carried the false phrase in a
   heading as well, which this item had not found. All three now describe fixed
   order, length prefixed sections with no tags, and the byte order is documented
   as host order with a note that a `.nbin` is not portable across byte orders.
   The code comment claiming little endian helpers says so too. No byte swapping
   was added; that is a format change. The manual also gained the validation
   table, generated from the `fail()` call sites, and the version policy.
8. **Small docstring undercount from U1:** `op_mapping.py` says six ops are
   missing but discusses `Identity` as import blocking in the same breath;
   the spec's U7 list names seven. One line fix next time the file is
   touched.
9. **The CI coverage step greps the first `lines:` figure from the gcovr
   log**, which breaks silently if gcovr's summary format shifts or another
   figure prints first. Switch to a `--json-summary` parse when U4 touches
   measurement anyway.

### 13.5 Updated order of work

Nothing here changes the tiers in section 7 or the phase order in
`UPGRADE_SPEC_V3.md`. It sharpens the immediate queue to this:

1. **Finish U3.** Fix the five defects in 13.2 (overflow safe
   `shapeElements`, gate or retire the scratchpad auto expansion so the
   result bounds check can fire, align the test constant, compare operand
   read extents in `validate()`, replace the `assert(false)` trap with the
   documented graceful refusal), get 42 of 42, move the work to its phase
   branch, and land it together with the U3 documentation items, including
   the ISA manual and header corrections from 13.4 item 7.

   **Status 2026-08-09, done** (upgrade parts 1 to 4). All five defects fixed
   plus the two adversarial findings and the three smaller items, 51 of 51
   encoding tests rather than the 42 this item asked for, the work moved to
   `upgrade/u3-harden-the-binary-interface` without ever committing to `main`,
   and the phase merged after the full 11.3 gate. Item 2 below is next.
2. **Regenerate and republish the numbers.** Rerun the six cells on the clean
   tree, rebuild both PDFs from them, and commit results, tex, and PDFs as
   their own commits (13.3, 13.4 item 3). That finally retires the 8095dbec
   numbers from the tree.

   **Status 2026-08-09, done** (upgrade part 5). Three separate commits as
   asked: results, generated tex plus figures, PDFs. No number moved. The only
   remaining mentions of `8095dbec` are historical narrative in
   `docs/ASSESSMENT.md`, `docs/DESIGN_DECISIONS.md`, `docs/ENGINEERING_LOG.md`,
   `CHANGELOG.md`, `UPGRADE_SPEC_V3.md`, and the docstring of the new
   `test_no_result_traces_to_a_missing_commit`, which exists so that a result
   citing a commit that does not exist can never pass again. Item 3 below is
   next and needs push access.
3. **Push and prove.** Push everything, dispatch `llvm-image.yml`, get the
   new CI green on GitHub, then run the four fault class proofs of spec
   section 10.2 and record the URLs. Until this step completes, the badge
   claims more than has been demonstrated, which is the exact failure mode U2
   exists to end.
4. **Proceed to U4** (measurement integrity), which now also picks up 13.4
   items 4, 5, and 9 explicitly.
