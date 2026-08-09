# Engineering log

Dated entries recorded as problems happen: symptom, root cause, options considered,
chosen fix and why, commit, verification. This log is the raw material for the debug
report (report_debug) assembled at Phase 11.

## 2026-08-09 Phase U3: the overflow guard was itself the overflow

**Symptom.** `Validation.RejectsShapeThatWouldOverflow` failed, in the most
pointed way available: the program it plants is the exact case the guard exists
for, and `validate()` accepted it. Built with `-fsanitize=undefined` the same
test also reported signed integer overflow at `Program.cpp:197`.

**Root cause.** `shapeElements()` accumulated the product and then checked it:

```cpp
n *= d;
if (n > kLimit)
  return std::nullopt;
```

A check can only run on a value that has already been computed, and computing
that value is the undefined behaviour. For a shape like `{2^40, 2^24}` the
multiply overflows int64 and wraps, so what the comparison sees is a small
number and the function returns a plausible element count for a shape it exists
to refuse. Downstream that count is multiplied by 4 and compared against a
region bound, where a wrapped product compares as comfortably in range. The
failure mode is therefore not a crash at decode but a program that validates
and then walks off the end of a buffer at run time, which is precisely what
phase U3 exists to make impossible.

Worth recording that the test was written before the implementation and was red
for a real reason, not a wrong expectation. A guard exercised only with inputs
an order of magnitude below its own limit looks correct forever.

**Options considered.**

1. Accumulate in a wider type (`__int128`) or in unsigned arithmetic and keep
   checking after the multiply. Rejected: it buys correctness with a compiler
   extension, or with wrapping semantics that are defined but still leave the
   check written the wrong way round for the next person to copy.
2. Test `n > kLimit / d` before the multiply. Chosen. It is the standard
   division based overflow test, it needs no wider type, and the `d <= 0`
   rejection immediately above it guarantees the divisor is at least 1.
3. Lower `kLimit` far enough that no product can overflow. Rejected: it changes
   an encoding contract to work around an arithmetic bug, and any cap large
   enough to be useful is still large enough to overflow when squared.

**Chosen fix.** Test the headroom before consuming it, so the guard never
performs the operation it is guarding:

```cpp
if (d <= 0 || d > kLimit)
  return std::nullopt;
if (n > kLimit / d)
  return std::nullopt;
n *= d;
```

`kLimit` stays at `int64_t{1} << 40` and the signature is unchanged, so the
three callers (the region check, the constant data check, and the instruction
result check) keep the contract they were written against. The comment above
the function now says the cap is inclusive, because that is the one part of the
behaviour a reader cannot recover without working through the integer division.

**Verification.** `Validation.RejectsShapeThatWouldOverflow` shown failing
before the change ("expected result-shape to reject this") and passing after.
Added `Validation.RejectsShapeAtTheOverflowBoundary`, which pins both sides of
the cap: `{2^20, 2^20}` is exactly 2^40 and has to fall through the shape rule
to `result-in-range`, while `{2^20, 2^21}` is one bit past and has to be caught
by `result-shape`. Asserting which rule fires is what keeps the boundary from
drifting silently, since either shape is rejected either way. A rebuilt
`build-san` running `Validation.*` under ASan and UBSan reports no diagnostic of
any kind, in particular nothing at `Program.cpp`.

## 2026-08-08 Phase U0: a regression net before any upgrade work starts

**Symptom.** Not a bug. The upgrade specification (`UPGRADE_SPEC_V3.md`) asks for a
long sequence of changes to a repository that currently works, under a prime
directive of not breaking it. Before Phase U0 there was no way to answer "did that
change move anything?" other than eyeballing test output. The four suites report
pass or fail but say nothing about instruction counts, cycles, DRAM traffic, or
numerics, which are exactly the things an optimizer change moves silently.

**Root cause.** The project has assertions but no recorded expectations. The end to
end pytest asserts agreement with onnxruntime at `rtol=1e-3`, five orders of
magnitude looser than the 3e-8 actually observed, so a numerics regression of four
orders of magnitude would pass. Nothing at all watched cycles or DRAM bytes.

**Options considered.**

1. Rely on the existing suites and read the diffs by hand each phase. Rejected:
   the interesting quantities are not asserted anywhere, so there is nothing to
   diff.
2. Add tighter assertions to the existing tests and stop there. Rejected as
   insufficient on its own: it catches numerics but not instruction counts,
   cycles, or DRAM traffic, and it would not notice a test quietly disappearing.
3. Record a machine readable snapshot of everything the repository does today and
   diff against it at every phase gate. Chosen.

**Chosen fix.** `scripts/regression-baseline.sh` (a thin wrapper that finds the
venv and the prebuilt lit) over `scripts/regression_baseline.py` (the engine).
It records per suite pass, fail, and skip counts plus the full test name list, the
cost model constants parsed straight out of `CostModel.h`, the tool versions, and
for each of the six LeNet cells the instruction count, simulated cycles, DRAM
bytes read and written, and the max absolute error against onnxruntime. The
simulated output tensor of every cell is frozen as a `.npy` golden file.

Three details worth recording:

- All four suites emit JUnit or XUnit XML natively (`llvm-lit --xunit-xml-output`,
  `--gtest_output=xml:`, `pytest --junitxml=`), so one parser covers all of them
  and the recorded shape is uniform. That was better than scraping four different
  human readable summaries.
- The instruction count comes from the simulator's `stats.instructions`, not from
  the regex in `run_benchmarks.py`. The two disagree badly: the regex reports
  91 / 82 / 70 for `-O0` / `-O1` / `-O2` and the simulator reports 28 / 25 / 21,
  because the regex counts `npuisa.const` (data, not an instruction) and matches
  inside type strings such as `!npuisa.buffer`. Recording the wrong number as the
  baseline would have baked the error in.
- The git sha and the tool versions are recorded but are notes rather than drift.
  The sha moves with every commit, and a machine update is not a behaviour change.
  `--strict-tools` promotes tool changes to failures when that is wanted.

**Verification.** Recorded at `7de39c6`: 37 passed, 0 failed across the five
suites. Ran `--check` immediately on unchanged code and it reported zero drift.
Then perturbed one cost model constant (`macsPerCycle` 256 to 255, chosen because
it feeds every conv and matmul cycle estimate while changing no numerics) and
`--check` failed with 10 items, catching it on three independent axes: the
constant itself, the `CostModelArithmetic::MatchesFormulas` GoogleTest flipping to
failed, and all six cells reporting `simulated_cycles` up by exactly 5. Reverted,
and `--check` went clean again. The net catches things.

**Incidental finding.** The first `ninja baseline-check` run reported a cmake
version change that the direct run did not. The cause is that a pip installed
cmake 4.4.0 in `~/.local/bin` shadows the apt cmake 4.2.3 on `PATH`, but only in
login shells. So the project can be configured by either of two cmake versions
depending on how the build was invoked, which is a real reproducibility hazard on
this machine. The baseline now records each tool as "resolved path: version" so
this surfaces as a legible note rather than a mysterious version change.

## 2026-08-08 Phase U2: ASan fires inside MLIR, and it is not our bug

**Symptom.** The new `sanitizers` CI job, run locally before pushing, aborts:

```
ERROR: AddressSanitizer: use-after-poison on address 0x70459d3e15b8
WRITE of size 32 at 0x70459d3e15b8 thread T0
    #1 llvm::SmallVectorImpl<std::pair<mlir::TypeID, void*>>::operator=(...)
    #2 mlir::Dialect::addType(mlir::TypeID, mlir::AbstractType&&)
    #5 mlir::BuiltinDialect::initialize()
    #10 EncodeFunction_LowersSmallProgram_Test::TestBody() EncodingTest.cpp:100
```

Every frame between the test and the fault is MLIR's own. `NPUSimulatorTests`
runs clean; only the one encoding test that constructs an `MLIRContext` fails.

**Root cause.** An instrumentation mismatch, not a memory bug. `llvm/Support/
Compiler.h` defines `LLVM_ADDRESS_SANITIZER_BUILD` purely from
`__SANITIZE_ADDRESS__`, which gcc defines whenever `-fsanitize=address` is on.
`BumpPtrAllocator` in `llvm/Support/Allocator.h` is header only and, under that
macro, poisons each new slab (line 351) and unpoisons per allocation. So the
copy of the allocator instantiated in our instrumented translation units
believes LLVM is an ASan build and poisons slab memory, while the prebuilt
`libMLIRIR.a`, compiled with no instrumentation and no matching unpoison, writes
straight into it. The shadow map shows exactly that: a clean allocation followed
by a poisoned tail being written.

**Options considered.**

1. Build a second LLVM with `-DLLVM_USE_SANITIZER=Address`. Correct, and the
   only thing that makes MLIR itself sanitizable. Rejected on cost: a second
   multi hour image, hours of CI per tag change, to find bugs in LLVM rather
   than in this project.
2. `ASAN_OPTIONS=detect_container_overflow=0`, the usual advice for linking
   instrumented code against uninstrumented LLVM. Rejected: that suppresses
   container annotation reports, and this is a `use-after-poison` from
   `BumpPtrAllocator`'s explicit poison calls, which the option does not cover.
3. Disable the annotations by defining the macro away. Rejected: `Compiler.h`
   defines it unconditionally inside the `__SANITIZE_ADDRESS__` branch, so it
   cannot be overridden without patching LLVM.
4. Scope the job to the code the job exists for. Chosen.

**Chosen fix.** The sanitizer job runs `NPUEncodingTests
--gtest_filter=-EncodeFunction.*` and the whole of `NPUSimulatorTests`. Nothing
is actually given up. ASan is here for the memory unsafe surface, which is
`Program::decode`, the `Program::validate` and fuzz corpus work landing in phase
U3, and the simulator's raw `spAt`/`dramAt` pointer arithmetic. None of it
touches MLIR. `EncodeFunction.*` still runs uninstrumented in `build-and-test`.
The exclusion is commented in `ci.yml` and in `docs/BUILD.md` with the reason,
because a bare `--gtest_filter` that looks like someone hiding a failure is
worse than the failure.

**Verification.** `NPUEncodingTests --gtest_filter=-EncodeFunction.*` passes 3
tests and `NPUSimulatorTests` passes 7, both clean under
`-fsanitize=address,undefined` with `halt_on_error=1` and `detect_leaks=1`.

## 2026-08-08 Phase U2: what the first reachability run found

**Symptom.** Not a failure so much as a measurement. The first run of
`scripts/check-reachability.py` reported six unreachable ops out of twelve, where
`docs/ASSESSMENT.md` section 2.2 had identified three.

**The three the audit already knew about**, `transpose`, `concat`, and
`batch_norm`, are missing every layer: no importer converter, no lowering
pattern, no encoder case, no simulator kernel.

**The three it did not.** `add` and `mul` are fully lowered, encoded, and
simulated, and have no ONNX converter at all, so nothing can produce one from a
real model; they exist only for hand written IR and for the lit tests. And
`avg_pool2d` is present at every layer and is exercised by nothing, because
LeNet uses max pooling throughout. That last one is the interesting category:
not broken, never run on a real graph, and invisible to a test suite that only
asks whether the tests pass.

**Chosen fix.** None yet, deliberately. Phase U2's job is the check, not the
implementation, and the gaps belong to U7 and U8. All six carry dated exemptions
in `docs/DESIGN_DECISIONS.md` and the check fails once a date passes.

One design note. The checker refuses to run at all if an op in `NPUOps.td` has no
entry in its `NPUISA_EQUIVALENT` table, rather than inferring a mapping. Adding
an op to the dialect should force an answer to "what does this become in the
ISA". An inferring checker would have quietly passed `transpose` on the day it
was added, which is the whole failure being guarded against.

## 2026-08-08 Phase U1: the published numbers came from a commit that never existed

**Symptom.** Phase U1 was meant to be routine cleanup: track the benchmark results,
and make the harness stop reusing stale ones. I wrote the staleness check first and
a test asserting the committed results match the tree. It failed, which was
expected, since `docs/ASSESSMENT.md` section 4.2 had already found the results were
three commits behind. What was not expected was the reason it gave:

```
generated at unknown commit 8095dbec, HEAD is d93e73de
```

**Root cause.** `git cat-file -e 8095dbec^{commit}` fails. That sha is not in this
repository. The assessment had read the manifest and assumed an old commit; in fact
the results were produced from a working tree that was never committed in the form
that produced them, probably an amended or discarded Phase 9 state. So the six
numbers the README headline table, the evaluation section, and both PDFs are built
from were reproducible from no point in the history. This is worse than stale. A
stale result can at least be reproduced by checking out its commit.

**Options considered.**

1. Compare the manifest sha to HEAD exactly, as `UPGRADE_SPEC_V3.md` section 8.1
   item 4 words it. Rejected after trying it. A result can only be committed by a
   commit that comes after the run that produced it, so under this rule every
   result is stale the moment it lands, every invocation regenerates the whole
   suite, and the committed files never match their own commit. The rule is
   self defeating as literally written.
2. Compare only the cost model constants. Rejected: it would not have caught this,
   since the cost model was unchanged throughout.
3. Compare the sha, and when it differs, ask whether anything that can actually
   move a number changed between then and the working tree. Chosen.

**Chosen fix.** `staleness()` replaces `valid()` and returns the reason rather than
a bool, so the harness can print why it is regenerating. A result is reusable when
its cost model constants match and either its manifest sha is HEAD, or `git diff`
between that sha and the tree touches nothing under `RESULT_INPUTS`: the dialect,
the passes, the encoder, the simulator, the tools, the frontend, and the harness
itself. Uncommitted edits count, because a result produced from a dirty tree is not
reproducible from any commit either. A sha the repository does not recognise is
stale by definition, which is what caught this. Deviation from the spec's literal
wording recorded in `docs/DESIGN_DECISIONS.md`.

**Verification.** Nine tests in `test/Python/test_benchmarks.py`, including one that
walks recent history for a documentation only commit and asserts the filter sees
through it, and `test_committed_results_are_current`, which is the standing guard.
Shown failing before the regeneration and passing after.

## 2026-08-08 Phase U1: refusing a batch size rather than lying about it

**Symptom.** `docs/ASSESSMENT.md` section 2.1: a 2 batch conv plus relu returns
1.19e-07 error on image 0 and 1.85 on image 1. No diagnostic anywhere.

**Root cause.** `Simulator.cpp` conv2d has `int64_t n = 0; // batch is 1 for the
supported models`, and `pool()` iterates `C = inS[1]` without ever seeing the batch
dimension. Neither is a coding mistake so much as a faithful implementation of a
specification whose stated scope was one LeNet at batch 1. Nothing else in the
pipeline had a reason to disagree, so the importer, the verifiers, the lowering,
and the allocator all accepted N greater than 1 and the allocator even sized the
buffers correctly for it.

**Options considered.**

1. Fix the kernels now. Rejected for this phase, not on merit: it is phase U6, it
   needs batched cost model changes and new GoogleTests, and U1's job is to stop
   the compiler being wrong, not to widen it. Shipping the guard first means the
   silent failure closes today rather than in two phases.
2. Reject any rank 4 tensor whose leading dimension is not 1. Rejected: a
   convolution weight is OIHW, so LeNet's 6x1x5x5 first kernel would be read as a
   batch of six and every real model would be refused. There is a test pinning
   this, because it is the obvious wrong way to write the guard.
3. Guard activations only, at the ops that are actually wrong. Chosen.

**Chosen fix.** `check_unbatched_activation` in the importer, called from the conv
and both pool converters, and `verifyUnbatchedActivation` in `NPUOps.cpp`, called
from the `conv2d`, `max_pool2d`, `avg_pool2d`, and `batch_norm` verifiers. Two
layers, as the spec asks, and both name the tensor and its shape and say the
limitation is tracked rather than permanent.

Deliberately not guarded: `matmul` and the elementwise ops. `matmul` iterates `M`
correctly, so a rank 2 batch is genuinely fine there, and relu, add, and mul are
elementwise over the whole buffer. Refusing those would be refusing something that
works, which the no silent failure rule does not ask for.

**Verification.** Four pytest cases and four lit cases, all shown failing before the
guard and passing after. Two of them are controls: batch 1 still imports, and a
6x1x5x5 weight is not mistaken for a batch.

## 2026-07-15 Phase 0: environment reconciliation

**Symptom.** The build spec (Section 3) assumes Ubuntu 24.04, a WSL budget of 20 GB
RAM / 24 processors, `LLVM_PARALLEL_COMPILE_JOBS=20`, and Python 3.10+. The actual
target machine reports different facts on every one of those axes, and one of them is
safety critical.

**Findings (verified 2026-07-15).**

- Disk is a non issue here. The WSL2 rootfs lives on `/dev/sdd` with about 932 GB free,
  and Windows C: has about 135 GB free. None of the spec's disk mitigations are needed.
- WSL2 Ubuntu 26.04 (not 24.04), kernel 6.18, user `elijah`, passwordless sudo.
- Toolchain present: gcc/g++ 15.2.0, cmake 4.2.3, ninja 1.13.2, git 2.53.0,
  python3 3.14.4, venv working. Missing and installed via apt: lld 21.1.8, ccache 4.12.3.
- The host `~/.wslconfig` deliberately caps WSL2 at `memory=12GB`, `processors=8`,
  `swap=8GB`, `autoMemoryReclaim=gradual`.

**Root cause of the deviation that matters.** The `.wslconfig` cap is not an oversight.
Its own header records that its absence crashed the machine on 2026-07-14: with no budget,
vmmemWSL grew to about 12.78 GB and never released it, leaving 2.2 GB free of 31.7 GB, and
the host paged itself to death (aggravated by GPU benchmark allocations spilling through
WDDM). The file explicitly instructs keeping build parallelism at or below 6 jobs.

**Options considered.**

1. Follow the spec literally: raise `.wslconfig` to 20 GB / 24 proc, build with 20 compile
   jobs. Rejected: directly reintroduces the condition that crashed the machine one day
   earlier, and 20 parallel gcc jobs compiling TableGen heavy MLIR translation units would
   exceed a 12 GB budget regardless.
2. Keep `.wslconfig` as is and adapt the build to the real budget. Chosen.

**Chosen fix.** Do not touch `.wslconfig`. Configure the one time LLVM/MLIR build with
`LLVM_PARALLEL_COMPILE_JOBS=6` and `LLVM_PARALLEL_LINK_JOBS=1`, link with lld, and enable
ccache. Accept that the build takes longer than the spec's 1 to 3 hour estimate at this
parallelism. Pin the LLVM tag to `llvmorg-22.1.8` (current tip of `release/22.x`) rather
than the moving branch, for reproducibility.

**Open risk carried forward.** Python is 3.14.4. torch, onnxruntime, and nanobind wheel
availability for 3.14 is unverified and is a risk for Phase 5 and later. To be checked
before the ONNX frontend work, with a 3.12 venv as the fallback plan.

**Verification.** `df`, `free`, `nproc`, `lsb_release`, tool `--version` probes captured
above. LLVM pinned tag confirmed by `git ls-remote`; shallow clone HEAD resolves to
`llvmorg-22.1.8`.

## 2026-07-15 Phase 0: repository relocated to WSL native storage (spaces break lit)

**Symptom.** After the LLVM build finished, `npu-opt` built and `npu-opt --help` worked,
but the very first lit test failed with `FileCheck: Too many positional arguments
specified!` and a shell error splitting a path at a space. The build tree sat under the
Windows project folder whose ancestors contain spaces (`Corrected Projects`,
`MLIR Backend for a Simulated Edge NPU`).

**Root cause.** LLVM's lit expands the `%s` substitution to the test file's real path and
does not quote it. Under the external `sh`, `npu-opt /mnt/c/.../Corrected Projects/...mlir`
splits into multiple arguments, so both `npu-opt` and `FileCheck` see too many positional
arguments. LLVM lit and FileCheck do not support spaces in paths, and lit resolves symlinks
to the real path, so a space-free symlink over the spaced directory did not help either
(verified: `%s` still expanded to the spaced `/mnt/c` realpath).

**Options considered.**

1. Keep the repo in the Windows folder and patch lit substitutions to quote `%s`. Rejected:
   fights the framework, fragile, and would have to be re-done for every future config.
2. Put the repo at a space-free Windows path such as `C:/Users/jidro/npu-mlir`. Space-free
   and my editing tools stay native, but WSL builds and lit still run over the slow 9p
   `/mnt/c` mount for the whole 12 phase project.
3. Host the canonical repo in WSL native storage at `/home/elijah/npu-mlir`. Space-free,
   native build and test speed, and the standard recommended layout for WSL2 development.
   Chosen.

**Chosen fix.** Relocated the git working tree (with history) to `/home/elijah/npu-mlir`.
Builds go to `/home/elijah/npu-mlir/build`. From Windows the repo is reachable at
`\\wsl.localhost\Ubuntu\home\elijah\npu-mlir`, which the Windows side editing tools handle
correctly (verified by reading and enumerating files over that path). A pointer file in the
original Windows folder records the new location. The original folder keeps only the spec
and that pointer, so there is a single source of truth.

**Verification.** After relocation, `ninja check-npu` reports `Passed: 1 (100.00%)` and
`npu-opt --help` lists the registered dialects. All paths in the generated
`lit.site.cfg.py` and in the `%s` expansion are now space-free.

## 2026-07-15 Phase 1: two ODS surprises building the first ops

Two failures worth recording while standing up the npu dialect, both of the "the
TableGen half and the C++ half of an interface are separate includes" family.

1. **`Variable not defined: 'SameOperandsAndResultType'`** from mlir-tblgen. This trait is
   not in `mlir/IR/OpBase.td` (it appears there only in FIXME comments). It is defined in
   `mlir/Interfaces/InferTypeOpInterface.td`, which must be included from the ops `.td`.

2. **`'InferTypeOpInterface' is not a member of 'mlir'`** at C++ compile time, after fixing
   1. Reason: `SameOperandsAndResultType` is a trait list that also mixes in
   `InferTypeOpInterface::Trait`, so the generated op class references
   `::mlir::InferTypeOpInterface`. The TableGen include is not enough; the C++ side needs
   `#include "mlir/Interfaces/InferTypeOpInterface.h"` in the ops header and the
   `MLIRInferTypeOpInterface` library on the dialect's `LINK_LIBS`.

**Lesson for later ops.** Any convenience trait may drag in an interface with a matching
C++ header and link library. When a trait is added in ODS, add its interface header and
link library at the same time rather than after the next build failure.

**Verification.** `ninja check-npu` reports `Passed: 2 (100.00%)`; the core ops round trip
through two `npu-opt` passes without change.

## 2026-07-15 Phase 2: CommonFolders poison template argument

**Symptom.** Using `constFoldBinaryOp<FloatAttr>` and `constFoldUnaryOp<FloatAttr>` from
`mlir/Dialect/CommonFolders.h` to fold the pointwise ops failed to compile with a static
assertion: "PoisonAttr is undefined, either add a dependency on UB dialect or pass void as
template argument to opt-out from poison semantics."

**Root cause.** In this LLVM release these helpers gained a `PoisonAttr` template parameter
that defaults to `ub::PoisonAttr`, whose definition is only available if the UB dialect is
linked. The parameter is the third one, after `AttrElementT` and `ElementValueT`. My first
fix passed `void` as the second argument, which wrongly set `ElementValueT` to void.

**Chosen fix.** Opt out of poison propagation by passing all three explicitly:
`constFoldBinaryOp<FloatAttr, APFloat, void>(...)` and the unary equivalent. This avoids
taking a dependency on the UB dialect for folds that cannot produce poison.

**Verification.** `-canonicalize` folds `add`, `mul`, and `relu` over constants to a single
`npu.constant`, and the canonicalization patterns (relu idempotence, reshape identity and
reshape of reshape) fire. The BatchNorm folding pass turns `bn(conv(x, W))` into one conv
with hand verified weights `[2, 6]` and bias `[-0.5, -1.5]`. `check-npu` reports 5 of 5.

## 2026-07-15 Phase 9: benchmark harness caught a spill correctness bug

**Symptom.** The first benchmark run at a tight scratchpad budget (140 KB, which forces the
allocator to spill) showed the maximum absolute error against onnxruntime jumping from
2.98e-08 (exact up to fp rounding) to about 0.08 at optimization levels O1 and O2. The
default 1 MB budget, which does not spill, stayed exact.

**Root cause.** The instruction encoder assigned DRAM offsets only to inputs, constants, and
the values returned from the function. A spill inserts a `dma_store` of an intermediate
buffer followed by a `dma_load` reload, and that spill store's result is a DRAM temporary
that is neither an input, a constant, nor a return value. With no offset assigned, the
`DenseMap` lookup returned the default 0, so every spill wrote to DRAM offset 0, clobbering
the model input, and the reload read the input back instead of the spilled buffer.

The scratchpad allocation lit test had checked only that spilling inserted the right number
of DMA instructions, never the numerics, so it did not catch this.

**Chosen fix.** Give every spill temporary its own DRAM region in the encoder. Scan the
block for `dma_store` ops whose result is not returned, assign each a fresh DRAM offset in
the unified `dramOffset` map, and lay them out after the constants and before the outputs.
The reload `dma_load` reads from the same map, so it now finds the spill region.

**Verification.** After the fix the objdump of a spilling program shows the spill store going
to a distinct DRAM offset (`dram[0x800]`, separate from the inputs at 0x0 and 0x400 and the
outputs), and every benchmark cell, spilling or not, is back to 2.98e-08 against onnxruntime.
Added an end to end pytest at a 140 KB budget to lock the fix in.
