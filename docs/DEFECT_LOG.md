<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Defect log

Every defect found in this project's own code gets an entry here, including a
defect found by a test that then passes. Entries are checked off with a dated
status marker and are never deleted, because this file is the audit trail and
it is the raw material the debug report is built from. A deleted entry is a
defect the report cannot talk about.

Each entry carries an identifier, the date it was found, how to reproduce it,
what was wrong, and how it was resolved. The reproduction is what makes an
entry worth having a year later.

This file starts at P0, in the v3 rebuild. It has no v1 entries because it did
not exist in v1; the v1 era defects are in `ENGINEERING_LOG.md`, which does go
back that far.

## Open

**D-0049**, the timing gap bound assumes the process had the CPU.
Reproduced under load, explained, and deliberately not fixed here, because
the fix is a change to a gate and a red at a gate is not answered by
widening it. The entry carries the reproduction and the proposed
precondition.

**D-0050**, the binary format cannot express a buffer written in pieces, so a
tiled program cannot be encoded. Escalated rather than fixed: the fix needs a
`Program::kVersion` bump, which P14's gate forbids by name and which the
format's own design claim says should never be needed. An owner level conflict
between three documents rather than a phase's call.

## Resolved

### D-0001 lit exits nonzero on a genuinely empty test suite

- **Found:** 2026-08-19, phase P0.
- **Status:** resolved 2026-08-19.
- **Reproduce:** configure the project with no `.mlir` file anywhere under
  `test/`, then run `ninja -C build check-npu`. llvm-lit prints `error: did not
  discover any tests for provided path(s)` and exits 2, so the build target
  fails.
- **What was wrong:** the build specification's P0 gate expects an empty suite
  to pass, and llvm-lit will not do that. Adding `--allow-empty-runs` does not
  help: `llvm/utils/lit/tests/selecting.py` in the pinned LLVM has a test
  asserting that the flag deliberately does not suppress this particular error.
  The flag covers a suite that discovered tests and then filtered all of them
  out, which is a different condition.
- **Resolution:** the suite carries one real test, `test/Smoke/
  npu-opt-roundtrip.mlir`, which round trips a two operation module through
  `npu-opt` and checks the result with FileCheck. That tests exactly what P0
  delivers, which is the build harness, so the gate is met by something that
  ran rather than by something that was skipped. `--allow-empty-runs` stays set
  because the filtered-to-empty case it does cover will come up once there are
  per directory suites.

### D-0002 the pytest PYTHONPATH wiring was configured in a key nothing reads

- **Found:** 2026-08-19, phase P0.
- **Status:** resolved 2026-08-19.
- **Reproduce:** put `env = ["PYTHONPATH=..."]` under
  `[tool.pytest.ini_options]` in `pyproject.toml` without installing
  `pytest-env`, then run `python -m pytest --collect-only`. pytest prints
  `PytestConfigWarning: Unknown config option: env` and carries on. Nothing
  sets PYTHONPATH.
- **What was wrong:** `env` belongs to the `pytest-env` plugin, which is not in
  this environment. pytest treats an unknown key as a warning rather than an
  error, so the wiring that Section 3.3 of the build specification calls not
  optional would have read as present in review and done nothing at run time.
  This is the exact failure the specification predicts, arriving one layer
  earlier than expected: not a missing path, but a path set somewhere nothing
  looks.
- **Resolution:** `test/Python/conftest.py` sets it instead, which needs no
  plugin. It resolves the path from `MLIR_PYTHON_PACKAGES_DIR` if that is set,
  then from this repository's own CMake cache, then from the default location,
  so pytest and lit agree on the path by construction rather than by two
  hardcoded strings staying in sync. `test/Python/test_bindings_wiring.py`
  imports `mlir.ir` and builds a module, so a future break in the wiring fails
  a test that names the wiring instead of failing whatever test ran first.
  Verified by running pytest with `PYTHONPATH` unset in the environment.

### D-0003 a clean install from the lock file could not resolve torch

- **Found:** 2026-08-19, phase P0.
- **Status:** resolved 2026-08-19.
- **Reproduce:** write `pip freeze` straight into `requirements-lock.txt`, then
  `python3 -m venv /tmp/check && /tmp/check/bin/pip install -r
  requirements-lock.txt`. pip reports `No matching distribution found for
  torch==2.13.0+cpu`.
- **What was wrong:** the installed torch is the CPU build and freezes as
  `2.13.0+cpu`. That local version exists only on the PyTorch CPU index and
  never on PyPI, so a lock file naming it with no index directive cannot be
  installed anywhere, which makes it a lock file that locks nothing.
- **Resolution:** `--extra-index-url https://download.pytorch.org/whl/cpu` is
  written into the lock file above the pins. The tempting alternative, dropping
  the `+cpu` suffix so PyPI resolves it, is worse than the broken state: it
  would silently install the CUDA build, change what reference generation runs
  on, and pull roughly two gigabytes of CUDA wheels onto a machine whose GPU
  this project does not use. Verified by installing into a clean venv and
  importing torch, onnx, onnxruntime and numpy there.

### D-0004 build-and-test failed inside the container because run steps fell back to sh

- **Found:** 2026-08-19, phase P0, by CI itself on the first run with the
  published image (run 32213043383).
- **Status:** resolved 2026-08-19.
- **Reproduce:** run the build-and-test job of ci.yml in the
  npu-mlir-llvm container without a shell default. The first step dies at
  `set: Illegal option -o pipefail` with exit code 2 before any real work.
- **What was wrong:** on a plain runner VM the default shell for run steps is
  bash, and every step in this job was written against bash. Inside a job
  container the runner instead invokes `sh -e {0}`, and sh on this image is
  dash, which does not implement `set -o pipefail`. The scripts were correct
  for the shell they assumed and never ran under it.
- **Resolution:** a job level `defaults.run.shell: bash` on build-and-test.
  bash is present in the image, it just has to be requested by name. The
  failing run is the before evidence and the next push run is the after
  evidence; both URLs are in the engineering log.

### D-0005 check-npu in CI asked for the build tree llvm-lit that an installed LLVM does not have

- **Found:** 2026-08-19, phase P0, by CI on the first run that got past D-0004
  (run 32213209291).
- **Status:** resolved 2026-08-19.
- **Reproduce:** configure this repository against an installed LLVM prefix
  (the /opt/llvm of the CI image) without LLVM_EXTERNAL_LIT and run
  `ninja -C build check-npu`. The lit target invokes
  `build/bin/llvm-lit`, which does not exist: that script is generated in an
  LLVM build tree and is not part of `ninja install`.
- **What was wrong:** the local build links against the LLVM build tree at
  `~/llvm-project/build`, which carries `llvm-lit`, so the default lit path
  works there and the omission was invisible. The CI image carries an install
  tree, where the default is wrong and the pip installed `lit` is the intended
  runner.
- **Resolution:** the CI configure step passes
  `-DLLVM_EXTERNAL_LIT="$(command -v lit)"`. The local configure stays as it
  is, because against a build tree the default is correct and pinning the pip
  lit there would add a dependency the local flow does not need.

### D-0006 the CI reachability guard named a path that never existed

- **Found:** 2026-08-19, phase P1, on switching the step on.
- **Status:** resolved 2026-08-19.
- **Reproduce:** check out the P0 tree and read the `check-reachability
  --skip-models` step of `.github/workflows/ci.yml`. Its guard is
  `[ -f include/npu/NPUOps.td ]`. No file has ever lived at that path: Section 6
  of the build specification puts the operation definitions at
  `include/NPU/Dialect/NPU/IR/NPUOps.td`, and the P0 scaffold created
  `include/NPU/` accordingly.
- **What was wrong:** the guard was written at P0 from the specification's prose
  rather than from its repository layout, and the two differ in case and in
  depth. Nothing caught it, because a guard that is never satisfied prints the
  "OFF until P1" line and the job stays green, which is indistinguishable from
  the intended behaviour right up until the phase that was supposed to switch it
  on. The step would have gone on reporting that it was waiting for P1 for the
  rest of the project.

  This is the exact failure mode the activation table exists to prevent, arriving
  through the table's own mechanism. A step that is off says so in the log, and
  this one did, truthfully and forever.
- **Resolution:** the guard now names the real path. It is also no longer a
  silent fallthrough: from P1 the step is on, so an unsatisfied guard exits 1
  with a message saying a file moved, rather than printing an off line for a
  phase that has already passed. The engineering log entry for 2026-08-19 notes
  the general lesson, which is that a guard whose condition can never be true is
  a check that has been deleted rather than deferred, and that the way to catch
  one is to prove it red on the day it activates.

### D-0007 a pooling shape diagnostic printed a rank where it meant an extent

- **Found:** 2026-08-19, phase P1, by reading the output of a verifier probe
  before writing the tests against it.
- **Status:** resolved 2026-08-19.
- **Reproduce:** at commit `a7fdbba`, run `npu-opt` over a `npu.max_pool2d`
  whose result spatial extents are wrong, for example an 8 by 8 input with a
  2 by 2 kernel and stride 2 declaring a 3 by 4 result. The diagnostic reads

      result spatial extents must be 4 by 4, computed from the input 2
      dimensional window arithmetic, but got 3 by 4

  The phrase "the input 2 dimensional window arithmetic" is not a description of
  anything. The 2 is `actual.size()`, the rank of the spatial extent vector,
  printed into a sentence that reads as though it were describing the input.
- **What was wrong:** a stray value in a format string. Nothing computed wrongly
  and no shape was accepted or refused incorrectly; the arithmetic behind the
  message was right, and only the message was wrong.

  It goes in this log anyway, and the reason is the standard this project holds
  diagnostics to. Every failure message names the operation and quotes the
  offending numbers, and a message that quotes a number which is not one of the
  offending numbers is worse than one that quotes none: a reader who tries to
  reconcile "2 dimensional" against the shapes in front of them is being sent to
  look for a problem that is not there.
- **Resolution:** the message now quotes the input extents, the kernel, the
  strides, the dilations, the pads and the ceil mode, which together are every
  input to the computation whose answer it is reporting. The two lit cases in
  `test/Dialect/NPU/invalid.mlir` that assert on this message quote it in full,
  so a future edit that drops a term fails a test rather than degrading quietly.

### D-0008 a null memory space crashed the operand type predicate

- **Found:** 2026-08-19, phase P2, by running the inherited and uncommitted
  `test/Dialect/NPUISA/invalid.mlir` for the first time.
- **Status:** resolved 2026-08-19.
- **Reproduce:** at commit `00cce3b`, write a memref with no memory space into a
  position the dialect constrains to one of the two named spaces:

      func.func @f(%src: memref<4x4xf32, #npu.dram>, %dst: memref<4x4xf32>) {
        npuisa.dma_load %src, %dst
          : memref<4x4xf32, #npu.dram> to memref<4x4xf32>
        return
      }

  `npu-opt` on that file dies with a segmentation fault inside
  `DmaLoadOp::verifyInvariants` and prints no diagnostic at all. Under lit it
  showed up as the whole `invalid.mlir` file failing, because the crash takes the
  process down partway through a `-split-input-file` run and the remaining
  sections never execute.
- **What was wrong:** the memory space predicate in `NPUISATypes.td` was written
  as

      ::llvm::isa<::mlir::npu::ScratchpadAttr>(
          ::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace())

  A memref written without a memory space has a **null** memory space attribute.
  `llvm::isa` on a null `Attribute` is undefined behaviour: it dereferences the
  attribute to reach its type id. In a build with assertions that trips one and
  names the problem. This project's default configuration sets no
  `CMAKE_BUILD_TYPE` and therefore builds without `NDEBUG` handling either way,
  and what happened here was a read through a null pointer.

  The general lesson, which is the reason this is in the log rather than fixed
  silently: `isa` is not a total function on MLIR's attribute and type handles.
  It is total on a *non null* one. Any predicate that reads an optional part of a
  type, and a memory space is optional by construction, has to use
  `isa_and_present`. The whole family of `NPUISA_MemRefInSpace` predicates had
  the same bug, generated once per space and per element type set, so a fix that
  reached only the one case the test happened to exercise would have left the
  others crashing.
- **Resolution:** every predicate now uses `isa_and_present`, which answers false
  for null and lets the ordinary operand type diagnostic report the wrong space.
  Five regression cases in `test/Dialect/NPUISA/invalid.mlir` cover it from both
  spaces and from three operation shapes: `@dma_load_into_the_default_space`,
  `@dma_store_from_the_default_space`, `@relu_in_the_default_space`,
  `@const_in_the_default_space`, and `@dma_load_with_a_dynamic_extent` for the
  static shape half of the same predicate. All five failed with a segmentation
  fault before the fix and produce their expected diagnostic after it.

### D-0009 the overlap scan made two transfers in flight unrepresentable

- **Found:** 2026-08-19, phase P2, by writing the canonicalization test for the
  case where the fold must *not* fire.
- **Status:** resolved 2026-08-19.
- **Reproduce:** at commit `00cce3b`, write the double buffering shape, two
  asynchronous loads to disjoint destinations with both awaits after both
  producers:

      %t1 = npuisa.dma_load_async %src1, %dst1 : ...
      %t2 = npuisa.dma_load_async %src2, %dst2 : ...
      npuisa.await %t1
      npuisa.await %t2

  `npu-opt` rejects it:

      'npuisa.dma_load_async' op the operation npuisa.await lies between this
      asynchronous transfer and its npuisa.await and does not implement
      MemoryEffectOpInterface, so it cannot be shown not to touch the destination
      buffer

  The destinations are two distinct function arguments and provably disjoint. The
  program is correct and the verifier refuses it.
- **What was wrong:** the intervening operation scan of rule 4 treats an
  operation that does not implement `MemoryEffectOpInterface` as a possible
  conflict, which is right in general: an operation that cannot say what it
  touches has not said it touches nothing. But `npuisa.await` declares no memory
  effect *by design*, per Section 8, because what it does is order an effect the
  asynchronous operation already declared. So the conservative branch caught the
  one operation that is guaranteed harmless, and it caught it in exactly the
  configuration asynchronous DMA exists for.

  That last part is what makes this worth a log entry rather than a one line fix.
  The rule was tested only in the shape where a compute instruction sits between
  a transfer and its await. Nothing exercised two transfers outstanding at once,
  and two transfers outstanding at once is not an edge case: it is the double
  buffering of Section 5.1, the whole reason the asynchronous form is in the
  dialect. A verifier can be wrong about its own reason for existing and still
  pass every test somebody thought to write, and the way this one surfaced was a
  test written for a different rule entirely.
- **Resolution:** the scan skips an intervening `npuisa.await` by name, with the
  soundness argument written at the skip: an await touches no memory, and the
  bytes the transfer it waits for implies are already accounted for by that
  transfer's own producer, which the same scan visits and checks. Counting them
  twice would reject correct programs without catching a single incorrect one.

  Both directions are pinned. `@two_transfers_in_flight` in
  `test/Dialect/NPUISA/canonicalize.mlir` is the legal shape and asserts that
  neither transfer folds; `@two_transfers_racing` in
  `test/Dialect/NPUISA/invalid.mlir` is two transfers whose destinations are
  views overlapping by 32 bytes over one flat buffer, and it is still rejected,
  caught through the *other* asynchronous operation's declared write rather than
  through the await. `NPUISAInterfaceTest.TheAwaitDeclaresNoEffectOfItsOwn`
  asserts that the await still declares no effects, so a later change that gives
  it one fails a test rather than silently making the skip unsound.

### D-0010 the bindings image rebuild died at configure for want of python3-dev

- **Found:** 2026-08-19, phase P2, by the llvm-image workflow on the first
  rebuild with MLIR_ENABLE_BINDINGS_PYTHON=ON (run 32222527819).
- **Status:** resolved 2026-08-19.
- **Reproduce:** build docker/Dockerfile.llvm at the P2 revision that turns the
  bindings on. CMake fails inside MLIRDetectPythonEnv at FindPython3, and the
  configure stops before a single source file compiles.
- **What was wrong:** the builder stage installed python3, python3-pip and
  python3-venv but not python3-dev. The bindings compile nanobind modules
  against the interpreter's development headers, and FindPython3's Development
  component refuses an interpreter that has none. The P0 image never noticed
  because the bindings were off, so the interpreter-only install was
  sufficient right up until the flag flipped.
- **Resolution:** python3-dev added to the builder stage's package list. The
  final stage is unchanged: it ships the built bindings, which need only the
  interpreter and numpy at run time, not the headers.

### D-0011 the coverage job flaked red on a gcov negative branch counter

- **Found:** 2026-08-19, phase P2, by the coverage job on a docs only commit
  (run 32290939959), two runs after the same code passed the same job.
- **Status:** resolved 2026-08-19.
- **Reproduce:** not deterministically. gcov's branch counters can go negative
  under counter merging (gcc bug 68080); when one does, the text report says
  "branch 2 taken -1" and gcovr's strict parser raises NegativeHits and exits
  64. The trigger run hit it in the report for NPUISAOps.cpp.
- **What was wrong:** nothing in this repository's code, which is the point of
  recording it: the failing commit touched two markdown files. The defect is in
  the gcov tool, and the coverage script's strictness turned a known upstream
  artifact into a job failure.
- **Resolution:** gcovr is invoked with
  --gcov-ignore-parse-errors=negative_hits.warn_once_per_file, the remedy
  gcovr's own error message names. The warn form keeps the artifact visible in
  the log; the threshold arm and rule 2 of Section 17.7 (no percentage from a
  failing suite) are unaffected.

### D-0012 the dialect's no broadcast rule made Section 11's carve out unrepresentable

- **Found:** 2026-08-19, phase P3, while writing the importer's broadcasting
  helper against Section 11.
- **Status:** resolved 2026-08-19.
- **Reproduce:** on `main` at `316f3b8`, ask `npu-opt` to verify the IR that
  Section 11's carve out and Section 15's ResNet block both describe:

  ```mlir
  func.func @f(%a: tensor<2x8x4x4xf32>, %s: tensor<8xf32>,
               %d: tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32> {
    %0 = npu.mul ins(%a, %s : tensor<2x8x4x4xf32>, tensor<8xf32>)
                 outs(%d : tensor<2x8x4x4xf32>) -> tensor<2x8x4x4xf32>
    return %0 : tensor<2x8x4x4xf32>
  }
  ```

  It is rejected with `'npu.mul' op does not broadcast, so the rhs shape must
  equal the result shape`.
- **What was wrong:** two layers of the specification disagreed and the
  disagreement sat in the tree for two phases. Section 11 keeps a rank 1 channel
  shaped initializer unexpanded because `-npu-fuse-bias` guards on a channel
  shaped constant addend, and Section 15 puts the same carve out on a per
  channel `Mul` in the ResNet block. P1's `NPUOps.td` instead required both
  operands of `npu.add` and `npu.mul` to have the result shape exactly, and
  recorded that the carve out "is expressed as a bias operand on the consuming
  convolution". That reading fails twice. Folding a `Conv` plus rank 1 `Add`
  into the convolution's bias at import leaves `-npu-fuse-bias` nothing to fuse,
  which is exactly the failure the carve out exists to prevent; and a per
  channel scale has no bias operand anywhere to be folded into, so the rule had
  no answer for `Mul` except expansion, which Section 11 forbids in the same
  paragraph.

  Nothing was silently wrong at runtime, because nothing had run: there was no
  importer to emit the IR and no pass to consume it. What was wrong is that the
  earliest layer had closed off a shape two later layers require, and it would
  have surfaced at P6 as a fusion pass with a zero ablation row rather than as
  an error anybody could act on.
- **Resolution:** `npu.add` and `npu.mul` now accept a rank 1 rhs whose length
  equals the result's channel extent under its layout, against a rank 4 result,
  and refuse everything else. Only the rhs may be rank 1, which preserves P1's
  actual concern that one fact should not have two representations. The
  regression coverage is `@add_channel_broadcast`,
  `@add_channel_broadcast_nhwc` and `@mul_channel_broadcast` in
  `test/Dialect/NPU/ops.mlir`, which the previous verifier rejects and this one
  accepts, plus five negative cases in `test/Dialect/NPU/invalid.mlir` that
  bound the relaxation. Recorded as
  `docs/adr/0005-channel-broadcast-on-add-and-mul.md`.

### D-0013 a failed import left MLIR's insertion point stack unwound, and the process crashed at exit

- **Found:** 2026-08-19, phase P3, by the first full run of the new pytest
  suite.
- **Status:** resolved 2026-08-19.
- **Reproduce:** delete the `self._insertion.close()` line from
  `ModuleBuilder.__exit__` in `python/npu_frontend/builder.py`, then run
  `python -m pytest test/Python/test_onnx_importer.py -q`. The suite reports
  `74 passed` and the process then exits **139**, a segmentation fault, with no
  traceback and nothing naming a test.
- **What was wrong:** `ModuleBuilder` entered MLIR's `InsertionPoint` context in
  `begin_function` and left it in `end_function`, which is the happy path only.
  Every converter that raises, and about a third of this suite's tests exist to
  make one raise, unwound out of the builder without ever reaching
  `end_function`, so the insertion point stayed on MLIR's thread local stack
  pointing into a module that was then freed. Nothing failed at the time. The
  crash came at interpreter shutdown, long after the test that caused it had
  reported a clean expected failure, and the exit code was the only symptom.

  The shape of this is worse than the bug. A test suite that reports every test
  passing and then segfaults is one whose exit code is the only thing that
  disagrees, and a runner that checked only the summary line would have called
  it green.
- **Resolution:** the builder holds two `contextlib.ExitStack`s, an outer one
  for the context and the location and an inner one for the function's insertion
  point. `end_function` closes the inner one on the happy path and `__exit__`
  closes both unconditionally, which works because `ExitStack.close` is
  idempotent. Verified both ways: patched back to the old teardown the suite
  exits 139, and restored it exits 0 on the same 74 tests.

### D-0014 the broadcast carve out matched a rank 1 initializer, which ONNX broadcasts over the width

- **Found:** 2026-08-19, phase P3, while writing the fixtures for the
  broadcasting tests.
- **Status:** resolved 2026-08-19.
- **Reproduce:** put `(channels,)` back into the accepted set in
  `_channel_broadcast_length` in `python/npu_frontend/op_mapping.py`, then run
  `python -m pytest test/Python/test_onnx_importer.py -q -k literally_rank_one`.
  The test fails: an `Add` of a `1 x 4 x 3 x 4` activation and an initializer of
  dims `[4]` imports to `npu.add` with a rank 1 operand, where the operand is a
  per column vector and the rank 1 form means per channel.
- **What was wrong:** Section 11 describes the carve out as "a rank 1
  initializer of length C broadcasting against a rank 4 activation over the
  channel axis", and I implemented the first clause without checking that it
  implies the second. It does not. ONNX broadcasting aligns from the **trailing**
  axis, so an initializer of dims `[C]` against an `N x C x H x W` activation
  broadcasts over the width, not over the channels. The shapes that broadcast
  over the channel axis are `[C, 1, 1]` and `[1, C, 1, 1]`, and `[1, C, 1, 1]`
  is what the exporter actually writes.

  The consequence is a wrong answer that typechecks. On any model where the
  channel count and the width are equal, a per column constant would have been
  imported as a per channel one, and the emitted IR is legal, verifies, and
  computes something the model did not ask for. It would have been invisible
  until an end to end comparison against onnxruntime at P8, and on a model where
  the two extents differ it would have raised a shape error somewhere unrelated
  instead.
- **Resolution:** the accepted set is `{(C, 1, 1), (1, C, 1, 1)}` and a rank 1
  initializer is expanded like any other broadcast.
  `test_a_literally_rank_one_initializer_is_a_width_broadcast_not_a_channel_one`
  is the regression test, and it is deliberately written on a `1 x 4 x 3 x 4`
  activation, where the channel count and the width are both 4, so the two
  readings are distinguishable only by knowing the rule. It fails on the
  previous implementation and passes on this one.

### D-0015 a dynamic extent reached npu.reshape, which aborted instead of diagnosing

- **Found:** 2026-08-20, phase P4, while writing the lowering's refusal cases.
  The test that found it was asking a different question: it wanted an operation
  whose result type the lowering could not assign a memory space to, and a
  reshape of a `tensor<?x4xf32>` was the obvious way to build one.
- **Status:** resolved 2026-08-20.
- **Reproduce:** put `RankedTensorOf` back in place of `StaticShapeTensorOf` in
  `include/NPU/Dialect/NPU/IR/NPUTypes.td`, rebuild, and run

  ```
  npu-opt /dev/stdin <<< 'func.func @f(%x: tensor<?x4xf32>) -> tensor<4x4xf32> {
    %0 = npu.reshape %x : tensor<?x4xf32> to tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }'
  ```

  The tool aborts inside `ReshapeOp::verify` with
  `Assertion 'hasStaticShape() && "cannot get element count of dynamic shaped
  type"' failed`, an LLVM stack trace, and no diagnostic. In a build without
  assertions the same call returns a product over a dynamic extent instead.
- **What was wrong:** `NPUTypes.td` said, in a comment carried since P1, that
  "a dynamic dimension is refused at the type level rather than by a verifier,
  because nothing below this dialect can represent one". The three constraints
  under that comment were written with `RankedTensorOf`, which requires a static
  **rank** and says nothing about the extents. So the comment described a rule
  nothing enforced, and `ReshapeOp::verify` then read the element count of such a
  tensor through `getNumElements()`, which asserts rather than answering.

  The failure mode is the specific one every verifier in this dialect is written
  to avoid. It is not a wrong answer, which a test would catch, and it is not a
  diagnostic, which a reader could act on. It is an abort with a stack trace
  pointing into LLVM, from IR that parsed.

  It was never reachable from the frontend, which refuses a dynamic extent at
  import and has a pytest saying so. It was reachable from any hand written
  `.mlir`, which is most of this project's test surface.
- **Resolution:** the three constraints are `StaticShapeTensorOf` now, which is
  what the comment always claimed and what the `npuisa` side has done since P2,
  where every memref constraint ANDs `HasStaticShapePred` for the same reason.
  The two levels now say the same thing in the same place.

  Three regression cases in `test/Dialect/NPU/invalid.mlir`:
  `relu_with_a_dynamic_extent` proves the constraint fires on an ordinary
  compute operation, and `reshape_from_a_dynamic_extent` and
  `reshape_to_a_dynamic_extent` prove it fires on the operation that aborted,
  from each side of it, since a reshape can take its dynamic extent on either
  the operand or the result. All three abort against the old spelling and are
  refused with `operand #0 must be statically shaped tensor of ...` against this
  one.

  `docs/DIALECT_REFERENCE.md` is regenerated in the same commit, because the
  constraint description is part of it: 46 rows change from "ranked tensor of"
  to "statically shaped tensor of".

### D-0016 the batch norm decomposition left four constants behind, and each became a DRAM transfer

- **Found:** 2026-08-20, phase P4, by reading the lowering's own output on a
  batch norm rather than by a failing test. Nothing was failing: the program was
  correct and every test passed.
- **Status:** resolved 2026-08-20.
- **Reproduce:** delete the loop in `expand()` in
  `lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp` that erases an
  `npu.constant` with no uses, rebuild, and run
  `npu-opt test/Dialect/NPUISA/lowering.mlir --npu-lower-to-npuisa`. The
  function `an_unfolded_batch_norm_loads_two_constants_not_four` comes out with
  **seven** `npuisa.dma_load` operations instead of three: the argument, the four
  batch norm parameters, and the two constants the decomposition computed. Its
  `CHECK-COUNT-3` then fails.
- **What was wrong:** the decomposition consumes gamma, beta, mean and variance
  at rewrite time, computing the multiplier and the addend from their values, and
  it left the four `npu.constant` operations in the block with no uses. The
  conversion that follows does not care whether a constant is used: it is an
  illegal operation, so the pattern fires, and out comes an `npuisa.const` in
  DRAM and an `npuisa.dma_load` bringing it on chip.

  Nothing downstream would have removed them either, and that is the part worth
  recording. A transfer has memory effects, by design and for good reasons set
  out in this project's own architecture notes, so it is not dead code that a
  canonicalizer is entitled to delete. The four transfers would have survived to
  the encoder and been executed by the simulator.

  The consequence is not a wrong answer. It is four transfers per unfolded batch
  norm of data no instruction reads, which is DRAM traffic, which moves the byte
  counts and the energy numbers this project publishes. Section 8 is explicit
  that unexplained DRAM traffic is a defect rather than a style question, and it
  names exactly three permitted producers of DMA; this was a fourth, hiding
  inside the first.
- **Resolution:** `expand()` erases every `npu.constant` whose result has no uses
  before the conversion sees it. One pass suffices, because a constant has no
  operands and erasing one therefore cannot make another dead.

  `an_unfolded_batch_norm_loads_two_constants_not_four` in
  `test/Dialect/NPUISA/lowering.mlir` is the regression test. It asserts three
  loads and then `CHECK-NOT: npuisa.dma_load`, so it fails at seven and passes at
  three, and it is named after the claim rather than after the mechanism so that
  a later reader knows what it is defending.

### D-0017 an int64_t pass option printed into a diagnostic as a character

- **Found:** 2026-08-20, phase P5.
- **Status:** resolved 2026-08-20.
- **Reproduce:** on the parent of the fix, run

  ```
  npu-opt test/Dialect/NPUISA/scratchpad-alloc.mlir \
      --npu-allocate-scratchpad=alignment=48
  ```

  The diagnostic reads `the alignment must be a positive power of two, but it
  is 0`. The alignment given was 48. At 63 it reads `?`, at 96 a backtick, at
  100 the letter `d`. Every one of them is the ASCII character whose code is the
  number.
- **What was wrong:** an ODS pass option of C++ type `int64_t` is generated as
  an `llvm::cl::opt<int64_t>`, which converts to its data type through a user
  defined conversion operator. Streaming one straight into an
  `mlir::InFlightDiagnostic` picks the `char` overload of `Diagnostic::operator
  <<` rather than the integer one, and the value is printed as a character.

  Nothing about the allocation was wrong: the check itself read the right
  number and refused the right values. What was wrong was the message, which is
  the whole of what that code path exists to produce. A diagnostic that
  misreports the number it was given is worse than no diagnostic, because
  somebody acts on it: told the alignment is 0 when they passed 48, the obvious
  conclusion is that the option was not read at all, and the next thing they do
  is go looking in the pass manager rather than at their own value.

  It is worth naming the shape of this rather than only the instance. Every
  numeric pass option in this project is one `<<` away from the same bug, and
  nothing about the code looks wrong at the point of use.
- **Resolution:** the option is copied into a plain `const int64_t` before it is
  used or printed, in both `readOptions` and `readBudget`, with a comment saying
  why so that a later reader does not simplify it back. The regression test is
  `test/Dialect/NPUISA/alloc-unknown-option.mlir`, which asserts the exact text
  `but it is 48` under `-verify-diagnostics`. It fails against the character
  form and passes against the fix.
### D-0018 a subview's byte range was measured by the elements it holds, not the bytes it reaches

- **Found:** 2026-08-20, phase P5, while adding the `memref.reinterpret_cast`
  case that P4's handoff left open.
- **Status:** resolved 2026-08-20.
- **Reproduce:** take two `2 by 2` subviews of one `4 by 4` `f32` buffer, the
  first at `[0, 0]` and the second at `[1, 0]`. They share the whole of row 1,
  elements 4 and 5. On the parent of the fix, `mlir::npuisa::overlaps` on the
  two returns `Disjoint`. `unittests/Dialect/NPUISA/InterfaceTest.cpp`, test
  `OverlappingSubviewsAreNotReportedDisjoint`, is that case and it fails there.
- **What was wrong:** `byteSize` computed a memref's size as the product of its
  extents times its element width. For a contiguous buffer that is right. For a
  **subview** it is the number of elements the view holds, which is not the
  number of bytes it reaches across: a sub block of a larger buffer skips the
  remainder of every row, so its first and last elements are further apart than
  its element count suggests. The first subview above holds 4 elements and spans
  6; the analysis said 16 bytes where the truth is 24.

  The direction of the error is what makes it a defect rather than an
  imprecision. A range that is too **small** can be reported disjoint from a
  range it genuinely intersects, and the only consumer of this analysis is the
  rule that decides whether an operation between an asynchronous transfer and
  its `await` races with the destination. An under reported range there permits
  a real race, silently. `Unknown` would have been safe; a confident wrong
  answer is not.

  It had never fired because no pass in this project emits a `memref.subview`
  yet. The tiling pass of Section 13.2 is the one that will, at P13, and it
  would have inherited a memory model that quietly permitted the race it is most
  likely to create.
- **Resolution:** the size is computed from the strides rather than the extents,
  by `byteSpan`: `1 + sum((extent - 1) * stride)` elements when the type carries
  a layout map, and the product of the extents when it does not, which are the
  same number for a contiguous buffer. The same change is what makes a stride 0
  broadcast view span the C floats it addresses rather than the tensor it is
  shaped like, and `docs/ARCHITECTURE.md` records both halves as a marked P5
  extension.

  Three tests in `InterfaceTest.cpp` fail before the change and pass after:
  `AStrideZeroBroadcastSpansTheBufferItIsOver`, `TwoAdjacentBroadcastsAreDisjoint`
  and `OverlappingSubviewsAreNotReportedDisjoint`. Shown red by reverting
  `byteSpan` to the extent product and rerunning, then shown green again.

### D-0019 the npuisa windowed verifier reordered its pads and rejected every asymmetric one

- **Found:** 2026-08-20, phase P5, by running the model suite through the
  allocator to measure the fragmentation ratio Section 13.1 asks for.
- **Status:** resolved 2026-08-20.
- **Reproduce:** compile `dilated_stack`, which is the one model in the suite
  with asymmetric padding:

  ```
  python experiments/allocator_fragmentation.py --models dilated_stack
  ```

  On the parent of the fix the lowering's output does not verify:

  ```
  'npuisa.conv2d' op the destination width extent must be 5, the extent this
  window implies, but the destination is 'memref<1x5x3x6xf32, #npu.scratchpad>'
  ```

  The window is input 11 by 13, kernel 3 by 3, strides 2 by 2, dilations 3 by 2,
  pads `[0, 1, 0, 1]`. The width is `(13 + 1 + 1 - 5) / 2 + 1`, which is 6, and
  the lowering produced 6. The `npuisa` verifier computed 5.
- **What was wrong:** `computeWindowedShape` takes its pads in ONNX order, all
  begins then all ends, and reads `pads[axis]` and `pads[rank + axis]` itself.
  The `npu` verifier one level up passes them straight through, which is right.
  The `npuisa` verifier **reordered them into per axis pairs** before the call,
  `{pads[0], pads[2], pads[1], pads[3]}`, so the helper then read the first
  entry of each pair as a begin and the pair after it as the ends. Height was
  given `(padTop, padBottom)` interpreted as entries 0 and 2 of an ONNX array,
  which they are not.

  The code sat under a comment reading "Getting this reordering wrong is the
  single easiest mistake in the whole file and it is why it is written once here
  instead of at each call site". The comment was correct about the risk and the
  line under it was the mistake, which is worth recording exactly that way: a
  comment asserting that a thing is careful is not evidence that it is.

  It survived from P2 to P5 because **every pad in every test was symmetric**.
  Under a symmetric pad both orders produce the same numbers, so a test suite
  built from `pads = [1, 1, 1, 1]` and `pads = [0, 0, 0, 0]` cannot distinguish
  them. The suite model that carries asymmetric padding is `dilated_stack`, and
  Section 15 put it there precisely because it forces a case nothing else
  reaches; it did that, three phases after it was written, and the case it
  forced was in a verifier rather than in the importer.

  The consequence was a whole model that could not be lowered. It is not a
  numerics defect: the verifier refused, loudly, rather than accepting a wrong
  shape. That is the failure mode the shared helper was designed for, working
  in the direction of safety while still being wrong.
- **Resolution:** the pads are passed straight through, exactly as `NPUOps.cpp`
  does. `conv2d_asymmetric_pads` and `pool_max_asymmetric_pads` in
  `test/Dialect/NPUISA/ops.mlir` are the regression tests: a 4 by 8 input with
  `pads = [0, 1, 2, 0]`, whose correct destination is 4 by 7 and which the
  reordered form demands be 3 by 8. Both fail against the reordering and pass
  against the fix, shown both ways. The pooling case is there because the two
  operations share the verifier body and a fix that missed one would be
  invisible otherwise.

### D-0020 a reshape could lose elements and validate

- **Found:** 2026-08-20, phase P6.
- **Status:** resolved 2026-08-20.
- **Reproduce:** build a `Program` holding a `DMA_LOAD` of a 4 by 4 f32 buffer,
  then a `RESHAPE` reading that buffer and declaring a result shape of `{4}`,
  then a `DMA_STORE` of four elements, with an output region of four elements
  so that nothing downstream is out of range. Call `validate()`. Before the
  fix it returned nothing: sixteen elements went in, four came out, and the
  file was accepted. `Validation.ResultShapeCatchesAReshapeThatLosesElements`
  in `unittests/Encoding/ValidationTest.cpp` is that program.
- **What was wrong:** `docs/ISA_MANUAL.md` and the `shapeRule` field of the
  `RESHAPE` record in `include/NPU/Encoding/NPUISADescription.td` both state
  that a reshape's element counts agree in and out. Nothing enforced it.
  `Program::validate()` dispatched `RESHAPE` to the case that means "nothing
  semantic beyond the generated rules", which is true of `RELU` and was not
  true here.

  It is a documentation defect and a validation defect at the same time, and
  that pairing is what makes it worth an entry: the generated manual said a
  rule existed, so a reader had every reason to believe it did. Section 9.4's
  whole argument is that a description which cannot lie is better than four
  hand maintained places that agree by discipline, and this is the residue of
  that argument. The description carries the rule as prose in a `shapeRule`
  string, prose is not generated into code, and the gap between the two is
  exactly where this sat.

  The consequence was bounded rather than unbounded, which is worth being
  precise about. A short reshape does not read or write out of range: the
  operand extent check bounds what it reads and the result range check bounds
  what it writes. What it does is copy the wrong number of elements and leave
  the rest of the destination holding whatever was there, which is a silent
  wrong answer at P7 rather than a trap.
- **Why the first regression test proved nothing, which is the more useful
  half of this entry.** The case was found while writing the corpus of Section
  17.3, as `reshape that loses elements`, built by changing the opcode and the
  result shape of the base program's relu and nothing else. That case is
  rejected with or without the fix, because leaving the rest of the program
  alone leaves the following `DMA_STORE` reading sixty four bytes out of the
  sixteen the reshape wrote, and `operand-extent` refuses it one instruction
  later. The corpus test passed before the fix and after it.

  A regression test that passes before the fix is not a regression test. The
  one that counts had to be built so that the reshape's element count is the
  **only** rule the program breaks: the store reads exactly what the reshape
  wrote, and the output region is exactly what the store writes. It fails
  before and passes after, shown both ways.
- **Resolution:** `checkReshape` in `lib/Encoding/Validation.cpp`, dispatched
  from the per opcode semantic switch, reporting `result-shape`. That name
  rather than a new one because Section 9.2's list is what it is and inventing
  a name would be inventing a rule the specification does not have; and
  `result-shape` is what is wrong, since the operand is whatever it is and the
  result shape is the one field a reshape gets to choose. The check's
  description in the ISA description was extended to say so, which regenerates
  into the manual.

### D-0021 the validator's own range messages overflowed a signed integer

- **Found:** 2026-08-20, phase P6, by UndefinedBehaviorSanitizer on the first
  real run of the sanitizers job's build.
- **Status:** resolved 2026-08-20.
- **Reproduce:** build with clang and
  `-fsanitize=address,undefined -fno-omit-frame-pointer`, then run
  `NPUEncodingTests`. The corpus case `result address at the signed limit` sets
  a result address of `INT64_MAX` in a program whose result is sixty four
  bytes. `Program::validate()` rejects it, correctly, and then builds the
  message:

  ```
  lib/Encoding/Validation.cpp:734:65: runtime error: signed integer overflow:
  9223372036854775807 + 64 cannot be represented in type 'int64_t'
  ```

- **What was wrong:** four diagnostics said "the result runs from A to A plus
  N", computing the end of a range that the check immediately above had just
  established does not fit in memory. The addition is exactly the arithmetic
  the range check exists to refuse, performed one line after refusing it.

  A fifth site was not a message. `operand.address + span > *end` in the
  operand extent check is a **comparison**, and it overflows for the same
  reason: a file is free to declare a memory of nearly 2^64 bytes, so
  `fitsInMemory` can pass an address near the signed limit through to a
  comparison that adds a span to it.

  **The consequence is small in practice and the reason it is worth an entry is
  not the consequence.** On the machines this runs on the addition wraps, the
  message prints a negative number, and the file is still rejected by the check
  that already decided. What signed overflow actually is, though, is undefined
  behaviour, and a compiler is entitled to assume it cannot happen. The
  assumption it would draw from the comparison site is that
  `operand.address + span` does not overflow, which is to say that
  `operand.address` is small, which is a fact the compiler could then use to
  simplify the range check above it. That is how a bounds check disappears, and
  it is the shape of thing this project's whole validation layer exists to
  prevent.

  It is also a direct hit for the sanitizers job, which Section 19.0 activates
  at this phase. The job found a real defect in the code it was switched on to
  watch, on its first real run, in a path no hand written test reaches: nothing
  in `ValidationTest.cpp` uses an address near the signed limit, because a
  human writing a test picks a number like 4096.
- **Resolution:** the four messages say "runs from A for N bytes" instead of
  naming the end, which removes the arithmetic from the message entirely. The
  comparison is `span > *end - operand.address`, which cannot overflow: the end
  comes from a written span that contains the address, so the difference is at
  least one. The regression test is the corpus case above, which UBSan already
  fails against the old code and passes against the new, and the whole corpus
  now runs clean under both sanitizers.

### D-0022 the disassembler multiplied an unvalidated shape without a guard

- **Found:** 2026-08-20, phase P6, by `fuzz/nbin_decode_fuzzer` under
  UndefinedBehaviorSanitizer, on a run seeded from a corpus a previous run had
  grown.
- **Status:** resolved 2026-08-20.
- **Reproduce:** `fuzz/corpus/regression_d0022_stride_overflow.nbin`, the
  minimized crash input, 2580 bytes. Or directly:
  `Disassembly.AStrideProductThatOverflowsDoesNotUndefineTheListing` in
  `unittests/Encoding/EncodingTest.cpp`, which builds an operand with shape
  `{8935141660703064067, 3}` and strides `{3, 1}`.

  ```
  lib/Encoding/Disassembler.cpp:98:16: runtime error: signed integer overflow:
  3 * 8935141660703064067 cannot be represented in type 'long'
  ```

- **What was wrong:** `operandText` decides whether to print an operand's
  strides by walking them against the contiguous layout its shape implies,
  accumulating `expected *= operand.shape[index]`. Every other multiplication
  of an extent in this subsystem is guarded, because Section 9.2's first rule
  says to test before multiplying. This one was not, and the reason it was not
  is worth naming: the rest of the disassembler's inputs come from a program
  that validated, and it is easy to forget that this particular function does
  not.

  **`npu-objdump` decodes without validating.** That is the single reason
  `decodeUnvalidated` exists. So every extent this function sees is whatever
  the file claimed, and a claimed extent of nine quintillion is not an unlikely
  input, it is the ordinary case for the tool's whole purpose.

  The strides needed to be exactly the contiguous ones for the walk to reach
  the overflowing extent rather than break early, which is why no hand written
  test found it: a person writing a malformed operand writes a *wrong* stride
  vector, and this one is right.
- **Resolution:** the same overflow safe form the validator uses, testing
  `expected > Program::kShapeLimit / extent` before multiplying, with a
  non positive extent treated as not contiguous. The regression test is the
  named case above, plus `contiguous strides over an overflowing shape` in the
  corpus, which reaches it through the same unvalidated path the tool takes.
  The minimized input is committed as a seed, per Section 17.3.

### D-0023 an instruction with a missing operand disassembled to a blank line

- **Found:** 2026-08-20, phase P6, while reading the listing of the minimized
  crash input from D-0022.
- **Status:** resolved 2026-08-20.
- **Reproduce:**
  `Disassembly.AMissingMandatoryOperandStillPrintsTheInstruction` in
  `unittests/Encoding/EncodingTest.cpp`: a `DMA_LOAD` with no operands at all,
  which is what a corrupt file carries and what the `arity` check refuses.
  Before the fix its listing line was `0000  ` and nothing else.
- **What was wrong:** the format string renderer drops a brace group whose
  operand reference is absent, which is how an optional bias disappears without
  leaving an empty slot. `renderTokens` implemented that by building into a
  local string and returning false **before** appending anything, so the same
  early return at the **top level** discarded everything already rendered for
  that instruction, the opcode mnemonic included.

  The consequence is small and lands in the worst possible place. `npu-objdump`
  exists so that a file which does not validate can still be looked at, and the
  instruction somebody is looking for is the broken one. Printing a blank line
  for exactly that instruction is the tool failing at its only job, silently.

  It is also a good argument for reading a tool's output rather than only
  asserting on it. Every disassembly test in this phase was written against
  programs whose operands are present, because a test author writes the case
  they are thinking about. This turned up in three seconds of looking at a real
  listing.
- **Resolution:** `renderTokens` takes a `dropOnMissing` flag. Group contents
  keep the drop behaviour; the top level substitutes `<missing operand N>` and
  prints everything else. The regression test asserts both that the mnemonic
  appears and that the line is not blank.

### D-0024 a CHECK-NOT below the listing it was meant to guard

- **Found:** 2026-08-20, phase P6, while measuring how far the product side
  activation fault reached.
- **Status:** resolved 2026-08-20.
- **Reproduce:** swap the two adjacent `i32` writes in `Program::encode` so that
  `requantShift` is written before `requantMultiplier`, rebuild, and run
  `ninja -C build check-npu`. Before the fix the suite reported 18 of 18
  passing, against an encoder that produced a file failing its own validator.
- **What was wrong:** `test/Encoding/objdump.mlir` carried
  `// DUMP-NOT: WARNING` **after** the last `DUMP-NEXT`. A `CHECK-NOT` only
  covers the span between the directives around it, so that one covered the
  tail of the listing and nothing else, and the warning block it was written to
  forbid appears at the very top.

  The chain that made it invisible is worth writing out, because each link is
  reasonable on its own. `npu-translate` validates the `Program` **in memory**
  before it writes anything, which is the right thing to do and catches encoder
  defects that corrupt the record. It does not validate the bytes it then
  writes, so an encoder that serialises a correct record incorrectly gets past
  it. `npu-objdump` reads those bytes, fails validation, and prints its warning
  block above a listing whose every line still matched, because swapping two
  fields that both round trip through the same instruction does not move any
  address or shape. The suite scanned underneath the warning and saw nothing.
- **What the finding actually is.** The P5 handoff's lesson says a product side
  fault proves the whole net and usually lights the earliest gate, and this one
  lit nothing except the step it was aimed at. Section 19.1's rule is that a
  fault caught by a different job than expected is a finding to record rather
  than a fault to adjust, and the same applies with more force to a fault caught
  by no other job at all: the net in front of the encoder's *output* was one
  badly placed directive deep, and only the unit tests were holding it.
- **Resolution:** the `DUMP-NOT` moved above the first `DUMP`, where it covers
  the span from the start of the input to the first match, which is where the
  warning block is. Re-measured with the same fault: `check-npu` now reports 17
  of 18 with `NPU :: Encoding/objdump.mlir` red, and restoring returns 18 of 18.

### D-0025 the sanitizers job died at its first link for want of the clang runtime archives

- **Found:** 2026-08-20, phase P6, by the sanitizers job on its first ever CI
  run (run 32339476756), at the CMake compiler check, before any project file
  compiled.
- **Status:** resolved 2026-08-20.
- **Reproduce:** in the CI image, compile any file with
  `clang++ -fsanitize=address,undefined` and link it. The link fails with
  `cannot find libclang_rt.asan_static-x86_64.a`.
- **What was wrong:** Ubuntu ships clang and its sanitizer runtime archives in
  separate packages, and the image installed `clang` without
  `libclang-rt-18-dev`. The local rehearsal ran under WSL's clang 21, whose
  runtimes are installed, so the gap was invisible until the job ran where it
  ships. Compiling works; only the link fails; CMake's try-compile catches it
  at configure.
- **Resolution:** `libclang-rt-18-dev` baked into the image's final stage per
  the environment pinning rule (a job time apt install would resolve
  differently as mirrors move), image republished, dev image repinned to the
  new digest.

### D-0026 the simulator aborted on an instruction with no operands

- **Found:** 2026-08-31, phase P7, by an audit of the unvalidated execution
  path rather than by a failing test.
- **Status:** resolved 2026-08-31.
- **Reproduce:** build a `Program` whose single `RELU` carries no operands at
  all, hand it to `Simulator::runUnvalidated`, and run the binary. Before the fix
  the process aborted inside libstdc++, and gtest printed its crash handler's
  frame list, through `abort` and `pthread_kill`, instead of a result.
- **What was wrong:** every kernel indexes its operands positionally.
  `kernelRELU` reads `instruction.operands.front()`, the elementwise and windowed
  kernels read `operands[0]` and `operands[1]`, and none of them looked first. On
  the validated path that is correct and deliberate: `Program::validate()`
  refuses an instruction with too few operands and names the `arity` check, so
  the guarantee is already bought and re-checking it per element would be paying
  for it a second time in the inner loop.

  `Simulator::runUnvalidated` carries no such guarantee, and it is not a hole
  left by accident either. Section 9.3 requires the bounds checked accessors to
  refuse gracefully in every build mode and requires a test to prove it, and a
  program that has passed `validate()` cannot reach that path at all, so a test
  that could only submit validated programs would be asserting that a mechanism
  exists rather than that it works. That entry point exists to reach the last
  line of defence. It reached past it.

  **The failure is worse than a crash, which is why it is a defect and not a
  hardening task.** `_GLIBCXX_ASSERTIONS` is defined in the default build, so
  `vector::front()` on an empty vector aborts, and an abort is exactly the
  assertion Section 9.3 says the trap path must not have: graceful refusal is
  the contract in a release build and in an assertions build alike. In a build
  without that macro the same call is silent undefined behaviour, which is the
  version that would have reached a report.
- **Resolution:** `Simulator::execute` compares `instruction.operands.size()`
  against `opcodeInfo(opcode).minOperands` before dispatching and records a trap
  naming both numbers when it is short. The bound comes from the table generated
  out of `NPUISADescription.td`, which is the description `validate()` reads, so
  appending an opcode creates no second arity table to keep in agreement with the
  first. `Trap.AnInstructionWithTooFewOperandsTrapsGracefully` submits the case
  through `runUnvalidated` and
  `Trap.TheValidatorRefusesAnInstructionWithTooFewOperands` submits the same
  program through `run`, which is the pair the rest of that file is written in.

### D-0027 a docstring claimed a check that did not exist

- **Found:** 2026-08-31, phase P7, while auditing what the cost model mirror
  actually asserts.
- **Status:** resolved 2026-08-31.
- **Reproduce:** read the module docstring of
  `python/npu_frontend/cost_model.py` as it was first written. It said the
  charges below it are "checked against the simulator's own output on a real
  program, so a formula that drifted from the C++ would fail rather than quietly
  produce a second cost model". Then read
  `test/Python/test_cost_model_mirror.py`: it parsed the constants out of
  `CostModel.h` and compared them, and it checked the Python formulas against
  literals written beside them. Nothing ran the simulator.
- **What was wrong:** the constants were mechanically checked and the formulas
  were not. A mirror reproducing every constant and then charging with them
  differently would have passed every assertion in the file, which is the more
  expensive half of the drift the one home rule exists to prevent: the numbers
  agree, the answers do not, and the plots of the later phases are drawn from the
  answers.

  The docstring is the defect rather than a symptom of it. A file that describes
  a check it does not have is worse than a file with no check, because the next
  person reads the claim and stops looking.
- **Resolution:** `test_the_mirror_reproduces_the_machines_own_numbers` runs
  `npu-sim` over an exported differential case and reconstructs the statistics it
  prints from the Python mirror's own arithmetic: the raw MAC count exactly, and
  the occupancy terms and both timeline totals to the precision the tool prints
  at. The case is `matmul_narrow_bias`, and it is narrow on purpose: 5 by 19 by 3
  folds into two tiles and neither fills the array, so a mirror that had dropped
  the utilization or the preload term disagrees. Against a tile that fills the
  array both terms are 1 and a mirror without them would still have passed. The
  docstring now says what the file does, and names this defect for the half it
  did not.

### D-0028 the release build kept its assertions, and the gate's NDEBUG run proved nothing

- **Found:** 2026-08-31, phase P7, by reading the compile flags of a build whose
  tests had already passed.
- **Status:** resolved 2026-08-31.
- **Reproduce:** configure against the assertions LLVM at `~/llvm-project/build`
  with `-DCMAKE_BUILD_TYPE=Release`, then read the compile line for any object
  with `ninja -C build-ndebug -t commands
  lib/Simulator/CMakeFiles/obj.NPUSimulator.dir/Memory.cpp.o`. `-DNDEBUG` is
  there, and so are `-D_DEBUG`, `-D_GLIBCXX_ASSERTIONS` and a `-UNDEBUG` that
  comes **after** it. Compiling a file whose first lines are `#ifndef NDEBUG` and
  `#error` with those same flags fails.
- **What was wrong:** Section 9.3 requires the bounds checked accessors to hold
  in every build mode and the P7 gate asks for the trap tests in an assertions
  build and an NDEBUG build with all four runs shown. The first attempt at the
  NDEBUG half configured `-DCMAKE_BUILD_TYPE=Release`, built, ran the four tests,
  and watched them pass. They passed in an assertions build.

  `LLVMConfig.cmake` sets `LLVM_ENABLE_ASSERTIONS` as a plain variable from the
  LLVM tree this project is configured against, which shadows the cache value
  `-DLLVM_ENABLE_ASSERTIONS=OFF` sets, so the obvious override does not take
  either. `HandleLLVMOptions` then reads it, adds the two macros and the
  `-UNDEBUG`, and the last `-D` or `-U` on a command line wins. The project could
  not produce a build without assertions at all, and nothing said so.
- **What the finding actually is.** A proof that proved nothing, which is the
  same shape as P3's, P5's and two of P6's, caught by the same habit those
  recorded: read the thing the gate is about rather than the result it produced.
  A green test in the wrong build looks exactly like a green test in the right
  one, and the gate's wording is satisfied by either.
- **Resolution:** `NPU_FORCE_NDEBUG`, off by default, in the top level
  `CMakeLists.txt`. It sets `LLVM_ENABLE_ASSERTIONS` to `OFF` **before**
  `include(HandleLLVMOptions)`, which is where those flags are decided and which
  also rebuilds `LLVM_DEFINITIONS` from the same switch, and it defines `NDEBUG`
  explicitly so the option means what its name says under any build type. The
  configure log prints a line beginning `NDEBUG:` saying which way it went, and
  `docs/BUILD.md` carries the configure line and the reasoning, because the next
  person will otherwise reach for `-DCMAKE_BUILD_TYPE=Release` exactly as this
  one did.

  Two attempts in between are worth recording because each looked correct.
  Putting the block **after** `include(HandleLLVMOptions)` sets a variable
  nothing reads again. And rewriting `LLVM_DEFINITIONS` with `string(REPLACE)`
  does remove the macros, and then hands `add_definitions` one argument full of
  spaces, which CMake stops parsing at the first `=` and passes to the compiler
  as a definition of `_GLIBCXX_USE_CXX11_ABI` whose value is the rest of the
  line: twelve redefinition warnings out of a build that otherwise did exactly
  what was wanted. The forced path splits the string into a list itself for that
  reason, and the unforced path is left alone, because it works.

### D-0029 the differential oracle's random inputs were all negative

- **Found:** 2026-08-31, phase P7, while rehearsing the product side activation
  fault. It was not what the rehearsal was looking for.
- **Status:** resolved 2026-08-31.
- **Reproduce:** run the exporter with `NPU_DIFFERENTIAL_OUT` set and read any
  `.in0.bin` back as f32. Before the fix every value in every file was in
  `[-1, 0)`, against a comment that said `[-1, 1)`.
- **What was wrong:** one bit. The generator is a linear congruential step
  followed by

  ```
  const uint32_t bits = static_cast<uint32_t>(state >> 33);
  return static_cast<float>(bits) / 2147483648.0f - 1.0f;
  ```

  A shift of 33 leaves 31 significant bits, so `bits` is at most `2^31 - 1`, the
  division lands in `[0, 1)` and the subtraction in `[-1, 0)`. The intended
  shift is 32, which keeps all thirty two bits, divides into `[0, 2)` and
  subtracts into `[-1, 1)`. The same generator was copied into
  `DeterminismTest.cpp`, so both carried it.

  **Nothing about the output looked wrong**, which is the whole difficulty. The
  values were random, deterministic, reproducible from the seed, and inside the
  interval the comment named at one end. Every case still ran, every comparison
  still passed, and the suite reported twenty four cases agreeing.

  What it cost is specific. The `relu` case compared all zeros against all
  zeros: an input drawn entirely from `[-1, 0)` makes both the reference and the
  simulator produce a buffer of zeros, and the assertion that they agree is an
  assertion about nothing. The two `max_pool2d` cases never saw a window whose
  maximum was positive, which is precisely the case an accumulator initialised
  to zero rather than to negative infinity gets wrong. And the determinism test
  convolved negative inputs against negative weights, so every product in the
  reduction carried the same sign, which is the easiest possible case for a
  summation order to survive.
- **How it was found, because the route matters.** The product side activation
  fault for `NPUSimulatorTests` was chosen to be `POOL_MAX` starting its
  accumulator at zero, on the prediction that the hand computed tests would
  catch one case and the differential suite would catch the two pooling cases on
  randomized inputs. The differential did go red, and it reported **54 of 54**
  elements mismatched on `max_pool2d`. That number is wrong for the fault: with
  inputs spread over `[-1, 1)` and windows of four, about one window in sixteen
  has an all negative maximum, so the expected figure was three or four. A
  rehearsal is supposed to confirm a prediction, and the useful part of one is
  the number that does not match.
- **Resolution:** the shift is 32 in both files. Three guards, because the
  defect is invisible in the output and a comment is not a mechanism.
  `Differential.TheStreamSpansBothSigns` draws a hundred thousand values and
  asserts the extremes reach past 0.99 in both directions, so a range that
  narrowed rather than shifted is caught too.
  `test_the_exported_inputs_straddle_zero` makes the same check against the
  bytes that actually reached the files, pooled across the export and then per
  operand for the operands of sixteen elements or more, and it survives a
  rewrite of the C++ that keeps the comment and loses the property. The per
  operand rule is conditioned on size deliberately: the three element bias of
  `matmul_narrow_bias` is all positive in this export, which happens to a fair
  three element sample one time in four. And the same test asserts the `relu`
  case's reference output is not entirely zero, which is the case that goes
  vacuous first and the one a reader would least expect to.

### D-0030 a test's subprocess found the package only because the caller had exported it

- **Found:** 2026-08-31, phase P8, by `scripts/coverage.sh` on its first run
  with a Python threshold. Not by the test suite, which had been green.
- **Status:** resolved 2026-08-31.
- **Reproduce:** run `python -m pytest test/Python/test_input_classes.py` from a
  shell that has **not** exported `PYTHONPATH`.
  `test_the_seed_survives_a_new_process` fails with a `CalledProcessError`, and
  the child's stderr says `ModuleNotFoundError: No module named 'npu_frontend'`.
- **What was wrong:** the test spawns a second interpreter to prove the cell
  seed is stable across processes, which is the property `zlib.crc32` has and
  `hash` does not. It spawned it with the inherited environment. `pythonpath` in
  `pyproject.toml` puts this project's package root on **pytest's own**
  `sys.path` and exports nothing, so the child found `npu_frontend` only when
  the caller happened to have exported `PYTHONPATH` themselves. The developer
  wrapper used all through this phase does export it. `scripts/coverage.sh` does
  not, and neither does the CI coverage job.
- **Why it is worth an entry.** The failure mode is the one this project keeps
  finding: a test that passes for a reason that is not the reason it claims. It
  was green on every run anybody had done, it would have gone red in exactly one
  CI job, and the message there would have named a missing module rather than a
  missing variable. A test whose result depends on the shell that started it is
  not a test of the code, and the two other subprocess spawning tests added in
  this phase were checked for the same fault: `test_compile_driver.py` launches
  `scripts/npu-compile`, which puts the package root on `sys.path` itself, and
  `test_end_to_end.py` launches pytest, which reads `pythonpath` from
  `pyproject.toml`. Neither depends on the caller.
- **Resolution:** the child's `PYTHONPATH` is constructed from the repository
  layout rather than inherited, and the subprocess runs with `check=False` so a
  failure surfaces the child's stderr instead of a `CalledProcessError`
  traceback with the interesting half in a variable nobody prints. The docstring
  says why.

### D-0031 the NDEBUG and sanitizer directories cannot build anything that links MLIR

- **Found:** 2026-08-31, phase P8, on the first attempt to run the end to end
  pipeline against the `build-ndebug` binaries.
- **Status:** resolved 2026-08-31 as a documented limit, not as a code change.
  The real fix is out of scope and is named below.
- **Reproduce:**

  ```bash
  ninja -C build-ndebug -j6 npu-opt
  echo 'module {}' | ./build-ndebug/bin/npu-opt -
  # double free or corruption (!prev), exit 134
  ```

  And the mirror, under the sanitizers:

  ```bash
  ninja -C build-fuzz -j6 npu-opt
  echo 'module {}' | ./build-fuzz/bin/npu-opt -
  # AddressSanitizer: use-after-poison, inside mlir::BuiltinDialect::initialize
  ```

- **What was wrong, and it is not this project's code.** Both directories set a
  compile time option in **this project's** translation units that the prebuilt
  LLVM they link does not share.

  `build-ndebug` turns `_GLIBCXX_ASSERTIONS` off, which is exactly what
  `NPU_FORCE_NDEBUG` exists to do, while `~/llvm-project/build` is
  `LLVM_ENABLE_ASSERTIONS=ON` and its archives were compiled with the macro on.
  Mixing libstdc++ hardening across a link changes the definition of standard
  containers between translation units, which is an ODR violation and undefined
  however carefully the flags are written.

  `build-fuzz` has the mirror: AddressSanitizer's container annotations are on
  in this project's translation units and off in the LLVM archives, so an
  instrumented translation unit poisons a `SmallVector`'s unused capacity and
  uninstrumented MLIR code then writes into it.

- **The evidence that it is the directory and not the compiler**, because the
  distinction is the whole entry:
  1. `echo 'module {}' | build-ndebug/bin/npu-opt -` aborts. That input reaches
     no operation of this dialect, no pass, no pipeline and no tool code.
  2. The AddressSanitizer stack is `main`, `MlirOptMain`,
     `MLIRContext::MLIRContext`, `BuiltinDialect::initialize`,
     `addTypes<..., Float8E5M2Type, ...>`. Every frame is MLIR's, and it is
     reached before this project's code runs at all.
  3. The same sources built in `build` run both inputs at exit 0, and have done
     so several thousand times across this phase.
- **Why it went unnoticed for two phases.** `build-ndebug` was created at P7 to
  prove Section 9.3's "every build mode" claim about the simulator's bounds
  checked accessors, and the simulator links the format library and LLVM's
  `Support` and no MLIR at all. Nothing had ever built an MLIR linking target in
  it. P8 is the first phase whose end to end run wants `npu-opt`, so P8 is the
  first phase that could find this. The CI sanitizers job builds three targets
  by name and none of them links MLIR, which is why CI has never seen it either.
- **Resolution.** The limit is made loud rather than papered over. The
  `NPU_FORCE_NDEBUG` branch of the top level `CMakeLists.txt` emits a
  `message(WARNING)` naming the two sound targets and the failure mode,
  `docs/BUILD.md` carries the reproduction and the reasoning, and
  `docs/PHASE_STATE.md` records it as an open question.

  **The real fix is a second LLVM tree**, built with
  `-DLLVM_ENABLE_ASSERTIONS=OFF` for the NDEBUG half and with the sanitizers for
  the other. That is an hour of build time per tree and a decision with a cost,
  like the third CI build P7 left open, so it is left to be taken deliberately
  rather than taken at the end of a phase. Nothing any gate has asked for is
  blocked by it: Section 9.3's four runs are the simulator's, and the simulator
  is one of the two targets these directories build correctly.

- **2026-09-01, interphase P9b: the second LLVM tree is declined, and this entry
  stays open as a limit.** The decision and its argument are
  `docs/adr/0009-ndebug-coverage-without-a-second-llvm-tree.md`. In short:
  Section 9.3's contract lives entirely in `NPUSimulatorTests` and
  `NPUEncodingTests`, which are exactly the two targets this directory builds
  soundly, so a non-assertions LLVM would buy coverage of MLIR linking tools
  that no clause asks about, at an hour of runner time per build, a second
  published image to keep in step with the first, and a second `LLVM_IMAGE`
  reference for every future phase that moves the LLVM tag to move.

  **What changed in CI instead** is the `ndebug` job of
  `.github/workflows/ci.yml`, which configures `-DNPU_FORCE_NDEBUG=ON`, asserts
  in its own configure log that the option took, and builds and runs those two
  binaries and nothing else. The two release side runs of Section 9.3's four now
  happen on every push rather than on whichever developer remembered the second
  directory.

  **This entry is not resolved by that** and the status line above is unchanged.
  The MLIR linking tools still cannot be built here without assertions, and a
  later phase that writes a clause about those binaries in a non-assertions build
  should reopen ADR 0009 rather than work around this. The difference P9b makes
  is that the cost is now declined against a stated requirement instead of
  deferred against an unstated one.

### D-0032 three tool discoveries disagreed, and only one of them said so

- **Found:** 2026-08-31, phase P8, **by CI**, on the first run of the coverage
  job's new Python arm. Run 33367169622, job 99449908485. Every local run had
  been green, and no local run could have found it.
- **Status:** resolved 2026-08-31.
- **Reproduce:** in a checkout with **no** `build/` directory, which is what the
  coverage job has because it configures `build-coverage/` and nothing else:

  ```bash
  git worktree add --detach /tmp/p8cov HEAD
  cd /tmp/p8cov
  export MLIR_DIR=... LLVM_DIR=...      # what the container image supplies
  unset NPU_BUILD_DIR NPU_OPT PYTHONPATH
  bash scripts/coverage.sh 85 90
  ```

  The C++ half passes, then:

  ```
  ERROR collecting test/Python/test_metamorphic.py
  test/Python/test_metamorphic.py:248: in <module>
      @pytest.mark.parametrize("level", levels_that_eliminate_dead_code())
  python/npu_frontend/compile.py:156: in describe_pipeline
      tool = find_tool("npu-opt")
  E   VerificationError: npu-opt was not found ... Looked at $NPU_OPT (unset or
  E   not a file), /tmp/p8cov/build/bin/npu-opt , and PATH.
  Interrupted: 1 error during collection
  STEP EXIT=2
  ```

- **What was wrong.** `scripts/coverage.sh` builds into `build-coverage/` and
  never told the Python suite so. The suite looks for a binary at
  `$NPU_BUILD_DIR/bin`, then `<repo>/build/bin`, then `PATH`, and in that job
  none of the three held anything.

  **It worked on a developer machine only because `build/` happens to sit beside
  `build-coverage/` there.** That is the same failure class as D-0030 two entries
  up: a result that depended on what else was lying around rather than on what
  the command was given.

- **Why it surfaced at collection rather than as a failing test**, which is the
  detail that made it fatal instead of merely red. `test_metamorphic.py`
  parametrizes a test on `levels_that_eliminate_dead_code()`, which shells out
  to `npu-opt`. A parametrization is evaluated when the module is imported, so
  the tool lookup ran in the **collection** phase, where there is no test to
  attach a failure or a skip to. pytest exits 2 and the whole run stops. Reading
  the level set from the compiler is what makes the P9 check fill itself without
  an edit, so the parametrization stays; what changes is that the environment
  now supplies the tool.

- **The two instances hiding behind the loud one, which are the worse half.**
  Three places in this project knew how to find a built binary:
  `npu_frontend.find_tool`, and a hand written `build_directory()` in each of
  `test_refexec_differential.py` and `test_cost_model_mirror.py`. The copies
  looked at `$NPU_BUILD_DIR` and `<repo>/build` and nothing else, and they
  **skipped** rather than failing when they found neither.

  Measured, by naming `npu-opt` directly so collection survived and leaving
  everything else as the job leaves it:

  ```
  SKIPPED test/Python/test_cost_model_mirror.py:214
  SKIPPED test/Python/test_refexec_differential.py:134
  SKIPPED test/Python/test_refexec_differential.py:162
  SKIPPED test/Python/test_refexec_differential.py:223
  SKIPPED test/Python/test_refexec_differential.py:252
  484 passed, 12 skipped
  ```

  Five tests, including `test_every_case_agrees`, which is the Phase P7 gate
  item that the reference interpreter and the simulator agree on every
  operation. **The run would have reported success and attached a coverage
  number to it**, and that number would have described a suite in which the
  differential oracle and the cost model mirror did not execute. Section 17.7
  says coverage is only counted from a run where every test passed. Every test
  did pass, and five of them passed by not running.

- **Resolution, in two parts, because there are two faults.**

  The **environment**: `scripts/coverage.sh` exports `NPU_BUILD_DIR` set to its
  own build directory, derives `MLIR_PYTHON_PACKAGES_DIR` from that directory's
  CMake cache when the caller has not set one, and asserts that the four
  binaries it just built are present before it asks the suite to find them. A
  missing one is now this script having failed to build it, which is a better
  sentence than a lookup failure three layers down.

  The **duplication**: `test/Python/tools.py` is the one rule, wrapping
  `find_tool`, and both copies are deleted. It also carries the policy that
  closes the silent half: a missing binary is a **skip** when nobody named a
  build directory, which is a developer running the pure Python tests, and a
  **failure** when somebody did, because a caller that names a build directory
  is asserting that the build is there.

  `test/Python/test_tool_discovery.py` is the mechanism rather than the
  intention. It scans the test modules for a file locating a build directory for
  itself, and separately for a file constructing a path to a binary by hand, so
  a fourth copy is a red test. `scripts/` is out of scope and the test says why:
  `regression_baseline.py` needs a *configured* directory with a `CMakeCache.txt`
  and a `test/` subdirectory, which is a different question from where a binary
  is.

- **Verified under the job's own conditions**, in a worktree with no `build/`
  and with `MLIR_PYTHON_PACKAGES_DIR` unset as well, which is stricter than CI:

  ```
  coverage: PASS. C++ 86.1 percent is at or above the threshold of 85 percent.
  coverage: MLIR bindings from /tmp/p8cov/build-coverage/CMakeCache.txt
  495 passed, 7 skipped
  coverage: PASS. Python 90.60 percent is at or above the threshold of 90.
  STEP EXIT=0
  ```

  495 passed where the broken state gave 484, and the Python coverage headroom
  widened from 0.27 points to 0.61 for the plainest of reasons: the tests that
  had been skipping now run.

- **What this says about the phase's own claims.** `docs/PHASE_STATE.md` recorded
  the coverage arm's activation rehearsal as matching its prediction, and it did,
  on a machine where `build/` existed. The prediction was about the threshold
  arithmetic and it was right about that; it was never a test of the arm's
  environment, and the handoff has been corrected to say so. **The first run of a
  new CI step is the first time anybody learns what that step's environment
  actually is**, which is the argument for switching steps on one at a time and
  is why the activation table exists.

---

### D-0033 the dialect could not materialise a constant, so `-sccp` did nothing

- **Found:** 2026-09-01, phase P9, by measuring rather than by a failing test.
  `-sccp` is one of the four upstream passes Section 12 puts in `-O2`, and
  before writing its Section 12 negative test I ran it over a module of `npu`
  operations to see what its positive case looked like. There was not one: the
  output was byte identical to the input on every module I gave it, including
  one built specifically to have a constant to propagate.
- **Status:** resolved 2026-09-01.
- **Reproduce:** a private function whose only caller passes a constant.

  ```mlir
  func.func private @scaled(%x: tensor<2xf32>) -> tensor<2xf32> {
    %c = npu.constant dense<[2.0, 3.0]> : tensor<2xf32>
    %d = tensor.empty() : tensor<2xf32>
    %r = npu.mul ins(%x, %c : tensor<2xf32>, tensor<2xf32>)
                 outs(%d : tensor<2xf32>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
  func.func @main() -> tensor<2xf32> {
    %c = npu.constant dense<[2.0, 3.0]> : tensor<2xf32>
    %r = call @scaled(%c) : (tensor<2xf32>) -> tensor<2xf32>
    return %r : tensor<2xf32>
  }
  ```

  `npu-opt --sccp` on that returned it unchanged. The argument should have
  become the constant, and the lattice said so.

- **Root cause.** `NPUDialect` implemented no `materializeConstant`. MLIR's
  constant propagation is two halves: a lattice that decides a value is a known
  constant, and a materialiser that builds an operation to hold it. The first
  half worked. The second had nowhere to go, so the solver asked the dialect for
  an operation, got null, and left the value alone. No diagnostic, no warning,
  and an exit code of zero: the pass reached the right answer and declined to
  write it down.

- **Why nothing caught it before.** `-sccp` was not in a pipeline until P9, so
  nothing ran it. That is the honest answer and it is also the uncomfortable
  one: the hook has been missing since P1 and would have stayed missing if P9's
  gate had not required a **positive** test per pass. A gate that asked only for
  the negative case would have been met by a pass that never fires, which is
  precisely why Section 12 asks for both.

- **Resolution.** `let hasConstantMaterializer = 1;` on the dialect and four
  lines in `NPUDialect.cpp`. The guard is narrow because `npu.constant`'s
  verifier requires the attribute's own type to equal the result type, so an
  attribute that is not an `ElementsAttr`, or one whose type is not the
  requested type, returns null rather than an operation that fails to verify.

- **What it would have cost.** `-sccp` is one of Section 16.2's ablatable
  passes, so at P10 it would have produced an ablation row of exactly zero on
  every cell. A reader would have concluded that constant propagation is worth
  nothing on this workload, which happens to be nearly true and would have been
  true for the wrong reason. `test/Transforms/level-passes.mlir` is the test
  that now separates the two.

---

### D-0034 two operations shared one destination and therefore one buffer

- **Found:** 2026-09-01, phase P9, by reading the `-O2` output of
  `test/Pipeline/opt-levels.mlir` while writing its CHECK lines. Not by a test:
  every test still passed, because the one shape that reaches it in the model
  suite computes the right answer anyway.
- **Status:** resolved 2026-09-01, before `-O2` was registered.
- **Reproduce:** two convolutions in a chain, at `-O2`.

  ```mlir
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %c0 = npu.conv2d ins(%x, %w : ...) outs(%d0 : ...) {...} -> tensor<1x2x4x4xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %c1 = npu.conv2d ins(%c0, %w : ...) outs(%d1 : ...) {...} -> tensor<1x2x4x4xf32>
  ```

  came out of `npu-opt --npu-O2` as

  ```mlir
  npuisa.conv2d ins(%view, %view_0 : ...) outs(%view_1 : ...)
  npuisa.conv2d ins(%view_1, %view_0 : ...) outs(%view_1 : ...)
  ```

  The second convolution reads that buffer through a three by three window and
  writes it at the same time.

- **Root cause, in two correct halves that are wrong together.** `-cse` merged
  the two `tensor.empty` operations, which is right: they are identical
  operations with no operands, `npu` operations are `Pure` and take their
  destination by value, and a destination has no contents, so two operations
  sharing one are two pure functions of the same meaningless input. Then
  `-npu-lower-to-npuisa` converted one `tensor.empty` into one `memref.alloc`,
  which had been right for four phases because the importer emits one
  `tensor.empty` per compute operation and nothing had ever merged them.

- **Why the tests did not catch it.** The shape that reaches this is a chain
  whose second operation reads its own output buffer, and for `npu.relu` that is
  elementwise and gives the right answer. Every model in the suite that fused
  produced the relu shape. The convolution shape above was written by hand while
  diagnosing something else.

- **Resolution.** In `-npu-lower-to-npuisa`'s pre conversion stage, every use of
  a `tensor.empty` after the first gets its own clone. The fix is in the
  lowering rather than in `-cse` for the same reason the aliasing rule is: this
  is the layer where a value becomes a buffer, and it is the layer that has to
  say one writer, one buffer. `test/Dialect/NPUISA/lowering.mlir` carries the
  convolution chain as its regression case.

- **The lesson, because it will recur.** Two passes each correct in isolation
  can be wrong in composition, and the composition only exists once a level runs
  both. Every pass P9 added is correct alone and had a lit test proving it; this
  defect is in the pipeline, and it was found by looking at what the pipeline
  produced rather than at what the passes did.

---

### D-0035 hoisted constants serialised every transfer ahead of every computation

- **Found:** 2026-09-01, phase P9, by measuring `-O1` against `-O0` on the seven
  model suite before recording the baseline. LeNet's cycles went from 17766.25
  to 24392.75, a 37 percent regression, from an optimization level.
- **Status:** resolved 2026-09-01, before `-O2` was registered.
- **Reproduce:** compile LeNet at `-O1` and read the statistics.

  ```
  O0 cycles 17766.25  dma_cycles 16441  compute_cycles 7955.75  overlap 0.8334
  O1 cycles 24392.75  dma_cycles 16441  compute_cycles 7955.75  overlap 0.0005
  ```

  The same twenty five instructions, the same transfer budget, the same compute
  budget, and none of it overlapping.

- **Root cause.** `-npu-lower-to-npuisa` emits one `npuisa.const` and one
  `npuisa.dma_load` at the position of each `npu.constant`, so where a constant
  sits in the block decides when its bytes are fetched. The two port cost model
  of Section 10.1 charges exactly that: a transfer overlaps a computation when
  the computation does not depend on it. MLIR's canonicalizer hoists every
  `ConstantLike` operation to the top of the block, which is right for an
  operation whose cost is zero and wrong for one that becomes a DRAM transfer.
  It moved all eleven of LeNet's constants above all of its compute, in an order
  that put the **last** layer's weights first, so the first convolution waited
  for essentially the whole transfer budget.

- **And the second half, found by the same measurement one step later.**
  `-npu-fuse-ops` takes every value its region reads as an operand, destinations
  included, so both destinations of a fused chain are defined above the region
  and the flattening leaves them there. Section 13.1's liveness runs from the
  allocation, so a destination defined earlier than it is written is a buffer
  held out of everyone's way for longer than the program needs it. On LeNet at
  `-O2` that raised the sweep line peak from 194624 bytes to 195040 and **the
  tight budget cell stopped compiling**: "the scratchpad budget of 194624 bytes
  is too small ... the requirement is therefore at least 195104". An
  optimization level that could not compile a program `-O0` compiled, at a
  budget Section 15 froze at P8.

- **Resolution, one rule for both.** In the same pre conversion stage, each
  `npu.constant` is moved back down above the run of constants and destinations
  that immediately precedes its first reader, and each `tensor.empty` is moved
  down to the operation that writes it. Both sink past **computation and nothing
  else**, which is why both are no operations at `-O0`, where the importer's
  placement is already this one, and why no `-O0` cell of the baseline moved.

  There is nothing to schedule here and no scheduling pass is implied: this pass
  chooses where to put a transfer and an allocation it is about to create, and
  the answer is where the data is used.

- **What it says about the phase.** Both halves are the same mistake seen twice:
  a pass moved an operation whose position was free at the level it was
  reasoning about and expensive one level down. Neither pass is wrong. What was
  missing is that the layer which turns a position into a cost had no opinion
  about position. It has one now, and `test/Dialect/NPUISA/lowering.mlir`
  carries the hoisted shape by hand so the rule is tested without needing a
  pipeline to produce it.

- **A note on where the tight budgets stand.** They are unchanged and still the
  numbers `docs/adr/0008-per-model-tight-scratchpad-budgets.md` froze at P8.
  This defect is the reason they did not have to move: the alternative to fixing
  the lowering would have been re measuring every tight budget against `-O2`,
  which would have moved every tight budget cell in the project's history to
  accommodate a placement artefact.

---

### D-0036 CI named an NDEBUG build it did not have, in a comment, for two phases

- **Found:** 2026-09-01, interphase P9b, while writing the `ndebug` job that
  replaces the claim. Not by a test, and no test could have found it: the fault
  was a sentence.
- **Status:** resolved 2026-09-01.
- **Reproduce:** configure this project the way the sanitizers job of
  `.github/workflows/ci.yml` does, against an assertions LLVM, and read the
  compile line rather than the result.

  ```bash
  cmake -G Ninja -S . -B /tmp/relwithdebinfo \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DMLIR_DIR=... -DLLVM_DIR=... \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  ninja -C /tmp/relwithdebinfo -t commands \
        lib/Simulator/CMakeFiles/obj.NPUSimulator.dir/Memory.cpp.o \
    | tr ' ' '\n' | grep -E '^-(D|U)(NDEBUG|_DEBUG|_GLIBCXX_ASSERTIONS)'
  ```

  Measured on 2026-09-01, the flags end
  `-DNDEBUG -D_DEBUG -D_GLIBCXX_ASSERTIONS -UNDEBUG`, and the last `-D` or `-U`
  wins. The configure also prints `NDEBUG: not forced` in so many words, which is
  the line this project added at P7 and which nobody read against this job.

- **What was wrong.** Two comments in `ci.yml`, written at P7, said the
  sanitizers job was the NDEBUG half of Section 9.3's "every build mode" clause,
  "which configures RelWithDebInfo and therefore compiles with `-DNDEBUG`". The
  therefore does not hold against an assertions LLVM. It is exactly D-0028, which
  this project found, fixed and documented **in the same phase**, and which then
  survived in a workflow comment because a comment is not a mechanism and nothing
  checks one.

  So CI had no NDEBUG coverage at all. The trap tests ran three times in three
  jobs, all with assertions on, and an accessor that had quietly become
  assert-only would have been green in every one of them.

- **Why it is worth an entry.** The failure class is this project's most
  frequent one and this is its purest instance: a claim that reads as a
  measurement and is an assumption. D-0028 was a proof that proved nothing
  because it was made in the wrong build; this is the same wrong build,
  described in prose, sitting in the file that decides what CI does. The
  distance between the two is one phase and one comment.

  It also says something about where to look. The reasoning in the comment was
  sound for a project whose LLVM has no assertions, and it was written by
  somebody who had just fixed the defect that makes it unsound here. Correct
  general knowledge applied to a specific configuration is not a mistake anybody
  reviews out.

- **Resolution.** The `ndebug` job, which forces the option and greps its own
  configure log for the line beginning `NDEBUG:`, so the claim is now made by a
  build that either has assertions compiled out or fails. Both comments are
  rewritten to say what their jobs actually cover and to name this entry.
  `docs/adr/0009-ndebug-coverage-without-a-second-llvm-tree.md` is the decision
  behind the job, including why a second LLVM tree is declined.

- **Proven red before it was believed.** The product side rehearsal compiled the
  range trap's diagnostic out behind `#ifdef NDEBUG` in `readBytes`, leaving the
  check and the null return in place so every caller still behaves. The
  assertions build reported 55 tests, 54 passed, 1 skipped, exit 0, and the
  NDEBUG build reported one failure,
  `Trap.AnOutOfRangeOperandAddressTrapsGracefully`, exit 1. One fault, one net,
  and the net is the one that did not exist before this entry.

---

### D-0037 the local coverage number was the union of every run the directory had ever seen

- **Found:** 2026-09-01, interphase P9b, by `scripts/coverage.sh` aborting in the
  middle of the closing verification matrix. Not by CI, and CI could not have
  found it.
- **Status:** resolved 2026-09-01.
- **Reproduce:** run `bash scripts/coverage.sh 85 90` three or four times in the
  same checkout without deleting anything in between. Somewhere after the third
  the collection dies:

  ```
  gcovr.formats.gcov.parser.common.SuspiciousHits:
    lib/Simulator/Kernels.cpp:87 Got suspicious hit value in:
      for (size_t axis = 0; axis < strides.size(); ++axis)
  (ERROR) Error occurred while reading reports: Worker thread raised exception
  ```

  and the script exits 64 with no percentage. Asking gcov directly for that line
  gives the reason:

  ```
  Kernels.cpp:87 count = 5896524226
  ```

  which is past gcovr's suspicious hits threshold of 2^32.

- **What was wrong.** gcov **accumulates**. A `.gcda` left in the build directory
  is added to by the next run rather than replaced, and this script had never
  deleted them, so `build-coverage/` held the sum of every run since P8: the
  lit suite, five GoogleTest binaries and the whole pytest matrix, several times
  over, in one set of counters. `Kernels.cpp:87` is the stride loop inside the
  odometer, which is the hottest line in the project, and it got there first.

- **The crash is the harmless half.** The half worth the entry is that a
  percentage collected from accumulated counters is a percentage about the union
  of every suite the directory has ever run. **A line covered by a test that was
  later deleted stays covered**, because the count that covered it is still in
  the file. This phase deleted a test, which is what makes the point concrete
  rather than theoretical, and the counter for whatever it exercised would have
  gone on being counted for as long as the object was not recompiled. The
  measured evidence for that is small and real: 42 `.gcda` files before the
  deletion and 41 after a clean run, so one object had counters and no longer
  runs at all.

  Measured either way, the number did not actually move on this tree: C++ 86.5
  and Python 90.50 against a clean run, which are the P9 figures. So this is a
  defect in what the number **means** rather than in what it currently **says**,
  and that is exactly the kind that survives until it is looked for.

- **The mirror of D-0030 and D-0032, and worth naming as such.** Those were
  results that depended on what else was lying around in CI, invisible on a
  developer machine. This is a result that depends on what is lying around on a
  developer machine, invisible in CI: the coverage job checks out a fresh tree
  and has no previous run to inherit, so several phases of green CI said nothing
  about it and could not have. The class is the same and the direction is
  reversed, which is a useful thing to know about a class this project keeps
  meeting.

- **Resolution.** `scripts/coverage.sh` deletes every `.gcda` under its build
  directory after the build and before the suites run, so the number describes
  the run that produced it. **Not** a wider
  `--gcov-suspicious-hits-threshold`, which would have silenced the crash and
  left the meaning wrong; the existing
  `--gcov-ignore-parse-errors=negative_hits.warn_once_per_file` stays, because
  D-0011 is a genuine tool artifact and this was not.

  The deletion is after the build rather than before it because ninja may
  recompile a source and the tree has to be final first.

- **Verified:** with 42 accumulated `.gcda` present, exit 64 and no percentage.
  After the deletion, `coverage: PASS. C++ 86.5 percent`, `coverage: PASS.
  Python 90.50425671250818 percent`, exit 0, and the run leaves 41 files behind
  that the next run will clear.

---

### D-0038 the dash linter was written in a Python the CI container does not have

- **Found:** 2026-09-01, interphase P9b, **by CI**, on the first run of the
  `regression-baseline --check` step. Run 33458934438, job 99704772026. Every
  local run had been green, every `lint` job had been green, and the pre-commit
  hook had been green on every commit since P0.
- **Status:** resolved 2026-09-01.
- **Reproduce:** run the linter under the interpreter the CI image ships, which
  is Ubuntu 24.04's, on a box with no project venv:

  ```bash
  docker run --rm -v "$PWD:/work:ro" -w /work \
    ubuntu:24.04@sha256:1e0a86e5... bash -c \
    'apt-get update -qq && apt-get install -qq --yes python3 git &&
     bash scripts/dash-lint.sh'
  ```

  ```
  File "/work/scripts/dash_lint.py", line 235
      except subprocess.CalledProcessError, FileNotFoundError:
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  SyntaxError: multiple exception types must be parenthesized
  ```

  Exit 1 for both `dash-lint.sh` and `dash-lint.sh --self-test`, which is
  exactly the `0 passed 2 failed` the drift report named.

- **What was wrong.** Two `except A, B:` clauses without parentheses, at
  `dash_lint.py:235` and `:255`. PEP 758 made that spelling legal in **Python
  3.14**; it is a `SyntaxError` in every earlier version. The module did not
  parse, so the linter never ran, and both invocations failed for one reason.

- **What it is not**, because four plausible mechanisms were checked and
  discarded before the fifth was found. Not the locale: `LANG` is unset in the
  container and Python still reports `utf-8` for both the filesystem encoding
  and the preferred encoding, so the unicode scanning was never at risk. Not
  `PATH`, not the working directory, and not a GNU `grep -P` dependence, because
  the linter shells out to `git ls-files` and does its scanning in Python. There
  is no `grep` in it at all.

- **Why several phases of green said nothing.** Four places run this linter and
  the fault is invisible in three of them.

  | Where | Interpreter | Why |
  |---|---|---|
  | developer machine | 3.14 | `dash-lint.sh` prefers `$HOME/npu-venv/bin/python` |
  | pre-commit | 3.14 | the same venv |
  | the `lint` job | 3.14 | `actions/setup-python` at `3.14`, on a plain runner |
  | the `build-and-test` container | **3.12** | the image's `python3`, and the script falls back to it because there is no venv |

  **The container had never run the linter.** The `lint` job does not run in a
  container and nothing else in `build-and-test` invoked it, so the
  `regression-baseline --check` step is the first thing in this project's
  history to run `dash-lint.sh` inside the image. It found this on its first
  attempt.

- **Why no tool caught it, which is the half worth fixing.** `pyproject.toml`
  declared `requires-python = ">=3.11"` and then configured black, ruff and mypy
  alike at `py314`. The promise was in a field nothing reads and every checker
  was pointed at the developer's interpreter rather than at the floor. The
  comment beside `target-version` argued for py314 **deliberately**, against the
  v1 tree's py312, on the grounds that an older grammar target applied to code
  running on 3.14 shows up as a formatting argument rather than as an actionable
  error. That reasoning is sound and it looks in one direction only. **The
  interpreter that matters is the lowest one that runs the code, not the
  highest.**

- **Resolution, in three parts because there are three faults.**

  The **syntax**: both clauses are parenthesised, which is what
  `requires-python` already promised.

  The **mechanism**: `[tool.ruff]` and `[tool.black]` are at `py311`, so the
  declared floor is enforced by a tool instead of promised in a field. Measured
  before the fix, at py311 ruff reports exactly these two errors over the whole
  tree and nothing else, and `black --check --target-version py311` leaves all
  forty two files unchanged, so the formatting argument the old comment feared
  does not arise here. mypy goes to `3.12` and not `3.11`, because at 3.11 it
  stops inside **numpy's own shipped stubs**, `numpy/__init__.pyi:737: Type
  statement is only supported in Python 3.12 and greater`, and checks nothing
  further. 3.12 is clean and is exactly the interpreter the CI image ships.

  The **diagnosability**: `run_dash_lint` in `scripts/regression_baseline.py`
  prints the child's output when it fails. Every other suite the baseline runs
  writes a machine readable file, so a failing test reaches the drift report by
  name; this one contributes a count, and the CI log said `suite dash-lint:
  passed 2 -> 0` while the `SyntaxError` explaining it went to a pipe nobody
  read. That is the standard the golden drift lines were rewritten to meet
  earlier on this branch, applied to the one suite that had been missed.

- **The baseline is unchanged and deliberately not re-recorded.** The recorded
  `2 passed 0 failed` was always right and the container was wrong. Re-recording
  here would have written a broken environment into the file as if it were
  correct, which is the thing `regression-baseline.sh` warns about in its own
  words.

- **What it says about the step that found it.** The `--check` step was switched
  on to answer a question about floating point reproducibility across hosts. It
  answered that question, in the affirmative, on the same run, and then caught a
  defect with nothing to do with it in a script five phases old, because it is
  the first thing that ever ran that script in that environment. **A step that
  runs the whole suite somewhere new is worth more than the reason it was added
  for.** That is the third time on this branch that the environment rather than
  the code turned out to be the thing under test.

- **Verified** under the conditions that reproduced it: a container from the
  pinned base digest, Python 3.12.3, no venv. `dash-lint: clean` exit 0,
  `self-test: all 8 expectations met` exit 0, and the baseline runner's own
  invocation reporting `suite dash-lint  2 passed  0 failed`, which is the line
  the recorded baseline expects. Unchanged on 3.14 in the venv.

---

### D-0039 the baseline compared a field with one end outside this project

- **Found:** 2026-09-01, interphase P9b, **by CI**, by the two `pull_request`
  activation proof runs for the step that carries the field. Runs 33461200759
  (PR 15, the NDEBUG fault) and 33461203436 (PR 16, the golden ulp fault). Both
  intended faults fired exactly as predicted; this arrived beside them, in both
  runs, unpredicted.
- **Status:** resolved 2026-09-01.
- **Reproduce:** run `scripts/regression-baseline.sh --check` against a baseline
  recorded on different CPU hardware. Eighteen cells report a moved
  `max_abs_error_vs_onnxruntime`:

  ```
  cell conv_bn_relu_stack-O0-default: max_abs_error_vs_onnxruntime
      1.894070e-07 -> 8.940697e-08
  cell inception_block-O2-tight: max_abs_error_vs_onnxruntime
      5.160464e-07 -> 5.960464e-07
  ```

  Three models, `conv_bn_relu_stack`, `inception_block` and `resnet_block`, at
  every level and both budgets, which is three times three times two. The
  movement is between 1e-8 and 1e-7 and it goes in **both directions**:
  `inception_block` came out *closer* to the oracle on the other hardware.
  **No golden tensor drifted and no cycle count moved.**

- **What was wrong, and it is the step's design rather than a line of code.**
  `max_abs_error_vs_onnxruntime` is the distance between this compiler's answer
  and `onnxruntime`'s. It has two ends and only one of them belongs to this
  project. Comparing it for **equality** asserted that both ends hold still.

  **This compiler's end does hold still, and the same file proves it.** The
  golden tensors pin every default budget cell's output bit for bit at a
  tolerance of zero, and `test/Python/test_tight_budgets.py` pins each tight
  budget answer to its default budget one. So with green goldens this field
  **cannot** move because of anything the compiler did. Every difference it can
  still report is a change at the other end.

  **The other end is `onnxruntime`, which dispatches its CPU kernels on what the
  host supports**, and GitHub's hosted runners are not homogeneous. The two
  earlier push runs of this step happened to land on hardware matching the
  recording host, which is why the first CI run reported zero drift on every
  numeric field and the cross host question looked answered in full.

- **The knowledge already existed in this repository and the baseline did not
  inherit it.** `test/Python/test_end_to_end.py` set its tolerances at ten and
  six times the observed maxima and said why, at P8, in these words: "this suite
  runs on at least two hosts ... and `onnxruntime` chooses its own vectorisation
  per host. A bound two times the observed value on one machine is a bound that
  goes red on another for a reason that is not a defect." The regression
  baseline then recorded the same quantity and compared it at a bound of
  **zero**. One file argued for a wide band on a number and another asserted
  equality on it, two phases apart, and nothing connected them.

- **Why the activation proofs found it and the ordinary runs did not.** The
  first two runs were pushes and both landed on matching hardware. The proof
  runs are two more pull requests, so these were the third and fourth samples of
  a population nobody had noticed was a population. **Two green cross host runs
  are one sample and not a proof**, and a step that runs on a fleet of
  heterogeneous machines has a distribution rather than an answer.

- **Resolution, and what is deliberately not changed.**

  `GOLDEN_TOLERANCE` stays at **zero** and every other cell field stays compared
  for **equality**. Those proved themselves on the same runs and nothing here
  touches them.

  `ORACLE_FIELD` names the one exception. It is checked against Section 17.4's
  absolute end to end band instead, which is the only thing the number ever
  meant, and the band is `npu_frontend.tolerances.ABSOLUTE_TOLERANCE`,
  **imported** rather than restated. The constants moved out of
  `test/Python/test_end_to_end.py` into the package for exactly that reason: a
  script cannot import a test module, and a second copy of a tolerance is the
  duplication D-0032's fix built `test_tool_discovery.py` to hunt for. A
  tolerance is the worst thing in a project to have two of, because the copies
  agree until somebody widens one.

  The recorded value **stays in the baseline**, as documentation of what the
  recording host measured. That is the status `tool_versions` already has in
  this file and it is the same argument: recorded because a reader wants it, not
  compared because it describes the machine rather than the project.

  A movement inside the band is **printed** rather than passed over, with its
  magnitude and its direction. A check that was switched off has to say so in
  its own output, which is Section 19.0's rule about silence and success not
  looking alike, applied to a field rather than to a step.

- **What is lost, stated rather than glossed.** The field can no longer catch an
  `onnxruntime` upgrade that moved the oracle. Two things still can: the version
  is recorded in `tool_versions`, and an upgrade is a diff in
  `requirements-lock.txt`, which is reviewed and which the pip cache key hashes.
  Neither is as loud, and that is the price of the fix.

- **Verified**, three ways, under the step's own script.

  1. Unperturbed: the numeric half is unchanged and the field reports nothing.
  2. With the recorded values perturbed by the magnitudes CI reported, in both
     directions, on exactly the three models CI named: **eighteen notes and no
     drift**, each naming the direction and the magnitude.
  3. With the band tightened to 1e-9 so the real measured values fall outside
     it: every affected cell produces a drift line naming the value, the band
     and the recorded figure, and the step goes red. The field is bounded, not
     ignored.

  And in `test/Python/test_regression_baseline.py`, four tests: a move inside
  the band in **both** directions is not drift, a move is reported as a note, a
  value outside the band is drift and is not also a note, and the band is the
  same object the end to end matrix imports.

### D-0040 seven tests were marked slow at P3 and CI has never run one of them

- **Found:** 2026-09-01, phase P10, while rehearsing the `pytest slow cells` step
  the activation table turns on at this phase. Found by a **prediction that was
  wrong**: I predicted the step would report two slow tests, the two P10 adds,
  and it reported nine.
- **Status:** resolved 2026-09-01 by the step being switched on.
- **Reproduce:** at any commit from P3 to P9b, ask which tests carry the marker
  and then ask which step in CI would run them.

  ```
  $ python -m pytest test/Python -q --collect-only -m slow | grep ::
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[conv_bn_relu_stack]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[depthwise_separable]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[dilated_stack]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[inception_block]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[lenet]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[lenet_batched]
  test/Python/test_model_generator.py::test_every_model_imports_at_a_second_seed[resnet_block]
  ```

  `ci.yml`'s `pytest` step runs the default marker expression, which deselects
  `slow`. The only step that runs `-m 'slow or not slow'` is `pytest slow cells`,
  and the activation table of Section 19.0 has that step off until P10. So these
  seven tests have existed since the model suite landed and no CI run has ever
  executed one.

- **What was wrong, and it is worth being precise about.** Nothing lied. The
  activation table said the step was off, the step said in its own log that it
  was off, and Section 19.0's rule that a step which is off says so was honoured
  throughout. What nobody noticed is the **consequence**: marking a test `slow`
  before the step that runs slow tests exists is the same as deleting it from
  CI, and the marker gives no hint of that at the point of use.

  These seven are not trivial. `test_every_model_imports_at_a_second_seed`
  regenerates every model at a different seed and imports it, which is what
  catches an importer that works on the committed weights and not on the shape of
  the graph. They ran on the developer machine, where the closing verification
  matrix of every phase uses `-m 'slow or not slow'`, so they were not
  unexercised; they were unexercised **in the environment that differs**.

- **Why this is D-0037's mirror, and the set is now six.** D-0030 and D-0032 were
  results that depended on what else was lying around in CI and were invisible
  locally. D-0037 was the reverse, a result that depends on what is lying around
  locally and is invisible in CI. This one is the reverse of the reverse: a
  *test* that only ever ran locally, so any environment dependent failure in it
  had exactly one place to hide and that place was the only place nobody looked.

- **The fix, and what it is not.** The step is on, so the seven run in the
  container from this phase. That is the whole fix and it is one guard line more
  than switching the step on: the step counts the collected slow tests first and
  fails when the count is zero, because a step whose marker expression selected
  nothing would exit 0 having repeated the step above it, which is how this
  situation would recur silently after somebody removed the last marker.

  What the fix is **not** is a rule against the `slow` marker. Section 17.4 asks
  for it and P10 adds two more uses of it that are worth having. The lesson is
  narrower and it is about ordering: **a marker that excludes a test from a
  default run is only safe once something runs the excluded set**, and this
  project introduced the marker six phases before it introduced that something.

- **Verified.** Rehearsed under the step's own script. With the markers in place
  the step reports nine and the suite runs green at 954 passed, 18 skipped. With
  every `@pytest.mark.slow` removed, the count guard reports zero and the step
  exits 1 with the message naming why a zero count is a failure rather than a
  quiet pass. Restored, and `git status` clean.

### D-0041 the first CI run of P10's tests asked questions a shallow checkout cannot answer

- **Found:** 2026-09-02, phase P10, by **the first CI run of this phase's tests**,
  run 33559636835 on `phase/p10-measurement`. Eight unique failures across the
  `pytest` and `pytest slow cells` arms, every one of them in a P10 test file,
  every one of them green locally.
- **Status:** resolved 2026-09-02.
- **Reproduce:** a shallow clone is the whole of it. `actions/checkout` defaults
  to `fetch-depth: 1`.

  ```
  $ git clone --depth 1 file:///home/elijah/npu-mlir-v2 /tmp/p10-shallow
  $ cd /tmp/p10-shallow
  $ git rev-parse --is-shallow-repository
  true
  $ git rev-list --count HEAD
  1
  $ git cat-file -e d4210f352957b95e69185003bb1b960a2a3286be^{commit}
  fatal: Not a valid object name
  ```

  and the four failures fall straight out of it:

  ```
  FAILED test_predictions.py::test_every_result_that_names_a_prediction_predates_it
  FAILED test_predictions.py::test_the_ancestor_check_refuses_a_sha_that_is_not_an_ancestor
  FAILED test_traceability.py::test_the_macros_sha_resolves_to_a_real_commit
  FAILED test_result_schema.py::test_every_manifest_git_sha_resolves
  ```

- **What was wrong, and it is two things rather than one.**

  **The checkout.** P10 is the first phase whose artifacts and tests name
  historical commits. Law 3 of Section 0.2 asserts that every published number
  traces to a commit that resolves, and law 4's mechanism is `git merge-base
  --is-ancestor` between the commit a prediction landed in and the commit a
  result was measured at. Neither question has an answer in a checkout that
  fetched one commit. Nothing before this phase asked, which is why nine phases
  of green CI say nothing about it.

  **The code, and this half is the more serious.** Three functions were asked
  something the checkout could not answer and each answered anyway.

  - `commit_exists` returned False, which reads as "this commit does not exist"
    when the truth is "this commit is not here".
  - `is_ancestor` returned False. `git merge-base --is-ancestor` exits nonzero
    both for "no" and for "I have never heard of that commit", and collapsing
    those turned "unobservable" into "the prediction does not predate its
    measurement", which is a serious finding reported where there was none.
  - `landing_sha` **did not fail at all.** `git log --diff-filter=A` against a
    truncated history attributes every file to the graft commit, because that
    commit has no parent to have differed from. So it returned the checkout's own
    tip: a plausible looking sha that is not the answer. Measured on a
    `pull_request` merge ref at depth 1, it returned the merge commit for an entry
    that landed six commits earlier.

  That third one is the reason this is a defect in the project rather than a
  setting in a workflow file. **A `prediction_sha` of the graft commit would have
  satisfied the ancestor test while recording a provenance link to the wrong
  commit**, which is worse than the failure it replaced, and no checkout option
  prevents it. It is the fault this project forbids everywhere else, an absent
  measurement that cannot be told apart from a real one, in the one mechanism
  whose entire purpose is provenance.

- **Four checkout shapes were measured before the fix was chosen**, because the
  `pull_request` trigger checks out a synthetic merge commit whose parents are
  the base and the branch, and that is a different shape from a `push`.

  | Shape | depth | shallow | `landing_sha` | historical sha resolves |
  |---|---|---|---|---|
  | `push` | 1 | yes | the graft commit | no |
  | `push` | 0 | no | `f92de42`, correct | yes |
  | `pull_request` merge ref | 1 | yes | the merge commit | no |
  | `pull_request` merge ref | 0 | no | `f92de42`, correct | yes |

  The merge ref at full depth answers every question correctly, including
  `HEAD~1`, which on a merge commit is the base branch: a parent, and an
  ancestor, so the refusal test holds there too.

- **The fix, both halves.**

  **`fetch-depth: 0` on the three jobs that run the suite**: `build-and-test` and
  `coverage` in `ci.yml`, and `full-matrix` in `nightly.yml`. The other four
  checkouts are left at the default because no step in them asks about history.

  **Depth 0 rather than a narrower fetch**, and the reason is that a narrow fetch
  would have to name the commits to deepen to. Those commits are the shas
  recorded in result files and prediction entries, so the fetch would need
  updating every time a result is re-recorded, and would be wrong in exactly the
  situation it exists to serve. The repository is a few hundred commits.

  **`require_full_history`**, which refuses once, readably, naming the checkout
  and the fix, before any of the three functions answers. `is_ancestor` now
  raises rather than returning False for an unresolvable reference, and
  `landing_sha` refuses rather than returning the graft commit. A shallow
  checkout now produces one diagnosable message instead of eight assertions about
  shas, and the message says `fetch-depth: 0` and names this defect.

- **Why the ancestor refusal test failed, since it looked like a test design
  issue and partly was.** `test_the_ancestor_check_refuses_a_sha_that_is_not_an_
  ancestor` asserts that a parent is an ancestor of its child, using `HEAD~1`.
  In a one commit checkout `HEAD~1` does not resolve, so `is_ancestor` returned
  False and the test failed claiming a parent is not an ancestor of its child,
  which is nonsense on its face and was the clearest signal in the whole run that
  the environment rather than the logic was wrong. The test was not wrong to use
  `HEAD~1`; the function was wrong to conflate two answers. Both are fixed: the
  function raises, and the test guards first.

- **Verified**, under all four checkout shapes, built as real fetches from a bare
  mirror rather than as mocks, because what is under test is what git does.

  - At depth 0, `push` and `pull_request` alike: **43 passed, 0 failed**, and
    `run_benchmarks.py` completes and exits 0.
  - At depth 1, after the fix: the suite reports the shallow refusal by name
    rather than eight failures about shas, and the harness refuses before
    measuring anything instead of exiting 2 with a message about an uncommitted
    prediction.
  - `test_the_ancestor_check_refuses_to_guess_in_a_shallow_checkout` makes a real
    `--depth 1` clone inside the test and asserts all three refusals, so the
    guard is exercised by the suite on every run rather than only by this entry.

- **The set is now seven, and this one sits with D-0037 and D-0040.** D-0030 and
  D-0032 depended on what was lying around in CI and were invisible locally.
  D-0037 was the reverse. D-0040 was a test that only ever ran locally. This is
  the fourth in that family and the sharpest: code that is correct in every
  environment where the history is present, run for the first time in an
  environment where it is not. **In all four the code under test was fine and the
  environment differed**, which is the argument for CI existing at all, and for
  this project's habit of writing down what a red run actually measured.

### D-0042 a git fatal was read as an answer, and the probe could not have told anyway

- **Found:** 2026-09-02, phase P10, by **the second CI run**, 33571635111, on the
  commit that fixed D-0041. Same eight tests red, but the evidence had moved:
  `fetch-depth: 0` had taken, the commits existed on the remote and were
  ancestors of the pushed tip, and the tests were printing D-0041's **new**
  message, "not a commit in this repository. The repository is not shallow, so
  the commit is genuinely absent rather than merely unfetched." Every clause of
  that sentence was false.
- **Status:** resolved 2026-09-02.
- **Reproduce:** the runner's shape is a workspace owned by the runner user and a
  job running as root inside a container. In the pinned image
  `ghcr.io/olajide-badejo/npu-mlir-llvm:llvmorg-22.1.8`:

  ```
  $ docker run --rm -v /tmp/ws:/__w -w /__w/repo -u 0:0 $IMG bash -c '
      git cat-file -e f92de427d1f315d9d6621c44516e54f886f18a9c^{commit}; echo $?'
  fatal: detected dubious ownership in repository at '/__w/repo'
  128
  ```

  and every helper in `npu_frontend.predictions`, run against it:

  ```
  repository_is_shallow       -> False        (should have refused)
  commit_exists(present)      -> False        (the commit is there)
  is_ancestor(pred, harness)  -> "not a commit ... genuinely absent"
  head_sha                    -> ""           (empty string)
  landing_sha(p10)            -> None         (the assert None is not None)
  require_full_history        -> passed       (it could not even see the shallow flag)
  ```

- **Two faults, and the second is the one that made the first unfixable.**

  **git's exit code was read as an answer.** `git cat-file -e` and `git
  merge-base --is-ancestor` both exit nonzero for "no" and for "I could not
  look", and this module read any nonzero as "no". A repository git refused to
  open therefore reported: the commit does not exist, the prediction does not
  predate its measurement, the repository is not shallow. Three findings, none
  true, out of one unreadable repository. `git rev-parse
  --is-shallow-repository` was worse: it prints its fatal to **stdout**, so the
  comparison against `"true"` was False and `require_full_history`, the guard
  D-0041 had just added, passed and let everything downstream run.

  **The probe could not have made the distinction at all.** Measured on
  2026-09-02, in a readable repository and in an unreadable one:

  | invocation | present | absent | unreadable |
  |---|---|---|---|
  | `cat-file -e <sha>^{commit}` | 0 | **128** | **128** |
  | `cat-file -e <sha>` | 0 | 1 | 128 |
  | `rev-parse --verify --quiet <sha>^{commit}` | 0 | **1** | **128** |

  `cat-file -e` with a `^{commit}` peel returns 128 for a well formed but absent
  object, the same code an unreadable repository gives. **So the first attempt at
  this fix, which read exit 1 as absence and 128 as a refusal, still could not
  separate them**: the exit codes were being read correctly and the probe was the
  wrong tool. `git rev-parse --verify --quiet` separates them, and additionally
  answers 1 for a sha that resolves to something that is not a commit, which is
  the right answer to the question being asked and one more thing
  `cat-file -e <sha>` would have said yes to.

- **Why the log for the first run pointed away from this.** Two facts made it
  look like anything but a git refusal. `regression-baseline --check` runs git in
  the same job and printed shas happily, and the pytest failures named specific
  shas as absent, which reads as a data problem. Both have the same explanation
  and it is a step ordering: `git config --global --add safe.directory` was first
  set by the `DIALECT_REFERENCE.md staleness` step at line 577, and `pytest` and
  `pytest slow cells` are at 463 and 520. **Every step above line 577 had a
  repository git would not read, and every step below it was fine.** The suite was
  the only thing above that line that asked git anything.

- **The fix, three parts.**

  **`_git` refuses to read a fatal as an answer.** One runner for every git call
  in the module, taking the set of exit codes that are genuinely answers. Anything
  else raises with git's own stderr in the message, because git explains itself
  better than a paraphrase and its message carries the fix.

  **`commit_exists` uses `rev-parse --verify --quiet`**, per the table above.

  **`safe.directory` is set immediately after the checkout**, in all three
  container jobs that run the suite: `build-and-test` and `coverage` in `ci.yml`,
  `full-matrix` in `nightly.yml`. The `coverage` job had never set it at all. The
  two later settings are left in place, because `git config --global --add` of a
  value already present is a no operation and a step that reads on its own is
  worth more than one fewer line, but they are no longer what makes anything work.

  **`run_benchmarks.git_sha` raises rather than recording `unknown`**, which is
  the same fault in the harness: that string would have gone into the manifest of
  every committed cell, and law 3 is that every published number traces to a
  commit that resolves.

- **This is the second time this project folded a nonzero exit into a boolean**,
  and the first was two days ago. D-0041 fixed `is_ancestor` returning False for
  "cannot tell" in a shallow checkout; this is the same sentence with a different
  cause. The lesson is not about git: **a helper that returns a bool for a
  question with three answers will eventually be asked the third one**, and this
  project already knows that, because Section 16.1 spends a paragraph forbidding
  exactly it for result fields and `values_of` refuses to average a null. The
  discipline existed and had not been applied to a subprocess call.

- **Verified**, in the pinned image with the workspace owned by uid 1001 and the
  container running as root, which is the runner's actual shape.

  | | before the fix | after |
  |---|---|---|
  | `repository_is_shallow` | `False` | refuses, naming git's fatal |
  | `commit_exists(present)` | `False` | refuses, then `True` once trusted |
  | `is_ancestor` | "genuinely absent" | refuses, then `True` once trusted |
  | `head_sha` | `""` | refuses, then the sha |
  | `landing_sha` | `None` | refuses, then `f92de427d1f3` |

  `test_a_git_fatal_is_never_read_as_an_answer` exercises all five entry points
  against a directory that is not a repository, which produces the same class of
  fatal without needing ownership games, and asserts that a well formed absent
  sha still comes back as absent rather than as a refusal. That last assertion is
  what chose the probe.

### D-0043 a bound between two clocks that ignored the quantum of the coarser one

- **Found:** 2026-09-02, phase P10, by **the third CI run**, 33575891610. That run
  is the one where `build-and-test` went green including every D-0041 and D-0042
  test, so those two are proven; the coverage job was red on one test.
- **Status:** resolved 2026-09-02.
- **Reproduce:**

  ```
  test_pass_instrumentation.py:273: AssertionError: assert 5.3 >= 5.301691
  ```

  Read the two numbers. `5.3` has one decimal; `5.301691` has six. They are not
  two measurements that disagree, they are one measurement and one **rounding**
  of another, compared as though both were exact.

  Locally, against `build-coverage`, ten runs of the same cell:

  ```
  run 0: mlir  4.1000  instr 3.978771  shortfall -0.121229
  run 2: mlir  5.5000  instr 5.297579  shortfall -0.202421
  run 3: mlir  3.2000  instr 3.262634  shortfall +0.062634   <-- fails a strict >=
  run 7: mlir  3.0000  instr 2.859018  shortfall -0.140982
  ```

  One run in ten. The gcov instrumented build is slower and jitterier, which does
  not create the fault but moves the sum across the boundary often enough to be
  seen.

- **What was wrong.** `--mlir-timing` prints **seconds to four decimal places**,
  so every figure this project parses out of it is a multiple of 0.1 ms and
  stands within 0.05 ms of a number it cannot see. Measured, the eleven parsed
  values of an `-O2` run: `0.1, 0.2, 0.1, 0.1, 0.4, 0.4, 0.2, 0.3, 0.2, 0.5, 0.2`.
  Not one of them has a digit finer than the quantum.

  The instrumentation's own figures are `steady_clock` differences at
  microsecond resolution.

  The direction the cross check asserts is sound and is not what changed: MLIR's
  timer opens before this instrumentation is called and closes after it, so
  `true_mlir >= instrumented`. What was wrong is that the comparison was made
  against the **printed** figure as though it were the true one. For one pass the
  error is at most half a unit in the last place; for a sum of `n` passes it is
  at most `n` halves, which at `-O2` is 0.55 ms. The assertion allowed zero.

  Two bounds beside it were **magic numbers that happened to work**, which is the
  same fault in a milder form: `TIMING_RESOLUTION_MS = 0.15` and
  `TIMING_FLOOR_MS = 0.15` were chosen loosely at three times a quantum nobody
  had written down. They are gone.

- **The fix, and the shape of it is the point.** The quantum is **derived from
  the text that was parsed**, per report, rather than written into the module.
  `parse_mlir_timing` now returns a `TimingReport` carrying the rows, the number
  of decimals it actually read, and half a unit in the last place computed from
  it. Every bound in the cross check is expressed in those terms:

  | bound | before | after |
  |---|---|---|
  | per pass, MLIR below ours | `0.15` | `half_ulp` |
  | per pass, MLIR above ours | `0.15 + 0.5 * mlir` | `half_ulp + 0.5 * mlir` |
  | totals | strict `>=` | `>= instrumented - n * half_ulp` |

  Measured against those: the worst per pass deficit over ten coverage build runs
  is **0.039971 ms** against a bound of 0.05, and the worst total shortfall is
  **+0.062634 ms** against an allowance of 0.55.

  **Deriving it rather than fixing it at 0.05 is not ceremony.** A report printed
  to two decimals of seconds has a quantum of 10 ms, and a constant would have
  gone on asserting a precision the figures no longer had, which is this defect
  again rather than a smaller version of it.
  `test_the_print_precision_decides_the_bound_rather_than_a_constant` feeds the
  parser a two decimal report and asserts the half unit comes back as 5 ms.

- **What was deliberately not done.** The bound was not widened to a number that
  makes the observed failure pass. 0.05 ms per pass is not a tolerance chosen
  against data; it is the largest error a rounded figure can carry, and the
  containment argument makes it exact: `printed >= true - half_ulp >=
  instrumented - half_ulp`. A deficit larger than that still fails, and still
  means the two clocks are not measuring the same run. **The cross check clause
  of P10's gate is only worth having if its bound is principled**, and it is
  tighter now than it was: the per pass bound went from 0.15 to 0.05.

- **Why the coverage job and not `build-and-test`.** Nothing about gcov is
  special here. The instrumented build makes each pass slower and noisier, which
  moves the sum of eleven roundings across zero more often. `build-and-test` has
  been running the same flawed assertion since P10 landed and passing it by luck,
  which is the more uncomfortable half of this entry: **a bound that is wrong by
  construction can be green for days.**

- **Verified.** 27 tests in `test_pass_instrumentation.py`, including the two new
  ones; ten runs against `build-coverage` with no failure and the numbers above;
  the full suite at 957 passed, 18 skipped.

### D-0044 SCALE-Sim v3 does not run under numpy 2, and says so by exiting zero

- **Found:** 2026-09-02, phase P11, on the first attempt to run the pinned tool,
  before any export existed to blame.
- **Status:** worked around, upstream unfixed. The workaround is
  `scripts/patch-scalesim.py` and it is applied by hand rather than by anything
  automatic.
- **Reproduce.** The tool's own shipped example, at the pinned sha, in
  `~/npu-venv`:

  ```
  python -m scalesim.scale \
      -c configs/scale.cfg \
      -t topologies/conv_nets/Resnet_test.csv \
      -l layouts/conv_nets/test.csv -p /tmp/out

  File ".../scalesim/memory/double_buffered_scratchpad_mem.py", line 307,
      in service_memory_requests
    self.total_cycles = int(max(ofmap_serviced_cycles))
  TypeError: only 0-dimensional arrays can be converted to Python scalars
  ```

- **What is wrong.** Three expressions convert a numpy array to a Python integer
  with the builtin `int`. numpy 2.0 removed that conversion for arrays of rank
  one or more, and `~/npu-venv` has numpy 2.5.1. `read_buffer.py:423`,
  `double_buffered_scratchpad_mem.py:279` and `:307`. Both code paths reach one
  of them, so neither the estimate bandwidth mode nor the user bandwidth mode
  runs. Checked against every upstream branch at the pinned sha, including
  `origin/3.1`: none of them fixes it.

- **The second half, and it is the one worth remembering.** While measuring the
  above, the tool was also run with no layout file:

  ```
  ERROR: scalesim.scale.py: Layout file not found
  Input file:./layouts/conv_nets/test.csv
  Exiting
  ```

  Exit status **0**. `scale_sim.set_params` calls the builtin `exit()` on every
  input validation failure, and `exit()` with no argument is status zero. So this
  tool reports a hard input error the same way it reports a successful run. An
  uncaught exception does exit 1, which makes the two failure modes report
  differently from each other, which is worse than either.

  **This is D-0040 through D-0043's shape, arriving from outside.** A channel
  that loses information, read as though it had not. So
  `experiments/scalesim_export.py` never treats the exit status as an answer: it
  requires `COMPUTE_REPORT.csv` to exist, to carry the header the parser was
  written against, and to hold exactly one row per exported layer, and it raises
  with the tool's own stdout and stderr when any of the three fails.
  `test_a_zero_exit_with_no_report_is_not_an_answer` proves the refusal.

- **Why the environment was not changed instead.** numpy 2.5.1 is pinned in
  `requirements-lock.txt` and is recorded in the manifest of all 175 committed
  results. Moving it to satisfy an external tool would make every recorded number
  describe an environment that no longer exists. A second interpreter with numpy
  1.x was the other option and is not available: Ubuntu 26.04 ships CPython 3.14
  only, and no numpy 1.x wheel exists for 3.14. Running the tool in a container
  was possible and was rejected as a larger change than the two line one, for a
  dependency this project runs by hand rather than in CI.

- **How a reader knows what ran.** The upstream git sha alone would describe code
  that is not the code that produced the numbers. So every result manifest
  carries **both** `scalesim` at the upstream sha and
  `scalesim_installed_tree_sha256`, a hash over the installed package as it was
  when the run happened. A modified install is visible in the record rather than
  hidden behind a sha that is very nearly true.

- **Deviation from Section 16.3, recorded rather than absorbed.** That section
  says to read the example topologies from the **installed path** of the pinned
  version. The pinned version's wheel ships the package and not its
  `topologies/` or `layouts/` directories, so the examples are read from the
  pinned source clone instead. `test_the_column_order_is_the_pinned_versions_own`
  reads the real file and asserts the exporter's header still matches it, so the
  rule that matters, that the column order is copied and never remembered, is
  enforced from the pinned source rather than dropped.

### D-0045 the cost model charges the array's weight preload once per instruction, and SCALE-Sim charges it per fold

- **Found:** 2026-09-02, phase P11, by the SCALE-Sim cross validation. This is
  the defect that cross validation exists to find, and it was found by the tool
  rather than by reading the code.
- **Status:** **withdrawn 2026-09-04, phase P13. The cost model does not do what
  this entry says it does.** The heading and the body below are left exactly as
  P11 wrote them, because this file is an audit trail and an entry rewritten to
  be right is an entry that cannot be learned from. What was wrong with it, and
  how it was found, is **D-0048**. Read that entry with this one.

  It was open deliberately at P11, for the reason Section 16.5 states for the
  same situation with ZigZag: do not silently retune a model against an external
  tool, because it invalidates every ablation and every number already recorded.
  That restraint was right and it is the reason nothing was broken by this. P13
  was handed the charge to change and measured it before changing it, which is
  D-0047's practice applied to D-0047's own successor phase.
- **Reproduce.** `resnet_block-O2-default-n1-fp32-normal`, layer `node_conv2d`:

  ```
  macs 36864   ideal cycles 144.0 (36864 / 256)
  this project  478.0 cycles   utilization 0.47   delta 0.80
  SCALE-Sim    1465   cycles   overall utilization 0.098
  ```

  Both tools are charged the same 36864 multiply accumulates;
  `check_the_same_arithmetic` reconciles SCALE-Sim's own utilization figure
  against that count before any of this is compared.

- **What the difference is.** The convolution presents a `72 by 8` weight matrix
  to a 16 by 16 array, which folds into five row tiles each occupying half the
  columns. `gemmCharge` computes `delta = m / (m + kWeightPreloadCycles)` **once
  per instruction** and applies it to every tile, so the sixteen cycle pipeline
  fill is amortised across the whole layer no matter how many times the array is
  actually refilled. SCALE-Sim refills the array per fold and charges each fill.
  With five folds the two accounts differ by roughly a factor of three, and that
  is the dominant term in the `array_fragmentation` column of the decomposition:
  minus 435825 cycles summed over the suite.

- **Which one is right.** SCALE-Sim's, on the mechanism: a weight stationary
  array physically has to push a new tile in before it can stream against it, and
  five tiles means five pushes. This project's `delta` is defensible as an
  average over an instruction and is documented as an assumption in
  `CostModel.h`, but the assumption is visibly optimistic for narrow deep GEMMs,
  which is most of this suite.

- **What it does not affect.** Nothing about correctness, and nothing about the
  golden files: `delta` enters the cycle charge and never the arithmetic. It
  affects `simulated_cycles`, `effective_macs` and `utilization`, and therefore
  every performance claim taken from them, which is why the entry says so here
  rather than in a report footnote.

- **What was deliberately not done.** The charge was not changed, the band in
  `experiments/predictions/p11-scalesim-divergence.md` was not widened, and no
  cell was excluded from the comparison after the fact. The prediction is
  answered as it was written, including where it is wrong.

### D-0046 the suite was green only on the machine that had the external tools

- **Found:** 2026-09-03, phase P11, by **CI**, on this branch's first push: run
  33691128405 and the pull request shape 33691234670. Three red clusters, one
  cause.
- **Status:** resolved 2026-09-03.
- **Reproduce.** Not on the developer machine, which is the whole point. In an
  interpreter that cannot reach SCALE-Sim or Accelergy:

  ```
  FAILED test_benchmarks.py::test_a_rerun_is_byte_identical_apart_from_the_timestamp_and_the_timing
  FAILED test_benchmarks.py::test_the_run_fails_when_it_exceeds_its_budget
  FileNotFoundError: [Errno 2] No such file or directory: 'accelergy'

  scripts/patch-scalesim.py:73: error: Unused "type: ignore" comment  [unused-ignore]
  scripts/patch-scalesim.py:73: error: Cannot find implementation or library stub
      for module named "scalesim"  [import-not-found]
  ```

- **What was wrong, and it is one sentence three times.** P11 installed two
  external tools on the development machine and then wrote code that assumed
  their presence in three different ways, none of them deliberate.

  1. **Two tests that have nothing to do with energy drove the whole harness.**
     `test_the_run_fails_when_it_exceeds_its_budget` tests the Section 2 budget
     gate and `test_a_rerun_is_byte_identical...` tests determinism, and both ran
     `run_benchmarks.py` without `--skip-external`, so both invoked Accelergy.
     The harness failing loudly on a missing tool is correct, per Section 16.4;
     what was wrong was asking it to.
  2. **A line level `# type: ignore` was correct in one environment and wrong in
     the other, twice.** Where SCALE-Sim is installed the import resolves and
     ships no `py.typed`, so mypy reports `import-untyped` and the ignore is
     used. Where it is absent mypy reports `import-not-found`, which the ignore
     does not cover, **and** `warn_unused_ignores` then reports the ignore
     itself. Two errors on one line, neither visible locally.
  3. **The coverage job was the same two tests under the instrumented run.**
     Verified rather than assumed: Python line coverage in an environment
     without the tools is **91.5561 percent**, identical to the figure with them,
     because `--cov=python/npu_frontend` measures the frontend package and the
     tool driven code lives in `experiments/`. Nothing hid behind the two
     failures.

- **This is D-0032 and D-0040 happening again one layer out**, and that is the
  reason it gets an entry rather than three small fixes. D-0032 was three copies
  of a tool lookup that skipped where one failed, and its fix was one policy:
  **skip when nobody promised the tool, fail when somebody did.** D-0040 was a
  set of tests that only ever ran on the developer machine. This is both: a set
  of tests that could only pass where the author's machine was special, and no
  mechanism to say which environments are supposed to have the tools.

- **The fix, in three parts.**

  - `test/Python/tools.py` gains `require_external_tools()` and
    `NPU_EXTERNAL_TOOLS`, which is `BUILD_DIR_VARIABLE`'s policy applied to a
    second kind of tool. A tool counts as reachable only when the module imports
    **and** its binary is on `PATH`, because Accelergy is driven as a subprocess
    and an importable package with no binary is a tool this project cannot run.
  - The two harness tests pass `--skip-external`, and what that gives up is
    recovered by `test_a_rerun_reproduces_the_external_fields_too`, which runs
    the same determinism check with the P11 fields included and is guarded by the
    policy above. `test_the_opt_out_records_a_null_and_a_reason` checks the flag
    those two now depend on: the fields are present and null, each reason names
    the flag, and `values_of` still refuses them.
  - `pyproject.toml` carries a per module `ignore_missing_imports` override for
    `scalesim` and the line level ignore is gone. **No global strictness setting
    moved.** The override names exactly one module, because the first version
    also listed `scalesim.*`, `accelergy` and `accelergy.*` and
    `warn_unused_configs` reported all three as unused sections.

- **Rehearsed in the environment that found it**, which is now the second
  environment this project has to stay green in. A meta path finder refusing
  `scalesim` and `accelergy` plus a `PATH` without the venv's `bin` reproduces
  both observable facts of the CI image: the imports fail and the binary is not
  found. Before the fix: the same two failures and the same two mypy errors.
  After: **997 passed, 29 skipped, 0 failed**, mypy clean, coverage 91.5561.
  With the tools present and promised: **1008 passed, 18 skipped**. And the third
  branch is proven rather than argued: with `NPU_EXTERNAL_TOOLS=1` set in the
  tool free environment, the guard **fails** naming the variable instead of
  skipping.

- **What was deliberately not done.** The harness was not made to tolerate a
  missing tool silently. Section 16.4 says a missing external tool fails loudly
  naming the dependency, and it still does; the opt out is a flag a caller sets,
  recorded in the result as a null with a reason naming the flag, and never a
  fallback the code chooses on its own. A test that could not find a tool is a
  skip; a run that was not told to skip is a failure.

#### The second half, found by run 33707070166: the rehearsal was wrong by two tests

The fix above went green in CI for `lint`, `coverage`, `sanitizers` and `ndebug`.
`regression-baseline --check` stayed red, and the drift was confined to the
pytest suite row: **996 passed and 31 skipped in CI, against a rehearsal that
predicted 998 and 29.** A rehearsal wrong by two tests cannot be trusted about
the other thousand, so the two were found rather than absorbed.

- **The two tests are `test_the_column_order_is_the_pinned_versions_own` and
  `test_the_layout_header_is_the_pinned_versions_own`.**
- **The mechanism is that neither imports `scalesim`.** They read the example
  topology and layout CSVs out of the **pinned source clone**, because the
  pinned wheel ships the package without its `topologies/` and `layouts/`
  directories, which is D-0044's recorded deviation from Section 16.3. The shim
  modelled the two things this phase had been thinking about, the import and the
  binary; the clone is a third thing, and `~/npu-external/` is a developer
  machine artefact no CI image has. So the meta path finder never touched those
  two tests, they ran here and skipped there, and every tool guard in the suite
  agreed the environments matched.
- **The shim models the clone now**, by pointing `NPU_SCALESIM_SOURCE` at a path
  that does not exist, and its comments say what it models and what it
  deliberately does not. Corrected, it predicts **996 passed and 31 skipped**,
  which is CI's row exactly.
- **The guard was wrong as well as the shim.** Those two tests skipped on a bare
  `is_file()` check, so `NPU_EXTERNAL_TOOLS=1` could not turn the skip into a
  failure. `tools.require_source_tree()` applies the policy the rest of the suite
  uses: skip where nobody promised, fail where somebody did.

**And the deeper cause, of which the two tests were the visible part.** The
baseline records suite counts and is checked in both environments, so it could
never be green in both: thirteen tests run on a machine with the external tools
and skip on one without, and recording either shape makes the other red. Copying
CI's numbers in would have made the developer machine permanently red instead,
which is the same defect facing the other way.

So the baseline records **which environment it was taken in**, and `compare`
compares `passed` and `skipped` only between two environments that can run the
same tests. `failed` is compared always. The test **name lists** are compared
always, so a test disappearing is still drift; and within one environment the
counts are still exact, so a test silently starting to skip is still caught,
which is the D-0040 shape this must not give up. The difference between two
environments is printed by `suite_notes` rather than discarded, because a
comparison this script has stopped making is a check that was switched off and
this project's rule is that such a step says so in its own output. That is the
treatment `max_abs_error_vs_onnxruntime` has had since P9b, arriving at the
suite counts.

`python/npu_frontend/external_tools.py` is the one home for "can this
environment reach the tools", because `test/Python/tools.py` and
`scripts/regression_baseline.py` both need the answer and two copies of it would
be the duplication D-0032's fix built a test to hunt for.

- **Verified in both environments against one baseline.** Developer machine:
  1013 passed, 18 skipped, **no drift**, counts compared exactly. CI shape: 1000
  passed, **31 skipped**, **no drift**, with both environments named in a note
  and the count difference printed. The skip count matches CI's 31 exactly; the
  pass count is 1000 rather than 996 because this fix adds four tests.
- **Four tests cover the change itself**: an environment difference is not
  drift, the same environment still compares exactly, a failure is drift in any
  environment, and a baseline recorded before the field existed reads as the
  developer machine rather than as something that compares equal to everything.

#### The third half, found by run 33711091899: the fix made a tree environment dependent

`build-and-test` went green including `--check`, so the environment aware
baseline is proven. `coverage` stayed red: **`python/npu_frontend` at 92.9004 in
CI against a threshold of 93**, with `scripts` at 16.12 against 14 and
`experiments` at 58.45 against 58 both passing.

**The threshold was not wrong when it was set; the tree moved under it.** 93 came
from a measurement taken before `python/npu_frontend/external_tools.py` existed,
and the commit that fixed the second half added that module to the tree the
threshold gates. Its whole purpose is to decide what an environment can reach, so
its tools present branches cannot execute in CI and its tools absent branches
cannot execute here. The prose beside the threshold claimed the frontend measured
identically in both environments, which had been true and had quietly stopped
being true.

**And re-measuring found a second cause the first does not explain.** With
`external_tools.py` covered the local figure was 93.3333 against CI's 92.9004, a
gap of about six statements the module cannot account for.
`pass_stats.interpreter_is_traced` asks two questions because CPython has two
tracing mechanisms, and **which one is live depends on the interpreter version**:
`coverage` uses `sys.settrace` on the 3.12 the CI image ships and
`sys.monitoring` on the 3.14 this machine runs, so the function answers on the
first question there and reaches the second one here.

Measured rather than deduced: `COVERAGE_CORE=ctrace` on 3.14 moves
`pass_stats.py` from 16 missing lines to 20, and the four are exactly the
`sys.monitoring` block. Two `# pragma: no cover` comments came off in the same
change, because the branches they excused are tested now.

- **Both are fixed by covering the branches, not by moving the gate.**
  `test/Python/test_external_tools.py` substitutes `find_spec`, `which` and the
  environment; `test_every_way_an_interpreter_can_be_traced` substitutes both
  tracing mechanisms. Neither consults the real environment, so both run
  everywhere. A test that asserted "the tools are reachable here" would pass on
  this machine and fail in CI for a reason that is not a defect, which is the
  shape of the thing being fixed.
- **Measured at the tip, and equal across four combinations** rather than two:
  CI shape and developer shape, each under both tracing backends.

  | Tree | CI shape | Developer | Threshold |
  |---|---|---|---|
  | `python/npu_frontend` | 93.4313 | 93.4313 | **93** |
  | `scripts` | 16.1191 | 16.1191 | **16** |
  | `experiments` | 58.4488 | 73.1302 | **58** |

  `scripts` moves from 14 to 16 because the environment aware comparison added
  tested lines to `regression_baseline.py`. Only `experiments` is environment
  dependent, which is by design and is where the tool driven code lives.
- **`external_tools.py` stays in the measured frontend package**, and the reason
  is recorded where the thresholds are so it is not re-litigated: the import
  graph forces it, because `scripts/regression_baseline.py` needs the same answer
  and a script must not import from `test/`. Moving it to `scripts/` would
  satisfy the import graph and drop it from a tree gated at 93 into one gated at
  16, which is hiding a measurement rather than making it true.

**The lesson this adds to the two above.** The first half was a suite green only
where the machine was special. The second was a rehearsal that modelled two of
three differences. This one is the same failure applied to a **number**: a
threshold is a measurement of a tree, and a commit that adds environment
dependent code to that tree invalidates the measurement as surely as it would
invalidate a recorded cycle count. **The prose beside a threshold is part of the
threshold**, and when it claims a property the code no longer has it is wrong in
the way a stale comment is never merely cosmetic.

### D-0047 the convolution kernel was never compiled with OpenMP, and the test that proves it is bitwise stable was comparing two serial runs

- **Found:** 2026-09-04, phase P12, by measuring a speedup that was not there.
  The first `experiments/kernel_threads.py` run reported 0.98 to 1.03 times at
  twenty eight threads against one, on all seven models. The models are small
  and that is exactly what a small model looks like, which is why the next step
  was `nm` rather than a shrug.
- **Status:** resolved 2026-09-04.
- **Reproduce**, at any commit from `078b78d` to `d67bb3c`:

  ```
  $ grep -A4 'Kernels.cpp.o: CXX' build/build.ninja | grep -c fopenmp
  0
  $ nm -C build/lib/Simulator/CMakeFiles/obj.NPUSimulator.dir/Kernels.cpp.o \
      | grep -ci 'GOMP\|omp_'
  0
  $ ldd build/bin/npu-sim | grep -c gomp
  0
  $ cmake -S . -B build | grep OpenMP
  -- OpenMP: found 4.5. The convolution kernel parallelises over the batch and
     output channel dimensions; the reductions stay sequential ...
  $ ./build/bin/NPUSimulatorTests --gtest_filter='Determinism*'
  [          ] OpenMP is on and reports 28 threads available.
  [       OK ] Determinism.OneThreadAndMaxThreadsAgreeBitwise
  ```

  Five lines saying the kernel is parallel, and an object file with no OpenMP
  in it.

- **What was wrong.** `lib/Simulator/CMakeLists.txt` wired OpenMP with one line,
  `target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)`.
  `add_mlir_library` compiles the four sources of `lib/Simulator` in an object
  library named `obj.NPUSimulator` and then assembles `NPUSimulator` from the
  objects, so a usage requirement attached to `NPUSimulator` reaches **everything
  that links it** and does not reach **what it is made of**. The imported
  target's `-fopenmp` therefore landed on every consumer and on none of the
  kernels. `#ifdef _OPENMP` around the convolution's `parallel for` was false in
  every build in every environment from P7 to P12.

- **The half that makes it a defect worth an entry rather than a missing flag.**
  `unittests/Simulator/DeterminismTest.cpp` is Section 10.3's assertion that one
  thread and the maximum produce bitwise equal buffers. It links `NPUSimulator`,
  so it **did** receive `-fopenmp`, so its own `_OPENMP` was defined, so it took
  the branch that prints a thread count and calls `omp_set_num_threads(1)` and
  then `omp_set_num_threads(28)`. Both calls succeeded. Both runs were single
  threaded, because the kernel between them had no parallel region. The test
  passed for three phases while asserting nothing, and its comment header said in
  as many words that a test which silently becomes vacuous is worse than no test.

  Nothing local could see it. `_OPENMP` is a property of a translation unit, and
  every translation unit that asked the question was answering it correctly about
  itself. The two that disagreed never compared answers.

- **This is the fourth appearance of P10's shape and the first one inside the
  build.** D-0040 through D-0043 were each a value that arrived through a channel
  which loses information, treated as though it had not. Here the lost
  information is *which target a compile option reached*, the channel is CMake's
  distinction between a target and the object library it is built from, and the
  code that treated it as exact was a `#ifdef` in one file and an
  `omp_get_max_threads()` in another. `docs/PHASE_STATE.md` at P11 wrote that the
  determinism test "asserts at full strength everywhere now". It did not, and it
  never had.

- **The fix, in three parts, and the third is the one that matters.**

  1. `add_compile_options` at directory scope in `lib/Simulator/CMakeLists.txt`,
     which is where `-Werror=switch` already lives for the same reason: a
     directory scope option applies to every target created after it, including
     the object library. The `target_link_libraries` line stays, because putting
     the runtime on a consumer's link line is a different job and this project
     needed both all along.
  2. `nbin::kernelsUseOpenMP()` and `nbin::kernelThreadCount()`, defined in
     `Kernels.cpp` and declared in `NPU/Simulator/Simulator.h`. They answer for
     the translation unit that has the parallel region in it, which is the one
     question no caller could previously ask.
  3. `Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`, which compares this
     test's `_OPENMP` against the kernels' answer and fails when they differ. It
     is red at every commit from P7 to P12 and green after the CMake fix, and it
     is the assertion that makes the two beneath it mean something.
     `npu-sim --kernel-info` is the same answer on a command line, so a harness
     or a person can ask it without linking anything.

- **What it did not move, which is the P12 claim.** With the parallel region
  compiled for the first time, `scripts/regression-baseline.sh --check` reports
  no drift across 42 cells and 21 golden tensors: not one cycle count, DRAM byte
  count, instruction count or golden byte. That is the inertness the phase is
  required to have, and it is now evidence rather than an expectation, because
  the thing it is evidence about finally happens.

- **A second, smaller fault the fix exposed**, and it is recorded here rather
  than as its own entry because it did not exist until the first one was fixed:
  an uncapped team made five of the seven suite models **slower than serial** at
  twenty eight threads, by as much as seven times on `depthwise_separable`, whose
  convolutions have eight and sixteen output channels and were being handed
  twenty eight threads each. The collapsed loop has `batch * outputChannels`
  iterations, so every thread past that count has no iteration to run and still
  pays for the region's entry and its closing barrier.
  `num_threads(min(tiles, max))` with `if (teamSize > 1)` is the fix and it
  carries no tuned constant: it is the number of independent output tiles the
  instruction has. Neither clause can move a bit, because neither changes which
  iterations exist, what one computes, or the order of the reductions inside it.

### D-0048 D-0045 named a mechanism the cost model does not have, and quoted a cell it does not match

- **Found:** 2026-09-04, phase P13, by measuring the charge before changing it.
  P13's brief was to fix D-0045 under the full declare then re-record
  governance, on the understanding that fixing it changes the cost model. It
  does not, because there is nothing there to fix.
- **Status:** resolved 2026-09-04. **The cost model is unchanged.** What was
  wrong was an entry in this file and the reading of the divergence
  decomposition that rested on it. D-0045 is marked withdrawn and its body is
  left exactly as P11 wrote it.

- **Reproduce, part one: the arithmetic.** D-0045 says `gemmCharge` "computes
  `delta = m / (m + kWeightPreloadCycles)` **once per instruction** and applies
  it to every tile, so the sixteen cycle pipeline fill is amortised across the
  whole layer no matter how many times the array is actually refilled". The
  premise is true and the conclusion does not follow from it.

  At the f32 peak the array's area and the peak are the same number, which
  `FrozenConstants.TheCostModelsNumbers` already asserts:
  `kPeakMacsPerCycleF32 == kArrayDim * kArrayDim`. So for any tile, whole or
  partial, `utilization * peak` is exactly `tileRows * tileColumns`, and the
  tile's charge reduces:

  ```
  tileMacs / (utilization * delta * peak)
      = (m * rows * columns) / (rows * columns * delta)
      = m / delta
      = m + kWeightPreloadCycles
  ```

  With `T` folds the instruction is charged `T * (m + kWeightPreloadCycles)`.
  That is the fill counted `T` times, once per refill, which is what a weight
  stationary array does and what SCALE-Sim charges. Applying the same
  **fraction** to every tile is not the same thing as counting the fill once,
  and the entry reasoned from the first to the second.

  Checked numerically as well as symbolically, over every combination of
  `m` in {1, 2, 7, 16, 64, 196, 1024}, `k` in {1, 8, 16, 17, 72, 144, 256} and
  `n` in {1, 6, 8, 16, 17, 120, 256}: **343 shapes, and the charge equals the
  explicit per fold accounting on all 343.** It differs from the once per
  instruction accounting on every shape with more than one fold, by exactly
  `(folds - 1) * kWeightPreloadCycles`. On D-0045's own `72 by 8` shape at
  `m = 64` that is 400 cycles against 336.

- **Reproduce, part two: the cell.** D-0045 names
  `resnet_block-O2-default-n1-fp32-normal`, layer `node_conv2d`, and quotes
  SCALE-Sim at **1465 cycles, overall utilization 0.098**. The committed result
  for that cell and that layer says something else, and has since P11:

  ```
                                          scalesim   utilization   stalls
  resnet_block-O2-default-n1  node_conv2d      549        0.2623        0
  resnet_block-O2-tight-n1    node_conv2d     1465        0.0983      916
  ```

  **1465 is the same layer at the tight budget, and 1465 minus 916 is 549.** The
  entry crossed a tight budget SCALE-Sim reading with a default budget
  analytical one. This project's 478 is the default budget figure, and it is a
  DMA bound charge rather than a compute one: the layer's
  `analytical_compute_cycles` is 404, of which 400 is the array and 4 is the
  issue overhead. So the entry compared 478 DMA bound cycles against 1465
  cycles of which 916 is SCALE-Sim waiting on memory, and attributed the whole
  difference to the weight preload.

  The 916 is not an anomaly of one layer. Across the 550 layer rows of the
  committed suite, **66 carry SCALE-Sim stall cycles and every one of the 66 is
  a tight budget cell**; the default budget cells carry none at all. Their
  median divergence is -72.42 percent against +11.59 percent for the 484 that
  do not stall.

- **Why neither the decomposition nor the headline moved because of it.** The
  stall term enters `decompose()` twice with opposite signs and cancels.
  `array_fragmentation` is `analytical_compute - (matched_total - stalls)`, so
  the stalls enter it positively; `double_buffering` is
  `max(0, dma - compute) - stalls`, so they enter it negatively. Summed over the
  suite the stalls are 107206 cycles against terms of plus 442289 and minus
  435825. That sign structure is a large part of why the two dominant terms are
  nearly equal and opposite, and it is worth stating beside the near
  cancellation rather than leaving the reader to find it: the cancellation is
  partly a property of how the terms are written and not only of the physics.
  The decomposition still sums to the total with a residual of zero, because it
  is a partition and the residual is defined as the remainder.

- **What the array fragmentation term actually is, restated.** With the stalls
  removed on SCALE-Sim's side the term is budget independent, which is what it
  was designed to be: `resnet_block`'s `node_conv2d` contributes 404 minus 549
  at both budgets. The two tools do disagree about the compute time of the same
  36864 multiply accumulates, by a factor of 1.36 on that layer and by as much
  as 8.65 on `dilated_stack`'s `conv1`. **That disagreement is real and it is
  unexplained.** What is now known is that the weight preload is not it, because
  both tools charge it once per fold. Naming the mechanism is left open below
  rather than guessed at, which is the state D-0045 should have been left in.

- **What was wrong, in one sentence.** An entry stated a mechanism that was
  inferred from reading one line of the code rather than from evaluating it, and
  supported it with a pair of numbers taken from two different cells.

- **It is the sixth appearance of P10's shape and the second in a claim rather
  than in code.** D-0040 through D-0043 were each a value that arrived through a
  channel which loses information, treated as though it had not. D-0047 was a
  build property that no reader could observe from inside the process that cared
  about it, and the test that asserted it had been vacuous for three phases.
  This one is nearer to D-0047 than to the other four: the claim was checkable
  from inside the artefact the whole time, in six lines of arithmetic, and
  nothing ever asked. **The frozen constants test could not have caught it**,
  and that is the part worth carrying: `FrozenConstants.TheCostModelsNumbers`
  pins `kWeightPreloadCycles` at 16.0 and says nothing about where the 16 is
  charged, so the accounting was never under any assertion at all.

- **The fix is a test, in the pattern P9 named.**
  `CostModel.TheWeightPreloadIsChargedOncePerFold` in
  `unittests/Simulator/CostModelTest.cpp` and
  `test_the_weight_preload_is_charged_once_per_fold` in
  `test/Python/test_cost_model_mirror.py` assert the per fold accounting **and
  assert it apart from the once per instruction accounting**, which is the half
  that matters: the two agree whenever there is exactly one fold, and every
  shape small enough to check by hand has exactly one fold. Both go red if a
  later phase changes the accounting, whether deliberately or by reverting to
  the model D-0045 described.

- **Rehearsed by injecting the defect D-0045 claimed was there.** The prediction
  was written first: pull `delta` out of the per tile divisor and add
  `kWeightPreloadCycles` once at the end of `gemmCharge`, and the new test goes
  red on the four multi fold cases while `FrozenConstants.TheCostModelsNumbers`
  stays **green**, because no constant moved. That is exactly what happened.
  The test named each shape and printed the difference: 16 cycles on `64 by 32
  by 16`, 64 on D-0045's `64 by 72 by 8`, and 2032 on the `16 by 256 by 120`
  tail of a fully connected layer, each of them `(folds - 1) * 16`. The two
  single fold cases stayed green, which is the reason the discriminating
  assertion is there. `test_the_mirror_reproduces_the_machines_own_numbers` went
  red in the same tree, because the machine moved and the Python mirror did not;
  the mirror's own copy of the per fold assertion stayed green, since it tests
  the mirror rather than the machine, and the mirror against machine test is
  what couples the two. Restored, tree clean.

- **What this changes about P13's brief.** P13 was handed a cost model change
  and the full declare then re-record sequence to run for it. **None of that
  sequence runs, because no charge moves.** No entry goes in
  `docs/BREAKING_CHANGES.md`, no baseline is re-recorded for this reason, and
  the pre-registered band of `p11-scalesim-divergence.md` is not re-versioned
  against new constants, because the constants and the accounting are both
  exactly what they were. Section 16.5's rule against retuning a model to match
  an external tool is what P11 obeyed when it left this open, and it is the same
  rule that says not to change the charge now on the strength of a diagnosis
  that does not survive being checked.

- **What is still open, and it is the real question D-0045 was reaching for.**
  The two tools disagree about the compute time of the same MAC count, widest on
  `dilated_stack` at 8.65 times and on `inception_block`'s 5 by 5 at 6.14, and
  in the other direction on the 1 by 1 convolutions where this project charges
  as much as 4.3 times what SCALE-Sim does. The dilation approximation already
  has its own term and its own second SCALE-Sim run, so it is accounted for
  separately and is not the answer. Whatever the mechanism is, it is not the
  weight preload, and the next phase to look at it should start by measuring the
  charge rather than by reading it.

### D-0049 the timing gap bound assumes the process had the CPU, and says so about a tracer but not about a busy machine

- **Found:** 2026-09-04, phase P13, by a single unexplained red in a full suite
  run, and then reproduced deliberately under load rather than left as a flake.
- **Status:** **open.** Reproduced, explained and **not fixed here**, because
  the fix is a change to a gate and the rule this project has about gates is
  that a red is not answered by widening the bound. The proposed fix is below
  and it is not a widening.

- **Reproduce.** Load the machine and run the test eight times:

  ```
  for i in $(seq 1 24); do ( while :; do :; done ) & done
  for run in $(seq 1 8); do
    python -m pytest \
      test/Python/test_benchmarks.py::test_the_opt_out_records_a_null_and_a_reason \
      -q -m 'slow or not slow'
  done
  ```

  **One red in eight under load; none in three on the idle machine**, and none
  in any of the four full suite runs this session took on an idle one. The
  failure:

  ```
  npu_frontend.pass_stats.PassStatisticsError: --mlir-timing reports
  Canonicalizer at 4.5000 ms and this project's instrumentation at 0.4496 ms, a
  gap of 4.0504 ms against a bound of 2.3000 ms, which is 0.0500 ms of display
  rounding plus 50% of MLIR's figure.
  ```

- **Which bound this is, because it is not the one P12 asked P13 to watch.**
  `cross_check_against_mlir_timing` has **two** bounds and they point in
  opposite directions. The **deficit** bound catches the instrumentation
  reading *above* MLIR, is derived from the print quantum, and is D-0043; that
  is the one whose margin narrowed across P11 and P12 at 0.1577, 0.1177 and
  0.1856 ms against 0.2000. **This is the other one**, the upper bound, which
  catches MLIR reading far *above* the instrumentation and is
  `half_ulp + 50 percent of MLIR's figure`. Nothing here is a fourth data point
  on D-0043's margin, and reading it as one would be reading the wrong number.

- **What is wrong, and the code already contains the argument.** The upper
  bound's premise is stated in its own message: the gap **is** the
  instrumentation's own operation walk, the one thing inside MLIR's window and
  outside this project's. `pass_stats.py` already knows that premise can fail
  and already refuses to check the bound when it does, for a tracer:

  > **It is not checked under a traced interpreter, and that is a precondition
  > rather than an exemption.** Under a tracer everything else in that window is
  > stretched too, so the gap becomes the walk plus whatever the tracer did, and
  > the bound stops measuring what it says.

  **A busy machine does the same thing for the same reason.** MLIR's timer is
  wall clock and brackets the whole pass, so when the process is descheduled
  mid pass that time lands inside MLIR's window; the instrumentation's own
  figure does not grow with it. The gap becomes the walk plus whatever the
  scheduler did, which is exactly the sentence above with one word changed.
  4.5000 ms for a canonicalization that this project measured at 0.4496 is not
  a canonicalization that took four milliseconds.

- **Where it matters, which is not this machine.** The test carries
  `@pytest.mark.slow` and CI's `pytest slow cells` step runs slow tests, on
  shared four vCPU runners. A developer machine with nothing on it is the least
  likely place for this to fire and CI is among the most likely, which is
  D-0037 and D-0040's shape again: a check whose behaviour depends on the
  machine, validated on the machine where it behaves.

- **The proposed fix, and it is a precondition rather than a wider bound.**
  Measure the compiling process's own CPU time against the wall clock across the
  invocation, and skip the **upper** bound when the ratio shows the process did
  not have the processor, exactly as it is already skipped under a tracer and
  with the same message. The deficit bound stays active in both cases, because
  its premise survives: MLIR's window contains this project's whatever the
  scheduler is doing in between. **What must not happen is
  `TIMING_GAP_FRACTION` moving from 0.5 to a number chosen to make this run
  green**, which would discard the only measurement the check exists to make.

- **What was done instead of fixing it.** It is recorded, reproduced with a
  written prediction, and handed to the flake governance of Section 17.9 at
  P15, which owns the quarantine marker with its owner and expiry fields. The
  one thing this entry adds beyond the reproduction is the discrimination: the
  test is not flaky, the bound is conditional, and the condition is not being
  checked.

- **A note on how it was nearly lost.** The first observation was a single red
  in a battery script that tailed three lines of output, so the message was
  gone before anybody read it and the session recorded it as unexplained.
  Capturing the whole failure is what turned a flake into an entry, and the
  cost of not doing it the first time was a second run of everything.

### D-0050 the binary format cannot express a buffer written in pieces, so a tiled program cannot be encoded

- **Found:** 2026-09-05, phase P13, by trying to encode a tiled program rather
  than by reading the format. The four things it took are below and each was a
  separate refusal.
- **Status:** **open, and escalated rather than fixed.** The fix needs a
  `Program::kVersion` bump, which reseeds the fuzz corpus and re-records the
  binary stability test, in the phase immediately before the one whose gate is
  written around that constant not moving. That makes it an owner decision
  rather than a phase's call, and P13 stopped at it deliberately.

- **Reproduce, and the order matters because each refusal hides the next.**

  1. A `memref.subview` of a DRAM argument, as the source of a `dma_load`,
     **parses, verifies and allocates**. `npu-translate` then refuses it:

     ```
     error: this DRAM buffer has no address in the DRAM map. The map holds the
     function's arguments, the npuisa.const results, and the allocator's
     npuisa.spill_slot allocations, and nothing else may live off chip
     ```

     That one is small and is not the problem. `dramAddressOf` is a map lookup
     while `scratchpadAddressOf` walks the view chain through
     `npuisa::computeBufferRange` and returns a base and an offset. Teaching the
     DRAM side the same walk is the change that was authorised, and it needs no
     format change: `Operand` already carries `address`, `shape` **and a stride
     per dimension**, so a sub region is `base + byteOffset` with the parent's
     strides, and `addressedByteSpan` already computes the span of a
     non contiguous view.

  2. **A tile writes into a sub region of a larger buffer, and the result side
     of an instruction has no strides.** `Instruction` carries `resultSpace`,
     `resultElementType`, `resultAddress` and `resultShape`, and no
     `resultStrides`. `FunctionEncoder::setResult` builds a full `Operand`,
     strides included, and then copies four of its five fields. So a strided
     write is not representable, and every spatially tiled convolution needs
     one.

  3. **Deeper, and this is the finding: the validation model assumes a buffer is
     written whole by one instruction.** ISA checks 8 and 9,
     `operand-defined` and `operand-extent`, ask whether a consumer's need fits
     "the element count actually written to the buffer it reads". A tiled
     program writes one buffer in pieces by construction, so the count at the
     base address is one tile's and the consumer wants all of them:

     ```
     error: the encoder produced a program that does not validate:
     operand-extent: operand 0 reads 2048 bytes from 0 and the buffer written
     there ends at 512 (instruction 3)
     ```

  4. **It is not fixed by strides alone**, which is the measurement that settles
     the scope. Channel tiling at batch 1 produces a **contiguous** sub region,
     because the channel axis is dimension 1 and everything under it is whole,
     so the strides a tile carries differ from the contiguous ones only on a
     dimension of extent one. That case needs no `resultStrides` at all, and it
     is refused anyway, by the same check, at 1024 bytes written against 2048
     read. **So the blocking constraint is the write model and not the layout.**

- **What is right about the current behaviour, and it is worth saying.** Nothing
  miscompiles. The encoder's own validator catches the inconsistency and
  `npu-translate` prints `this is a defect in the encoder rather than in the
  input, and no file has been written`. A format that could not express this and
  emitted a wrong program quietly would be far worse than one that refuses.

- **Why it is an owner level conflict rather than a phase's decision.** Three
  documents disagree once this is known.

  - **The format's own claim is narrower than it first looks, and it is quoted
    rather than paraphrased here, because overstating a conflict is the exact
    mistake D-0048 was about.** `Program.h` says the element types are present
    from version one "together with `requantMultiplier` and `requantShift`, and
    those specific fields **and nothing broader** are what let Phase P14 land
    without bumping `kVersion`". So the format does **not** promise that version
    1 carries every field any later phase might need. It promises P14's fields
    specifically, and about those it is right.
  - **P14's gate requires `Program::kVersion` unmoved**, with
    `test_binary_stability` green to prove it, on the grounds that the
    requantization fields have been present since version 1. A P13 bump would
    not contradict that clause, because P14 would still move nothing. What it
    would do is change the baseline the clause is measured against, and reseed
    the fuzz corpus and re-record the binary stability test inside a phase whose
    own gate says the goldens are byte identical.
  - Checks 8 and 9 are **declared** ISA checks in
    `include/NPU/Encoding/NPUISADescription.td`, numbered, and mirrored into
    `docs/ISA_MANUAL.md` and `docs/ISA_OPCODES.json` with
    `scripts/check-isa-staleness.sh` keeping the three in step. Changing what
    they mean is changing the declared ISA, not an implementation detail, and
    the fuzz corpus is seeded against the current one.

- **What a fix would have to be**, recorded so the decision has something
  concrete to weigh rather than a direction.

  - `resultStrides` on `Instruction`, written and read symmetrically with the
    operand strides that already exist, and `setResult` keeping the fifth field
    it currently drops.
  - `operand-defined` and `operand-extent` tracking **written ranges** per
    buffer rather than a single count per address, so that N disjoint writes
    covering a buffer satisfy a consumer that reads all of it. That is strictly
    more precise than the present rule and would still refuse the reshape case
    this file already records, where a short reshape writes fewer elements than
    its consumer reads.
  - A `kVersion` bump, the fuzz corpus reseeded, and `test_binary_stability`
    re-recorded against the new version.

- **What P13 did instead.** Stopped. The tiling pass stays implemented and in no
  `-O` level, the lowering patterns are not written, and no `kVersion` moved.
  Writing the patterns without the write model would have produced a compiler
  that emits programs the encoder refuses, which is a worse state than one that
  does not emit them.
