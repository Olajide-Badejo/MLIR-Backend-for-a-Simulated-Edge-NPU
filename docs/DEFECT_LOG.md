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

None.

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
