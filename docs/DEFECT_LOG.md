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
