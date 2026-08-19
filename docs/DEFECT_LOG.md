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
