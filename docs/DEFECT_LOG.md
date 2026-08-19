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
