<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Building and testing

*Diataxis type: tutorial plus how to.*

This is the build documentation for **this hardware**: a Windows 11 host running
a WSL2 Ubuntu 26.04 guest with a 15 GB memory cap, 28 processors, and an LLVM
already built at `~/llvm-project/build`. Nothing here is a general porting
guide. The versions it names are the resolved matrix of
[`adr/0003-resolved-tool-matrix.md`](adr/0003-resolved-tool-matrix.md).

**Every command below is annotated with the phase at which it starts existing.**
The project is mid rebuild, so a command marked P6 does not work today and that
is not a broken environment. This page carries only the commands that exist as
of Phase P0. The full list lives in Section 3.3 of the build specification, and
commands are copied here as their phase lands.

## Before anything else: which directory

Everything happens in **`~/npu-mlir-v2`**. That is the working clone, and it is
the only repository this project writes to.

`~/npu-mlir` is the **frozen v1 fallback**. Nothing writes to it, ever, and only
the owner may retire it. Record 0001 states the rule in full along with its HEAD
sha. The Windows side `npu-mlir\` folder is an independent stale clone and is
read only wreckage; its path contains spaces, which breaks `lit` and `FileCheck`
outright, so nothing is built there either.

## First build, from a fresh shell

```bash
cd ~/npu-mlir-v2

cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld

ninja -C build -j6                 # P0
ninja -C build check-npu           # P0, lit and FileCheck
```

Three things about that configure line are worth knowing rather than copying.

`MLIR_DIR` and `LLVM_DIR` point at an existing build tree, not at an install
prefix. This is an out of tree MLIR project in the shape of
`mlir/examples/standalone`, and it finds MLIR through `MLIRConfig.cmake` in that
tree. If the configure fails at `find_package(MLIR)`, the build tree moved or
was deleted, and the recovery is the contingency in Section 3.2 of the build
specification and not a path fixup.

`LLVM_USE_LINKER=lld` matters more here than it looks. `lld` is meaningfully
faster than the default linker and, more importantly on a memory capped guest, it uses
less memory to do the same link.

`-j6` is not a suggestion. See the memory ceiling below.

The build produces `build/bin/npu-opt`. Check it answers:

```bash
./build/bin/npu-opt --help         # P0
```

## The Python environment

The virtual environment is `~/npu-venv`, CPython 3.14.4. It is **reused, not
recreated**, and it lives outside both repositories, so it survives anything
done to either of them.

```bash
source ~/npu-venv/bin/activate
export PYTHONPATH=$HOME/llvm-project/build/tools/mlir/python_packages/mlir_core
```

**That `PYTHONPATH` is not optional and it is not a convenience.** The MLIR
Python bindings are built by the LLVM build and are not pip installed anywhere.
They resolve only out of that path. Without it, every test under `test/Python`
fails at import, with a traceback that says nothing about the code under test.

The wiring exists in three places so that an interactive shell is not the only
thing that knows about it: `CMakeLists.txt` derives
`MLIR_PYTHON_PACKAGES_DIR` from `MLIR_DIR` at configure time, `test/lit.cfg.py`
reads it for the lit suite, and `test/Python/conftest.py` sets it for pytest.
The conftest resolves the path from `MLIR_PYTHON_PACKAGES_DIR` if that is set in
the environment, then from this repository's own CMake cache, then from the
default location. It does that rather than hardcoding a string so that pytest
and lit agree by construction. Exporting the variable by hand, as above, still
works and still wins.

One trap, already paid for once and recorded as D-0002: the wiring must **not**
be written as `env = [...]` under `[tool.pytest.ini_options]`. That key belongs
to the `pytest-env` plugin, which is not installed here, and pytest accepts an
unknown key with a warning and then ignores it. The wiring would read as present
in review and do nothing at run time.

### Installing from the lock file

`pyproject.toml` pins the dependencies I chose. `requirements-lock.txt` pins
everything those choices resolved to. To reproduce the environment from
scratch:

```bash
python3 -m venv ~/npu-venv
~/npu-venv/bin/pip install -r requirements-lock.txt
```

The lock file carries an `--extra-index-url https://download.pytorch.org/whl/cpu`
line above its pins, and that line is load bearing. torch is pinned as
`2.13.0+cpu` because the CPU build is what is installed, and a `+cpu` local
version exists only on the PyTorch CPU index, never on PyPI. Removing the suffix
so PyPI resolves would silently install the CUDA build and pull roughly two
gigabytes of wheels onto a machine whose GPU this project does not use. That is
D-0003.

Missing packages are installed at the phase that first needs them rather than
all at P0, and the lock file is regenerated in the same commit.

## The checks that run on every commit

```bash
bash scripts/dash-lint.sh              # P0, lint every tracked file
bash scripts/dash-lint.sh --self-test  # P0, check the linter against its fixture
reuse lint                             # P0, every file carries SPDX tags
```

`dash-lint.sh` enforces ground rule 3: no em dashes and no en dashes anywhere,
and no TeX ligature hazard in `.tex` prose. It carries a fixture that proves
the verbatim exemption in both directions, so a `--check` flag inside a fenced
code block passes and an em dash inside the same block fails. Run the self test
after touching the linter and not otherwise.

`reuse lint` enforces the SPDX headers. Every new file needs
`SPDX-FileCopyrightText` and `SPDX-License-Identifier`, in a comment for code
and in an HTML comment for markdown. A file that genuinely cannot carry a header
gets an entry in `REUSE.toml` instead, and the test for which of the two applies
is simply whether a header would work.

Both are wired into `.pre-commit-config.yaml`, so:

```bash
pre-commit run --all-files             # P0, from the venv
```

runs them along with black, ruff, and the whitespace hooks.

## Regenerating the generated documents

```bash
python scripts/gen-design-decisions.py           # P0, write the ADR index
python scripts/gen-design-decisions.py --check   # P0, fail if it is stale
ninja -C build npu-dialect-doc                   # P1, DIALECT_REFERENCE.md
ninja -C build npu-isa-doc                       # P6, the ISA_MANUAL.md opcode table
```

`docs/DESIGN_DECISIONS.md` is a build artifact. Edit the record under
`docs/adr/` and regenerate; an edit made in the index is lost on the next run.

## The memory ceiling, and what to do when a build is killed

`~/.wslconfig` sets `memory=15GB` with 8 GB of swap. **That is a hard ceiling.**
It was 12 GB from July 2026, set after the machine crashed running WSL2
uncapped, until 2026-08-19, when the owner raised it to 15 GB. That file is
edited only on the owner's instruction; this project plans around the ceiling
rather than working against it. The parallelism numbers below were measured
under the 12 GB cap and stay in force at 15 GB until something re-measures them.

What it means in practice:

- **Compile jobs stay at six or fewer.** `ninja -C build -j6`, always. This
  project itself builds in seconds and is nowhere near the ceiling, so the six
  is habit rather than necessity today. It becomes necessary the moment
  anything large is compiled.
- **Test parallelism may use all 28 threads.** Tests are not memory hungry, and
  the ceiling does not bind there.
- **Link jobs stay at one** for anything LLVM sized. Parallel links of tools
  like `mlir-opt` are the classic out of memory kill on a small guest, and one
  is the only setting under which this machine has ever completed an LLVM link.
  Two is not a conservative alternative; it is an out of memory kill waiting for
  the wrong link to land at the wrong moment.

**If a build dies with the compiler killed and no error message**, that is the
out of memory killer, not a compiler bug. Look for the signature: a `ninja`
job that reports `signal: Killed` or a bare nonzero exit with no diagnostic, and
`dmesg | tail` naming `Out of memory: Killed process`. The fix is to lower the
job count, never to raise the memory cap.

### The LLVM contingency

**LLVM is never rebuilt on this machine.** It was built once at
`llvmorg-22.1.8`, the build at `~/llvm-project/build` is verified working, and
this project links against it. Record 0001 has the reasoning, and it is not a
preference: a rebuild is an hour or more of wall clock time on a guest that can
plausibly run out of memory partway through.

The one exception is deliberate and is not on this machine:
`docker/Dockerfile.llvm` builds the same tag inside CI, on GitHub's hardware, so
that the build and test workflow can pull an image instead of spending an hour
per run.

If the existing build is ever lost or found broken, follow **Section 3.2 of the
build specification** exactly, inside WSL2, with `LLVM_PARALLEL_COMPILE_JOBS=6`
and `LLVM_PARALLEL_LINK_JOBS=1`. Do not attempt a native Windows LLVM build. If
free disk is under 80 GB, stop and report before starting.

## Running a multi line command from the Windows side

Write it to a `.sh` file and run `wsl -d Ubuntu -- bash <path>`. Nested quoting
through PowerShell mangles anything more complicated than a single command, and
the failures it produces look like the script being wrong rather than the
quoting being wrong.
