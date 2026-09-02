<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 3. The resolved tool matrix, as installed on 2026-08-19

- **Status:** Accepted
- **Date:** 2026-08-19
- **Diataxis type:** explanation

## Context

The version policy in the build specification is a rule and not a list: pin the
newest stable release of every tool, resolved at Phase P0 on the machine that
will use it, never a version copied out of a document. Version numbers in a
specification go stale the moment they are written. A resolution rule does not.

Phase P0 is where that rule produces an actual list, and this record is that
list. Everything below was read off the installed tool on this machine, not
looked up. Where an entry is a floor rather than a pin, it says so.

The reason for writing it down at all is reproducibility with a name attached.
When a phase gate moves eighteen months from now, the question is going to be
which compiler produced the earlier number, and "GCC, presumably a recent one"
is not an answer that lets anyone reproduce anything.

## Decision

Pin the toolchain at the versions below, all verified installed and working on
**2026-08-19**.

### Host and platform

| Item | Resolved |
|---|---|
| OS | Windows 11 Pro, WSL2 guest Ubuntu 26.04 |
| Memory | 15 GB cap plus 8 GB swap, set in `~/.wslconfig` (12 GB from 2026-07-14 until 2026-08-19, raised by the owner) |
| Processors | 28 |
| Free disk | 895 GB |

The cap is a hard ceiling. It exists because this machine crashed in July 2026
when WSL2 ran uncapped, and it was 12 GB until 2026-08-19, when the owner
explicitly raised it to 15 GB, trading host headroom for guest capacity.
**`~/.wslconfig` is edited only on the owner's instruction; no part of this
project changes it on its own.** The practical consequence lives in
`docs/BUILD.md`: keep memory hungry parallelism at six jobs or fewer, and cap
link jobs at one if the LLVM contingency of Section 3.2 ever runs. Those
numbers were measured under the 12 GB cap and remain the safe defaults; nothing
has re-measured them at 15 GB.

### Build toolchain

| Tool | Version |
|---|---|
| GCC | 15.2.0 |
| clang | 21.1.8 |
| `lld` | 21.1.8 |
| CMake | 4.2.3 |
| Ninja | 1.13.2 |
| `ccache` | 4.12.3 |
| git | 2.53.0 |
| docker | 29.6.1 |

CMake 4.x is safe here. The LLVM standalone example this project's build is
derived from declares a 3.20 minimum, comfortably above the 3.5 floor that
CMake 4 removed.

LLVM and MLIR are pinned separately, at `llvmorg-22.1.8`, with the reasoning in
record 0001. That is the one entry in this matrix whose resolution is expensive
enough to deserve its own record.

### Python

| Item | Version |
|---|---|
| CPython | 3.14.4 |
| torch | 2.13.0+cpu |
| onnx | 1.22.0 |
| onnxruntime | 1.27.0 |
| nanobind | 2.13.0 |
| numpy | 2.5.1 |
| pytest | 9.1.1 |

The interpreter floor is 3.11, set by `zigzag-dse`. The installed 3.14.4 clears
it. `pyproject.toml` says `target-version = "py314"` for both black and ruff,
which matters more than it looks: the v1 tree said `py312`, and a formatter
applying an older grammar target to code running on a 3.14 interpreter shows up
as an argument about formatting rather than as an error anyone can act on.

torch is the CPU build. That is deliberate, and it means reference generation
runs on CPU: the installed onnxruntime carries the CPU and Azure execution
providers only, and the machine's GPU is not in use. It also has a consequence
for the lock file, because `2.13.0+cpu` is a local version that exists only on
the PyTorch CPU index and never on PyPI, so the lock file carries an
`--extra-index-url` line above its pins.

`pyproject.toml` pins the direct dependencies and `requirements-lock.txt` pins
the full transitive closure. Two layers, because one of them says what I chose
and the other says what that resolved to.

### Report engines

| Tool | Version |
|---|---|
| `tectonic` | 0.16.9 |
| TeX Live | 2025, with `latexmk` |

Both are present. `tectonic` is the primary engine and TeX Live is the fallback,
which is the order the build specification sets.

### Remote and CI

The GitHub remote is
`https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU.git`.
The repository is public, Actions are enabled, and `gh` is authenticated as
`Olajide-Badejo`. Verified 2026-08-19.

Publishing the LLVM image to GHCR happens through the CI `GITHUB_TOKEN` rather
than from this machine. The local token deliberately lacks `write:packages`,
which is a scope I would rather not hold on a development machine for the sake
of a step that CI can do with a token scoped to one repository and one run.

## Consequences

**This matrix is recorded again in every result manifest from Phase P10
onward.** That is the mechanism that makes it useful rather than decorative: a
number in the report resolves to a result file, and the result file carries the
compiler, the interpreter, the LLVM tag and the package versions it was produced
under. A matrix that lived only in this record would tell a reader what the
tools were on 2026-08-19 and nothing about what they were when the number was
measured.

The docker base images are pinned by `sha256` digest rather than by tag in both
Dockerfiles, from P0, because the image is published at P0 and re-pinning a
published image later means republishing it. Pinning an LLVM tag pins the
source, not the environment: any package install inside an image resolves
differently once mirrors rotate.

**A reproducible environment does not imply a bit identical output binary.**
This project claims the former and does not claim the latter, and the
distinction is stated here so no later document overclaims it.

Tools that install from source rather than from a package index, which is
Accelergy and its plug ins, are recorded by git sha rather than by package
metadata when the phase that needs them arrives.

### The external cross validation tools, installed at P11

*Added 2026-09-02.* Section 16.1 records these by git sha rather than by
installed metadata, because Accelergy is not on PyPI and its version string has
not moved in a long time, and because SCALE-Sim has no tagged GitHub releases.
Every one below was installed from source into `~/npu-venv` from a clone of its
upstream repository at the sha given.

| Tool | Version it declares | Git sha | Installed |
|---|---|---|---|
| SCALE-Sim | 3.0.0 | `9f98c4371055a54c75209c2e02b640b897550532` | 2026-09-02 |
| Accelergy | 0.4 | `6911d15686ee7efdceba7d95605102df4472ae3a` | 2026-09-02 |
| `accelergy-library-plug-in` | 0.1 | `ba4e9dac1b2e7a3076fb8b7816a5228211623055` | 2026-09-02 |
| `accelergy-aladdin-plug-in` | 0.1 | `5e2e1263ddcc896ba3b8ce95954d76cdeebe03ab` | 2026-09-02 |
| `accelergy-cacti-plug-in` | 0.1 | `7649b2c02a389f3c3d585d7ff4ececacfb01e6ea` | 2026-09-02 |
| `accelergy-table-based-plug-ins` | 0.1 | `bad19e941043045e130ea999852331f203d8c3fe` | 2026-09-02 |
| CACTI, vendored by the plug in above | none | `1ffd8dfb10303d306ecd8d215320aea07651e878` | 2026-09-02 |

**The clones live outside this repository, in `~/npu-external/`.** Vendoring
another project's source into this tree would put code this project does not
maintain under this project's licence header rules and its dash linter, for no
benefit a recorded sha does not already give.

**Two of the six needed something beyond `pip install .`, and both are recorded
rather than smoothed over.** `accelergy-table-based-plug-ins` imports `yaml` in
its `setup.py`, which pip's isolated build environment does not have, so it
installs with `--no-build-isolation`. `accelergy-cacti-plug-in` copies a built
CACTI binary its clone does not contain, so its submodule is initialised and
`make` is run before the install.

**SCALE-Sim's installed tree is not byte identical to its sha.** It does not run
under numpy 2 and `scripts/patch-scalesim.py` changes three expressions to make
it. D-0044 carries the account. Every result manifest therefore records
`scalesim_installed_tree_sha256` beside the sha, so the record says the tool was
modified rather than showing a sha that does not describe the code that ran.

**Every install used a constraints file built from `requirements-lock.txt`**, so
that no external tool could move a pin the 175 committed results were measured
under. Nothing in the lock file moved.
