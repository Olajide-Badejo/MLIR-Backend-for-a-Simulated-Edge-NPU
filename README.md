<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# MLIR Backend for a Simulated Edge NPU

This repository is mid rebuild. I am rewriting the compiler from scratch
against the v3 build specification, and this README is a stub that says where
the work stands rather than a description of something finished. The v1 release
is tagged `v1.0.0` and stays reachable in git history; the rebuild releases as
`v2.0.0`.

**Current phase: P0, foundations.** What exists right now is the skeleton: an
out of tree MLIR project that configures against a prebuilt LLVM, an `npu-opt`
that registers the upstream dialects and passes, an empty `lit` suite wired as
`check-npu`, the dash linter with its fixture, REUSE headers on every file, and
a resolved lock file for the Python environment. There is no `npu` dialect yet.
That is P1.

I am not putting any numbers here until there are numbers that come from a
recorded experiment. The v3 specification forbids a hand typed number in this
file, and the machinery that makes a number quotable does not land until P8.

## Layout

The tree follows Section 6 of the build specification. `include/NPU` and `lib`
hold the dialects and passes, `tools` holds the drivers, `test` holds the lit
suite, `python/npu_frontend` holds the ONNX importer, `experiments` holds the
benchmark harness and its recorded results, and `report` holds the LaTeX that
reads those results. Directories arrive at the phase that fills them, so most
of that list is not present yet.

## Building

```
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld
ninja -C build -j6
ninja -C build check-npu
```

The build links against an existing LLVM at `llvmorg-22.1.8` and never builds
LLVM itself.

## Licence

MIT, in `LICENSE`. Files derived from the LLVM standalone example keep their
own Apache-2.0 with LLVM-exception licence and say so in their headers. Every
file carries SPDX tags and `reuse lint` runs in CI, so the provenance is
checkable rather than asserted.
