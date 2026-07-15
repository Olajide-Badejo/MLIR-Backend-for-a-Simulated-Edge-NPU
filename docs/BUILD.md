# Build instructions

This project is a small out of tree MLIR/CMake project. It links against a prebuilt
LLVM/MLIR that you build exactly once at the pinned tag. The project build itself takes
seconds to minutes after that.

## 0. Pinned versions

- LLVM/MLIR tag: `llvmorg-22.1.8` (current tip of `release/22.x`).
- Build inside WSL2 Ubuntu. Do not attempt a native Windows LLVM build.

## 1. Target machine budget (read this first)

This machine runs WSL2 under a deliberate resource cap set in `~/.wslconfig`:
`memory=12GB`, `processors=8`, `swap=8GB`, `autoMemoryReclaim=gradual`. That cap exists
because its absence crashed the host on 2026-07-14 (WSL held roughly 12.8 GB and never
released it, and Windows paged to death). Do not raise it to build faster.

Because of the 12 GB ceiling, the classic out of memory failure is parallel links of large
tools such as `mlir-opt`. Two mitigations, both applied below:

- `LLVM_PARALLEL_COMPILE_JOBS=6`: at or below the 6 job guidance in `.wslconfig`.
- `LLVM_PARALLEL_LINK_JOBS=1`: one link at a time keeps peak link memory bounded.

If you still see the kernel OOM killer fire during linking, drop compile jobs to 4.

## 2. One time LLVM/MLIR build

Prerequisites (already present or installed on this machine): cmake 3.24+, ninja, gcc or
clang with C++17, git, lld, ccache, python3 with venv.

```bash
cd ~/llvm-project
cmake -G Ninja -S llvm -B build \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_COMPILE_JOBS=6 \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_CCACHE_BUILD=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$HOME/npu-venv/bin/python

ninja -C build check-mlir
```

`LLVM_TARGETS_TO_BUILD="Native"` keeps the build to this host's architecture only, which
saves both time and disk. `MLIR_ENABLE_BINDINGS_PYTHON=ON` requires nanobind and numpy in
the venv pointed to by `Python3_EXECUTABLE`, plus the Python development headers.

Watch Ninja's native progress display. Do not wrap it.

## 3. Building this project against that LLVM

The canonical repository lives in WSL native storage at `~/npu-mlir` (that is,
`/home/elijah/npu-mlir`). It is deliberately not under the Windows `/mnt/c` project folder:
that folder's path contains spaces, and LLVM's lit and FileCheck do not support spaces in
paths. WSL native storage is also much faster to build in than the 9p `/mnt/c` mount. From
Windows the repo is reachable at `\\wsl.localhost\Ubuntu\home\elijah\npu-mlir`.

```bash
cd ~/npu-mlir
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld
ninja -C build npu-opt
ninja -C build check-npu
```

## 4. Measured wall clock times

Recorded as they become real, replacing the spec's estimates. Do not quote estimates as if
they were measurements.

- One time LLVM/MLIR build (tag `llvmorg-22.1.8`, Release with assertions, mlir only,
  Native target, lld, ccache cold, 6 compile / 1 link jobs, WSL2 at 12 GB / 8 proc):
  about 28 minutes wall clock on the i7-14700K (measured 2026-07-15). Peak resident memory
  stayed near 3 GB of the 12 GB budget.
