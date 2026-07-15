#!/usr/bin/env bash
#
# Measure line coverage on the C++ pass and backend implementations and on the
# Python frontend. Builds an instrumented copy of the project with gcov, runs the
# lit and GoogleTest suites to collect data, and reports with gcovr and pytest-cov.
#
# Usage: scripts/coverage.sh
# Requires the prebuilt LLVM/MLIR (see docs/BUILD.md) and a Python with gcovr and
# pytest-cov. Point MLIR_DIR/LLVM_DIR at the toolchain if not the defaults below.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
build=build-coverage
mlir_dir="${MLIR_DIR:-$HOME/llvm-project/build/lib/cmake/mlir}"
llvm_dir="${LLVM_DIR:-$HOME/llvm-project/build/lib/cmake/llvm}"

cmake -G Ninja -S . -B "$build" \
  -DMLIR_DIR="$mlir_dir" -DLLVM_DIR="$llvm_dir" -DLLVM_USE_LINKER=lld \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"

ninja -C "$build" npu-opt npu-translate npu-objdump NPUEncodingTests NPUSimulatorTests
ninja -C "$build" check-npu || true
"./$build/bin/NPUEncodingTests"
"./$build/bin/NPUSimulatorTests"

echo
echo "==== C++ coverage (pass and backend implementations) ===="
gcovr --root . \
  --filter 'lib/Dialect/' --filter 'lib/Encoding/' --filter 'lib/Simulator/' \
  --txt --print-summary

echo
echo "==== Python coverage (frontend and driver) ===="
python -m pytest test/Python --cov=npu_frontend --cov-report=term-missing -q
