#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The regression baseline of Section 17.6.
#
#   bash scripts/regression-baseline.sh            build, measure, record
#   bash scripts/regression-baseline.sh --check    build, measure, diff, fail on drift
#   bash scripts/regression-baseline.sh --help     this text
#
# This is the entry point and it does three things: it builds, it puts the MLIR
# Python bindings on PYTHONPATH, and it hands over to
# scripts/regression_baseline.py, which is where all the measuring and diffing
# lives. The split is Section 6's tree, and the reason it is worth keeping is
# that everything interesting here reads JSON and writes JSON, and a shell
# script that did it would be parsing logs.
#
# **It builds first, deliberately.** Section 17.6 says the baseline builds the
# project and then runs every suite. A baseline recorded against a stale build
# is a baseline describing a binary nobody has, and --check against one is a
# diff of two different compilers.
#
# **There is no `|| true` in this file.** A suite that failed and was recorded
# anyway is a baseline that records what is broken as if it were correct, which
# is the same rule scripts/coverage.sh states for the coverage number.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/.." && pwd)"

usage() {
  sed -n '6,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1-}" in
  --help | -h)
    usage
    exit 0
    ;;
esac

build_dir="${NPU_BUILD_DIR:-${root}/build}"
if [ ! -f "${build_dir}/CMakeCache.txt" ]; then
  echo "regression-baseline: ${build_dir} is not configured." >&2
  echo "regression-baseline: see docs/BUILD.md, or set NPU_BUILD_DIR." >&2
  exit 2
fi

jobs="${NPU_BASELINE_JOBS:-6}"
echo "regression-baseline: building ${build_dir} with -j${jobs}"
ninja -C "${build_dir}" "-j${jobs}"

# The bindings resolve only out of the LLVM build tree and are not pip
# installed. The cache is read rather than guessed, so this script, lit and the
# pytest conftest agree on one directory rather than three.
bindings="$(sed -n 's/^MLIR_PYTHON_PACKAGES_DIR:[^=]*=//p' "${build_dir}/CMakeCache.txt" | head -n 1)"
if [ -n "${bindings}" ]; then
  export MLIR_PYTHON_PACKAGES_DIR="${bindings}"
  export PYTHONPATH="${bindings}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${root}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 "${here}/regression_baseline.py" "$@"
