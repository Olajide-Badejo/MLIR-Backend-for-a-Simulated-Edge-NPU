#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The C++ line coverage measurement of Section 17.7.
#
#   bash scripts/coverage.sh              measure, report, gate at the default
#   bash scripts/coverage.sh 85           measure, report, fail below 85 percent
#   bash scripts/coverage.sh --help       this text
#
# The threshold defaults to 0, which is what the activation table of Section 19.0
# asks for between P2 and P8: the job runs from P2 so the number is recorded and
# its trend is visible from the start, and P8 is where the real thresholds are
# set from what P8 measures. A threshold of 0 still gates on something real,
# because everything below still has to succeed for a number to exist at all.
#
# Three rules from Section 17.7 shape this script, and each is a line in it
# rather than a paragraph somebody remembers:
#
# 1. **A separate build directory.** Coverage instrumentation changes the
#    compiler flags, so measuring in the ordinary build directory would either
#    rebuild the world twice per switch or, worse, leave instrumented objects
#    behind for the next ordinary build to link. build-coverage/ is its own tree
#    and the ordinary one is never touched.
#
# 2. **Coverage is only counted from a run where every test passed.** There is
#    no `|| true` anywhere in this file, ever. A suite that failed and still
#    printed a percentage is a suite whose percentage describes a program that
#    does not work. `set -e` plus the explicit test invocations below is the
#    whole mechanism.
#
# 3. **CI parses a JSON summary rather than grepping a percentage out of a log**,
#    so a format change fails loudly instead of silently reporting zero. gcovr
#    writes the summary to coverage.json and this script reads the number back
#    out of it with python's json module, which raises on a malformed file.
#
# And the honesty note Section 17.7 insists on, repeated here because this is
# where somebody reads the number: line coverage measures execution, not
# assertion. A suite that runs every line and asserts nothing scores 100 percent.
# The percentage is a project management artifact and not a correctness argument.
# The real adequacy signal is the mutation testing that lands at P15.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/.." && pwd)"

usage() {
  sed -n '6,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

threshold="0"
case "${1-}" in
  --help | -h)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    threshold="$1"
    ;;
esac

# The threshold is compared numerically later, so it has to be a number now. A
# typo that reached the comparison would be read as 0 by some shells and as an
# error by others, and either way the gate would stop meaning what it says.
if ! printf '%s' "${threshold}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
  echo "coverage: the threshold must be a number, got '${threshold}'" >&2
  exit 2
fi

build_dir="${root}/build-coverage"
summary_json="${build_dir}/coverage.json"
summary_html="${build_dir}/coverage.html"

for tool in cmake ninja gcovr; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "coverage: ${tool} is not on PATH" >&2
    echo "coverage: gcovr comes from the project venv or from pip install gcovr" >&2
    exit 2
  fi
done

# MLIR_DIR and LLVM_DIR are taken from the environment when set, so that CI can
# point this at the image's install prefix without editing the script, and fall
# back to the ordinary build directory's cache otherwise. Reading them from the
# existing cache rather than hardcoding a path is what lets the same script work
# on a developer machine against an LLVM build tree and in CI against an install.
# The cache entry type is not part of the pattern. cmake records a variable
# passed on the command line as UNINITIALIZED and one declared with a type as
# PATH, and which of the two a given tree has depends on how it was configured
# rather than on anything meaningful. Matching `:` followed by anything up to the
# `=` reads both, where a pattern naming PATH silently found nothing against a
# tree configured the ordinary way.
cache_value() {
  sed -n "s/^$1:[^=]*=//p" "$2" | head -n 1
}

mlir_dir="${MLIR_DIR:-}"
llvm_dir="${LLVM_DIR:-}"
if [ -f "${root}/build/CMakeCache.txt" ]; then
  if [ -z "${mlir_dir}" ]; then
    mlir_dir="$(cache_value MLIR_DIR "${root}/build/CMakeCache.txt")"
  fi
  if [ -z "${llvm_dir}" ]; then
    llvm_dir="$(cache_value LLVM_DIR "${root}/build/CMakeCache.txt")"
  fi
fi
if [ -z "${mlir_dir}" ] || [ -z "${llvm_dir}" ]; then
  echo "coverage: MLIR_DIR and LLVM_DIR are neither set nor in build/CMakeCache.txt" >&2
  exit 2
fi

echo "coverage: configuring ${build_dir}"
echo "coverage: threshold is ${threshold} percent"

# --coverage on both compile and link flags. GCC needs it at link time too, to
# pull in libgcov, and a build that has it on only the compile line fails at the
# link with undefined __gcov symbols rather than with anything that names
# coverage.
#
# -O0 rather than the ordinary build type. An optimising compiler merges and
# reorders lines, so a coverage number measured at -O2 is a number about the
# object code rather than about the source the report will quote.
cmake_args=(
  -G Ninja
  -S "${root}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Debug
  -DMLIR_DIR="${mlir_dir}"
  -DLLVM_DIR="${llvm_dir}"
  -DCMAKE_C_FLAGS="-O0 -g --coverage"
  -DCMAKE_CXX_FLAGS="-O0 -g --coverage"
  -DCMAKE_EXE_LINKER_FLAGS="--coverage"
  -DCMAKE_SHARED_LINKER_FLAGS="--coverage"
)
if [ -n "${LLVM_EXTERNAL_LIT:-}" ]; then
  cmake_args+=(-DLLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT}")
elif command -v lit >/dev/null 2>&1; then
  cmake_args+=(-DLLVM_EXTERNAL_LIT="$(command -v lit)")
fi

cmake "${cmake_args[@]}"

jobs="${NPU_COVERAGE_JOBS:-4}"
echo "coverage: building with -j${jobs}"
ninja -C "${build_dir}" "-j${jobs}"

# The suites. Every one of them has to pass, and the absence of `|| true` here is
# the whole of rule 2 above. `check-npu` is the lit suite; the GoogleTest
# binaries are run when they exist, which is how this script keeps working as
# later phases add NPUAllocatorTests, NPUEncodingTests and NPUSimulatorTests
# without a change here. A binary that does not exist is not a failure; a binary
# that exists and fails stops the script, and the trap below says so.
echo "coverage: running the lit suite"
ninja -C "${build_dir}" check-npu

for binary in NPUInterfaceTests NPUTilingTests NPUAllocatorTests \
              NPUEncodingTests NPUSimulatorTests; do
  if [ -x "${build_dir}/bin/${binary}" ]; then
    echo "coverage: running ${binary}"
    "${build_dir}/bin/${binary}"
  else
    echo "coverage: ${binary} is not built in this tree, skipping"
  fi
done

# The filters are Section 17.7's three directories: lib/Dialect, lib/Encoding
# and lib/Simulator. The last two do not exist yet and gcovr is content with a
# filter that matches nothing, so naming all three now means the number covers
# them from the phase they arrive in rather than from the phase somebody
# remembers to add them.
#
# --exclude on the generated .inc files is not a way of flattering the number.
# Those files are tablegen output; they are covered by the tests that exercise
# the operations, and counting thousands of generated lines would make the
# percentage a measure of how much tablegen emitted rather than of what the
# suite reaches.
echo "coverage: collecting with gcovr"
gcovr \
  --root "${root}" \
  --filter "${root}/lib/Dialect/" \
  --filter "${root}/lib/Encoding/" \
  --filter "${root}/lib/Simulator/" \
  --exclude '.*\.inc$' \
  --exclude '.*/build-coverage/.*' \
  --json-summary-pretty \
  --json-summary "${summary_json}" \
  --html-details "${summary_html}" \
  --print-summary

# Rule 3: the number comes out of the JSON, never out of a log line. A gcovr
# that changed its summary schema makes this raise rather than silently produce
# a zero that would pass every threshold below 1.
read -r line_percent branch_percent <<EOF
$(python3 - "${summary_json}" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)

# KeyError here is the point. A schema change must fail loudly rather than be
# papered over with a default of zero, which would pass every threshold this
# script is ever likely to be given.
line = summary["line_percent"]
branch = summary.get("branch_percent", 0.0)
print(f"{line} {branch}")
PYTHON
)
EOF

echo ""
echo "coverage: line coverage   ${line_percent} percent"
echo "coverage: branch coverage ${branch_percent} percent"
echo "coverage: JSON summary    ${summary_json}"
echo "coverage: HTML report     ${summary_html}"
echo ""
echo "coverage: line coverage measures execution, not assertion. The percentage"
echo "coverage: is a project management artifact and not a correctness argument."

# The comparison is done in python rather than in the shell, because the numbers
# are floating point and the shell's test builtin only compares integers.
if ! python3 - "${line_percent}" "${threshold}" <<'PYTHON'
import sys

measured = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if measured >= threshold else 1)
PYTHON
then
  echo ""
  echo "coverage: FAIL. Line coverage ${line_percent} percent is below the" >&2
  echo "coverage: threshold of ${threshold} percent." >&2
  exit 1
fi

echo ""
echo "coverage: PASS. ${line_percent} percent is at or above the threshold of ${threshold} percent."
