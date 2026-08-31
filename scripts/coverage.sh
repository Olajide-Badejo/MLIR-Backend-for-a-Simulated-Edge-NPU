#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The coverage measurement of Section 17.7, C++ and Python.
#
#   bash scripts/coverage.sh              measure, report, gate at the defaults
#   bash scripts/coverage.sh 85           gate C++ line coverage at 85 percent
#   bash scripts/coverage.sh 85 90        and gate Python at 90 percent
#   bash scripts/coverage.sh --help       this text
#
# Both thresholds default to 0, which is what the activation table of Section
# 19.0 asked for between P2 and P8: the job ran from P2 so the number was
# recorded and its trend visible from the start. **P8 set the real ones**, from
# what P8 measured, and they are passed in by `.github/workflows/ci.yml` rather
# than written here, so that the policy lives with the job and the measurement
# lives with the script. A threshold of 0 still gates on something real, because
# everything below still has to succeed for a number to exist at all.
#
# Measured on 2026-08-31, on this toolchain, at the commit that set the
# thresholds:
#
#   C++ lines     86.0 percent over lib/Dialect, lib/Encoding and lib/Simulator
#   C++ branches  77.0 percent over the same three
#   Python lines  90.3 percent over python/npu_frontend
#
# The C++ threshold is 85, which is Section 17.7's floor, one point below the
# measurement. The Python threshold is 90, which is Section 17.7's rule for it:
# the measured value rounded down to a whole percent and never above what the
# suite achieves.
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

case "${1-}" in
  --help | -h)
    usage
    exit 0
    ;;
esac

threshold="${1:-${NPU_COVERAGE_THRESHOLD:-0}}"
python_threshold="${2:-${NPU_PYTHON_COVERAGE_THRESHOLD:-0}}"

# A threshold is compared numerically later, so it has to be a number now. A
# typo that reached the comparison would be read as 0 by some shells and as an
# error by others, and either way the gate would stop meaning what it says.
for value in "${threshold}" "${python_threshold}"; do
  if ! printf '%s' "${value}" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    echo "coverage: a threshold must be a number, got '${value}'" >&2
    exit 2
  fi
done

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
echo "coverage: C++ line threshold is ${threshold} percent"
echo "coverage: Python line threshold is ${python_threshold} percent"

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
# --gcov-ignore-parse-errors=negative_hits.warn_once_per_file tolerates a known
# gcov defect, gcc bug 68080: gcov's branch counters can go negative under
# counter merging, gcov prints "taken -1", and gcovr's strict parser refuses the
# file. It surfaced nondeterministically in CI on a docs only commit, run
# 32290939959, which is what marks it as a tool artifact rather than a property
# of this code. The warn form keeps the artifact visible in the log while the
# run proceeds; the threshold arm below is unaffected. Recorded as D-0011.
echo "coverage: collecting with gcovr"
gcovr \
  --root "${root}" \
  --filter "${root}/lib/Dialect/" \
  --filter "${root}/lib/Encoding/" \
  --filter "${root}/lib/Simulator/" \
  --exclude '.*\.inc$' \
  --exclude '.*/build-coverage/.*' \
  --gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
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
echo "coverage: C++ line coverage   ${line_percent} percent"
echo "coverage: C++ branch coverage ${branch_percent} percent"
echo "coverage: JSON summary        ${summary_json}"
echo "coverage: HTML report         ${summary_html}"

# Section 17.7: branch coverage is reported alongside line coverage **for the
# allocator and the decoder**, where the error paths matter most. Two more
# gcovr runs over narrower filters, reported and not gated: a per file branch
# threshold would be a second policy to argue about and the section asks for a
# report. The two are named by file rather than by directory, because "the
# allocator" is two files in a directory that holds the lowering as well and
# "the decoder" is two files in a directory that holds the encoder.
# The per file text report goes to a file rather than to /dev/null, because
# gcovr unlinks its output path before writing it and unlinking /dev/null is a
# permission error rather than a no-op. Keeping the file is no loss: it is the
# per line detail behind the summary, in the same directory as the HTML report.
report_branches() {
  local label="$1"
  local slug="$2"
  shift 2
  echo ""
  echo "coverage: branch coverage for ${label}"
  gcovr \
    --root "${root}" \
    "$@" \
    --exclude '.*\.inc$' \
    --exclude '.*/build-coverage/.*' \
    --gcov-ignore-parse-errors=negative_hits.warn_once_per_file \
    --print-summary \
    --txt "${build_dir}/branch-${slug}.txt"
}

report_branches "the allocator" allocator \
  --filter "${root}/lib/Dialect/NPUISA/Transforms/ScratchpadAllocation.cpp" \
  --filter "${root}/lib/Dialect/NPUISA/Transforms/AllocateScratchpad.cpp"

report_branches "the decoder" decoder \
  --filter "${root}/lib/Encoding/Program.cpp" \
  --filter "${root}/lib/Encoding/Validation.cpp"

echo ""
echo "coverage: line coverage measures execution, not assertion. The percentage"
echo "coverage: is a project management artifact and not a correctness argument."

# The comparison is done in python rather than in the shell, because the numbers
# are floating point and the shell's test builtin only compares integers.
above_threshold() {
  python3 - "$1" "$2" <<'PYTHON'
import sys

measured = float(sys.argv[1])
threshold = float(sys.argv[2])
sys.exit(0 if measured >= threshold else 1)
PYTHON
}

if ! above_threshold "${line_percent}" "${threshold}"; then
  echo ""
  echo "coverage: FAIL. C++ line coverage ${line_percent} percent is below the" >&2
  echo "coverage: threshold of ${threshold} percent." >&2
  exit 1
fi

echo ""
echo "coverage: PASS. C++ ${line_percent} percent is at or above the threshold of ${threshold} percent."

# ---------------------------------------------------------------------------
# Python, per Section 17.7's second half.
#
# pytest-cov over python/npu_frontend, which is the one package root this
# project has. `scripts/` is deliberately outside the measurement: those files
# are command line entry points whose bodies are exercised by running them
# rather than by the suite, and folding them in would make the number a
# statement about how many scripts have a `--help` test.
#
# **A threshold that cannot be measured is not a gate**, so a missing dependency
# is a failure when a threshold was asked for and an off line when it was not.
# That is the same rule the CI activation table states: silence and success must
# not look the same.
# ---------------------------------------------------------------------------

python_json="${build_dir}/python-coverage.json"

# ---------------------------------------------------------------------------
# D-0032. **This script builds into build-coverage/ and therefore has to say
# so.** The suite finds a built binary through NPU_BUILD_DIR, then through
# <repo>/build, then through PATH. This script configures build-coverage/ and
# nothing else, and in CI there is no build/ beside it, so without the export
# below every lookup fails: loudly in the frontend, which took the run down at
# collection, and **silently** in the two test modules that skip when a binary
# is missing. The silent half is the dangerous one, because a coverage number
# taken from a run where five tests skipped describes a smaller suite than the
# one anybody thinks was measured.
#
# It worked on a developer machine only because build/ happens to sit beside
# build-coverage/ there. That is the same failure class as D-0030: a result that
# depended on what else was lying around.
# ---------------------------------------------------------------------------
export NPU_BUILD_DIR="${build_dir}"

# The MLIR bindings are resolved the same way and for the same reason.
# conftest.py reads this variable first and a CMake cache second, and the cache
# it should read is the one belonging to the build directory in use. Deriving it
# here means this script does not depend on the job's environment happening to
# set it.
if [ -z "${MLIR_PYTHON_PACKAGES_DIR:-}" ]; then
  bindings="$(cache_value MLIR_PYTHON_PACKAGES_DIR "${build_dir}/CMakeCache.txt")"
  if [ -n "${bindings}" ]; then
    export MLIR_PYTHON_PACKAGES_DIR="${bindings}"
    echo "coverage: MLIR bindings from ${build_dir}/CMakeCache.txt"
  fi
fi

# And the binaries this script just built are asserted to be where it says they
# are, before the suite is asked to find them. A missing one here is this script
# having failed to build it, which is a better sentence than a lookup failure
# inside a test module three layers down.
for required in npu-opt npu-translate npu-sim NPUSimulatorTests; do
  if [ ! -x "${build_dir}/bin/${required}" ]; then
    echo "coverage: FAIL. ${build_dir}/bin/${required} is missing, so the" >&2
    echo "coverage: Python suite would skip or fail on it. That is this script" >&2
    echo "coverage: having not built it rather than a lookup problem. D-0032." >&2
    exit 1
  fi
done

echo ""
if ! python3 -c "import pytest, pytest_cov, torch, onnx, onnxruntime" >/dev/null 2>&1; then
  if [ "${python_threshold}" != "0" ]; then
    echo "coverage: FAIL. A Python threshold of ${python_threshold} was asked" >&2
    echo "coverage: for and the suite's dependencies are not importable, so" >&2
    echo "coverage: there is no number to compare. Install" >&2
    echo "coverage: requirements-lock.txt, or pass 0." >&2
    exit 1
  fi
  echo "coverage: Python coverage skipped, the suite's dependencies are not"
  echo "coverage: importable and the threshold is 0. Nothing was measured."
  exit 0
fi

echo "coverage: measuring Python coverage over python/npu_frontend"
(
  cd "${root}"
  # The whole matrix, not the fast subset. A coverage number taken from the
  # default marker expression would be a number about the subset.
  python3 -m pytest test/Python -q -p no:cacheprovider -m 'slow or not slow' \
    --cov=python/npu_frontend \
    --cov-report="json:${python_json}"
)

read -r python_percent <<EOF
$(python3 - "${python_json}" <<'PYTHON'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)

# KeyError here is the point, for the same reason it is above.
print(summary["totals"]["percent_covered"])
PYTHON
)
EOF

echo ""
echo "coverage: Python line coverage ${python_percent} percent"
echo "coverage: JSON summary         ${python_json}"

if ! above_threshold "${python_percent}" "${python_threshold}"; then
  echo ""
  echo "coverage: FAIL. Python line coverage ${python_percent} percent is below" >&2
  echo "coverage: the threshold of ${python_threshold} percent." >&2
  exit 1
fi

echo ""
echo "coverage: PASS. Python ${python_percent} percent is at or above the threshold of ${python_threshold} percent."
