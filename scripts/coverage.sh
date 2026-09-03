#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The coverage measurement of Section 17.7, C++ and Python.
#
#   bash scripts/coverage.sh              measure, report, gate at the defaults
#   bash scripts/coverage.sh 85           gate C++ line coverage at 85 percent
#   bash scripts/coverage.sh 85 93        and gate python/npu_frontend at 93
#   bash scripts/coverage.sh 85 93 14 58  and gate scripts at 14, experiments 58
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
# measurement. Every Python threshold follows Section 17.7's rule for them: the
# measured value rounded down to a whole percent and never above what the suite
# achieves.
#
# **The Python thresholds are set from what CI can execute, not from what this
# machine can**, and the difference is real. Measured 2026-09-03 by running this
# script twice, the whole matrix, subprocess coverage on:
#
#   tree                  CI shape    developer machine   threshold
#   python/npu_frontend   93.2337     93.2337             93
#   scripts               14.6597     14.6597             14
#   experiments           58.4488     73.1302             58
#
# "CI shape" is an interpreter that cannot reach SCALE-Sim or Accelergy, with
# the `--skip-external` path exercised, which is what the image has.
# `experiments/` differs by **14.7 points** between the two, because
# `scalesim_export.py` and `accelergy_energy.py` only execute where the tools do:
# 58.2 against 84.3 and 37.8 against 79.3 per file. Setting the gate from the
# developer figure would make CI red for having less installed, which is not a
# fall in coverage and not something a contributor could act on.
#
# The other two trees measure **identically** in both, which is the useful check
# on that reasoning: only the tree with tool dependent code moves.
#
# `python/npu_frontend` at 93 leaves 0.23 points of headroom, which is tighter
# than the 1.5 the old blended threshold of 90 had. That is deliberate and it is
# the rule rather than a preference: the measured value rounded down to a whole
# percent. **If it goes red, the question is which test stopped running**, not
# what the threshold should be.
#
# **`scripts` at 14 is a real number and a weak gate, and the difference is
# stated where the threshold is.** Five of its seven files measure exactly 0.0,
# because they are driven by shell scripts and CI steps rather than by pytest;
# the figure is close to a statement about `regression_baseline.py` alone. It is
# gated anyway, as a ratchet: it catches that one file's coverage collapsing,
# which is the part of the tree pytest can see.
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

# The second and third Python trees, added at P11. `${2}` stays the frontend's
# threshold, so every existing caller keeps meaning what it meant.
#
# **Three numbers and never one blended figure**, which is the whole reason this
# is three variables rather than one. `python/npu_frontend` is 2262 statements
# and `scripts` is 955: a blend would let the frontend's 93 percent carry the
# scripts tree's 14 and report a middling number nobody could act on, and a fall
# in either would hide behind the other's size.
scripts_threshold="${3:-${NPU_SCRIPTS_COVERAGE_THRESHOLD:-0}}"
experiments_threshold="${4:-${NPU_EXPERIMENTS_COVERAGE_THRESHOLD:-0}}"

# A threshold is compared numerically later, so it has to be a number now. A
# typo that reached the comparison would be read as 0 by some shells and as an
# error by others, and either way the gate would stop meaning what it says.
for value in "${threshold}" "${python_threshold}" "${scripts_threshold}" \
             "${experiments_threshold}"; do
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

# **The execution counters of any previous run are deleted here**, and that is
# D-0037 rather than tidiness. gcov accumulates: a `.gcda` left in place is
# added to by the next run rather than replaced, so a second run of this script
# in the same directory reports the union of both. Two consequences, and the
# second is the one that matters.
#
# The visible one is that a hot inner loop's counter grows without bound and
# eventually passes gcovr's suspicious hits threshold, which is 2^32, at which
# point the collection below aborts with a stack trace about a gcov defect. On
# this machine `lib/Simulator/Kernels.cpp:87` reached 5896524226 across the runs
# of P8, P9 and P9b and did exactly that.
#
# The quiet one is worse, and it is why this is a deletion rather than a wider
# `--gcov-suspicious-hits-threshold`. A line executed by a test that has since
# been deleted keeps the count the run that executed it gave it, so the
# percentage describes the union of every suite this directory has ever run
# rather than the suite that just ran. That is the failure class of D-0030 and
# D-0032 seen from the other side: those were results that depended on what else
# was lying around in CI, and this is one that depends on what is lying around
# locally. **CI never sees it**, because the coverage job checks out a fresh
# tree and has no previous run to inherit, which is why several phases of green
# CI said nothing about it.
#
# Deleting after the build rather than before it is deliberate: ninja may
# recompile a source, and the deletion has to happen once the tree is final.
echo "coverage: clearing the execution counters of any previous run"
find "${build_dir}" -name '*.gcda' -delete

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
# pytest-cov over **three trees, reported and gated separately**:
# `python/npu_frontend`, `scripts` and `experiments`.
#
# **The P8 reasoning for measuring one tree is recorded here and is superseded.**
# It said `scripts/` holds command line entry points whose bodies are exercised
# by running them rather than by the suite, so folding them in would make the
# number a statement about how many scripts have a `--help` test. That was true
# of P8's `scripts/`. It is not a reason to leave 955 statements in `scripts/`
# and 1439 in `experiments/` unmeasured at P11: `regression_baseline.py` alone is
# 393 statements of real logic, and `experiments/` now carries the roofline, the
# SCALE-Sim export and the energy path, which are where this project's published
# numbers come from.
#
# **The P8 observation was right about the shape and that is why each tree gets
# its own number.** Measured on 2026-09-03, five of the seven files in `scripts/`
# are at exactly 0.0 percent, because `dash_lint.py`, `check-reachability.py`,
# `gen-design-decisions.py`, `build-model-ir.py` and `patch-scalesim.py` are
# driven by shell scripts and CI steps rather than by pytest. So the `scripts/`
# figure is close to a statement about `regression_baseline.py` alone, its
# threshold is set from that measurement rather than from an aspiration, and
# this paragraph is what stops the number being read as "86 percent of the
# scripts tree is untested by anything". Most of it is tested by the CI steps
# that run it; none of that is visible to pytest.
#
# **Subprocess coverage is wired**, because it would otherwise be invisible in
# the other direction. `coverage`'s `a1_coverage.pth` calls `process_startup()`
# when `COVERAGE_PROCESS_START` names a config, and `parallel = true` keeps each
# process's data separate until pytest-cov combines them. Without it the
# frontend measured 91.5561 percent and with it 93.3687, so the wiring is worth
# 1.8 points of numbers that were being executed and not counted.
#
# **Stale data is erased first, which is D-0037 on the Python side.** That defect
# was `.gcda` files from a previous build being folded into a new run's number.
# `coverage` has exactly the same trap with `.coverage.*` parallel files, and the
# erase below is what keeps the figure about this run.
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
  if [ "${python_threshold}" != "0" ] || [ "${scripts_threshold}" != "0" ] \
     || [ "${experiments_threshold}" != "0" ]; then
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

coverage_rc="${build_dir}/python-coverage.rc"
coverage_data="${build_dir}/python-coverage-data/.coverage"

echo "coverage: measuring Python coverage over python/npu_frontend, scripts"
echo "coverage: and experiments, reported per tree"

# D-0037 on the Python side. A `.coverage.<host>.<pid>` left by a previous run
# would be combined into this one's number exactly as a stale `.gcda` was.
rm -rf "${build_dir}/python-coverage-data"
rm -f "${root}/.coverage" "${root}"/.coverage.*
mkdir -p "${build_dir}/python-coverage-data"

cat > "${coverage_rc}" <<RC
[run]
parallel = true
data_file = ${coverage_data}
source =
    python/npu_frontend
    scripts
    experiments
omit =
    */__pycache__/*
    scripts/testdata/*
RC

(
  cd "${root}"
  # Subprocess coverage. `coverage`'s installed `.pth` calls `process_startup()`
  # when this names a config, so a script this suite runs as a subprocess is
  # measured rather than reported as dead code.
  export COVERAGE_PROCESS_START="${coverage_rc}"
  # The whole matrix, not the fast subset. A coverage number taken from the
  # default marker expression would be a number about the subset.
  python3 -m pytest test/Python -q -p no:cacheprovider -m 'slow or not slow' \
    --cov-config="${coverage_rc}" --cov \
    --cov-report="json:${python_json}"
)

# Per tree, one line each, and never a blended figure. The three numbers are read
# out in one pass so that a tree present in the config and absent from the report
# is a loud KeyError rather than a silent zero, which is the same rule the C++
# half applies above.
read -r frontend_percent scripts_percent experiments_percent <<EOF
$(python3 - "${python_json}" <<'PYTHON'
import json
import sys

TREES = ("python/npu_frontend", "scripts", "experiments")

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)

# KeyError here is the point, for the same reason it is above.
files = report["files"]

percents = []
for tree in TREES:
    statements = missing = 0
    for name, entry in files.items():
        normalised = name.replace("./", "")
        if not normalised.startswith(tree + "/"):
            continue
        statements += entry["summary"]["num_statements"]
        missing += entry["summary"]["missing_lines"]
    if statements == 0:
        # A tree that measured nothing is a tree whose name stopped matching, and
        # reporting 100 or 0 for it would both be wrong. Section 16.4's rule, in
        # a shell script: an absent number must never look like a measured one.
        raise SystemExit(
            f"coverage: no statements were measured for {tree}. The tree is "
            f"named in the coverage config and nothing in the report matches "
            f"it, so the path moved or the source list is stale."
        )
    percents.append(str((statements - missing) / statements * 100))

print(" ".join(percents))
PYTHON
)
EOF

if [ -z "${experiments_percent}" ]; then
  echo "coverage: FAIL. The per tree figures could not be read from" >&2
  echo "coverage: ${python_json}." >&2
  exit 1
fi

echo ""
echo "coverage: Python line coverage per tree"
echo "coverage:   python/npu_frontend  ${frontend_percent} percent, threshold ${python_threshold}"
echo "coverage:   scripts              ${scripts_percent} percent, threshold ${scripts_threshold}"
echo "coverage:   experiments          ${experiments_percent} percent, threshold ${experiments_threshold}"
echo "coverage: JSON summary         ${python_json}"

# **Every tree is checked before any of them fails the script**, so one run
# reports every tree that is below rather than the first one. A gate that stops
# at the first failure makes a reader run it three times to see three numbers.
python_failed=0
check_tree() {
  if ! above_threshold "${2}" "${3}"; then
    echo "coverage: FAIL. ${1} line coverage ${2} percent is below the" >&2
    echo "coverage: threshold of ${3} percent." >&2
    python_failed=1
  fi
}

echo ""
check_tree "python/npu_frontend" "${frontend_percent}" "${python_threshold}"
check_tree "scripts" "${scripts_percent}" "${scripts_threshold}"
check_tree "experiments" "${experiments_percent}" "${experiments_threshold}"

if [ "${python_failed}" -ne 0 ]; then
  exit 1
fi

echo "coverage: PASS. Every Python tree is at or above its threshold."
