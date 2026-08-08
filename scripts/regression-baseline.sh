#!/usr/bin/env bash
#
# regression-baseline.sh
#
# The safety net for the upgrade work. Builds the project, runs all four test
# suites plus dash-lint, compiles and simulates every LeNet cell in the
# optimization level times scratchpad budget grid, and records the lot as a
# machine readable baseline in test/baseline/baseline.json plus golden output
# tensors in test/baseline/golden/.
#
#   scripts/regression-baseline.sh           record a new baseline
#   scripts/regression-baseline.sh --check   re-measure and diff, nonzero on drift
#
# Every upgrade phase gate re-runs the --check form. A phase whose baseline
# regressed is not complete unless the regression is deliberate and written up
# in docs/BREAKING_CHANGES.md.
#
# The real work is in scripts/regression_baseline.py; this wrapper only finds
# the virtualenv and the prebuilt toolchain. Override either with the
# environment variables below.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"

VENV="${NPU_VENV:-$HOME/npu-venv}"
LIT="${NPU_LIT:-$HOME/llvm-project/build/bin/llvm-lit}"
BUILD_DIR="${NPU_BUILD_DIR:-$repo/build}"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "regression-baseline: no virtualenv at $VENV, using the ambient python" >&2
fi

exec python "$repo/scripts/regression_baseline.py" \
  --build-dir "$BUILD_DIR" \
  --lit "$LIT" \
  "$@"
