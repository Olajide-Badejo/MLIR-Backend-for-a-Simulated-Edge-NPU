#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The dash linter of ground rule 3. Section 3.3 of the build specification
# names this file, so this is the entry point, but the rules live in
# dash_lint.py next to it: the verbatim exemption needs a state machine and a
# brace matcher, and writing those in shell would make the one thing that has
# to be exactly right the hardest thing in the repository to read.
#
#   bash scripts/dash-lint.sh              lint every tracked file
#   bash scripts/dash-lint.sh --self-test  check the linter against its fixture
#   bash scripts/dash-lint.sh FILE...      lint the named files
#
# Exits nonzero on any violation, which is what pre-commit and CI read.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the project venv when it is there, so the linter runs on the same
# interpreter as everything else, but do not require it: this script has no
# third party dependencies and CI may call it before any venv exists.
if [ -x "$HOME/npu-venv/bin/python" ]; then
  python="$HOME/npu-venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python="python3"
else
  echo "dash-lint: no python3 on PATH" >&2
  exit 127
fi

exec "$python" "$here/dash_lint.py" "$@"
