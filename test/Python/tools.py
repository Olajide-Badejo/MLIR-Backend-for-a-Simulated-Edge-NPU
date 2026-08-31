# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""One rule for finding a built binary, and one policy about not finding it.

*Added at P8, after D-0032.*

**Why this file exists rather than a helper in each test module.** Before it,
three places in this suite knew how to find a binary: `npu_frontend.find_tool`,
which the frontend uses, and a hand written `build_directory()` in
`test_refexec_differential.py` and another in `test_cost_model_mirror.py`. The
two copies looked at `$NPU_BUILD_DIR` and then at `<repo>/build`, and at nothing
else. So a caller that pointed the suite at a build directory the copies did not
know about got three different answers, and only one of them was loud about it.

That is not hypothetical. `scripts/coverage.sh` configures `build-coverage/` and
nothing else, and in CI there is no `build/` beside it. The frontend's lookup
raised, which took the whole run down at collection and is how the defect was
found. **The two copies would have skipped**, quietly, and the coverage number
would then have been measured from a run in which the differential suite and the
cost model mirror did not execute. A number measured from a suite that silently
did less is worse than no number.

**The policy, stated once.**

- If the binary is found, return it.
- If it is not found and **nobody named a build directory**, skip. That is a
  developer running the pure Python tests without a build, which is a legitimate
  thing to do and is what the skip has always been for.
- If it is not found and **somebody did name one**, fail. A caller that sets
  `NPU_BUILD_DIR` is asserting that a build is there; a skip in that case is the
  suite quietly doing less than the caller asked for, which is exactly the
  failure this file exists to close.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from npu_frontend import find_tool
from npu_frontend.diagnostics import VerificationError

#: The variable a caller sets to say which build directory to use. Named here
#: as a constant because three files test for it and a fourth exports it.
BUILD_DIR_VARIABLE = "NPU_BUILD_DIR"


def named_build_directory() -> str | None:
    """The build directory a caller named, or None when nobody did."""
    return os.environ.get(BUILD_DIR_VARIABLE) or None


def tool(name: str) -> Path:
    """Locate one built binary, or skip, or fail. See the module docstring.

    The lookup itself is `npu_frontend.find_tool`, which is the one rule: an
    environment variable named after the binary, then `$NPU_BUILD_DIR/bin`, then
    `<repo>/build/bin`, then `PATH`. Nothing here reimplements any of that.
    """
    try:
        return find_tool(name)
    except VerificationError as failure:
        named = named_build_directory()
        if named is not None:
            raise AssertionError(
                f"{name} was not found, and {BUILD_DIR_VARIABLE} is set to "
                f"{named!r}. A caller that names a build directory is asserting "
                f"that the build is there, so this is a failure rather than a "
                f"skip: skipping here would mean the suite quietly ran less "
                f"than the caller asked for, and any coverage number taken from "
                f"the run would describe a smaller suite.\n\n{failure}"
            ) from failure
        pytest.skip(
            f"{name} was not found and no {BUILD_DIR_VARIABLE} was set, so this "
            f"test has no built binary to run against. Build with "
            f"`ninja -C build`, or set {BUILD_DIR_VARIABLE} to a build "
            f"directory.\n\n{failure}"
        )
