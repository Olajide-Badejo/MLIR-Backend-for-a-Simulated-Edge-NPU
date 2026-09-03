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
from npu_frontend import external_tools, find_tool
from npu_frontend.diagnostics import VerificationError

#: The variable a caller sets to say which build directory to use. Named here
#: as a constant because three files test for it and a fourth exports it.
BUILD_DIR_VARIABLE = "NPU_BUILD_DIR"


def named_build_directory() -> str | None:
    """The build directory a caller named, or None when nobody did."""
    return os.environ.get(BUILD_DIR_VARIABLE) or None


# ---------------------------------------------------------------------------
# The same policy, for the external cross validation tools. Added at P11 after
# D-0046.
# ---------------------------------------------------------------------------

#: **`BUILD_DIR_VARIABLE`'s rule applied to a second kind of tool**, because
#: D-0046 was D-0032 happening again one layer out. The developer machine has the
#: external tools and the CI image does not, so `pytest.importorskip` alone means
#: the only environment that runs these tests is the one place nobody is
#: watching, which is exactly the shape of D-0040.
#:
#: **The answer itself lives in `npu_frontend.external_tools`** and not here,
#: because `scripts/regression_baseline.py` needs the same answer to record which
#: environment a baseline was taken in. Two copies of it would be the duplication
#: D-0032's fix built a test to hunt for. What stays here is the pytest policy,
#: which is the half that belongs to the suite.
EXTERNAL_TOOLS_VARIABLE = external_tools.EXTERNAL_TOOLS_VARIABLE
EXTERNAL_TOOLS = external_tools.EXTERNAL_TOOLS
external_tools_promised = external_tools.tools_promised
missing_external_tools = external_tools.missing_tools


def require_source_tree() -> None:
    """Skip, or fail, on a missing pinned SCALE-Sim source clone.

    *Added after CI run 33707070166.* The same policy as `require_external_tools`
    below, for a third thing that is neither of the two tools: the clone the
    example topologies are read from, per Section 16.3 and D-0044's deviation.

    **It needs its own guard because it is genuinely independent.** The two tests
    that read those CSVs never import `scalesim`, so every mechanism that reasons
    about the package leaves them alone. They ran on the developer machine and
    skipped in CI, and the rehearsal shim modelled the import and the binary and
    not the clone, so it predicted 998 passed where CI produced 996. Those two
    tests are the whole of that difference.
    """
    if external_tools.source_tree_present():
        return
    where = external_tools.source_tree()
    if external_tools.tools_promised():
        raise AssertionError(
            f"the pinned SCALE-Sim source clone is not at {where}, and "
            f"{EXTERNAL_TOOLS_VARIABLE} is set. A caller that names that "
            f"variable is asserting this environment has the external tools, and "
            f"the clone is where Section 16.3's example topologies are read "
            f"from, so this is a failure rather than a skip.\n\n"
            f"Clone it at the sha in docs/adr/0003-resolved-tool-matrix.md, or "
            f"set {external_tools.SOURCE_TREE_VARIABLE} to where it is."
        )
    pytest.skip(
        f"the pinned SCALE-Sim source clone is not at {where}, and no "
        f"{EXTERNAL_TOOLS_VARIABLE} was set, so nobody claimed this environment "
        f"has it. The pinned wheel ships the package without its topologies/ and "
        f"layouts/ directories, which is D-0044, so this check can only run "
        f"where the clone is."
    )


def require_external_tools() -> None:
    """Skip, or fail, on a missing external tool. The policy of this module.

    - Present: return, and the caller runs the real thing.
    - Absent and **nobody promised them**: skip. That is the CI image, and a
      developer without them, and both are legitimate.
    - Absent and **somebody promised them**: fail. A caller that sets
      `NPU_EXTERNAL_TOOLS` is asserting the tools are there, and a skip in that
      case is the suite quietly doing less than the caller asked for.

    The third branch is the whole point. Without it the external tool tests can
    only ever be skipped in CI, which means the day the image gains the tools
    nothing notices, and the day the tools break nothing notices either.
    """
    absent = missing_external_tools()
    if not absent:
        return
    if external_tools_promised():
        raise AssertionError(
            f"these external tools are not reachable: {absent}, and "
            f"{EXTERNAL_TOOLS_VARIABLE} is set to "
            f"{os.environ[EXTERNAL_TOOLS_VARIABLE]!r}. A caller that names that "
            f"variable is asserting the tools are installed, so this is a "
            f"failure rather than a skip: skipping here would mean the suite ran "
            f"less than the caller asked for, and the external cross validation "
            f"is the part of this phase that only exists when they run.\n\n"
            f"Install them from the pinned shas in "
            f"docs/adr/0003-resolved-tool-matrix.md, and remember "
            f"scripts/patch-scalesim.py."
        )
    pytest.skip(
        f"these external tools are not reachable: {absent}, and no "
        f"{EXTERNAL_TOOLS_VARIABLE} was set, so nobody claimed this environment "
        f"has them. Section 16.4: a missing external tool fails loudly naming "
        f"the dependency when something asks it to run, and a test that has no "
        f"tool to drive skips. Set {EXTERNAL_TOOLS_VARIABLE}=1 to turn this skip "
        f"into a failure."
    )


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
