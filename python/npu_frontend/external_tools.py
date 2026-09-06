# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Whether this environment can reach the external cross validation tools.

*Added at P11 after D-0046, and moved here from `test/Python/tools.py` when the
regression baseline needed the same answer.* Two readers now ask this question:
the test suite, to decide between skipping and failing, and
`scripts/regression_baseline.py`, to record which environment a baseline was
taken in. A second copy of the answer is exactly the duplication D-0032's fix
built a test to hunt for, so there is one home and two importers, the same
arrangement Section 5.5 uses for the cost model.

**The answer is not "is the package installed".** Accelergy is driven as a
subprocess, so an importable package whose `accelergy` binary is not on `PATH` is
a tool this project cannot run, and calling it present is how D-0046 turned a
readable skip into a `FileNotFoundError` in the middle of a benchmark. A tool
counts as reachable only when both halves are there.

**And the source clone is a third thing.** Section 16.3 says to read the example
topologies from the installed path of the pinned version, and that version's
wheel ships the package without its `topologies/` and `layouts/` directories, so
the examples come from the pinned source clone instead. D-0044 records the
deviation. The clone is a developer machine artefact: an environment can have
SCALE-Sim importable and no clone at all, which is what the CI image would look
like if the tools were ever added to it without the sources.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Final

#: The variable a caller sets to say this environment has the external tools, so
#: a test that needs one must **fail** rather than skip. `BUILD_DIR_VARIABLE`'s
#: rule, for a second kind of tool.
EXTERNAL_TOOLS_VARIABLE: Final[str] = "NPU_EXTERNAL_TOOLS"

#: The variable that names the pinned SCALE-Sim source clone.
SOURCE_TREE_VARIABLE: Final[str] = "NPU_SCALESIM_SOURCE"

#: Where that clone lives on the machine these tools were installed on.
DEFAULT_SOURCE_TREE: Final[Path] = Path.home() / "npu-external" / "scale-sim-v2"

#: Each external tool as an importable module and the binary that has to be on
#: `PATH` for it to answer. `None` means the tool is a library with no binary.
#:
#: **`zigzag` joins here at P13 and it is the first entry installed from a
#: package index rather than from a clone.** Section 16.5 pins the package as
#: `zigzag-dse`, which is not the same name as the project and is not the name it
#: imports under, so the key is the module and the pin lives in
#: `docs/adr/0003-resolved-tool-matrix.md` beside the git shas of the others. It
#: ships no console script, so the binary half is `None` and the reachability
#: question is the import alone, which is the whole of what this project drives
#: it through.
EXTERNAL_TOOLS: Final[dict[str, str | None]] = {
    "scalesim": None,
    "accelergy": "accelergy",
    "zigzag": None,
}


def source_tree() -> Path:
    """The pinned SCALE-Sim source clone, wherever a caller says it is."""
    named = os.environ.get(SOURCE_TREE_VARIABLE)
    return Path(named) if named else DEFAULT_SOURCE_TREE


def source_tree_present() -> bool:
    """Whether the pinned clone's example topologies are readable here."""
    return (source_tree() / "topologies" / "conv_nets" / "alexnet.csv").is_file()


def tools_promised() -> bool:
    """Whether a caller asserted this environment has the external tools."""
    return bool(os.environ.get(EXTERNAL_TOOLS_VARIABLE))


def missing_tools() -> list[str]:
    """Which external tools this environment cannot reach, by name.

    Both halves are required. A module that imports with no binary on `PATH` is
    reported as missing and says which half is absent, because "not installed"
    and "installed and unreachable" want different responses from whoever reads
    the message.
    """
    absent: list[str] = []
    for module, binary in EXTERNAL_TOOLS.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            absent.append(module)
            continue
        if binary is not None and shutil.which(binary) is None:
            absent.append(f"{module} (the {binary} binary is not on PATH)")
    return absent


def tools_reachable() -> bool:
    """Whether every external tool can actually be run here."""
    return not missing_tools()


def environment() -> dict[str, bool]:
    """What a recorded artefact says about the environment it was taken in.

    Recorded in the regression baseline so that a comparison between two
    environments can tell an environment difference from a regression. Section
    17.6's baseline exists to catch a number moving silently, and a suite count
    that differs because one machine has an optional dependency is not a number
    moving: it is two machines running different amounts of the same suite.
    """
    return {
        "external_tools_reachable": tools_reachable(),
        "scalesim_source_tree_present": source_tree_present(),
    }
