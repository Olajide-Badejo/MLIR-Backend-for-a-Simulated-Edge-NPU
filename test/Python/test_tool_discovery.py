# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""There is one rule for finding a built binary, and it is enforced here.

*Added at P8, after D-0032.*

The defect was not that a lookup failed. It was that there were **three** of
them: `npu_frontend.find_tool`, and a hand written `build_directory()` in each
of two test modules. They disagreed about where to look, so a caller that
pointed the suite at `build-coverage/` got one loud failure and five silent
skips, and the run reported success with a coverage number attached to it.

Discipline would be "remember not to write a fourth one". This file is the
mechanism instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import BUILD_DIR_VARIABLE, named_build_directory, tool

TEST_DIR = Path(__file__).resolve().parent

#: The file allowed to know how a binary is found. Everything else asks it.
DISCOVERY_MODULE = "tools.py"

#: Files allowed to locate a **build directory**, with the reason each is.
#:
#: `conftest.py` is here because it answers a different question: which CMake
#: cache carries the MLIR bindings path. That is not a binary, the bindings are
#: not built by this project, and forcing the two to share an implementation
#: would be tidiness rather than correctness. It is still held to the binary
#: path rule below, so a binary lookup added there is caught.
BUILD_DIRECTORY_ALLOWED = {DISCOVERY_MODULE, "conftest.py"}

#: A file deciding for itself where a build directory is.
SECOND_DISCOVERY = re.compile(
    r"def build_directory\(|"
    r"environ(?:\.get\(|\[)[\"']NPU_BUILD_DIR[\"']|"
    r"REPO_ROOT\s*/\s*[\"']build[\"']"
)

#: A file constructing the path to a binary by hand. Forbidden everywhere but
#: the discovery module, `conftest.py` included, because this is the half that
#: actually produced D-0032.
BINARY_PATH = re.compile(r"/\s*[\"']bin[\"']\s*/")


def _scan(pattern: re.Pattern[str], allowed: set[str]) -> list[str]:
    """Every non comment line matching `pattern` outside `allowed`."""
    offenders: list[str] = []
    for path in sorted(TEST_DIR.glob("*.py")):
        if path.name in allowed or path.name == Path(__file__).name:
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{path.name}:{index}: {line.strip()}")
    return offenders


def test_no_test_module_locates_a_build_directory_for_itself() -> None:
    """A fourth copy is a red test rather than a thing somebody notices later.

    `scripts/` is deliberately out of scope. `regression_baseline.py` has its own
    `build_directory()` and legitimately so: it needs a **configured** build
    directory, with a `CMakeCache.txt` it reads and a `test/` subdirectory it
    hands to lit, which is a different question from "where is this binary".
    """
    offenders = _scan(SECOND_DISCOVERY, BUILD_DIRECTORY_ALLOWED)
    assert not offenders, (
        "these files decide for themselves where a build directory is, and "
        f"{sorted(BUILD_DIRECTORY_ALLOWED)} are the only ones allowed to. That "
        "is D-0032: three discoveries disagreed, one failed loudly and two "
        "skipped silently, and a coverage number was taken from the run "
        "anyway.\n\n" + "\n".join(offenders)
    )


def test_no_test_module_builds_a_path_to_a_binary_by_hand() -> None:
    """The half that actually caused the defect, and the stricter of the two.

    Nothing outside `tools.py` may write `<somewhere> / "bin" / name`. That is
    the line both deleted copies had, and it is the line that made them disagree
    with `find_tool` about where to look.
    """
    offenders = _scan(BINARY_PATH, {DISCOVERY_MODULE})
    assert not offenders, (
        f"these files construct a path to a binary by hand, and only "
        f"{DISCOVERY_MODULE} may. Ask `tools.tool(name)` instead.\n\n"
        + "\n".join(offenders)
    )


def test_a_named_build_directory_makes_a_missing_binary_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The policy that closes the silent half.

    A caller that sets the variable is asserting that a build is there. Skipping
    then would be the suite quietly running less than it was asked to, which is
    exactly what happened under `scripts/coverage.sh`.
    """
    monkeypatch.setenv(BUILD_DIR_VARIABLE, str(tmp_path))
    monkeypatch.delenv("NPU_OPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    # The repository's own build/ is the next place `find_tool` looks, and in an
    # ordinary tree it holds a real npu-opt. Moving the root aside is what makes
    # this test about the policy rather than about the developer's checkout.
    monkeypatch.setattr("npu_frontend.builder.REPO_ROOT", tmp_path, raising=True)

    with pytest.raises(AssertionError) as failure:
        tool("npu-opt")
    message = str(failure.value)
    assert BUILD_DIR_VARIABLE in message
    assert str(tmp_path) in message


def test_no_named_build_directory_still_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A developer with no build at all can still run the pure Python tests.

    The skip is not deleted, it is scoped. It applies to the case it was always
    for and not to the case a caller has ruled out.
    """
    monkeypatch.delenv(BUILD_DIR_VARIABLE, raising=False)
    monkeypatch.delenv("NPU_OPT", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("npu_frontend.builder.REPO_ROOT", tmp_path, raising=True)

    with pytest.raises(pytest.skip.Exception):
        tool("npu-opt")


def test_the_discovery_finds_the_real_binaries_in_this_run() -> None:
    """Whatever else is true, this run can see the tools it is testing.

    Trivial when it passes and the whole point when it does not: it is the one
    assertion that would have gone red in CI's coverage job with a sentence
    naming the cause, instead of a collection error three layers down.
    """
    for name in ("npu-opt", "npu-translate", "npu-sim"):
        assert tool(name).is_file(), name


def test_an_empty_variable_counts_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`named_build_directory` is what the policy branches on, so it is checked.

    An exported but empty variable is a shell accident rather than a caller
    naming a directory, and reading it as a name would turn a stray `export
    NPU_BUILD_DIR=` into a suite that fails instead of skipping.
    """
    monkeypatch.setenv(BUILD_DIR_VARIABLE, "")
    assert named_build_directory() is None

    monkeypatch.setenv(BUILD_DIR_VARIABLE, "/somewhere")
    assert named_build_directory() == "/somewhere"

    monkeypatch.delenv(BUILD_DIR_VARIABLE, raising=False)
    assert named_build_directory() is None
