# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Whether this environment can reach the external tools, tested in both shapes.

*Added at P11 after CI run 33711091899.* The module under test decides what an
environment can run, so left to itself it is only ever exercised in the shape of
whichever machine ran the suite: the tools present branches cannot execute in the
CI image and the tools absent ones cannot execute here. That made
`python/npu_frontend` the one tree whose coverage depends on the environment,
one commit after its threshold had been set from a measurement that predated the
module.

**Covering the branches is the fix, not moving the gate.** Every function here is
a thin decision over three inputs: `importlib.util.find_spec`, `shutil.which` and
the environment. All three substitute cleanly, so every branch runs in every
environment and the module stops being a source of environment dependent
coverage. That is strictly better than lowering a threshold, because it makes the
gate mean the same thing in both places rather than less in both.

**These tests never consult the real environment.** A test that asserted "the
tools are reachable here" would pass on this machine and fail in CI for a reason
that is not a defect, which is the shape of the thing being fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from npu_frontend import external_tools


@pytest.fixture
def two_tools(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    """A controlled tool table: one library, one with a binary.

    The real table has exactly this shape, and pinning it here keeps these tests
    about the decision rather than about which tools the project happens to use
    this phase.
    """
    table: dict[str, str | None] = {"a_library": None, "a_tool": "a_tool_binary"}
    monkeypatch.setattr(external_tools, "EXTERNAL_TOOLS", table)
    return table


def fake_finder(present: set[str]) -> Any:
    def find_spec(name: str) -> object | None:
        return object() if name in present else None

    return find_spec


def fake_which(present: set[str]) -> Any:
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in present else None

    return which


# ---------------------------------------------------------------------------
# Which tools are missing, which is the decision everything else rests on.
# ---------------------------------------------------------------------------


def test_a_module_that_does_not_import_is_missing(
    two_tools: dict[str, str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", fake_finder(set()))
    monkeypatch.setattr("shutil.which", fake_which(set()))
    assert external_tools.missing_tools() == ["a_library", "a_tool"]
    assert external_tools.tools_reachable() is False


def test_a_module_that_imports_with_its_binary_is_reachable(
    two_tools: dict[str, str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "importlib.util.find_spec", fake_finder({"a_library", "a_tool"})
    )
    monkeypatch.setattr("shutil.which", fake_which({"a_tool_binary"}))
    assert external_tools.missing_tools() == []
    assert external_tools.tools_reachable() is True


def test_an_importable_module_whose_binary_is_absent_is_missing(
    two_tools: dict[str, str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The case D-0046 was made of**, and the one neither environment reaches.

    Accelergy is driven as a subprocess. A machine where the package imports and
    the binary is not on `PATH` has a tool this project cannot run, and calling
    it present is what turned a readable skip into a `FileNotFoundError` in the
    middle of a benchmark. Neither the developer machine nor the CI image is in
    that state, so without this test the branch that catches it never executes
    anywhere.
    """
    monkeypatch.setattr(
        "importlib.util.find_spec", fake_finder({"a_library", "a_tool"})
    )
    monkeypatch.setattr("shutil.which", fake_which(set()))
    missing = external_tools.missing_tools()
    assert missing == ["a_tool (the a_tool_binary binary is not on PATH)"]
    assert external_tools.tools_reachable() is False
    # The message says which half is absent, because "not installed" and
    # "installed and unreachable" want different responses from whoever reads it.
    assert "not on PATH" in missing[0]


def test_a_library_with_no_binary_needs_no_binary(
    two_tools: dict[str, str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCALE-Sim is imported, never executed, so `PATH` is not its problem."""
    monkeypatch.setattr("importlib.util.find_spec", fake_finder({"a_library"}))
    monkeypatch.setattr("shutil.which", fake_which(set()))
    assert external_tools.missing_tools() == ["a_tool"]


def test_a_finder_that_raises_is_read_as_absent(
    two_tools: dict[str, str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`find_spec` raises on a namespace package it cannot resolve.

    Read as absent rather than propagated, because the question this module
    answers is "can this environment run the tool", and an import system that
    cannot answer is a "no" for that purpose. It is the one place here where
    swallowing an exception is the honest reading.
    """

    def raising(name: str) -> object | None:
        raise ValueError(f"{name} is not resolvable")

    monkeypatch.setattr("importlib.util.find_spec", raising)
    assert external_tools.missing_tools() == ["a_library", "a_tool"]


# ---------------------------------------------------------------------------
# The source clone, which is a third thing and not either tool.
# ---------------------------------------------------------------------------


def test_the_source_tree_defaults_and_can_be_named(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(external_tools.SOURCE_TREE_VARIABLE, raising=False)
    assert external_tools.source_tree() == external_tools.DEFAULT_SOURCE_TREE

    monkeypatch.setenv(external_tools.SOURCE_TREE_VARIABLE, str(tmp_path))
    assert external_tools.source_tree() == tmp_path


def test_the_source_tree_is_present_only_when_the_example_is_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The check is the file the exporter reads, not the directory existing.

    A clone that was interrupted has the directory and not the CSV, and this
    project reads the column order out of that CSV. Section 16.3.
    """
    monkeypatch.setenv(external_tools.SOURCE_TREE_VARIABLE, str(tmp_path))
    assert external_tools.source_tree_present() is False

    example = tmp_path / "topologies" / "conv_nets" / "alexnet.csv"
    example.parent.mkdir(parents=True)
    example.write_text("Layer name, IFMAP Height,\n", encoding="utf-8")
    assert external_tools.source_tree_present() is True


def test_the_promise_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(external_tools.EXTERNAL_TOOLS_VARIABLE, raising=False)
    assert external_tools.tools_promised() is False

    monkeypatch.setenv(external_tools.EXTERNAL_TOOLS_VARIABLE, "1")
    assert external_tools.tools_promised() is True

    # Empty is not a promise. A caller that exported the variable and left it
    # blank has said nothing, and reading that as an assertion would turn every
    # skip into a failure on a machine nobody configured.
    monkeypatch.setenv(external_tools.EXTERNAL_TOOLS_VARIABLE, "")
    assert external_tools.tools_promised() is False


# ---------------------------------------------------------------------------
# What the baseline records.
# ---------------------------------------------------------------------------


def test_the_environment_block_is_the_two_facts_the_baseline_compares(
    two_tools: dict[str, str | None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`environment()` is called by the baseline recorder and by nothing else.

    So the suite never reaches it in either shape, and it is exactly the kind of
    line that reads as covered on the machine that wrote it and is not. Both
    states are asserted here rather than whichever one this machine is in.
    """
    monkeypatch.setattr("importlib.util.find_spec", fake_finder(set()))
    monkeypatch.setattr("shutil.which", fake_which(set()))
    monkeypatch.setenv(external_tools.SOURCE_TREE_VARIABLE, str(tmp_path))
    assert external_tools.environment() == {
        "external_tools_reachable": False,
        "scalesim_source_tree_present": False,
    }

    monkeypatch.setattr(
        "importlib.util.find_spec", fake_finder({"a_library", "a_tool"})
    )
    monkeypatch.setattr("shutil.which", fake_which({"a_tool_binary"}))
    example = tmp_path / "topologies" / "conv_nets" / "alexnet.csv"
    example.parent.mkdir(parents=True)
    example.write_text("Layer name,\n", encoding="utf-8")
    assert external_tools.environment() == {
        "external_tools_reachable": True,
        "scalesim_source_tree_present": True,
    }


def test_the_real_table_is_the_three_tools_this_project_installs() -> None:
    """The controlled table above must not drift from the real one.

    Every test here substitutes `EXTERNAL_TOOLS`, so a tool added to the real
    table would be tested by none of them. This is the assertion that notices,
    and it noticed: `zigzag` arriving at P13 turned it red, which is why this
    test is named for a count rather than for a property.

    **`zigzag` is a library with no binary and that is not an omission.** The
    `zigzag-dse` wheel ships no console script, so the only half of the
    reachability question it has is the import, and writing a binary name here
    that does not exist would make the tool permanently unreachable in every
    environment including the one that has it.
    """
    assert external_tools.EXTERNAL_TOOLS == {
        "scalesim": None,
        "accelergy": "accelergy",
        "zigzag": None,
    }
