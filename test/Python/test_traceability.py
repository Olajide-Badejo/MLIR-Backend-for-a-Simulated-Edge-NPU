# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Law 3 of Section 0.2, which is traceability, and its two mechanisms.

Section 20 states the rule and the two tests in one sentence: **the README must
never contain a number that is not reproducible from `experiments/results/`.** A
pytest parses the numeric cells of the README headline table and asserts each
appears in a committed result file, and another asserts the sha in
`report/generated/macros.tex` equals the manifest sha of the committed results.
Discipline is not a mechanism; these are.

**Why two tests and not one.** They fail in different situations and a project
that had only one of them would be exposed in the direction the other covers.

The README test catches a number typed by a person: somebody quotes a figure from
a run they did locally, or from a table that was true last month, and the number
has no provenance at all. The macros test catches the opposite, a number with
provenance that has gone stale: the results are re-recorded, `macros.tex` is not
regenerated, and every macro in it still names the previous measurement while
looking exactly as authoritative as before.

**What "appears in a committed result file" is taken to mean.** The README prints
rounded numbers, so an exact string comparison against a float would fail on
every value that needed rounding. What is asserted instead is that the printed
number is a **faithful rounding of a recorded one**, at the precision it was
printed to. That keeps the test strict about provenance without making it strict
about formatting, which is the distinction the rule is actually about.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import pytest
from npu_frontend.predictions import require_full_history

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RESULTS_DIR: Final[Path] = REPO_ROOT / "experiments" / "results"
README: Final[Path] = REPO_ROOT / "README.md"
MACROS: Final[Path] = REPO_ROOT / "report" / "generated" / "macros.tex"

sys.path.insert(0, str(REPO_ROOT / "experiments"))

import results_to_tex  # noqa: E402

#: The fields a README or a report is allowed to quote. A number that matches
#: nothing in this list is not traceable even if it happens to appear somewhere
#: in a result file, and that is deliberate: the point is that a reader can find
#: the number, not that the number exists.
QUOTABLE: Final[tuple[str, ...]] = (
    "instruction_count",
    "simulation.simulated_cycles",
    "simulation.dram_bytes_read",
    "simulation.dram_bytes_written",
    "simulation.dram_bytes_total",
    "simulation.macs",
    "simulation.effective_macs",
    "simulation.utilization",
    "simulation.spill_count",
    "simulation.fragmentation_ratio",
    "simulation.overlap_fraction",
    "accuracy.max_abs_error_vs_onnxruntime",
    "accuracy.max_rel_error_vs_onnxruntime",
    "accuracy.mean_abs_error_vs_onnxruntime",
    "accuracy.max_abs_movement_vs_o0",
    "accuracy.sqnr_db_vs_fp32_simulated",
    "cell.batch",
    "cell.opt_level",
    "cell.scratchpad_budget_bytes",
    "deltas.instruction_count",
    "deltas.simulated_cycles",
    "deltas.dram_bytes_total",
    "deltas.macs",
)


def committed_cells() -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(RESULTS_DIR.glob("*.json"))
    ]


def quotable_values(cells: list[dict[str, Any]]) -> set[float]:
    """Every number a document is allowed to print, from every committed cell."""
    values: set[float] = set()
    for cell in cells:
        for path in QUOTABLE:
            block: Any = cell
            for part in path.split("."):
                if not isinstance(block, dict):
                    block = None
                    break
                block = block.get(part)
            if isinstance(block, bool) or block is None:
                continue
            if isinstance(block, (int, float)):
                values.add(float(block))
    return values


def headline_table(text: str) -> list[str]:
    """The rows of the README's first markdown table.

    The **first** table, because that is the headline one and the rule is about
    headline numbers. A README that grew a second table of, say, supported opset
    versions should not have its version numbers hunted for in a result file.
    """
    rows: list[str] = []
    seen_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if seen_header and rows:
                break
            continue
        seen_header = True
        if set(stripped) <= set("|-: "):
            continue
        rows.append(stripped)
    return rows


#: A number in a table cell: an integer, a decimal, or scientific notation.
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.])")


def is_a_faithful_rounding(printed: str, recorded: set[float]) -> bool:
    """Whether the printed number is some recorded value, at its own precision.

    `1234.0625` has to match exactly. `5.960e-07` matches a recorded
    `5.9604644775390625e-07`, because that is what printing it to three decimal
    places produces and re-rounding the candidate is how the check stays about
    provenance rather than about formatting.
    """
    value = float(printed)
    for candidate in recorded:
        if candidate == value:
            return True
        if "e" in printed.lower():
            digits = len(printed.lower().split("e")[0].split(".")[-1])
            if f"{candidate:.{digits}e}" == f"{value:.{digits}e}":
                return True
        elif "." in printed:
            digits = len(printed.split(".")[-1])
            if round(candidate, digits) == value:
                return True
    return False


# ---------------------------------------------------------------------------
# The first mechanism: the README's numbers all come from somewhere.
# ---------------------------------------------------------------------------


def test_the_readme_has_a_headline_table() -> None:
    """The table the next test parses, so a deleted table is not a green run.

    Without this, removing the table from the README would make the test below
    pass over an empty list, which is the classic way a parsing test stops
    asserting anything.
    """
    rows = headline_table(README.read_text(encoding="utf-8"))
    assert rows, "the README has no table, so the test that parses it checks nothing"
    assert len(rows) >= 7, (
        f"the headline table has {len(rows)} rows and the model suite has seven, "
        f"so the table no longer covers the suite"
    )
    assert any(_NUMBER.search(row) for row in rows), "the table holds no numbers"


def test_every_number_in_the_readme_table_is_in_a_committed_result() -> None:
    """Section 20's rule, as the mechanism that enforces it.

    The README must never contain a number that is not reproducible from
    `experiments/results/`. This parses the numeric cells and requires each one
    to be a faithful rounding of a field of some committed cell.
    """
    cells = committed_cells()
    if not cells:
        pytest.skip("no results recorded yet; run experiments/run_benchmarks.py")
    recorded = quotable_values(cells)

    unsourced: list[tuple[str, str]] = []
    checked = 0
    for row in headline_table(README.read_text(encoding="utf-8")):
        for match in _NUMBER.finditer(row):
            printed = match.group(1)
            checked += 1
            if not is_a_faithful_rounding(printed, recorded):
                unsourced.append((printed, row))

    assert checked, "no numbers were checked, so this test asserted nothing"
    assert not unsourced, (
        "these numbers are in the README's headline table and in no committed "
        "result file, so they are not reproducible from experiments/results/:\n"
        + "\n".join(f"  {number} in {row}" for number, row in unsourced)
    )


def test_a_number_that_is_in_no_result_is_caught() -> None:
    """The mechanism shown refusing, because a check that has only passed is not one."""
    cells = committed_cells()
    if not cells:
        pytest.skip("no results recorded yet; run experiments/run_benchmarks.py")
    recorded = quotable_values(cells)

    assert not is_a_faithful_rounding("123456789.5", recorded), (
        "a number no cell records was accepted, so the README check would pass "
        "over a figure somebody typed"
    )
    # And a real one, rounded the way the README rounds it, is accepted.
    worst = max(cell["accuracy"]["max_abs_error_vs_onnxruntime"] for cell in cells)
    assert is_a_faithful_rounding(f"{worst:.3e}", recorded)


# ---------------------------------------------------------------------------
# The second mechanism: the generated macros name the measurement they came from.
# ---------------------------------------------------------------------------


def test_the_macros_sha_is_the_manifest_sha_of_the_committed_results() -> None:
    """Section 20's second test, and the reason `report/generated/` is tracked.

    Section 6: `report/generated/` is tracked rather than gitignored, and that is
    what makes law 3 enforceable, because a test can then assert that the sha
    recorded in `macros.tex` equals the manifest sha of the committed results.
    """
    cells = committed_cells()
    if not cells:
        pytest.skip("no results recorded yet; run experiments/run_benchmarks.py")
    assert MACROS.is_file(), (
        "report/generated/macros.tex does not exist. It is tracked rather than "
        "generated at build time precisely so that this test can compare it "
        "against the results."
    )

    found = re.search(
        r"\\newcommand\{\\" + results_to_tex.SHA_MACRO + r"\}\{([0-9a-f]{7,40})\}",
        MACROS.read_text(encoding="utf-8"),
    )
    assert found, f"macros.tex records no {results_to_tex.SHA_MACRO}"
    recorded = found.group(1)

    manifest_shas = {cell["manifest"]["git_sha"] for cell in cells}
    assert len(manifest_shas) == 1, (
        f"the committed results were measured at more than one commit: "
        f"{sorted(manifest_shas)}"
    )
    assert recorded == manifest_shas.pop(), (
        "macros.tex names a different commit than the results it was generated "
        "from, which means the results were re-recorded and the macros were not "
        "regenerated. Every number a report takes from that file would be the "
        "previous measurement wearing the current one's authority. Regenerate "
        "with python experiments/results_to_tex.py."
    )


def test_the_macros_sha_resolves_to_a_real_commit() -> None:
    """A generated number tracing to a commit that does not exist is the failure
    Section 16.1's own resolution test exists to prevent, applied to this file.

    The history guard is D-0041. In a shallow checkout this sha is absent like
    every other historical one, and reporting it as a commit that does not exist
    would blame the generated file for the checkout's depth.
    """
    if not MACROS.is_file():
        pytest.skip("macros.tex not generated yet")
    require_full_history("resolving the commit macros.tex names")
    found = re.search(
        r"\\newcommand\{\\" + results_to_tex.SHA_MACRO + r"\}\{([0-9a-f]{7,40})\}",
        MACROS.read_text(encoding="utf-8"),
    )
    assert found
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{found.group(1)}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"macros.tex names {found.group(1)}, which is not a commit in this "
        f"repository, and the repository is not shallow"
    )


def test_the_macros_file_is_not_stale() -> None:
    """Regenerated and diffed, which catches a moved number as well as a moved sha.

    The sha test above catches a re-recorded suite. This catches the subtler
    case: the results move, the suite is re-recorded at a commit that happens to
    match, and a macro body is now wrong while the sha is right.
    """
    cells = committed_cells()
    if not cells:
        pytest.skip("no results recorded yet; run experiments/run_benchmarks.py")
    assert results_to_tex.main(["--check"]) == 0, (
        "report/generated/macros.tex is stale. Regenerate it with "
        "python experiments/results_to_tex.py, in the same commit as the "
        "results that moved."
    )


def test_the_generated_file_says_it_is_generated() -> None:
    """A tracked generated file that does not say so is one somebody will edit."""
    text = MACROS.read_text(encoding="utf-8")
    assert "GENERATED by experiments/results_to_tex.py" in text
    assert "Do not edit" in text
