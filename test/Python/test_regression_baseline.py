# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The diffing half of `scripts/regression_baseline.py`, over synthetic data.

The measuring half takes about a minute and needs a build; it is exercised by
running the script, and the run and its deliberate failure are recorded in
`docs/PHASE_STATE.md`. What is exercised here is the part that decides whether
something moved, because that is the part with the interesting failure mode: a
comparison that silently ignored a field would report no drift forever, and it
would do so while looking exactly like a comparison that works.

The committed baseline is also checked for the shape Section 17.6 requires,
which is the cheap half of the same claim: the schema is versioned, the level
is `-O0` only, and the fields no phase has computed are named as absent rather
than recorded as zero.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import regression_baseline as baseline  # noqa: E402


def minimal() -> dict[str, Any]:
    return {
        "schema_version": baseline.SCHEMA_VERSION,
        "git_sha": "0" * 40,
        "generator_version": "1.0.0",
        "tool_versions": {"python": "3.14.4"},
        "absent_fields": dict(baseline.ABSENT_FIELDS),
        "suites": {
            "check-npu": {
                "passed": 2,
                "failed": 0,
                "skipped": 0,
                "tests": ["a", "b"],
            }
        },
        "levels": [0, 2],
        "cells": [
            {
                "model": "lenet",
                "level": 0,
                "budget": "default",
                "instructions": 25,
                "cycles": 17766.25,
                "max_abs_error_vs_onnxruntime": 3e-08,
                "max_abs_movement_vs_o0": 0.0,
            },
            {
                "model": "lenet",
                "level": 2,
                "budget": "default",
                "instructions": 25,
                "cycles": 17766.25,
                "max_abs_error_vs_onnxruntime": 3e-08,
                "max_abs_movement_vs_o0": 4.47e-08,
            },
        ],
    }


@pytest.fixture
def empty_goldens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the comparison at a directory with no recorded tensors.

    The golden half of `compare` reads the repository's own directory, and a
    test over synthetic cells must not diff against the real goldens.
    """
    directory = tmp_path / "golden"
    directory.mkdir()
    monkeypatch.setattr(baseline, "GOLDEN_DIR", directory)
    return directory


# ---------------------------------------------------------------------------
# The comparison.
# ---------------------------------------------------------------------------


def test_an_unchanged_baseline_reports_no_drift(empty_goldens: Path) -> None:
    recorded = minimal()
    assert baseline.compare(recorded, copy.deepcopy(recorded), {}) == []


def test_an_unknown_schema_version_is_refused_rather_than_guessed(
    empty_goldens: Path,
) -> None:
    """Section 17.6 asks for a loud failure and this is it.

    A version this script does not recognise is not compared with best effort:
    a field that appeared in a later version would be read as a regression from
    zero, which is the exact failure the versioning exists to prevent.
    """
    recorded = minimal()
    recorded["schema_version"] = baseline.SCHEMA_VERSION + 99
    with pytest.raises(baseline.BaselineError, match="does not recognise"):
        baseline.compare(recorded, minimal(), {})


def test_only_the_fields_present_in_both_are_compared(empty_goldens: Path) -> None:
    """What lets a P8 baseline still be readable at P14.

    A field a later phase adds is not a regression from zero, and a field an
    older baseline carries that this one does not is not a deletion. Both are
    the schema growing.
    """
    recorded = minimal()
    current = minimal()
    current["cells"][0]["energy_picojoules"] = 4200.0
    assert baseline.compare(recorded, current, {}) == []

    older = minimal()
    older["cells"][0]["some_field_p6_recorded"] = 7
    assert baseline.compare(older, minimal(), {}) == []


def test_a_moved_metric_is_reported_with_both_values(empty_goldens: Path) -> None:
    recorded = minimal()
    current = minimal()
    current["cells"][0]["cycles"] = 18624.35
    drift = baseline.compare(recorded, current, {})
    assert len(drift) == 1
    assert "17766.25" in drift[0]
    assert "18624.35" in drift[0]
    assert "lenet-O0-default" in drift[0]


def test_a_deleted_test_is_drift(empty_goldens: Path) -> None:
    """The failure mode a suite count alone does not catch.

    A suite that lost one test and gained another has the same pass count. The
    name lists are what turn that into a red run, which is why Section 17.6
    asks for them.
    """
    recorded = minimal()
    current = minimal()
    current["suites"]["check-npu"]["tests"] = ["a", "c"]
    drift = baseline.compare(recorded, current, {})
    assert any("test gone: b" in line for line in drift)
    assert any("test added: c" in line for line in drift)


def test_a_suite_that_stopped_running_is_drift(empty_goldens: Path) -> None:
    current = minimal()
    del current["suites"]["check-npu"]
    drift = baseline.compare(minimal(), current, {})
    assert any("no longer runs" in line for line in drift)


def test_a_cell_that_stopped_being_produced_is_drift(empty_goldens: Path) -> None:
    current = minimal()
    current["cells"] = []
    drift = baseline.compare(minimal(), current, {})
    assert any("no longer produced" in line for line in drift)


def test_a_level_set_change_is_drift_and_says_why(empty_goldens: Path) -> None:
    """*Added at P9 with the `levels` field.*

    A compiler that stopped building a level would produce fewer cells, and the
    per cell comparison would report each of them as "recorded and no longer
    produced" without ever saying that the *set of levels* had changed. One line
    that names the cause above forty that name its consequences is the readable
    report Section 17.6 asks for.
    """
    current = minimal()
    current["levels"] = [0]
    current["cells"] = [cell for cell in current["cells"] if cell["level"] == 0]
    drift = baseline.compare(minimal(), current, {})
    assert any("levels: [0, 2] -> [0]" in line for line in drift)


def test_a_generator_version_change_is_reported_first(empty_goldens: Path) -> None:
    """Because it makes every cell below it a measurement of a different suite."""
    current = minimal()
    current["generator_version"] = "2.0.0"
    drift = baseline.compare(minimal(), current, {})
    assert "GENERATOR_VERSION" in drift[0]


def test_a_moved_golden_is_drift_with_no_tolerance(empty_goldens: Path) -> None:
    """P8's golden tolerance is zero, and a moved bit is a moved bit."""
    recorded = np.array([1.0, 2.0], dtype=np.float32)
    np.save(empty_goldens / "lenet-O0-out0.npy", recorded)

    same = baseline.compare(minimal(), minimal(), {"lenet-O0-out0": recorded})
    assert same == []

    moved = recorded.copy()
    moved[1] = np.nextafter(moved[1], np.float32(3.0))
    drift = baseline.compare(minimal(), minimal(), {"lenet-O0-out0": moved})
    assert any("golden lenet-O0-out0" in line for line in drift)


def test_a_golden_that_is_no_longer_produced_is_drift(empty_goldens: Path) -> None:
    np.save(empty_goldens / "gone-O0-out0.npy", np.zeros(2, dtype=np.float32))
    drift = baseline.compare(minimal(), minimal(), {})
    assert any("no longer produced" in line for line in drift)


# ---------------------------------------------------------------------------
# The committed baseline's shape.
# ---------------------------------------------------------------------------


def committed() -> dict[str, Any]:
    if not baseline.BASELINE_PATH.is_file():
        pytest.skip(f"{baseline.BASELINE_PATH} has not been recorded")
    return json.loads(baseline.BASELINE_PATH.read_text(encoding="utf-8"))


def test_the_committed_baseline_is_at_this_schema_version() -> None:
    assert committed()["schema_version"] == baseline.SCHEMA_VERSION


def test_the_committed_baseline_records_every_level_the_compiler_builds() -> None:
    """Section 17.6's per level fields, which arrived at P9 with the levels.

    Until P9 this read `== {0}` and said that a baseline claiming a level the
    compiler cannot emit is a baseline nobody can re-record. The claim is the
    same and the answer moved: the set is read from the compiler on both sides,
    so a level added without a re-record fails here rather than producing a
    baseline that covers less than the compiler does.
    """
    from npu_frontend import implemented_levels

    recorded = committed()
    assert recorded["levels"] == implemented_levels()
    assert {cell["level"] for cell in recorded["cells"]} == set(implemented_levels())


def test_the_committed_baseline_names_its_absent_fields() -> None:
    """Explicitly absent, with the phase that adds each.

    Not zero, and not missing without explanation. A baseline that claimed
    energy before P11 would be recording a number no phase had computed.

    `per_level` was here until P9 and left with the levels themselves, which is
    the removal Section 17.6 asks for in the same commit that adds the fields.
    """
    absent = committed()["absent_fields"]
    assert set(absent) == {"energy"}
    assert "P11" in absent["energy"]
    for cell in committed()["cells"]:
        assert not any("energy" in key for key in cell)


def test_the_committed_baseline_records_this_phases_numerics_movement() -> None:
    """P9's gate: numerics within 1e-6 at every level, with the largest observed
    movement recorded.

    This is the recording. Every cell carries how far its answer sits from the
    same model's at `-O0` under the same budget, it is zero at `-O0` by
    construction, and the largest of them is inside the band Section 17.6 sets
    for this phase. `docs/BREAKING_CHANGES.md` names the pass it comes from.

    The lower bound matters as much as the upper one: a phase in which every
    cell moved by exactly zero would be a phase whose `-O2` did nothing to any
    model's arithmetic, and the declaration in `BREAKING_CHANGES.md` would be
    describing a movement that did not happen.
    """
    cells = committed()["cells"]
    for cell in cells:
        assert "max_abs_movement_vs_o0" in cell, cell
        if cell["level"] == 0:
            assert cell["max_abs_movement_vs_o0"] == 0.0, cell

    worst = max(float(cell["max_abs_movement_vs_o0"]) for cell in cells)
    assert 0.0 < worst <= 1e-6, (
        f"the largest movement against -O0 is {worst:.3e}. Section 17.6 puts P9 "
        f"in the within 1e-6 class, and a movement of exactly zero everywhere "
        f"would mean no level changed any model's arithmetic."
    )


def test_the_committed_baseline_covers_every_model_at_every_level_and_budget() -> None:
    from npu_frontend import MODELS, implemented_levels

    cells = committed()["cells"]
    assert {(cell["model"], cell["level"], cell["budget"]) for cell in cells} == {
        (name, level, budget)
        for name in MODELS
        for level in implemented_levels()
        for budget in ("default", "tight")
    }


def test_every_committed_golden_has_a_cell() -> None:
    recorded = {path.stem for path in baseline.GOLDEN_DIR.glob("*.npy")}
    if not recorded:
        pytest.skip("no goldens have been recorded")
    cells = committed()["cells"]
    models = {cell["model"] for cell in cells}
    levels = {int(cell["level"]) for cell in cells}
    for name in recorded:
        model, _, rest = name.partition("-O")
        assert model in models
        assert int(rest.split("-")[0]) in levels
