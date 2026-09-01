# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The benchmark harness of Sections 16.1 and 16.2, and the gates it enforces.

Three of P10's gate clauses live in `experiments/run_benchmarks.py` rather than
in the compiler, and each is tested here on both sides.

**Every ablatable `-O2` pass has an ablation row at every budget, with the
ablatable set read from the driver at run time.** Tested by recomputing the cell
set from the compiler's own description and comparing, so a pass added to `-O2`
and marked ablatable makes this red rather than quietly leaving the table one row
short.

**The suite runtime is measured against the 90 minute budget and the run fails if
it exceeds it.** Tested by arithmetic in the fast subset and by a real run with
the budget set to zero in the slow one, because a gate whose failure path has
never executed is a gate nobody has seen work.

**A determinism test asserts that re-running a cell produces a byte identical
result file apart from the timestamp and the timing object.** Section 16.1 says
that is what makes the split between counted and timed metrics meaningful rather
than decorative, so the test compares the two halves separately: everything
outside `timing` and `timestamp` for equality, and the timing object only for
being present and shaped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from npu_frontend import ablatable_passes, implemented_levels
from npu_frontend.model_generator import DEFAULT_BUDGET, MODELS, TIGHT_BUDGETS
from npu_frontend.results import RESULTS_DIR

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_benchmarks  # noqa: E402


@pytest.fixture(scope="module")
def built() -> None:
    """The harness reads the compiler's description, so it needs the compiler."""
    tool("npu-opt")


# ---------------------------------------------------------------------------
# The cell set.
# ---------------------------------------------------------------------------


def test_the_planned_cells_are_the_computed_cross_product(built: None) -> None:
    """A silently dropped cell fails, which is Section 17.4's rule for a matrix.

    The expectation is recomputed here from the registries and the compiler, not
    copied from the harness, so a filter applied inside `planned_cells` is a red
    test rather than a quieter suite.
    """
    cells = run_benchmarks.planned_cells()
    levels = implemented_levels()
    ablatable = ablatable_passes(2)

    expected: set[str] = set()
    for model in MODELS:
        declared = int(MODELS[model].input_shape[0])
        combinations = [("default", batch) for batch in (1, 4)]
        combinations.append(("tight", declared))
        for budget, batch in combinations:
            for level in levels:
                expected.add(f"{model}-O{level}-{budget}-n{batch}-fp32-normal")
        for budget in ("default", "tight"):
            for ablated in ablatable:
                expected.add(
                    f"{model}-O2-{budget}-n{declared}-fp32-normal-ablate-{ablated}"
                )

    assert {cell.name for cell in cells} == expected
    assert len({cell.name for cell in cells}) == len(cells), "a cell is duplicated"


def test_the_counts_are_what_adr_0010_says(built: None) -> None:
    """The arithmetic, written out once so a change to the rule is visible.

    Seven models times three levels times three budget and batch combinations is
    63 benchmark cells; eight ablatable passes times seven models times two
    budgets is 112 ablation cells; 175 in total. Section 2's 238 assumed eleven
    ablatable passes and a free budget times batch product, and both differences
    are recorded in `docs/adr/0010` and in the harness docstring.
    """
    cells = run_benchmarks.planned_cells()
    ablation = [cell for cell in cells if cell.key.ablated_pass is not None]
    benchmark = [cell for cell in cells if cell.key.ablated_pass is None]

    assert len(ablatable_passes(2)) == 8
    assert len(benchmark) == len(MODELS) * len(implemented_levels()) * 3 == 63
    assert len(ablation) == 8 * len(MODELS) * 2 == 112
    assert len(cells) == 175


def test_every_ablatable_pass_has_a_row_at_every_budget(built: None) -> None:
    """The gate clause, and the sweep is the driver's set rather than a list here.

    Section 16.2 asks for the rows at every budget in those words, and gives the
    reason: passes can behave oppositely at a tight budget and a table that only
    reports the generous one hides it.
    """
    cells = run_benchmarks.planned_cells()
    for model in MODELS:
        for budget in ("default", "tight"):
            rows = {
                cell.key.ablated_pass
                for cell in cells
                if cell.key.model == model
                and cell.key.scratchpad_budget == budget
                and cell.key.ablated_pass is not None
            }
            assert rows == set(ablatable_passes(2)), (
                f"{model} at the {budget} budget is missing ablation rows for "
                f"{sorted(set(ablatable_passes(2)) - rows)}"
            )


def test_a_tight_cell_runs_at_the_models_declared_batch(built: None) -> None:
    """ADR 0010, asserted rather than left in prose.

    The measurement behind it is in that record: six of the seven models do not
    allocate at batch 4 under their recorded tight budget, because the tight
    budget is the smallest budget at which **that program** allocates and a model
    at batch 4 is a different program.
    """
    for cell in run_benchmarks.planned_cells():
        declared = int(MODELS[cell.key.model].input_shape[0])
        if cell.key.scratchpad_budget == "tight":
            assert cell.key.batch == declared
            assert cell.budget_bytes == TIGHT_BUDGETS[cell.key.model]
        else:
            assert cell.budget_bytes == DEFAULT_BUDGET
        if cell.key.ablated_pass is not None:
            assert cell.key.batch == declared, (
                "an ablation row at a batch its baseline was not measured at "
                "would be a subtraction with one operand"
            )


def test_every_ablation_row_has_its_baseline_in_the_run(built: None) -> None:
    names = {cell.name for cell in run_benchmarks.planned_cells()}
    checked = 0
    for cell in run_benchmarks.planned_cells():
        baseline = cell.key.baseline_cell
        if baseline is None:
            continue
        assert baseline in names, f"{cell.name} names a baseline outside the run"
        checked += 1
    assert checked == 112


def test_an_unknown_model_is_refused(built: None) -> None:
    with pytest.raises(run_benchmarks.BenchmarkError) as failure:
        run_benchmarks.planned_cells(["not_a_model"])
    assert "not models of the suite" in str(failure.value)


# ---------------------------------------------------------------------------
# The 90 minute budget, as a gate.
# ---------------------------------------------------------------------------


def test_the_budget_defaults_to_section_2s_ninety_minutes() -> None:
    assert run_benchmarks.DEFAULT_BUDGET_MINUTES == 90.0


def test_the_budget_verdict_passes_inside_and_fails_outside() -> None:
    """The arithmetic of the gate, without paying for a run to exercise it."""
    assert run_benchmarks.budget_verdict(60.0, 90.0) is None
    assert run_benchmarks.budget_verdict(90.0 * 60.0, 90.0) is None

    over = run_benchmarks.budget_verdict(90.0 * 60.0 + 1.0, 90.0)
    assert over is not None
    assert "against a budget of 90" in over
    assert "Do not widen the budget to make this pass" in over


@pytest.mark.slow
def test_the_run_fails_when_it_exceeds_its_budget(built: None, tmp_path: Path) -> None:
    """The gate's failure path, executed.

    A budget of zero minutes is exceeded by any run at all, which is the cheapest
    way to reach the branch without making the suite slow on purpose. What is
    asserted is the exit code and the message, because the gate's job is to fail
    the run rather than to print a warning somebody scrolls past.
    """
    status = run_benchmarks.main(
        [
            "--models",
            "conv_bn_relu_stack",
            "--results",
            str(tmp_path),
            "--budget-minutes",
            "0",
        ]
    )
    assert status == 1, "a suite over its budget must fail the run"
    assert list(tmp_path.glob("*.json")), (
        "the results are written before the budget is checked, so a run over "
        "budget still leaves the measurement behind for whoever has to decide"
    )


@pytest.mark.slow
def test_a_rerun_is_byte_identical_apart_from_the_timestamp_and_the_timing(
    built: None, tmp_path: Path
) -> None:
    """Section 16.1's determinism test, which is what makes the split meaningful.

    Everything outside `timing` and `manifest.timestamp` compares for equality,
    with no tolerance. The timing object is checked for shape rather than for
    value, which is the whole reason it is a separate object: a wall clock that
    reproduced exactly would mean it was not being measured.
    """
    one_model = ["--models", "conv_bn_relu_stack", "--results", str(tmp_path)]
    assert run_benchmarks.main(one_model) == 0
    first = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.glob("*.json"))
    }
    assert run_benchmarks.main([*one_model, "--force"]) == 0
    second = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.glob("*.json"))
    }

    assert set(first) == set(second)
    moved: list[str] = []
    for name in first:
        before, after = _stripped(first[name]), _stripped(second[name])
        if before != after:
            moved.append(name)
    assert not moved, f"these cells did not reproduce: {moved}"

    for name, cell in second.items():
        assert cell["timing"]["compile_ms"]["n_trials"] >= 10
        assert cell["manifest"]["timestamp"] != first[name]["manifest"]["timestamp"]


def _stripped(cell: dict[str, Any]) -> dict[str, Any]:
    """One cell with the two things a rerun is allowed to move removed.

    The timestamp and the timing objects, which are exactly the two Section 16.1
    names. `run_order_position` is deliberately **not** stripped: the order is
    shuffled under a seed this comparison holds fixed, so the position is
    reproducible too, and stripping it would hide a seed that had stopped
    determining the order.
    """
    copy = json.loads(json.dumps(cell))
    copy.pop("timing", None)
    copy.get("manifest", {}).pop("timestamp", None)
    for entry in copy.get("passes", []):
        entry.pop("timing", None)
    return copy


# ---------------------------------------------------------------------------
# The two findings the harness fails a run for.
# ---------------------------------------------------------------------------


def test_an_ablation_outside_the_band_fails_the_run() -> None:
    """Section 16.2: it means the pass is load bearing for correctness.

    Driven on a doctored pair of cells rather than by breaking a pass, because
    the check is what is under test and a pass whose removal really did break the
    numerics would be a defect rather than a fixture.
    """
    from npu_frontend.tolerances import ABSOLUTE_TOLERANCE

    inside = _cell("lenet-O2-default-n1-fp32-normal", None, ABSOLUTE_TOLERANCE / 2.0)
    outside = _cell(
        "lenet-O2-default-n1-fp32-normal-ablate-cse", "cse", ABSOLUTE_TOLERANCE * 10.0
    )
    findings = run_benchmarks.check_ablation_numerics(
        {inside["cell"]["name"]: inside, outside["cell"]["name"]: outside}
    )
    assert len(findings) == 1
    assert "cse" in findings[0]
    assert "load bearing for correctness rather than for performance" in findings[0]

    ok = _cell(
        "lenet-O2-default-n1-fp32-normal-ablate-cse", "cse", ABSOLUTE_TOLERANCE / 2.0
    )
    assert run_benchmarks.check_ablation_numerics({ok["cell"]["name"]: ok}) == []


def test_an_ablation_whose_baseline_is_missing_is_refused() -> None:
    orphan = _cell("lenet-O2-default-n1-fp32-normal-ablate-cse", "cse", 0.0)
    with pytest.raises(run_benchmarks.BenchmarkError) as failure:
        run_benchmarks.fill_deltas({orphan["cell"]["name"]: orphan})
    assert "subtraction with one operand" in str(failure.value)


def _cell(name: str, ablated: str | None, distance: float) -> dict[str, Any]:
    """The two fields these checks read, and nothing else.

    Deliberately not a whole schema instance. A fixture that had to be a valid
    result would go stale the next time the schema grew a field, and neither
    function under test reads anything else.
    """
    return {
        "cell": {
            "name": name,
            "ablated_pass": ablated,
            "baseline_cell": (
                None if ablated is None else name.split("-fp32-")[0] + "-fp32-normal"
            ),
        },
        "accuracy": {"max_abs_error_vs_onnxruntime": distance},
    }


# ---------------------------------------------------------------------------
# What the committed run recorded about itself.
# ---------------------------------------------------------------------------


def test_the_committed_run_records_its_own_measured_cost() -> None:
    """The number that replaces Section 2's 15 second planning figure.

    Recorded in the tree rather than only printed, because the handoff quotes it
    and a number that lives in a terminal scrollback is a number nobody can check.
    """
    runtime = run_benchmarks.runtime_path(RESULTS_DIR)
    if not runtime.is_file():
        pytest.skip("no run recorded yet; run experiments/run_benchmarks.py")
    recorded = json.loads(runtime.read_text(encoding="utf-8"))
    assert recorded["cells_total"] == 175
    assert recorded["seconds_per_cell"] > 0.0
    assert recorded["budget_minutes"] == 90.0
    assert (
        recorded["suite_seconds"] < recorded["budget_minutes"] * 60.0
    ), "the committed run is outside the budget it was measured against"
