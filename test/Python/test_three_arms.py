# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The parts of `experiments/three_arms.py` that decide what the grid means.

The grid itself is seven models times a budget list times six configurations of
compile and simulate, and it runs in twelve seconds, so it is exercised by
running it. What is tested here is the reading: which arm overtook which, and
whether the shape of the experiment is still the shape Section 13.3 asks for.

**The rule that needs a test most is the one about not placing.** Section 13.3
says a program an arm cannot place is reported as exactly that and never as a
slower one, so a comparison that read an absence as a large number would answer
the question backwards while looking like it worked.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "python"))

import three_arms  # noqa: E402


def run(
    model: str,
    budget: int,
    configuration: str,
    *,
    placed: bool = True,
    cycles: float = 0.0,
) -> three_arms.Run:
    return three_arms.Run(
        model=model,
        budget=budget,
        arm=three_arms.ARM_OF[configuration],
        configuration=configuration,
        placed=placed,
        cycles=cycles,
    )


# ---------------------------------------------------------------------------
# The shape of the experiment.
# ---------------------------------------------------------------------------


def test_all_three_arms_are_present_because_two_is_an_incomplete_experiment() -> None:
    """Section 13.3 says a two arm result is reported as incomplete.

    The cheapest way for that to happen by accident is a configuration table
    that lost an arm, so the table is asserted to carry all three rather than
    trusted to.
    """
    arms = {three_arms.ARM_OF[name] for name in three_arms.ARMS}
    assert arms == {"one, spill", "two, tile", "three, no halo"}


def test_every_configuration_belongs_to_an_arm() -> None:
    assert set(three_arms.ARMS) == set(three_arms.ARM_OF)


def test_arm_one_runs_under_both_spill_heuristics() -> None:
    """Section 13.1 has two, and Section 13.3 calls them two configurations of
    one arm rather than two arms."""
    heuristics = {
        options["spill_heuristic"]
        for name, options in three_arms.ARMS.items()
        if three_arms.ARM_OF[name] == "one, spill"
    }
    assert heuristics == {"longest-range", "cost"}


def test_arm_two_runs_with_fusion_present_and_ablated() -> None:
    """The two configurations Section 13.3's resolution names.

    At `-O2` fusion hides thirty of the forty four compute operations from the
    tiling pass, so a tiling arm run only as the compiler stands would be
    reporting the fusion pass and calling it a tiling result.
    """
    tiling = [name for name, arm in three_arms.ARM_OF.items() if arm == "two, tile"]
    ablated = {three_arms.ARMS[name].get("ablate") for name in tiling}
    assert ablated == {None, "npu-fuse-ops"}


def test_every_model_is_swept_at_its_own_frozen_tight_budget() -> None:
    """The published cell has to appear in the table beside the swept ones.

    Otherwise the experiment answers a question about budgets nobody records
    results at, and the two cells `docs/NUMBERS.md` carries would have nothing
    in this table to be compared against.
    """
    from npu_frontend.model_generator import MODELS

    for model, budgets in three_arms.SWEPT.items():
        assert model in MODELS, model
        assert MODELS[model].tight_budget in budgets, model


# ---------------------------------------------------------------------------
# The reading.
# ---------------------------------------------------------------------------


def test_an_arm_that_places_beats_one_that_does_not_whatever_the_cycles() -> None:
    """The rule Section 13.3 states, and the one worth a test.

    Spilling produces no program at this budget and tiling produces a slow one.
    A comparison that ranked them on cycles would have to invent a cycle count
    for the absence, and whatever it invented would decide the answer.
    """
    rows = [
        run("resnet_block", 6144, "spill-longest-range", placed=False),
        run("resnet_block", 6144, "tile-unfused-recompute", cycles=99999.0),
    ]
    answer = three_arms.crossover(rows, "resnet_block", "tile-unfused-recompute")
    assert answer.budget == 6144


def test_tiling_that_is_merely_slower_does_not_count_as_overtaking() -> None:
    """A tie goes to spilling, because the burden is on the arm adding transfers."""
    rows = [
        run("resnet_block", 6464, "spill-longest-range", cycles=2018.0),
        run("resnet_block", 6464, "tile-fused-recompute", cycles=2660.0),
    ]
    answer = three_arms.crossover(rows, "resnet_block", "tile-fused-recompute")
    assert answer.budget is None
    assert "never costs less" in answer.reason


def test_a_tie_is_not_an_overtake() -> None:
    rows = [
        run("depthwise_separable", 8192, "spill-longest-range", cycles=1324.0),
        run("depthwise_separable", 8192, "tile-fused-recompute", cycles=1324.0),
    ]
    answer = three_arms.crossover(rows, "depthwise_separable", "tile-fused-recompute")
    assert answer.budget is None


def test_the_crossover_is_the_lowest_budget_at_which_tiling_leads() -> None:
    """Reported as the budget the answer changes at, walking down from the top."""
    rows = [
        run("inception_block", 6144, "spill-longest-range", cycles=3799.0),
        run("inception_block", 6144, "tile-fused-recompute", cycles=3395.0),
        run("inception_block", 6000, "spill-longest-range", placed=False),
        run("inception_block", 6000, "tile-fused-recompute", placed=False),
    ]
    answer = three_arms.crossover(rows, "inception_block", "tile-fused-recompute")
    assert answer.budget == 6144
    assert 6144 in answer.checked


def test_a_missing_row_is_skipped_rather_than_read_as_a_zero() -> None:
    """An unmeasured point is not a point where tiling lost.

    `Run` defaults every count to zero, so a comparison that treated a missing
    row as a row would read zero cycles and conclude tiling won everywhere.
    """
    rows = [run("lenet", 194624, "spill-longest-range", cycles=17766.25)]
    answer = three_arms.crossover(rows, "lenet", "tile-fused-recompute")
    assert answer.budget is None
    assert answer.checked == []


# ---------------------------------------------------------------------------
# The allocator's own numbers, off the text it wrote them into.
# ---------------------------------------------------------------------------


def test_the_allocator_attributes_are_read_off_the_function() -> None:
    text = (
        "func.func @main() attributes {npuisa.fragmentation_ratio = 1.0e+00 : f64, "
        "npuisa.scratchpad_budget = 6464 : i64, "
        "npuisa.scratchpad_bytes = 6432 : i64, "
        "npuisa.scratchpad_peak_bytes = 6432 : i64, "
        "npuisa.spill_count = 1 : i64, npuisa.spill_dma_count = 3 : i64} {"
    )
    found = three_arms.attributes(text)
    assert found["scratchpad_peak_bytes"] == 6432
    assert found["spill_count"] == 1
    assert found["spill_dma_count"] == 3
