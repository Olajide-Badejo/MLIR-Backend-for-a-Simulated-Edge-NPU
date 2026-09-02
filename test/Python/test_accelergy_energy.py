# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The energy path of Section 16.4, and the one law it must never break.

**The law is Section 5.5's**: `macs` is raw, and it is the only MAC figure the
energy path ever sees. `effective_macs` sits in the same block of the same
dictionary, is larger by the reciprocal of the array occupancy, and feeding it to
Accelergy would overstate the energy of exactly the layers this evaluation cares
most about. So the first test here builds a cell whose two differ by a factor of
four and checks which one the answer followed. A code review can miss that; this
cannot.

The rest is the same discipline the SCALE-Sim tests apply to the other tool. A
zero exit is not an answer, because Accelergy's own shipped example crashes on
this install and exits zero. Every parsed coefficient carries the file, the
component, the action and the units it came from. And the sanity check Section
16.4 asks for is run against the published reference table, including where it
does not pass, because a check that is quietly dropped when it fails is worse
than no check.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import accelergy_energy as energy  # noqa: E402


def require_accelergy() -> None:
    pytest.importorskip(
        "accelergy",
        reason=(
            "Accelergy is not installed in this interpreter. Section 16.4's rule "
            "is that a missing external tool fails loudly naming the dependency, "
            "and the module does that; a suite with no tool to drive skips."
        ),
    )


def fake_cell(
    *,
    macs: int,
    effective_macs: float,
    scratchpad_read: int = 1024,
    scratchpad_written: int = 512,
    dram_read: int = 4096,
    dram_written: int = 64,
    budget: int = 65536,
) -> dict[str, Any]:
    """A cell shaped like a result, with the two MAC figures set apart."""
    return {
        "cell": {"name": "synthetic", "scratchpad_budget_bytes": budget},
        "simulation": {
            "macs": macs,
            "effective_macs": effective_macs,
            "scratchpad_elements_read": scratchpad_read,
            "scratchpad_elements_written": scratchpad_written,
            "dram_bytes_read": dram_read,
            "dram_bytes_written": dram_written,
            "simulated_cycles": 1000.0,
        },
    }


# ---------------------------------------------------------------------------
# The law.
# ---------------------------------------------------------------------------


def test_the_energy_path_never_sees_the_scaled_count() -> None:
    """Section 5.5, asserted rather than reviewed.

    Two cells, identical except that one records `effective_macs` four times its
    `macs` and the other records them equal. The action counts have to be the
    same, because the count Accelergy is given is the raw one.
    """
    scaled = energy.counts_for(fake_cell(macs=1000, effective_macs=4000.0))
    unscaled = energy.counts_for(fake_cell(macs=1000, effective_macs=1000.0))
    assert scaled == unscaled
    assert scaled["mac_array"]["mac"] == 1000

    # And the other direction: the count follows `macs` and not the other field.
    assert (
        energy.counts_for(fake_cell(macs=250, effective_macs=4000.0))["mac_array"][
            "mac"
        ]
        == 250
    )


def test_a_partial_dram_access_is_paid_in_full() -> None:
    """Rounded up, because a DRAM cannot fetch part of a word.

    **This test replaced one that asserted a refusal**, and the replacement is
    the finding rather than a relaxation. The first version raised on a byte
    count that was not a whole number of accesses, on the grounds that rounding
    must never invent or discard one. Then `dilated_stack` moved 5004 bytes: its
    buffers are f32 tensors with odd element counts and 1251 floats is 5004
    bytes. The traffic is correct and the refusal was reading a remainder as a
    bug. Rounding up is the physical answer and is the direction that does not
    flatter the result.
    """
    assert energy.dram_accesses(4096) == 512
    assert energy.dram_accesses(4095) == 512
    assert energy.dram_accesses(4089) == 512
    assert energy.dram_accesses(5004) == 626
    assert energy.dram_accesses(0) == 0
    assert energy.dram_accesses(1) == 1

    counts = energy.counts_for(fake_cell(macs=1, effective_macs=1.0, dram_read=5004))
    assert counts["main_memory"]["read"] == 626


def test_a_null_scratchpad_count_raises_rather_than_becoming_zero() -> None:
    cell = fake_cell(macs=1, effective_macs=1.0)
    cell["simulation"]["scratchpad_elements_read"] = None
    with pytest.raises(energy.AccelergyError, match="scratchpad_elements_read"):
        energy.counts_for(cell)


# ---------------------------------------------------------------------------
# The inputs.
# ---------------------------------------------------------------------------


def test_the_scratchpad_depth_is_rounded_up_and_never_down() -> None:
    """An under sized SRAM would report an energy per access the design could not
    achieve, which is the direction that flatters the result."""
    assert energy.scratchpad_depth(65536) == 16384
    assert energy.scratchpad_depth(65540) == 32768
    assert energy.scratchpad_depth(1) == 1
    for byte_count in (6432, 8036, 8480, 194592, 200800, 1048576):
        words = -(-byte_count // 4)
        depth = energy.scratchpad_depth(byte_count)
        assert depth >= words
        assert depth & (depth - 1) == 0


def test_the_architecture_carries_the_pinned_node_and_clock() -> None:
    text = energy.architecture(65536)
    assert f'technology: "{energy.TECHNOLOGY_NODE}"' in text
    assert f"global_cycle_seconds: {energy.GLOBAL_CYCLE_SECONDS}" in text
    for component in energy.COMPONENTS:
        assert f"- name: {component}" in text


# ---------------------------------------------------------------------------
# The tool.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_three_components_are_estimated_and_say_by_what(tmp_path: Path) -> None:
    """Energy and area per component, with the plug in that answered recorded.

    Section 16.1 asks for the registered estimator list because two runs at the
    same Accelergy sha with different plug ins are two different measurements.
    The same reasoning one level down is why the per component estimator is
    recorded: a coefficient whose source is unrecorded cannot be compared with
    the published table it is supposed to have come from.
    """
    require_accelergy()
    cell = fake_cell(macs=100_000, effective_macs=400_000.0)
    estimate = energy.run_accelergy(
        scratchpad_bytes=65536,
        counts=energy.counts_for(cell),
        directory=tmp_path / "run",
    )
    assert set(estimate.energy_per_action_pj) == set(energy.COMPONENTS)
    assert set(estimate.area_um2) == set(energy.COMPONENTS)
    assert estimate.estimators["mac_array"] == "Aladdin_table"
    assert estimate.estimators["scratchpad"] == "CactiSRAM"
    assert estimate.estimators["main_memory"] == "CactiDRAM"
    assert all(
        value > 0.0 for value in estimate.energy_per_action_pj["mac_array"].values()
    )


@pytest.mark.slow
def test_the_cached_table_gives_the_same_answer_as_a_fresh_run(tmp_path: Path) -> None:
    """The linearity the per budget caching rests on, measured rather than assumed.

    `Estimator` runs Accelergy once per distinct scratchpad budget and reuses the
    reference table for every cell at that budget, on the grounds that the
    architecture depends on the budget alone and the energy is linear in the
    action counts. That is a claim about a tool this project did not write, so
    here is a second cell at the same budget with different counts, run fresh,
    compared against the reconstruction from the first cell's table.
    """
    require_accelergy()
    first = fake_cell(macs=100_000, effective_macs=100_000.0)
    second = fake_cell(
        macs=37_713,
        effective_macs=37_713.0,
        scratchpad_read=9_001,
        scratchpad_written=333,
        dram_read=8_192,
        dram_written=128,
    )

    table = energy.run_accelergy(
        scratchpad_bytes=65536,
        counts=energy.counts_for(first),
        directory=tmp_path / "first",
    )
    fresh = energy.run_accelergy(
        scratchpad_bytes=65536,
        counts=energy.counts_for(second),
        directory=tmp_path / "second",
    )

    reconstructed = energy.energy_for(second, table)
    for component in energy.COMPONENTS:
        assert reconstructed.energy_pj_per_component[component] == pytest.approx(
            fresh.energy_pj[component], rel=1e-5
        ), (
            f"{component}: the cached table reconstructs "
            f"{reconstructed.energy_pj_per_component[component]} pJ and a fresh "
            f"Accelergy run says {fresh.energy_pj[component]}. The per budget "
            f"caching assumes the energy is linear in the counts, and this is "
            f"the assumption failing."
        )


@pytest.mark.slow
def test_the_array_area_is_scaled_by_this_project_and_the_energy_is_not(
    tmp_path: Path,
) -> None:
    """One processing element in, `ARRAY_DIM * ARRAY_DIM` of them out, for area only.

    Accelergy is asked about one `fpmac`. The array has 256 of them, so its area
    is 256 times one PE's; its energy is **not**, because the action count is the
    number of multiplies that happened and multiplying it by the array size would
    charge every MAC 256 times.
    """
    require_accelergy()
    from npu_frontend import cost_model

    cell = fake_cell(macs=1000, effective_macs=1000.0)
    estimate = energy.run_accelergy(
        scratchpad_bytes=65536,
        counts=energy.counts_for(cell),
        directory=tmp_path / "run",
    )
    answer = energy.energy_for(cell, estimate)

    positions = cost_model.ARRAY_DIM * cost_model.ARRAY_DIM
    assert answer.area_mm2_per_component["mac_array"] == pytest.approx(
        estimate.area_um2["mac_array"] / 1e6 * positions
    )
    assert answer.energy_pj_per_component["mac_array"] == pytest.approx(
        estimate.energy_per_action_pj["mac_array"]["mac"] * 1000
    )


@pytest.mark.slow
def test_the_sanity_check_of_section_16_4_including_where_it_does_not_pass(
    tmp_path: Path,
) -> None:
    """Per action energy against the published reference table at 45 nm.

    Section 16.4 asks for this check and names the figures: an 8 bit integer add
    about 0.03 pJ, an 8 bit integer multiply about 0.2 pJ, a 32 bit float
    multiply about 3.7 pJ, a 32 KB cache access about 20 pJ, and a 64 bit DRAM
    access on the order of 1.3 to 2.6 nJ.

    **Two of the three hold and one does not, and the one that does not is
    asserted as it is rather than as it should be.** The fp32 MAC coefficient
    comes out at about 10.7 times the sum of the published multiply and add
    figures. That is not a defect in this project and it is not a bound to widen:
    Aladdin's number is a synthesised three stage pipelined unit at a 1 ns clock,
    including the registers, and the published figure is a combinational datapath.
    `docs/NUMBERS.md` carries the account. The assertion here pins the measured
    ratio so that it moving is a failure, which is a stronger check than the
    order of magnitude one it replaces, rather than a weaker one that passes.
    """
    require_accelergy()
    cell = fake_cell(macs=1, effective_macs=1.0, budget=32768)
    estimate = energy.run_accelergy(
        scratchpad_bytes=32768,
        counts=energy.counts_for(cell),
        directory=tmp_path / "run",
    )

    # DRAM: within an order of magnitude of the published range. It holds.
    dram = estimate.energy_per_action_pj["main_memory"]["read"]
    assert energy.REFERENCE_PJ["dram_access_low"] / 10 <= dram
    assert dram <= energy.REFERENCE_PJ["dram_access_high"] * 10

    # A 32 kB scratchpad against the published 32 kB cache access. It holds.
    scratchpad = estimate.energy_per_action_pj["scratchpad"]["read"]
    assert energy.REFERENCE_PJ["cache_32kb_access"] / 10 <= scratchpad
    assert scratchpad <= energy.REFERENCE_PJ["cache_32kb_access"] * 10

    # The MAC, which does not hold. Pinned rather than bounded, and the reason is
    # in the docstring and in docs/NUMBERS.md.
    per_mac = estimate.energy_per_action_pj["mac_array"]["mac"]
    reference = energy.REFERENCE_PJ["fp32_multiply"] + energy.REFERENCE_PJ["fp32_add"]
    assert per_mac == pytest.approx(49.286, rel=1e-4), (
        "the fp32 MAC coefficient moved. It is pinned rather than bounded "
        "because it fails Section 16.4's order of magnitude check at a factor of "
        f"{per_mac / reference:.2f}, and a moving value under a failing check is "
        "two questions at once."
    )
    assert per_mac / reference > 10.0, (
        "the fp32 MAC coefficient now passes Section 16.4's order of magnitude "
        "check. That is good news and it means docs/NUMBERS.md is out of date: "
        "the finding recorded there says it does not."
    )


@pytest.mark.slow
def test_a_run_that_writes_no_output_is_refused(tmp_path: Path) -> None:
    """A zero exit is not an answer from this tool. Proved, not assumed.

    The fault is an action count naming an action no component class defines,
    which Accelergy cannot answer and which it reports by crashing. What matters
    here is not which status it exits with, and that is the point: this project
    does not look. The condition is that the three output files were written and
    carry every component that was asked about, and that condition is what fails.
    """
    require_accelergy()
    with pytest.raises(energy.AccelergyError) as failure:
        energy.run_accelergy(
            scratchpad_bytes=65536,
            counts={
                "mac_array": {"there_is_no_such_action": 1},
                "scratchpad": {"read": 1, "write": 1},
                "main_memory": {"read": 1, "write": 1},
            },
            directory=tmp_path / "broken",
        )
    # The tool's own output travels with the refusal, because Accelergy explains
    # itself better than a paraphrase of it would.
    assert "stdout" in str(failure.value)


@pytest.mark.slow
def test_the_estimator_list_is_read_from_the_tool_and_not_written_here() -> None:
    require_accelergy()
    found = energy.registered_estimators()
    assert found
    assert "Aladdin_table" in found
    assert "CactiSRAM" in found
    assert "CactiDRAM" in found


# ---------------------------------------------------------------------------
# The schema block.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_schema_block_carries_no_null_reason(tmp_path: Path) -> None:
    require_accelergy()
    cell = fake_cell(macs=100_000, effective_macs=400_000.0)
    with tempfile.TemporaryDirectory(prefix="npu-acc-test-") as directory:
        estimator = energy.Estimator(Path(directory))
        answer = estimator.energy(cell)
    block = answer.as_schema(1000.0)
    assert not any(key.endswith("_null_reason") for key in block)
    assert all(value is not None for value in block.values())
    assert block["technology_node"] == energy.TECHNOLOGY_NODE
    # edp is energy times latency, and the latency is cycles times the pinned
    # clock. The unit is picojoule seconds and the arithmetic is checked here so
    # that a reader meeting a bare `edp` can find out what it is.
    assert block["edp"] == pytest.approx(
        block["energy_pj"] * 1000.0 * energy.GLOBAL_CYCLE_SECONDS
    )
