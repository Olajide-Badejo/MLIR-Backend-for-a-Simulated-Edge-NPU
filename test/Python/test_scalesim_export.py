# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The SCALE-Sim export of Section 16.3, and the things about SCALE-Sim that bite.

Three groups of test here and they are worth telling apart.

**The export describes this program.** The column order is checked against the
pinned version's own example file rather than against memory, the MAC count the
topology implies is checked against the MAC count this project charged, and the
mappings Section 16.3 spells out for grouping, dilation and matmul each have a
case.

**The tool's failure modes, which are not the ones a caller would guess.** A
missing input file makes SCALE-Sim print an error and exit **zero**, so a test
proves that a zero exit with no report is refused rather than read as an answer.
This is the D-0040 shape of defect living inside a dependency, and the only
defence is to check the answer rather than the status.

**The inputs this project does not understand.** SCALE-Sim v3 requires a layout
file and then ignores it while custom layout is off. "Ignores it" is a claim, so
it is measured: the same cell runs twice with two different layout rows and the
cycle counts have to be identical. An input handed to a tool whose answer this
project publishes is exactly the kind of thing that turns out to matter after the
number is in a report.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
from npu_frontend import npuisa_walk
from npu_frontend.results import RESULTS_DIR, load_result

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import roofline  # noqa: E402
import scalesim_export as export  # noqa: E402

#: Where the pinned SCALE-Sim source tree is, for the files the wheel does not
#: ship. Section 16.3 says to read the example topologies from the installed path
#: of the pinned version; the pinned version's wheel carries the package and not
#: its `topologies/` or `layouts/` directories, so the source clone is where they
#: are. That deviation is recorded in `docs/DEFECT_LOG.md` D-0044 rather than
#: papered over by writing the column order out from memory.
SOURCE_TREE = Path(
    os.environ.get(
        "NPU_SCALESIM_SOURCE", str(Path.home() / "npu-external" / "scale-sim-v2")
    )
)


def require_scalesim() -> None:
    pytest.importorskip(
        "scalesim",
        reason=(
            "SCALE-Sim is not installed in this interpreter. Section 16.4's rule "
            "is that a missing external tool fails loudly naming the dependency, "
            "and the exporter does that; a test suite that has no tool to drive "
            "skips rather than asserting the refusal message twice."
        ),
    )


def cell(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{name} is not a committed cell in this checkout")
    return load_result(path)


def walked(result: dict, directory: Path) -> list[npuisa_walk.Operation]:
    tool("npu-opt")
    text = roofline.Compiler(directory).allocated_ir(result)
    operations = npuisa_walk.attribute_transfers(npuisa_walk.walk(text), text)
    npuisa_walk.check_against_result(operations, result)
    return operations


# ---------------------------------------------------------------------------
# The export describes this program.
# ---------------------------------------------------------------------------


def test_the_column_order_is_the_pinned_versions_own() -> None:
    """Copied from the example topology, not composed from the field names.

    Section 16.3 is explicit: read the bundled example topologies from the
    installed path of the pinned version and copy their column order exactly, do
    not invent a column order from memory. This is the assertion that keeps that
    true after the sentence has been forgotten.
    """
    example = SOURCE_TREE / "topologies" / "conv_nets" / "alexnet.csv"
    if not example.is_file():
        pytest.skip(f"the pinned SCALE-Sim source tree is not at {SOURCE_TREE}")
    header = example.read_text(encoding="utf-8").splitlines()[0].strip()
    assert export.TOPOLOGY_HEADER == header


def test_the_layout_header_is_the_pinned_versions_own() -> None:
    example = SOURCE_TREE / "layouts" / "conv_nets" / "test.csv"
    if not example.is_file():
        pytest.skip(f"the pinned SCALE-Sim source tree is not at {SOURCE_TREE}")
    header = example.read_text(encoding="utf-8").splitlines()[0].strip()
    assert export.LAYOUT_HEADER == header


def test_a_matmul_maps_to_one_gemm_shaped_row() -> None:
    """Section 16.3's mapping, with the reading a column name does not give.

    IFMAP `M` by 1, filter 1 by 1, `K` channels, `N` filters, stride 1. None of
    `M`, `K` or `N` is an image extent, and two of them sit in columns named for
    one, which is why the mapping is stated in the module docstring and asserted
    here.
    """
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(cell("lenet-O2-default-n1-fp32-normal"), Path(directory))
    topology = export.export(operations)
    matmuls = [operation for operation in operations if operation.op == "matmul"]
    assert matmuls, "lenet has three matmuls and the walk found none"

    by_name = {row.name: row for row in topology.rows}
    for operation in matmuls:
        row = by_name[operation.name]
        assert row.ifmap_height == operation.result.shape[0]
        assert row.ifmap_width == 1
        assert row.filter_height == 1
        assert row.filter_width == 1
        assert row.channels == operation.operands[0].shape[1]
        assert row.num_filters == operation.result.shape[1]
        assert row.stride == 1
        assert export.row_macs(row) == operation.macs


def test_a_grouped_convolution_becomes_one_row_per_group() -> None:
    """`<layer>_g<i>`, each carrying `C / group` channels and `O / group` filters.

    `depthwise_separable` is the model that exercises it: its first convolution
    has `group == C == 8`, so eight rows each presenting a single column to a
    sixteen column array.
    """
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(
            cell("depthwise_separable-O2-default-n1-fp32-normal"), Path(directory)
        )
    topology = export.export(operations)

    grouped = [
        operation
        for operation in operations
        if operation.op == "conv2d" and operation.attributes.get("group", 1) > 1
    ]
    assert len(grouped) == 1
    operation = grouped[0]
    group = int(operation.attributes["group"])

    rows = [row for row in topology.rows if row.source_position == operation.position]
    assert len(rows) == group
    assert {row.name for row in rows} == {
        f"{operation.name}_g{index}" for index in range(group)
    }
    assert all(row.channels == operation.operands[0].shape[1] // group for row in rows)
    assert all(row.num_filters == operation.result.shape[1] // group for row in rows)
    # And the whole layer's arithmetic is preserved across the split.
    assert sum(export.row_macs(row) for row in rows) == operation.macs


def test_dilation_is_an_approximation_and_says_so() -> None:
    """The effective extent `d * (k - 1) + 1`, never presented as exact."""
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(
            cell("dilated_stack-O2-default-n1-fp32-normal"), Path(directory)
        )
    topology = export.export(operations)

    dilated = [entry for entry in topology.approximations if entry.reason == "dilation"]
    assert dilated, "dilated_stack dilates and the exporter recorded no approximation"
    for entry in dilated:
        detail = entry.detail
        expected = [
            detail["dilations"][axis] * (detail["true_filter"][axis] - 1) + 1
            for axis in (0, 1)
        ]
        assert detail["effective_filter"] == expected

    # The dilated rows carry the effective extent and the matched rows carry the
    # true one, which is what makes the second run a measurement of the
    # approximation rather than a repeat of the first.
    positions = {entry.position for entry in dilated}
    for position in positions:
        effective = [row for row in topology.rows if row.source_position == position]
        true = [
            row for row in topology.undilated_rows if row.source_position == position
        ]
        assert any(
            a.filter_height != b.filter_width or a.filter_height != b.filter_height
            for a, b in zip(effective, true, strict=True)
        )


def test_everything_without_a_systolic_representation_is_skipped_with_its_cost() -> (
    None
):
    """Section 16.3: skipped and returned in a list, with the cycles it carried.

    The cycles are the point. A `skipped` list of names says the export dropped
    something; a `skipped` list with the analytical cycles beside each name is
    what lets the divergence read as a sum.
    """
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(cell("lenet-O2-default-n1-fp32-normal"), Path(directory))
    topology = export.export(operations)

    exported = {row.source_position for row in topology.rows}
    skipped = {entry.position for entry in topology.skipped}
    assert exported | skipped == {operation.position for operation in operations}
    assert not exported & skipped
    assert all(entry.analytical_cycles > 0.0 for entry in topology.skipped)
    assert {entry.op for entry in topology.skipped} >= {
        "relu",
        "pool_max",
        "dma_load",
        "reshape",
    }


def test_a_topology_that_describes_other_arithmetic_is_refused() -> None:
    """The check that makes a divergence figure mean anything, proved red.

    A row with the wrong IFMAP height describes a layer with a different number
    of output positions, so SCALE-Sim would return the cycle count of a layer
    this program does not contain. That has to be a refusal rather than a
    divergence with a plausible size and no cause.
    """
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(
            cell("resnet_block-O2-default-n1-fp32-normal"), Path(directory)
        )
    topology = export.export(operations)

    broken = export.Topology(
        rows=list(topology.rows),
        skipped=list(topology.skipped),
        approximations=list(topology.approximations),
        undilated_rows=[
            (
                row
                if index != 0
                else export.TopologyRow(
                    **{**row.__dict__, "ifmap_height": row.ifmap_height + row.stride}
                )
            )
            for index, row in enumerate(topology.undilated_rows)
        ],
    )
    with pytest.raises(export.ScaleSimError, match="same arithmetic"):
        export.check_macs(broken, operations)


# ---------------------------------------------------------------------------
# The architecture configuration.
# ---------------------------------------------------------------------------


def test_the_configuration_restates_the_cost_model_constants() -> None:
    """Section 16.3: the architecture configuration restates them from one source.

    Including the unit conversion, which is the part that would be silent if it
    were wrong: SCALE-Sim's `Bandwidth` is words per cycle and this project's
    constant is bytes per cycle, so a figure handed over unconverted would be a
    four times bandwidth error that every downstream number would absorb.
    """
    import configparser

    from npu_frontend import cost_model

    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read_string(export.architecture_config(65536))

    architecture = parser["architecture_presets"]
    assert int(architecture["ArrayHeight"]) == cost_model.ARRAY_DIM
    assert int(architecture["ArrayWidth"]) == cost_model.ARRAY_DIM
    assert architecture["Dataflow"] == "ws"
    assert float(architecture["Bandwidth"]) == (
        cost_model.DRAM_BANDWIDTH_BYTES_PER_CYCLE / npuisa_walk.ELEMENT_BYTES
    )
    assert int(architecture["IfmapSramSzkB"]) == 65536 // 1024
    assert parser["run_presets"]["InterfaceBandwidth"] == "USER"


# ---------------------------------------------------------------------------
# The tool, and what it does when it fails.
# ---------------------------------------------------------------------------


def test_a_zero_exit_with_no_report_is_not_an_answer(tmp_path: Path) -> None:
    """**Measured, not assumed: this tool's input validation exits zero.**

    `scale_sim.set_params` calls the builtin `exit()` when an input file is
    missing, which is exit status zero. A wrapper that checked the status would
    read a hard failure as a success and then find no report, or worse, find a
    stale one from a previous run. So the condition is the report, and this
    proves the refusal fires.

    The fault injected here is a topology with no rows, which SCALE-Sim accepts
    and then writes a report with no layers in.
    """
    require_scalesim()
    with pytest.raises(export.ScaleSimError):
        export.run_scalesim(
            topology_csv=export.TOPOLOGY_HEADER + "\n",
            config_text=export.architecture_config(65536),
            layout_csv=export.LAYOUT_HEADER + "\n",
            directory=tmp_path / "empty",
            expected_layers=1,
        )


def test_a_report_with_a_moved_column_order_is_refused(tmp_path: Path) -> None:
    """A column order that changed upstream must not be read as a cycle count."""
    report = tmp_path / "COMPUTE_REPORT.csv"
    report.write_text("LayerID, Total Cycles, Stall Cycles,\n0, 10, 0,\n")
    import subprocess

    completed = subprocess.CompletedProcess(["true"], 0, "", "")
    with pytest.raises(export.ScaleSimError, match="header this parser"):
        export._parse_compute_report(report, 1, completed)


@pytest.mark.slow
def test_the_layout_file_changes_nothing(tmp_path: Path) -> None:
    """SCALE-Sim v3 demands a layout file and then ignores it. Proved, not assumed.

    `IfmapCustomLayout` and `FilterCustomLayout` are both false in the
    configuration this project writes, and under that setting the layout arrays
    are loaded and never consulted. That is a claim about a tool this project did
    not write, made about an input it does not understand, whose answer it
    publishes. So it is measured: two runs of the same cell with different
    intraline factors have to return identical cycle counts.
    """
    require_scalesim()
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        operations = walked(
            cell("depthwise_separable-O2-default-n1-fp32-normal"), Path(directory)
        )
    topology = export.export(operations)
    config = export.architecture_config(1048576)

    first = export.run_scalesim(
        topology_csv=topology.csv_text(),
        config_text=config,
        layout_csv=export.layout_text(topology.rows),
        directory=tmp_path / "a",
        expected_layers=len(topology.rows),
    )
    second = export.run_scalesim(
        topology_csv=topology.csv_text(),
        config_text=config,
        layout_csv=export.layout_text(
            topology.rows,
            row_values="1, 1, 1, 1, 1, 1, 0, 1, 2, 4, 5, 3, 3, 2, 1, 0, 4, 5, 6, 7,",
        ),
        directory=tmp_path / "b",
        expected_layers=len(topology.rows),
    )
    assert [report.total_cycles for report in first] == [
        report.total_cycles for report in second
    ]


@pytest.mark.slow
def test_scalesim_did_the_arithmetic_it_was_given(tmp_path: Path) -> None:
    """The one check here that can fail, run end to end on a real cell.

    SCALE-Sim's `Overall Util %` times its cycle count times the array positions
    recovers the MAC count it believed it was doing, and that has to be the MAC
    count this project charged. Without it, every divergence figure could be two
    tools answering about two different layers.
    """
    require_scalesim()
    result = cell("inception_block-O2-default-n1-fp32-normal")
    with tempfile.TemporaryDirectory(prefix="npu-ss-test-") as directory:
        text = roofline.Compiler(Path(directory)).allocated_ir(result)
    answer = export.divergence_for(result, text, tmp_path)

    assert answer.layers
    assert 0.0 < answer.covered_cycle_fraction <= 1.0
    assert 0.0 < answer.covered_op_fraction <= 1.0
    # Section 16.3 forbids an agreement figure in a row without a coverage
    # fraction beside it, and the schema block is where that is enforced.
    block = answer.as_schema()
    assert block["scalesim_covered_cycle_fraction"] is not None
    assert block["scalesim_covered_op_fraction"] is not None
    assert not any(key.endswith("_null_reason") for key in block)
    # And the partition is complete.
    assert abs(answer.terms["residual"]) < 1e-6 * max(
        1.0, abs(answer.terms["total_divergence"])
    )


# ---------------------------------------------------------------------------
# Rank fidelity.
# ---------------------------------------------------------------------------


def test_rank_fidelity_on_orderings_a_reader_can_check() -> None:
    identical = export.rank_fidelity([(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)])
    assert identical["kendall_tau_b"] == pytest.approx(1.0)
    assert identical["pairwise_accuracy"] == pytest.approx(1.0)

    reversed_order = export.rank_fidelity([(1.0, 30.0), (2.0, 20.0), (3.0, 10.0)])
    assert reversed_order["kendall_tau_b"] == pytest.approx(-1.0)
    assert reversed_order["pairwise_accuracy"] == pytest.approx(0.0)

    # A tie on one side is not a pair the other side got right or wrong, which is
    # why tau b is used and why the pairwise denominator excludes it. This suite
    # has many such pairs: the same program at three optimization levels ties
    # exactly on both sides.
    tied = export.rank_fidelity([(1.0, 10.0), (1.0, 20.0), (2.0, 30.0)])
    assert tied["tied_on_this_project"] == 1.0
    assert tied["pairwise_accuracy"] == pytest.approx(1.0)
    assert tied["kendall_tau_b"] < 1.0
