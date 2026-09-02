# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""SCALE-Sim v3 cross validation, per Section 16.3.

    python experiments/scalesim_export.py --models lenet --verbose
    python experiments/scalesim_export.py --json /tmp/scalesim.json

This walks the allocated `npuisa` IR, emits a SCALE-Sim topology CSV plus an
architecture configuration, runs the pinned SCALE-Sim, parses its per layer cycle
counts, and decomposes the divergence from this project's own analytical model
into named terms. The IR driven export has direct prior art and is cited rather
than reinvented silently: see `report/references.bib`, `samajdar2020scalesim` and
`raj2025scalesimv3`.

## The column order is copied, not remembered

Section 16.3 says to read the bundled example topologies from the installed path
of the pinned version and copy their column order exactly rather than inventing
one from memory. `TOPOLOGY_HEADER` below is that header, byte for byte from
`topologies/conv_nets/alexnet.csv` at the pinned sha, and
`test_scalesim_export.py` asserts it still matches the file rather than trusting
this comment.

## What is exported, and what is not

- `npuisa.conv2d`: IFMAP height and width from the activation operand, filter
  height and width plus channels and filter count from the weight, stride from
  the attribute.
- `group > 1`: one row per group, named `<layer>_g<i>`, each carrying `C / group`
  channels and `O / group` filters.
- `dilation > 1`: SCALE-Sim models no dilation, so the row carries the
  **effective filter extent** `d * (k - 1) + 1` and the layer is returned in the
  `approximations` list. It is never presented as exact, and the cost of the
  approximation is **measured** rather than argued: the layer is run a second
  time at its true tap extent and the difference is the `dilation_approximation`
  term of the decomposition.
- `npuisa.matmul`: one GEMM shaped row. IFMAP `M` by 1, filter 1 by 1, `K`
  channels, `N` filters, stride 1. The mapping is stated here because a reader
  meeting `IFMAP Height = 400` on a fully connected layer deserves to be told it
  is the reduction depth and not an image.
- Elementwise operations, pooling, DMA, constants and `HALT` have no systolic
  representation. They are skipped and returned in the `skipped` list **with the
  analytical cycles each one carries**, so the divergence account can name the
  gap rather than discover it.

## Coverage is reported beside every agreement figure

Section 16.3 forbids a `cost_model_agreement` ratio in a row without
`scalesim_covered_cycle_fraction` beside it, and this module never returns one
without the other. An agreement computed over a topology that silently dropped
pooling, elementwise work and DMA is an agreement on an unstated subset.

## Reading SCALE-Sim's answer, and what its answer costs to trust

Two things about this tool were measured rather than assumed, and both are the
D-0040 shape of failure sitting in an external dependency:

1. **A missing input file makes SCALE-Sim print an error and exit 0.**
   `scale_sim.set_params` calls the builtin `exit()`, which is exit status zero.
   Measured on 2026-09-02: a run with no layout file printed
   `ERROR: scalesim.scale.py: Layout file not found` and exited **0**, while an
   uncaught exception in the same tool exited 1. So this module never treats a
   zero exit as an answer: it requires `COMPUTE_REPORT.csv` to exist, to carry
   the header it expects, and to hold exactly one row per exported layer, and it
   raises with the tool's own stdout and stderr otherwise.
2. **SCALE-Sim v3 does not run under numpy 2.** Three `int(max(...))` casts over
   numpy arrays raise `TypeError: only 0-dimensional arrays can be converted to
   Python scalars`, on the tool's own shipped example, on every upstream branch
   at the time of writing. `scripts/patch-scalesim.py` is the three expression
   compatibility fix and `docs/DEFECT_LOG.md` D-0044 carries the whole account.
   The installed tree's sha256 is recorded in every manifest beside the upstream
   git sha, so a reader is told the tool was modified and by how much rather than
   being shown a sha that does not describe the code that ran.

Every number parsed here carries its provenance: `COMPUTE_REPORT.csv`, the column
name, and the fact that SCALE-Sim prints cycle counts as **exact integers**, so
they are read with `int` and compared without a band. The bandwidth report's
figures are floats and are not used for any comparison.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from roofline import Compiler, _mlir_python_packages_dir  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

from npu_frontend import cost_model, npuisa_walk  # noqa: E402
from npu_frontend.results import (  # noqa: E402
    RESULTS_DIR,
    ResultSchemaError,
    load_result,
)

#: The header of `topologies/conv_nets/alexnet.csv` at the pinned sha, copied
#: rather than composed. `test_scalesim_export.py` reads the installed file and
#: asserts this still matches it.
TOPOLOGY_HEADER = (
    "Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, "
    "Channels, Num Filter, Strides,"
)

#: The header SCALE-Sim writes at the top of `COMPUTE_REPORT.csv`. Checked rather
#: than assumed, because a column order change upstream would otherwise be read
#: as a cycle count.
COMPUTE_REPORT_HEADER = [
    "LayerID",
    "Total Cycles (incl. prefetch)",
    "Total Cycles",
    "Stall Cycles",
    "Overall Util %",
    "Mapping Efficiency %",
    "Compute Util %",
]

#: Where SCALE-Sim lives, so that a missing install fails loudly naming the
#: dependency rather than producing a null nobody asked for. Section 16.4 states
#: that rule for Accelergy and it is the same rule here.
SCALESIM_MODULE = "scalesim.scale"

#: The one place the run name is spelled. SCALE-Sim writes its reports into
#: `<top_path>/<run_name>/`, so a caller that does not know the name cannot find
#: the answer.
RUN_NAME = "npu_mlir_p11"


class ScaleSimError(Exception):
    """SCALE-Sim could not be run, or its answer could not be read."""


# ---------------------------------------------------------------------------
# The export.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyRow:
    """One row of the topology CSV, in the pinned version's column order."""

    name: str
    ifmap_height: int
    ifmap_width: int
    filter_height: int
    filter_width: int
    channels: int
    num_filters: int
    stride: int
    #: The operation this row came from, so a per layer divergence can be
    #: attributed back to the program rather than to a row number.
    source_position: int

    def as_csv(self) -> str:
        return (
            f"{self.name},{self.ifmap_height},{self.ifmap_width},"
            f"{self.filter_height},{self.filter_width},{self.channels},"
            f"{self.num_filters},{self.stride},"
        )


@dataclass(frozen=True)
class Skipped:
    """An operation with no systolic representation, and what it cost here."""

    name: str
    op: str
    position: int
    #: The analytical cycles this project charged for it. Section 16.3 asks the
    #: exporter to emit exactly this, so the divergence reads as a sum rather
    #: than as one headline percentage.
    analytical_cycles: float
    reason: str


@dataclass(frozen=True)
class Approximation:
    """A layer the topology represents inexactly, named as such."""

    name: str
    op: str
    position: int
    reason: str
    detail: dict[str, Any]


@dataclass
class Topology:
    """What one cell exports, with everything it could not export beside it."""

    rows: list[TopologyRow] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    approximations: list[Approximation] = field(default_factory=list)
    #: The rows a second run uses to measure what the dilation approximation
    #: cost, carrying the true tap extent instead of the effective one.
    undilated_rows: list[TopologyRow] = field(default_factory=list)

    @property
    def has_dilation(self) -> bool:
        return any(
            approximation.reason == "dilation" for approximation in self.approximations
        )

    def csv_text(self, rows: list[TopologyRow] | None = None) -> str:
        chosen = self.rows if rows is None else rows
        return "\n".join([TOPOLOGY_HEADER, *(row.as_csv() for row in chosen)]) + "\n"


def export(operations: list[npuisa_walk.Operation]) -> Topology:
    """Section 16.3's export, from one walk of the allocated IR."""
    topology = Topology()

    for operation in operations:
        if operation.op == "conv2d":
            _export_conv2d(topology, operation)
        elif operation.op == "matmul":
            _export_matmul(topology, operation)
        elif operation.op in npuisa_walk.TRANSFER_OPS:
            topology.skipped.append(
                Skipped(
                    name=operation.name,
                    op=operation.op,
                    position=operation.position,
                    analytical_cycles=operation.cycles,
                    reason=(
                        "a DMA between DRAM and the scratchpad. SCALE-Sim models "
                        "its own memory hierarchy from the topology it is given "
                        "and has no representation for an explicit transfer "
                        "instruction."
                    ),
                )
            )
        elif operation.op in npuisa_walk.POOL_OPS:
            topology.skipped.append(
                Skipped(
                    name=operation.name,
                    op=operation.op,
                    position=operation.position,
                    analytical_cycles=operation.cycles,
                    reason=(
                        "SCALE-Sim does not model pooling at all, which is the "
                        "second mechanism the divergence prediction names."
                    ),
                )
            )
        else:
            topology.skipped.append(
                Skipped(
                    name=operation.name,
                    op=operation.op,
                    position=operation.position,
                    analytical_cycles=operation.cycles,
                    reason=(
                        "an elementwise or shape operation. A systolic array "
                        "topology has no row for it."
                    ),
                )
            )

    if not topology.rows:
        raise ScaleSimError(
            "nothing in this program exported to a topology row. Every model in "
            "this suite has at least one convolution or matmul, so an empty "
            "topology is a walk that went wrong rather than a program with no "
            "systolic work in it."
        )
    check_macs(topology, operations)
    return topology


def row_macs(row: TopologyRow) -> int:
    """The multiply accumulates one topology row describes.

    SCALE-Sim's output extents are `(ifmap - filter) / stride + 1` on each axis,
    with no padding term, and every output element accumulates
    `filter_height * filter_width * channels` products for each of `num_filters`
    filters. Written out here rather than taken from SCALE-Sim's own report,
    because the point is to check the export against this project independently
    of what the tool then does with it.
    """
    output_height = (row.ifmap_height - row.filter_height) // row.stride + 1
    output_width = (row.ifmap_width - row.filter_width) // row.stride + 1
    return (
        output_height
        * output_width
        * row.filter_height
        * row.filter_width
        * row.channels
        * row.num_filters
    )


def check_macs(topology: Topology, operations: list[npuisa_walk.Operation]) -> None:
    """The exported topology describes the same arithmetic this project charged.

    **This is the check that makes a divergence figure mean anything.** A
    topology whose rows imply a different MAC count from the program they came
    from produces a cycle count for a different layer, and the difference then
    arrives as a divergence with a plausible size and no cause. The extents are
    derived from the output positions precisely so that this holds, and it is
    asserted rather than trusted.

    The comparison is on `undilated_rows`, because those carry the layer's true
    tap count. The dilated rows deliberately do not match, which is the whole
    content of the dilation approximation, and what that costs is measured by the
    second SCALE-Sim run rather than by this check.
    """
    charged: dict[int, int] = {}
    for row in topology.undilated_rows:
        charged[row.source_position] = charged.get(row.source_position, 0) + row_macs(
            row
        )

    disagreements = []
    for operation in operations:
        if not operation.is_compute:
            continue
        exported = charged.get(operation.position)
        if exported != operation.macs:
            disagreements.append(
                f"{operation.name} ({operation.op}) at position "
                f"{operation.position}: this project charged {operation.macs} "
                f"MACs and the exported topology implies {exported}"
            )
    if disagreements:
        raise ScaleSimError(
            "the exported topology does not describe the same arithmetic as the "
            "program it came from:\n  "
            + "\n  ".join(disagreements)
            + "\n\nSCALE-Sim has no padding field and no batch field, so the "
            "IFMAP extents are derived from the output positions rather than "
            "copied from the activation. A mismatch here means that derivation "
            "is wrong, and every cycle count taken from this topology would be "
            "the cycle count of a layer this program does not contain."
        )


def _export_conv2d(topology: Topology, operation: npuisa_walk.Operation) -> None:
    """One convolution, as however many topology rows its groups need.

    **The IFMAP extents are derived from the output rather than copied from the
    input, and that is the difference between a comparison and an anecdote.**
    SCALE-Sim's topology has no padding field and no batch field, and this
    machine has both. Copying the activation's own height and width across would
    hand SCALE-Sim a layer with `(H - k) / s + 1` output positions where this one
    has `(H + pads - k) / s + 1` of them, times the batch, so the two tools would
    be charged for different amounts of arithmetic and every divergence figure
    would carry that difference without naming it.

    So the extents are chosen to make SCALE-Sim's own output size formula produce
    exactly this layer's output positions, with the batch folded into the row
    dimension because that is what the cost model of Section 5.5 does: it streams
    `batch * oH * oW` activation rows against one stationary weight tile.

        ifmap_height = (batch * oH - 1) * stride + filter_height
        ifmap_width  = (oW - 1) * stride + filter_width

    `_check_macs` then asserts the MAC count implied by the exported row equals
    the MAC count this project charged, which is what turns the paragraph above
    into a checked claim.
    """
    activation = operation.operands[0]
    weight = operation.operands[1]
    result = operation.result

    batch = result.shape[0]
    output_channels = result.shape[1]
    output_height = result.shape[2]
    output_width = result.shape[3]
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]
    input_channels = activation.shape[1]
    group = int(operation.attributes.get("group", 1))
    strides = operation.attributes.get("strides", (1, 1))
    dilations = operation.attributes.get("dilations", (1, 1))

    if strides[0] != strides[1]:
        topology.approximations.append(
            Approximation(
                name=operation.name,
                op=operation.op,
                position=operation.position,
                reason="anisotropic_stride",
                detail={
                    "strides": list(strides),
                    "note": (
                        "a SCALE-Sim topology row carries one Strides column and "
                        "this layer strides differently on the two axes. The row "
                        "carries the row stride and the column extent is chosen "
                        "against it, so the output position count is still "
                        "exact and only the shape of the window walk differs."
                    ),
                },
            )
        )
    stride = int(strides[0])

    # Section 16.3: SCALE-Sim models no dilation, so record the effective filter
    # extent and return the layer as an approximation rather than as exact.
    effective_height = kernel_height
    effective_width = kernel_width
    if dilations != (1, 1):
        effective_height = dilations[0] * (kernel_height - 1) + 1
        effective_width = dilations[1] * (kernel_width - 1) + 1
        topology.approximations.append(
            Approximation(
                name=operation.name,
                op=operation.op,
                position=operation.position,
                reason="dilation",
                detail={
                    "dilations": list(dilations),
                    "true_filter": [kernel_height, kernel_width],
                    "effective_filter": [effective_height, effective_width],
                    "note": (
                        "SCALE-Sim models no dilation. The row carries the "
                        "effective extent d * (k - 1) + 1, which presents the "
                        "array with a larger filter than the layer has taps for. "
                        "What that costs is measured by a second run at the true "
                        "extent rather than argued."
                    ),
                },
            )
        )

    channels_per_group = input_channels // group
    filters_per_group = output_channels // group

    for index in range(group):
        name = operation.name if group == 1 else f"{operation.name}_g{index}"
        for extents, target in (
            ((effective_height, effective_width), topology.rows),
            ((kernel_height, kernel_width), topology.undilated_rows),
        ):
            target.append(
                TopologyRow(
                    name=name,
                    ifmap_height=(batch * output_height - 1) * stride + extents[0],
                    ifmap_width=(output_width - 1) * stride + extents[1],
                    filter_height=extents[0],
                    filter_width=extents[1],
                    channels=channels_per_group,
                    num_filters=filters_per_group,
                    stride=stride,
                    source_position=operation.position,
                )
            )


def _export_matmul(topology: Topology, operation: npuisa_walk.Operation) -> None:
    """One GEMM shaped row, per Section 16.3.

    IFMAP `M` by 1, filter 1 by 1, `K` channels, `N` filters, stride 1. `M` is
    the row count the activation streams, `K` is the reduction depth and `N` is
    the output width, and none of the three is an image extent even though two of
    them sit in columns named for one.
    """
    rows = operation.result.shape[0]
    columns = operation.result.shape[1]
    reduction = operation.operands[0].shape[1]

    row = TopologyRow(
        name=operation.name,
        ifmap_height=rows,
        ifmap_width=1,
        filter_height=1,
        filter_width=1,
        channels=reduction,
        num_filters=columns,
        stride=1,
        source_position=operation.position,
    )
    topology.rows.append(row)
    topology.undilated_rows.append(row)


# ---------------------------------------------------------------------------
# The architecture configuration, restated from the cost model's one home.
# ---------------------------------------------------------------------------


def architecture_config(scratchpad_bytes: int, *, run_name: str = RUN_NAME) -> str:
    """The `CostModel` constants, restated in SCALE-Sim's own format.

    Section 16.3 asks the configuration to restate the constants from their
    single source, which is what this does: nothing here is a literal that
    `python/npu_frontend/cost_model.py` does not carry.

    **The bandwidth is converted and the conversion is stated.** SCALE-Sim's
    `Bandwidth` is in **words** per cycle and this project's
    `DRAM_BANDWIDTH_BYTES_PER_CYCLE` is in bytes, so the figure is divided by the
    four byte f32 element. A number handed over in the wrong unit would be a
    sixteen times bandwidth error that every downstream figure would absorb
    silently, which is the whole reason this docstring names the units on both
    sides.

    The three SRAM sizes are the cell's own scratchpad budget, in kB, because
    that is the memory this machine has. SCALE-Sim wants three separate
    capacities and this machine has one flat scratchpad, so the budget is given
    to each rather than split, and the layer sizes in this suite are small enough
    that no split would bind. That is an approximation and it is named here.
    """
    words_per_cycle = (
        cost_model.DRAM_BANDWIDTH_BYTES_PER_CYCLE / npuisa_walk.ELEMENT_BYTES
    )
    kilobytes = max(1, scratchpad_bytes // 1024)

    parser = configparser.ConfigParser()
    parser.optionxform = str  # SCALE-Sim's keys are case sensitive.
    parser["general"] = {"run_name": run_name}
    parser["architecture_presets"] = {
        "ArrayHeight": str(cost_model.ARRAY_DIM),
        "ArrayWidth": str(cost_model.ARRAY_DIM),
        "IfmapSramSzkB": str(kilobytes),
        "FilterSramSzkB": str(kilobytes),
        "OfmapSramSzkB": str(kilobytes),
        "IfmapOffset": "0",
        "FilterOffset": "10000000",
        "OfmapOffset": "20000000",
        # Weight stationary, because Section 5.5 pins weight stationary and ADR
        # 0007 records it as a pinned assumption. Handing SCALE-Sim a different
        # dataflow would be comparing this machine against a different one.
        "Dataflow": "ws",
        "Bandwidth": f"{words_per_cycle:g}",
        "MemoryBanks": "1",
        "ReadRequestBuffer": "32",
        "WriteRequestBuffer": "32",
    }
    parser["layout"] = {
        "IfmapCustomLayout": "False",
        "IfmapSRAMBankBandwidth": f"{words_per_cycle:g}",
        "IfmapSRAMBankNum": "1",
        "IfmapSRAMBankPort": "1",
        "FilterCustomLayout": "False",
        "FilterSRAMBankBandwidth": f"{words_per_cycle:g}",
        "FilterSRAMBankNum": "1",
        "FilterSRAMBankPort": "1",
    }
    parser["sparsity"] = {
        "SparsitySupport": "false",
        "SparseRep": "ellpack_block",
        "OptimizedMapping": "false",
        "BlockSize": "8",
        "RandomNumberGeneratorSeed": "40",
    }
    parser["run_presets"] = {
        # USER, not CALC: the bandwidth is this project's constant and letting
        # SCALE-Sim estimate its own would be comparing against a machine with a
        # different interconnect from the one the cost model charges for.
        "InterfaceBandwidth": "USER",
        "UseRamulatorTrace": "False",
    }

    buffer = io.StringIO()
    parser.write(buffer)
    return buffer.getvalue()


#: The layout header of `layouts/conv_nets/test.csv` at the pinned sha.
LAYOUT_HEADER = (
    "Layer name, IFMAP Height Intraline Factor, IFMAP Width Intraline Factor, "
    "Filter Height Intraline Factor, Filter Width Intraline Factor, "
    "Channel Intraline Factor, Num Filter Intraline Factor, "
    "IFMAP Height Intraline Order, IFMAP Width Intraline Order, "
    "Channel Intraline Order, IFMAP Height Interline Order, "
    "IFMAP Width Interline Order, Channel Interline Order, "
    "Num Filter Intraline Order, Channel Intraline Order, "
    "Filter Height Intraline Order, Filter Width Intraline Order, "
    "Num Filter Interline Order, Channel Interline Order, "
    "Filter Height Interline Order, Filter Width Interline Order,"
)

#: The per layer layout row, which SCALE-Sim v3 requires a file for and then does
#: not use while `IfmapCustomLayout` and `FilterCustomLayout` are both false.
#:
#: **That it is unused is proved rather than assumed.**
#: `test_scalesim_export.py` runs one cell twice with two different layout rows
#: and asserts the cycle counts are identical. An input this project does not
#: understand, handed to a tool whose answer it publishes, is exactly the kind of
#: thing that turns out to matter after the number is in a report.
LAYOUT_ROW = "2, 2, 1, 1, 16, 4, 0, 1, 2, 4, 5, 3, 3, 2, 1, 0, 4, 5, 6, 7,"


def layout_text(rows: list[TopologyRow], row_values: str = LAYOUT_ROW) -> str:
    return (
        "\n".join([LAYOUT_HEADER, *(f"{row.name}, {row_values}" for row in rows)])
        + "\n"
    )


# ---------------------------------------------------------------------------
# Running the tool and reading its answer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerReport:
    """One row of `COMPUTE_REPORT.csv`, with where every field came from.

    Section 16.1's rule about provenance, applied to a parsed external number:
    the file, the column, the units and the tool's own printed resolution travel
    with the value rather than being remembered by whoever wrote the parser.
    """

    layer_id: int
    total_cycles: int
    total_cycles_with_prefetch: int
    stall_cycles: int
    overall_utilization_percent: float
    mapping_efficiency_percent: float
    compute_utilization_percent: float

    #: What this row is, in the terms Section 16.1 asks a parsed number to carry.
    provenance: str = (
        "COMPUTE_REPORT.csv written by SCALE-Sim v3. The three cycle columns are "
        "printed as exact integers, so they are read with int and compared "
        "without a band; the three percentages are printed as full precision "
        "floats and are used only to attribute the divergence, never to assert "
        "an agreement."
    )


def installed_tree_sha256() -> str:
    """A hash over the installed SCALE-Sim sources, recorded in every manifest.

    **The upstream git sha does not describe the code that ran**, because
    `scripts/scalesim-numpy2.patch` is applied to the install. Recording only the
    sha would be recording a provenance that is very nearly true, which is worse
    than one that is either true or visibly qualified. This hash changes if
    anything under the installed package changes, patch included.
    """
    module = _installed_root()
    digest = hashlib.sha256()
    for path in sorted(module.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(module)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _installed_root() -> Path:
    try:
        import scalesim
    except ImportError as failure:  # pragma: no cover
        raise ScaleSimError(
            "SCALE-Sim is not installed in this interpreter. Section 16.4's rule "
            "is that a missing external tool fails loudly naming the dependency, "
            "and this is that failure. Install it from the pinned sha recorded in "
            "docs/adr/0003-resolved-tool-matrix.md."
        ) from failure
    return Path(scalesim.__file__).parent


def run_scalesim(
    *,
    topology_csv: str,
    config_text: str,
    layout_csv: str,
    directory: Path,
    expected_layers: int,
) -> list[LayerReport]:
    """One SCALE-Sim run, with its answer checked before it is believed.

    The exit code is checked **and is not sufficient**, for the reason the module
    docstring measures: this tool's own input validation path prints an error and
    exits zero. So the report has to exist, carry the header this parser expects,
    and hold one row per exported layer, and any of the three failing raises with
    everything the tool said.
    """
    directory.mkdir(parents=True, exist_ok=True)
    topology_path = directory / "topology.csv"
    config_path = directory / "arch.cfg"
    layout_path = directory / "layout.csv"
    output = directory / "out"

    topology_path.write_text(topology_csv, encoding="utf-8")
    config_path.write_text(config_text, encoding="utf-8")
    layout_path.write_text(layout_csv, encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        SCALESIM_MODULE,
        "-c",
        str(config_path),
        "-t",
        str(topology_path),
        "-l",
        str(layout_path),
        "-p",
        str(output),
        # No traces. One layer of one model produced 124 MB of them, and the
        # suite has 550 layers.
        "-s",
        "N",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        # Never the repository or the SCALE-Sim clone: a cwd holding a `scalesim`
        # directory shadows the installed package, and the run then measures a
        # copy of the tool this project did not hash.
        cwd=str(directory),
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )

    report_path = output / RUN_NAME / "COMPUTE_REPORT.csv"
    if completed.returncode != 0 or not report_path.is_file():
        raise ScaleSimError(
            f"SCALE-Sim exited {completed.returncode} and "
            f"{'wrote no' if not report_path.is_file() else 'wrote a'} "
            f"COMPUTE_REPORT.csv. **A zero exit is not an answer from this "
            f"tool**: its own input validation calls the builtin exit(), which "
            f"is status zero, so the report having been written is the condition "
            f"that matters and it is checked here rather than inferred.\n\n"
            f"command:\n  {' '.join(command)}\n\n"
            f"stdout:\n{completed.stdout.strip()[-4000:] or '(nothing)'}\n\n"
            f"stderr:\n{completed.stderr.strip()[-4000:] or '(nothing)'}"
        )

    return _parse_compute_report(report_path, expected_layers, completed)


def _parse_compute_report(
    path: Path, expected_layers: int, completed: subprocess.CompletedProcess[str]
) -> list[LayerReport]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            [cell.strip() for cell in row if cell.strip() != ""]
            for row in csv.reader(handle)
            if any(cell.strip() for cell in row)
        ]

    if not rows or rows[0] != COMPUTE_REPORT_HEADER:
        raise ScaleSimError(
            f"{path} does not carry the header this parser was written against.\n"
            f"expected: {COMPUTE_REPORT_HEADER}\n"
            f"found:    {rows[0] if rows else '(the file is empty)'}\n\n"
            f"A column order that moved upstream would otherwise be read as a "
            f"cycle count, which is a wrong number rather than a failure."
        )

    reports: list[LayerReport] = []
    for row in rows[1:]:
        if len(row) != len(COMPUTE_REPORT_HEADER):
            raise ScaleSimError(f"{path}: a row has {len(row)} fields: {row}")
        reports.append(
            LayerReport(
                layer_id=int(row[0]),
                total_cycles_with_prefetch=int(row[1]),
                total_cycles=int(row[2]),
                stall_cycles=int(row[3]),
                overall_utilization_percent=float(row[4]),
                mapping_efficiency_percent=float(row[5]),
                compute_utilization_percent=float(row[6]),
            )
        )

    if len(reports) != expected_layers:
        raise ScaleSimError(
            f"{path} holds {len(reports)} layers and the topology exported "
            f"{expected_layers}. SCALE-Sim answered a different question from "
            f"the one it was asked, and a per layer divergence taken across that "
            f"mismatch would attribute one layer's cycles to another.\n\n"
            f"stdout:\n{completed.stdout.strip()[-2000:] or '(nothing)'}"
        )
    return reports


# ---------------------------------------------------------------------------
# The divergence, decomposed into named terms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerDivergence:
    """One covered layer, both models' answers, and the gap between them."""

    name: str
    position: int
    op: str
    macs: int
    #: The later of this machine's two timelines for this layer, which is what
    #: `experiments/roofline.py` compares against a bound and is the quantity
    #: SCALE-Sim's own per layer total is comparable with.
    analytical_cycles: float
    analytical_compute_cycles: float
    analytical_dma_cycles: float
    #: What SCALE-Sim answered for the topology it can actually be asked, which
    #: on a dilated layer carries the effective filter extent.
    scalesim_cycles: int
    #: What it answered for the topology describing the same arithmetic this
    #: project charged. The same number except on a dilated layer, where the
    #: difference between the two is the measured dilation approximation.
    scalesim_arithmetic_matched_cycles: int
    scalesim_stall_cycles: int
    scalesim_overall_utilization: float

    @property
    def divergence(self) -> float:
        return self.analytical_cycles - self.scalesim_cycles

    @property
    def relative_divergence(self) -> float:
        if self.scalesim_cycles == 0:
            return float("inf")
        return self.divergence / self.scalesim_cycles


@dataclass(frozen=True)
class CellDivergence:
    """Everything SCALE-Sim says about one cell, and everything it did not see."""

    cell: str
    layers: tuple[LayerDivergence, ...]
    skipped: tuple[Skipped, ...]
    approximations: tuple[Approximation, ...]
    terms: dict[str, float]
    covered_cycle_fraction: float
    covered_op_fraction: float
    analytical_serial_total: float
    simulated_cycles: float
    scalesim_total: int

    @property
    def whole_model_relative_divergence(self) -> float:
        """Over the covered layers only, which is the figure the prediction names."""
        covered = sum(layer.analytical_cycles for layer in self.layers)
        if self.scalesim_total == 0:
            return float("inf")
        return (covered - self.scalesim_total) / self.scalesim_total

    def as_schema(self) -> dict[str, Any]:
        """The `external` fields Section 16.1 names for SCALE-Sim."""
        return {
            "scalesim_cycles": self.scalesim_total,
            "scalesim_cycles_per_layer": [
                {
                    "name": layer.name,
                    "position": layer.position,
                    "op": layer.op,
                    "macs": layer.macs,
                    "scalesim_cycles": layer.scalesim_cycles,
                    "scalesim_arithmetic_matched_cycles": (
                        layer.scalesim_arithmetic_matched_cycles
                    ),
                    "scalesim_stall_cycles": layer.scalesim_stall_cycles,
                    "scalesim_overall_utilization": layer.scalesim_overall_utilization,
                    "analytical_cycles": layer.analytical_cycles,
                    "analytical_compute_cycles": layer.analytical_compute_cycles,
                    "analytical_dma_cycles": layer.analytical_dma_cycles,
                    "divergence_cycles": layer.divergence,
                    "relative_divergence": layer.relative_divergence,
                }
                for layer in self.layers
            ],
            "scalesim_skipped": [asdict(entry) for entry in self.skipped],
            "scalesim_approximations": [asdict(entry) for entry in self.approximations],
            "scalesim_covered_cycle_fraction": self.covered_cycle_fraction,
            "scalesim_covered_op_fraction": self.covered_op_fraction,
        }


def check_the_same_arithmetic(
    *,
    cell: str,
    topology: Topology,
    matched: list[LayerReport],
) -> None:
    """SCALE-Sim's own utilization says it did the MAC count this project charged.

    **This is the one check here that can fail, and it is what every divergence
    figure rests on.** SCALE-Sim reports `Overall Util %` as the useful multiply
    accumulates over the array positions times the cycles it took, so
    `total_cycles * util * array_height * array_width` recovers the MAC count it
    believed it was doing. Comparing that with `row_macs` closes the loop from
    the other side: `check_macs` asserts the export describes this program, and
    this asserts the tool agreed about what it was given.

    **The band is derived from the tool's printed precision and from nothing
    else.** `Overall Util %` is printed at full float repr, so the only slack
    needed is the rounding of one multiply and one divide, and the check is
    against half a MAC. A percentage printed to two decimals would need a band
    three orders wider, which is exactly the kind of thing that has to be read
    off the file rather than assumed, so the header check above pins the format
    this reasoning depends on.
    """
    positions = sorted({row.source_position for row in topology.undilated_rows})
    by_position: dict[int, list[tuple[TopologyRow, LayerReport]]] = {
        position: [] for position in positions
    }
    for row, report in zip(topology.undilated_rows, matched, strict=True):
        by_position[row.source_position].append((row, report))

    array_positions = cost_model.ARRAY_DIM * cost_model.ARRAY_DIM
    disagreements = []
    for position, pairs in by_position.items():
        exported = sum(row_macs(row) for row, _ in pairs)
        implied = sum(
            report.total_cycles
            * (report.overall_utilization_percent / 100.0)
            * array_positions
            for _, report in pairs
        )
        if abs(implied - exported) > 0.5:
            disagreements.append(
                f"position {position}: the topology describes {exported} MACs "
                f"and SCALE-Sim's own utilization implies {implied:.3f}"
            )
    if disagreements:
        raise ScaleSimError(
            f"{cell}: SCALE-Sim did not do the arithmetic it was given.\n  "
            + "\n  ".join(disagreements)
            + "\n\nEvery divergence figure below compares two charges for the "
            "same work, and that is only true while this holds."
        )


def decompose(
    *,
    cell: str,
    operations: list[npuisa_walk.Operation],
    topology: Topology,
    reports: list[LayerReport],
    undilated_reports: list[LayerReport] | None,
    result: dict[str, Any],
) -> CellDivergence:
    """Section 16.3's decomposition: a sum of named terms, not one percentage.

    "18 percent equals 11 pooling plus 4 double buffering plus 3 residual" is an
    analysis; "18 percent" is a number. Each term below is measured from a
    quantity one of the two tools reported, and each is the same physical
    quantity on both sides so that the subtraction means something.

    **The residual is zero by construction and that is stated rather than
    presented as a result.** These terms partition the difference; they are not a
    fit to it. Every cycle either belongs to work SCALE-Sim never saw (the
    pooling, elementwise and uncovered DMA terms), to the filter extent the
    dilation approximation gave it, to the compute time difference on the layers
    it did see (fragmentation), or to the memory time neither model hid
    (double buffering). A nonzero residual would mean this function had lost
    cycles, so it is asserted rather than reported as evidence.

    **The check that is not free is `check_the_same_arithmetic`**, which
    reconciles SCALE-Sim's own utilization figure against the MAC count this
    project charged. That one can fail, and it is what makes the fragmentation
    term a comparison of two charges for the same work rather than a comparison
    of two different workloads.
    """
    # **Two runs, and which one each term is taken from is load bearing.**
    #
    # `reports` is SCALE-Sim's answer to the question it can actually be asked:
    # a topology with no dilation in it, so a dilated layer arrives as a larger
    # filter with more taps than the layer has. That is the headline figure,
    # because it is what this tool says about this program.
    #
    # `matched` is the run whose topology describes the **same arithmetic** this
    # project charged, which is the same run except on the dilated cells. Every
    # decomposed term except the dilation one is taken from it, because a
    # fragmentation term computed against a topology doing three times the
    # multiplies would absorb the dilation approximation a second time. It did,
    # in the first version of this function: `dilated_stack` came out with a
    # residual of -1838 cycles against named terms of +1524 and -1410, which is
    # the shape of a decomposition that has counted one effect twice and left the
    # difference in the residual.
    matched = reports if undilated_reports is None else undilated_reports
    check_the_same_arithmetic(cell=cell, topology=topology, matched=matched)

    by_position: dict[int, list[LayerReport]] = {}
    matched_by_position: dict[int, list[LayerReport]] = {}
    for row, report in zip(topology.rows, reports, strict=True):
        by_position.setdefault(row.source_position, []).append(report)
    for row, report in zip(topology.undilated_rows, matched, strict=True):
        matched_by_position.setdefault(row.source_position, []).append(report)

    covered = {operation.position: operation for operation in operations}

    layers: list[LayerDivergence] = []
    fragmentation = 0.0
    double_buffering = 0.0

    for position, group_reports in sorted(by_position.items()):
        operation = covered[position]
        matched_reports = matched_by_position[position]
        # The batch is already inside the exported extents, and the groups of a
        # grouped convolution are separate rows, so the layer's answer is the sum
        # over its rows with no scaling of any kind applied afterwards.
        scalesim_cycles = sum(report.total_cycles for report in group_reports)
        matched_cycles = sum(report.total_cycles for report in matched_reports)
        scalesim_stalls = sum(report.stall_cycles for report in matched_reports)
        utilization = sum(
            report.overall_utilization_percent for report in matched_reports
        ) / len(matched_reports)

        analytical = max(operation.cycles, operation.attributed_dma_cycles)
        layers.append(
            LayerDivergence(
                name=operation.name,
                position=position,
                op=operation.op,
                macs=operation.macs,
                analytical_cycles=analytical,
                analytical_compute_cycles=operation.cycles,
                analytical_dma_cycles=operation.attributed_dma_cycles,
                scalesim_cycles=scalesim_cycles,
                scalesim_arithmetic_matched_cycles=matched_cycles,
                scalesim_stall_cycles=scalesim_stalls,
                scalesim_overall_utilization=utilization / 100.0,
            )
        )

        # **Array fragmentation.** Both tools are charged the same MAC count,
        # which `check_macs` asserts and `check_the_same_arithmetic` confirms
        # from SCALE-Sim's own utilization figure. For the same MACs the only
        # reason two compute times differ is how each model charges for array
        # occupancy, so the difference between the two compute times **is** the
        # fragmentation term. SCALE-Sim's compute time is its total minus its
        # stalls, because its total carries the memory time that the next term
        # accounts for and counting it here as well was a double count the first
        # version of this function made.
        scalesim_compute = matched_cycles - scalesim_stalls
        fragmentation += operation.cycles - scalesim_compute

        # **Double buffering.** SCALE-Sim models it and this project does not
        # until P13, so the term is this machine's unhidden DMA time against
        # SCALE-Sim's own stall cycles, which is the same quantity on both sides:
        # memory time that could not be hidden behind compute.
        unhidden = max(0.0, operation.attributed_dma_cycles - operation.cycles)
        double_buffering += unhidden - scalesim_stalls

    # The dilation approximation, measured rather than argued: the same topology
    # at the true tap extent, and the difference is what the effective extent
    # cost SCALE-Sim's answer. The sign is negative because a larger filter makes
    # SCALE-Sim charge more, which widens the gap in the direction of this
    # project reading low.
    dilation = float(
        sum(layer.scalesim_arithmetic_matched_cycles for layer in layers)
        - sum(layer.scalesim_cycles for layer in layers)
    )

    # A transfer whose bytes were charged to a covered layer is already inside
    # that layer's analytical cycles through `attributed_dma_cycles`, so counting
    # it again here would double count it. One that feeds a pooling or
    # elementwise operation, or that nothing consumed, is genuinely outside the
    # comparison and is a term of it.
    inside_a_covered_layer = {
        operation.position
        for operation in operations
        if operation.op in npuisa_walk.TRANSFER_OPS
        and operation.attributed_to is not None
        and operation.attributed_to != operation.position
        and operations[operation.attributed_to].is_compute
    }

    pooling = sum(
        entry.analytical_cycles
        for entry in topology.skipped
        if entry.op in npuisa_walk.POOL_OPS
    )
    transfers = sum(
        entry.analytical_cycles
        for entry in topology.skipped
        if entry.op in npuisa_walk.TRANSFER_OPS
        and entry.position not in inside_a_covered_layer
    )
    elementwise = sum(
        entry.analytical_cycles
        for entry in topology.skipped
        if entry.op not in npuisa_walk.POOL_OPS
        and entry.op not in npuisa_walk.TRANSFER_OPS
    )

    # The comparison is between two serialized totals. This project's own
    # `simulated_cycles` is the later of two overlapped timelines at HALT and is
    # reported beside this rather than used as the left hand side, because
    # SCALE-Sim returns one number per layer and summing them is a serial total.
    analytical_serial = sum(
        (
            max(operation.cycles, operation.attributed_dma_cycles)
            if operation.is_compute
            else operation.cycles
        )
        for operation in operations
        if operation.position not in inside_a_covered_layer
    )
    scalesim_total = sum(layer.scalesim_cycles for layer in layers)

    named = {
        "pooling_gap": pooling,
        "elementwise_gap": elementwise,
        "uncovered_dma_gap": transfers,
        "dilation_approximation": dilation,
        "array_fragmentation": fragmentation,
        "double_buffering": double_buffering,
    }
    total_divergence = analytical_serial - scalesim_total
    named["residual"] = total_divergence - sum(named.values())
    named["total_divergence"] = total_divergence

    # The partition is complete or this function has lost cycles. The band is on
    # the representation only: every term is a sum of at most a few hundred
    # doubles against totals in the thousands, so anything above a millionth of a
    # cycle is a bookkeeping error rather than rounding.
    if abs(named["residual"]) > 1e-6 * max(1.0, abs(total_divergence)):
        raise ScaleSimError(
            f"{cell}: the divergence decomposition does not sum. The total is "
            f"{total_divergence:.6f} cycles and the named terms account for "
            f"{sum(value for key, value in named.items() if key != 'residual'):.6f}, "
            f"leaving a residual of {named['residual']:.6f}. These terms are a "
            f"partition rather than a fit, so a residual means cycles were lost "
            f"between them rather than that an effect went unnamed."
        )

    covered_cycles = sum(layer.analytical_cycles for layer in layers)
    all_cycles = covered_cycles + pooling + elementwise + transfers

    return CellDivergence(
        cell=cell,
        layers=tuple(layers),
        skipped=tuple(topology.skipped),
        approximations=tuple(topology.approximations),
        terms=named,
        covered_cycle_fraction=covered_cycles / all_cycles if all_cycles else 0.0,
        covered_op_fraction=len(layers) / len(operations) if operations else 0.0,
        analytical_serial_total=analytical_serial,
        simulated_cycles=float(result["simulation"]["simulated_cycles"]),
        scalesim_total=scalesim_total,
    )


# ---------------------------------------------------------------------------
# Rank fidelity, which Section 16.3 asks for beside the absolute error.
# ---------------------------------------------------------------------------


def rank_fidelity(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Kendall tau and pairwise comparison accuracy between two orderings.

    **Why this is reported and not only the absolute error.** The cost model
    exists to make compiler decisions, so what matters is whether it *orders*
    candidates the way the reference does. A learned cost model with an R squared
    of minus 1818 nonetheless achieved the best pairwise accuracy and produced
    the fastest programs in its comparison set, which is the evidence Section
    16.3 cites for the two coming apart. A project that only reported absolute
    error could not tell the difference.

    Tau b, so that ties on either side are handled rather than assumed away:
    this suite has many cells that are the same program at different optimization
    levels, and those tie exactly on both sides. Counting a tie as a discordance
    would report a disagreement where both tools said the same thing.

    Pairwise accuracy is the fraction of pairs the two orderings agree the sign
    of, over the pairs where **neither** is tied, because a pair where one tool
    is undecided is not a pair the other got right or wrong.
    """
    concordant = 0
    discordant = 0
    tied_left = 0
    tied_right = 0
    for index, (left_a, right_a) in enumerate(pairs):
        for left_b, right_b in pairs[index + 1 :]:
            left = (left_a > left_b) - (left_a < left_b)
            right = (right_a > right_b) - (right_a < right_b)
            if left == 0 and right == 0:
                continue
            if left == 0:
                tied_left += 1
                continue
            if right == 0:
                tied_right += 1
                continue
            if left == right:
                concordant += 1
            else:
                discordant += 1

    decided = concordant + discordant
    denominator = (
        (concordant + discordant + tied_left) * (concordant + discordant + tied_right)
    ) ** 0.5
    return {
        "n": float(len(pairs)),
        "kendall_tau_b": (
            (concordant - discordant) / denominator if denominator else 0.0
        ),
        "pairwise_accuracy": concordant / decided if decided else 0.0,
        "concordant": float(concordant),
        "discordant": float(discordant),
        "tied_on_this_project": float(tied_left),
        "tied_on_scalesim": float(tied_right),
    }


# ---------------------------------------------------------------------------
# One cell, end to end.
# ---------------------------------------------------------------------------


def divergence_for(
    result: dict[str, Any], npuisa_text: str, directory: Path
) -> CellDivergence:
    """Export one cell, run SCALE-Sim over it, and decompose the gap."""
    operations = npuisa_walk.attribute_transfers(
        npuisa_walk.walk(npuisa_text), npuisa_text
    )
    npuisa_walk.check_against_result(operations, result)

    topology = export(operations)
    config = architecture_config(int(result["cell"]["scratchpad_budget_bytes"]))

    reports = run_scalesim(
        topology_csv=topology.csv_text(),
        config_text=config,
        layout_csv=layout_text(topology.rows),
        directory=directory / "effective",
        expected_layers=len(topology.rows),
    )

    undilated = None
    if topology.has_dilation:
        undilated = run_scalesim(
            topology_csv=topology.csv_text(topology.undilated_rows),
            config_text=config,
            layout_csv=layout_text(topology.undilated_rows),
            directory=directory / "undilated",
            expected_layers=len(topology.undilated_rows),
        )

    return decompose(
        cell=result["cell"]["name"],
        operations=operations,
        topology=topology,
        reports=reports,
        undilated_reports=undilated,
        result=result,
    )


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scalesim_export.py",
        description=(
            "Section 16.3's SCALE-Sim cross validation over the committed cells."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        return _run(arguments)
    except (ScaleSimError, ResultSchemaError, npuisa_walk.WalkError) as failure:
        print(f"scalesim: {failure}", file=sys.stderr)
        return 2


def _run(arguments: argparse.Namespace) -> int:
    import tempfile

    paths = sorted(Path(arguments.results).glob("*.json"))
    results = [load_result(path) for path in paths]
    if arguments.models:
        wanted = set(arguments.models)
        results = [result for result in results if result["cell"]["model"] in wanted]
    if arguments.limit is not None:
        results = results[: arguments.limit]
    if not results:
        raise ScaleSimError("no committed cells matched")

    print(f"scalesim: installed tree sha256 {installed_tree_sha256()[:16]}")

    answers: list[CellDivergence] = []
    with tempfile.TemporaryDirectory(prefix="npu-scalesim-") as directory:
        root = Path(directory)
        compiler = Compiler(root / "models")
        for index, result in enumerate(results):
            answer = divergence_for(
                result, compiler.allocated_ir(result), root / f"cell{index}"
            )
            answers.append(answer)
            if arguments.verbose:
                print(
                    f"  {answer.cell}: divergence "
                    f"{answer.whole_model_relative_divergence * 100:+.2f}% over "
                    f"coverage {answer.covered_cycle_fraction:.3f}"
                )

    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                [
                    {
                        "cell": answer.cell,
                        "terms": answer.terms,
                        "covered_cycle_fraction": answer.covered_cycle_fraction,
                        "covered_op_fraction": answer.covered_op_fraction,
                        "analytical_serial_total": answer.analytical_serial_total,
                        "simulated_cycles": answer.simulated_cycles,
                        "scalesim_total": answer.scalesim_total,
                        "whole_model_relative_divergence": (
                            answer.whole_model_relative_divergence
                        ),
                        "layers": [asdict(layer) for layer in answer.layers],
                        "skipped": [asdict(entry) for entry in answer.skipped],
                        "approximations": [
                            asdict(entry) for entry in answer.approximations
                        ],
                    }
                    for answer in answers
                ],
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )

    worst = max(answers, key=lambda answer: abs(answer.whole_model_relative_divergence))
    print(
        f"scalesim: {len(answers)} cells. Worst whole model divergence "
        f"{worst.whole_model_relative_divergence * 100:+.2f}% on {worst.cell}, "
        f"at coverage {worst.covered_cycle_fraction:.3f}."
    )

    # Section 16.3: rank fidelity beside the absolute error, and no agreement
    # figure printed without a coverage fraction beside it.
    cells = rank_fidelity(
        [
            (answer.analytical_serial_total, float(answer.scalesim_total))
            for answer in answers
        ]
    )
    layers = rank_fidelity(
        [
            (layer.analytical_cycles, float(layer.scalesim_cycles))
            for answer in answers
            for layer in answer.layers
        ]
    )
    coverage = sum(answer.covered_cycle_fraction for answer in answers) / len(answers)
    print(
        f"scalesim: rank fidelity over cells, tau_b "
        f"{cells['kendall_tau_b']:.4f}, pairwise "
        f"{cells['pairwise_accuracy']:.4f}; over layers, tau_b "
        f"{layers['kendall_tau_b']:.4f}, pairwise "
        f"{layers['pairwise_accuracy']:.4f}. Mean covered cycle fraction "
        f"{coverage:.3f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
