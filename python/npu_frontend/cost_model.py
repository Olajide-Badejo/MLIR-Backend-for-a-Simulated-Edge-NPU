# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The Python mirror of ``include/NPU/Simulator/CostModel.h``.

Section 5.5 puts the machine's constants in exactly one home and requires a
Python mirror that a test asserts equal to the header. This is that mirror, and
``test/Python/test_cost_model_mirror.py`` is that test: it parses the constant
block out of the header and compares it, name by name and value by value, with
the table below.

**Why a mirror at all, when one home was the point.** The analysis, plotting and
reporting code of the later phases is Python and it needs these numbers: Section
16 computes cycles per instruction and arithmetic intensity, Phase P11 hands MAC
counts to Accelergy, and Phase P13's tiling experiments plot against the same
model the simulator charges with. Without a mirror each of those would grow its
own copy of whichever constant it happened to need, which is the drift the one
home rule exists to prevent, in the language the rule was not written in. A
mirror plus a test that fails when the two disagree is one home with two
readers; two hand copied sets of numbers is how a report starts lying.

The formulas are here as well as the constants, for the same reason and with the
same guarantee. ``test_cost_model_mirror.py`` parses the constants out of the
header and compares them name by name, and it then runs ``npu-sim`` over a real
program and asserts that the charges below reproduce the machine's own reported
statistics, so a formula that drifted from the C++ fails rather than quietly
producing a second cost model. That second half was missing when this docstring
first claimed it, which is D-0027.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The constants. Every one of these is an assumption with a stated uncertainty
# in the header, never a measurement, and the report says so wherever it quotes
# one. The reasoning lives in the header rather than being repeated here,
# because two copies of a justification drift exactly like two copies of a
# number.
# ---------------------------------------------------------------------------

#: The systolic array is this many processing elements on a side.
ARRAY_DIM = 16

#: Peak multiply accumulate throughput per cycle at f32.
PEAK_MACS_PER_CYCLE_F32 = 256

#: Peak multiply accumulate throughput per cycle at int8, four per f32 lane.
PEAK_MACS_PER_CYCLE_I8 = 1024

#: DRAM bandwidth in bytes per cycle, for a contiguous burst.
DRAM_BANDWIDTH_BYTES_PER_CYCLE = 16.0

#: The fixed cost of issuing one DMA descriptor, in cycles.
DMA_DESCRIPTOR_CYCLES = 64.0

#: The extra cost per element when a transfer's innermost stride is not one.
DMA_STRIDED_ELEMENT_CYCLES = 0.5

#: How many elements the elementwise unit consumes per cycle.
ELEMENTWISE_LANE_WIDTH = 16

#: The fixed issue overhead charged to every instruction, in cycles.
ISSUE_OVERHEAD_CYCLES = 4.0

#: The depth of the weight preload pipeline, in cycles.
WEIGHT_PRELOAD_CYCLES = 16.0

#: The name each constant carries in the C++ header, so the mirror test can
#: compare the two tables without either side guessing at the other's spelling.
#: A constant added to one side and not the other fails that test, which is the
#: whole mechanism.
HEADER_NAMES = {
    "kArrayDim": "ARRAY_DIM",
    "kPeakMacsPerCycleF32": "PEAK_MACS_PER_CYCLE_F32",
    "kPeakMacsPerCycleI8": "PEAK_MACS_PER_CYCLE_I8",
    "kDramBandwidthBytesPerCycle": "DRAM_BANDWIDTH_BYTES_PER_CYCLE",
    "kDmaDescriptorCycles": "DMA_DESCRIPTOR_CYCLES",
    "kDmaStridedElementCycles": "DMA_STRIDED_ELEMENT_CYCLES",
    "kElementwiseLaneWidth": "ELEMENTWISE_LANE_WIDTH",
    "kIssueOverheadCycles": "ISSUE_OVERHEAD_CYCLES",
    "kWeightPreloadCycles": "WEIGHT_PRELOAD_CYCLES",
}

#: The value of every mirrored constant, keyed by its Python name.
VALUES: dict[str, float] = {
    "ARRAY_DIM": ARRAY_DIM,
    "PEAK_MACS_PER_CYCLE_F32": PEAK_MACS_PER_CYCLE_F32,
    "PEAK_MACS_PER_CYCLE_I8": PEAK_MACS_PER_CYCLE_I8,
    "DRAM_BANDWIDTH_BYTES_PER_CYCLE": DRAM_BANDWIDTH_BYTES_PER_CYCLE,
    "DMA_DESCRIPTOR_CYCLES": DMA_DESCRIPTOR_CYCLES,
    "DMA_STRIDED_ELEMENT_CYCLES": DMA_STRIDED_ELEMENT_CYCLES,
    "ELEMENTWISE_LANE_WIDTH": ELEMENTWISE_LANE_WIDTH,
    "ISSUE_OVERHEAD_CYCLES": ISSUE_OVERHEAD_CYCLES,
    "WEIGHT_PRELOAD_CYCLES": WEIGHT_PRELOAD_CYCLES,
}


# ---------------------------------------------------------------------------
# The charges.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeCharge:
    """What one compute instruction costs.

    ``macs`` is raw and stays raw: utilization describes how long the array was
    busy, not how many multiplies happened, and nothing in the energy path ever
    sees ``effective_macs``.
    """

    cycles: float
    macs: int
    effective_macs: float
    utilization: float
    delta: float


def dma_cycles(byte_count: int, element_count: int, innermost_stride: int) -> float:
    """Bytes over bandwidth, plus the descriptor, plus the stride penalty."""
    if byte_count <= 0 or element_count <= 0:
        return DMA_DESCRIPTOR_CYCLES
    cycles = byte_count / DRAM_BANDWIDTH_BYTES_PER_CYCLE + DMA_DESCRIPTOR_CYCLES
    if innermost_stride != 1:
        cycles += element_count * DMA_STRIDED_ELEMENT_CYCLES
    return cycles


def elementwise_cycles(element_count: int) -> float:
    """Elements over lane width."""
    if element_count <= 0:
        return 0.0
    return element_count / ELEMENTWISE_LANE_WIDTH


def gemm_charge(rows: int, reduction: int, columns: int, peak: int) -> ComputeCharge:
    """The weight stationary charge for ``rows`` activation rows.

    The weight matrix is ``reduction`` by ``columns`` and is folded into tiles
    of at most ``ARRAY_DIM`` by ``ARRAY_DIM``. Each tile is charged
    ``tile_macs / (utilization * delta * peak)``, so a tile occupying a quarter
    of the array takes four times as long as its MAC count alone suggests.
    """
    if rows <= 0 or reduction <= 0 or columns <= 0 or peak <= 0:
        return ComputeCharge(0.0, 0, 0.0, 1.0, 1.0)

    delta = rows / (rows + WEIGHT_PRELOAD_CYCLES)
    cycles = 0.0
    macs = 0
    weighted_utilization = 0.0

    for base_row in range(0, reduction, ARRAY_DIM):
        tile_rows = min(ARRAY_DIM, reduction - base_row)
        for base_column in range(0, columns, ARRAY_DIM):
            tile_columns = min(ARRAY_DIM, columns - base_column)
            utilization = (tile_rows / ARRAY_DIM) * (tile_columns / ARRAY_DIM)
            tile_macs = rows * tile_rows * tile_columns
            macs += tile_macs
            weighted_utilization += utilization * tile_macs
            cycles += tile_macs / (utilization * delta * peak)

    return ComputeCharge(
        cycles=cycles,
        macs=macs,
        effective_macs=cycles * peak,
        utilization=weighted_utilization / macs if macs else 1.0,
        delta=delta,
    )


def overlap_fraction(dma_total: float, compute_total: float, total: float) -> float:
    """The fraction of the shorter timeline hidden underneath the longer one.

    0 means the two timelines ran end to end and 1 means the shorter one is
    entirely hidden. A program with nothing on one port reports 0, because there
    was no shorter timeline to hide.
    """
    shorter = min(dma_total, compute_total)
    if shorter <= 0.0:
        return 0.0
    return (dma_total + compute_total - total) / shorter
