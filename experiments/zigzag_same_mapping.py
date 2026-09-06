# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""ZigZag under the mapping this compiler chose, per Section 16.5.

    python experiments/zigzag_same_mapping.py --verbose
    python experiments/zigzag_same_mapping.py --json experiments/results-zigzag/same-mapping.json

Section 16.5 asks for one thing and forbids another. It asks for the mapping the
tiling pass actually chose to be exported in ZigZag and Timeloop mapping form and
for the **cost under that one mapping** to be compared. It forbids comparing two
totals produced by two mappers, which is the comparison that looks like this one
and answers a different question: a mapper that searched a different space
telling you its best is not a second opinion on your mapping, it is a first
opinion on its own.

So every ZigZag run here is given a **complete** temporal ordering, which is the
mapping the pass recorded in `npu.tiling_choice`, plus the spatial mapping the
array imposes. `LayerTemporalOrdering.is_complete` is ZigZag's own predicate for
"this ordering fixes the mapping", and when it holds ZigZag skips its search
engine entirely and evaluates exactly the loop nest it was handed. This module
asserts the same completeness itself before every run, dimension by dimension,
because a factor that silently did not multiply out would put ZigZag back into
its own search and the comparison would quietly become the forbidden one.

## What is compared

`npu.tiling_choice.makespan_cycles`, which is the Section 5.5 two port makespan
the tiling search chose the mapping by, against ZigZag's `latency_total2` for the
same layer under the same loop nest. Both are cycles for one layer on a sixteen
by sixteen array with one scratchpad and one DRAM, and both include the transfer
time rather than only the arithmetic.

**Energy is not compared and the accelerator description says so by carrying
zeros.** This project has no per access energy model at this level; Section 16.4
owns energy and reaches it through Accelergy with real coefficients. Handing
ZigZag invented `r_cost` numbers and reporting the joules that came back would be
reporting the invention. The prediction registered for this comparison
deliberately makes no energy claim for the same reason.

## The one thing the translation cannot preserve, named and measured

This project folds a weight matrix of `reduction` by `columns` onto the array in
bands of at most sixteen, and charges each band by how much of the array it
fills: a reduction of 72 is four full bands and one of eight, and the last band
costs what a full one does. **That is an uneven fold and ZigZag cannot express
it**, because a ZigZag spatial mapping unrolls a loop dimension by a factor and a
factor has to divide its dimension.

So the spatial mapping exported here is the largest product of divisors of the
reduction dimensions that fits the array, and it is generally **smaller** than
what this project's own model packs onto the same hardware. The gap is recorded
per layer as `array_fill_here` against `array_fill_zigzag` rather than
absorbed into the cycle comparison, because a divergence with a named mechanism
beside it is a result and the same divergence without one is a discrepancy.

## What bounds the exploration

The layers, and nothing else. Section 16.5 says to bound it to the layers
actually tiled, so the population is exactly the layers carrying
`npu.tiling_choice` at the suite's tight cells and at every budget
`experiments/three_arms.py` sweeps, deduplicated by the mapping itself: two
budgets that produce the same loop nest on the same layer are one run.

Wall clock and peak resident memory are recorded per layer and in total. The
ceiling is `MEMORY_CEILING_MB` below, and it is this machine's, not a target: if
a run crosses it the module **stops and records why**. It does not raise the
ceiling, and it never proposes changing the WSL configuration, which is
Section 16.5's instruction in those words.

## If ZigZag prefers a different mapping

`--search` asks the second question, and it is a different one: ZigZag's own
`loma` engine over the same layer with no ordering supplied, scored by **ZigZag's
cost model on both sides**. That is one mapper's opinion of two mappings rather
than two mappers' totals, so it is a legitimate question about this project's
search and it is reported separately from the same mapping comparison. A margin
over `MATERIAL_FRACTION` is recorded as a defect candidate with the layer, the
budget and the ordering that reproduces it. **The search is never retuned to
close it.**
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from roofline import _mlir_python_packages_dir  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

from npu_frontend import cost_model  # noqa: E402
from npu_frontend.compile import CompileError, compile_model  # noqa: E402
from npu_frontend.model_generator import MODELS, generate_model  # noqa: E402
from three_arms import ARMS, SWEPT  # noqa: E402

#: The array this project has, from the one place its size is written down.
ARRAY_DIM: int = cost_model.ARRAY_DIM

#: Every value in this compiler is fp32, and ZigZag counts precision in bits.
ELEMENT_BITS: int = 32

#: DRAM bandwidth, in bits per cycle, from this project's own constant. ZigZag
#: port bandwidths are bits per cycle, so the conversion is the only arithmetic.
DRAM_BITS_PER_CYCLE: int = int(cost_model.DRAM_BANDWIDTH_BYTES_PER_CYCLE * 8)

#: What the scratchpad has to sustain to keep the array fed, derived rather than
#: chosen: a weight stationary array of `ARRAY_DIM` rows consumes one activation
#: per row per cycle and produces one partial sum per column per cycle.
SCRATCHPAD_BITS_PER_CYCLE: int = ARRAY_DIM * ELEMENT_BITS

#: DRAM, which this machine models as unbounded. ZigZag needs a number.
DRAM_BITS: int = 10_000_000_000

#: The memory ceiling for this comparison, in mebibytes. It is a fraction of
#: what this machine has, so that a run which needs more is a result rather than
#: a machine that has to be reconfigured to produce one.
MEMORY_CEILING_MB: int = 8192

#: The wall clock ceiling for the whole comparison, in seconds.
WALL_CLOCK_CEILING_SECONDS: float = 900.0

#: How much better a different mapping has to be before it is worth a defect.
MATERIAL_FRACTION: float = 0.10

#: The configurations of `experiments/three_arms.py` in which tiling can fire.
#: The two spilling configurations ablate the tiling pass, so they carry no
#: mapping to export and are not swept here.
TILING_CONFIGURATIONS: tuple[str, ...] = tuple(
    name
    for name, options in ARMS.items()
    if options.get("ablate") != "npu-tile-to-scratchpad"
)

#: ADR 0008's tight budgets, which are the suite's cells. Read off the registry
#: rather than written down, for the reason `tight_budget_floor.py` states.
TIGHT: dict[str, int] = {name: spec.tight_budget for name, spec in MODELS.items()}


class ExportError(Exception):
    """Raised when the export cannot be made faithful rather than approximate."""


# ---------------------------------------------------------------------------
# Reading the compiler's own answer.
# ---------------------------------------------------------------------------

_LOCATION = re.compile(r'^#(?P<id>loc\d*) = loc\("(?P<name>[^"]*)"\)', re.MULTILINE)

_OPERATION = re.compile(
    r"npu\.(?P<op>conv2d|matmul)\s+ins\((?P<ins>[^)]*)\)\s+"
    r"outs\((?P<outs>[^)]*)\)\s*(?P<attributes>\{.*\})?\s*->\s*"
    r"(?P<result>tensor<[^>]*>)\s+loc\(#(?P<loc>loc\d*)\)"
)

_TENSOR = re.compile(r"tensor<(?P<shape>[0-9x]*)x?(?P<element>[a-z0-9]+)>")

_ARRAY = re.compile(r"(?P<name>[a-z_]+) = array<i64:(?P<values>[^>]*)>")

_INTEGER = re.compile(r"(?P<name>[a-z_]+) = (?P<value>-?\d+) : i64")

_FLOAT = re.compile(r"(?P<name>[a-z_]+) = (?P<value>-?[\d.]+e?[+-]?\d*) : f64")

_STRING = re.compile(r'(?P<name>[a-z_]+) = "(?P<value>[^"]*)"')

_CHOICE = re.compile(r"npu\.tiling_choice = \{(?P<body>[^}]*)\}")


def _shape(text: str) -> tuple[int, ...]:
    """The extents of one tensor type."""
    match = _TENSOR.fullmatch(text.strip())
    if match is None:
        raise ExportError(f"{text!r} is not a tensor type this module can read.")
    dimensions = match.group("shape")
    if not dimensions:
        return ()
    return tuple(int(part) for part in dimensions.split("x") if part)


def _operand_shapes(clause: str) -> list[tuple[int, ...]]:
    """The types of an `ins` or `outs` clause, in order."""
    if ":" not in clause:
        raise ExportError(f"{clause!r} has no type list.")
    _, types = clause.rsplit(":", 1)
    return [_shape(part) for part in types.split(",") if part.strip()]


@dataclass(frozen=True)
class Choice:
    """`npu.tiling_choice`, as the pass wrote it."""

    temporal_tiles: tuple[int, ...]
    spatial_factors: tuple[int, ...]
    loop_order: str
    tile_count: int
    tile_bytes: int
    makespan_cycles: float
    strategy: str


def _choice(attributes: str) -> Choice | None:
    """The mapping attribute off one operation's attribute dictionary."""
    match = _CHOICE.search(attributes)
    if match is None:
        return None
    body = match.group("body")
    arrays = {
        name: tuple(int(value) for value in values.split(","))
        for name, values in _ARRAY.findall(body)
    }
    integers = {name: int(value) for name, value in _INTEGER.findall(body)}
    floats = {name: float(value) for name, value in _FLOAT.findall(body)}
    strings = dict(_STRING.findall(body))
    missing = [
        field_name
        for field_name, source in (
            ("temporal_tiles", arrays),
            ("spatial_factors", arrays),
            ("tile_count", integers),
            ("tile_bytes", integers),
            ("makespan_cycles", floats),
            ("loop_order", strings),
            ("strategy", strings),
        )
        if field_name not in source
    ]
    if missing:
        raise ExportError(
            f"npu.tiling_choice is missing {missing}, which means the attribute "
            f"changed shape and this exporter reads a mapping that is no longer "
            f"the one the pass writes: {body!r}"
        )
    return Choice(
        temporal_tiles=arrays["temporal_tiles"],
        spatial_factors=arrays["spatial_factors"],
        loop_order=strings["loop_order"],
        tile_count=integers["tile_count"],
        tile_bytes=integers["tile_bytes"],
        makespan_cycles=floats["makespan_cycles"],
        strategy=strings["strategy"],
    )


@dataclass
class Problem:
    """One layer's own shape, read off the untiled program."""

    kind: str
    name: str
    #: Convolution: batch, output channels per group, groups, output width,
    #: output height, input channels per group, kernel width, kernel height.
    #: Matmul: reduction, rows, columns, in ZigZag's own C, D, K order.
    sizes: dict[str, int]
    strides: tuple[int, int] = (1, 1)
    dilations: tuple[int, int] = (1, 1)
    pads: tuple[int, int, int, int] = (0, 0, 0, 0)
    input_height: int = 0
    input_width: int = 0


def problems(text: str) -> dict[str, Problem]:
    """Every convolution and matmul in one program, by the name its location
    carries.

    The location is what joins a tiled operation to its own untiled shape, and
    it survives the rewrite because MLIR propagates it onto the operations a
    pattern builds. `test_zigzag_same_mapping.py` asserts that rather than
    trusting it.
    """
    names = {
        match.group("id"): match.group("name") for match in _LOCATION.finditer(text)
    }
    found: dict[str, Problem] = {}
    for match in _OPERATION.finditer(text):
        name = names.get(match.group("loc"), match.group("loc"))
        attributes = match.group("attributes") or ""
        operands = _operand_shapes(match.group("ins"))
        result = _shape(match.group("result"))
        if match.group("op") == "conv2d":
            found[name] = _conv_problem(name, operands, result, attributes)
        else:
            found[name] = _matmul_problem(name, operands, result)
    return found


def _integer_attribute(attributes: str, name: str, default: int) -> int:
    for found, value in _INTEGER.findall(attributes):
        if found == name:
            return int(value)
    return default


def _array_attribute(
    attributes: str, name: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    for found, values in _ARRAY.findall(attributes):
        if found == name:
            return tuple(int(value) for value in values.split(","))
    return default


def _conv_problem(
    name: str,
    operands: list[tuple[int, ...]],
    result: tuple[int, ...],
    attributes: str,
) -> Problem:
    activation, weight = operands[0], operands[1]
    group = _integer_attribute(attributes, "group", 1)
    strides = _array_attribute(attributes, "strides", (1, 1))
    dilations = _array_attribute(attributes, "dilations", (1, 1))
    pads = _array_attribute(attributes, "pads", (0, 0, 0, 0))
    return Problem(
        kind="conv2d",
        name=name,
        sizes={
            "B": result[0],
            "K": result[1] // group,
            "G": group,
            "OY": result[2],
            "OX": result[3],
            "C": weight[1],
            "FY": weight[2],
            "FX": weight[3],
        },
        strides=(int(strides[0]), int(strides[1])),
        dilations=(int(dilations[0]), int(dilations[1])),
        pads=tuple(int(value) for value in pads),  # type: ignore[arg-type]
        input_height=activation[2],
        input_width=activation[3],
    )


def _matmul_problem(
    name: str, operands: list[tuple[int, ...]], result: tuple[int, ...]
) -> Problem:
    activation, weight = operands[0], operands[1]
    return Problem(
        kind="matmul",
        name=name,
        sizes={"C": weight[0], "D": activation[0], "K": result[1]},
    )


# ---------------------------------------------------------------------------
# The mapping, translated.
# ---------------------------------------------------------------------------


def divisors(value: int) -> list[int]:
    """Every divisor of `value`, ascending."""
    found = [candidate for candidate in range(1, value + 1) if value % candidate == 0]
    return found


def largest_divisor_within(value: int, limit: int) -> int:
    """The largest divisor of `value` that is at most `limit`."""
    return max(candidate for candidate in divisors(value) if candidate <= limit)


def fill_within(extents: list[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    """The largest product of one divisor per extent that fits `limit`.

    Ties go to the assignment that unrolls the earlier dimension further, which
    makes the answer a function of the layer rather than of iteration order.
    This is where this project's uneven fold becomes an even one: a reduction of
    72 over three dimensions of 8, 3 and 3 can put 12 on the array and not 16,
    and the shortfall is reported rather than absorbed.
    """
    best: list[tuple[str, int]] = [(name, 1) for name, _ in extents]
    best_product = 1

    def walk(index: int, chosen: list[tuple[str, int]], product: int) -> None:
        nonlocal best, best_product
        if index == len(extents):
            if product > best_product:
                best, best_product = list(chosen), product
            return
        name, extent = extents[index]
        for factor in sorted(divisors(extent), reverse=True):
            if product * factor > limit:
                continue
            chosen.append((name, factor))
            walk(index + 1, chosen, product * factor)
            chosen.pop()

    walk(0, [], 1)
    return best


@dataclass
class Mapping:
    """One layer's mapping, in the form both tools can be handed."""

    spatial: dict[str, list[tuple[str, int]]]
    #: Innermost first, which is the order ZigZag's `temporal_ordering` takes.
    temporal: list[tuple[str, int]]
    #: Where in `temporal` the tile loops begin. Everything below it runs inside
    #: one tile and everything at or above it walks the tiles, which is the one
    #: place the two levels of a Timeloop mapping can be cut. Carried rather than
    #: recomputed, because a boundary found by halving the list is right only
    #: while the loops happen to be balanced.
    tile_boundary: int
    array_fill_here: float
    array_fill_zigzag: float


def conv_mapping(problem: Problem, choice: Choice) -> Mapping:
    """The convolution's mapping, from the five tile extents the pass recorded.

    `temporal_tiles` is in the pass's own domain order, which its search states
    as (batch, group, channel, height, width), and the entries are tile
    **extents** rather than counts: `tile_count` is the product of each extent
    divided into its dimension.
    """
    sizes = problem.sizes
    batch_tile, group_tile, channel_tile, height_tile, width_tile = (
        choice.temporal_tiles
    )

    columns = largest_divisor_within(channel_tile, ARRAY_DIM)
    reduction = fill_within(
        [("C", sizes["C"]), ("FY", sizes["FY"]), ("FX", sizes["FX"])], ARRAY_DIM
    )
    spatial = {"D1": [("K", columns)], "D2": reduction}

    unrolled = dict(reduction)
    temporal: list[tuple[str, int]] = [
        # Inside one tile, in the order `cost_model.gemm_charge` walks: the
        # activation rows stream through a loaded array, so the output positions
        # are innermost; then the column bands; then the reduction bands, which
        # are the loop a weight stationary array pays a preload for. The pass
        # does not choose this order and could not: it is what the hardware is.
        ("OX", width_tile),
        ("OY", height_tile),
        ("B", batch_tile),
        ("K", channel_tile // columns),
        ("C", sizes["C"] // unrolled.get("C", 1)),
        ("FY", sizes["FY"] // unrolled.get("FY", 1)),
        ("FX", sizes["FX"] // unrolled.get("FX", 1)),
        ("G", group_tile),
        # The tile loops, in the pass's own domain order, width innermost.
        ("OX", sizes["OX"] // width_tile),
        ("OY", sizes["OY"] // height_tile),
        ("K", sizes["K"] // channel_tile),
        ("G", sizes["G"] // group_tile),
        ("B", sizes["B"] // batch_tile),
    ]

    here = _fold_fill(sizes["C"] * sizes["FY"] * sizes["FX"], sizes["K"])
    product = math.prod(factor for _, factor in reduction) * columns
    return Mapping(
        spatial=spatial,
        temporal=temporal,
        tile_boundary=8,
        array_fill_here=here,
        array_fill_zigzag=product / (ARRAY_DIM * ARRAY_DIM),
    )


def matmul_mapping(problem: Problem, choice: Choice) -> Mapping:
    """The matmul's mapping. Its five slot record is (M, 1, N, 1, 1)."""
    sizes = problem.sizes
    row_tile, _, column_tile, _, _ = choice.temporal_tiles

    columns = largest_divisor_within(column_tile, ARRAY_DIM)
    reduction = fill_within([("C", sizes["C"])], ARRAY_DIM)
    spatial = {"D1": [("K", columns)], "D2": reduction}

    unrolled = dict(reduction)
    temporal: list[tuple[str, int]] = [
        # The same order, with the matmul's own names: rows innermost, then the
        # column bands, then the reduction bands.
        ("D", row_tile),
        ("K", column_tile // columns),
        ("C", sizes["C"] // unrolled.get("C", 1)),
        ("K", sizes["K"] // column_tile),
        ("D", sizes["D"] // row_tile),
    ]

    here = _fold_fill(sizes["C"], sizes["K"])
    product = math.prod(factor for _, factor in reduction) * columns
    return Mapping(
        spatial=spatial,
        temporal=temporal,
        tile_boundary=3,
        array_fill_here=here,
        array_fill_zigzag=product / (ARRAY_DIM * ARRAY_DIM),
    )


def _fold_fill(reduction: int, columns: int) -> float:
    """How much of the array this project's own fold occupies, on average.

    `cost_model.gemm_charge` walks the weight matrix in bands of `ARRAY_DIM` and
    charges each band by its own occupancy, so the average over the bands
    weighted by their multiply accumulates is the number the cycle count already
    contains. It is computed here rather than read out of a charge because the
    comparison needs it per layer and beside ZigZag's, not summed.
    """
    total = 0
    filled = 0
    for base_row in range(0, reduction, ARRAY_DIM):
        rows = min(ARRAY_DIM, reduction - base_row)
        for base_column in range(0, columns, ARRAY_DIM):
            wide = min(ARRAY_DIM, columns - base_column)
            macs = rows * wide
            total += ARRAY_DIM * ARRAY_DIM
            filled += macs
    return filled / total if total else 0.0


def check_complete(problem: Problem, mapping: Mapping) -> None:
    """Every dimension's temporal factors times its spatial factor is its size.

    This is ZigZag's `is_complete` asserted from this side, and it is the whole
    guarantee that the comparison is under one mapping: an ordering that fails it
    is not rejected by ZigZag, it is treated as a hint and the search runs, which
    would turn this into the comparison Section 16.5 forbids.
    """
    product: dict[str, int] = {}
    for name, factor in mapping.temporal:
        product[name] = product.get(name, 1) * factor
    for entries in mapping.spatial.values():
        for name, factor in entries:
            product[name] = product.get(name, 1) * factor
    for name, size in problem.sizes.items():
        if product.get(name, 1) != size:
            raise ExportError(
                f"{problem.name}: the exported mapping covers {name} "
                f"{product.get(name, 1)} times and the layer is {size}. A "
                f"mapping that does not multiply out is a hint to ZigZag rather "
                f"than an ordering, and this comparison is only meaningful under "
                f"one mapping."
            )


# ---------------------------------------------------------------------------
# The two files ZigZag reads, and the one Timeloop would.
# ---------------------------------------------------------------------------


def accelerator_text(budget: int) -> str:
    """This machine, in ZigZag's accelerator form.

    Every number is this project's own: the array is `ARRAY_DIM` by `ARRAY_DIM`
    at `ELEMENT_BITS`, the scratchpad is the budget the cell was compiled at, its
    port to DRAM runs at this project's DRAM bandwidth, and its ports to the
    array run at what the array consumes. The energy costs are zero because this
    project has no energy model at this level and Section 16.4 owns the one it
    does have.
    """
    return f"""name: npu_p13

memories:
  scratchpad:
    size: {budget * 8}
    r_cost: 0.0
    w_cost: 0.0
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: {SCRATCHPAD_BITS_PER_CYCLE}
        bandwidth_max: {SCRATCHPAD_BITS_PER_CYCLE}
        allocation:
          - I1, tl
      - name: rw_port_2
        type: read_write
        bandwidth_min: {SCRATCHPAD_BITS_PER_CYCLE}
        bandwidth_max: {SCRATCHPAD_BITS_PER_CYCLE}
        allocation:
          - I2, tl
      - name: rw_port_3
        type: read_write
        bandwidth_min: {SCRATCHPAD_BITS_PER_CYCLE}
        bandwidth_max: {SCRATCHPAD_BITS_PER_CYCLE}
        allocation:
          - O, tl
          - O, fl
      - name: rw_port_4
        type: read_write
        bandwidth_min: {DRAM_BITS_PER_CYCLE}
        bandwidth_max: {DRAM_BITS_PER_CYCLE}
        allocation:
          - I1, fh
          - I2, fh
          - O, fh
          - O, th
    served_dimensions: [D1, D2]

  dram:
    size: {DRAM_BITS}
    r_cost: 0.0
    w_cost: 0.0
    area: 0
    latency: 1
    operands: [I1, I2, O]
    ports:
      - name: rw_port_1
        type: read_write
        bandwidth_min: {DRAM_BITS_PER_CYCLE}
        bandwidth_max: {DRAM_BITS_PER_CYCLE}
        allocation:
          - I1, fh
          - I1, tl
          - I2, fh
          - I2, tl
          - O, fh
          - O, tl
          - O, fl
          - O, th
    served_dimensions: [D1, D2]

operational_array:
  input_precision: [{ELEMENT_BITS}, {ELEMENT_BITS}]
  unit_energy: 0.0
  unit_area: 1
  dimensions: [D1, D2]
  sizes: [{ARRAY_DIM}, {ARRAY_DIM}]
"""


def mapping_text(mapping: Mapping, *, ordered: bool = True) -> str:
    """The mapping, in ZigZag's mapping form.

    `ordered` off leaves the temporal ordering out, which is what `--search`
    hands ZigZag when it asks the second question. Nothing else differs between
    the two, so the spatial mapping is held fixed across both.
    """
    lines = ["- name: default", "  spatial_mapping:"]
    for dimension in sorted(mapping.spatial):
        entries = mapping.spatial[dimension]
        if not entries:
            continue
        lines.append(f"    {dimension}:")
        for name, factor in entries:
            lines.append(f"      - {name}, {factor}")
    lines += [
        "  memory_operand_links:",
        "    O: O",
        "    W: I2",
        "    I: I1",
    ]
    if ordered:
        lines.append("  temporal_ordering:")
        for name, factor in mapping.temporal:
            if factor > 1:
                lines.append(f"    - [{name}, {factor}]")
    return "\n".join(lines) + "\n"


def workload_entry(problem: Problem) -> dict[str, Any]:
    """One layer, in ZigZag's workload form.

    The convolution's equation, loop dimension names and padding fields are
    copied from `zigzag.parser.onnx.conv_parser` and the matmul's from
    `gemm_parser`, rather than composed here, for the reason
    `scalesim_export.TOPOLOGY_HEADER` gives about the column order: an input
    format written from memory beside the file that defines it is the shape of
    error this project keeps finding.
    """
    sizes = problem.sizes
    if problem.kind == "conv2d":
        return {
            "id": 0,
            "name": problem.name,
            "operator_type": "Conv",
            "equation": "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]",
            "loop_dims": ["B", "K", "G", "OX", "OY", "C", "FX", "FY"],
            "loop_sizes": [
                sizes["B"],
                sizes["K"],
                sizes["G"],
                sizes["OX"],
                sizes["OY"],
                sizes["C"],
                sizes["FX"],
                sizes["FY"],
            ],
            "dimension_relations": [
                f"ix={problem.strides[0]}*ox+{problem.dilations[0]}*fx",
                f"iy={problem.strides[1]}*oy+{problem.dilations[1]}*fy",
            ],
            "operand_precision": {
                "W": ELEMENT_BITS,
                "I": ELEMENT_BITS,
                "O_final": ELEMENT_BITS,
                "O": ELEMENT_BITS,
            },
            "pr_loop_dims": ["IX", "IY"],
            "pr_loop_sizes": [problem.input_width, problem.input_height],
            "padding": [
                [problem.pads[0], problem.pads[2]],
                [problem.pads[1], problem.pads[3]],
            ],
        }
    return {
        "id": 0,
        "name": problem.name,
        "operator_type": "Gemm",
        "equation": "O[d][k]+=I[d][c]*W[c][k]",
        "loop_dims": ["C", "D", "K"],
        "loop_sizes": [sizes["C"], sizes["D"], sizes["K"]],
        "dimension_relations": [],
        "operand_precision": {
            "W": ELEMENT_BITS,
            "I": ELEMENT_BITS,
            "O_final": ELEMENT_BITS,
            "O": ELEMENT_BITS,
        },
    }


#: Timeloop's own names for the loop dimensions of a convolution, which are not
#: ZigZag's. The pairs are written here once so the export is a translation with
#: a table rather than a substitution scattered through a formatter.
TIMELOOP_NAMES: dict[str, str] = {
    "B": "N",
    "K": "K",
    "C": "C",
    "OY": "P",
    "OX": "Q",
    "FY": "R",
    "FX": "S",
    "G": "G",
    "D": "P",
}


def timeloop_text(problem: Problem, mapping: Mapping) -> str:
    """The same mapping in Timeloop's mapping form.

    Section 16.5 asks for both forms and this project installs no Timeloop, so
    this is an artefact rather than an input to a run here: the file is written,
    its factors are the same factors, and nothing in this module claims a
    Timeloop number. `docs/adr/0003-resolved-tool-matrix.md` records why the tool
    is absent, and Section 16.5's own warning about `pip install timeloop`
    fetching an unrelated scheduler is the reason it is not casually installed.
    """
    spatial: list[str] = []
    for entries in mapping.spatial.values():
        for name, factor in entries:
            spatial.append(f"{TIMELOOP_NAMES.get(name, name)}={factor}")

    inner: dict[str, int] = {}
    outer: dict[str, int] = {}
    for index, (name, factor) in enumerate(mapping.temporal):
        target = inner if index < mapping.tile_boundary else outer
        key = TIMELOOP_NAMES.get(name, name)
        target[key] = target.get(key, 1) * factor

    def factors(values: dict[str, int]) -> str:
        return " ".join(f"{name}={factor}" for name, factor in values.items())

    lines = [
        f"# {problem.name}, {problem.kind}, exported from npu.tiling_choice.",
        "mapping:",
        "  - target: DRAM",
        "    type: temporal",
        f"    factors: {factors(outer)}",
        f"    permutation: {''.join(outer)}",
        "  - target: scratchpad",
        "    type: temporal",
        f"    factors: {factors(inner)}",
        f"    permutation: {''.join(inner)}",
        "  - target: PE",
        "    type: spatial",
        f"    factors: {' '.join(spatial)}",
        f"    permutation: {''.join(part.split('=')[0] for part in spatial)}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Running it.
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One layer, one budget, and the mapping the pass chose for it."""

    model: str
    budget: int
    layer: str
    kind: str
    problem: Problem
    choice: Choice
    configurations: list[str] = field(default_factory=list)


def npu_stage(onnx: Path, budget: int | None, ablate: str | None) -> str:
    """The tensor level half at one budget, which is where the mapping lives."""
    return compile_model(onnx, level=2, emit="npu", budget=budget, ablate=ablate).text


def collect(models: list[str]) -> list[Cell]:
    """Every layer the pass tiled, at every budget this phase swept.

    Deduplicated by the mapping rather than by the cell: two budgets that produce
    the same loop nest on the same layer are one ZigZag run, and both budgets are
    recorded against it. That is what bounds the exploration to the layers
    actually tiled without running the same layer twenty times.
    """
    cells: dict[tuple[Any, ...], Cell] = {}
    with tempfile.TemporaryDirectory(prefix="zigzag-mapping-") as directory:
        work = Path(directory)
        for model in models:
            batch = int(MODELS[model].input_shape[0])
            onnx = generate_model(model, work, batch=batch)
            budgets = sorted({*SWEPT.get(model, ()), TIGHT[model]})
            shapes: dict[str | None, dict[str, Problem]] = {}
            for configuration in TILING_CONFIGURATIONS:
                ablate = ARMS[configuration].get("ablate")
                if ablate not in shapes:
                    shapes[ablate] = problems(npu_stage(onnx, None, ablate))
                for budget in budgets:
                    try:
                        text = npu_stage(onnx, budget, ablate)
                    except CompileError:
                        continue
                    for name, choice in _choices(text).items():
                        problem = shapes[ablate].get(name)
                        if problem is None:
                            raise ExportError(
                                f"{model} at {budget} tiled {name!r} and the "
                                f"untiled program has no layer of that name, so "
                                f"the shape the mapping refers to cannot be read."
                            )
                        key = (
                            model,
                            budget,
                            name,
                            choice.temporal_tiles,
                            problem.kind,
                        )
                        if key in cells:
                            cells[key].configurations.append(configuration)
                            continue
                        cells[key] = Cell(
                            model=model,
                            budget=budget,
                            layer=name,
                            kind=problem.kind,
                            problem=problem,
                            choice=choice,
                            configurations=[configuration],
                        )
    return sorted(
        cells.values(), key=lambda cell: (cell.model, cell.budget, cell.layer)
    )


def _choices(text: str) -> dict[str, Choice]:
    """One mapping per layer name, and a refusal if two tiles disagree."""
    names = {
        match.group("id"): match.group("name") for match in _LOCATION.finditer(text)
    }
    found: dict[str, Choice] = {}
    for match in _OPERATION.finditer(text):
        choice = _choice(match.group("attributes") or "")
        if choice is None:
            continue
        name = names.get(match.group("loc"), match.group("loc"))
        if name in found and found[name] != choice:
            raise ExportError(
                f"{name!r} carries two different mappings in one program, which "
                f"means the tiles of one layer were not tiled by one decision."
            )
        found[name] = choice
    return found


@dataclass
class Comparison:
    """One layer's two cycle counts, under one mapping."""

    model: str
    budget: int
    layer: str
    kind: str
    configurations: list[str]
    loop_sizes: dict[str, int]
    temporal_tiles: list[int]
    tile_count: int
    tile_bytes: int
    spatial_mapping: dict[str, list[str]]
    temporal_ordering: list[list[Any]]
    here_cycles: float
    zigzag_cycles: float
    zigzag_ideal_cycles: float
    zigzag_ideal_temporal_cycles: float
    array_fill_here: float
    array_fill_zigzag: float
    seconds: float
    evaluated_ordering: list[str] = field(default_factory=list)
    searched_cycles: float | None = None
    searched_margin: float | None = None
    searched_ordering: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """This project's cycles over ZigZag's, under the one mapping."""
        return self.here_cycles / self.zigzag_cycles if self.zigzag_cycles else 0.0


def peak_memory_mb() -> float:
    """This process's high water mark, in mebibytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_zigzag(
    problem: Problem,
    mapping: Mapping,
    budget: int,
    work: Path,
    *,
    ordered: bool = True,
) -> dict[str, Any]:
    """One ZigZag evaluation, under the mapping it is handed."""
    from zigzag.api import get_hardware_performance_zigzag

    accelerator = work / "accelerator.yaml"
    accelerator.write_text(accelerator_text(budget), encoding="utf-8")
    mapping_file = work / ("mapping.yaml" if ordered else "mapping-free.yaml")
    mapping_file.write_text(mapping_text(mapping, ordered=ordered), encoding="utf-8")

    previous = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        answer = get_hardware_performance_zigzag(
            [workload_entry(problem)],
            str(accelerator),
            str(mapping_file),
            opt="latency",
            dump_folder=str(work / "out"),
            pickle_filename=str(work / "out" / "cmes.pickle"),
            loma_show_progress_bar=False,
        )
    finally:
        logging.disable(previous)

    evaluation = answer[-1][0][0]
    per_layer = answer[-1][0][1][0][0]
    evaluated = evaluated_ordering(per_layer)
    if ordered:
        requested = [
            f"{name}, {factor}" for name, factor in mapping.temporal if factor > 1
        ]
        if evaluated != requested:
            raise ExportError(
                f"ZigZag evaluated {evaluated} and was handed {requested}. A "
                f"mismatch means the ordering was read as a hint and the search "
                f"ran, which is the comparison Section 16.5 forbids rather than "
                f"the one it asks for."
            )
    return {
        "latency": float(evaluation.latency_total2),
        "ideal": float(evaluation.ideal_cycle),
        "ideal_temporal": float(evaluation.ideal_temporal_cycle),
        "evaluated": evaluated,
    }


def evaluated_ordering(evaluation: Any) -> list[str]:
    """The loop nest ZigZag actually evaluated, innermost first.

    Read back off the evaluation rather than assumed, because "the ordering was
    complete so ZigZag used it" is exactly the kind of claim this project keeps
    finding to be false in the one case nobody checked. `mapping_dic_origin` is
    the nest as the allocator received it, per operand and per memory level; the
    operands agree on it here because every loop is above the innermost level.
    """
    nest = evaluation.temporal_mapping.mapping_dic_origin
    for levels in nest.values():
        return [
            f"{dimension}, {factor}" for level in levels for dimension, factor in level
        ]
    return []


def compare(
    cells: list[Cell], *, search: bool = False
) -> tuple[list[Comparison], dict[str, Any]]:
    """Every cell, under its own mapping, with the budget it was compiled at."""
    results: list[Comparison] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="zigzag-run-") as directory:
        work = Path(directory)
        for cell in cells:
            if time.perf_counter() - started > WALL_CLOCK_CEILING_SECONDS:
                raise ExportError(
                    f"the comparison passed {WALL_CLOCK_CEILING_SECONDS} seconds "
                    f"with {len(cells) - len(results)} layers left. Recorded as a "
                    f"stop rather than continued: the bound is the machine's."
                )
            if peak_memory_mb() > MEMORY_CEILING_MB:
                raise ExportError(
                    f"peak resident memory reached {peak_memory_mb():.0f} MiB, "
                    f"over this machine's ceiling of {MEMORY_CEILING_MB} MiB, "
                    f"with {len(cells) - len(results)} layers left. This step "
                    f"stops here and the reason is recorded. The machine is not "
                    f"reconfigured to make it fit."
                )
            mapping = (
                conv_mapping(cell.problem, cell.choice)
                if cell.kind == "conv2d"
                else matmul_mapping(cell.problem, cell.choice)
            )
            check_complete(cell.problem, mapping)
            layer_started = time.perf_counter()
            answer = run_zigzag(cell.problem, mapping, cell.budget, work)
            elapsed = time.perf_counter() - layer_started

            comparison = Comparison(
                model=cell.model,
                budget=cell.budget,
                layer=cell.layer,
                kind=cell.kind,
                configurations=sorted(set(cell.configurations)),
                loop_sizes=dict(cell.problem.sizes),
                temporal_tiles=list(cell.choice.temporal_tiles),
                tile_count=cell.choice.tile_count,
                tile_bytes=cell.choice.tile_bytes,
                spatial_mapping={
                    dimension: [f"{name}, {factor}" for name, factor in entries]
                    for dimension, entries in mapping.spatial.items()
                },
                temporal_ordering=[
                    [name, factor] for name, factor in mapping.temporal if factor > 1
                ],
                here_cycles=cell.choice.makespan_cycles,
                zigzag_cycles=answer["latency"],
                zigzag_ideal_cycles=answer["ideal"],
                zigzag_ideal_temporal_cycles=answer["ideal_temporal"],
                array_fill_here=mapping.array_fill_here,
                array_fill_zigzag=mapping.array_fill_zigzag,
                seconds=elapsed,
                evaluated_ordering=list(answer["evaluated"]),
            )
            if search:
                free = run_zigzag(
                    cell.problem, mapping, cell.budget, work, ordered=False
                )
                comparison.searched_cycles = float(free["latency"])
                comparison.searched_margin = (
                    float(answer["latency"] - free["latency"]) / answer["latency"]
                    if answer["latency"]
                    else 0.0
                )
                comparison.searched_ordering = list(free["evaluated"])
            results.append(comparison)

    ratios = [row.ratio for row in results if row.ratio > 0]
    summary = {
        "layers": len(results),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "memory_ceiling_mb": MEMORY_CEILING_MB,
        "ratio_min": round(min(ratios), 4) if ratios else 0.0,
        "ratio_max": round(max(ratios), 4) if ratios else 0.0,
        "ratio_geometric_mean": (
            round(math.exp(sum(math.log(value) for value in ratios) / len(ratios)), 4)
            if ratios
            else 0.0
        ),
        "reads_above_on": sum(1 for value in ratios if value > 1.0),
        "reads_below_on": sum(1 for value in ratios if value < 1.0),
    }
    if search:
        material = [
            row
            for row in results
            if row.searched_margin is not None
            and row.searched_margin > MATERIAL_FRACTION
        ]
        summary["material_fraction"] = MATERIAL_FRACTION
        summary["materially_better_on"] = len(material)
        summary["materially_better_layers"] = [
            f"{row.model} {row.layer} at {row.budget}" for row in material
        ]
    return results, summary


def report(results: list[Comparison], summary: dict[str, Any]) -> None:
    print(
        f"{'model':20}{'budget':>8}  {'layer':22}{'here':>10}{'zigzag':>10}"
        f"{'ratio':>8}{'fill here':>11}{'fill zz':>9}"
    )
    print("-" * 98)
    for row in results:
        print(
            f"{row.model:20}{row.budget:>8}  {row.layer:22}"
            f"{row.here_cycles:>10.1f}{row.zigzag_cycles:>10.1f}"
            f"{row.ratio:>8.2f}{row.array_fill_here:>11.3f}"
            f"{row.array_fill_zigzag:>9.3f}"
        )
    print()
    for name, value in summary.items():
        print(f"  {name}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zigzag_same_mapping.py",
        description="Section 16.5's cross check, under the mapping this compiler chose.",
    )
    parser.add_argument("--models", nargs="+", default=sorted(SWEPT))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--mappings", type=Path, default=None)
    parser.add_argument(
        "--search",
        action="store_true",
        help="also ask ZigZag's own engine for a mapping, scored by ZigZag on "
        "both sides, and report a margin over the material fraction.",
    )
    arguments = parser.parse_args(argv)

    unknown = [name for name in arguments.models if name not in MODELS]
    if unknown:
        print(f"zigzag: unknown models {unknown}.", file=sys.stderr)
        return 2

    try:
        cells = collect(list(arguments.models))
        if not cells:
            print("zigzag: no layer was tiled at any swept budget.", file=sys.stderr)
            return 1
        results, summary = compare(cells, search=arguments.search)
    except ExportError as failure:
        print(f"zigzag: {failure}", file=sys.stderr)
        return 1

    report(results, summary)

    if arguments.mappings is not None:
        arguments.mappings.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            mapping = (
                conv_mapping(cell.problem, cell.choice)
                if cell.kind == "conv2d"
                else matmul_mapping(cell.problem, cell.choice)
            )
            stem = f"{cell.model}-{cell.budget}-{cell.layer}"
            (arguments.mappings / f"{stem}.zigzag.yaml").write_text(
                mapping_text(mapping), encoding="utf-8"
            )
            (arguments.mappings / f"{stem}.timeloop.yaml").write_text(
                timeloop_text(cell.problem, mapping), encoding="utf-8"
            )
        print(f"\nzigzag: wrote {2 * len(cells)} mapping files to {arguments.mappings}")

    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(
            json.dumps(
                {
                    "comparisons": [asdict(row) for row in results],
                    "summary": summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"zigzag: wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
