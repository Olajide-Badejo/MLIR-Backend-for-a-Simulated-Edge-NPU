# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The quantized cells' energy if four int8 elements shared a scratchpad access.

*Added at P14, as a sensitivity and not a recorded figure.* The recorded
energy charges one scratchpad access per element whatever its type, which the
2026-09-30 entry in `docs/BREAKING_CHANGES.md` declares and which is the
direction that does not flatter the quantized result. This script answers the
other end of the range: the same cells with every int8 element access packed
four to a 32 bit word, and nothing else changed.

**What is packed and what is not.** Only int8 operand and result traffic: an
operand or a result of element type `i8` in the scratchpad, which is the int8
activations and weights a contraction reads and writes, the int8 side of
`QUANT` and `DEQUANT`, and the int8 destination of a transfer. Not the int32
bias, not the int32 rescale table, and not the f32 side of `QUANT` and
`DEQUANT`, which are 32 bit elements and already one to a word.

**The method.** `Stats` counts scratchpad element accesses without their type,
so the split is computed here from the program: each committed quantized cell
is recompiled from its profile, every instruction's scratchpad operands and
scratchpad result are read off the allocated `npuisa` module with their element
types, the way `lib/Simulator/Simulator.cpp` counts them, and the per type
totals are asserted to sum to the cell's recorded `scratchpad_elements_read` and
`scratchpad_elements_written` exactly before anything is computed from them. A
packed operand costs `ceil(elements / 4)` accesses. The energy is the cell's own
Accelergy estimate at those counts, from the same reference table, because the
energy is linear in the counts.

No schema field, no cost model term and no recorded number moves: this prints
a table, and `docs/NUMBERS.md` carries it beside the recorded ratios.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import accelergy_energy  # noqa: E402
import roofline  # noqa: E402
from npu_frontend import npuisa_walk  # noqa: E402
from npu_frontend.results import RESULTS_DIR, load_result  # noqa: E402

#: Four int8 elements to one 32 bit scratchpad word.
INT8_PER_ACCESS = accelergy_energy.SCRATCHPAD_WORD_BITS // 8


class PackingError(Exception):
    """The split by type does not reproduce what the machine counted."""


@dataclass(frozen=True)
class Split:
    """One program's scratchpad element accesses, by element type."""

    read_by_type: dict[str, int]
    written_by_type: dict[str, int]
    #: The int8 accesses at four elements a word, operand by operand.
    int8_read_packed: int
    int8_written_packed: int

    @property
    def read(self) -> int:
        return sum(self.read_by_type.values())

    @property
    def written(self) -> int:
        return sum(self.written_by_type.values())

    @property
    def packed_read(self) -> int:
        return self.read - self.read_by_type.get("i8", 0) + self.int8_read_packed

    @property
    def packed_written(self) -> int:
        return (
            self.written - self.written_by_type.get("i8", 0) + self.int8_written_packed
        )


def _packed(elements: int) -> int:
    return -(-elements // INT8_PER_ACCESS)


def split(npuisa_text: str) -> Split:
    """Every scratchpad operand and result of the program, by element type."""
    read: dict[str, int] = {}
    written: dict[str, int] = {}
    read_packed = 0
    written_packed = 0
    for operation in npuisa_walk.walk(npuisa_text):
        for operand in operation.operands:
            if operand.space != "scratchpad":
                continue
            read[operand.element] = read.get(operand.element, 0) + operand.elements
            if operand.element == "i8":
                read_packed += _packed(operand.elements)
        result = operation.result
        if result.space == "scratchpad":
            written[result.element] = written.get(result.element, 0) + result.elements
            if result.element == "i8":
                written_packed += _packed(result.elements)
    return Split(
        read_by_type=read,
        written_by_type=written,
        int8_read_packed=read_packed,
        int8_written_packed=written_packed,
    )


def check(answer: Split, result: dict[str, Any]) -> None:
    """The split sums to what the machine counted, or nothing is computed."""
    simulation = result["simulation"]
    for label, mine, theirs in (
        ("read", answer.read, int(simulation["scratchpad_elements_read"])),
        ("written", answer.written, int(simulation["scratchpad_elements_written"])),
    ):
        if mine != theirs:
            raise PackingError(
                f"{result['cell']['name']}: the program's scratchpad operands sum "
                f"to {mine} elements {label} and the machine counted {theirs}. A "
                f"split that does not partition the recorded count is a split of "
                f"something else."
            )


@dataclass(frozen=True)
class Row:
    cell: str
    int8_share_read: float
    int8_share_written: float
    energy_pj: float
    energy_pj_packed: float
    scratchpad_pj: float
    scratchpad_pj_packed: float
    twin_energy_pj: float

    @property
    def ratio(self) -> float:
        return self.energy_pj / self.twin_energy_pj

    @property
    def ratio_packed(self) -> float:
        return self.energy_pj_packed / self.twin_energy_pj


def analyse(results_dir: Path, models: set[str] | None) -> list[Row]:
    paths = sorted(results_dir.glob("*-int8-normal.json"))
    rows: list[Row] = []
    with tempfile.TemporaryDirectory(prefix="npu-int8-packing-") as directory:
        root = Path(directory)
        compiler = roofline.Compiler(root / "models")
        estimator = accelergy_energy.Estimator(root / "accelergy")
        for path in paths:
            result = load_result(path)
            if models is not None and result["cell"]["model"] not in models:
                continue
            answer = split(compiler.allocated_ir(result))
            check(answer, result)
            recorded = estimator.energy(result)
            if recorded.energy_pj != result["external"]["energy_pj"]:
                raise PackingError(
                    f"{result['cell']['name']}: the estimate at the recorded counts "
                    f"is {recorded.energy_pj!r} pJ and the cell records "
                    f"{result['external']['energy_pj']!r}, so the table is not the "
                    f"one the cell was measured with."
                )
            packed_cell = copy.deepcopy(result)
            packed_cell["simulation"]["scratchpad_elements_read"] = answer.packed_read
            packed_cell["simulation"][
                "scratchpad_elements_written"
            ] = answer.packed_written
            packed = accelergy_energy.energy_for(
                packed_cell, estimator.estimate(result)
            )
            twin = load_result(path.with_name(path.name.replace("-int8-", "-fp32-")))
            rows.append(
                Row(
                    cell=result["cell"]["name"],
                    int8_share_read=answer.read_by_type.get("i8", 0) / answer.read,
                    int8_share_written=(
                        answer.written_by_type.get("i8", 0) / answer.written
                    ),
                    energy_pj=recorded.energy_pj,
                    energy_pj_packed=packed.energy_pj,
                    scratchpad_pj=recorded.energy_pj_per_component["scratchpad"],
                    scratchpad_pj_packed=packed.energy_pj_per_component["scratchpad"],
                    twin_energy_pj=float(twin["external"]["energy_pj"]),
                )
            )
    if not rows:
        raise PackingError(f"no committed quantized cell under {results_dir}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="int8_scratchpad_packing.py",
        description="The quantized cells' energy with int8 scratchpad accesses "
        "packed four to a word, as a sensitivity.",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)
    try:
        rows = analyse(
            arguments.results, set(arguments.models) if arguments.models else None
        )
    except (PackingError, accelergy_energy.AccelergyError) as failure:
        print(f"int8-packing: {failure}", file=sys.stderr)
        return 2
    for row in rows:
        print(
            f"{row.cell:44s} int8 share read {row.int8_share_read:.3f} written "
            f"{row.int8_share_written:.3f} | int8/fp32 energy {row.ratio:.3f} "
            f"recorded, {row.ratio_packed:.3f} packed"
        )
    ratios = [row.ratio for row in rows]
    packed = [row.ratio_packed for row in rows]
    print(
        f"int8-packing: {len(rows)} cells, every split partitions the recorded "
        f"counts. int8/fp32 energy {min(ratios):.3f} to {max(ratios):.3f} recorded, "
        f"{min(packed):.3f} to {max(packed):.3f} with int8 accesses packed."
    )
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                [
                    asdict(row) | {"ratio": row.ratio, "ratio_packed": row.ratio_packed}
                    for row in rows
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
