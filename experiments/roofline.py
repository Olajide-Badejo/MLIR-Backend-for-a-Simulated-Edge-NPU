# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The roofline check of Section 16.6: a physical bound, not a second opinion.

    python experiments/roofline.py
    python experiments/roofline.py --models lenet --verbose

*Added at P11, and built first.* Section 16.6 says it in those words: the
roofline is the frame the other three tools are presented inside, and building
the frame after the pictures is how the pictures end up framing themselves.

Comparing two simulators is the weakest form of validation there is. A roofline
is stronger in kind, because it is a bound the hardware could not have beaten
rather than another model's opinion, and it costs no dependency at all.

## The two branches, and which one is worth anything

Per layer the bound is `effective_macs / min(intensity * bw_peak, comp_peak)`,
which is the same number as `max(bytes / bw_peak, effective_macs / comp_peak)`
written the way Section 16.6 writes it. The two branches are **not** worth the
same and this module records which one bound each layer:

- the **compute** branch is `effective_macs / comp_peak`, and Section 16.6 warns
  that it is partly the cost model grading its own homework. This project's
  measurement is stronger than that warning: under the cost model of Section 5.5
  `effective_macs` is *defined* as `cycles * peak`, so the compute branch is
  identically the kernel's own cycle count and the comparison is a tautology up
  to the per instruction issue overhead. It is computed and recorded because
  Section 16.6 asks for the branch that bound each layer, and it is never read as
  evidence. **`docs/NUMBERS.md` records what that leaves the check worth**, and
  it is not nothing: it is a regression bound that fires the day a pass claims
  cycles below the traffic it still moves.
- the **memory** branch is `dram_bytes / bw_peak`. The byte count is a fact the
  simulator counted rather than a factor it chose, so a cycle count below it
  asserts that bytes moved faster than the modelled interconnect can move them.
  This is the independently binding half and a violation of it is the serious
  one.

## Where the per layer numbers come from

`npu_frontend.npuisa_walk`, which walks the allocated `npuisa` module the cell
was compiled from and charges each operation through the Python mirror of
`CostModel.h`. Every walk is checked against the cell it belongs to before a
single bound is computed from it: raw MACs, DRAM bytes read and written, and the
instruction count have to agree with the machine exactly. A per layer bound
derived from a walk that cannot reproduce the program's totals is not a
measurement, so the check raises rather than warning.

The **cell level** bound is computed from the recorded fields alone,
`simulation.effective_macs` and `simulation.dram_bytes_total` against
`simulation.simulated_cycles`, and touches the walk not at all. So the two
answers come from two paths and the per model verdict does not depend on this
module's parser being right.

## What a violation does

Fails the run, per Section 16.6 and Section 16.1 both. `roofline_verdict` records
that the check ran; a check that has not run is `null` with its reason, which is
a different statement from `below_bound` and the schema keeps them apart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mlir_python_packages_dir() -> Path:
    """Where the MLIR Python bindings live.

    The same three step resolution `experiments/run_benchmarks.py` uses, and
    carried here for the same reason it is carried there: an experiment is not
    run under pytest and cannot take it from a conftest.
    """
    override = os.environ.get("MLIR_PYTHON_PACKAGES_DIR")
    if override:
        return Path(override)

    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    if cache.is_file():
        pattern = re.compile(r"^MLIR_PYTHON_PACKAGES_DIR:[^=]*=(.*)$")
        for line in cache.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match and match.group(1).strip():
                return Path(match.group(1).strip())

    return (
        Path.home()
        / "llvm-project"
        / "build"
        / "tools"
        / "mlir"
        / "python_packages"
        / "mlir_core"
    )


sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

from npu_frontend import cost_model, npuisa_walk  # noqa: E402
from npu_frontend.compile import compile_model  # noqa: E402
from npu_frontend.model_generator import generate_model  # noqa: E402
from npu_frontend.results import (  # noqa: E402
    RESULTS_DIR,
    ResultSchemaError,
    load_result,
)
from npu_frontend.tolerances import ROOFLINE_RELATIVE_SLACK  # noqa: E402

#: Section 16.1's two verdicts, spelled once.
AT_OR_ABOVE = "at_or_above_bound"
BELOW = "below_bound"

#: Which half of the bound bound a layer. Recorded beside the verdict because
#: Section 16.6 asks for it and because a memory bound violation is the serious
#: one.
COMPUTE_BRANCH = "compute"
MEMORY_BRANCH = "memory"


class RooflineError(Exception):
    """The check cannot be run, or a cell is below its bound."""


# ---------------------------------------------------------------------------
# The bound.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bound:
    """One roofline bound, with the branch that produced it."""

    bound_cycles: float
    compute_branch_cycles: float
    memory_branch_cycles: float
    bound_by: str
    operational_intensity: float | None
    charged_cycles: float
    verdict: str

    @property
    def headroom(self) -> float:
        """How far above the bound the charge sits, as a fraction of the bound.

        Negative means below, which is the failure. Reported rather than only the
        verdict, because a cell sitting a thousandth above its bound and one
        sitting ten times above it are both `at_or_above_bound` and are not the
        same situation.
        """
        if self.bound_cycles <= 0.0:
            return float("inf")
        return (self.charged_cycles - self.bound_cycles) / self.bound_cycles


def bound_for(
    *, effective_macs: float, dram_bytes: int, charged_cycles: float, peak: int
) -> Bound:
    """Section 16.6's bound, for one layer or for one whole cell.

    `operational_intensity` is `effective_macs / dram_bytes` and is `None` when
    no bytes moved, which is a real state on this machine: the allocator has
    already staged a convolution's operands into the scratchpad, so a layer whose
    every operand was staged for an earlier layer touches DRAM not at all. A
    division by zero recorded as a large number would be an intensity this
    project did not measure appearing in a plot as if it had.
    """
    compute_branch = effective_macs / peak if peak > 0 else 0.0
    memory_branch = dram_bytes / cost_model.DRAM_BANDWIDTH_BYTES_PER_CYCLE
    if memory_branch > compute_branch:
        bound_cycles, bound_by = memory_branch, MEMORY_BRANCH
    else:
        bound_cycles, bound_by = compute_branch, COMPUTE_BRANCH

    intensity = effective_macs / dram_bytes if dram_bytes else None

    # The slack is on the representation and never on the finding. See
    # `npu_frontend.tolerances.ROOFLINE_RELATIVE_SLACK` for where the number
    # comes from.
    floor = bound_cycles * (1.0 - ROOFLINE_RELATIVE_SLACK)
    verdict = AT_OR_ABOVE if charged_cycles >= floor else BELOW

    return Bound(
        bound_cycles=bound_cycles,
        compute_branch_cycles=compute_branch,
        memory_branch_cycles=memory_branch,
        bound_by=bound_by,
        operational_intensity=intensity,
        charged_cycles=charged_cycles,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# One cell.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerBound:
    """One layer's row of the roofline table."""

    name: str
    op: str
    position: int
    macs: int
    effective_macs: float
    dram_bytes: int
    #: What the array was charged for this layer's arithmetic.
    compute_cycles: float
    #: What the DMA port was charged for the transfers that exist to feed this
    #: layer or to drain it.
    dma_cycles: float
    bound: Bound


@dataclass(frozen=True)
class CellRoofline:
    """Everything the roofline says about one cell."""

    cell: str
    layers: tuple[LayerBound, ...]
    whole: Bound

    @property
    def violations(self) -> list[str]:
        found = [
            f"{self.cell}: layer {layer.name} ({layer.op}) charges "
            f"{layer.bound.charged_cycles:.4f} cycles against a "
            f"{layer.bound.bound_by} bound of {layer.bound.bound_cycles:.4f}"
            for layer in self.layers
            if layer.bound.verdict == BELOW
        ]
        if self.whole.verdict == BELOW:
            found.append(
                f"{self.cell}: the cell charges {self.whole.charged_cycles:.4f} "
                f"cycles against a {self.whole.bound_by} bound of "
                f"{self.whole.bound_cycles:.4f}"
            )
        return found

    def as_schema(self) -> dict[str, Any]:
        """The `roofline` block Section 16.1 names, ready to be written.

        Four keys and no nulls, because a cell this function ran on is a cell
        that was checked. The `_null_reason` siblings are absent by construction
        rather than deleted, and `npu_frontend.results.validate_result` refuses a
        field carrying both.
        """
        return {
            "roofline_bound_cycles": self.whole.bound_cycles,
            "roofline_bound_cycles_per_layer": [
                {
                    "name": layer.name,
                    "op": layer.op,
                    "position": layer.position,
                    "macs": layer.macs,
                    "effective_macs": layer.effective_macs,
                    "dram_bytes": layer.dram_bytes,
                    "compute_cycles": layer.compute_cycles,
                    "dma_cycles": layer.dma_cycles,
                    "bound_cycles": layer.bound.bound_cycles,
                    "compute_branch_cycles": layer.bound.compute_branch_cycles,
                    "memory_branch_cycles": layer.bound.memory_branch_cycles,
                    "bound_by": layer.bound.bound_by,
                    "operational_intensity": layer.bound.operational_intensity,
                    "charged_cycles": layer.bound.charged_cycles,
                    "verdict": layer.bound.verdict,
                }
                for layer in self.layers
            ],
            "operational_intensity": self.whole.operational_intensity,
            "roofline_verdict": (BELOW if self.violations else AT_OR_ABOVE),
        }


class Compiler:
    """Compiles a cell back to its allocated IR, caching what it can.

    A cell's roofline needs the program, and a result file records the numbers
    rather than the program. Recompiling is cheap and deterministic; caching on
    the four things that decide the IR keeps the whole suite to one compile per
    distinct program rather than one per cell.
    """

    def __init__(self, work: Path) -> None:
        self._work = work
        self._onnx: dict[tuple[str, int], Path] = {}
        self._ir: dict[tuple[str, int, int, int, str | None], str] = {}

    def allocated_ir(self, result: dict[str, Any]) -> str:
        cell = result["cell"]
        model = cell["model"]
        batch = int(cell["batch"])
        level = int(cell["opt_level"])
        budget = int(cell["scratchpad_budget_bytes"])
        ablated = cell["ablated_pass"]

        key = (model, batch, level, budget, ablated)
        if key not in self._ir:
            onnx_key = (model, batch)
            if onnx_key not in self._onnx:
                self._onnx[onnx_key] = generate_model(model, self._work, batch=batch)
            compiled = compile_model(
                self._onnx[onnx_key],
                level=level,
                emit="npuisa",
                budget=budget,
                ablate=ablated,
            )
            self._ir[key] = compiled.stages["npuisa"]
        return self._ir[key]


def roofline_for(result: dict[str, Any], npuisa_text: str) -> CellRoofline:
    """The roofline of one cell, per layer and whole.

    The walk is checked against the cell before any bound is computed from it,
    which is the only thing that makes a per layer number here comparable with a
    recorded one.
    """
    operations = npuisa_walk.attribute_transfers(
        npuisa_walk.walk(npuisa_text), npuisa_text
    )
    npuisa_walk.check_against_result(operations, result)

    peak = cost_model.PEAK_MACS_PER_CYCLE_F32
    layers = tuple(
        LayerBound(
            name=operation.name,
            op=operation.op,
            position=operation.position,
            macs=operation.macs,
            effective_macs=operation.effective_macs,
            dram_bytes=operation.attributed_dram_bytes,
            compute_cycles=operation.cycles,
            dma_cycles=operation.attributed_dma_cycles,
            bound=bound_for(
                effective_macs=operation.effective_macs,
                dram_bytes=operation.attributed_dram_bytes,
                # **The later of the two timelines, and not the compute charge
                # alone.** Section 5.5 accumulates cycles on two independent
                # ports and reports the later of them at `HALT`, so the time one
                # layer occupies is the later of the array's charge for its
                # arithmetic and the DMA port's charge for its bytes. Comparing
                # the compute charge alone against a bound that has a memory
                # branch in it would compare a compute time with a memory time,
                # and every matmul in `lenet` reads below its bound when it is
                # done that way: `node_linear` charges 3404 cycles for arithmetic
                # over a 400 by 120 weight matrix whose 192480 bytes take 12030
                # cycles to move. The bytes were not moved faster than the
                # interconnect; they were moved by an instruction the comparison
                # had left out.
                charged_cycles=max(operation.cycles, operation.attributed_dma_cycles),
                peak=peak,
            ),
        )
        for operation in operations
        if operation.is_compute
    )
    if not layers:
        raise RooflineError(
            f"{result['cell']['name']} has no convolution or matmul, so there is "
            f"no layer to bound. Every model in this suite has at least one, so "
            f"this is a walk that went wrong rather than a program that is "
            f"genuinely all elementwise."
        )

    simulation = result["simulation"]
    whole = bound_for(
        effective_macs=float(simulation["effective_macs"]),
        dram_bytes=int(simulation["dram_bytes_total"]),
        charged_cycles=float(simulation["simulated_cycles"]),
        peak=peak,
    )
    return CellRoofline(cell=result["cell"]["name"], layers=layers, whole=whole)


# ---------------------------------------------------------------------------
# The whole suite.
# ---------------------------------------------------------------------------


def analyse(results: list[dict[str, Any]], compiler: Compiler) -> list[CellRoofline]:
    """Every cell's roofline, in the order the cells were given."""
    return [roofline_for(result, compiler.allocated_ir(result)) for result in results]


def summarise(rooflines: list[CellRoofline]) -> dict[str, Any]:
    """What the whole run says, for the engineering log and for `docs/NUMBERS.md`."""
    layer_rows = [layer for roofline in rooflines for layer in roofline.layers]
    memory_bound = [
        layer for layer in layer_rows if layer.bound.bound_by == MEMORY_BRANCH
    ]
    tightest = min(layer_rows, key=lambda layer: layer.bound.headroom, default=None)
    return {
        "cells": len(rooflines),
        "layers": len(layer_rows),
        "layers_memory_bound": len(memory_bound),
        "layers_compute_bound": len(layer_rows) - len(memory_bound),
        "cells_memory_bound": sum(
            1 for roofline in rooflines if roofline.whole.bound_by == MEMORY_BRANCH
        ),
        "violations": [line for roofline in rooflines for line in roofline.violations],
        "tightest_layer": (
            {
                "layer": tightest.name,
                "headroom": tightest.bound.headroom,
                "bound_by": tightest.bound.bound_by,
            }
            if tightest is not None
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="roofline.py",
        description=(
            "Section 16.6's roofline over the committed cells. Reports the bound "
            "per layer and per cell and FAILS on any cell below its bound."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="write the whole table here as JSON, for the report to read",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print one row per layer rather than one per cell",
    )
    arguments = parser.parse_args(argv)

    try:
        return _run(arguments)
    except (RooflineError, ResultSchemaError, npuisa_walk.WalkError) as failure:
        print(f"roofline: {failure}", file=sys.stderr)
        return 2


def _run(arguments: argparse.Namespace) -> int:
    import tempfile

    paths = sorted(Path(arguments.results).glob("*.json"))
    if not paths:
        raise RooflineError(
            f"no result files under {arguments.results}. The roofline is computed "
            f"from the committed cells and there are none, so an empty pass here "
            f"would be a check that stopped checking."
        )
    results = [load_result(path) for path in paths]
    if arguments.models:
        results = [
            result
            for result in results
            if result["cell"]["model"] in set(arguments.models)
        ]
        if not results:
            raise RooflineError(f"no committed cells for {arguments.models}")

    with tempfile.TemporaryDirectory(prefix="npu-roofline-") as directory:
        rooflines = analyse(results, Compiler(Path(directory)))

    summary = summarise(rooflines)
    print(
        f"roofline: {summary['cells']} cells, {summary['layers']} layers, "
        f"{summary['layers_memory_bound']} memory bound and "
        f"{summary['layers_compute_bound']} compute bound"
    )
    if arguments.verbose:
        for roofline in rooflines:
            for layer in roofline.layers:
                print(
                    f"  {roofline.cell} {layer.name:<24} {layer.op:<8} "
                    f"charged {layer.bound.charged_cycles:12.2f} "
                    f"bound {layer.bound.bound_cycles:12.2f} "
                    f"by {layer.bound.bound_by:<8} "
                    f"headroom {layer.bound.headroom:8.3f} "
                    f"{layer.bound.verdict}"
                )

    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "cells": [
                        {
                            "cell": roofline.cell,
                            "whole": asdict(roofline.whole),
                            "layers": [asdict(layer) for layer in roofline.layers],
                        }
                        for roofline in rooflines
                    ],
                },
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )

    if summary["violations"]:
        print()
        for line in summary["violations"]:
            print(f"roofline: {line}", file=sys.stderr)
        print(
            "roofline: FAIL. Section 16.6: any pass that appears to improve "
            "cycles past the physical bound fails the run. A memory bound "
            "violation is the serious one, because the byte count is a fact the "
            "simulator counted rather than a factor it chose.",
            file=sys.stderr,
        )
        return 1

    tightest = summary["tightest_layer"]
    print(
        f"roofline: every cell at or above its bound. Tightest layer "
        f"{tightest['layer']} at {tightest['headroom']:.4f} headroom over its "
        f"{tightest['bound_by']} bound."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
