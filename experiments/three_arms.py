# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 13.3's three arms, per model and per budget, at `-O2`.

The question Section 13.3 asks is what a compiler should do when a program does
not fit: **spill**, **tile**, or **recompute**. This runs all three over the
budgets where any of them changes anything and reports what each costs.

**The arms, and why each is two configurations rather than one.**

- **Arm one, spill.** `-npu-tile-to-scratchpad` left out, so the allocator is the
  only thing standing between the program and the budget. Run under both spill
  heuristics of Section 13.1, `longest-range` and `cost`, which Section 13.3
  calls two configurations of one arm rather than two arms.
- **Arm two, tile.** The level as it stands, and the level with `-npu-fuse-ops`
  left out. At `-O2` fusion puts 30 of the 44 convolutions and matrix
  multiplications inside `npu.fused_op` regions where the tiling pass does not
  look, so the first configuration is what the compiler does today and the second
  is what tiling is worth when nothing hides the operands from it.
- **Arm three, recompute.** `halo=cache` against the level's default of
  `halo=recompute`. `recompute` lets the search split the output spatial axes and
  pay for the overlapping input rows per tile; `cache` refuses to split them so
  that no halo is created. Both configurations of arm two are run under both.

**A program an arm cannot place is reported as exactly that.** Section 13.3 says
so and it is the reason `placed` is a field rather than an absence: an arm that
produces no program is not a slower arm, and a table that had nowhere to put
"there is no program here" would have to put a number there.

**Two arms is an incomplete experiment.** Where an arm cannot run at all the
report says which and why, rather than comparing the two that could.

Usage:

    python experiments/three_arms.py
    python experiments/three_arms.py --models resnet_block inception_block
    python experiments/three_arms.py --json experiments/results-three-arms/arms.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mlir_python_packages_dir() -> Path:
    """Where the MLIR Python bindings live.

    The same three step resolution the other experiments use, repeated rather
    than imported because a conftest is pytest's file and an experiment is not
    run under pytest.
    """
    override = os.environ.get("MLIR_PYTHON_PACKAGES_DIR")
    if override:
        return Path(override)
    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    if cache.is_file():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("MLIR_PYTHON_PACKAGES_DIR:"):
                return Path(line.split("=", 1)[1])
    return Path.home() / "llvm-project/build/tools/mlir/python_packages/mlir_core"


sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

from npu_frontend.compile import (  # noqa: E402
    CompileError,
    compile_model,
    run_program,
)
from npu_frontend.input_classes import make_inputs  # noqa: E402
from npu_frontend.model_generator import MODELS, generate_model  # noqa: E402

#: The budget at which nothing in this suite is over budget, and therefore the
#: control row of every configuration.
#:
#: **Without it the fusion ablated rows have no control.** With `-npu-fuse-ops`
#: left out the tiling pass fires at every budget in the swept ranges below, so
#: comparing those rows against each other says what a smaller budget costs and
#: never what tiling costs. At the default budget the pass runs, finds nothing
#: over budget and answers no, which is the same configuration with the arm
#: switched off by the input rather than by a pass list.
CONTROL_BUDGET = 1048576

#: The budgets each model is measured at.
#:
#: The swept ranges are the measured ones: `docs/PHASE_STATE.md` records that
#: tiling starts firing between 6000 and 6464 on `resnet_block` and
#: `conv_bn_relu_stack` and between 4000 and 6000 on `inception_block`. Every
#: model also gets its own frozen tight budget, so that the published cells
#: appear in the table beside the swept ones; the floors
#: `experiments/tight_budget_floor.py` measured, so that the bottom of the range
#: is a budget something still places at rather than a round number; and the
#: control budget above.
SWEPT: dict[str, tuple[int, ...]] = {
    "resnet_block": (6000, 6144, 6272, 6400, 6464, CONTROL_BUDGET),
    "conv_bn_relu_stack": (
        4672,
        4928,
        6000,
        6144,
        6272,
        6400,
        6464,
        CONTROL_BUDGET,
    ),
    "inception_block": (4000, 4608, 5120, 5632, 6000, 6144, CONTROL_BUDGET),
    "lenet": (194240, 194624, CONTROL_BUDGET),
    "lenet_batched": (199872, 200832, CONTROL_BUDGET),
    "depthwise_separable": (8064, 8192, CONTROL_BUDGET),
    "dilated_stack": (7936, 8064, CONTROL_BUDGET),
}

#: The six configurations, as the level options that make each one.
#:
#: `ablate` is one pass by design, which is Section 16.2's rule and is why there
#: is no configuration with both fusion and tiling removed: that is not
#: expressible through the level, and assembling a pass list beside it would
#: measure a pass list rather than the compiler. The sweep supplies the control
#: instead, because above each model's tiling threshold the pass fires on nothing
#: and the row is the untiled one.
ARMS: dict[str, dict[str, str]] = {
    "spill-longest-range": {
        "ablate": "npu-tile-to-scratchpad",
        "spill_heuristic": "longest-range",
    },
    "spill-cost": {
        "ablate": "npu-tile-to-scratchpad",
        "spill_heuristic": "cost",
    },
    "tile-fused-recompute": {"halo": "recompute"},
    "tile-fused-cache": {"halo": "cache"},
    "tile-unfused-recompute": {"ablate": "npu-fuse-ops", "halo": "recompute"},
    "tile-unfused-cache": {"ablate": "npu-fuse-ops", "halo": "cache"},
}

#: Which arm each configuration belongs to, for the report.
ARM_OF: dict[str, str] = {
    "spill-longest-range": "one, spill",
    "spill-cost": "one, spill",
    "tile-fused-recompute": "two, tile",
    "tile-unfused-recompute": "two, tile",
    "tile-fused-cache": "three, no halo",
    "tile-unfused-cache": "three, no halo",
}

_ATTRIBUTE = re.compile(r"npuisa\.(\w+)\s*=\s*([0-9.e+-]+)")

#: The input class every point of the grid runs at.
#:
#: One class rather than the matrix's four, because what is being compared here
#: is arrangements of the same arithmetic: the arms move transfers and offsets
#: around and none of them changes what a kernel computes, so a second input
#: class would multiply the grid by four and answer the same question again.
INPUT_CLASS = "normal"


@dataclass
class Run:
    """One model, at one budget, under one configuration."""

    model: str
    budget: int
    arm: str
    configuration: str
    placed: bool
    refusal: str = ""
    instructions: int = 0
    cycles: float = 0.0
    dram_bytes: int = 0
    scratchpad_peak_bytes: int = 0
    scratchpad_bytes: int = 0
    spill_count: int = 0
    spill_dma_count: int = 0


def attributes(npuisa_text: str) -> dict[str, float]:
    """The allocator's own numbers, off the function it allocated."""
    return {name: float(value) for name, value in _ATTRIBUTE.findall(npuisa_text)}


def declared_batch(model: str) -> int:
    """The batch the model's own shape declares, which its tight budget was at."""
    return int(MODELS[model].input_shape[0])


def one_run(model: str, onnx: Path, budget: int, configuration: str) -> Run:
    """Compile and simulate one point of the grid.

    A refusal is caught and recorded rather than raised, because a budget at
    which an arm produces no program is the answer for that point and not an
    error in the experiment. The allocator's own diagnostic is what goes in the
    row, because "does not place" with the reason beside it is the report
    Section 13.3 asks for and a bare boolean is not.
    """
    options = ARMS[configuration]
    row = Run(
        model=model,
        budget=budget,
        arm=ARM_OF[configuration],
        configuration=configuration,
        placed=False,
    )
    try:
        program = compile_model(
            onnx,
            level=2,
            emit="nbin",
            budget=budget,
            ablate=options.get("ablate"),
            spill_heuristic=options.get("spill_heuristic"),
            halo=options.get("halo"),
        )
    except CompileError as failure:
        row.refusal = " ".join(str(failure).split())[:400]
        return row

    assert program.binary is not None
    allocation = attributes(program.stages["npuisa"])
    inputs = make_inputs(
        INPUT_CLASS,
        program.input_shapes,
        model=model,
        batch=declared_batch(model),
    )
    simulation = run_program(program.binary, inputs, program.output_shapes)

    row.placed = True
    row.instructions = int(simulation.stats["instructions"])
    row.cycles = float(simulation.stats["cycles"])
    row.dram_bytes = int(simulation.stats["dram_bytes_read"]) + int(
        simulation.stats["dram_bytes_written"]
    )
    row.scratchpad_peak_bytes = int(allocation.get("scratchpad_peak_bytes", 0))
    row.scratchpad_bytes = int(allocation.get("scratchpad_bytes", 0))
    row.spill_count = int(allocation.get("spill_count", 0))
    row.spill_dma_count = int(allocation.get("spill_dma_count", 0))
    return row


def measure(models: list[str]) -> list[Run]:
    rows: list[Run] = []
    with tempfile.TemporaryDirectory(prefix="three-arms-") as directory:
        work = Path(directory)
        for model in models:
            onnx = generate_model(model, work, batch=declared_batch(model))
            for budget in SWEPT[model]:
                for configuration in ARMS:
                    rows.append(one_run(model, onnx, budget, configuration))
    return rows


@dataclass
class Crossover:
    """Where tiling overtakes spilling on one model, or that it never does."""

    model: str
    configuration: str
    budget: int | None
    reason: str
    checked: list[int] = field(default_factory=list)


#: The two tiling configurations a crossover is computed for.
#:
#: Both, because they are different questions. The fused one asks what tiling is
#: worth in the compiler as it stands, where fusion hides most of the operands
#: from it; the unfused one asks what it is worth when nothing does.
TILING_CONFIGURATIONS = ("tile-fused-recompute", "tile-unfused-recompute")


def crossover(rows: list[Run], model: str, configuration: str) -> Crossover:
    """The lowest budget at which tiling is at least as good as spilling.

    Compared against `spill-longest-range`, which is the allocator's default
    heuristic and therefore what the compiler does when tiling is not there.

    **Placing beats not placing, and cycles decide the rest.** An arm that
    produces a program where the other produces none has overtaken it in the only
    sense Section 13.3 cares about at that budget, and no cycle count is
    comparable with an absence. Where both place, cycles decide and a tie is read
    as tiling not overtaking: the burden is on the arm that adds transfers.
    """
    checked: list[int] = []
    best: int | None = None
    for budget in sorted(SWEPT[model], reverse=True):
        spill = _find(rows, model, budget, "spill-longest-range")
        tile = _find(rows, model, budget, configuration)
        if spill is None or tile is None:
            continue
        checked.append(budget)
        if not tile.placed:
            continue
        if not spill.placed:
            best = budget
            continue
        if tile.cycles < spill.cycles:
            best = budget
    if best is None:
        return Crossover(
            model=model,
            configuration=configuration,
            budget=None,
            reason=(
                "tiling never costs less than spilling, and never places where "
                "spilling does not, at any budget swept here"
            ),
            checked=checked,
        )
    return Crossover(
        model=model,
        configuration=configuration,
        budget=best,
        reason=(
            "tiling either costs fewer cycles than spilling here or places "
            "where spilling does not"
        ),
        checked=checked,
    )


def _find(rows: list[Run], model: str, budget: int, configuration: str) -> Run | None:
    for row in rows:
        if (
            row.model == model
            and row.budget == budget
            and row.configuration == configuration
        ):
            return row
    return None


def report(rows: list[Run]) -> None:
    models = sorted({row.model for row in rows})
    for model in models:
        print()
        print(f"=== {model} ===")
        print(
            f"{'budget':>8}  {'configuration':24}{'placed':>7}{'instr':>7}"
            f"{'cycles':>11}{'dram':>9}{'peak':>9}{'spills':>8}"
        )
        print("-" * 84)
        for budget in sorted(SWEPT[model], reverse=True):
            for configuration in ARMS:
                row = _find(rows, model, budget, configuration)
                if row is None:
                    continue
                if not row.placed:
                    print(
                        f"{budget:>8}  {configuration:24}{'no':>7}"
                        f"{'   does not place':>36}"
                    )
                    continue
                print(
                    f"{budget:>8}  {configuration:24}{'yes':>7}"
                    f"{row.instructions:>7}{row.cycles:>11.1f}"
                    f"{row.dram_bytes:>9}{row.scratchpad_peak_bytes:>9}"
                    f"{row.spill_count:>8}"
                )
        for configuration in TILING_CONFIGURATIONS:
            answer = crossover(rows, model, configuration)
            where = "none" if answer.budget is None else f"{answer.budget} bytes"
            print(f"  crossover, {configuration}: {where}. {answer.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="three_arms.py",
        description="Section 13.3's three arms, per model and per budget, at -O2.",
    )
    parser.add_argument("--models", nargs="+", default=sorted(SWEPT))
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    unknown = [name for name in arguments.models if name not in SWEPT]
    if unknown:
        print(f"three-arms: unknown models {unknown}.", file=sys.stderr)
        return 2
    missing = [name for name in arguments.models if name not in MODELS]
    if missing:
        print(f"three-arms: {missing} are not in the suite.", file=sys.stderr)
        return 2

    rows = measure(list(arguments.models))
    report(rows)
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "runs": [asdict(row) for row in rows],
            "crossovers": [
                asdict(crossover(rows, model, configuration))
                for model in sorted(arguments.models)
                for configuration in TILING_CONFIGURATIONS
            ],
        }
        arguments.json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nthree-arms: wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
