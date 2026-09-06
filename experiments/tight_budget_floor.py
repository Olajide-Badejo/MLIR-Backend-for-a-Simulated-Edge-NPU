# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The smallest budget each model still allocates at, re-measured where tiling is.

ADR 0008 defines a tight budget as **the smallest budget at which that program
allocates**, swept at 64 byte granularity, and freezes the seven constants it
found on 2026-08-31. It also says what would make them worth measuring again, in
as many words: the recorded fraction "becomes a live knob at P13, when tiling
gives the compiler a way to fit an instruction whose operands exceed the budget".

This script is that re-measurement. **It is an input to Section 13.3 and it moves
nothing.** The suite's recorded budgets stay exactly where ADR 0008 put them,
because the tiling disabled ablation row is only checkable at the budgets the old
numbers were measured at, and because moving them would move every tight budget
cell in the project's history at once. What the floor is worth is telling Section
13.3's sweep where the interesting budgets are.

**The `--ablate` option is what makes the answer a measurement of a pass rather
than of the tree.** The floor at `-O2` as it stands is one number; the floor with
`-npu-tile-to-scratchpad` left out is the same number without the pass, and the
difference between them is the budget tiling buys. Section 16.2's leave one out
ablation takes one pass at a time by design, so a configuration with two passes
removed is not expressible through the level, and this script does not assemble a
pass list beside it: `docs/PASSES.md` says why, which is that an ablation
assembled outside the level measures a pass list rather than the compiler.

**The method, and where it differs from ADR 0008's.** ADR 0008 swept downward in
64 byte steps. This brackets the floor by bisection and then **checks the
boundary exhaustively**, because bisection alone assumes the predicate is
monotone in the budget and **on this tree it is not**: with `-npu-fuse-ops` left
out, `conv_bn_relu_stack` fails to allocate at 4928 bytes and allocates at 4736.
A smaller budget makes the tiling search size the operation differently, and a
mapping that fits can sit below one that does not. When the window finds such a
point the search stops bisecting and walks down in 64 byte steps until its
patience runs out, and the number it reports is then the smallest budget seen to
allocate rather than a proof that nothing below it does. The report says which of
the two any given row is.

Usage:

    python experiments/tight_budget_floor.py
    python experiments/tight_budget_floor.py --models resnet_block lenet
    python experiments/tight_budget_floor.py --ablate npu-tile-to-scratchpad
    python experiments/tight_budget_floor.py --json experiments/results-floors.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL = REPO_ROOT / "build" / "bin" / "npu-opt"
MODELS_DIR = REPO_ROOT / "experiments" / "models"

#: The allocator's own alignment, which is what ADR 0008 swept in and the only
#: quantum at which a smaller budget can change a placement at all.
QUANTUM = 64

#: How far below the bisected answer every 64 byte step is compiled. Ten steps is
#: 640 bytes, which is ten times the alignment. It is a check on the bisection
#: rather than a search, so it is deliberately small and exhaustive.
WINDOW_STEPS = 10

#: How many consecutive failures end the exhaustive descent a non monotone point
#: triggers. Twenty four steps is 1536 bytes. It is a stopping rule rather than a
#: proof, and the report says so.
PATIENCE_STEPS = 24

#: ADR 0008's constants, quoted here so the report can put the two side by side.
#: A test holds this table and `model_generator.MODELS` in step, so a constant
#: that moved in one and not the other is a red rather than a quiet
#: disagreement.
FROZEN: dict[str, int] = {
    "lenet": 194624,
    "depthwise_separable": 8192,
    "resnet_block": 6464,
    "inception_block": 6144,
    "conv_bn_relu_stack": 6464,
    "dilated_stack": 8064,
    "lenet_batched": 200832,
}

#: One budget in, allocates or does not out. The search below takes this rather
#: than a tool path and a model name, so that a test can drive it over a
#: predicate chosen to be awkward instead of over a compiler.
Allocates = Callable[[int], bool]

_STAT = re.compile(r"\(S\)\s+(\d+)\s+(\S+)")


@dataclass
class Floor:
    """One model's smallest allocatable budget, and what it does there."""

    model: str
    ablated: str
    frozen: int
    floor: int | None
    bisected: int | None = None
    non_monotone: list[int] = field(default_factory=list)
    descended: bool = False
    compiles: int = 0
    tiled_ops_at_floor: int = 0
    spills_at_floor: int = 0
    prefetched_at_floor: int = 0
    tiled_ops_at_frozen: int = 0
    spills_at_frozen: int = 0
    prefetched_at_frozen: int = 0

    @property
    def moved(self) -> int | None:
        """How far the floor fell, or `None` when it could not be found."""
        return None if self.floor is None else self.frozen - self.floor


# ---------------------------------------------------------------------------
# The search, over a predicate.
# ---------------------------------------------------------------------------


def bisect_floor(allocates: Allocates, ceiling: int) -> int | None:
    """The smallest budget that allocates, bracketed by bisection.

    `ceiling` is a budget expected to allocate, and `None` comes back when it
    does not: a search for a floor under a ceiling that is itself below the floor
    has no answer and must not invent one.
    """
    if not allocates(ceiling):
        return None

    low = ceiling
    candidate = QUANTUM
    while low > QUANTUM:
        candidate = max(QUANTUM, (low // 2) // QUANTUM * QUANTUM)
        if not allocates(candidate):
            break
        low = candidate
    else:
        return QUANTUM

    bad, good = candidate, low
    while good - bad > QUANTUM:
        middle = (bad + good) // 2 // QUANTUM * QUANTUM
        if middle in (bad, good):
            break
        if allocates(middle):
            good = middle
        else:
            bad = middle
    return good


def verify_window(allocates: Allocates, floor: int) -> list[int]:
    """Every 64 byte step below the answer, compiled.

    Returns the budgets below `floor` that allocate anyway, which is what a non
    monotone predicate looks like from here. An empty list is the bisection
    confirmed over the window.
    """
    surprises: list[int] = []
    for step in range(1, WINDOW_STEPS + 1):
        candidate = floor - step * QUANTUM
        if candidate < QUANTUM:
            break
        if allocates(candidate):
            surprises.append(candidate)
    return surprises


def descend(allocates: Allocates, start: int) -> int:
    """Every 64 byte step down from a known success, until the patience runs out.

    Returns the smallest budget seen to allocate. Exhaustive rather than
    bisected, because the whole reason this is reached is that the predicate is
    not monotone, and bisecting a non monotone predicate answers a question
    nobody asked.
    """
    smallest = start
    misses = 0
    candidate = start
    while candidate > QUANTUM and misses < PATIENCE_STEPS:
        candidate -= QUANTUM
        if allocates(candidate):
            smallest = candidate
            misses = 0
        else:
            misses += 1
    return smallest


# ---------------------------------------------------------------------------
# The compiler, behind that predicate.
# ---------------------------------------------------------------------------


def statistics(
    tool: Path, model: str, budget: int, ablated: str = ""
) -> tuple[bool, dict[str, int]]:
    """Whether the model allocates at this budget, and the pass statistics.

    The exit code is the answer. An allocator that cannot place emits its own
    diagnostic and fails the pass, which is what ADR 0008's sweep read too.
    """
    source = MODELS_DIR / f"{model}-O0.npu.mlir"
    leave_out = f" ablate={ablated}" if ablated else ""
    completed = subprocess.run(
        [
            str(tool),
            str(source),
            f"--pass-pipeline=builtin.module(npu-O2{{budget={budget}{leave_out}}})",
            "-mlir-pass-statistics",
            "-mlir-pass-statistics-display=list",
            "-o",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    counts = {name: int(value) for value, name in _STAT.findall(completed.stderr)}
    return completed.returncode == 0, counts


def measure(tool: Path, models: list[str], ablated: str = "") -> list[Floor]:
    results: list[Floor] = []
    for model in models:
        frozen = FROZEN[model]
        compiles = 0

        def allocates(budget: int, name: str = model) -> bool:
            nonlocal compiles
            compiles += 1
            ok, _ = statistics(tool, name, budget, ablated)
            return ok

        ok_frozen, at_frozen = statistics(tool, model, frozen, ablated)
        compiles += 1
        if not ok_frozen:
            print(
                f"tight-budget-floor: {model} does not allocate at its own frozen "
                f"budget of {frozen}. That is a finding rather than a floor, and "
                f"the sweep stops here for this model.",
                file=sys.stderr,
            )
            results.append(
                Floor(
                    model=model,
                    ablated=ablated,
                    frozen=frozen,
                    floor=None,
                    compiles=compiles,
                )
            )
            continue

        bisected = bisect_floor(allocates, frozen)
        surprises = verify_window(allocates, bisected) if bisected else []
        floor = bisected
        descended = False
        if surprises:
            descended = True
            floor = descend(allocates, min(surprises))

        at_floor: dict[str, int] = {}
        if floor is not None:
            _, at_floor = statistics(tool, model, floor, ablated)
            compiles += 1

        results.append(
            Floor(
                model=model,
                ablated=ablated,
                frozen=frozen,
                floor=floor,
                bisected=bisected,
                non_monotone=surprises,
                descended=descended,
                compiles=compiles,
                tiled_ops_at_floor=at_floor.get("tiled-ops", 0),
                spills_at_floor=at_floor.get("spilled-buffers", 0),
                prefetched_at_floor=at_floor.get("prefetched", 0),
                tiled_ops_at_frozen=at_frozen.get("tiled-ops", 0),
                spills_at_frozen=at_frozen.get("spilled-buffers", 0),
                prefetched_at_frozen=at_frozen.get("prefetched", 0),
            )
        )
    return results


def report(results: list[Floor]) -> None:
    ablated = {row.ablated for row in results}
    label = f"with {sorted(ablated)[-1]} left out" if any(ablated) else "as it stands"
    print()
    print(f"configuration: -O2 {label}")
    print(
        f"{'model':22}{'frozen':>9}{'floor':>9}{'fell by':>9}{'tiled':>7}"
        f"{'spills':>8}{'prefetch':>10}{'compiles':>10}"
    )
    print("-" * 84)
    for row in results:
        floor = "none" if row.floor is None else str(row.floor)
        fell = "" if row.moved is None else str(row.moved)
        print(
            f"{row.model:22}{row.frozen:>9}{floor:>9}{fell:>9}"
            f"{row.tiled_ops_at_floor:>7}{row.spills_at_floor:>8}"
            f"{row.prefetched_at_floor:>10}{row.compiles:>10}"
        )
    print()
    print(
        "The frozen column is ADR 0008's constant and this script does not move "
        f"it. The floor column is bisected and then checked over a "
        f"{WINDOW_STEPS * QUANTUM} byte window in {QUANTUM} byte steps."
    )
    for row in results:
        if row.non_monotone:
            print(
                f"  {row.model}: the bisected boundary is {row.bisected} and it "
                f"allocates at {row.non_monotone} below that, so allocatability "
                f"is not monotone in the budget. The search walked down from the "
                f"lowest of those in {QUANTUM} byte steps until "
                f"{PATIENCE_STEPS} consecutive failures and reports {row.floor}, "
                f"which is the smallest budget seen to allocate rather than a "
                f"proof that nothing below it does."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tight_budget_floor.py",
        description=(
            "Re-measure the smallest allocatable budget per model, as an input "
            "to Section 13.3. ADR 0008's constants are not changed by this."
        ),
    )
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--models", nargs="+", default=sorted(FROZEN))
    parser.add_argument(
        "--ablate",
        default="",
        help=(
            "leave one ablatable pass out of -O2, by its argument, so that the "
            "floor is a measurement of that pass rather than of the tree"
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if not arguments.tool.is_file():
        print(f"tight-budget-floor: {arguments.tool} is not built.", file=sys.stderr)
        return 2
    unknown = [name for name in arguments.models if name not in FROZEN]
    if unknown:
        print(f"tight-budget-floor: unknown models {unknown}.", file=sys.stderr)
        return 2

    results = measure(arguments.tool, list(arguments.models), arguments.ablate)
    report(results)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps([row.__dict__ for row in results], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"tight-budget-floor: wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
