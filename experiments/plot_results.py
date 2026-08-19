"""Plot the benchmark results into figures used by the README and the report.

Reads experiments/results/*.json and writes PNGs to docs/images/. All numbers
are simulated estimates from the analytical cost model.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
OUT = REPO / "docs" / "images"

BLUE = "#2563eb"
SLATE = "#64748b"
GREEN = "#16a34a"


def _all_rows() -> list[dict]:
    return [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS / "*.json"))]


def load(budget: int) -> dict[int, dict]:
    """The three full cells at a budget, keyed by optimization level.

    Ablation records carry model lenet, opt_level 2, and a scratchpad_budget
    too, so they have to be excluded explicitly. Without the filter one of them
    would replace the real -O2 row here, and which one would depend on glob
    order, which is the kind of thing that produces a plot nobody can reproduce.
    """
    return {
        r["opt_level"]: r
        for r in _all_rows()
        if r["model"] == "lenet"
        and r["scratchpad_budget"] == budget
        and "ablated_pass" not in r
    }


def load_ablations(budget: int) -> dict[str, dict]:
    return {
        r["ablated_pass"]: r
        for r in _all_rows()
        if r["model"] == "lenet"
        and r["scratchpad_budget"] == budget
        and "ablated_pass" in r
    }


def bars(ax, values, color, title, ylabel, fmt="{:,}"):
    levels = ["-O0", "-O1", "-O2"]
    xs = range(len(levels))
    ax.bar(xs, values, color=color, width=0.62, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(levels)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    top = max(values)
    for x, v in zip(xs, values, strict=True):
        ax.text(
            x,
            v + top * 0.02,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, top * 1.15)


def ablation_panel(ax, budget: int, metric: str, title: str, ylabel: str, color, scale=1.0):
    """One grouped panel: full -O2 plus one bar per ablated pass."""
    full = load(budget)[2]
    ablations = load_ablations(budget)
    names = sorted(ablations)
    labels = ["full -O2"] + [f"no\n{n}" for n in names]
    values = [full[metric] / scale] + [ablations[n][metric] / scale for n in names]

    xs = range(len(labels))
    colors = [color] + ["#cbd5e1" if v == values[0] else SLATE for v in values[1:]]
    ax.bar(xs, values, color=colors, width=0.62, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    top = max(values) if max(values) else 1
    for x, v in zip(xs, values, strict=True):
        ax.text(x, v + top * 0.02, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, top * 1.18)


def ablation_figure() -> None:
    """Leave one out ablations, both budgets.

    A bar equal to the full -O2 reference is drawn pale: it means removing that
    pass changed nothing, which is a result rather than a gap in the data.
    """
    budgets = [(1048576, "1 MB scratchpad"), (143360, "140 KB scratchpad")]
    if not load_ablations(budgets[0][0]):
        raise SystemExit("no ablation results; run experiments/run_benchmarks.py first")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (budget, label) in enumerate(budgets):
        ablation_panel(axes[row][0], budget, "instruction_count",
                       f"Instructions ({label})", "count", SLATE)
        ablation_panel(axes[row][1], budget, "simulated_cycles",
                       f"Simulated cycles ({label})", "cycles", BLUE)
        ablation_panel(axes[row][2], budget, "dram_bytes_total",
                       f"DRAM traffic ({label})", "KB", GREEN, scale=1024)
    fig.suptitle(
        "What each -O2 pass buys: full pipeline against the pipeline without it\n"
        "(LeNet, simulated estimates; a pale bar means removing the pass changed nothing)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "ablations.png", dpi=150)
    print(f"wrote {OUT / 'ablations.png'}")


def main() -> None:
    d = load(1048576)
    if {0, 1, 2} - set(d):
        raise SystemExit("run experiments/run_benchmarks.py first")
    OUT.mkdir(parents=True, exist_ok=True)

    cycles = [d[i]["simulated_cycles"] for i in (0, 1, 2)]
    dram = [d[i]["dram_bytes_total"] / 1024 for i in (0, 1, 2)]
    instrs = [d[i]["instruction_count"] for i in (0, 1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    bars(axes[0], instrs, SLATE, "Instructions", "count")
    bars(axes[1], cycles, BLUE, "Simulated cycles", "cycles")
    bars(axes[2], dram, GREEN, "DRAM traffic", "KB", fmt="{:.0f}")
    fig.suptitle(
        "What each optimization level buys (LeNet, 1 MB scratchpad, simulated estimates)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "optimization_levels.png", dpi=150)
    print(f"wrote {OUT / 'optimization_levels.png'}")

    ablation_figure()


if __name__ == "__main__":
    main()
