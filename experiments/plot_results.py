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


def load(budget: int) -> dict[int, dict]:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS / "*.json"))]
    return {
        r["opt_level"]: r
        for r in rows
        if r["model"] == "lenet" and r["scratchpad_budget"] == budget
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
        ax.text(x, v + top * 0.02, fmt.format(v), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, top * 1.15)


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
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "optimization_levels.png", dpi=150)
    print(f"wrote {OUT / 'optimization_levels.png'}")


if __name__ == "__main__":
    main()
