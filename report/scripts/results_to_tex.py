"""Turn experiments/results/*.json into LaTeX included by the report.

Nothing in the report is hand copied: this script regenerates the results table
and a few macros from the recorded experiments. It is idempotent.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "results"
OUT = Path(__file__).resolve().parent.parent / "generated"


def load_rows() -> list[dict]:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS / "*.json"))]
    return sorted(
        rows, key=lambda r: (r["model"], r["scratchpad_budget"], r["opt_level"])
    )


def results_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & O & Budget (KB) & Instrs & Cycles & DRAM (KB) & Max error \\",
        r"\midrule",
    ]
    for r in rows:
        budget_kb = r["scratchpad_budget"] // 1024
        dram_kb = r["dram_bytes_total"] / 1024
        err = f'{r["max_abs_error_vs_onnxruntime"]:.1e}'
        lines.append(
            f'{r["model"]} & {r["opt_level"]} & {budget_kb} & '
            f'{r["instruction_count"]} & {r["simulated_cycles"]} & '
            f"{dram_kb:.1f} & {err} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def macros(rows: list[dict]) -> str:
    # Key numbers cited inline, taken from the default budget LeNet cells.
    default = {
        r["opt_level"]: r
        for r in rows
        if r["model"] == "lenet" and r["scratchpad_budget"] == 1048576
    }
    out = []
    if {0, 1, 2} <= set(default):
        out.append(rf"\newcommand{{\OaInstr}}{{{default[0]['instruction_count']}}}")
        out.append(rf"\newcommand{{\OcInstr}}{{{default[2]['instruction_count']}}}")
        out.append(
            rf"\newcommand{{\OaDram}}{{{default[0]['dram_bytes_total'] // 1024}}}"
        )
        out.append(
            rf"\newcommand{{\ObDram}}{{{default[1]['dram_bytes_total'] // 1024}}}"
        )
        out.append(rf"\newcommand{{\OaCycles}}{{{default[0]['simulated_cycles']}}}")
        out.append(rf"\newcommand{{\OcCycles}}{{{default[2]['simulated_cycles']}}}")
    manifest = rows[0]["manifest"] if rows else {}
    out.append(rf"\newcommand{{\LlvmTag}}{{{manifest.get('llvm_tag', 'unknown')}}}")
    out.append(rf"\newcommand{{\GitSha}}{{{manifest.get('git_sha', 'unknown')[:12]}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("no results found; run experiments/run_benchmarks.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_table.tex").write_text(results_table(rows))
    (OUT / "macros.tex").write_text(macros(rows))
    print(f"wrote {OUT}/results_table.tex and macros.tex from {len(rows)} cells")


if __name__ == "__main__":
    main()
