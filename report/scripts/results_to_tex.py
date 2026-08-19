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
    """The full cells only.

    Ablation records live in the same directory and carry model, opt_level, and
    scratchpad_budget too, so they would otherwise appear as extra rows in the
    results table and corrupt the macros, which pick the -O2 cell by key.
    """
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS / "*.json"))]
    rows = [r for r in rows if "ablated_pass" not in r]
    return sorted(
        rows, key=lambda r: (r["model"], r["scratchpad_budget"], r["opt_level"])
    )


def load_ablations() -> list[dict]:
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS / "*.json"))]
    rows = [r for r in rows if "ablated_pass" in r]
    return sorted(
        rows, key=lambda r: (-r["scratchpad_budget"], r["ablated_pass"])
    )


def ablation_table(rows: list[dict]) -> str:
    """Leave one out ablation deltas, both budgets, against full -O2.

    Deltas are ablated minus baseline, so a positive instruction delta means
    removing the pass costs instructions, which is the pass earning its place.
    """
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Budget & Pass removed & Instrs & $\Delta$ & Cycles & $\Delta$ & "
        r"$\Delta$ DRAM (KB) \\",
        r"\midrule",
    ]
    for r in rows:
        budget_kb = r["scratchpad_budget"] // 1024
        pass_name = r["ablated_pass"].replace("_", r"\_")
        lines.append(
            f'{budget_kb} KB & \\texttt{{{pass_name}}} & '
            f'{r["instruction_count"]} & {r["delta_instruction_count"]:+d} & '
            f'{r["simulated_cycles"]} & {r["delta_simulated_cycles"]:+d} & '
            f'{r["delta_dram_bytes_total"] / 1024:+.1f} ' + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def ablation_macros(rows: list[dict]) -> list[str]:
    """Macros for the evaluation prose, so no number there is hand typed."""
    out = []
    default = [r for r in rows if r["scratchpad_budget"] == 1048576]
    if not default:
        return out
    # The pass that costs the most cycles when removed is the one buying most.
    best = max(default, key=lambda r: r["delta_simulated_cycles"])
    out.append(
        rf"\newcommand{{\BestPass}}{{\texttt{{{best['ablated_pass']}}}}}"
    )
    out.append(rf"\newcommand{{\BestPassCycles}}{{{best['delta_simulated_cycles']}}}")
    out.append(
        rf"\newcommand{{\BestPassInstrs}}{{{best['delta_instruction_count']}}}"
    )
    # The same pass at the tight budget, by name rather than by "best". At the
    # tight budget every delta is zero or negative, so a max() would nominate a
    # pass that bought nothing and read as though it had won.
    tight = {
        r["ablated_pass"]: r for r in rows if r["scratchpad_budget"] == 143360
    }
    same = tight.get(best["ablated_pass"])
    if same:
        out.append(
            rf"\newcommand{{\BestPassTightCycles}}"
            rf"{{{same['delta_simulated_cycles']}}}"
        )
        out.append(
            rf"\newcommand{{\BestPassTightDram}}"
            rf"{{{same['delta_dram_bytes_total'] / 1024:.1f}}}"
        )
    return out


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
    out += ablation_macros(load_ablations())
    return "\n".join(out) + "\n"


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("no results found; run experiments/run_benchmarks.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_table.tex").write_text(results_table(rows))
    (OUT / "macros.tex").write_text(macros(rows))
    ablations = load_ablations()
    (OUT / "ablation_table.tex").write_text(ablation_table(ablations))
    print(
        f"wrote {OUT}/results_table.tex, macros.tex, and ablation_table.tex "
        f"from {len(rows)} cells and {len(ablations)} ablations"
    )


if __name__ == "__main__":
    main()
