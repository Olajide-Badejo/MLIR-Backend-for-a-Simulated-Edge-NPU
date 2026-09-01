# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Turn the recorded results into LaTeX macros. Nothing is hand copied.

    python experiments/results_to_tex.py
    python experiments/results_to_tex.py --check

`report/generated/macros.tex` is **tracked, not gitignored**, and Section 6 says
why in one sentence: that is what makes law 3 of Section 0.2 enforceable, because
a test can then assert that the sha recorded in `macros.tex` equals the manifest
sha of the committed results.

**The sha is the load bearing part of this file.** Every macro below is a number
a report will print, and the point of generating them is that a number in a
report traces to a result file that traces to a commit. `\\npuResultsSha` is that
link, and `test/Python/test_traceability.py` is what stops it going stale: if
somebody re-records the results and does not regenerate this file, the sha here
names the old measurement and the test fails.

**Why a generator and not a table somebody types.** Section 20: the README must
never contain a number that is not reproducible from `experiments/results/`, and
Section 21.1 says the report's tables and figures come from here with nothing
hand copied. A hand typed number is not wrong because typing is unreliable; it is
wrong because it has no provenance, and a reader cannot tell a stale number from
a current one by looking at it.

The full report lands at P16. What lands here is the mechanism and the macros the
evaluation will use, because the mechanism is what P10's gate asks for and a
mechanism whose first user is six phases away is a mechanism nobody has run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = REPO_ROOT / "experiments" / "results"
GENERATED_DIR: Final[Path] = REPO_ROOT / "report" / "generated"
MACROS_PATH: Final[Path] = GENERATED_DIR / "macros.tex"

#: The macro name a reader looks for to find out which measurement a number came
#: from. Named as a constant because the test that enforces law 3 reads it.
SHA_MACRO: Final[str] = "npuResultsSha"

#: LaTeX macro names cannot hold digits or underscores, so a model name has to be
#: transliterated. Written out rather than computed so that the mapping is
#: reviewable and stable: a generated macro name that changed because somebody
#: improved the transliteration would silently break every reference in the
#: report.
_DIGITS: Final[dict[str, str]] = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}


class TexError(Exception):
    """The results cannot be turned into macros."""


def macro_name(*parts: str) -> str:
    """A LaTeX safe macro name from a cell key's pieces."""
    joined = "".join(
        segment.capitalize()
        for part in parts
        for segment in re.split(r"[^A-Za-z0-9]+", part)
        if segment
    )
    return "npu" + "".join(_DIGITS.get(character, character) for character in joined)


def load_cells() -> list[dict[str, Any]]:
    cells = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        cells.append(json.loads(path.read_text(encoding="utf-8")))
    if not cells:
        raise TexError(
            f"{RESULTS_DIR} holds no results, so there is nothing to generate "
            f"from. Run experiments/run_benchmarks.py first."
        )
    return cells


def results_sha(cells: list[dict[str, Any]]) -> str:
    """The one commit every committed cell was measured at.

    More than one is a refusal rather than a choice between them. A generated
    file naming one sha over results measured at two would be a provenance link
    that is wrong for some of what it covers, which is worse than none.
    """
    shas = {cell["manifest"]["git_sha"] for cell in cells}
    if len(shas) != 1:
        raise TexError(
            f"the committed results were measured at {len(shas)} different "
            f"commits: {sorted(shas)}. Law 3 needs one sha to name, so re-run "
            f"the whole suite rather than generating a file that is right about "
            f"some of it."
        )
    return shas.pop()


def macros(cells: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every macro this file emits, as (name, body) in emission order."""
    by_name = {cell["cell"]["name"]: cell for cell in cells}
    plain = [cell for cell in cells if cell["cell"]["ablated_pass"] is None]

    entries: list[tuple[str, str]] = [
        (SHA_MACRO, results_sha(cells)),
        ("npuResultsCells", str(len(cells))),
        ("npuResultsSchemaVersion", str(cells[0]["schema_version"])),
        ("npuResultsContentHash", cells[0]["content_hash"]),
        ("npuResultsGeneratorVersion", cells[0]["manifest"]["generator_version"]),
    ]

    # Per model, at each level, at the default budget and the declared batch.
    # That is the cell a headline number means when it says "lenet at -O2".
    for cell in plain:
        key = cell["cell"]
        if key["scratchpad_budget"] != "default":
            continue
        if key["batch"] != _declared_batch(by_name, key["model"]):
            continue
        stem = macro_name(key["model"], f"O{key['opt_level']}")
        entries.append((f"{stem}Instructions", str(cell["instruction_count"])))
        entries.append(
            (f"{stem}Cycles", _number(cell["simulation"]["simulated_cycles"]))
        )
        entries.append(
            (f"{stem}DramBytes", str(cell["simulation"]["dram_bytes_total"]))
        )
        entries.append((f"{stem}Macs", str(cell["simulation"]["macs"])))
        entries.append(
            (
                f"{stem}MaxAbsErrorVsOnnxruntime",
                _scientific(cell["accuracy"]["max_abs_error_vs_onnxruntime"]),
            )
        )

    # The ablation table's nonzero rows, which are the numbers a report argues
    # from. The zero rows are a fact about the table rather than a number, and
    # `docs/NUMBERS.md` states them in prose beside it.
    for cell in cells:
        key = cell["cell"]
        if key["ablated_pass"] is None or key["scratchpad_budget"] != "default":
            continue
        if cell["deltas"]["instruction_count"] == 0:
            continue
        stem = macro_name("ablate", key["ablated_pass"], key["model"])
        entries.append(
            (f"{stem}Instructions", str(cell["deltas"]["instruction_count"]))
        )
        entries.append((f"{stem}Cycles", _number(cell["deltas"]["simulated_cycles"])))

    duplicated = [
        name for name, _ in entries if [n for n, _ in entries].count(name) > 1
    ]
    if duplicated:
        raise TexError(
            f"two cells generated the same macro name: {sorted(set(duplicated))}"
        )
    return entries


def _declared_batch(by_name: dict[str, dict[str, Any]], model: str) -> int:
    """The batch the tight budget cell of this model was measured at.

    Read out of the results rather than out of the model registry, because this
    script's job is to describe what was recorded and a disagreement between the
    two is something a reader of the results wants to see rather than have
    silently resolved.
    """
    for cell in by_name.values():
        if (
            cell["cell"]["model"] == model
            and cell["cell"]["scratchpad_budget"] == "tight"
        ):
            return int(cell["cell"]["batch"])
    raise TexError(
        f"{model} has no tight budget cell, so its declared batch is unknown"
    )


def _number(value: float) -> str:
    """A float as LaTeX, without a trailing `.0` on a whole number."""
    if float(value).is_integer():
        return str(int(value))
    return repr(round(float(value), 4))


def _scientific(value: float) -> str:
    """A small number in the form the README and the report print it."""
    return f"{float(value):.3e}"


def render(cells: list[dict[str, Any]]) -> str:
    # The header this writes into the generated file is itself an SPDX tag, and
    # `reuse lint` scans *this* file too and reads the string literal below as a
    # malformed expression, because it ends in a quote and a comma. The
    # REUSE-IgnoreStart and REUSE-IgnoreEnd markers are the tool's own answer to
    # a file that contains a tag it does not own, and they are here rather than
    # a cleverer construction because the alternative is a generator whose
    # output header is assembled from fragments nobody can read.
    #
    # REUSE-IgnoreStart
    lines = [
        "% SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>",
        "%",
        "% SPDX-License-Identifier: MIT",
        "%",
        "% GENERATED by experiments/results_to_tex.py from experiments/results/.",
        "% Do not edit. Every number here is a field of a committed result file,",
        "% and npuResultsSha names the commit those results were measured at.",
        "% test/Python/test_traceability.py asserts that sha equals the manifest",
        "% sha of the committed results, which is law 3 of Section 0.2.",
        "",
    ]
    # REUSE-IgnoreEnd
    for name, body in macros(cells):
        lines.append(f"\\newcommand{{\\{name}}}{{{body}}}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="results_to_tex.py",
        description="Generate report/generated/macros.tex from experiments/results/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "regenerate and diff against the committed file instead of writing. "
            "This is the staleness gate: a result re-recorded without this file "
            "being regenerated leaves a sha naming the previous measurement."
        ),
    )
    arguments = parser.parse_args(argv)

    try:
        rendered = render(load_cells())
    except TexError as failure:
        print(f"results-to-tex: {failure}", file=sys.stderr)
        return 2

    if arguments.check:
        if not MACROS_PATH.is_file():
            print(f"results-to-tex: {MACROS_PATH} does not exist", file=sys.stderr)
            return 1
        current = MACROS_PATH.read_text(encoding="utf-8")
        if current != rendered:
            print(
                "results-to-tex: report/generated/macros.tex is stale. It is "
                "generated from experiments/results/, so a difference means the "
                "results moved and this file did not, and every number a report "
                "takes from it would name the previous measurement.",
                file=sys.stderr,
            )
            return 1
        print("results-to-tex: macros.tex is up to date")
        return 0

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    MACROS_PATH.write_text(rendered, encoding="utf-8")
    print(f"results-to-tex: wrote {MACROS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
