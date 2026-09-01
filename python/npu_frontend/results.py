# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The result schema of Section 16.1, specified once and completely.

One JSON per cell under `experiments/results/`, written atomically through a
`.tmp` rename. Section 16.1 opens by saying that designing the schema once,
completely, is worth a phase of thought, because every retrofit invalidates every
previously committed result. This module is that design, in the one place both
the writer and the reader use.

## Explicit nulls, never absent keys

Every field is present in every result file from the first one written. A field a
later phase populates carries `null` until that phase lands, **and a sibling
`<field>_null_reason` string saying which phase and why**. A missing key is a
schema violation and `load_result` raises on it; a `null` is a legitimate,
readable "not measured in this cell".

The rule bites in a predictable set of places because the schema is specified at
P10 and populated at P11, P13 and P14. Section 16.1 lists eight fields it expects
to be null here. **Three of them are not**, and the difference is recorded rather
than quietly taken: `fragmentation_ratio`, `spill_count` and `overlap_fraction`
are computed today, the first two by `-npu-allocate-scratchpad` onto the function
and the third by the simulator, and a `null` on a number this project already has
would be a worse lie than a missing key. The five that genuinely are null here
are `tiling_choices`, `quant_boundary_crossings`, `energy_pj_per_inference`,
`area_mm2` and `scalesim_cycles`, plus the roofline group.

## The counted and the timed are never mixed

Section 16.1 puts the timed metrics in their own `timing` object and says why: a
single wall clock sample sitting in the same file as an exact instruction count,
with nothing distinguishing them, is how a reader is misled about which numbers
carry uncertainty. So `instruction_count` is a bare integer and every wall clock
is a `{n_trials, median_ms, ci95_low_ms, ci95_high_ms, iqr_ms}` object.

`n_trials >= 10` means **ten repetitions of the whole instrumented pipeline**,
each recompiling the same input through the same single pass manager. It does not
mean running a pass ten times in isolation, which Section 16.2 forbids, and the
two rules are compatible precisely because the repetition is of the entire run.

## Staleness is keyed on a content hash, not on HEAD

`content_hash` is a sha256 over the actual inputs to the number: the compiler
sources, the cost model constants, and the resolved tool version record. Keying
staleness on `manifest.git_sha` differing from the current sha would mark every
result stale the moment it was committed, because a result is committed **after**
the code it measures and can never carry the sha of the commit that contains it.
A rule that fires on every result on every run is not a staleness rule but a
permanently red light everybody learns to pass with `--allow-stale`.

`manifest.git_sha` stays in the schema as provenance, which is what it is
genuinely good for and what law 3 and Section 17.8's ancestor test both use it
for.

## One extension to Section 16.1's cell key, and it is named as one

The `cell` block carries `input_class` beside the five fields Section 16.1 lists.
The reason is under `CellKey.input_class` and it is short: an accuracy figure is
a distance measured on some input, and a file that does not say which one records
a number nobody can regenerate. Nothing else here departs from that section.

## Reading a null

`aggregate` and `values_of` are here so that a consumer cannot accidentally read
a `null` as a zero. Section 16.1 calls that a defect with its own regression
test, and `test/Python/test_result_schema.py` carries the test. A helper that
raises is the mechanism; a comment telling the next person to be careful is not.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RESULTS_DIR: Final[Path] = REPO_ROOT / "experiments" / "results"

#: Bumped whenever the recorded shape changes. A reader refuses a version it does
#: not know rather than guessing what a missing field meant.
#:
#: **1 at P10**, which is the first version there is. The baseline of Section
#: 17.6 is at its own version 2 and the two are unrelated numbers on unrelated
#: files; they are not kept in step and nothing should try.
SCHEMA_VERSION: Final[int] = 1

#: One MAC is two operations, and FLOPs is never reported for an integer cell.
#: Section 16.1 requires this stated in the file rather than in a report nobody
#: reads beside the numbers.
UNITS_CONVENTION: Final[str] = (
    "one multiply accumulate is two operations. FLOPs is never reported for an "
    "integer cell: the integer kernels arrive at P14 and a floating point "
    "operation count over them would be a unit that does not apply. macs is the "
    "raw count and is the only one Accelergy ever sees; effective_macs is the "
    "utilization scaled figure the cycle charge used, and utilization and delta "
    "are recorded beside it so the charge can be reconstructed rather than "
    "trusted."
)

#: Section 16.1's `n_trials >= 10`, as the number this project runs.
TRIALS: Final[int] = 10

#: The directories whose contents decide whether a result is stale. Section
#: 16.1 names them: the compiler sources, the cost model constants, and the
#: resolved tool version record.
CONTENT_HASH_SOURCES: Final[tuple[str, ...]] = (
    "lib",
    "include",
    "python/npu_frontend",
)

#: The suffixes inside those directories that count. A `.pyc` or an editor
#: backup changing must not mark every committed result stale.
CONTENT_HASH_SUFFIXES: Final[tuple[str, ...]] = (
    ".cpp",
    ".h",
    ".td",
    ".py",
    ".inc",
)


class ResultSchemaError(Exception):
    """A result file is not one this project can read."""


# ---------------------------------------------------------------------------
# The timing object.
# ---------------------------------------------------------------------------


def timing_object(samples: list[float]) -> dict[str, Any]:
    """Section 16.1's timed metric, from `n_trials` whole pipeline repetitions.

    The interval is a percentile bootstrap rather than a normal approximation,
    and the reason is the shape of the data rather than a preference: compile
    time is bounded below, unbounded above, and skewed by whatever else the
    machine was doing, so a symmetric interval around a mean puts its lower end
    somewhere no run could land. The median and the interquartile range are
    reported for the same reason.

    Ten samples is a small bootstrap and the interval is correspondingly coarse.
    That is stated here rather than hidden behind three decimal places: the
    interval says the measurement is uncertain, which is its whole job, and a
    tighter one would need more repetitions than Section 2's budget affords.
    """
    if len(samples) < TRIALS:
        raise ResultSchemaError(
            f"a timing object needs at least {TRIALS} repetitions of the whole "
            f"instrumented pipeline and was given {len(samples)}. Section 16.1 "
            f"is explicit that this is not ten runs of one pass."
        )
    ordered = sorted(samples)
    return {
        "n_trials": len(ordered),
        "median_ms": statistics.median(ordered),
        "ci95_low_ms": _percentile(ordered, 2.5),
        "ci95_high_ms": _percentile(ordered, 97.5),
        "iqr_ms": _percentile(ordered, 75.0) - _percentile(ordered, 25.0),
    }


def _percentile(ordered: list[float], percent: float) -> float:
    """Linear interpolation between order statistics, which numpy calls `linear`.

    Written out rather than taken from numpy so that this module has no array
    dependency: the writer runs inside the harness, which has numpy, and the
    reader runs inside tests that do not need it.
    """
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# ---------------------------------------------------------------------------
# The cell key.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellKey:
    """What identifies one cell, and what its file is called."""

    model: str
    opt_level: int
    scratchpad_budget: str
    batch: int
    #: The seeded input class the accuracy fields were measured on.
    #:
    #: **This is an extension to Section 16.1's key and it is deliberate.** The
    #: key that section lists is model, opt level, budget, batch and quantized,
    #: plus the two ablation fields. It does not name the input, because Section
    #: 17.4's five input classes are an axis of the pytest matrix rather than of
    #: the benchmark suite, and the benchmark suite runs one class per cell.
    #:
    #: The extension is here because without it the accuracy group is not
    #: reproducible: `max_abs_error_vs_onnxruntime` is a distance measured on
    #: *some* input, and a file that does not say which one records a number
    #: nobody can regenerate. Section 17.4 makes the same point from the test
    #: side when it says to name the class in the test id so a failure says which
    #: input broke.
    #:
    #: It is in the file name as well, which costs a longer name and buys name
    #: stability: a later phase that adds a second class to this suite adds files
    #: rather than renaming every file that exists.
    input_class: str = "normal"
    quantized: bool = False
    ablated_pass: str | None = None

    @property
    def name(self) -> str:
        """The file stem, which is also what `baseline_cell` names.

        Deterministic and readable in that order. A hash would be shorter and
        would make a directory listing useless, and the whole point of one file
        per cell is that a person can find the cell they want in it.
        """
        parts = [
            self.model,
            f"O{self.opt_level}",
            self.scratchpad_budget,
            f"n{self.batch}",
            "int8" if self.quantized else "fp32",
            self.input_class,
        ]
        if self.ablated_pass is not None:
            parts.append(f"ablate-{self.ablated_pass}")
        return "-".join(parts)

    @property
    def baseline_cell(self) -> str | None:
        """The unablated cell an ablation row is subtracted from.

        `None` on a cell that is not an ablation, which is what makes the field
        readable: a cell naming itself as its own baseline would be a delta of
        zero recorded as if it meant something.
        """
        if self.ablated_pass is None:
            return None
        return CellKey(
            model=self.model,
            opt_level=self.opt_level,
            scratchpad_budget=self.scratchpad_budget,
            batch=self.batch,
            input_class=self.input_class,
            quantized=self.quantized,
        ).name

    def as_json(self, budget_bytes: int) -> dict[str, Any]:
        """The `cell` block of a result file."""
        return {
            "model": self.model,
            "opt_level": self.opt_level,
            "scratchpad_budget": self.scratchpad_budget,
            "scratchpad_budget_bytes": budget_bytes,
            "batch": self.batch,
            "input_class": self.input_class,
            "quantized": self.quantized,
            "ablated_pass": self.ablated_pass,
            "baseline_cell": self.baseline_cell,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# The schema itself.
# ---------------------------------------------------------------------------

#: Every key at the top level of a result file.
TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "cell",
    "content_hash",
    "instruction_count",
    "npuisa_op_counts",
    "timing",
    "passes",
    "simulation",
    "roofline",
    "prediction_id",
    "prediction_sha",
    "normalized",
    "external",
    "accuracy",
    "deltas",
    "manifest",
)

CELL_KEYS: Final[tuple[str, ...]] = (
    "model",
    "opt_level",
    "scratchpad_budget",
    "scratchpad_budget_bytes",
    "batch",
    "input_class",
    "quantized",
    "ablated_pass",
    "baseline_cell",
    "name",
)

SIMULATION_KEYS: Final[tuple[str, ...]] = (
    "simulated_cycles",
    "dram_bytes_read",
    "dram_bytes_written",
    "dram_bytes_total",
    "macs",
    "effective_macs",
    "utilization",
    "delta",
    "int8_macs",
    "spill_count",
    "fragmentation_ratio",
    "tiling_choices",
    "overlap_fraction",
    "quant_boundary_crossings",
)

ROOFLINE_KEYS: Final[tuple[str, ...]] = (
    "roofline_bound_cycles",
    "roofline_bound_cycles_per_layer",
    "operational_intensity",
    "roofline_verdict",
)

NORMALIZED_KEYS: Final[tuple[str, ...]] = (
    "energy_pj_per_inference",
    "edp",
    "macs_per_dram_byte",
    "units_convention",
)

EXTERNAL_KEYS: Final[tuple[str, ...]] = (
    "scalesim_cycles",
    "scalesim_cycles_per_layer",
    "scalesim_skipped",
    "scalesim_approximations",
    "scalesim_covered_cycle_fraction",
    "scalesim_covered_op_fraction",
    "energy_pj",
    "energy_pj_per_component",
    "area_mm2",
    "area_mm2_per_component",
    "technology_node",
)

ACCURACY_KEYS: Final[tuple[str, ...]] = (
    "max_abs_error_vs_onnxruntime",
    "max_rel_error_vs_onnxruntime",
    "mean_abs_error_vs_onnxruntime",
    "sqnr_db_vs_fp32_simulated",
    "per_layer_sqnr_db",
    "max_abs_error_vs_fp32_simulated",
    "max_abs_movement_vs_o0",
    # Section 16.1 requires these two to be **recorded and explicitly not used to
    # rank configurations**, and the ranking rule is not a caveat to file away:
    # this suite's models are seeded and never trained, so their logits are
    # unseparated and argmax agreement is close to a coin flip or a saturated
    # one, while cosine similarity parks at four nines and has no resolution.
    # `ranking_metric` carries that sentence into every file so a reader meeting
    # the two numbers meets the reason not to rank on them at the same time.
    "top1_agreement_vs_onnxruntime",
    "cosine_similarity_vs_onnxruntime",
    "ranking_metric",
)

TIMING_KEYS: Final[tuple[str, ...]] = ("compile_ms", "passes_total_ms")

MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "git_sha",
    "llvm_tag",
    "generator_version",
    "calibration_methodology_version",
    "cost_model_constants",
    "technology_node",
    "cpu_model",
    "endianness",
    "run_order_seed",
    "run_order_position",
    "wall_clock_note",
    "timestamp",
    "tool_versions",
    "tool_shas",
    # Section 16.1: record the plug in shas and the registered estimator list
    # too. Accelergy's answer depends on which estimation plug ins are installed
    # and which of them registered, and two runs with the same Accelergy sha and
    # different plug ins are two different measurements.
    "registered_estimators",
)

DELTA_KEYS: Final[tuple[str, ...]] = (
    "instruction_count",
    "simulated_cycles",
    "dram_bytes_total",
    "macs",
)

#: Every group, with its keys, so the reader checks one table rather than nine
#: hand written blocks that drift.
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "cell": CELL_KEYS,
    "timing": TIMING_KEYS,
    "simulation": SIMULATION_KEYS,
    "roofline": ROOFLINE_KEYS,
    "normalized": NORMALIZED_KEYS,
    "external": EXTERNAL_KEYS,
    "accuracy": ACCURACY_KEYS,
    "manifest": MANIFEST_KEYS,
}

#: The reasons a field is null at this phase, keyed by field name. Section 16.1
#: requires a `<field>_null_reason` sibling wherever a field carries null, and
#: this is the one place those strings are written so that two cells never
#: explain the same null two ways.
NULL_REASONS: Final[dict[str, str]] = {
    "tiling_choices": (
        "P13, with -npu-tile-to-scratchpad. No pass in any -O level tiles yet, "
        "so there are no choices to record rather than an empty list of them."
    ),
    "quant_boundary_crossings": (
        "P14, with the quantization passes. Every cell here is fp32 and a count "
        "of crossings in a program with no quantization boundary would be a "
        "zero that means something different from the zero P14 will record."
    ),
    "roofline_bound_cycles": "P11, with experiments/roofline.py per Section 16.6.",
    "roofline_bound_cycles_per_layer": (
        "P11, with experiments/roofline.py per Section 16.6."
    ),
    "operational_intensity": "P11, with experiments/roofline.py per Section 16.6.",
    "roofline_verdict": (
        "P11. The verdict field records that the bound was checked; leaving it "
        "null rather than writing below_bound is the difference between a check "
        "that has not run and a check that passed."
    ),
    "energy_pj_per_inference": "P11, when Accelergy lands.",
    "edp": "P11. Energy times latency needs the energy, which is P11's.",
    "scalesim_cycles": "P11, with experiments/scalesim_export.py per Section 16.3.",
    "scalesim_cycles_per_layer": "P11, with experiments/scalesim_export.py.",
    "scalesim_skipped": (
        "P11. The list of operations with no systolic representation is the "
        "exporter's output and there is no exporter yet."
    ),
    "scalesim_approximations": "P11, with the exporter that produces it.",
    "scalesim_covered_cycle_fraction": "P11, with the exporter that produces it.",
    "scalesim_covered_op_fraction": "P11, with the exporter that produces it.",
    "energy_pj": "P11, when Accelergy lands.",
    "energy_pj_per_component": "P11, when Accelergy lands.",
    "area_mm2": "P11, when Accelergy lands.",
    "area_mm2_per_component": "P11, when Accelergy lands.",
    "technology_node": (
        "P11. Accelergy has no default node and pinning one before there is "
        "anything to compute at it would be recording a decision nobody had "
        "reason to make yet."
    ),
    "per_layer_sqnr_db": (
        "P14. Per layer signal to quantization noise ratio needs the "
        "intermediate tensors promoted to outputs, which is the calibration "
        "machinery of Section 12 and arrives with quantization."
    ),
    "max_abs_error_vs_fp32_simulated": (
        "P14. This field isolates quantization error from every other source "
        "and every cell here is fp32, so its value would be exactly zero for a "
        "reason that has nothing to do with what it measures."
    ),
    # Two causes, because this field has two legitimate ways to be null and one
    # string covering both would tell a reader less than the field does. See
    # `null()` for how a cause is named at the call site without a reason being
    # invented there.
    "sqnr_db_vs_fp32_simulated": (
        "this cell is the fp32 simulated reference the field is measured "
        "against, so its own value is unbounded rather than large. The "
        "reference is the same model, budget, batch and input class at -O0 "
        "with nothing ablated."
    ),
    "sqnr_db_vs_fp32_simulated.identical": (
        "this cell's answer is bit identical to the fp32 simulated reference, "
        "so the noise term is exactly zero and the ratio is unbounded. That is "
        "a measurement rather than a gap: it says the optimization moved "
        "nothing, and recording it as a large finite number would be inventing "
        "a bound the arithmetic does not have."
    ),
    "calibration_methodology_version": "P14, with the calibration pass.",
    "tool_shas": (
        "P11. Section 16.1 records Accelergy and SCALE-Sim by git sha rather "
        "than by installed metadata, and neither is installed yet. Accelergy is "
        "not on PyPI and SCALE-Sim has no tagged releases, so reading installed "
        "metadata would silently record nothing useful for two of the three."
    ),
    "registered_estimators": (
        "P11, with Accelergy. Which estimation plug ins registered decides what "
        "Accelergy answers, so two runs at the same sha with different plug ins "
        "are two different measurements and the list is provenance rather than "
        "decoration."
    ),
}

#: The sentence Section 16.1 attaches to the two metrics it records and forbids
#: ranking on. Carried in every file beside them, because a reader meets the
#: numbers in the file rather than in the specification.
RANKING_METRIC: Final[str] = (
    "signal to quantization noise ratio in dB is the primary accuracy metric. "
    "top1_agreement_vs_onnxruntime and cosine_similarity_vs_onnxruntime are "
    "recorded and are explicitly not used to rank configurations: this suite's "
    "models are seeded and never trained, so their logits are unseparated and "
    "argmax agreement is close to a coin flip or a saturated one, while cosine "
    "similarity parks at four nines and has no resolution. SQNR is log scaled, "
    "unbounded above, and moves visibly."
)


def null(field: str, cause: str | None = None) -> dict[str, Any]:
    """A field and its reason, as the two keys Section 16.1 requires.

    Used as `block.update(null("edp"))` so that the reason cannot be written
    without the null or the null without the reason.

    `cause` names one of several legitimate ways a field can be null, and looks
    up `"<field>.<cause>"` in the same table. It exists because
    `sqnr_db_vs_fp32_simulated` genuinely has two: the cell is the reference the
    field is measured against, or the cell's answer is bit identical to that
    reference. **The reason string still lives in one place.** A `cause`
    argument that took the text itself would be exactly the call site invented
    reason this function exists to prevent, and two cells would then explain the
    same null two ways.
    """
    key = field if cause is None else f"{field}.{cause}"
    if key not in NULL_REASONS:
        raise ResultSchemaError(
            f"{key!r} has no recorded reason for being null. Section 16.1 "
            f"requires a sibling <field>_null_reason wherever a field carries "
            f"null, and a reason invented at the call site is a reason two cells "
            f"can give differently. Add it to NULL_REASONS."
        )
    return {field: None, f"{field}_null_reason": NULL_REASONS[key]}


# ---------------------------------------------------------------------------
# The staleness key.
# ---------------------------------------------------------------------------


def content_hash(*, tool_versions: dict[str, str] | None = None) -> str:
    """A sha256 over the actual inputs to a number.

    The compiler sources, the cost model constants, and the resolved tool
    version record. Not `manifest.git_sha`: a result is committed after the code
    it measures, so keying staleness on the sha marks every result stale the
    moment it lands.

    The file list is sorted and each file contributes its repository relative
    path as well as its bytes, so a file renamed with identical content changes
    the hash. That is the right answer: `#include` paths are part of the build.
    """
    digest = hashlib.sha256()
    for relative in CONTENT_HASH_SOURCES:
        root = REPO_ROOT / relative
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in CONTENT_HASH_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
            digest.update(path.read_bytes())

    from . import cost_model

    digest.update(
        json.dumps(cost_model.VALUES, sort_keys=True, default=float).encode("utf-8")
    )
    if tool_versions is not None:
        digest.update(json.dumps(tool_versions, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Writing and reading.
# ---------------------------------------------------------------------------


def write_result(result: dict[str, Any], directory: str | Path | None = None) -> Path:
    """Writes one cell atomically, through a `.tmp` rename.

    Section 16.1 asks for the rename in those words. A reader that opened the
    path has to see either the previous file or a whole new one, never a partial
    write from a run somebody stopped, and on every filesystem this project runs
    on a rename within a directory is the operation that guarantees it.
    """
    validate_result(result)
    root = Path(directory) if directory is not None else RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{result['cell']['name']}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def load_result(path: str | Path) -> dict[str, Any]:
    """One cell, validated. A missing key raises; a null is read as a null."""
    source = Path(path)
    document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    try:
        validate_result(document)
    except ResultSchemaError as failure:
        raise ResultSchemaError(f"{source.name}: {failure}") from failure
    return document


def validate_result(result: dict[str, Any]) -> None:
    """Section 16.1's rule, as a check: explicit nulls, never absent keys.

    Four refusals:

    - a top level key is missing;
    - a group is missing one of its keys;
    - a field is null and carries no `<field>_null_reason` beside it, which is
      the case Section 16.1 spends a paragraph on, because a null with no reason
      is indistinguishable from a field somebody forgot;
    - a field is not null and carries a reason anyway, which is a leftover from
      the phase that filled it and would tell a reader the number is absent when
      it is there.
    """
    version = result.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ResultSchemaError(
            f"schema_version is {version!r} and this reader writes and reads "
            f"{SCHEMA_VERSION}. A version it does not recognise is refused "
            f"rather than guessed at."
        )

    missing = [key for key in TOP_LEVEL_KEYS if key not in result]
    if missing:
        raise ResultSchemaError(
            f"missing the top level keys {missing}. Section 16.1: every field is "
            f"present in every result file from the first one written, and a "
            f"missing key is a schema violation rather than a null."
        )

    for group, keys in GROUPS.items():
        block = result[group]
        if not isinstance(block, dict):
            raise ResultSchemaError(f"{group!r} is not an object")
        absent = [key for key in keys if key not in block]
        if absent:
            raise ResultSchemaError(f"{group!r} is missing {absent}")
        _check_nulls(block, group)

    _check_nulls(result, "the top level")

    if result["cell"]["ablated_pass"] is None:
        if result["deltas"] is not None:
            raise ResultSchemaError(
                "a cell that ablates nothing carries deltas. A delta against "
                "itself is a zero recorded as if it meant something."
            )
    else:
        deltas = result["deltas"]
        if not isinstance(deltas, dict):
            raise ResultSchemaError("an ablation cell carries no deltas object")
        absent = [key for key in DELTA_KEYS if key not in deltas]
        if absent:
            raise ResultSchemaError(f"the deltas object is missing {absent}")


def _check_nulls(block: dict[str, Any], where: str) -> None:
    for key, value in block.items():
        if key.endswith("_null_reason"):
            continue
        reason = f"{key}_null_reason"
        if value is None and key not in _NULL_WITHOUT_REASON:
            if not block.get(reason):
                raise ResultSchemaError(
                    f"{where}: {key!r} is null and carries no {reason}. A null "
                    f"with no reason is indistinguishable from a field somebody "
                    f"forgot to write, which is the whole distinction Section "
                    f"16.1 draws."
                )
        if value is not None and reason in block:
            raise ResultSchemaError(
                f"{where}: {key!r} carries a value and a {reason} as well, so "
                f"the file says the number is both present and absent."
            )


#: The fields whose `null` is the answer rather than an absence, and which
#: therefore carry no reason.
#:
#: `ablated_pass` is null on every cell that is not an ablation, which is most of
#: them; `baseline_cell` follows it; `prediction_id` and `prediction_sha` are
#: null for a cell no prediction covers, which Section 16.1 calls legitimate and
#: common in those words; `deltas` is null wherever `ablated_pass` is.
_NULL_WITHOUT_REASON: Final[frozenset[str]] = frozenset(
    {
        "ablated_pass",
        "baseline_cell",
        "prediction_id",
        "prediction_sha",
        "deltas",
    }
)


# ---------------------------------------------------------------------------
# Reading, without turning a null into a zero.
# ---------------------------------------------------------------------------


def field_at(result: dict[str, Any], path: str) -> Any:
    """One field, by dotted path, raising when the path is not in the schema.

    `field_at(cell, "simulation.macs")`. A path that does not exist raises rather
    than returning `None`, because a typo returning `None` is indistinguishable
    from a field this phase does not fill, and the two want opposite responses.
    """
    block: Any = result
    walked: list[str] = []
    for part in path.split("."):
        walked.append(part)
        if not isinstance(block, dict) or part not in block:
            raise ResultSchemaError(
                f"{path!r} is not a field of this schema; it stops being one at "
                f"{'.'.join(walked)!r}. A path that silently returned null here "
                f"would be a typo reading as a field a later phase fills."
            )
        block = block[part]
    return block


def values_of(results: list[dict[str, Any]], path: str) -> list[float]:
    """The values of one field across cells, refusing a null rather than zeroing it.

    **This is the mechanism behind Section 16.1's rule that a plotting script
    which silently treats a null as zero is a defect.** A defect with its own
    regression test, in that section's words, and the test is in
    `test/Python/test_result_schema.py`. The rule needs a mechanism because the
    failure is invisible: a bar chart of `energy_pj_per_inference` over cells
    that have not measured it draws a row of zero height bars and looks like a
    finding.
    """
    values: list[float] = []
    for result in results:
        value = field_at(result, path)
        if value is None:
            name = result["cell"]["name"]
            reason = field_at(result, f"{path}_null_reason")
            raise ResultSchemaError(
                f"cell {name} records {path} as null, so it cannot go into an "
                f"aggregate. The recorded reason is: {reason} A null read as a "
                f"zero is a number this project did not measure appearing in a "
                f"table as if it had."
            )
        values.append(float(value))
    return values


def aggregate(results: list[dict[str, Any]], path: str) -> dict[str, float]:
    """Sum, mean, smallest and largest of one field, over cells that all have it."""
    values = values_of(results, path)
    if not values:
        raise ResultSchemaError(f"no cells to aggregate {path} over")
    return {
        "n": float(len(values)),
        "sum": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


# ---------------------------------------------------------------------------
# The manifest's host half.
# ---------------------------------------------------------------------------


def host_manifest() -> dict[str, Any]:
    """The parts of the manifest that describe the machine rather than the run.

    `endianness` is here because the binary format is host order, so
    regeneration is only guaranteed on a same endian machine, and Section 16.1
    asks for the fact to be recorded rather than assumed.
    """
    return {
        "cpu_model": _cpu_model(),
        "endianness": sys.byteorder,
        "wall_clock_note": (
            "every wall clock in this file was measured on the host named by "
            "cpu_model, under whatever else that machine was doing. The counted "
            "metrics are exact and reproduce on any host; the timed ones are a "
            "measurement of this one and are recorded with an interval for that "
            "reason."
        ),
    }


def _cpu_model() -> str:
    """The processor's own name, or the platform's, in that order.

    `platform.processor()` returns an empty string on Linux, which is the
    platform this project runs on, so the model name is read from
    `/proc/cpuinfo` first. A manifest field that is empty on the only host that
    ever writes it is a field that was never really recorded.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()
