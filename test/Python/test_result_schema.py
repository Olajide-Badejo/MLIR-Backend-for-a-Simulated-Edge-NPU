# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The result schema of Section 16.1, and the rules that make it worth having.

Three of the rules in that section are the reason this file exists rather than a
JSON schema file, because each is a claim about *why* a field is the way it is
and a validator that only checked shapes would pass a file that broke all three.

**Explicit nulls, never absent keys.** A missing key is a schema violation and
the loader raises. A `null` is a legitimate "not measured in this cell", and it
carries a `<field>_null_reason` beside it so that a reader is told which phase
fills it rather than left to infer that the number is zero.

**The counted and the timed are never mixed.** `instruction_count` is an exact
integer and every wall clock is an object with an interval. A reader who cannot
tell which numbers carry uncertainty has been misled by the file's shape.

**A null read as a zero is a defect with its own regression test**, in Section
16.1's own words, and `test_a_null_never_aggregates_as_a_zero` is that test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from npu_frontend.predictions import commit_exists, require_full_history
from npu_frontend.results import (
    ACCURACY_KEYS,
    CELL_KEYS,
    EXTERNAL_KEYS,
    GROUPS,
    MANIFEST_KEYS,
    NORMALIZED_KEYS,
    NULL_REASONS,
    NULL_WITHOUT_REASON,
    RESULTS_DIR,
    ROOFLINE_KEYS,
    SCHEMA_VERSION,
    SIMULATION_KEYS,
    TOP_LEVEL_KEYS,
    TRIALS,
    CellKey,
    ResultSchemaError,
    aggregate,
    content_hash,
    field_at,
    load_result,
    null,
    timing_object,
    validate_result,
    values_of,
    write_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def committed_results() -> list[Path]:
    return sorted(RESULTS_DIR.glob("*.json"))


@pytest.fixture(scope="module")
def results() -> list[dict[str, Any]]:
    """Every committed cell, loaded through the validating reader."""
    files = committed_results()
    if not files:
        pytest.skip(
            "experiments/results/ is empty, so there is nothing to validate. "
            "Run experiments/run_benchmarks.py."
        )
    return [load_result(path) for path in files]


# ---------------------------------------------------------------------------
# Every field, in every file.
# ---------------------------------------------------------------------------


def test_every_committed_result_validates(results: list[dict[str, Any]]) -> None:
    """The gate clause: every schema field present in every result file."""
    assert results
    for result in results:
        validate_result(result)


def test_every_group_carries_every_key(results: list[dict[str, Any]]) -> None:
    """Checked against the tables rather than against one sample file.

    A file written from a stale copy of the writer would validate against itself
    and fail here, which is the direction that matters: the schema is the
    authority and the files are the things held to it.
    """
    for result in results:
        assert set(TOP_LEVEL_KEYS) <= set(result)
        for group, keys in GROUPS.items():
            assert set(keys) <= set(result[group]), (
                f"{result['cell']['name']} is missing "
                f"{sorted(set(keys) - set(result[group]))} from {group!r}"
            )


def test_the_groups_are_the_ones_section_16_1_names() -> None:
    """The tables themselves, so a field dropped from one is caught here.

    Written out, deliberately, because this is the one place in the project where
    a hardcoded list is the right answer: it is a transcription of a
    specification section, and its whole job is to disagree with the code when
    somebody edits the code without editing the specification.
    """
    assert set(CELL_KEYS) == {
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
    }
    assert {"macs", "effective_macs", "utilization", "delta"} <= set(SIMULATION_KEYS)
    assert {"tiling_choices", "quant_boundary_crossings"} <= set(SIMULATION_KEYS)
    assert {"roofline_verdict", "operational_intensity"} <= set(ROOFLINE_KEYS)
    assert {"energy_pj_per_inference", "edp", "macs_per_dram_byte"} <= set(
        NORMALIZED_KEYS
    )
    assert {"units_convention"} <= set(NORMALIZED_KEYS)
    assert {"scalesim_covered_cycle_fraction", "scalesim_covered_op_fraction"} <= set(
        EXTERNAL_KEYS
    )
    assert {
        "max_abs_error_vs_onnxruntime",
        "max_rel_error_vs_onnxruntime",
        "mean_abs_error_vs_onnxruntime",
        "sqnr_db_vs_fp32_simulated",
        "per_layer_sqnr_db",
        "max_abs_error_vs_fp32_simulated",
    } <= set(ACCURACY_KEYS)
    assert {"cpu_model", "endianness", "run_order_seed", "wall_clock_note"} <= set(
        MANIFEST_KEYS
    )


def test_the_counted_and_the_timed_are_not_mixed(results: list[dict[str, Any]]) -> None:
    """Section 16.1's reason for the `timing` object, asserted about the files.

    `instruction_count` is a bare `int` and every wall clock is an object with an
    interval. A reader who cannot tell which numbers carry uncertainty has been
    misled by the shape of the file, which is the failure this separation exists
    to prevent.
    """
    for result in results:
        assert isinstance(result["instruction_count"], int)
        assert not isinstance(result["instruction_count"], bool)
        for name in ("compile_ms", "passes_total_ms"):
            block = result["timing"][name]
            assert set(block) == {
                "n_trials",
                "median_ms",
                "ci95_low_ms",
                "ci95_high_ms",
                "iqr_ms",
            }
            assert block["n_trials"] >= TRIALS
            assert block["ci95_low_ms"] <= block["median_ms"] <= block["ci95_high_ms"]
            assert block["iqr_ms"] >= 0.0
        for entry in result["passes"]:
            assert entry["timing"]["n_trials"] >= TRIALS
            assert entry["pass_timing_source"].startswith("measured:")


# ---------------------------------------------------------------------------
# The nulls, and the four refusals.
# ---------------------------------------------------------------------------


def test_every_null_carries_a_reason(results: list[dict[str, Any]]) -> None:
    """The clause Section 16.1 spends a paragraph on.

    The exceptions come from `NULL_WITHOUT_REASON` rather than from a copy here.
    They are the fields whose null *is* the answer: a cell that ablates nothing
    has no `ablated_pass`, and a cell no prediction covers has no
    `prediction_id`, which Section 16.1 calls legitimate and common. A second
    copy of that set in a test would pass while the validator used a different
    one.
    """
    checked = 0
    for result in results:
        for group in GROUPS:
            for key, value in result[group].items():
                if key.endswith("_null_reason") or value is not None:
                    continue
                if key in NULL_WITHOUT_REASON:
                    continue
                reason = result[group].get(f"{key}_null_reason")
                assert reason, f"{result['cell']['name']}: {group}.{key} has no reason"
                assert len(reason) > 20, "a reason that says nothing is not a reason"
                checked += 1
    assert checked, "no null was checked, so this test asserted nothing"


def test_a_missing_key_is_refused(results: list[dict[str, Any]]) -> None:
    doctored = json.loads(json.dumps(results[0]))
    del doctored["simulation"]["macs"]
    with pytest.raises(ResultSchemaError) as failure:
        validate_result(doctored)
    assert "macs" in str(failure.value)


def test_a_null_without_a_reason_is_refused(results: list[dict[str, Any]]) -> None:
    """The refusal shown, because a check that has only ever passed is not one."""
    doctored = json.loads(json.dumps(results[0]))
    doctored["simulation"]["macs"] = None
    with pytest.raises(ResultSchemaError) as failure:
        validate_result(doctored)
    assert "macs_null_reason" in str(failure.value)
    assert "indistinguishable from a field somebody forgot" in str(failure.value)


def test_a_value_and_a_reason_together_are_refused(
    results: list[dict[str, Any]],
) -> None:
    """A leftover reason from the phase that filled the field.

    It would tell a reader the number is absent while the number sits beside it,
    which is worse than either half alone.
    """
    doctored = json.loads(json.dumps(results[0]))
    doctored["simulation"]["macs_null_reason"] = "P11, when Accelergy lands."
    with pytest.raises(ResultSchemaError) as failure:
        validate_result(doctored)
    assert "both present and absent" in str(failure.value)


def test_an_unknown_schema_version_is_refused(results: list[dict[str, Any]]) -> None:
    doctored = json.loads(json.dumps(results[0]))
    doctored["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ResultSchemaError) as failure:
        validate_result(doctored)
    assert "refused" in str(failure.value)


def test_a_reason_cannot_be_invented_at_the_call_site() -> None:
    """`null()` is the single home for the strings, and it refuses to be bypassed."""
    with pytest.raises(ResultSchemaError) as failure:
        null("a_field_nobody_declared")
    assert "two cells can give differently" in str(failure.value)

    identical = null("sqnr_db_vs_fp32_simulated", cause="identical")
    reference = null("sqnr_db_vs_fp32_simulated")
    assert identical["sqnr_db_vs_fp32_simulated"] is None
    assert (
        identical["sqnr_db_vs_fp32_simulated_null_reason"]
        != reference["sqnr_db_vs_fp32_simulated_null_reason"]
    ), "one field with two legitimate causes must not give one reason for both"


def test_every_declared_reason_names_a_phase_or_the_measurement() -> None:
    """A reason that does not say when the field arrives is not usable."""
    for key, reason in NULL_REASONS.items():
        assert len(reason) > 20, key
        assert any(
            marker in reason for marker in ("P11", "P13", "P14", "this cell")
        ), f"{key} does not say which phase fills it or why it cannot be filled"


# ---------------------------------------------------------------------------
# A null is never a zero.
# ---------------------------------------------------------------------------


def test_a_null_never_aggregates_as_a_zero(results: list[dict[str, Any]]) -> None:
    """Section 16.1: a plotting script that treats a null as zero is a defect.

    A defect with its own regression test, in that section's words, and this is
    it. The failure is invisible without a mechanism: a chart of
    `energy_pj_per_inference` over cells that have not measured it draws a row of
    zero height bars and reads as a finding rather than as an absence.
    """
    with pytest.raises(ResultSchemaError) as failure:
        values_of(results, "normalized.energy_pj_per_inference")
    assert "cannot go into an aggregate" in str(failure.value)
    assert "P11" in str(failure.value), "the recorded reason is quoted in the refusal"

    with pytest.raises(ResultSchemaError):
        aggregate(results, "external.scalesim_cycles")

    # And the same helper over a field that is measured returns the numbers.
    summary = aggregate(results, "simulation.macs")
    assert summary["n"] == len(results)
    assert summary["sum"] > 0.0


def test_a_path_that_is_not_a_field_raises(results: list[dict[str, Any]]) -> None:
    """A typo must not read as a field a later phase fills."""
    with pytest.raises(ResultSchemaError) as failure:
        field_at(results[0], "simulation.mackssss")
    assert "is not a field of this schema" in str(failure.value)


# ---------------------------------------------------------------------------
# The cell key.
# ---------------------------------------------------------------------------


def test_the_cell_name_is_the_key_and_the_file_stem(
    results: list[dict[str, Any]],
) -> None:
    for path, result in zip(committed_results(), results, strict=True):
        assert path.stem == result["cell"]["name"]


def test_an_ablation_names_a_baseline_that_is_not_itself() -> None:
    ablated = CellKey(
        model="lenet",
        opt_level=2,
        scratchpad_budget="tight",
        batch=1,
        ablated_pass="cse",
    )
    plain = CellKey(model="lenet", opt_level=2, scratchpad_budget="tight", batch=1)
    assert ablated.baseline_cell == plain.name
    assert ablated.name != plain.name
    assert plain.baseline_cell is None, (
        "a cell that ablates nothing naming itself as its baseline would be a "
        "delta of zero recorded as if it meant something"
    )


def test_a_cell_that_ablates_nothing_carries_no_deltas(
    results: list[dict[str, Any]],
) -> None:
    for result in results:
        if result["cell"]["ablated_pass"] is None:
            assert result["deltas"] is None
        else:
            assert set(result["deltas"]) == {
                "instruction_count",
                "simulated_cycles",
                "dram_bytes_total",
                "macs",
            }


def test_the_deltas_are_the_subtraction_they_claim(
    results: list[dict[str, Any]],
) -> None:
    """Section 16.1 stores the delta beside the absolutes so nobody has to trust it.

    So this test does not trust it either: it recomputes every delta from the two
    cells and compares. A stored delta nobody checks is a stored delta that can
    be wrong.
    """
    by_name = {result["cell"]["name"]: result for result in results}
    checked = 0
    for result in results:
        baseline_name = result["cell"]["baseline_cell"]
        if baseline_name is None:
            continue
        baseline = by_name[baseline_name]
        assert result["deltas"]["instruction_count"] == (
            result["instruction_count"] - baseline["instruction_count"]
        )
        for field in ("simulated_cycles", "dram_bytes_total", "macs"):
            assert result["deltas"][field] == pytest.approx(
                result["simulation"][field] - baseline["simulation"][field]
            )
        checked += 1
    assert checked, "no ablation cells were checked, so this test asserted nothing"


# ---------------------------------------------------------------------------
# Provenance and staleness.
# ---------------------------------------------------------------------------


def test_every_manifest_git_sha_resolves(results: list[dict[str, Any]]) -> None:
    """Section 16.1, line for line: a committed number tracing to a commit that
    does not exist is the exact failure this test prevents.

    **The history guard comes first, and it is D-0041.** In a shallow checkout
    every one of these shas is absent, and without the guard this test reports
    each of them as a commit that does not exist. That is a true statement about
    the checkout and a false one about the repository, and the false reading is
    the one somebody acts on.
    """
    require_full_history("resolving every recorded manifest.git_sha")
    for result in results:
        sha = result["manifest"]["git_sha"]
        assert commit_exists(sha), (
            f"{result['cell']['name']} records git_sha {sha}, which is not a "
            f"commit in this repository. The repository is not shallow and git "
            f"answered the question rather than refusing it, so the commit is "
            f"genuinely absent."
        )


def test_the_content_hash_is_not_the_git_sha(results: list[dict[str, Any]]) -> None:
    """Staleness is keyed on content, and the two keys must not be confused.

    Section 16.1 corrects an earlier draft here: keying staleness on the sha
    marks every result stale the moment it is committed, because a result is
    committed after the code it measures and can never carry the sha of the
    commit that contains it. This asserts the file keeps both and that they are
    different things.
    """
    for result in results:
        assert result["content_hash"] != result["manifest"]["git_sha"]
        assert len(result["content_hash"]) == 64


def test_the_content_hash_moves_with_a_compiler_source(tmp_path: Path) -> None:
    """It is a hash of the inputs to the number, so a changed input changes it.

    Driven through the tool version record rather than by editing a source file,
    because a test that wrote into `lib/` would be a test that can leave the
    working tree dirty when it fails.
    """
    first = content_hash(tool_versions={"llvm": "22.1.8"})
    same = content_hash(tool_versions={"llvm": "22.1.8"})
    other = content_hash(tool_versions={"llvm": "22.1.9"})
    assert first == same
    assert first != other, (
        "the resolved tool version record is one of the three inputs Section "
        "16.1 names, so a moved tool version has to move the hash"
    )


def test_two_cells_from_the_same_tree_share_a_content_hash(
    results: list[dict[str, Any]],
) -> None:
    assert len({result["content_hash"] for result in results}) == 1, (
        "the committed cells were measured against more than one state of the "
        "sources, so at least one of them is stale"
    )


# ---------------------------------------------------------------------------
# The timing object's own arithmetic.
# ---------------------------------------------------------------------------


def test_a_timing_object_needs_ten_whole_pipeline_repetitions() -> None:
    with pytest.raises(ResultSchemaError) as failure:
        timing_object([1.0, 2.0, 3.0])
    assert "not ten runs of one pass" in str(failure.value)


def test_the_interval_brackets_the_median() -> None:
    samples = [float(value) for value in range(1, 21)]
    block = timing_object(samples)
    assert block["n_trials"] == 20
    assert block["ci95_low_ms"] <= block["median_ms"] <= block["ci95_high_ms"]
    assert block["iqr_ms"] == pytest.approx(9.5)


def test_a_result_is_written_atomically(
    results: list[dict[str, Any]], tmp_path: Path
) -> None:
    """The `.tmp` rename Section 16.1 asks for, and no `.tmp` left behind."""
    written = write_result(json.loads(json.dumps(results[0])), tmp_path)
    assert written.is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert load_result(written)["cell"]["name"] == results[0]["cell"]["name"]
