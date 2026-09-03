# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 16.2's instrumentation, and the two claims its gate clause makes.

The first claim is that the counts come from a `PassInstrumentation` in
`runBeforePass` and `runAfterPass` **and from no flag**. That is not something a
test can assert about the implementation, so what is asserted here is the
property that only that implementation has: a before and an after pair for every
pass in a running pipeline, with one pass's after count matching the next pass's
before count across a stage boundary. `--print-op-stats` prints one summary for
one invocation and cannot produce any of it.

The second claim is that **a pass present in the pipeline and absent from the
JSON raises**, and it is shown twice. Once against a file this test doctors, so
the raise is exercised with the pass named; and once against the real file, so
the real file is known to contain every pass rather than merely to be readable.

**The cross check against `--mlir-timing` runs on the same invocation**, not on
a second one. Two runs of a compiler give two wall clocks and comparing them
would be a measurement of the machine's variance. One run instrumented two ways
is a comparison of the two instruments.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from npu_frontend import (
    CompileError,
    PassRecord,
    PassStatisticsError,
    ablatable_passes,
    compile_model,
    cross_check_against_mlir_timing,
    expected_passes,
    generate_model,
    load_pass_stats,
    pass_stats,
)
from npu_frontend.pass_stats import (
    MLIR_TIMING_DECIMALS,
    REQUIRED_PASS_KEYS,
    parse_mlir_timing,
)

from tools import tool

#: One model, because this file is about the instrumentation rather than about
#: the models. `dilated_stack` is the one that gives every `-O2` pass something
#: to do, which is what makes its counts move rather than merely exist.
MODEL = "dilated_stack"


@dataclass(frozen=True)
class Instrumented:
    """One `-O2` compilation, instrumented and timed in the same run."""

    path: Path
    records: list[PassRecord]
    timing: str
    directory: Path
    onnx: Path


@pytest.fixture(scope="module")
def instrumented(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Instrumented]:
    tool("npu-opt")
    directory = tmp_path_factory.mktemp("instrumentation")
    onnx = generate_model(MODEL, directory)
    stats = directory / "stats.json"
    result = compile_model(
        onnx, level=2, emit="nbin", pass_stats_json=stats, mlir_timing=True
    )
    yield Instrumented(
        path=stats,
        records=load_pass_stats(stats, expected=expected_passes(2)),
        timing=result.mlir_timing_text,
        directory=directory,
        onnx=Path(onnx),
    )


# ---------------------------------------------------------------------------
# The counts.
# ---------------------------------------------------------------------------


def test_every_pass_has_a_before_and_an_after_count(
    instrumented: Instrumented,
) -> None:
    """The pair per pass, which is what no flag produces."""
    records = instrumented.records
    assert [record.name for record in records] == expected_passes(2)
    for record in records:
        assert record.ops_before_total == sum(record.ops_before.values())
        assert record.ops_after_total == sum(record.ops_after.values())
        assert record.ops_before_total > 0
        assert record.ops_after_total > 0
        assert record.wall_ms > 0.0
        assert record.pass_timing_source.startswith("measured:")


def test_the_counts_chain_across_the_pipeline(instrumented: Instrumented) -> None:
    """One pass's after is the next pass's before, once the anchor is allowed for.

    This is the property that shows the counts are taken from a running pipeline
    rather than from independent invocations. Where two neighbouring passes share
    an anchor operation the two totals are equal exactly. Where the anchor
    changes, from `func.func` to `builtin.module` or back, the module form counts
    one more operation, the module itself, and that difference is asserted rather
    than tolerated: a chain that broke by any other amount would mean something
    ran between two passes that this file does not know about.
    """
    records = instrumented.records
    for earlier, later in zip(records, records[1:], strict=False):
        if earlier.anchor_op == later.anchor_op:
            expected = earlier.ops_after_total
        elif later.anchor_op == "builtin.module":
            expected = earlier.ops_after_total + 1
        else:
            expected = earlier.ops_after_total - 1
        assert later.ops_before_total == expected, (
            f"{earlier.name} left {earlier.ops_after_total} operations under "
            f"{earlier.anchor_op} and {later.name} found "
            f"{later.ops_before_total} under {later.anchor_op}"
        )


def test_the_passes_that_change_the_count_are_the_ones_that_should(
    instrumented: Instrumented,
) -> None:
    """A census that is all zeros would satisfy every check above and nothing else.

    Two passes are expected to grow the number on this model and the reason is
    named for each. `-npu-fuse-ops` grows it, because a fused region adds the
    `npu.fused_op` and its `npu.yield` around operations it does not remove, and
    `-npu-lower-to-npuisa` grows it, because one tensor level operation becomes a
    transfer, a compute and a transfer back.
    """
    moved = {
        record.name: record.ops_delta
        for record in instrumented.records
        if record.ops_delta != 0
    }
    assert moved, (
        "no pass changed the operation count on a model chosen because every "
        "-O2 pass has something to do on it, which would mean the after count "
        "is the before count copied"
    )
    assert moved["npu-fuse-ops"] > 0
    assert moved["npu-lower-to-npuisa"] > 0


def test_the_op_names_are_real_operation_names(instrumented: Instrumented) -> None:
    """The census keys, so a count of `1` under a made up name would be caught."""
    first = instrumented.records[0]
    assert "func.func" in first.ops_before
    assert any(name.startswith("npu.") for name in first.ops_before)
    for name in first.ops_before:
        assert "." in name, f"{name!r} is not a dialect qualified operation name"


# ---------------------------------------------------------------------------
# The raise.
# ---------------------------------------------------------------------------


def test_a_pass_absent_from_the_json_raises(
    instrumented: Instrumented, tmp_path: Path
) -> None:
    """The gate clause, shown.

    The file is doctored rather than the compiler, because the fault this guards
    against is a pass that ran and was not recorded, and there is no way to make
    the instrumentation drop a pass without adding a way to make it drop a pass.
    A fault injection flag in the instrumentation would be a second code path
    inside the thing under test.
    """
    document = json.loads(instrumented.path.read_text(encoding="utf-8"))
    dropped = document["passes"].pop(4)
    doctored = tmp_path / "missing-a-pass.json"
    doctored.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PassStatisticsError) as failure:
        load_pass_stats(doctored, expected=expected_passes(2))
    message = str(failure.value)
    assert dropped["name"] in message
    assert "in the pipeline and absent from the file" in message


def test_a_pass_in_the_json_and_not_in_the_pipeline_raises(
    instrumented: Instrumented, tmp_path: Path
) -> None:
    """The same fault from the other side, which is what a failed ablation is."""
    document = json.loads(instrumented.path.read_text(encoding="utf-8"))
    doctored = tmp_path / "an-extra-pass.json"
    doctored.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PassStatisticsError) as failure:
        load_pass_stats(doctored, expected=expected_passes(2, ablated="cse"))
    assert "recorded and not in the pipeline" in str(failure.value)


@pytest.mark.parametrize("key", REQUIRED_PASS_KEYS)
def test_a_missing_field_is_refused_rather_than_defaulted(
    instrumented: Instrumented, tmp_path: Path, key: str
) -> None:
    """Section 16.2: a missing timing or op count is an error, never a zero."""
    document = json.loads(instrumented.path.read_text(encoding="utf-8"))
    del document["passes"][0][key]
    doctored = tmp_path / f"without-{key}.json"
    doctored.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PassStatisticsError) as failure:
        load_pass_stats(doctored)
    assert key in str(failure.value)


def test_a_run_that_did_not_complete_is_refused(
    instrumented: Instrumented, tmp_path: Path
) -> None:
    """The counts after a failed pass describe a half rewritten operation."""
    document = json.loads(instrumented.path.read_text(encoding="utf-8"))
    document["run_completed"] = False
    document["failed_pass"] = "npu-fuse-ops"
    doctored = tmp_path / "incomplete.json"
    doctored.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(PassStatisticsError) as failure:
        load_pass_stats(doctored)
    assert "did not complete" in str(failure.value)
    assert "npu-fuse-ops" in str(failure.value)


def test_an_absent_file_raises_rather_than_reading_as_an_empty_pipeline(
    tmp_path: Path,
) -> None:
    with pytest.raises(PassStatisticsError) as failure:
        load_pass_stats(tmp_path / "never-written.json")
    assert "never got as far as building a pipeline" in str(failure.value)


def test_the_import_stage_refuses_the_flag_rather_than_writing_nothing(
    instrumented: Instrumented,
) -> None:
    """An empty file and a pipeline whose passes vanished look the same."""
    with pytest.raises(CompileError) as failure:
        compile_model(
            instrumented.onnx,
            level=2,
            emit="import",
            pass_stats_json=instrumented.directory / "never.json",
        )
    assert "runs no pass manager at all" in str(failure.value)


# ---------------------------------------------------------------------------
# The cross check.
# ---------------------------------------------------------------------------


def test_the_two_clocks_agree_on_the_same_run(instrumented: Instrumented) -> None:
    """Section 16.2's cross check, on one invocation instrumented two ways."""
    check = cross_check_against_mlir_timing(instrumented.records, instrumented.timing)
    assert [row[0] for row in check.rows] == [
        record.pass_name for record in instrumented.records
    ]
    assert check.mlir_total_ms > 0.0
    assert check.instrumented_total_ms > 0.0

    # **The direction is what makes this a check; the allowance is what makes it
    # true.** MLIR's timer opens before this project's instrumentation is called
    # and closes after it, so MLIR's figure contains the operation walk and this
    # project's does not.
    #
    # The allowance is D-0043. MLIR prints seconds to four decimals, so each of
    # its figures is a multiple of 0.1 ms standing for a number this project
    # cannot see, and a sum of eleven of them is within 0.55 ms of the true sum.
    # This was a strict `>=` between that sum and an exact one, and it went red
    # in CI's coverage job on a margin of 1.7 microseconds, which is a
    # hundredth of one figure's rounding.
    assert check.totals_agree, (
        f"MLIR {check.mlir_total_ms:.6f} ms against instrumented "
        f"{check.instrumented_total_ms:.6f} ms over {len(check.rows)} passes, "
        f"allowance {check.rounding_allowance_ms:.4f} ms"
    )
    assert check.rounding_allowance_ms == pytest.approx(
        len(check.rows) * check.half_ulp_ms
    )
    assert check.half_ulp_ms > 0.0, (
        "a report with no printed precision would make every bound here zero, "
        "which is the check switching itself off rather than tightening"
    )


class _FakeMonitoring:
    """`sys.monitoring` with a controlled tool table.

    Substituted rather than consulted, because the real one reports whatever the
    interpreter running these tests happens to use, which is the thing being
    tested around.
    """

    _MAX_TOOLS = 6

    def __init__(self, tools: dict[int, str], raises: bool = False) -> None:
        self._tools = tools
        self._raises = raises

    def get_tool(self, tool_id: int) -> str | None:
        if self._raises:
            raise ValueError("this interpreter refuses the question")
        return self._tools.get(tool_id)


@pytest.mark.parametrize(
    ("gettrace", "threadtrace", "monitoring", "expected"),
    [
        (object(), None, _FakeMonitoring({}), True),
        (None, object(), _FakeMonitoring({}), True),
        (None, None, _FakeMonitoring({2: "coverage.py"}), True),
        (None, None, _FakeMonitoring({}), False),
        (None, None, None, False),
        (None, None, _FakeMonitoring({}, raises=True), False),
    ],
    ids=[
        "settrace",
        "thread-settrace",
        "sys-monitoring",
        "nothing-registered",
        "no-monitoring-module",
        "get-tool-raises",
    ],
)
def test_every_way_an_interpreter_can_be_traced(
    monkeypatch: pytest.MonkeyPatch,
    gettrace: object,
    threadtrace: object,
    monitoring: object,
    expected: bool,
) -> None:
    """All six branches, on any interpreter, which is the whole point.

    *Added at P11 after CI run 33711091899.* `interpreter_is_traced` asks two
    questions because CPython has two mechanisms, and **which one is live depends
    on the interpreter version**: `coverage` uses `sys.settrace` on 3.12, which
    the CI image ships, and `sys.monitoring` on 3.14, which this machine runs. So
    the function answers on the first question in CI and reaches the second one
    here, and `python/npu_frontend` measured 0.26 points higher locally than in
    CI for that reason alone.

    Measured rather than deduced: forcing `COVERAGE_CORE=ctrace` on 3.14 moves
    `pass_stats.py` from 16 missing lines to 20, and the four are exactly the
    `sys.monitoring` block.

    Substituting both mechanisms covers every branch wherever this runs, so the
    tree stops being interpreter dependent. That is better than lowering the
    coverage gate, which would have made it mean less in both environments
    rather than the same thing in both.
    """
    monkeypatch.setattr(pass_stats.sys, "gettrace", lambda: gettrace)
    monkeypatch.setattr(pass_stats.threading, "gettrace", lambda: threadtrace)
    if monitoring is None:
        monkeypatch.delattr(pass_stats.sys, "monitoring", raising=False)
    else:
        monkeypatch.setattr(pass_stats.sys, "monitoring", monitoring, raising=False)

    assert pass_stats.interpreter_is_traced() is expected


def test_the_gap_bound_does_not_run_under_a_tracer_and_says_so(
    instrumented: Instrumented, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per pass gap bound has a precondition, and a tracer breaks it.

    *Added at P11, when `experiments/` joined the coverage measurement.* That
    bound's premise, stated in its own message, is that the gap **is** this
    instrumentation's operation walk: the one thing inside MLIR's window and
    outside this project's. Under a tracer everything else in that window is
    stretched too, so the gap becomes the walk plus whatever the tracer did.

    Measured 2026-09-03, the same cells three times each: untraced worst gaps
    0.0658, 0.0729 and 0.0690 ms; under `coverage` 0.2757, 0.0844 and 0.0769.
    The tracer does not shift the gap, it produces occasional outliers, which
    makes the check flaky rather than wrong, and a flaky check is worth less
    than one that says it did not run.

    **A skipped bound is recorded, never silent.** Section 19.0's rule is that a
    check which did not run says so, and `run_benchmarks.py` prints the reason at
    the end of a run. This asserts both directions.
    """
    untraced = cross_check_against_mlir_timing(
        instrumented.records, instrumented.timing
    )
    if not pass_stats.interpreter_is_traced():
        assert untraced.upper_bound_skipped == "", (
            "nothing is tracing this interpreter and the bound reported itself "
            "as skipped, which would switch the check off in the ordinary run"
        )

    monkeypatch.setattr(pass_stats, "interpreter_is_traced", lambda: True)
    traced = cross_check_against_mlir_timing(instrumented.records, instrumented.timing)
    assert traced.upper_bound_skipped
    assert "traced" in traced.upper_bound_skipped
    assert "deficit bound and the totals were still checked" in (
        traced.upper_bound_skipped
    )
    # The rows survive, so a run can still report its worst gap. What is skipped
    # is the assertion about the gap, never the measurement of it.
    assert traced.rows == untraced.rows


def test_the_two_clocks_disagreeing_about_the_passes_raises(
    instrumented: Instrumented,
) -> None:
    """A comparison of two clocks that timed different things is meaningless."""
    truncated = "\n".join(instrumented.timing.splitlines()[:8])
    with pytest.raises(PassStatisticsError) as failure:
        cross_check_against_mlir_timing(instrumented.records, truncated)
    assert "disagree about which passes ran" in str(failure.value)


def test_the_timing_text_is_the_tree_display_and_is_parsed_as_one(
    instrumented: Instrumented,
) -> None:
    """The parser, against the thing it parses, so a format change fails loudly.

    The list display would satisfy a laxer test: it holds the same names and the
    same numbers, in a different order, with a `Pipeline Collection` row that is
    not a pass. What the tree display gives and the list does not is the
    pipeline's own order, which is what `position` is compared against.
    """
    report = parse_mlir_timing(instrumented.timing)
    assert report.rows, "no timing rows were parsed, so the cross check has no data"
    assert [name for name, _ in report.rows] == [
        record.pass_name for record in instrumented.records
    ]
    assert "Execution time report" in instrumented.timing
    assert all(milliseconds >= 0.0 for _, milliseconds in report.rows)

    # **The precision is read off the report rather than assumed**, D-0043, and
    # this is the assertion that would notice MLIR changing its format. Four
    # decimals of seconds is a quantum of 0.1 ms, so every parsed value is a
    # multiple of it and every bound derived from it moves if the format does.
    assert report.decimals == MLIR_TIMING_DECIMALS
    assert report.half_ulp_ms == pytest.approx(0.05)
    for _, milliseconds in report.rows:
        remainder = milliseconds % (2 * report.half_ulp_ms)
        assert remainder == pytest.approx(0.0) or remainder == pytest.approx(
            2 * report.half_ulp_ms
        ), f"{milliseconds} is not a multiple of the quantum this report prints"


def test_the_parser_ignores_what_is_not_a_pass() -> None:
    """The rows a tree display carries beside the passes, in one fixture.

    Written out rather than taken from a run, because the point is the rows a
    real run happens not to have on a small module: an analysis nested under a
    pass, and a pipeline group header.
    """
    text = "\n".join(
        [
            "  ----Wall Time----  ----Name----",
            "    0.0019 ( 25.5%)  Parser",
            "    0.0016 ( 21.5%)  'func.func' Pipeline",
            "    0.0001 (  1.9%)    NPUConstantFold",
            "    0.0000 (  0.0%)      (A) DominanceInfo",
            "    0.0010 ( 13.7%)  Output",
            "    0.0003 (  4.0%)  Rest",
            "    0.0074 (100.0%)  Total",
        ]
    )
    report = parse_mlir_timing(text)
    assert report.rows == [("NPUConstantFold", 0.1)]
    assert report.decimals == 4
    assert report.half_ulp_ms == pytest.approx(0.05)


def test_the_print_precision_decides_the_bound_rather_than_a_constant() -> None:
    """D-0043: the quantum comes from the text, so a format change moves it.

    A report printed to two decimals of seconds has a quantum of 10 ms and a
    half unit of 5 ms, and every bound in the cross check has to widen by a
    hundred fold with it. A constant written into this module would have stayed
    at 0.05 and asserted a precision the figures no longer had, which is the
    same fault as the one it replaced rather than a smaller version of it.
    """
    coarse = "\n".join(
        [
            "  ----Wall Time----  ----Name----",
            "    0.01 (  1.0%)  NPUConstantFold",
            "    0.02 (  2.0%)  CSE",
        ]
    )
    report = parse_mlir_timing(coarse)
    assert report.decimals == 2
    assert report.half_ulp_ms == pytest.approx(5.0)
    assert [milliseconds for _, milliseconds in report.rows] == [10.0, 20.0]


# ---------------------------------------------------------------------------
# The ablation, proven by measurement rather than asserted.
# ---------------------------------------------------------------------------


def test_an_ablation_removes_exactly_the_pass_it_names(
    instrumented: Instrumented,
) -> None:
    """The removal is read back out of the instrumentation.

    This is why `PipelineOptions::ablatedPass` does not have to refuse an
    impossible request in C++, where MLIR's pipeline registration gives it no way
    to. An ablation that silently did nothing produces a recorded pass list that
    still holds the pass, and the loader raises with the pass named.

    `-canonicalize` is one pass with two positions at `-O2`, so its row removes
    both and the recorded list is two shorter. That is asserted here rather than
    left as a comment in `Pipeline.cpp`.

    The set is swept rather than listed, and it is read from the compiler at run
    time, so a pass added to `-O2` and marked ablatable is swept the day it lands.
    """
    whole = expected_passes(2)
    ablatable = ablatable_passes(2)
    assert ablatable, "an empty ablatable set would make this test vacuous"

    for ablated in ablatable:
        stats = instrumented.directory / f"ablate-{ablated}.json"
        compile_model(
            instrumented.onnx,
            level=2,
            emit="nbin",
            pass_stats_json=stats,
            ablate=ablated,
        )
        records = load_pass_stats(stats, expected=expected_passes(2, ablated=ablated))
        names = [record.name for record in records]
        assert ablated not in names
        assert len(records) == len(whole) - whole.count(ablated)

    assert whole.count("canonicalize") == 2, (
        "-canonicalize has one position at -O2, so the two that Section 12's "
        "table marks as deliberate are no longer both there"
    )


def test_a_pass_that_is_not_ablatable_is_refused_by_the_driver(
    instrumented: Instrumented,
) -> None:
    """Section 16.2 names both, and the refusal quotes the reason rather than the rule."""
    with pytest.raises(CompileError) as failure:
        compile_model(instrumented.onnx, level=2, ablate="npu-allocate-scratchpad")
    assert "not an ablatable pass" in str(failure.value)
    assert "produces no program at all" in str(failure.value)


def test_the_ablatable_set_is_read_from_the_compiler_and_is_not_written_here() -> None:
    """Section 16.2's rule, asserted about this project rather than assumed.

    The list below is the one the compiler reports today. It is compared rather
    than used: a pass added to `-O2` and marked ablatable makes this red, which
    is a prompt to add its ablation row and its `docs/PASSES.md` entry, and that
    is the failure the rule exists to produce.
    """
    assert ablatable_passes(2) == [
        "npu-constant-fold",
        "canonicalize",
        "npu-fuse-bias",
        "npu-fold-batchnorm",
        "npu-fuse-ops",
        "cse",
        "sccp",
        "symbol-dce",
    ]
    assert ablatable_passes(0) == [], (
        "-O0's two passes are both marked not ablatable, because removing "
        "either produces no program at all"
    )
