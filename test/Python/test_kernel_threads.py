# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The thread scaling harness, and the one thing in it that is a gate.

`experiments/kernel_threads.py` produces two kinds of number and only one of
them is a claim about this project. The output bytes must be identical at every
thread count, on every model, because Section 10.3 parallelises over batch and
output channel and leaves the reductions sequential; that holds on any host. The
times are a measurement of one host and nothing gates on them.

So this file tests the byte comparison, the refusal it produces, and the parts
that decide what gets measured. It does **not** run the seven models: that needs
a build, several seconds, and a quiet machine, and a wall clock inside a pytest
run beside a hundred other tests is not a measurement of anything. The
measurement is a committed script the phase runs and records, in the shape
Section 13.1's compile time benchmark already established.

**`kernel_info` is tested against a substituted `npu-sim` rather than the real
one.** It is D-0047's out of process answer, and the fault D-0047 was is a build
answering a question about itself, so a test that read this build's real answer
would pass on any machine and prove nothing about the parsing. Both branches are
driven here, plus the refusal, which is what `test_external_tools.py` does for
the same reason at P11.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import kernel_threads  # noqa: E402


def _timing(
    model: str,
    seconds: dict[int, float],
    digest: dict[int, str],
) -> kernel_threads.ModelTiming:
    return kernel_threads.ModelTiming(model=model, seconds=seconds, digest=digest)


# ---------------------------------------------------------------------------
# The gate: the bytes.
# ---------------------------------------------------------------------------


def test_equal_bytes_at_every_thread_count_is_the_passing_case() -> None:
    """One digest across the whole row, which is what Section 10.3 requires."""
    timing = _timing("lenet", {1: 0.02, 28: 0.007}, {1: "abc", 28: "abc"})
    assert timing.outputs_agree()
    assert kernel_threads.verdict([timing]) == []


def test_one_moved_byte_names_the_model_it_moved_on() -> None:
    """A difference is a defect in the kernel and the report has to say where.

    Section 10.3's reason is that the accumulation order inside each output
    element determines its last bits, so a row that moved means a reduction went
    into the parallel region. Naming the model is what turns that into something
    somebody can reproduce.
    """
    good = _timing("lenet", {1: 0.02, 28: 0.007}, {1: "abc", 28: "abc"})
    bad = _timing("resnet_block", {1: 0.02, 28: 0.007}, {1: "abc", 28: "def"})
    assert not bad.outputs_agree()
    assert kernel_threads.verdict([good, bad]) == ["resnet_block"]


def test_a_single_thread_count_cannot_disagree_with_itself() -> None:
    """One column agrees trivially, and that is worth pinning rather than
    leaving as an accident of `len(set(...)) <= 1`.

    A run at one thread count is a legitimate thing to ask for and it must not
    be reported as a failure; it also proves nothing, which is why the default
    thread list has five entries.
    """
    assert _timing("lenet", {1: 0.02}, {1: "abc"}).outputs_agree()
    assert _timing("lenet", {}, {}).outputs_agree()


# ---------------------------------------------------------------------------
# Speedup, which is reported and never gated.
# ---------------------------------------------------------------------------


def test_the_speedup_is_one_thread_over_the_count_asked_for() -> None:
    timing = _timing("lenet", {1: 0.0233, 8: 0.0072, 28: 0.0074}, {})
    assert timing.speedup(8) == pytest.approx(0.0233 / 0.0072)
    assert timing.speedup(28) == pytest.approx(0.0233 / 0.0074)


def test_a_speedup_below_one_is_reported_rather_than_clamped() -> None:
    """A parallel run slower than a serial one is a result, not an error.

    P12 measured exactly that on `depthwise_separable`, whose whole simulation
    is a fraction of a millisecond of arithmetic inside a process that takes
    longer than that to start. Clamping it to 1.0 would hide the only row that
    says where the parallelism stops paying for itself.
    """
    timing = _timing("depthwise_separable", {1: 0.0016, 28: 0.0019}, {})
    speedup = timing.speedup(28)
    assert speedup is not None
    assert speedup < 1.0


def test_a_missing_or_zero_measurement_gives_no_speedup_rather_than_a_number() -> None:
    """A division nobody can do returns None instead of infinity or a zero.

    D-0043's standing lesson: a value that did not arrive is not a value, and
    the code around a schema kept not having the discipline the schema had.
    """
    assert _timing("lenet", {28: 0.007}, {}).speedup(28) is None
    assert _timing("lenet", {1: 0.02}, {}).speedup(28) is None
    assert _timing("lenet", {1: 0.02, 28: 0.0}, {}).speedup(28) is None


# ---------------------------------------------------------------------------
# What gets measured.
# ---------------------------------------------------------------------------


def test_zero_means_the_host_maximum_and_the_rest_mean_themselves() -> None:
    """The default list ends in 0 so that the top of the curve is the host's.

    Writing 28 into the defaults would make this file a measurement of one
    machine, and a four processor CI runner would then spend its time measuring
    oversubscription.
    """
    maximum = kernel_threads.host_thread_maximum()
    assert maximum >= 1
    assert kernel_threads.resolve_threads([1, 0]) == sorted({1, maximum})
    assert kernel_threads.resolve_threads([4, 2, 1]) == [1, 2, 4]


def test_a_small_host_collapses_the_default_list_without_duplicating_it() -> None:
    """On a host with two processors the default asks for 1, 2, 4, 8 and 2.

    The duplicate is dropped rather than measured twice, and the result stays
    sorted, so the report's last column is the largest count on any host.
    """
    resolved = kernel_threads.resolve_threads([1, 2, 4, 8, 2])
    assert resolved == [1, 2, 4, 8]
    assert resolved == sorted(set(resolved))


def test_a_thread_count_below_one_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="not one this script can ask for"):
        kernel_threads.resolve_threads([1, -4])


def test_the_default_thread_list_spans_the_range_and_ends_at_the_host() -> None:
    """A pair of numbers cannot tell saturation from no parallelism at all.

    P12 is the reason this is asserted rather than assumed: the run that found
    D-0047 looked exactly like a run on models too small to parallelise, and the
    intermediate columns are what separated the two.
    """
    assert kernel_threads.DEFAULT_THREADS[0] == 1
    assert kernel_threads.DEFAULT_THREADS[-1] == 0
    assert len(kernel_threads.DEFAULT_THREADS) >= 4


# ---------------------------------------------------------------------------
# kernel_info, which is D-0047 asked from outside the process.
# ---------------------------------------------------------------------------


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("kernel openmp: yes\nkernel threads: 28\n", (True, 28)),
        ("kernel openmp: no\nkernel threads: 1\n", (False, 1)),
    ],
)
def test_both_answers_the_tool_can_give_are_read(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected: tuple[bool, int],
) -> None:
    """Both branches run on any host, which is the point of substituting the tool.

    Half of them cannot execute in either environment otherwise: this machine's
    build has OpenMP and would only ever exercise the first. That is the same
    argument `test_external_tools.py` makes for `find_spec` and `which`, and it
    is why the P11 coverage round put those branches under test rather than
    behind a pragma.
    """
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, stdout=answer))
    assert kernel_threads.kernel_info(Path("npu-sim")) == expected


def test_a_build_without_the_flag_is_refused_rather_than_read_as_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero exit is not the answer "no".

    D-0042 is the standing example and it was expensive: a git fatal was read as
    a "no" and six helpers returned six wrong answers. A tool that refuses
    `--kernel-info` has not told this script that the kernels are serial, it has
    told it nothing, and defaulting to False here would reproduce exactly the
    fault D-0047 already is.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _Completed(1, stderr="npu-sim: Unknown command line argument"),
    )
    with pytest.raises(RuntimeError, match="predates the flag"):
        kernel_threads.kernel_info(Path("npu-sim"))


def test_an_answer_missing_a_line_falls_back_to_the_serial_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit zero and a line this script does not recognise is the cautious case.

    The tool answered, so this is not D-0042's shape; what is missing is one of
    the two fields, and the safe reading of a missing OpenMP line is that there
    is no OpenMP. It is reported and never gated on, so a wrong reading here
    costs a sentence in a table and not a verdict.
    """
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(0, stdout="kernel threads: 4\n")
    )
    assert kernel_threads.kernel_info(Path("npu-sim")) == (False, 4)


# ---------------------------------------------------------------------------
# The report and the JSON.
# ---------------------------------------------------------------------------


def test_the_report_says_which_column_is_the_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timings = [
        _timing("lenet", {1: 0.0233, 28: 0.0074}, {1: "a", 28: "a"}),
        _timing("resnet_block", {1: 0.0045, 28: 0.0017}, {1: "b", 28: "b"}),
    ]
    kernel_threads.report(
        timings, [1, 28], repeats=5, has_openmp=True, kernel_threads=28
    )
    printed = capsys.readouterr().out
    assert "equal" in printed
    assert "bytes column is the gate" in printed
    assert "not comparable" in printed
    assert "OpenMP on" in printed


def test_the_report_marks_a_row_whose_bytes_moved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    timings = [_timing("lenet", {1: 0.02, 28: 0.007}, {1: "a", 28: "z"})]
    kernel_threads.report(
        timings, [1, 28], repeats=5, has_openmp=True, kernel_threads=28
    )
    assert "DIFFER" in capsys.readouterr().out


def test_the_report_says_when_the_kernels_have_no_openmp(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A build without OpenMP is supported and the table says so rather than
    looking like a table of models too small to parallelise.

    That sentence is D-0047 in one line: for five phases the two were
    indistinguishable from the outside.
    """
    timings = [_timing("lenet", {1: 0.02}, {1: "a"})]
    kernel_threads.report(timings, [1], repeats=5, has_openmp=False, kernel_threads=1)
    assert "OpenMP off" in capsys.readouterr().out


def test_the_json_carries_the_host_beside_the_times() -> None:
    """A wall clock separated from its machine is the figure that ends up in a
    table next to one from a different machine.

    P11's rule for recorded numbers, applied to a file whose whole content is
    wall clocks.
    """
    timings = [_timing("lenet", {1: 0.0233, 28: 0.0074}, {1: "a", 28: "a"})]
    payload = kernel_threads.as_json(
        timings, [1, 28], repeats=5, has_openmp=True, kernel_threads=28
    )

    assert payload["kernel_openmp"] is True
    assert payload["kernel_default_threads"] == 28
    assert payload["host_process_cpu_count"] >= 1
    assert payload["input_seed"] == kernel_threads.INPUT_SEED
    assert payload["threads_measured"] == [1, 28]

    models = payload["models"]
    assert isinstance(models, list)
    row = models[0]
    assert row["model"] == "lenet"
    assert row["outputs_agree"] is True
    assert row["speedup_at_max"] == pytest.approx(0.0233 / 0.0074)
    assert row["seconds"] == {"1": 0.0233, "28": 0.0074}
    assert row["output_sha256"] == {"1": "a", "28": "a"}
