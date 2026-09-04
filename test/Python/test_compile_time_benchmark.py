# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The allocator growth benchmark's arithmetic, and the gate it carries.

Section 13.1 commits this benchmark and the P12 gate asks it for a **fitted**
growth exponent that is consistent with O(n log n). Two different things are
being held here.

The first is the fit itself: a least squares line through log time against log
size, with residuals and an r squared. It is checked against curves whose answer
is known before the code runs, which is the only way to test a fit at all, and
against the degenerate inputs a wall clock produces on a fast machine.

The second is the check's discrimination. The ceiling is derived from the sizes
rather than chosen, so `test_the_ceiling_separates_the_two_hypotheses_13_1_names`
asserts what it is for: a synthetic O(n log n) curve passes and a synthetic
quadratic one fails. A tolerance nobody can fail is not a gate, and this is the
test that would notice if the ceiling ever drifted up to where the naive nested
formulation fits underneath it.

**Nothing here runs `npu-opt`.** The measurement half needs a build and a quiet
machine and is a wall clock on one host; the arithmetic half is a pure function
and is what a regression would break silently. Only the second is pytest's.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import compile_time_benchmark as benchmark  # noqa: E402

SIZES = (500.0, 1000.0, 2000.0, 5000.0)


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exponent", [0.5, 1.0, 1.1365, 1.5, 2.0, 3.0])
def test_a_pure_power_law_is_recovered_exactly(exponent: float) -> None:
    """The fit's own definition, checked against curves it must reproduce.

    A least squares line through points that already lie on a line returns that
    line, so `t = 3 * n ** a` has to come back with a slope of `a` whatever `a`
    is. Six exponents rather than one, spanning both sides of the ceiling, so
    that a sign error or a swapped axis cannot survive by being right at 1.0.
    """
    times = [3.0 * size**exponent for size in SIZES]
    fit = benchmark.fit_power_law(SIZES, times)
    assert fit.exponent == pytest.approx(exponent, abs=1e-9)
    assert fit.intercept == pytest.approx(math.log(3.0), abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    assert fit.worst_residual == pytest.approx(0.0, abs=1e-9)


def test_the_intercept_is_a_scale_and_the_slope_ignores_it() -> None:
    """Doubling every time moves the intercept and leaves the exponent alone.

    It is the property that makes the exponent comparable between two machines
    even though no time on one is comparable with a time on the other, and it is
    the reason the report leads with the exponent rather than with a duration.
    """
    times = [size**1.2 for size in SIZES]
    slow = benchmark.fit_power_law(SIZES, [10.0 * value for value in times])
    fast = benchmark.fit_power_law(SIZES, times)
    assert slow.exponent == pytest.approx(fast.exponent, abs=1e-12)
    assert slow.intercept - fast.intercept == pytest.approx(math.log(10.0), abs=1e-12)


def test_the_residuals_show_a_bend_that_r_squared_almost_hides() -> None:
    """A curve that is not a power law is reported as one, with its shape visible.

    `n log n` is not `n ** a` for any `a`, so the fit through it must leave
    residuals with a sign pattern rather than noise. The pattern is the one a
    **concave** curve leaves, low at both ends and high in the middle, because
    `log(n log n)` is `log n + log log n` and the second term is concave in
    `log n`. The r squared stays above 0.999 all the same, which is exactly why
    the report prints the residuals beside it: an r squared that good would let
    a reader believe the curve is a power law, and it is not one.

    The direction is asserted rather than only its magnitude, because getting it
    backwards is how a reader would misread the sign pattern of a real
    measurement. The committed P12 curve leans the other way, convex, which is
    the pass' genuine superlinear term and not this shape.
    """
    times = [size * math.log(size) for size in SIZES]
    fit = benchmark.fit_power_law(SIZES, times)

    assert fit.r_squared > 0.999
    assert fit.worst_residual > 0.0
    assert fit.residuals[0] < 0.0
    assert fit.residuals[1] > 0.0
    assert fit.residuals[2] > 0.0
    assert fit.residuals[-1] < 0.0


def test_the_fit_refuses_what_a_log_cannot_take() -> None:
    """Zero and negative inputs are refused by name rather than raising ValueError
    from inside `math.log`.

    A pass time of zero is what a coarse timer reports for a fast pass, so it is
    a real input rather than a hypothetical one, and the refusal says what to do
    about it. Section 16.1's rule about values arriving through channels that
    lose information is the same rule: a zero here is the timer's resolution and
    not a measurement.
    """
    with pytest.raises(ValueError, match="strictly positive"):
        benchmark.fit_power_law(SIZES, [0.1, 0.0, 0.3, 0.4])
    with pytest.raises(ValueError, match="strictly positive"):
        benchmark.fit_power_law((0.0, 1.0, 2.0, 3.0), [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="at least two sizes"):
        benchmark.fit_power_law((500.0,), [0.1])
    with pytest.raises(ValueError, match="one time per size"):
        benchmark.fit_power_law(SIZES, [0.1, 0.2])
    with pytest.raises(ValueError, match="no curve to fit"):
        benchmark.fit_power_law((500.0, 500.0), [0.1, 0.2])


# ---------------------------------------------------------------------------
# The references and the ceiling.
# ---------------------------------------------------------------------------


def test_the_references_are_computed_at_the_sizes_measured() -> None:
    """n is exactly 1, n squared is exactly 2, and n log n is neither.

    The first two are the fit reproducing a power law it was handed. The third
    is the one that matters: `n log n`'s effective exponent is a property of the
    range, it is 1.1365 over 500 to 5000, and it is **not** 1. A check written
    against a rounded constant from another decade is a check written against
    nothing, which is why this is derived.
    """
    references = benchmark.reference_exponents(SIZES)
    assert references["n"] == pytest.approx(1.0, abs=1e-12)
    assert references["n squared"] == pytest.approx(2.0, abs=1e-12)
    assert references["n log n"] == pytest.approx(1.1365, abs=5e-4)
    assert references["n"] < references["n log n"] < references["n squared"]


def test_the_n_log_n_reference_does_not_depend_on_the_log_base() -> None:
    """Changing the base multiplies every value by a constant and moves no slope.

    Stated in the module and asserted here, because "which logarithm" is exactly
    the question a reader asks of an `n log n` reference and the answer is that
    it does not matter.
    """
    natural = benchmark.fit_power_law(
        SIZES, [size * math.log(size) for size in SIZES]
    ).exponent
    binary = benchmark.fit_power_law(
        SIZES, [size * math.log2(size) for size in SIZES]
    ).exponent
    assert natural == pytest.approx(binary, abs=1e-12)


def test_the_n_log_n_reference_falls_as_the_range_moves_out() -> None:
    """1.1365 is a fact about 500 to 5000 and a decade higher it is smaller.

    This is the whole argument for deriving the reference rather than pinning
    it, and it is asserted so that the argument is checkable rather than only
    written down.
    """
    here = benchmark.reference_exponents(SIZES)["n log n"]
    decade_up = benchmark.reference_exponents(
        tuple(size * 10 for size in SIZES),
    )["n log n"]
    assert decade_up < here


def test_the_ceiling_separates_the_two_hypotheses_13_1_names() -> None:
    """The sweep line passes and the naive nested formulation does not.

    Section 13.1 names both: the sweep line is O(n log n) for the sort and O(n)
    for the walk, and the naive formulation is O(instructions times buffers)
    recomputed inside the spill loop. The ceiling is the midpoint between their
    exponents, so a gate that cannot fail and a gate that fails on noise are
    both excluded, and this test is what says so.
    """
    ceiling = benchmark.consistency_ceiling(SIZES)
    references = benchmark.reference_exponents(SIZES)

    assert references["n log n"] < ceiling < references["n squared"]
    assert ceiling == pytest.approx(1.5683, abs=5e-4)

    linear = benchmark.fit_power_law(SIZES, [float(size) for size in SIZES])
    log_linear = benchmark.fit_power_law(
        SIZES, [size * math.log(size) for size in SIZES]
    )
    quadratic = benchmark.fit_power_law(SIZES, [float(size) ** 2 for size in SIZES])

    assert linear.exponent < ceiling
    assert log_linear.exponent < ceiling
    assert quadratic.exponent >= ceiling


def test_the_ceiling_leaves_room_for_wall_clock_noise() -> None:
    """An n log n curve with a tenth of a decade of scatter still passes.

    The measurement is a wall clock on a shared machine and the fit is over four
    points, so the question a ceiling has to answer is not only "does it exclude
    quadratic" but "does it survive a noisy afternoon". A multiplicative jitter
    of 1.1 on alternate points is larger than anything the committed runs have
    shown and it does not reach the ceiling.
    """
    jittered = [
        size * math.log(size) * (1.1 if index % 2 else 0.9)
        for index, size in enumerate(SIZES)
    ]
    fit = benchmark.fit_power_law(SIZES, jittered)
    assert fit.exponent < benchmark.consistency_ceiling(SIZES)


# ---------------------------------------------------------------------------
# The synthetic module, and the axis it is generated on.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buffers", [2, 500, 1000, 2000, 5000])
def test_a_request_in_buffers_produces_exactly_that_many_buffers(
    buffers: int,
) -> None:
    """The gate's unit, honoured by construction rather than by rounding.

    Section 13.1 and the P12 gate both say buffers. At P5 this script read its
    sizes as operations, and the chain allocates one buffer per two operations,
    so the committed P5 curve was measured at 249, 499, 999 and 2499 buffers
    while its table's first column said 500, 1000, 2000 and 5000. The arithmetic
    is now checked against the text the generator produced.
    """
    text = benchmark.synthetic_module(benchmark.operations_for_buffers(buffers))
    assert text.count("memref.alloc()") == buffers


def test_the_p5_axis_still_produces_the_p5_counts() -> None:
    """`--size-unit operations` reproduces the committed P5 table's two columns.

    The unit changed and the P5 record did not, so the old axis stays reachable
    and its buffer counts stay what the P5 entry in the engineering log says
    they were. A correction that made an earlier measurement irreproducible
    would be a second fault rather than a fix for the first.
    """
    for operations, buffers in ((500, 249), (1000, 499), (2000, 999), (5000, 2499)):
        text = benchmark.synthetic_module(operations)
        assert text.count("memref.alloc()") == buffers


def test_the_two_axes_disagree_by_the_factor_that_caused_the_correction() -> None:
    """Roughly two operations per buffer, which is why the P5 axis read low.

    Asserted rather than described, because it is the size of the error the P12
    correction fixes and a reader is entitled to see it as a number.
    """
    assert benchmark.operations_for_buffers(5000) == 10002
    assert benchmark.operations_for_buffers(500) == 1002


def test_a_buffer_count_below_the_chain_overhead_is_refused() -> None:
    """One buffer is fewer than the chain's fixed allocation, and it says so."""
    with pytest.raises(ValueError, match="fixed overhead"):
        benchmark.operations_for_buffers(1)
    with pytest.raises(ValueError, match="smaller than the fixed overhead"):
        benchmark.synthetic_module(5)


def test_the_generated_module_is_the_shape_the_benchmark_claims() -> None:
    """A chain: allocate, load, then relu into a fresh buffer, then store.

    The shape is the honest one for a growth curve because every live range is
    short and every buffer is the same size, so nothing here is measuring the
    spill loop. That claim is only true while the generator keeps producing this
    shape.
    """
    text = benchmark.synthetic_module(benchmark.operations_for_buffers(10))
    assert "npuisa.dma_load" in text
    assert "npuisa.dma_store" in text
    assert text.count("npuisa.relu") == 9
    assert text.count("npu.scratchpad") > 0
    assert text.rstrip().endswith("}")


def test_the_axis_follows_the_unit_it_is_asked_for() -> None:
    """The fit runs over buffers or over operations, and the two differ.

    `axis_of` is what decides which column the exponent is an exponent **of**,
    and a curve fitted against the wrong column is the P5 fault repeated in a
    different place.
    """
    results = [
        benchmark.Measurement(
            size=size,
            operations=benchmark.operations_for_buffers(size),
            buffers=size,
            pass_seconds=0.001 * size,
            total_seconds=0.01 * size,
        )
        for size in (500, 1000)
    ]
    assert benchmark.axis_of(results, "buffers") == [500.0, 1000.0]
    assert benchmark.axis_of(results, "operations") == [1002.0, 2002.0]


def test_an_unknown_size_unit_is_refused_by_name() -> None:
    """A typo must not silently select the axis nobody asked for.

    Section 13.1 states the rule for the allocator's own pass options and it is
    the same rule here: the refusal names the offending string and lists the
    values that exist.
    """
    with pytest.raises(ValueError, match="is not a size unit"):
        benchmark.run(Path("/nonexistent"), (500,), 1, "megabytes")


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


def _measurements(times: list[float]) -> list[benchmark.Measurement]:
    sizes = [500, 1000, 2000, 5000][: len(times)]
    return [
        benchmark.Measurement(
            size=size,
            operations=benchmark.operations_for_buffers(size),
            buffers=size,
            pass_seconds=seconds,
            total_seconds=seconds * 3.0,
        )
        for size, seconds in zip(sizes, times, strict=True)
    ]


def test_the_report_prints_the_fit_and_returns_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gate's phrase appears in the output, with the references beside it.

    The P12 gate asks for the fitted growth exponent to be **reported**, so the
    words are part of the deliverable and not decoration.
    """
    results = _measurements([0.0047, 0.0094, 0.0202, 0.0593])
    fit = benchmark.report(results, repeats=5, size_unit="buffers")

    assert fit is not None
    assert 1.0 < fit.exponent < 1.2
    printed = capsys.readouterr().out
    assert "Fitted growth exponent" in printed
    assert "r squared" in printed
    assert "n log n" in printed
    assert "Sizes are buffers" in printed
    assert "residual" in printed


def test_the_report_names_the_other_axis_when_it_is_used(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On the P5 axis the header says so, rather than claiming 13.1's unit."""
    benchmark.report(
        _measurements([0.0023, 0.0046, 0.0095, 0.0264]),
        repeats=5,
        size_unit="operations",
    )
    printed = capsys.readouterr().out
    assert "Sizes are operations" in printed
    assert "the axis P5 measured" in printed


def test_the_report_refuses_to_fit_one_point(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One size is a point, not a curve, and the report says that rather than
    printing an exponent it made up."""
    assert benchmark.report(_measurements([0.0047]), repeats=5) is None
    assert "at least two sizes" in capsys.readouterr().out


def test_the_report_refuses_to_fit_a_zero_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pass too fast for the timer is a missing measurement, not a fast one.

    `--mlir-timing` prints four decimals of seconds, so a pass under 50
    microseconds reports 0.0000, and D-0043 is this project's standing lesson
    about treating a value that arrived through a lossy channel as exact.
    """
    assert benchmark.report(_measurements([0.0, 0.0094]), repeats=5) is None
    assert "nonzero pass time" in capsys.readouterr().out
