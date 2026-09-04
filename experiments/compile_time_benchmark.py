# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The allocator's compile time growth curve, at four sizes.

Section 13.1 asks for this by name and gives the reason. The sweep line is
O(n log n) for the sort and O(n) for the walk; the naive nested formulation is
O(instructions times buffers) and gets recomputed inside the spill loop, which
makes the whole pass cubic in the worst case. A single measurement at one size
cannot tell those apart, so the benchmark runs at four sizes and prints the
curve.

**The four sizes are buffer counts and that is P12's correction.** Section 13.1
and the P12 gate both say "500, 1000, 2000, and 5000 buffers". At P5 this script
read `--sizes` as operation counts, and the chain below allocates one buffer per
two operations, so the committed P5 curve was measured at 249, 499, 999 and 2499
buffers. The numbers were right about the shape and wrong about the axis.
`--size-unit operations` reproduces the P5 table exactly; the default is the
gate's unit.

**What is measured, and what is not.** The headline number is the pass's own
wall time, read out of MLIR's `--mlir-timing` report, so parsing the module and
printing it back are excluded. Both are O(n) with a large constant and at these
sizes they dominate the total, which would hide exactly the growth this exists
to show. The total wall time is printed beside it so the difference is visible
rather than assumed.

**The reported exponent is fitted, over every point, and P12's gate asks for it
that way.** A ratio between two adjacent rows is one number from two
measurements and inherits both of their noise; the fit is a least squares line
through log time against log size over the whole curve, and it comes with the
residual at each point so a reader can see whether the curve is a power law at
all rather than being told that it is. The consecutive ratios are still printed,
because a fit that hides a bend is worse than two numbers that show one.

**"Consistent with O(n log n)" is a comparison and not an opinion.** The three
reference exponents printed beside the fit are what the same least squares would
return for n, for n log n, and for n squared **at these exact sizes**, so they
carry the range's own arithmetic rather than a rounded constant from a textbook.
`--check` fails when the fitted exponent reaches the midpoint between the n log n
reference and the quadratic one, which is precisely the discrimination Section
13.1 asks this benchmark to make and carries no hand chosen tolerance. Growing
more slowly than n log n is not a failure: O is an upper bound, and the check is
one sided for that reason.

**These are wall clock measurements on one machine, not a claim about
complexity.** A prediction of what the curve should be is committed as
`experiments/predictions/p5-allocator-compile-time.md`, strictly before the
first number was recorded, per ground rule 15.

Usage:

    python experiments/compile_time_benchmark.py
    python experiments/compile_time_benchmark.py --check
    python experiments/compile_time_benchmark.py --sizes 500 5000 --repeats 7
    python experiments/compile_time_benchmark.py --size-unit operations
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL = REPO_ROOT / "build" / "bin" / "npu-opt"

# The four sizes Section 13.1 names, in the unit Section 13.1 names them in.
# They are a default rather than a constant so that a machine with a different
# amount of patience can ask for fewer, but the committed benchmark is these
# four and the gate quotes them.
DEFAULT_SIZES = (500, 1000, 2000, 5000)

# What a size means. `buffers` is the gate's unit and the default; `operations`
# is what P5 measured and is kept so that the committed P5 table stays
# reproducible from this script rather than only from its own commit.
SIZE_UNITS = ("buffers", "operations")

# The buffer every step of the synthetic chain allocates: 1 x 4 x 4 x 4 f32,
# which is 64 elements and 256 bytes. Small on purpose. The point of the
# benchmark is the number of live ranges, not the number of bytes, and a large
# buffer would only make the default budget bind and turn the measurement into a
# measurement of the spill loop.
BUFFER_TYPE = "memref<1x4x4x4xf32, #npu.scratchpad>"
DRAM_TYPE = "memref<1x4x4x4xf32, #npu.dram>"

# The line `--mlir-timing --mlir-timing-display=list` prints for the pass.
#
# The column count is not fixed and that is worth knowing rather than
# discovering: MLIR prints a user time column beside the wall time column only
# when the pass manager actually ran threads, so a module with one function has
# one column and a module with several has two. The pattern therefore takes any
# number of `time (percent%)` columns and the caller reads the last one, which
# is wall time in both layouts.
_TIMING = re.compile(
    r"^\s*((?:[0-9.]+\s*\(\s*[0-9.]+%\)\s+)+)NPUAllocateScratchpad\s*$",
    re.MULTILINE,
)
_SECONDS = re.compile(r"([0-9.]+)\s*\(")


@dataclass(frozen=True)
class Measurement:
    """One size, measured."""

    size: int
    operations: int
    buffers: int
    pass_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class Fit:
    """A least squares line through log time against log size.

    `exponent` is its slope, which is the growth exponent the P12 gate asks to
    have reported. `residuals` are in log space and in the order of the points
    given, so a curve that bends shows as a sign pattern rather than as a
    slightly worse `r_squared`.
    """

    exponent: float
    intercept: float
    r_squared: float
    residuals: tuple[float, ...]

    @property
    def worst_residual(self) -> float:
        """The largest absolute residual, which is the fit's own error bar."""
        return max(abs(value) for value in self.residuals) if self.residuals else 0.0


def operations_for_buffers(buffers: int) -> int:
    """The operation count whose chain allocates exactly `buffers` buffers.

    The chain is one allocation, a load, then one allocation and one relu per
    step, then a store and a return. That is `buffers - 1` steps and
    `2 * buffers + 2` operations, and it is written as arithmetic here rather
    than found by search so that the table's two count columns agree with the
    request by construction.
    """
    if buffers < 2:
        raise ValueError(
            f"a buffer count of {buffers} is smaller than the chain's fixed "
            "overhead of one allocation before the loop"
        )
    return 2 * buffers + 2


def fit_power_law(sizes: Sequence[float], times: Sequence[float]) -> Fit:
    """Ordinary least squares of log(time) on log(size).

    The whole curve at once, which is the difference between this and the
    consecutive ratios beside it in the report. Four points is few, and that is
    exactly why the residuals travel with the slope: a reader who can see them
    can decide for themselves whether the line is a description or a hope.

    `r_squared` is 1.0 when the points already lie on a line, which is the
    degenerate case a two point curve always is, and is reported rather than
    suppressed because a perfect fit through two points is information about how
    many points there were.
    """
    if len(sizes) != len(times):
        raise ValueError("the fit needs one time per size")
    if len(sizes) < 2:
        raise ValueError("a growth exponent needs at least two sizes")
    if any(value <= 0 for value in sizes) or any(value <= 0 for value in times):
        raise ValueError(
            "a log fit needs every size and every time to be strictly positive. "
            "A pass time of zero means the timer's resolution is coarser than "
            "the pass, and the answer to that is a larger size, not a fit."
        )

    xs = [math.log(value) for value in sizes]
    ys = [math.log(value) for value in times]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)

    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0.0:
        raise ValueError("every size is the same, so there is no curve to fit")

    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    residuals = tuple(y - (intercept + slope * x) for x, y in zip(xs, ys, strict=True))

    total = sum((y - mean_y) ** 2 for y in ys)
    residual_sum = sum(value**2 for value in residuals)
    r_squared = 1.0 if total == 0.0 else 1.0 - residual_sum / total

    return Fit(
        exponent=slope,
        intercept=intercept,
        r_squared=r_squared,
        residuals=residuals,
    )


def reference_exponents(sizes: Sequence[float]) -> dict[str, float]:
    """What the same fit returns for n, n log n and n squared at these sizes.

    Derived at the sizes actually measured rather than quoted as a constant,
    because "the exponent of n log n" is not one number: it is 1.14 over 500 to
    5000 and 1.10 over 5000 to 50000, and a check written against a rounded
    figure from the wrong decade is a check written against nothing.

    The base of the logarithm inside `n log n` does not matter and that is worth
    knowing rather than rediscovering: changing it multiplies every value by the
    same constant, which moves the fitted intercept and leaves the slope alone.
    """
    return {
        "n": fit_power_law(sizes, [float(n) for n in sizes]).exponent,
        "n log n": fit_power_law(sizes, [n * math.log(n) for n in sizes]).exponent,
        "n squared": fit_power_law(sizes, [float(n) ** 2 for n in sizes]).exponent,
    }


def consistency_ceiling(sizes: Sequence[float]) -> float:
    """The exponent at which the curve stops being consistent with O(n log n).

    The midpoint between the n log n reference and the quadratic one, at these
    sizes. Section 13.1 states the two hypotheses this benchmark exists to
    separate, the sweep line and the naive nested formulation, and the midpoint
    between their exponents is the line that separates them with the largest
    margin available on either side. It is derived from the sizes rather than
    chosen, so no reader has to be told why a tolerance is the size it is.
    """
    references = reference_exponents(sizes)
    return (references["n log n"] + references["n squared"]) / 2.0


def synthetic_module(size: int) -> str:
    """A straight line function of roughly `size` operations.

    The shape is a chain: allocate a buffer, write it from the previous one,
    repeat. Every live range is short and every buffer is the same size, which
    is the cheapest shape for the allocator and therefore the honest one for a
    growth curve: a shape chosen to be hard would measure the worst case rather
    than the trend.

    The operation count is the number of operations in the block, counting the
    allocations, the instructions, the two transfers and the terminator, because
    that is what the pass walks.
    """
    if size < 6:
        raise ValueError(f"a size of {size} is smaller than the fixed overhead")

    # 4 fixed operations: the first allocation, the load, the store, the return.
    # Every step after that is an allocation plus a relu.
    steps = (size - 4) // 2
    lines = [
        f"func.func @synthetic_{size}(%in: {DRAM_TYPE}, %out: {DRAM_TYPE}) {{",
        f"  %a0 = memref.alloc() : {BUFFER_TYPE}",
        f"  npuisa.dma_load %in, %a0 : {DRAM_TYPE} to {BUFFER_TYPE}",
    ]
    for step in range(1, steps + 1):
        lines.append(f"  %a{step} = memref.alloc() : {BUFFER_TYPE}")
        lines.append(
            f"  npuisa.relu ins(%a{step - 1} : {BUFFER_TYPE}) "
            f"outs(%a{step} : {BUFFER_TYPE})"
        )
    lines.append(f"  npuisa.dma_store %a{steps}, %out : {BUFFER_TYPE} to {DRAM_TYPE}")
    lines.append("  return")
    lines.append("}")
    return "\n".join(lines) + "\n"


def measure(tool: Path, module: Path, repeats: int) -> tuple[float, float]:
    """The pass's own wall time and the total wall time, best of `repeats`.

    The minimum rather than the mean, which is the standard choice for a timing
    loop on a shared machine: the distribution's lower tail is the machine doing
    the work and nothing else, and every sample above it includes something that
    is not the thing being measured.
    """
    pass_times: list[float] = []
    total_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        completed = subprocess.run(
            [
                str(tool),
                str(module),
                "--npu-allocate-scratchpad",
                "--mlir-timing",
                "--mlir-timing-display=list",
                "-o",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        total_times.append(time.perf_counter() - started)

        found = _TIMING.search(completed.stderr)
        if found is None:
            raise RuntimeError(
                "the timing report has no NPUAllocateScratchpad line. The pass "
                "did not run, or --mlir-timing-display=list changed its format:\n"
                + completed.stderr
            )
        columns = _SECONDS.findall(found.group(1))
        pass_times.append(float(columns[-1]))
    return min(pass_times), min(total_times)


def run(
    tool: Path,
    sizes: tuple[int, ...],
    repeats: int,
    size_unit: str = "buffers",
) -> list[Measurement]:
    if size_unit not in SIZE_UNITS:
        raise ValueError(
            f"{size_unit!r} is not a size unit. The units are "
            + ", ".join(SIZE_UNITS)
            + "."
        )
    results: list[Measurement] = []
    with tempfile.TemporaryDirectory() as directory:
        for size in sizes:
            operation_count = (
                operations_for_buffers(size) if size_unit == "buffers" else size
            )
            text = synthetic_module(operation_count)
            module = Path(directory) / f"synthetic_{size}.mlir"
            module.write_text(text, encoding="utf-8")

            # The real counts, from the text rather than from the request, so a
            # rounding in the step arithmetic cannot make the table lie.
            operations = sum(
                1
                for line in text.splitlines()
                if line.startswith("  ") and line.strip()
            )
            buffers = text.count("memref.alloc()")
            pass_seconds, total_seconds = measure(tool, module, repeats)
            results.append(
                Measurement(size, operations, buffers, pass_seconds, total_seconds)
            )
    return results


def axis_of(results: Sequence[Measurement], size_unit: str) -> list[float]:
    """The x axis the fit runs over, taken from the modules rather than the ask.

    The counts come off the generated text, so a size the chain rounded is
    fitted at the count it actually produced. That is the same reason the table
    prints both columns.
    """
    if size_unit == "buffers":
        return [float(result.buffers) for result in results]
    return [float(result.operations) for result in results]


def report(
    results: list[Measurement],
    repeats: int,
    size_unit: str = "buffers",
) -> Fit | None:
    """Prints the curve, the fit and the comparison, and returns the fit."""
    print(f"Allocator compile time, best of {repeats} runs per size.")
    print("Pass time is the NPUAllocateScratchpad line of the --mlir-timing report.")
    print("Total is the whole npu-opt invocation, parse and print included.")
    if size_unit == "buffers":
        print("Sizes are buffers, which is the unit Section 13.1 states them in.")
    else:
        print("Sizes are operations, which is the axis P5 measured. Section 13.1")
        print("states the four sizes in buffers and the buffers column is that one.")
    print()
    header = (
        f"{'size':>6}  {'ops':>6}  {'buffers':>8}  {'pass s':>9}  "
        f"{'total s':>9}  {'step':>7}  {'residual':>9}"
    )

    axis = axis_of(results, size_unit)
    fit: Fit | None = None
    usable = len(results) > 1 and all(result.pass_seconds > 0 for result in results)
    if usable:
        fit = fit_power_law(axis, [result.pass_seconds for result in results])

    print(header)
    print("-" * len(header))
    for index, result in enumerate(results):
        step = ""
        if index > 0:
            previous = results[index - 1]
            if previous.pass_seconds > 0 and axis[index - 1] > 0:
                step = "%.2f" % (
                    math.log(result.pass_seconds / previous.pass_seconds)
                    / math.log(axis[index] / axis[index - 1])
                )
        residual = "" if fit is None else f"{fit.residuals[index]:+.4f}"
        print(
            f"{result.size:>6}  {result.operations:>6}  {result.buffers:>8}  "
            f"{result.pass_seconds:>9.4f}  {result.total_seconds:>9.4f}  "
            f"{step:>7}  {residual:>9}"
        )
    print()
    print(
        "The step column is log(t2 / t1) / log(n2 / n1) between consecutive "
        "rows. The residual column is that row's distance in log space from the "
        "fitted line, which is the fit's own error bar shown per point."
    )

    if fit is None:
        print(
            "\nNo fit: a growth exponent needs at least two sizes and a nonzero "
            "pass time at every one of them."
        )
        return None

    references = reference_exponents(axis)
    ceiling = consistency_ceiling(axis)
    print()
    print(f"Fitted growth exponent: {fit.exponent:.4f}")
    print(f"  r squared:            {fit.r_squared:.4f}")
    print(f"  worst residual:       {fit.worst_residual:+.4f} in log space")
    print(
        "  references at these sizes: "
        + ", ".join(f"{name} {value:.4f}" for name, value in references.items())
    )
    print(f"  O(n log n) is met while the fit stays below {ceiling:.4f}")
    print()
    print(
        "The fit describes the measured curve on this machine and is not a "
        "proof of a complexity class. What it can do is separate the two "
        "hypotheses Section 13.1 names, the sweep line and the naive nested "
        "formulation, and the ceiling above is the midpoint between them."
    )
    return fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tool",
        type=Path,
        default=DEFAULT_TOOL,
        help="the npu-opt binary (default: build/bin/npu-opt)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="sizes to measure, in --size-unit (default: 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--size-unit",
        choices=SIZE_UNITS,
        default="buffers",
        help="what a size counts. 'buffers' is Section 13.1's unit and the "
        "default; 'operations' is what P5 measured (default: buffers)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="runs per size, the fastest of which is reported (default: 5)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero unless the fitted exponent is consistent with "
        "O(n log n) at the sizes measured",
    )
    arguments = parser.parse_args(argv)

    if not arguments.tool.is_file():
        print(
            f"{arguments.tool} does not exist. Build it first: ninja -C build",
            file=sys.stderr,
        )
        return 1

    try:
        results = run(
            arguments.tool,
            tuple(arguments.sizes),
            arguments.repeats,
            arguments.size_unit,
        )
    except ValueError as failure:
        print(str(failure), file=sys.stderr)
        return 1

    fit = report(results, arguments.repeats, arguments.size_unit)

    if not arguments.check:
        return 0

    if fit is None:
        print(
            "compile-time-benchmark: --check has no fit to check. Measure at "
            "two sizes or more, at sizes large enough for the pass to take a "
            "measurable time.",
            file=sys.stderr,
        )
        return 1

    ceiling = consistency_ceiling(axis_of(results, arguments.size_unit))
    if fit.exponent >= ceiling:
        print(
            f"compile-time-benchmark: FAIL. The fitted exponent is "
            f"{fit.exponent:.4f} against a ceiling of {ceiling:.4f}, which is "
            "the midpoint between n log n and n squared at these sizes. "
            "Section 13.1's sweep line is O(n log n) for the sort and O(n) for "
            "the walk, so a curve this steep means the peak pressure "
            "computation has stopped being a sweep line, or that something "
            "nested has been recomputed inside the spill loop. It is not a "
            "number to widen the ceiling for.",
            file=sys.stderr,
        )
        return 1

    print(
        f"compile-time-benchmark: the fitted exponent {fit.exponent:.4f} is "
        f"consistent with O(n log n) at these sizes, exit 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
