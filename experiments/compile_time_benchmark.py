# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The allocator's compile time growth curve, at four sizes.

Section 13.1 asks for this by name and gives the reason. The sweep line is
O(n log n) for the sort and O(n) for the walk; the naive nested formulation is
O(instructions times buffers) and gets recomputed inside the spill loop, which
makes the whole pass cubic in the worst case. A single measurement at one size
cannot tell those apart, so the benchmark runs at 500, 1000, 2000 and 5000
operations and prints the curve.

**What is measured, and what is not.** The headline number is the pass's own
wall time, read out of MLIR's `--mlir-timing` report, so parsing the module and
printing it back are excluded. Both are O(n) with a large constant and at these
sizes they dominate the total, which would hide exactly the growth this exists
to show. The total wall time is printed beside it so the difference is visible
rather than assumed.

**These are wall clock measurements on one machine, not a claim about
complexity.** The implied exponent printed in the last column is
log(t2 / t1) / log(n2 / n1) between consecutive sizes, which is a description of
the measured curve. A prediction of what it should be is committed as
`experiments/predictions/p5-allocator-compile-time.md`, strictly before the
first number was recorded, per ground rule 15.

Usage:

    python experiments/compile_time_benchmark.py
    python experiments/compile_time_benchmark.py --sizes 500 5000 --repeats 7
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
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL = REPO_ROOT / "build" / "bin" / "npu-opt"

# The four sizes Section 13.1 names. They are a default rather than a constant
# so that a machine with a different amount of patience can ask for fewer,
# but the committed benchmark is these four.
DEFAULT_SIZES = (500, 1000, 2000, 5000)

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


def run(tool: Path, sizes: tuple[int, ...], repeats: int) -> list[Measurement]:
    results: list[Measurement] = []
    with tempfile.TemporaryDirectory() as directory:
        for size in sizes:
            text = synthetic_module(size)
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


def report(results: list[Measurement], repeats: int) -> None:
    print(f"Allocator compile time, best of {repeats} runs per size.")
    print("Pass time is the NPUAllocateScratchpad line of the --mlir-timing report.")
    print("Total is the whole npu-opt invocation, parse and print included.")
    print()
    header = (
        f"{'size':>6}  {'ops':>6}  {'buffers':>8}  {'pass s':>9}  "
        f"{'total s':>9}  {'exponent':>9}"
    )
    print(header)
    print("-" * len(header))
    for index, result in enumerate(results):
        exponent = ""
        if index > 0:
            previous = results[index - 1]
            if previous.pass_seconds > 0 and previous.operations > 0:
                exponent = "%.2f" % (
                    math.log(result.pass_seconds / previous.pass_seconds)
                    / math.log(result.operations / previous.operations)
                )
        print(
            f"{result.size:>6}  {result.operations:>6}  {result.buffers:>8}  "
            f"{result.pass_seconds:>9.4f}  {result.total_seconds:>9.4f}  "
            f"{exponent:>9}"
        )
    print()
    print(
        "The exponent is log(t2 / t1) / log(n2 / n1) between consecutive rows. "
        "It describes the measured curve on this machine and is not a proof of "
        "a complexity class."
    )
    if len(results) > 1:
        exponents = [
            math.log(later.pass_seconds / earlier.pass_seconds)
            / math.log(later.operations / earlier.operations)
            for earlier, later in zip(results[:-1], results[1:], strict=True)
            if earlier.pass_seconds > 0
        ]
        if exponents:
            print(f"Mean exponent over the curve: {statistics.fmean(exponents):.2f}")


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
        help="operation counts to measure (default: 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="runs per size, the fastest of which is reported (default: 5)",
    )
    arguments = parser.parse_args(argv)

    if not arguments.tool.is_file():
        print(
            f"{arguments.tool} does not exist. Build it first: ninja -C build",
            file=sys.stderr,
        )
        return 1

    results = run(arguments.tool, tuple(arguments.sizes), arguments.repeats)
    report(results, arguments.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
