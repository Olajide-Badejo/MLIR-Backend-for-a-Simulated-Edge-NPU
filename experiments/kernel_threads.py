# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The convolution kernel's thread scaling, and the bits it must not move.

Section 10.3 parallelises the convolution over the batch and output channel
dimensions and leaves the reductions sequential and in their original order, and
then requires that "the same input under one thread and under the maximum thread
count produces bitwise equal output buffers". `unittests/Simulator/Determinism
Test.cpp` asserts that on one synthetic convolution. This script asserts it on
the seven models of the suite, through `npu-sim`, and measures what the
parallelism is worth on each while it is there.

**Two different claims, and they are not equally portable.**

- **The bytes are a claim about this project.** Every thread count must produce
  the same output file, byte for byte, on every model. It holds on any host
  because it is a property of the loop nest rather than of the machine, and it
  is what `--check` gates on.
- **The times are a measurement of one host.** A wall clock carries the machine
  it was taken on, and no two rows from two machines are comparable. Nothing
  gates on a time here and `--check` ignores every one of them.

**Why this measures whole `npu-sim` invocations rather than the kernel.** The
number a reader wants is what the parallel kernel is worth to somebody running
the simulator, and a process' startup, its file reads and its serial kernels are
part of that. The consequence is worth stating rather than discovering: on a
small model the convolution is a minority of the run and the speedup here is
correspondingly small, which is a fact about the suite's models and not a fault
in the parallelism. The `--repeats` minimum is the usual choice for a timing
loop on a shared machine, because the distribution's lower tail is the machine
doing the work and nothing else.

**`--kernel-info` is asked first and the answer is printed with the table.**
D-0047 is the reason: between P7 and P12 the kernels were compiled without
OpenMP while everything that linked them was compiled with it, so a table of
flat speedups looked exactly like a table of small models. It is reported rather
than gated, because a build without OpenMP is a supported build.

Usage:

    python experiments/kernel_threads.py
    python experiments/kernel_threads.py --threads 1 4 28 --repeats 7
    python experiments/kernel_threads.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mlir_python_packages_dir() -> Path:
    """Where the MLIR Python bindings live.

    The same resolution `experiments/allocator_fragmentation.py` uses, and it is
    repeated for the same reason: a conftest is pytest's file and an experiment
    is not run under pytest. Section 3.3 names the places this wiring has to
    exist and a script that imports the frontend is one more of them.
    """
    override = os.environ.get("MLIR_PYTHON_PACKAGES_DIR")
    if override:
        return Path(override)

    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    if cache.is_file():
        pattern = re.compile(r"^MLIR_PYTHON_PACKAGES_DIR:[^=]*=(.*)$")
        for line in cache.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match and match.group(1).strip():
                return Path(match.group(1).strip())

    return (
        Path.home()
        / "llvm-project"
        / "build"
        / "tools"
        / "mlir"
        / "python_packages"
        / "mlir_core"
    )


sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

#: The thread counts measured by default. One and the host's maximum are the two
#: the Section 10.3 assertion is written about; the powers of two between them
#: are what turns a pair of numbers into a curve, and a curve is what shows
#: whether a flat result is saturation or is no parallelism at all.
DEFAULT_THREADS = (1, 2, 4, 8, 0)

#: The seed every input is drawn from. Fixed, because two runs of this script
#: comparing different bytes would be comparing nothing.
INPUT_SEED = 20260904

#: The level the models are compiled at. `-O2` is the level the suite reports
#: and the one whose kernels a reader is asking about.
DEFAULT_LEVEL = 2


@dataclass
class ModelTiming:
    """One model, at every thread count asked for."""

    model: str
    #: Thread count to best wall clock in seconds.
    seconds: dict[int, float] = field(default_factory=dict)
    #: Thread count to the sha256 of the concatenated output files.
    digest: dict[int, str] = field(default_factory=dict)

    def outputs_agree(self) -> bool:
        """Whether every thread count produced the same bytes."""
        return len(set(self.digest.values())) <= 1

    def speedup(self, threads: int) -> float | None:
        """Best time at one thread over best time at `threads`."""
        if 1 not in self.seconds or threads not in self.seconds:
            return None
        if self.seconds[threads] <= 0.0:
            return None
        return self.seconds[1] / self.seconds[threads]


def host_thread_maximum() -> int:
    """How many threads this host would give a parallel region.

    `os.process_cpu_count` rather than `os.cpu_count`, because a run inside a
    cpuset sees fewer processors than the machine has and libgomp's default
    follows the affinity mask rather than the hardware. A caller can override it
    with `--threads`, and the number that was used is printed with the table
    rather than left to be inferred from the column heading.

    It arrived in CPython 3.13 and this project's floor is 3.11, so the fallback
    is written out rather than assumed. The CI image ships 3.12 and takes the
    second branch, which is the branch a container without an affinity mask
    would have answered the same way anyway.
    """
    counter = getattr(os, "process_cpu_count", None)
    if counter is not None:
        return counter() or 1
    return os.cpu_count() or 1


def resolve_threads(requested: list[int]) -> list[int]:
    """Turns the request into thread counts, with 0 meaning the host maximum.

    Zero is the placeholder rather than a literal maximum, so that the default
    is a property of the host and not of this file. Duplicates are dropped,
    which is what a host with two processors does to the default list.
    """
    maximum = host_thread_maximum()
    resolved: list[int] = []
    for value in requested:
        count = maximum if value == 0 else value
        if count < 1:
            raise ValueError(
                f"a thread count of {value} is not one this script can ask for. "
                "Use a positive count, or 0 for the host maximum."
            )
        if count not in resolved:
            resolved.append(count)
    return sorted(resolved)


def kernel_info(simulator: Path) -> tuple[bool, int]:
    """Whether the kernels have OpenMP, and the threads they would use.

    Read out of `npu-sim --kernel-info`, which reads it out of `Kernels.cpp`.
    Asking the tool rather than this process is the whole point: D-0047 was a
    disagreement between one translation unit and every other, and a caller that
    answered from its own state would have reproduced it.
    """
    completed = subprocess.run(
        [str(simulator), "--kernel-info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"npu-sim --kernel-info exited {completed.returncode}. This build "
            "predates the flag, or the tool refused:\n" + completed.stderr.strip()
        )
    has_openmp = False
    threads = 1
    for line in completed.stdout.splitlines():
        if line.startswith("kernel openmp:"):
            has_openmp = line.split(":", 1)[1].strip() == "yes"
        elif line.startswith("kernel threads:"):
            threads = int(line.split(":", 1)[1].strip())
    return has_openmp, threads


def measure(
    simulator: Path,
    binary: Path,
    inputs: list[Path],
    outputs: list[Path],
    threads: int,
    repeats: int,
) -> tuple[float, str]:
    """Best of `repeats` runs at `threads`, and the digest of what it wrote.

    One warm up run before the loop, discarded, so that the first timed run is
    not paying for the page cache of a file the others read warm.
    """
    command = [str(simulator), str(binary), "--quiet"]
    for path in inputs:
        command += ["--input", str(path)]
    for path in outputs:
        command += ["--output", str(path)]

    environment = dict(os.environ, OMP_NUM_THREADS=str(threads))

    def once() -> float:
        started = time.perf_counter()
        completed = subprocess.run(
            command, env=environment, capture_output=True, text=True, check=False
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"npu-sim exited {completed.returncode} at {threads} threads. A "
                "nonzero exit is a refusal and the outputs after one are "
                "whatever the skipped writes left behind:\n"
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        return elapsed

    once()
    best = min(once() for _ in range(repeats))

    digest = hashlib.sha256()
    for path in outputs:
        digest.update(path.read_bytes())
    return best, digest.hexdigest()


def run(
    simulator: Path,
    models: list[str],
    threads: list[int],
    repeats: int,
    level: int,
) -> list[ModelTiming]:
    """Compiles each model once and times it at every thread count."""
    import numpy as np
    from npu_frontend.compile import compile_model
    from npu_frontend.model_generator import generate_model

    timings: list[ModelTiming] = []
    with tempfile.TemporaryDirectory(prefix="npu-threads-") as directory:
        work = Path(directory)
        for name in models:
            source = generate_model(name, str(work))
            compiled = compile_model(source, level=level, emit="nbin")

            binary = work / f"{name}.nbin"
            binary.write_bytes(compiled.binary)

            generator = np.random.default_rng(INPUT_SEED)
            inputs: list[Path] = []
            for index, shape in enumerate(compiled.input_shapes):
                path = work / f"{name}-in{index}.bin"
                values = generator.standard_normal(tuple(shape)).astype(np.float32)
                path.write_bytes(values.tobytes())
                inputs.append(path)

            timing = ModelTiming(model=name)
            for count in threads:
                outputs = [
                    work / f"{name}-out{index}-t{count}.bin"
                    for index in range(len(compiled.output_shapes))
                ]
                seconds, digest = measure(
                    simulator, binary, inputs, outputs, count, repeats
                )
                timing.seconds[count] = seconds
                timing.digest[count] = digest
            timings.append(timing)
    return timings


def report(
    timings: list[ModelTiming],
    threads: list[int],
    repeats: int,
    has_openmp: bool,
    kernel_threads: int,
) -> None:
    """The table, and the two sentences that say how to read it."""
    print(f"Kernel thread scaling, best of {repeats} runs per model, per count.")
    print(
        "The kernels report OpenMP "
        + ("on" if has_openmp else "off")
        + f" and a default of {kernel_threads} threads, read out of "
        "npu-sim --kernel-info."
    )
    print(
        "Times are whole npu-sim invocations on one host and are not comparable "
        "with a time from another."
    )
    print()

    widest = threads[-1]
    header = (
        f"{'model':<22}"
        + "".join(f"{f'{count}t s':>10}" for count in threads)
        + f"{f'x{widest}':>8}{'bytes':>9}"
    )
    print(header)
    print("-" * len(header))
    for timing in timings:
        speedup = timing.speedup(widest)
        print(
            f"{timing.model:<22}"
            + "".join(f"{timing.seconds.get(count, 0.0):>10.4f}" for count in threads)
            + f"{(speedup if speedup is not None else 0.0):>8.2f}"
            + f"{('equal' if timing.outputs_agree() else 'DIFFER'):>9}"
        )
    print()
    print(
        "The bytes column is the gate. Section 10.3 requires one thread and the "
        "maximum to agree bitwise, and a DIFFER on any row means a reduction "
        "moved into the parallel region."
    )


def verdict(timings: list[ModelTiming]) -> list[str]:
    """The models whose outputs moved with the thread count, if any."""
    return [timing.model for timing in timings if not timing.outputs_agree()]


def as_json(
    timings: list[ModelTiming],
    threads: list[int],
    repeats: int,
    has_openmp: bool,
    kernel_threads: int,
) -> dict[str, object]:
    """The rows as data, for a caller that reads them rather than looks at them.

    The host context travels with the numbers rather than beside them, because a
    wall clock separated from the machine it was taken on is the kind of figure
    that ends up in a table next to one from a different machine.
    """
    return {
        "repeats": repeats,
        "threads_measured": threads,
        "kernel_openmp": has_openmp,
        "kernel_default_threads": kernel_threads,
        "host_process_cpu_count": host_thread_maximum(),
        "input_seed": INPUT_SEED,
        "models": [
            {
                "model": timing.model,
                "seconds": {str(k): v for k, v in timing.seconds.items()},
                "output_sha256": {str(k): v for k, v in timing.digest.items()},
                "outputs_agree": timing.outputs_agree(),
                "speedup_at_max": timing.speedup(threads[-1]),
            }
            for timing in timings
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--simulator",
        type=Path,
        default=None,
        help="the npu-sim binary (default: the four step discovery of "
        "npu_frontend.builder.find_tool)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="models to measure (default: every model in the suite)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=list(DEFAULT_THREADS),
        help="thread counts to measure, where 0 means the host maximum "
        "(default: 1 2 4 8 0)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed runs per point, the fastest of which is reported (default: 5)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=DEFAULT_LEVEL,
        help=f"optimization level to compile at (default: {DEFAULT_LEVEL})",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the rows as JSON to this path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the verdict line on success. The byte comparison itself is "
        "not optional and a difference exits nonzero with or without this flag, "
        "because a flag that could switch it off would eventually be passed",
    )
    arguments = parser.parse_args(argv)

    from npu_frontend.builder import find_tool
    from npu_frontend.model_generator import MODELS

    simulator = (
        arguments.simulator if arguments.simulator is not None else find_tool("npu-sim")
    )
    if not Path(simulator).is_file():
        print(
            f"{simulator} does not exist. Build it first: ninja -C build",
            file=sys.stderr,
        )
        return 1

    try:
        threads = resolve_threads(list(arguments.threads))
    except ValueError as failure:
        print(str(failure), file=sys.stderr)
        return 1

    if arguments.repeats < 1:
        print("--repeats has to be at least 1", file=sys.stderr)
        return 1

    models = list(arguments.models) if arguments.models else sorted(MODELS)
    has_openmp, kernel_threads = kernel_info(Path(simulator))

    timings = run(Path(simulator), models, threads, arguments.repeats, arguments.level)
    report(timings, threads, arguments.repeats, has_openmp, kernel_threads)

    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                as_json(
                    timings, threads, arguments.repeats, has_openmp, kernel_threads
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    moved = verdict(timings)
    if moved:
        print(
            "\nthe output bytes moved with the thread count on: "
            + ", ".join(moved)
            + "\nSection 10.3 makes that a defect in the kernel and never a "
            "number to re-record.",
            file=sys.stderr,
        )
        return 1
    if arguments.check:
        print("\nkernel-threads: bitwise equal at every thread count, exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
