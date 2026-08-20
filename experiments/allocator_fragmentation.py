# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The fragmentation ratio of Section 13.1, per model and per strategy.

`fragmentation_ratio` is the assigned high water mark divided by the sweep line
peak, and Section 13.1 calls it the headline allocator metric: it is what
TelaMalloc, MiniMalloc and the TFLite Micro arena planner all report. This
script is what produces it per model at P5.

**Why a script here and not a field in a result cell.** Section 16.1's result
schema carries `fragmentation_ratio`, and Section 16.1 is explicit that the
field carries `null` until the phase that populates it lands. The schema is
specified at P10 and there is no end to end pipeline before P8, so at P5 the
honest arrangement is: the pass computes the number and writes it on the
function, and this script reads it off every model in the suite so that the
metric is exercised on real programs rather than only on the hand written lit
cases. At P10 the loop below is what `run_benchmarks.py` absorbs.

Each model is generated from the seed of `npu_frontend.model_generator`,
imported to the `npu` dialect, lowered, and allocated. Nothing is downloaded and
no `.onnx` file is committed.

Usage:

    python experiments/allocator_fragmentation.py
    python experiments/allocator_fragmentation.py --models lenet resnet_block
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOL = REPO_ROOT / "build" / "bin" / "npu-opt"


def _mlir_python_packages_dir() -> Path:
    """Where the MLIR Python bindings live.

    The same three step resolution `test/Python/conftest.py` uses, and it is
    repeated rather than imported because a conftest is pytest's file and an
    experiment is not run under pytest. An explicit environment variable wins,
    then the build tree this repository configured, then the default location.
    Section 3.3 names the places this wiring has to exist; a script that imports
    the frontend is one more of them.
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

from npu_frontend.model_generator import MODELS, generate_model  # noqa: E402
from npu_frontend.onnx_importer import import_model_file  # noqa: E402

STRATEGIES = ("pack", "interval")

_ATTRIBUTE = re.compile(r"npuisa\.(\w+) = ([-0-9.e+]+)")


@dataclass(frozen=True)
class Allocation:
    """What the allocator recorded on one function."""

    model: str
    strategy: str
    bytes_used: int
    peak_bytes: int
    ratio: float
    spill_count: int
    spill_dma_count: int


def allocate(tool: Path, module: str, strategy: str) -> Allocation | None:
    """Lower and allocate one module, and read the attributes back off it."""
    completed = subprocess.run(
        [
            str(tool),
            "--npu-lower-to-npuisa",
            f"--npu-allocate-scratchpad=strategy={strategy}",
        ],
        input=module,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stderr.strip()[:2000], file=sys.stderr)
        return None

    found = dict(_ATTRIBUTE.findall(completed.stdout))
    required = {
        "scratchpad_bytes",
        "scratchpad_peak_bytes",
        "fragmentation_ratio",
        "spill_count",
        "spill_dma_count",
    }
    if not required.issubset(found):
        print(
            "the allocated function is missing an attribute this script reads: "
            f"{sorted(required - set(found))}",
            file=sys.stderr,
        )
        return None

    return Allocation(
        model="",
        strategy=strategy,
        bytes_used=int(found["scratchpad_bytes"]),
        peak_bytes=int(found["scratchpad_peak_bytes"]),
        ratio=float(found["fragmentation_ratio"]),
        spill_count=int(found["spill_count"]),
        spill_dma_count=int(found["spill_dma_count"]),
    )


def run(tool: Path, models: list[str]) -> list[Allocation]:
    results: list[Allocation] = []
    with tempfile.TemporaryDirectory() as directory:
        for name in models:
            path = generate_model(name, directory)
            module = import_model_file(path)
            for strategy in STRATEGIES:
                allocation = allocate(tool, module, strategy)
                if allocation is None:
                    print(f"{name} under {strategy}: failed", file=sys.stderr)
                    continue
                results.append(
                    Allocation(
                        model=name,
                        strategy=allocation.strategy,
                        bytes_used=allocation.bytes_used,
                        peak_bytes=allocation.peak_bytes,
                        ratio=allocation.ratio,
                        spill_count=allocation.spill_count,
                        spill_dma_count=allocation.spill_dma_count,
                    )
                )
    return results


def report(results: list[Allocation]) -> None:
    header = (
        f"{'model':<22}  {'strategy':<9}  {'bytes':>9}  {'peak':>9}  "
        f"{'ratio':>7}  {'spills':>6}  {'dma':>4}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.model:<22}  {result.strategy:<9}  {result.bytes_used:>9}  "
            f"{result.peak_bytes:>9}  {result.ratio:>7.4f}  "
            f"{result.spill_count:>6}  {result.spill_dma_count:>4}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tool", type=Path, default=DEFAULT_TOOL)
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the rows as JSON to this path",
    )
    arguments = parser.parse_args(argv)

    if not arguments.tool.is_file():
        print(
            f"{arguments.tool} does not exist. Build it first: ninja -C build",
            file=sys.stderr,
        )
        return 1

    results = run(arguments.tool, list(arguments.models))
    if not results:
        return 1
    report(results)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps([result.__dict__ for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
