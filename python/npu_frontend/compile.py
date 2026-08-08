"""npu-compile: the end to end compiler driver.

Takes a trained ONNX model to an executable .nbin instruction stream, or stops
early at a chosen stage. Optimization levels control the npu dialect passes:

  -O0  import and verify only
  -O1  + canonicalize and constant fold
  -O2  + operator fusion and dead code elimination

The stages, selectable with --emit, are:

  import   the npu dialect IR straight from the ONNX importer
  npu      the npu dialect IR after the optimization level's passes
  npuisa   the npuisa instruction IR after lowering and scratchpad allocation
  nbin     the encoded binary (default)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from npu_frontend import onnx_importer

STAGES = ["import", "npu", "npuisa", "nbin"]

# The stage helper is generic over what its stage returns: the text stages give
# back str, the encode stage gives back bytes.
T = TypeVar("T")


def _default_bin_dir() -> Path:
    env = os.environ.get("NPU_BIN")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "build" / "bin"


def _run_opt(npu_opt: Path, text: str, passes: list[str]) -> str:
    proc = subprocess.run(
        [str(npu_opt), *passes], input=text, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"npu-opt failed:\n{proc.stderr}")
    return proc.stdout


def _passes_for_level(level: int) -> list[str]:
    passes: list[str] = []
    if level >= 1:
        passes.append("-canonicalize")  # canonicalize also constant folds
    if level >= 2:
        passes += ["-npu-fuse-ops", "-canonicalize", "-symbol-dce"]
    return passes


def compile_model(
    onnx_path: str | Path,
    *,
    opt_level: int = 2,
    emit: str = "nbin",
    output: str | Path | None = None,
    budget: int = 1048576,
    bin_dir: Path | None = None,
    verbose: bool = False,
) -> str | bytes:
    bin_dir = bin_dir or _default_bin_dir()
    npu_opt = bin_dir / "npu-opt"
    npu_translate = bin_dir / "npu-translate"

    def stage(name: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = fn()
        if verbose:
            ms = (time.perf_counter() - start) * 1000
            print(f"  {name:<10} {ms:8.1f} ms", file=sys.stderr)
        return result

    text = stage("import", lambda: onnx_importer.import_model(onnx_path))
    if emit == "import":
        return _write(output, text)

    text = stage(
        "optimize", lambda: _run_opt(npu_opt, text, _passes_for_level(opt_level))
    )
    if emit == "npu":
        return _write(output, text)

    isa_passes = [
        "-npu-lower-to-npuisa",
        f"-npu-allocate-scratchpad=budget={budget}",
    ]
    text = stage("lower", lambda: _run_opt(npu_opt, text, isa_passes))
    if emit == "npuisa":
        return _write(output, text)

    def encode() -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            isa = Path(tmp) / "program.isa.mlir"
            isa.write_text(text)
            nbin = Path(tmp) / "program.nbin"
            proc = subprocess.run(
                [str(npu_translate), str(isa), "-o", str(nbin)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"npu-translate failed:\n{proc.stderr}")
            return nbin.read_bytes()

    data = stage("encode", encode)
    if output:
        Path(output).write_bytes(data)
    return data


def _write(output: str | Path | None, text: str) -> str:
    # Write to a file when requested; never to stdout, so the function is quiet
    # when used as a library. The CLI handles stdout.
    if output:
        Path(output).write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="npu-compile", description=__doc__)
    parser.add_argument("model", help="input .onnx model")
    parser.add_argument(
        "-o", "--output", help="output path (default stdout for text stages)"
    )
    parser.add_argument(
        "-O",
        dest="opt",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="optimization level (default 2)",
    )
    parser.add_argument(
        "--emit", choices=STAGES, default="nbin", help="stage to emit (default nbin)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1048576,
        help="scratchpad size in bytes (default 1 MB)",
    )
    parser.add_argument("--bin-dir", help="directory holding npu-opt and npu-translate")
    parser.add_argument(
        "--verbose", action="store_true", help="print pass stage timings"
    )
    args = parser.parse_args(argv)

    if args.emit == "nbin" and not args.output:
        parser.error("--emit nbin requires -o OUTPUT")

    try:
        result = compile_model(
            args.model,
            opt_level=args.opt,
            emit=args.emit,
            output=args.output,
            budget=args.budget,
            bin_dir=Path(args.bin_dir) if args.bin_dir else None,
            verbose=args.verbose,
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"npu-compile: {e}", file=sys.stderr)
        return 1

    # Text stages go to stdout when no output file was given.
    if not args.output and isinstance(result, str):
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
