# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""`npu-compile`: one entry point from an ONNX file to a `.nbin`.

Section 6 settles the shape of this and the reasons are worth keeping here
rather than only in the specification.

**The driver is Python and the pipeline is C++.** The import step is Python by
design, because the MLIR bindings are, so a C++ driver would have to shell out
to Python for its first stage. The pass pipeline is not Python, because the
`PassInstrumentation` of Section 16.2 has to sit on the `PassManager` that
actually runs the passes; a pipeline assembled here out of one `npu-opt`
invocation per pass would be a different pipeline from the one under test, which
is exactly what Section 17.4 forbids in the other direction. So this file names
a level and `lib/Pipeline` builds it.

**The consequence, which is the reason the split matters:** the ablatable pass
set is read out of the compiler at run time, by
``npu-opt --npu-describe-pipeline``, rather than written down a second time
here. A list maintained by hand in Python stops covering a pass on the day one
is added, and nothing goes red.

**The stages.** ``--emit`` stops after one of four:

===========  ==========================================================
``import``   the IR the importer produced, verified by `npu-opt`
``npu``      the same IR after the level's `npu` level passes
``npuisa``   after lowering and scratchpad allocation
``nbin``     the binary, from `npu-translate`
===========  ==========================================================

At `-O0` ``import`` and ``npu`` are the same text, because `-O0` runs no `npu`
level pass at all. That is a property rather than an accident and
``test_compile_driver.py`` asserts it, so that the day `-O1` lands and the two
stop being equal, the assertion moves with the change rather than being noticed
later.

**Running the program is `npu-sim`'s job, not this one's.** ``run_program``
below wraps it for the harnesses, which need the outputs and the statistics as
arrays and a dictionary; the command line driver compiles and stops, the way a
compiler does. The statistics come back through ``npu-sim --json-stats`` and
never by parsing its human readable output, and a missing ``instructions``
raises by name, which is what Section 10.2 asks of every consumer.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
from onnx import ModelProto

from .builder import find_tool
from .diagnostics import ONNXImportError
from .onnx_importer import import_model

#: The stages `--emit` can stop after, in pipeline order.
EMIT_STAGES: Final[tuple[str, ...]] = ("import", "npu", "npuisa", "nbin")

#: The scratchpad budget the allocator uses when nobody names one. It is
#: repeated here from `Passes.td` rather than read out of it, and the repetition
#: is deliberate: this is the number `npu-compile --help` prints, and printing
#: "whatever the pass decides" would tell a reader nothing. A test asserts the
#: two agree.
DEFAULT_SCRATCHPAD_BUDGET: Final[int] = 1048576


class CompileError(Exception):
    """A stage of the driver refused. Carries the tool's own message."""


class SimulationError(Exception):
    """`npu-sim` refused, or said something this project will not read."""


@dataclass(frozen=True)
class StageTiming:
    """One stage's wall clock, for `--verbose`."""

    name: str
    seconds: float


@dataclass
class CompileResult:
    """What one compilation produced.

    ``text`` is set for the three IR stages and ``binary`` for ``nbin``. Both
    are held rather than one, because a caller that asked for ``nbin`` usually
    also wants the `npuisa` IR it came from, and recompiling to get it would
    mean the two could disagree.
    """

    level: int
    emit: str
    text: str | None = None
    binary: bytes | None = None
    #: The IR at each stage that ran, keyed by stage name.
    stages: dict[str, str] = field(default_factory=dict)
    timings: list[StageTiming] = field(default_factory=list)
    #: The graph's input shapes, in the order the `.nbin` declares its input
    #: regions, which is the order `npu-sim --input` wants them in.
    input_shapes: tuple[tuple[int, ...], ...] = ()
    #: Likewise for the outputs.
    output_shapes: tuple[tuple[int, ...], ...] = ()

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Writes whichever of the two forms this stage produced."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.binary is not None:
            target.write_bytes(self.binary)
        else:
            assert self.text is not None
            target.write_text(self.text, encoding="utf-8")
        return target


@dataclass(frozen=True)
class SimulationResult:
    """What one simulated run produced."""

    outputs: list[np.ndarray]
    stats: dict[str, Any]

    @property
    def instructions(self) -> int:
        """The instruction count, which Section 10.2 makes the only one."""
        return int(self.stats["instructions"])


# ---------------------------------------------------------------------------
# The pipeline description, read out of the compiler.
# ---------------------------------------------------------------------------


def describe_pipeline() -> dict[str, Any]:
    """The `-O` level table, as `npu-opt` describes it.

    Not cached. The table costs one process start and a compiler rebuilt
    between two calls in the same session is a compiler whose answer changed,
    which is a thing a developer does and a cache would hide.
    """
    tool = find_tool("npu-opt")
    completed = subprocess.run(
        [str(tool), "--npu-describe-pipeline"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CompileError(
            f"{tool} --npu-describe-pipeline exited {completed.returncode}. "
            "This is the compiler describing itself, so a failure here means "
            "the binary is not the one this driver was written against.\n\n"
            f"{completed.stderr.strip()}"
        )
    parsed: dict[str, Any] = json.loads(completed.stdout)
    return parsed


def level_description(level: int) -> dict[str, Any]:
    """One level's row of the table, or a refusal naming the levels there are."""
    table = describe_pipeline()
    for row in table["levels"]:
        if int(row["level"]) == level:
            return dict(row)
    known = ", ".join(f"-O{row['level']}" for row in table["levels"])
    raise CompileError(f"-O{level} is not an optimization level. There are {known}.")


def implemented_levels() -> list[int]:
    """The levels this compiler can build, as opposed to name."""
    return [
        int(row["level"]) for row in describe_pipeline()["levels"] if row["implemented"]
    ]


def ablatable_passes(level: int) -> list[str]:
    """The ablatable subset of a level's passes, read from the compiler.

    Section 16.2 requires the ablation set to be taken from the driver at run
    time and never hardcoded, because a hardcoded list silently stops covering
    a pass the day one is added. This is that read. At `-O0` it is empty, and
    correctly so: both of the level's passes are marked not ablatable, since
    removing either produces no program at all.
    """
    return [
        str(entry["pass"])
        for entry in level_description(level)["passes"]
        if entry["ablatable"]
    ]


# ---------------------------------------------------------------------------
# The stages.
# ---------------------------------------------------------------------------


def _boundary_shapes(
    model: ModelProto,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """The graph's input and output shapes, in declaration order.

    That order is the one the `.nbin` declares its regions in and therefore the
    one `npu-sim` wants its `--input` and `--output` arguments in. It is read
    from the ONNX graph rather than from the binary, because the binary carries
    extents and the caller needs them before it has a binary.

    An initializer that is also listed as a graph input is a constant in the
    pre IR version 4 spelling, and the importer does not make it an argument.
    Skipping it here is the same rule stated in the same order.
    """
    initializers = {tensor.name for tensor in model.graph.initializer}

    def extents(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
        return tuple(
            int(dimension.dim_value)
            for dimension in value_info.type.tensor_type.shape.dim
        )

    inputs = tuple(
        extents(info) for info in model.graph.input if info.name not in initializers
    )
    outputs = tuple(extents(info) for info in model.graph.output)
    return inputs, outputs


def _run(command: Sequence[str], stage: str, stdin: str | None = None) -> str:
    completed = subprocess.run(
        [str(part) for part in command],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CompileError(
            f"the {stage} stage failed: {command[0]} exited "
            f"{completed.returncode}.\n\n{completed.stderr.strip()}"
        )
    return completed.stdout


def compile_model(
    source: str | os.PathLike[str] | ModelProto,
    *,
    level: int = 0,
    emit: str = "nbin",
    budget: int | None = None,
    strip_debug: bool = False,
    function_name: str = "main",
    verbose: bool = False,
) -> CompileResult:
    """Compiles one ONNX model and stops after `emit`.

    `budget` of None means the allocator's own default, which is what leaving
    the flag off does on the command line as well.
    """
    if emit not in EMIT_STAGES:
        raise CompileError(
            f"{emit!r} is not a stage. The stages are " + ", ".join(EMIT_STAGES) + "."
        )

    row = level_description(level)
    if not row["implemented"]:
        raise CompileError(
            f"-O{level} is named by this compiler and not implemented: "
            f"{row['summary']}. It arrives at {row['arrives_at']}. The levels "
            "that exist are "
            + ", ".join(f"-O{value}" for value in implemented_levels())
            + "."
        )

    model = source if isinstance(source, ModelProto) else onnx.load(str(Path(source)))
    input_shapes, output_shapes = _boundary_shapes(model)

    result = CompileResult(
        level=level,
        emit=emit,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )

    def record(name: str, started: float, text: str) -> None:
        result.stages[name] = text
        result.timings.append(StageTiming(name, time.perf_counter() - started))

    # ---- import ----------------------------------------------------------
    started = time.perf_counter()
    try:
        imported = import_model(model, function_name=function_name)
    except ONNXImportError as failure:
        raise CompileError(f"the import stage failed: {failure}") from failure
    record("import", started, imported)
    if emit == "import":
        result.text = imported
        return _finish(result, verbose)

    npu_opt = find_tool("npu-opt")

    # ---- npu -------------------------------------------------------------
    #
    # The `npu` level passes of the chosen level, which at `-O0` are none. The
    # stage exists at `-O0` anyway and produces the imported text unchanged,
    # because a stage that appeared at `-O1` would make `--emit npu` mean two
    # different things at two levels.
    started = time.perf_counter()
    npu_level = imported
    record("npu", started, npu_level)
    if emit == "npu":
        result.text = npu_level
        return _finish(result, verbose)

    # ---- npuisa ----------------------------------------------------------
    #
    # One npu-opt invocation running the registered pipeline, not a pass list
    # assembled here. Section 17.4: a test that runs a hardcoded pass list
    # matching no optimization level enforces nothing, and a driver that did so
    # would make every such test vacuous at once.
    started = time.perf_counter()
    pipeline = str(row["pipeline"])
    argument = f"--{pipeline}"
    if budget is not None:
        argument += f"=budget={budget}"
    lowered = _run(
        [str(npu_opt), "-", argument, "--mlir-print-debuginfo"],
        "npuisa",
        stdin=npu_level,
    )
    record("npuisa", started, lowered)
    if emit == "npuisa":
        result.text = lowered
        return _finish(result, verbose)

    # ---- nbin ------------------------------------------------------------
    #
    # npu-translate writes no file on a failure and validates its own output
    # before it writes one on success, so there is nothing to clean up here on
    # either path. The temporary file exists only because the tool writes bytes
    # and a pipe would mean reading them back through text.
    started = time.perf_counter()
    translate = find_tool("npu-translate")
    with tempfile.TemporaryDirectory(prefix="npu-compile-") as directory:
        target = Path(directory) / "out.nbin"
        command = [str(translate), "-", "-o", str(target)]
        if strip_debug:
            command.append("--strip-debug")
        _run(command, "nbin", stdin=lowered)
        result.binary = target.read_bytes()
    result.timings.append(StageTiming("nbin", time.perf_counter() - started))
    return _finish(result, verbose)


def _finish(result: CompileResult, verbose: bool) -> CompileResult:
    """Prints the stage timings Section 18 asks `--verbose` for.

    To stderr, so that `--emit npu -o -` stays usable in a pipe. A driver whose
    progress reporting lands in its own output is a driver nobody pipes twice.
    """
    if verbose:
        total = sum(timing.seconds for timing in result.timings)
        print(
            f"npu-compile: -O{result.level}, stopping after {result.emit}",
            file=sys.stderr,
        )
        for timing in result.timings:
            print(
                f"npu-compile:   {timing.name:<8} {timing.seconds * 1000:8.1f} ms",
                file=sys.stderr,
            )
        print(f"npu-compile:   {'total':<8} {total * 1000:8.1f} ms", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Running what was compiled.
# ---------------------------------------------------------------------------


def run_program(
    program: str | os.PathLike[str] | bytes,
    inputs: Sequence[np.ndarray],
    output_shapes: Sequence[Sequence[int]],
    *,
    single_port: bool = False,
) -> SimulationResult:
    """Runs a `.nbin` under `npu-sim` and returns its outputs and statistics.

    `output_shapes` is required rather than inferred, and that is not laziness.
    The file declares its output regions with their extents, but reading them
    back out of the binary here would mean this module parsing the format, and
    the format has exactly one reader in this project on purpose. The caller
    compiled the model and therefore already knows the shapes; `CompileResult`
    carries them.
    """
    simulator = find_tool("npu-sim")

    with tempfile.TemporaryDirectory(prefix="npu-sim-") as directory:
        work = Path(directory)
        if isinstance(program, bytes):
            binary = work / "program.nbin"
            binary.write_bytes(program)
        else:
            binary = Path(program)

        command = [str(simulator), str(binary), "--quiet"]
        for index, array in enumerate(inputs):
            path = work / f"in{index}.bin"
            path.write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())
            command += ["--input", str(path)]

        output_paths = [work / f"out{index}.bin" for index in range(len(output_shapes))]
        for path in output_paths:
            command += ["--output", str(path)]

        stats_path = work / "stats.json"
        command += ["--json-stats", str(stats_path)]
        if single_port:
            command.append("--single-port")

        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise SimulationError(
                f"npu-sim exited {completed.returncode}. A nonzero exit is a "
                "refusal, and the outputs after a trap are whatever the "
                "skipped writes left behind, so nothing here reads "
                "them.\n\n" + (completed.stderr.strip() or completed.stdout.strip())
            )

        if not stats_path.is_file():
            raise SimulationError(
                "npu-sim exited 0 and wrote no statistics file. The file is "
                "written after a successful run, so its absence means the "
                "--json-stats argument did not reach the tool."
            )
        stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8"))

        # Section 10.2: stats.instructions is the only instruction count in
        # this project, and a harness raises when the field is missing rather
        # than falling back to counting lines of anything.
        if "instructions" not in stats:
            raise SimulationError(
                "npu-sim's statistics carry no 'instructions' field. Section "
                "10.2 makes it the only instruction count in this project and "
                "requires a harness to raise rather than fall back, because "
                "every fallback anybody has proposed counts something else: a "
                "regex over an IR dump matches inside type strings and counts "
                "constants the encoder treats as data.\n\nThe fields present "
                "were: " + ", ".join(sorted(stats)) + "."
            )

        outputs = [
            np.fromfile(path, dtype=np.float32).reshape(tuple(shape))
            for path, shape in zip(output_paths, output_shapes, strict=True)
        ]

    return SimulationResult(outputs=outputs, stats=stats)


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npu-compile",
        description=(
            "Compile an ONNX model to a .nbin for the simulated edge NPU. "
            "Running the result is npu-sim's job."
        ),
    )
    parser.add_argument(
        "model", nargs="?", help="the ONNX model to compile, at the pinned opset"
    )
    parser.add_argument(
        "-O",
        dest="level",
        type=int,
        default=0,
        metavar="LEVEL",
        help=(
            "the optimization level. -O0 is import and verify, then lowering "
            "and allocation. Higher levels arrive at P9 and are refused by "
            "name until they do."
        ),
    )
    parser.add_argument(
        "--emit",
        choices=EMIT_STAGES,
        default="nbin",
        help=(
            "stop after this stage. At -O0 'import' and 'npu' are the same "
            "text, because -O0 runs no npu level pass."
        ),
    )
    parser.add_argument(
        "-o",
        dest="output",
        help="write the result here. Defaults to stdout for the text stages.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        metavar="BYTES",
        help=(
            "the scratchpad budget. Defaults to the allocator's own, which is "
            f"{DEFAULT_SCRATCHPAD_BUDGET} bytes."
        ),
    )
    parser.add_argument(
        "--strip-debug",
        action="store_true",
        help="write an empty debug section into the .nbin",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the pipeline stages with their timings, on stderr",
    )
    parser.add_argument(
        "--describe-pipeline",
        action="store_true",
        help=(
            "print the -O level table as JSON and exit. The table comes from "
            "the compiler rather than from this script."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.describe_pipeline:
            json.dump(describe_pipeline(), sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
            return 0

        if not args.model:
            parser.error("a model is required unless --describe-pipeline is given")

        result = compile_model(
            args.model,
            level=args.level,
            emit=args.emit,
            budget=args.budget,
            strip_debug=args.strip_debug,
            verbose=args.verbose,
        )
    except (CompileError, ONNXImportError) as failure:
        print(f"npu-compile: {failure}", file=sys.stderr)
        return 1

    if args.output:
        written = result.write(args.output)
        if args.verbose:
            print(f"npu-compile: wrote {written}", file=sys.stderr)
        return 0

    if result.binary is not None:
        # A .nbin is bytes and stdout may be a terminal. Refusing is better
        # than a screenful of control characters that a reader then has to
        # decide was not a diagnostic.
        print(
            "npu-compile: --emit nbin writes a binary, so -o is required. "
            "Use --emit npuisa to see the IR it comes from.",
            file=sys.stderr,
        )
        return 1

    assert result.text is not None
    sys.stdout.write(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
