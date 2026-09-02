# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The ONNX frontend of the simulated edge NPU compiler.

This is the one Python package root in the project. Section 17.3a is explicit
about that: an earlier draft put the reference interpreter in a second package
called `python/npu/`, which would have meant two import roots, two mypy
configurations, and two places to forget to add a module. Everything Python in
this project lives here, including the reference interpreter when it arrives at
P7.

Three entry points matter today.

``import_model`` takes an ONNX model at the pinned opset and returns `npu`
dialect IR as text, verified by `npu-opt` on the way out. ``generate_model``
builds one of the suite's seeded models and writes it as ONNX. ``compile_model``
is `npu-compile`, the driver of Section 6: it runs the whole pipeline at a named
optimization level and stops after whichever stage was asked for.

    from npu_frontend import compile_model, generate_model, run_program

    path = generate_model("lenet", directory)
    program = compile_model(path, level=0, emit="nbin")
    answer = run_program(program.binary, [x], program.output_shapes)

The package needs two things on top of its pip dependencies: the MLIR Python
bindings on `sys.path`, which `test/Python/conftest.py` arranges for the test
suite and `PYTHONPATH` arranges for everything else, and a built
`./build/bin/npu-opt`, which is a runtime dependency rather than a test one and
is why `import_model` raises rather than degrading when it cannot find one.
"""

from __future__ import annotations

from . import attributes
from .builder import find_npu_opt, find_tool, verify
from .compile import (
    CompileError,
    CompileResult,
    SimulationError,
    SimulationResult,
    ablatable_passes,
    compile_model,
    describe_pipeline,
    implemented_levels,
    run_program,
)
from .diagnostics import ONNXImportError, VerificationError
from .model_generator import (
    GENERATOR_VERSION,
    INPUT_SHAPES,
    MODELS,
    generate_all,
    generate_model,
)
from .onnx_importer import PINNED_OPSET, import_model, import_model_file
from .op_mapping import CONVERTERS, DEFERRED
from .pass_stats import (
    PassRecord,
    PassStatisticsError,
    cross_check_against_mlir_timing,
    expected_passes,
    load_pass_stats,
)
from .tolerances import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE

__all__ = [
    "ABSOLUTE_TOLERANCE",
    "CONVERTERS",
    "DEFERRED",
    "GENERATOR_VERSION",
    "INPUT_SHAPES",
    "MODELS",
    "RELATIVE_TOLERANCE",
    "CompileError",
    "CompileResult",
    "ONNXImportError",
    "PINNED_OPSET",
    "PassRecord",
    "PassStatisticsError",
    "SimulationError",
    "SimulationResult",
    "VerificationError",
    "ablatable_passes",
    "attributes",
    "compile_model",
    "cross_check_against_mlir_timing",
    "describe_pipeline",
    "expected_passes",
    "find_npu_opt",
    "find_tool",
    "generate_all",
    "generate_model",
    "implemented_levels",
    "import_model",
    "import_model_file",
    "load_pass_stats",
    "run_program",
    "verify",
]
