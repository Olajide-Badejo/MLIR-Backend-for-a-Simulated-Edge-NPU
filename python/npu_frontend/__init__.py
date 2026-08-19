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

``import_model`` takes an ONNX model at the pinned opset and returns `npu`
dialect IR as text, verified by `npu-opt` on the way out.

    from npu_frontend import import_model_file

    print(import_model_file("model.onnx"))

The package needs two things on top of its pip dependencies: the MLIR Python
bindings on `sys.path`, which `test/Python/conftest.py` arranges for the test
suite and `PYTHONPATH` arranges for everything else, and a built
`./build/bin/npu-opt`, which is a runtime dependency rather than a test one and
is why `import_model` raises rather than degrading when it cannot find one.
"""

from __future__ import annotations

from . import attributes
from .builder import find_npu_opt, verify
from .diagnostics import ONNXImportError, VerificationError
from .onnx_importer import PINNED_OPSET, import_model, import_model_file
from .op_mapping import CONVERTERS, DEFERRED

__all__ = [
    "CONVERTERS",
    "DEFERRED",
    "ONNXImportError",
    "PINNED_OPSET",
    "VerificationError",
    "attributes",
    "find_npu_opt",
    "import_model",
    "import_model_file",
    "verify",
]
