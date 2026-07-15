# isort: skip_file
"""npu_frontend: ONNX to npu dialect importer and seeded test model generator.

Importing this package first puts the prebuilt MLIR Python bindings on the path
(via ``_mlir``) so that the submodules can ``from mlir import ir``. The import
order below is deliberate and must not be reordered.
"""

from __future__ import annotations

from . import _mlir  # noqa: F401  side effect: put the mlir bindings on sys.path
from . import model_generator, onnx_importer, op_mapping
from .op_mapping import UnsupportedOpError

__all__ = ["model_generator", "onnx_importer", "op_mapping", "UnsupportedOpError"]
