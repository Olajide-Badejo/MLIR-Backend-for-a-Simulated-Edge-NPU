"""Make the prebuilt MLIR Python bindings importable.

The bindings are produced by the one time LLVM/MLIR build and are not pip
installed. Their location is taken from the ``MLIR_PYTHON_PACKAGES`` environment
variable when set, otherwise from the default build tree path documented in
docs/BUILD.md. Importing this module for its side effect puts ``mlir`` on the
path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT = (
    Path.home()
    / "llvm-project"
    / "build"
    / "tools"
    / "mlir"
    / "python_packages"
    / "mlir_core"
)


def _bindings_dir() -> Path:
    env = os.environ.get("MLIR_PYTHON_PACKAGES")
    return Path(env) if env else _DEFAULT


def ensure_on_path() -> None:
    d = _bindings_dir()
    if not (d / "mlir").is_dir():
        raise RuntimeError(
            f"MLIR Python bindings not found at {d}. Build LLVM/MLIR with "
            "MLIR_ENABLE_BINDINGS_PYTHON=ON (see docs/BUILD.md), or set the "
            "MLIR_PYTHON_PACKAGES environment variable to the mlir_core directory."
        )
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


ensure_on_path()
