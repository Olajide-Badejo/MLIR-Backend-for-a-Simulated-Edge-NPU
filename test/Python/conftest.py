# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Put the MLIR Python bindings on sys.path for every test under test/Python.

The bindings are built by LLVM and are not pip installed. They resolve only
out of the LLVM build tree, so without this every test here fails at
``import mlir`` for a reason that has nothing to do with the code under test.
Section 3.3 of the build specification calls this wiring not optional and
names three places it has to exist: the MLIR_PYTHON_PACKAGES_DIR variable in
CMakeLists.txt, test/lit.cfg.py, and the pytest configuration. This file is
the third.

It is a conftest rather than an ``env`` key in pyproject.toml because that key
belongs to the pytest-env plugin, which this project does not install. pytest
warns about an unknown config key and then ignores it, so the key would look
like working wiring while doing nothing at all.

The path is resolved in this order, and the order is the point: an explicit
environment variable wins, then the build tree this repository configured,
then the default location. That way a developer with LLVM somewhere else sets
one variable, and everyone else needs no configuration.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLVM_BUILD = Path.home() / "llvm-project" / "build"
_PACKAGE_SUBPATH = Path("tools") / "mlir" / "python_packages" / "mlir_core"


def _from_cmake_cache() -> Path | None:
    """Read MLIR_PYTHON_PACKAGES_DIR out of this repository's CMake cache.

    Reading the cache rather than guessing means pytest and lit agree on the
    path by construction: the cache is the single value the configure step
    computed, and lit.site.cfg.py is generated from that same value.
    """
    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    if not cache.is_file():
        return None
    pattern = re.compile(r"^MLIR_PYTHON_PACKAGES_DIR:[^=]*=(.*)$")
    try:
        for line in cache.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match and match.group(1).strip():
                return Path(match.group(1).strip())
    except OSError:
        return None
    return None


def mlir_python_packages_dir() -> Path:
    override = os.environ.get("MLIR_PYTHON_PACKAGES_DIR")
    if override:
        return Path(override)
    from_cache = _from_cmake_cache()
    if from_cache is not None:
        return from_cache
    return DEFAULT_LLVM_BUILD / _PACKAGE_SUBPATH


def pytest_configure(config: object) -> None:
    bindings = str(mlir_python_packages_dir())
    if bindings not in sys.path:
        sys.path.insert(0, bindings)
    # Also export it, so a test that shells out to a subprocess, and the MLIR
    # bindings' own native extension loader, both see the same path.
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if bindings not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([bindings, *parts])
