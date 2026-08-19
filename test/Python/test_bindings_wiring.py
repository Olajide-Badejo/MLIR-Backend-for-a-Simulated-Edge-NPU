# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Prove the MLIR Python bindings resolve under pytest.

This is the test that catches the failure mode Section 3.3 warns about. If the
PYTHONPATH wiring in conftest.py breaks, every future test under test/Python
fails at import with ``ModuleNotFoundError: mlir``, and the error points at
whatever test happened to run first rather than at the wiring. This test
points at the wiring. It is also the reason conftest.py is exercised at P0
rather than first exercised at P3, when there is real frontend code to blame
for a failure instead.
"""

from __future__ import annotations

import mlir.ir


def test_bindings_import_and_build_a_module() -> None:
    """A context, a module and a type is the smallest end to end use."""
    with mlir.ir.Context():
        module = mlir.ir.Module.parse("func.func private @f(i32) -> i32")
        assert "@f" in str(module)


def test_f32_type_round_trips() -> None:
    with mlir.ir.Context():
        assert str(mlir.ir.F32Type.get()) == "f32"
