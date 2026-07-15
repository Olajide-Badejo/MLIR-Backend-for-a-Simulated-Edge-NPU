"""Shared pytest fixtures for the frontend tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the npu_frontend package importable without an install step.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


@pytest.fixture(scope="session")
def npu_opt() -> str:
    """Path to the built npu-opt binary; skip the test if it is not built."""
    candidate = os.environ.get("NPU_OPT") or str(REPO / "build" / "bin" / "npu-opt")
    if not Path(candidate).exists():
        pytest.skip(f"npu-opt not built at {candidate}")
    return candidate
