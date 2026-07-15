"""End to end numerical validation: compile and simulate a model, then compare
the simulator output to onnxruntime within tolerance.

This exercises the whole pipeline: ONNX import, optimization, lowering to npuisa,
scratchpad allocation, binary encoding, and simulation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from npu_frontend import model_generator, onnx_importer


def _compile_and_simulate(
    tmp_path: Path,
    npu_opt: str,
    name: str,
    seed: int,
    input_array: np.ndarray,
    budget: int = 1048576,
) -> np.ndarray:
    bindir = Path(npu_opt).parent
    npu_translate = bindir / "npu-translate"
    npu_sim = bindir / "npu-sim"
    for tool in (npu_translate, npu_sim):
        if not tool.exists():
            pytest.skip(f"{tool} not built")

    onnx_path = model_generator.export(name, tmp_path / f"{name}.onnx", seed=seed)
    mlir_path = tmp_path / f"{name}.mlir"
    mlir_path.write_text(onnx_importer.import_model(onnx_path))

    isa_path = tmp_path / f"{name}.isa.mlir"
    subprocess.run(
        [
            npu_opt,
            str(mlir_path),
            "-canonicalize",
            "-npu-fuse-ops",
            "-npu-lower-to-npuisa",
            f"-npu-allocate-scratchpad=budget={budget}",
            "-o",
            str(isa_path),
        ],
        check=True,
    )
    nbin_path = tmp_path / f"{name}.nbin"
    subprocess.run(
        [str(npu_translate), str(isa_path), "-o", str(nbin_path)], check=True
    )

    in_path = tmp_path / "input.bin"
    in_path.write_bytes(np.ascontiguousarray(input_array, dtype=np.float32).tobytes())
    out_path = tmp_path / "output.bin"
    subprocess.run(
        [
            str(npu_sim),
            str(nbin_path),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
        ],
        check=True,
    )
    return np.frombuffer(out_path.read_bytes(), dtype=np.float32)


def test_lenet_matches_onnxruntime(tmp_path, npu_opt):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 28, 28)).astype(np.float32)

    onnx_path = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    session = ort.InferenceSession(str(onnx_path))
    reference = session.run(None, {"input": x})[0]

    simulated = _compile_and_simulate(tmp_path, npu_opt, "lenet", 0, x)
    simulated = simulated.reshape(reference.shape)

    np.testing.assert_allclose(simulated, reference, rtol=1e-3, atol=1e-3)


def test_lenet_with_spilling_matches_onnxruntime(tmp_path, npu_opt):
    # A 140 KB budget is below the LeNet peak, so the allocator spills buffers to
    # DRAM and reloads them. The numerics must survive the round trip.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 28, 28)).astype(np.float32)

    onnx_path = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    reference = ort.InferenceSession(str(onnx_path)).run(None, {"input": x})[0]

    simulated = _compile_and_simulate(tmp_path, npu_opt, "lenet", 0, x, budget=143360)
    simulated = simulated.reshape(reference.shape)

    np.testing.assert_allclose(simulated, reference, rtol=1e-3, atol=1e-3)
