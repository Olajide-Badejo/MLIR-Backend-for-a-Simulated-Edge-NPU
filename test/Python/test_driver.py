"""Tests for the npu-compile driver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from npu_frontend import compile as npu_compile
from npu_frontend import model_generator


def _bindir(npu_opt: str) -> Path:
    bindir = Path(npu_opt).parent
    if not (bindir / "npu-translate").exists():
        pytest.skip("npu-translate not built")
    return bindir


def test_emit_stages(tmp_path, npu_opt):
    bindir = _bindir(npu_opt)
    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)

    npu_ir = npu_compile.compile_model(onnx, opt_level=2, emit="npu", bin_dir=bindir)
    assert "npu.conv2d" in npu_ir

    isa_ir = npu_compile.compile_model(onnx, opt_level=2, emit="npuisa", bin_dir=bindir)
    assert "npuisa.conv2d" in isa_ir
    assert "npu.conv2d" not in isa_ir  # fully lowered off the npu dialect

    data = npu_compile.compile_model(onnx, opt_level=2, emit="nbin", bin_dir=bindir)
    assert data[:4] == b"NPUB"


def test_opt_levels_change_the_ir(tmp_path, npu_opt):
    bindir = _bindir(npu_opt)
    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)

    o0 = npu_compile.compile_model(onnx, opt_level=0, emit="npu", bin_dir=bindir)
    o2 = npu_compile.compile_model(onnx, opt_level=2, emit="npu", bin_dir=bindir)

    # O2 fuses the standalone relu activations into their producers.
    assert o0.count("npu.relu") > o2.count("npu.relu")


def test_driver_nbin_matches_onnxruntime(tmp_path, npu_opt):
    bindir = _bindir(npu_opt)
    npu_sim = bindir / "npu-sim"
    if not npu_sim.exists():
        pytest.skip("npu-sim not built")

    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    nbin = tmp_path / "lenet.nbin"
    npu_compile.compile_model(onnx, opt_level=2, emit="nbin", output=nbin, bin_dir=bindir)

    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 1, 28, 28)).astype(np.float32)
    reference = ort.InferenceSession(str(onnx)).run(None, {"input": x})[0]

    import subprocess

    in_path = tmp_path / "in.bin"
    in_path.write_bytes(x.tobytes())
    out_path = tmp_path / "out.bin"
    subprocess.run(
        [str(npu_sim), str(nbin), "--input", str(in_path), "--output", str(out_path)],
        check=True,
    )
    simulated = np.frombuffer(out_path.read_bytes(), dtype=np.float32).reshape(
        reference.shape
    )
    np.testing.assert_allclose(simulated, reference, rtol=1e-3, atol=1e-3)
