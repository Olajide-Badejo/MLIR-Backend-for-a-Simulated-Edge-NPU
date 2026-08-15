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


def test_cli_main_emits_npu(tmp_path, npu_opt, monkeypatch):
    bindir = _bindir(npu_opt)
    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    out = tmp_path / "lenet.npu.mlir"
    monkeypatch.setenv("NPU_BIN", str(bindir))
    rc = npu_compile.main(
        [str(onnx), "-O2", "--emit", "npu", "-o", str(out), "--verbose"]
    )
    assert rc == 0
    assert "npu.conv2d" in out.read_text()


def test_opt_levels_change_the_ir(tmp_path, npu_opt):
    bindir = _bindir(npu_opt)
    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)

    o0 = npu_compile.compile_model(onnx, opt_level=0, emit="npu", bin_dir=bindir)
    o2 = npu_compile.compile_model(onnx, opt_level=2, emit="npu", bin_dir=bindir)

    # O2 fuses the standalone relu activations into their producers.
    assert o0.count("npu.relu") > o2.count("npu.relu")


def test_npu_sim_takes_one_input_per_declared_region(tmp_path, npu_opt):
    """npu-sim needs one --input per declared input, and says so when it is not.

    It used to keep a single input path, so a two input program ran with its
    second input left as zeros and reported nothing about it.
    """
    import subprocess

    bindir = _bindir(npu_opt)
    npu_sim = bindir / "npu-sim"
    if not npu_sim.exists():
        pytest.skip("npu-sim not built")

    src = tmp_path / "two_inputs.mlir"
    src.write_text(
        "func.func @main(%a: tensor<1x1x2x2xf32>, %b: tensor<1x1x2x2xf32>)\n"
        "    -> tensor<1x1x2x2xf32> {\n"
        "  %0 = npu.add %a, %b : tensor<1x1x2x2xf32>\n"
        "  return %0 : tensor<1x1x2x2xf32>\n"
        "}\n"
    )
    lowered = tmp_path / "two_inputs.isa.mlir"
    subprocess.run(
        [
            str(npu_opt),
            str(src),
            "-npu-lower-to-npuisa",
            "-npu-allocate-scratchpad",
            "-o",
            str(lowered),
        ],
        check=True,
    )
    nbin = tmp_path / "two_inputs.nbin"
    subprocess.run(
        [str(bindir / "npu-translate"), str(lowered), "-o", str(nbin)], check=True
    )

    a = np.array([1, 2, 3, 4], dtype=np.float32).reshape(1, 1, 2, 2)
    b = np.array([10, 20, 30, 40], dtype=np.float32).reshape(1, 1, 2, 2)
    a_path = tmp_path / "a.bin"
    b_path = tmp_path / "b.bin"
    a_path.write_bytes(a.tobytes())
    b_path.write_bytes(b.tobytes())
    out_path = tmp_path / "out.bin"

    # Two flags for two declared inputs. The result proves the second input was
    # actually read: with it left as zeros the sum would be a alone.
    subprocess.run(
        [
            str(npu_sim),
            str(nbin),
            "--input",
            str(a_path),
            "--input",
            str(b_path),
            "--output",
            str(out_path),
        ],
        check=True,
    )
    got = np.frombuffer(out_path.read_bytes(), dtype=np.float32)
    np.testing.assert_allclose(got, (a + b).reshape(-1))

    # One flag for two declared inputs is the case that used to run quietly.
    refused = subprocess.run(
        [str(npu_sim), str(nbin), "--input", str(a_path), "--output", str(out_path)],
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "2 input region(s)" in refused.stderr
    assert "1 --input flag(s)" in refused.stderr


def test_driver_nbin_matches_onnxruntime(tmp_path, npu_opt):
    bindir = _bindir(npu_opt)
    npu_sim = bindir / "npu-sim"
    if not npu_sim.exists():
        pytest.skip("npu-sim not built")

    onnx = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    nbin = tmp_path / "lenet.nbin"
    npu_compile.compile_model(
        onnx, opt_level=2, emit="nbin", output=nbin, bin_dir=bindir
    )

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
