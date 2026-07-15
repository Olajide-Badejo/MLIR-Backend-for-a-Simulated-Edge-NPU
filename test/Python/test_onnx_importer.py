"""Tests for the ONNX to npu importer.

Each imported module is verified by running it through npu-opt, which applies the
real dialect verifiers. Op counts are read back from the pretty printed output.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter

import onnx
import pytest
from npu_frontend import model_generator, onnx_importer, op_mapping
from onnx import TensorProto, helper


def _verify(mlir_text: str, npu_opt: str) -> str:
    proc = subprocess.run([npu_opt], input=mlir_text, capture_output=True, text=True)
    assert proc.returncode == 0, f"npu-opt rejected the module:\n{proc.stderr}"
    return proc.stdout


def _op_counts(mlir_text: str) -> Counter:
    return Counter(re.findall(r"npu\.[a-z_0-9]+", mlir_text))


def test_import_lenet_verifies(tmp_path, npu_opt):
    path = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    text = onnx_importer.import_model(path)
    out = _verify(text, npu_opt)

    counts = _op_counts(out)
    assert counts["npu.conv2d"] == 2
    assert counts["npu.matmul"] == 3
    assert counts["npu.max_pool2d"] == 2
    assert counts["npu.relu"] == 4
    assert counts["npu.reshape"] == 1


def _single_node_model(path, node, inputs, outputs, initializers=()):
    graph = helper.make_graph([node], "g", inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def test_averagepool_and_reshape_import(tmp_path, npu_opt):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node(
        "AveragePool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2]
    )
    path = _single_node_model(tmp_path / "avg.onnx", node, [x], [y])
    out = _verify(onnx_importer.import_model(path), npu_opt)
    assert _op_counts(out)["npu.avg_pool2d"] == 1


def test_unsupported_op_fails_loudly(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Sigmoid", ["x"], ["y"])
    path = _single_node_model(tmp_path / "sig.onnx", node, [x], [y])

    with pytest.raises(op_mapping.UnsupportedOpError) as excinfo:
        onnx_importer.import_model(path)
    assert "Sigmoid" in str(excinfo.value)
