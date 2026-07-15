"""Structural validation of the generated test models."""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from npu_frontend import model_generator
from onnx import numpy_helper


def test_lenet_exports_and_checks(tmp_path):
    path = model_generator.export("lenet", tmp_path / "lenet.onnx", seed=0)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    counts = Counter(n.op_type for n in model.graph.node)
    assert counts["Conv"] == 2
    assert counts["Gemm"] == 3
    assert counts["MaxPool"] == 2
    assert counts["Relu"] == 4

    opset = {i.domain: i.version for i in model.opset_import}
    assert opset[""] == 17


def test_seed_is_deterministic(tmp_path):
    a = model_generator.export("lenet", tmp_path / "a.onnx", seed=0)
    b = model_generator.export("lenet", tmp_path / "b.onnx", seed=0)

    def first_conv_weight(path):
        model = onnx.load(str(path))
        init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        return init["conv1.weight"]

    np.testing.assert_array_equal(first_conv_weight(a), first_conv_weight(b))


def test_unknown_model_raises():
    import pytest

    with pytest.raises(KeyError):
        model_generator.build("does-not-exist")
