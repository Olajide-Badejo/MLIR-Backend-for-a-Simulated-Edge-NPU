# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Small hand built ONNX graphs, for testing one converter at a time.

Section 17.1 asks for a pytest per importer operator in isolation as well as a
suite model that uses it, and the two catch different things. A suite model
proves the converter works on the shape an exporter actually produces; a graph
of one node proves the converter's own rules, including the ones no model in the
suite triggers, which is where every rejection lives.

These are graphs rather than files. Nothing here writes an `.onnx`, so nothing
here can leave one behind, and `import_model` takes a `ModelProto` for exactly
this reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import onnx
from npu_frontend import PINNED_OPSET
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

# What the dynamo exporter writes, and what the model generator pins its hand
# built models to. Kept the same here so a fixture cannot pass for a reason the
# suite models would not.
IR_VERSION = 10


def initializer(name: str, array: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array), name)


def value(name: str, shape: Sequence[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def make_model(
    nodes: Iterable[NodeProto],
    inputs: Iterable[onnx.ValueInfoProto],
    outputs: Iterable[onnx.ValueInfoProto],
    initializers: Iterable[TensorProto] = (),
    *,
    opset: int = PINNED_OPSET,
    check: bool = True,
) -> ModelProto:
    """One graph, checked unless the fixture exists to be malformed.

    `check=False` is for the fixtures that test the importer's own refusals of
    things `onnx.checker` also refuses, where checking here would raise the
    wrong exception from the wrong place.
    """
    graph = helper.make_graph(
        list(nodes),
        "fixture",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="fixture"
    )
    model.ir_version = IR_VERSION
    if check:
        onnx.checker.check_model(model, full_check=True)
    return model


def conv_model(
    *,
    input_shape: Sequence[int] = (1, 3, 8, 8),
    output_shape: Sequence[int] = (1, 4, 8, 8),
    weight_shape: Sequence[int] = (4, 3, 3, 3),
    with_bias: bool = True,
    check: bool = True,
    **attributes: object,
) -> ModelProto:
    """A single `Conv`, with sensible defaults every attribute can override."""
    kwargs: dict[str, object] = {
        "kernel_shape": [3, 3],
        "strides": [1, 1],
        "pads": [1, 1, 1, 1],
        "dilations": [1, 1],
        "group": 1,
    }
    kwargs.update(attributes)
    # An attribute passed as None is removed rather than set. ONNX refuses a
    # Conv that carries both `auto_pad` and `pads`, so the auto_pad fixture has
    # to be able to take the default `pads` away again.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    rng = np.random.default_rng(7)
    initializers = [
        initializer("w", rng.standard_normal(tuple(weight_shape)).astype(np.float32))
    ]
    node_inputs = ["x", "w"]
    if with_bias:
        initializers.append(
            initializer("b", rng.standard_normal(output_shape[1]).astype(np.float32))
        )
        node_inputs.append("b")

    return make_model(
        [helper.make_node("Conv", node_inputs, ["y"], name="conv0", **kwargs)],
        [value("x", input_shape)],
        [value("y", output_shape)],
        initializers,
        check=check,
    )
