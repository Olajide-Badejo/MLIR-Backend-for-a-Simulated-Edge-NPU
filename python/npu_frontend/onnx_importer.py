# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""ONNX at the pinned opset to `npu` dialect IR on tensors.

The driver. It checks the model, runs shape inference, refuses a graph that is
not at the pinned opset, and then walks the nodes handing each to the converter
`op_mapping.py` registered for its operator type. What comes out is the text
`npu-opt` printed, so nothing leaves here unverified.

The order of the first three steps is not interchangeable. `onnx.checker` runs
first because a malformed graph makes everything after it meaningless; the opset
check runs before shape inference because inference resolves an operator's shape
rule at the opset the model declares, and inferring at an opset this project
does not implement produces shapes that are right for a semantics the converters
do not have.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import onnx
from onnx import GraphProto, ModelProto, numpy_helper

from .builder import ModuleBuilder, verify
from .context import ConversionContext
from .diagnostics import graph_error
from .op_mapping import CONVERTERS, DEFERRED

# Resolved by measurement at P0 and recorded in
# docs/adr/0002-onnx-opset-pin.md: the exporter binds it, at 23. A model
# declaring anything else is refused rather than imported and hoped about,
# because operator semantics move between opsets and a compiler that does not
# know which revision of AveragePool it is reading cannot be correct about
# padding.
PINNED_OPSET: Final[int] = 23

# The two spellings of the default ONNX domain. They mean the same thing and a
# model may use either.
_DEFAULT_DOMAINS: Final[frozenset[str]] = frozenset({"", "ai.onnx"})


# What onnx raises when a model is malformed. Caught and re-raised as this
# package's own exception so that a caller has one type to catch: a graph that
# onnx refuses and a graph this importer refuses are the same event from the
# outside, and a caller should not have to know which of the two layers spoke.
# The original message is kept in full rather than summarised, because it is
# usually more specific than anything this file could say about the same fault.
_ONNX_REFUSALS: Final[tuple[type[Exception], ...]] = (
    onnx.checker.ValidationError,
    onnx.shape_inference.InferenceError,
)


def _check_model(model: ModelProto) -> None:
    try:
        onnx.checker.check_model(model, full_check=True)
    except _ONNX_REFUSALS as exc:
        raise graph_error(
            f"onnx.checker refused this model before import: {exc}"
        ) from exc


def _infer_shapes(model: ModelProto) -> ModelProto:
    try:
        return onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
    except _ONNX_REFUSALS as exc:
        raise graph_error(f"onnx shape inference refused this model: {exc}") from exc


def _check_opset(model: ModelProto) -> None:
    default_versions = [
        entry.version
        for entry in model.opset_import
        if entry.domain in _DEFAULT_DOMAINS
    ]
    other_domains = sorted(
        {
            entry.domain
            for entry in model.opset_import
            if entry.domain not in _DEFAULT_DOMAINS
        }
    )
    if other_domains:
        raise graph_error(
            f"the model imports the operator domains {other_domains}. This "
            "project implements the default ONNX domain only, and an operator "
            "from a custom domain has no converter and no meaning here."
        )
    if len(default_versions) != 1:
        raise graph_error(
            f"the model declares {len(default_versions)} versions of the "
            "default ONNX operator set and exactly one is expected"
        )
    if default_versions[0] != PINNED_OPSET:
        raise graph_error(
            f"the model is at opset {default_versions[0]} and this project is "
            f"pinned at {PINNED_OPSET}. The pin was resolved by measurement on "
            "this toolchain and is recorded in "
            "docs/adr/0002-onnx-opset-pin.md; importing a different opset "
            "would mean reading one revision of an operator's semantics with a "
            "converter written against another."
        )


def _static_shape(
    name: str, type_proto: onnx.TypeProto, role: str, *, require_float: bool
) -> tuple[int, ...] | None:
    """The static shape of one value info, or None when it is not one to keep.

    `require_float` is True at the graph boundary and False for an intermediate,
    and the asymmetry is deliberate. A graph input or output is a function
    argument or result, so its element type is part of the signature this
    project has to be able to write down. An intermediate of another element
    type is not automatically a problem: the only one the supported operator set
    can produce is `MaxPool`'s optional `Indices` output, and the right thing to
    say about that is that this project does not produce indices, which is what
    the `MaxPool` converter says. Refusing it here instead would answer a
    question about pooling with a message about dtypes.
    """
    if not type_proto.HasField("tensor_type"):
        if not require_float:
            return None
        raise graph_error(
            f"the {role} {name!r} is not a tensor. This project compiles tensor "
            "graphs; sequences, maps and optionals have no representation here."
        )
    tensor_type = type_proto.tensor_type
    if tensor_type.elem_type != onnx.TensorProto.FLOAT:
        if not require_float:
            return None
        raise graph_error(
            f"the {role} {name!r} has element type "
            f"{onnx.TensorProto.DataType.Name(tensor_type.elem_type)}, and this "
            "project's tensors are f32. The integer types arrive with the "
            "quantization phase."
        )

    extents: list[int] = []
    for axis, dimension in enumerate(tensor_type.shape.dim):
        if dimension.HasField("dim_param") or not dimension.HasField("dim_value"):
            if not require_float:
                return None
            raise graph_error(
                f"the {role} {name!r} has a dynamic extent on axis {axis}. "
                "Section 0.3's scope is static shapes at an arbitrary but fixed "
                "batch size, and nothing below the npu dialect can represent a "
                "dynamic dimension at all."
            )
        extents.append(dimension.dim_value)
    return tuple(extents)


def _collect_shapes(graph: GraphProto) -> dict[str, tuple[int, ...]]:
    """Every name in the graph mapped to the shape ONNX inferred for it.

    Initializers are read from their own tensor rather than from a value info,
    because an initializer that is not also a graph input has no value info and
    the converters need its shape to decide whether a broadcast is the channel
    one.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for initializer in graph.initializer:
        shapes[initializer.name] = tuple(int(extent) for extent in initializer.dims)
    for value_info, role, strict in (
        *((info, "graph input", True) for info in graph.input),
        *((info, "graph output", True) for info in graph.output),
        *((info, "intermediate value", False) for info in graph.value_info),
    ):
        if value_info.name in shapes:
            continue
        shape = _static_shape(
            value_info.name, value_info.type, role, require_float=strict
        )
        if shape is not None:
            shapes[value_info.name] = shape
    return shapes


def _collect_initializers(graph: GraphProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }


def _name_every_node(graph: GraphProto) -> None:
    """Give every node a name, so every diagnostic and every location has one.

    Section 11 requires every operation the importer creates to carry a
    `NameLoc` with the ONNX node name, and an exporter is not obliged to name
    its nodes. A synthesised name is deterministic, derived from the operator
    type and the node's index in the graph, so two imports of the same model
    produce the same locations and a golden file over the IR is stable.
    """
    for index, node in enumerate(graph.node):
        if not node.name:
            node.name = f"{node.op_type}_{index}"


def import_model(model: ModelProto, *, function_name: str = "main") -> str:
    """Import one ONNX model and return the verified `npu` dialect IR as text.

    The model is not mutated: shape inference returns a copy and every later
    step works on that.
    """
    _check_model(model)
    _check_opset(model)
    inferred = _infer_shapes(model)

    graph = inferred.graph
    _name_every_node(graph)

    initializers = _collect_initializers(graph)
    shapes = _collect_shapes(graph)

    # An initializer that is also listed as a graph input is the pre IR version
    # 4 spelling of a constant. It is a constant either way, so it must not
    # become a function argument.
    argument_names = [
        info.name for info in graph.input if info.name not in initializers
    ]
    result_names = [info.name for info in graph.output]
    if not result_names:
        raise graph_error("the graph has no outputs, so there is nothing to compile")

    with ModuleBuilder(function_name) as builder:
        arguments = builder.begin_function(
            [shapes[name] for name in argument_names],
            [shapes[name] for name in result_names],
        )
        ctx = ConversionContext(builder, shapes, initializers)
        for name, value in zip(argument_names, arguments, strict=True):
            ctx.bind(name, value)

        for node in graph.node:
            converter = CONVERTERS.get(node.op_type)
            if converter is None:
                deferred = DEFERRED.get(node.op_type)
                reason = (
                    deferred
                    if deferred is not None
                    else (
                        "this operator has no converter. The supported set is "
                        + ", ".join(sorted(CONVERTERS))
                        + "."
                    )
                )
                raise graph_error(f"{node.op_type} node {node.name!r}: {reason}")
            converter(ctx, node)

        results = []
        for name in result_names:
            if not ctx.is_bound(name):
                raise graph_error(
                    f"the graph output {name!r} is not produced by any node. An "
                    "output that is also an initializer is a constant function "
                    "and is not what this project compiles."
                )
            results.append(ctx.value(graph.node[-1], name))
        builder.end_function(results)

        return verify(builder.printed())


def import_model_file(
    path: str | os.PathLike[str], *, function_name: str = "main"
) -> str:
    """Import an ONNX model from a file. A convenience over `import_model`."""
    return import_model(onnx.load(str(Path(path))), function_name=function_name)
