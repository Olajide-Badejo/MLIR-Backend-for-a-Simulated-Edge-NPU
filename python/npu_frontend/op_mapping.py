"""Map supported ONNX operators to npu dialect operations.

Each ONNX node is translated to one or more npu ops built with the MLIR Python
bindings. The npu dialect is not registered in the Python context, so ops are
created in generic form; the authoritative verification is done by running the
result through npu-opt, which has the dialect and its verifiers. Any ONNX op that
is not in SUPPORTED_OPS fails loudly, naming the op and node, rather than emitting
silently wrong IR.

Supported ops (opset 17): Conv, Gemm, MatMul, Add, Relu, MaxPool, AveragePool,
BatchNormalization, Reshape, Flatten, GlobalAveragePool, Concat, Clip. The subset
needed by the core LeNet suite is implemented first; the rest raise a clear
"not yet implemented" until their phase.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mlir import ir


class UnsupportedOpError(Exception):
    """Raised when an ONNX op cannot be imported. Never emit wrong IR instead."""


# ---------------------------------------------------------------------------
# Type and attribute helpers
# ---------------------------------------------------------------------------


def f32() -> ir.Type:
    return ir.F32Type.get()


def i32() -> ir.Type:
    return ir.IntegerType.get_signless(32)


def i64() -> ir.Type:
    return ir.IntegerType.get_signless(64)


def tensor_type(shape) -> ir.RankedTensorType:
    dims = [int(d) for d in shape]
    if any(d <= 0 for d in dims):
        raise UnsupportedOpError(f"dynamic or unknown dimension in shape {shape}")
    return ir.RankedTensorType.get(dims, f32())


def i64_array(values) -> ir.ArrayAttr:
    return ir.ArrayAttr.get([ir.IntegerAttr.get(i64(), int(v)) for v in values])


def i64_attr(v) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(i64(), int(v))


def activation_attr(kind: int = 0) -> ir.IntegerAttr:
    # 0 = none, 1 = relu, matching the ODS I32EnumAttr storage.
    return ir.IntegerAttr.get(i32(), kind)


def _create(name: str, operands, result_types, attributes) -> ir.Value:
    op = ir.Operation.create(
        name,
        results=result_types,
        operands=list(operands),
        attributes=attributes,
    )
    return op.result


def make_constant(array: np.ndarray) -> ir.Value:
    """Materialize a numpy array as an npu.constant value."""
    arr = np.ascontiguousarray(array.astype(np.float32))
    ttype = ir.RankedTensorType.get(list(arr.shape), f32())
    value = ir.DenseElementsAttr.get(arr, type=ttype)
    return _create("npu.constant", [], [ttype], {"value": value})


# ---------------------------------------------------------------------------
# ONNX attribute access
# ---------------------------------------------------------------------------


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.HasField("i"):
                return a.i
            if a.HasField("f"):
                return a.f
    return default


# ---------------------------------------------------------------------------
# Per op converters. Each takes (node, ctx) and updates ctx.values.
# ctx exposes: values(name->Value), shape(name)->tuple, init(name)->np.ndarray or None
# ---------------------------------------------------------------------------


def _conv(node, ctx):
    x = ctx.values[node.input[0]]
    w = ctx.values[node.input[1]]
    operands = [x, w]
    if len(node.input) > 2 and node.input[2]:
        operands.append(ctx.values[node.input[2]])
    out_t = tensor_type(ctx.shape(node.output[0]))
    attrs = {
        "strides": i64_array(_attr(node, "strides", [1, 1])),
        "pads": i64_array(_attr(node, "pads", [0, 0, 0, 0])),
        "dilations": i64_array(_attr(node, "dilations", [1, 1])),
        "group": i64_attr(_attr(node, "group", 1)),
        "activation": activation_attr(0),
    }
    ctx.values[node.output[0]] = _create("npu.conv2d", operands, [out_t], attrs)


def _relu(node, ctx):
    x = ctx.values[node.input[0]]
    out_t = tensor_type(ctx.shape(node.output[0]))
    ctx.values[node.output[0]] = _create("npu.relu", [x], [out_t], {})


def _maxpool(node, ctx):
    x = ctx.values[node.input[0]]
    out_t = tensor_type(ctx.shape(node.output[0]))
    kernel = _attr(node, "kernel_shape")
    if kernel is None:
        raise UnsupportedOpError("MaxPool without kernel_shape")
    attrs = {
        "kernel_shape": i64_array(kernel),
        "strides": i64_array(_attr(node, "strides", kernel)),
        "pads": i64_array(_attr(node, "pads", [0, 0, 0, 0])),
    }
    ctx.values[node.output[0]] = _create("npu.max_pool2d", [x], [out_t], attrs)


def _averagepool(node, ctx):
    x = ctx.values[node.input[0]]
    out_t = tensor_type(ctx.shape(node.output[0]))
    kernel = _attr(node, "kernel_shape")
    if kernel is None:
        raise UnsupportedOpError("AveragePool without kernel_shape")
    attrs = {
        "kernel_shape": i64_array(kernel),
        "strides": i64_array(_attr(node, "strides", kernel)),
        "pads": i64_array(_attr(node, "pads", [0, 0, 0, 0])),
    }
    ctx.values[node.output[0]] = _create("npu.avg_pool2d", [x], [out_t], attrs)


def _gemm(node, ctx):
    # Y = alpha * (A' * B') + beta * C, with optional transpose of A and B.
    alpha = _attr(node, "alpha", 1.0)
    beta = _attr(node, "beta", 1.0)
    trans_a = _attr(node, "transA", 0)
    trans_b = _attr(node, "transB", 0)
    if float(alpha) != 1.0 or float(beta) != 1.0 or trans_a:
        raise UnsupportedOpError(
            "Gemm with alpha or beta not 1 or transA set is not supported"
        )
    a = ctx.values[node.input[0]]
    # B is a constant weight; if transB is set, transpose the constant at compile
    # time so the npu.matmul (which does a plain lhs * rhs) is correct.
    b_name = node.input[1]
    b_array = ctx.init(b_name)
    if b_array is None:
        raise UnsupportedOpError("Gemm with a non constant B is not supported")
    if trans_b:
        b_value = make_constant(np.ascontiguousarray(b_array.T))
    else:
        b_value = ctx.values[b_name]
    operands = [a, b_value]
    if len(node.input) > 2 and node.input[2]:
        operands.append(ctx.values[node.input[2]])
    out_t = tensor_type(ctx.shape(node.output[0]))
    ctx.values[node.output[0]] = _create(
        "npu.matmul", operands, [out_t], {"activation": activation_attr(0)}
    )


def _reshape(node, ctx):
    x = ctx.values[node.input[0]]
    out_t = tensor_type(ctx.shape(node.output[0]))
    ctx.values[node.output[0]] = _create("npu.reshape", [x], [out_t], {})


def _flatten(node, ctx):
    # Flatten collapses the dimensions from axis onward; the imported result type
    # already carries the flattened shape, so a reshape expresses it.
    _reshape(node, ctx)


CONVERTERS: dict[str, Callable] = {
    "Conv": _conv,
    "Relu": _relu,
    "MaxPool": _maxpool,
    "AveragePool": _averagepool,
    "Gemm": _gemm,
    "Reshape": _reshape,
    "Flatten": _flatten,
}

SUPPORTED_OPS = frozenset(CONVERTERS)


def convert_node(node, ctx) -> None:
    converter = CONVERTERS.get(node.op_type)
    if converter is None:
        raise UnsupportedOpError(
            f"unsupported ONNX op {node.op_type!r} at node {node.name or node.output[0]!r}; "
            f"supported: {sorted(SUPPORTED_OPS)}"
        )
    converter(node, ctx)
