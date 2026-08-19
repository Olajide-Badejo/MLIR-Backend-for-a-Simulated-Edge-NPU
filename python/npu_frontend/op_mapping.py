# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The converter registry: one ONNX operator to one `npu` operation.

Sixteen converters, at the pinned opset 23. The list below is the contract, and
`test_onnx_importer.py` asserts it agrees with `CONVERTERS` exactly, in both
directions, so a converter added without a line here fails a test rather than
going undocumented, and a line here for a converter that does not exist fails
the same test.

- ``Add`` to ``npu.add``, through the shared broadcasting policy.
- ``AveragePool`` to ``npu.avg_pool2d``.
- ``BatchNormalization`` to ``npu.batch_norm``, inference form only.
- ``Clip`` to ``npu.relu``, and only for the bounds that are a relu.
- ``Concat`` to ``npu.concat``, with the axis normalised to non negative.
- ``Conv`` to ``npu.conv2d``, rank 4, grouped, with an optional bias operand.
- ``Flatten`` to ``npu.reshape``, batch preserving.
- ``Gemm`` to ``npu.matmul``, with an optional bias operand.
- ``GlobalAveragePool`` to ``npu.avg_pool2d`` with a full extent kernel.
- ``Identity`` to no operation at all; it binds its output name to its input.
- ``MatMul`` to ``npu.matmul``, rank 2 by rank 2.
- ``MaxPool`` to ``npu.max_pool2d``.
- ``Mul`` to ``npu.mul``, through the shared broadcasting policy.
- ``Relu`` to ``npu.relu``.
- ``Reshape`` to ``npu.reshape``, batch preserving when it flattens.
- ``Transpose`` to ``npu.transpose``.

Everything else is refused by name. `DEFERRED` below carries the operators that
are refused with a reason more specific than "unsupported", because a message
naming the phase that will bring an operator is worth more than one that reads
as if the operator will never exist.

The rules in here are Section 11's, and where a rule looks arbitrary it is
usually load bearing. The three worth reading before editing anything are the
broadcasting policy and its channel carve out, the `Clip` policy, and the batch
preserving rule on `Reshape` and `Flatten`, which Section 11 calls the most
likely hidden bug in the whole frontend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Final

import numpy as np
from mlir import ir
from onnx import NodeProto

from .attributes import (
    get_float,
    get_int,
    get_ints,
    get_string,
    require_int,
    require_ints,
)
from .context import ConversionContext
from .diagnostics import node_error

Converter = Callable[[ConversionContext, NodeProto], None]

# Operators this project will support and does not yet, refused with the phase
# that brings them rather than with a generic message.
DEFERRED: Final[dict[str, str]] = {
    "QuantizeLinear": (
        "the quantization pair arrives with its converters, its integer "
        "kernels and its calibrated models together, at the quantization "
        "phase. The npu dialect has no quantize operation to import it to, so "
        "accepting it here would mean emitting something nothing can lower, "
        "encode or simulate."
    ),
    "DequantizeLinear": (
        "the quantization pair arrives with its converters, its integer "
        "kernels and its calibrated models together, at the quantization "
        "phase. The npu dialect has no dequantize operation to import it to."
    ),
    "Pad": (
        "Pad is not in this project's operator set and is not planned. Its "
        "opset 18 optional axes input changes the length of pads from twice "
        "the rank to twice the axis count, and opset 19 added a wrap mode, so "
        "a converter that guessed would misread the padding rather than fail. "
        "The dilated stack model reaches asymmetric padding through the pads "
        "attribute of Conv, which is why that model is built with the ONNX "
        "construction API rather than exported from PyTorch."
    ),
}


# =============================================================================
# Shared helpers.
# =============================================================================


def _reject_auto_pad(node: NodeProto) -> None:
    """`auto_pad` is NOTSET or the node is refused.

    SAME_UPPER, SAME_LOWER and VALID are all expressible as explicit pads, and
    an importer that resolved them would be reimplementing a padding rule that
    the exporter already resolved for every model in this suite. Refusing is the
    safe minimum Section 11 asks for; a wrong resolution is a shape that
    typechecks and computes the wrong thing.
    """
    auto_pad = get_string(node, "auto_pad", "NOTSET")
    if auto_pad != "NOTSET":
        raise node_error(
            node,
            f"auto_pad is {auto_pad!r} and this importer implements NOTSET "
            "only. Re-export with explicit pads: an importer that resolved the "
            "padding itself would be a second implementation of a rule that "
            "has to agree with the runtime exactly.",
        )


def _spatial(node: NodeProto, name: str, values: list[int], expected: int) -> list[int]:
    if len(values) != expected:
        raise node_error(
            node,
            f"attribute {name!r} must have {expected} entries for a two "
            f"dimensional operation, but it has {len(values)}: {values}. This "
            "project implements rank 4 activations and nothing else.",
        )
    return values


def _windowed_attributes(
    node: NodeProto, *, allow_dilations: bool
) -> tuple[list[int], list[int], list[int]]:
    """`strides`, `pads` and `dilations` for a windowed operation.

    `pads` is in ONNX order throughout, `[padTop, padLeft, padBottom, padRight]`,
    which is the order the `npu` operations declare and the order
    `NPUShapeUtils.cpp` reads. Keeping one order end to end is what stops the
    asymmetric padding case from being right on one side and transposed on the
    other.
    """
    strides = _spatial(node, "strides", get_ints(node, "strides", [1, 1]), 2)
    pads = _spatial(node, "pads", get_ints(node, "pads", [0, 0, 0, 0]), 4)
    dilations = _spatial(node, "dilations", get_ints(node, "dilations", [1, 1]), 2)
    if not allow_dilations and dilations != [1, 1]:
        raise node_error(
            node,
            f"dilations is {dilations}. AveragePool gained dilations at opset "
            "19 and the output shape formula gained the dilation term with it; "
            "this importer refuses a non default value rather than ignoring "
            "the attribute, because an operator whose specification changed "
            "and whose converter did not is a silent wrong answer.",
        )
    return strides, pads, dilations


def _channel_broadcast_length(
    initializer_shape: tuple[int, ...], result_shape: tuple[int, ...]
) -> int | None:
    """The channel length of a per channel initializer, or None.

    Section 11's carve out, and the one broadcast the importer does not expand.
    A constant that broadcasts against a rank 4 activation over the channel axis
    is emitted as a rank 1 `npu.constant` of length C, because `-npu-fuse-bias`
    guards on a channel shaped constant addend and an expanded one makes the
    pass structurally unfireable on every model in the suite.

    **The shapes matched here are the two an ONNX per channel constant can
    have, and the literally rank 1 one is deliberately not among them.** ONNX
    broadcasting aligns from the trailing axis, so an initializer of dims `[C]`
    against an `N x C x H x W` activation broadcasts over the **width**, not
    over the channels, and matching it here would import a per column constant
    as a per channel one on any model where the two happen to be the same
    length. The shapes that do broadcast over the channel axis are `[C, 1, 1]`
    and `[1, C, 1, 1]`, and `[1, C, 1, 1]` is what the dynamo exporter writes
    for a scale or a bias spelled `p.reshape(1, -1, 1, 1)` in PyTorch. So the
    carve out is defined by what the constant broadcasts as, not by the rank it
    was stored with, and rank 1 is what comes out rather than what goes in.

    `(1, 1, 1, 1)` against a single channel result is both a scalar broadcast
    and a channel one. The tie goes to the carve out, deterministically, and it
    is harmless: the values are identical and the rank 1 form is the one the
    fusion pass matches.
    """
    if len(result_shape) != 4:
        return None
    channels = result_shape[1]
    if initializer_shape in {(channels, 1, 1), (1, channels, 1, 1)}:
        return channels
    return None


def _binary_operands(
    ctx: ConversionContext, node: NodeProto, result_shape: tuple[int, ...]
) -> tuple[ir.Value, ir.Value]:
    """The broadcasting policy of Section 11, stated once, for Add and Mul.

    Three cases, in this order.

    1. Both operands already have the result shape. Nothing to do.
    2. One operand is an initializer that broadcasts against the other. If it
       broadcasts purely over the channel axis it stays rank 1, as the carve out
       above; otherwise it is materialised at import time as a same shaped
       constant, so the emitted IR needs no broadcasting operation.
    3. Anything else raises, naming the node and both shapes.

    The carve out puts the rank 1 operand on the right, commuting the node when
    it arrived on the left. `npu.add` and `npu.mul` accept a rank 1 rhs and
    refuse a rank 1 lhs on purpose, so a channel broadcast has one spelling and
    `-npu-fuse-bias` has one form to match. Both operations are commutative, so
    the commutation is not observable in the result.
    """
    names = (node.input[0], node.input[1])
    shapes = tuple(ctx.shape(node, name) for name in names)

    if shapes[0] == result_shape and shapes[1] == result_shape:
        return ctx.value(node, names[0]), ctx.value(node, names[1])

    # The carve out, tried on the right operand first so a graph that already
    # has the canonical order keeps it.
    for index in (1, 0):
        other = 1 - index
        if shapes[other] != result_shape:
            continue
        array = ctx.initializer(names[index])
        if array is None:
            continue
        channels = _channel_broadcast_length(shapes[index], result_shape)
        if channels is None:
            continue
        rank_one = ctx.builder.constant(
            np.reshape(np.asarray(array, dtype=np.float32), (channels,)), names[index]
        )
        return ctx.value(node, names[other]), rank_one

    resolved: list[ir.Value] = []
    for index in (0, 1):
        if shapes[index] == result_shape:
            resolved.append(ctx.value(node, names[index]))
            continue
        array = ctx.initializer(names[index])
        if array is None:
            raise node_error(
                node,
                f"operand {names[index]!r} has shape {list(shapes[index])} "
                f"against a result of {list(result_shape)}, and it is computed "
                "at run time rather than being an initializer. This project "
                "materialises a broadcast at import time or refuses it; it does "
                "not emit a broadcasting operation, because the npu dialect has "
                "one broadcast and it is the rank 1 channel one.",
            )
        try:
            expanded = np.broadcast_to(
                np.asarray(array, dtype=np.float32), result_shape
            )
        except ValueError as exc:
            raise node_error(
                node,
                f"operand {names[index]!r} has shape {list(shapes[index])}, "
                f"which does not broadcast against the result shape "
                f"{list(result_shape)}: {exc}",
            ) from exc
        resolved.append(ctx.builder.constant(expanded, names[index]))
    return resolved[0], resolved[1]


def _normalise_axis(node: NodeProto, axis: int, rank: int, name: str) -> int:
    """A possibly negative ONNX axis as a non negative one.

    Normalised here rather than carried into the IR, because `npu.concat`
    requires a non negative axis: an axis convention every consumer has to
    normalise again is a convention that one of them eventually forgets.
    """
    normalised = axis + rank if axis < 0 else axis
    if not 0 <= normalised < rank:
        raise node_error(
            node,
            f"{name} is {axis}, which is out of range for a rank {rank} tensor",
        )
    return normalised


# =============================================================================
# The converters.
# =============================================================================


def convert_conv(ctx: ConversionContext, node: NodeProto) -> None:
    _reject_auto_pad(node)
    input_name = node.input[0]
    input_shape = ctx.shape(node, input_name)
    if len(input_shape) != 4:
        raise node_error(
            node,
            f"the input has rank {len(input_shape)} and this project implements "
            "two dimensional convolution over rank 4 activations only",
        )

    filter_array = ctx.require_initializer(node, node.input[1], "filter")
    if filter_array.ndim != 4:
        raise node_error(
            node,
            f"the filter has rank {filter_array.ndim} and must be rank 4, as "
            "(outputChannels, inputChannels / group, kernelHeight, kernelWidth)",
        )

    kernel_shape = get_ints(node, "kernel_shape", list(filter_array.shape[2:]))
    if list(kernel_shape) != list(filter_array.shape[2:]):
        raise node_error(
            node,
            f"kernel_shape is {list(kernel_shape)} but the filter's spatial "
            f"extents are {list(filter_array.shape[2:])}. The two disagree, and "
            "guessing which one is right is how a convolution ends up reading "
            "the wrong window.",
        )

    strides, pads, dilations = _windowed_attributes(node, allow_dilations=True)
    group = get_int(node, "group", 1)

    operands = [ctx.value(node, input_name), ctx.value(node, node.input[1])]
    if len(node.input) >= 3 and node.input[2]:
        bias_array = ctx.require_initializer(node, node.input[2], "bias")
        if bias_array.ndim != 1:
            raise node_error(
                node,
                f"the bias has rank {bias_array.ndim} and must be rank 1 of "
                "length equal to the output channel count",
            )
        operands.append(ctx.value(node, node.input[2]))

    result_shape = ctx.shape(node, node.output[0])
    builder = ctx.builder
    ctx.bind(
        node.output[0],
        builder.compute(
            "conv2d",
            inputs=operands,
            result_shape=result_shape,
            attributes={
                "strides": builder.i64_array(strides),
                "pads": builder.i64_array(pads),
                "dilations": builder.i64_array(dilations),
                "group": builder.i64(group),
            },
            name=node.name,
        ),
    )


def _matmul_like(
    ctx: ConversionContext,
    node: NodeProto,
    lhs: ir.Value,
    rhs: ir.Value,
    bias: ir.Value | None,
) -> None:
    result_shape = ctx.shape(node, node.output[0])
    if len(result_shape) != 2:
        raise node_error(
            node,
            f"the result has rank {len(result_shape)} and npu.matmul is rank 2 "
            "by rank 2. A batched matmul is not in this project's operator set.",
        )
    inputs = [lhs, rhs] if bias is None else [lhs, rhs, bias]
    ctx.bind(
        node.output[0],
        ctx.builder.compute(
            "matmul", inputs=inputs, result_shape=result_shape, name=node.name
        ),
    )


def convert_matmul(ctx: ConversionContext, node: NodeProto) -> None:
    lhs_shape = ctx.shape(node, node.input[0])
    rhs_shape = ctx.shape(node, node.input[1])
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise node_error(
            node,
            f"MatMul is rank 2 by rank 2 here, but the operands have ranks "
            f"{len(lhs_shape)} and {len(rhs_shape)}. Rank 1 promotion and the "
            "batched forms are both refused rather than reshaped, because a "
            "reshape invented by the importer is a shape nobody wrote down.",
        )
    _matmul_like(
        ctx, node, ctx.value(node, node.input[0]), ctx.value(node, node.input[1]), None
    )


def convert_gemm(ctx: ConversionContext, node: NodeProto) -> None:
    alpha = get_float(node, "alpha", 1.0)
    beta = get_float(node, "beta", 1.0)
    if alpha != 1.0 or beta != 1.0:
        raise node_error(
            node,
            f"alpha is {alpha} and beta is {beta}, and this importer implements "
            "1.0 for both. Folding a scale into the weights at import would "
            "change the constants the report publishes for a reason invisible "
            "in the model.",
        )

    if get_int(node, "transA", 0) != 0:
        raise node_error(
            node,
            "transA is 1, which would need a transpose of the activation. That "
            "is a real operation with a real cost and this importer will not "
            "insert one silently.",
        )

    lhs = ctx.value(node, node.input[0])
    rhs_name = node.input[1]
    if get_int(node, "transB", 0) != 0:
        # The shape nn.Linear exports as. The weight is a constant, so the
        # transpose happens once here rather than on every inference, and the
        # emitted IR has no transpose in it at all.
        weights = ctx.require_initializer(node, rhs_name, "B operand with transB set")
        if weights.ndim != 2:
            raise node_error(
                node,
                f"the B operand has rank {weights.ndim} and Gemm is rank 2 by "
                "rank 2 here",
            )
        rhs = ctx.builder.constant(np.ascontiguousarray(weights.T), rhs_name)
    else:
        rhs = ctx.value(node, rhs_name)

    bias: ir.Value | None = None
    if len(node.input) >= 3 and node.input[2]:
        bias_array = ctx.require_initializer(node, node.input[2], "C operand")
        if bias_array.ndim != 1:
            raise node_error(
                node,
                f"the C operand has rank {bias_array.ndim} and npu.matmul's "
                "bias is rank 1 of length equal to the output column count. A "
                "rank 2 C is a general broadcast add and belongs in its own "
                "node.",
            )
        bias = ctx.value(node, node.input[2])

    _matmul_like(ctx, node, lhs, rhs, bias)


def _elementwise_binary(ctx: ConversionContext, node: NodeProto, mnemonic: str) -> None:
    result_shape = ctx.shape(node, node.output[0])
    lhs, rhs = _binary_operands(ctx, node, result_shape)
    ctx.bind(
        node.output[0],
        ctx.builder.compute(
            mnemonic, inputs=[lhs, rhs], result_shape=result_shape, name=node.name
        ),
    )


def convert_add(ctx: ConversionContext, node: NodeProto) -> None:
    _elementwise_binary(ctx, node, "add")


def convert_mul(ctx: ConversionContext, node: NodeProto) -> None:
    _elementwise_binary(ctx, node, "mul")


def convert_relu(ctx: ConversionContext, node: NodeProto) -> None:
    result_shape = ctx.shape(node, node.output[0])
    ctx.bind(
        node.output[0],
        ctx.builder.compute(
            "relu",
            inputs=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            name=node.name,
        ),
    )


def convert_clip(ctx: ConversionContext, node: NodeProto) -> None:
    """`Clip` is a relu or it is refused, and the diagnostic quotes the bounds.

    Since opset 11 the bounds are optional inputs rather than attributes, so
    they are read from the initializer list, and an input name that is present
    but empty means the bound was omitted. A lower bound of 0 with an absent or
    infinite upper bound is `npu.relu`. Anything else raises.

    There is deliberately no relu6 case. A general bounded activation belongs in
    the `fused_op` region, and a third enum case at this level would be a
    migration to undo later.
    """

    def bound(index: int, role: str) -> float | None:
        if len(node.input) <= index or not node.input[index]:
            return None
        array = ctx.require_initializer(node, node.input[index], f"{role} bound")
        if array.size == 0:
            return None
        if array.size != 1:
            raise node_error(
                node,
                f"the {role} bound has {array.size} elements and Clip's bounds "
                "are scalars here",
            )
        return float(array.reshape(-1)[0])

    minimum = bound(1, "minimum")
    maximum = bound(2, "maximum")

    is_relu = minimum == 0.0 and (maximum is None or math.isinf(maximum))
    if not is_relu:
        raise node_error(
            node,
            f"the bounds are minimum {minimum} and maximum {maximum}. This "
            "importer accepts a lower bound of 0 with an absent or infinite "
            "upper bound, which is npu.relu, and refuses every other pair. A "
            "general bounded activation belongs in a fused_op region rather "
            "than in a third activation enum case.",
        )

    result_shape = ctx.shape(node, node.output[0])
    ctx.bind(
        node.output[0],
        ctx.builder.compute(
            "relu",
            inputs=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            name=node.name,
        ),
    )


def convert_identity(ctx: ConversionContext, node: NodeProto) -> None:
    """`Identity` emits no operation; it binds its output to its input.

    Exporters and graph transforms leave `Identity` nodes behind routinely, and
    an importer that rejected them would reject otherwise valid models for no
    reason. Binding rather than emitting means a chain of them costs nothing and
    the operation that follows sees the original value.
    """
    ctx.bind(node.output[0], ctx.value(node, node.input[0]))


def _pool(
    ctx: ConversionContext,
    node: NodeProto,
    mnemonic: str,
    kernel: list[int],
    strides: list[int],
    pads: list[int],
    dilations: list[int],
    ceil_mode: int,
) -> None:
    result_shape = ctx.shape(node, node.output[0])
    builder = ctx.builder
    ctx.bind(
        node.output[0],
        builder.compute(
            mnemonic,
            inputs=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            attributes={
                "kernel": builder.i64_array(kernel),
                "strides": builder.i64_array(strides),
                "pads": builder.i64_array(pads),
                "dilations": builder.i64_array(dilations),
                "ceil_mode": builder.i64(ceil_mode),
            },
            name=node.name,
        ),
    )


def convert_max_pool(ctx: ConversionContext, node: NodeProto) -> None:
    _reject_auto_pad(node)
    if len(node.output) > 1 and node.output[1]:
        raise node_error(
            node,
            "the second output, Indices, is requested. This project's pooling "
            "produces values only, and an importer that dropped the indices "
            "output would silently change what the graph computes.",
        )
    if get_int(node, "storage_order", 0) != 0:
        raise node_error(
            node,
            "storage_order is 1, meaning column major indices. It is only "
            "meaningful for the Indices output, which this importer refuses "
            "anyway, so it is refused rather than ignored.",
        )
    kernel = _spatial(node, "kernel_shape", require_ints(node, "kernel_shape"), 2)
    strides, pads, dilations = _windowed_attributes(node, allow_dilations=True)
    _pool(
        ctx,
        node,
        "max_pool2d",
        kernel,
        strides,
        pads,
        dilations,
        get_int(node, "ceil_mode", 0),
    )


def convert_average_pool(ctx: ConversionContext, node: NodeProto) -> None:
    """`AveragePool`, with the two attributes Section 11 refuses.

    `dilations` is refused outright: it arrived at opset 19 along with a change
    to the output shape formula, and a converter that ignored it would be an
    operator whose specification moved and whose implementation did not.

    `count_include_pad = 1` is refused **when any pad is non zero**, and the
    condition is the whole content of the rule rather than a softening of it.
    The kernel this project simulates divides by the number of elements that
    actually contributed, which is the `count_include_pad = 0` behaviour, so the
    two settings disagree exactly when some window overlaps the padded region.
    With every pad at zero no window ever does, the two produce bit identical
    results, and refusing would reject a graph that is provably fine.

    That case is not hypothetical: the dynamo exporter writes
    `count_include_pad = 1` on every `AveragePool` it emits, pads or no pads, so
    an unconditional refusal would make every average pool this suite exports
    unimportable while proving nothing.
    """
    _reject_auto_pad(node)
    kernel = _spatial(node, "kernel_shape", require_ints(node, "kernel_shape"), 2)
    strides, pads, dilations = _windowed_attributes(node, allow_dilations=False)

    if get_int(node, "count_include_pad", 0) != 0 and any(pad != 0 for pad in pads):
        raise node_error(
            node,
            f"count_include_pad is 1 with pads {pads}. This project's average "
            "pool divides by the number of elements that contributed, which is "
            "the count_include_pad = 0 behaviour, and the two disagree exactly "
            "when a window overlaps the padding. Silently ignoring the "
            "attribute is what makes an importer disagree with ONNX.",
        )

    _pool(
        ctx,
        node,
        "avg_pool2d",
        kernel,
        strides,
        pads,
        dilations,
        get_int(node, "ceil_mode", 0),
    )


def convert_global_average_pool(ctx: ConversionContext, node: NodeProto) -> None:
    """`GlobalAveragePool` as an `npu.avg_pool2d` over the whole spatial extent.

    There is no separate operation for it in the dialect and there should not
    be: a pool whose kernel is the input's spatial extent, with unit stride and
    no padding, computes exactly this and reuses the shape arithmetic, the
    lowering, the encoding and the kernel that the general pool already has.
    """
    input_shape = ctx.shape(node, node.input[0])
    if len(input_shape) != 4:
        raise node_error(
            node,
            f"the input has rank {len(input_shape)} and this project pools rank "
            "4 activations only",
        )
    kernel = [input_shape[2], input_shape[3]]
    _pool(ctx, node, "avg_pool2d", kernel, [1, 1], [0, 0, 0, 0], [1, 1], 0)


def convert_batch_normalization(ctx: ConversionContext, node: NodeProto) -> None:
    """Inference form batch normalization, with the training form refused.

    Both refusals happen here rather than at lowering, because the earliest
    layer that can name the problem is the one that should: at this level the
    ONNX node name is still available, and three passes later it is a diagnostic
    about an operand of an operation nobody can trace back to a model.
    """
    training_mode = get_int(node, "training_mode", 0)
    if len(node.output) > 1 or training_mode != 0:
        # One check rather than two, because ONNX ties the two facts together:
        # the output count must be 3 when training_mode is 1 and 1 when it is 0,
        # so a graph that reaches here can never have exactly one of them wrong.
        # Two checks would mean one of them was unreachable.
        raise node_error(
            node,
            f"this node has {len(node.output)} outputs and training_mode "
            f"{training_mode}, which is the training form. This project "
            "compiles inference graphs, the two forms compute different things "
            "from the same inputs, and the running mean and variance outputs "
            "have no meaning in an inference graph.",
        )

    roles = ("scale", "B", "input_mean", "input_var")
    parameters = []
    for index, role in enumerate(roles, start=1):
        array = ctx.require_initializer(node, node.input[index], role)
        if array.ndim != 1:
            raise node_error(
                node,
                f"the {role} parameter has rank {array.ndim} and all four "
                "parameters are rank 1 of length equal to the channel extent",
            )
        parameters.append(ctx.value(node, node.input[index]))

    result_shape = ctx.shape(node, node.output[0])
    builder = ctx.builder
    ctx.bind(
        node.output[0],
        builder.compute(
            "batch_norm",
            inputs=[ctx.value(node, node.input[0]), *parameters],
            result_shape=result_shape,
            attributes={"epsilon": builder.f32_attr(get_float(node, "epsilon", 1e-5))},
            name=node.name,
        ),
    )


def _check_batch_preserved(
    node: NodeProto, input_shape: tuple[int, ...], result_shape: tuple[int, ...]
) -> None:
    """Refuse a flatten that folds the batch away.

    Section 11 calls this the most likely hidden bug in the whole frontend, and
    it is worth saying why it is invisible without a check. A rank 4 activation
    of `(N, C, H, W)` reshaped to `(1, N * C * H * W)` typechecks, feeds a
    `matmul` that typechecks, and produces one enormous row that is arithmetic
    nobody asked for. Every shape in the graph is consistent; only the meaning
    is wrong.

    So a reshape from rank 4 to rank 2 must keep dimension 0. That is a narrow
    rule on purpose: it fires on the flatten in front of a classifier, which is
    where the bug lives, and it says nothing about reshapes of other ranks.
    """
    if len(input_shape) == 4 and len(result_shape) == 2:
        if result_shape[0] != input_shape[0]:
            raise node_error(
                node,
                f"this reshapes {list(input_shape)} to {list(result_shape)}, "
                f"which folds the batch of {input_shape[0]} into the feature "
                "axis. A flatten in front of a classifier produces "
                "(N, features) and never (1, N * features): the second "
                "typechecks all the way through and computes one enormous row.",
            )


def convert_reshape(ctx: ConversionContext, node: NodeProto) -> None:
    """`Reshape`, with `allowzero` handled rather than assumed.

    `allowzero` arrived at opset 14, which is before this project's floor, so it
    is not part of the opset 23 migration and the importer has to handle it
    already. At 0, which is the default, a zero in the target shape copies the
    corresponding input extent. At 1 a zero means a literal zero extent, which
    is an empty tensor and is not representable here, so the two are
    distinguished rather than conflated.

    The dynamo exporter writes `allowzero = 1` on every `Reshape` it emits, and
    with no zero in the target shape the two settings are identical, so the
    refusal is conditioned on a zero actually being there.
    """
    input_shape = ctx.shape(node, node.input[0])
    target = ctx.require_initializer(node, node.input[1], "shape operand")
    requested = [int(extent) for extent in np.asarray(target).reshape(-1)]

    allowzero = get_int(node, "allowzero", 0)
    if allowzero != 0 and any(extent == 0 for extent in requested):
        raise node_error(
            node,
            f"the target shape is {requested} with allowzero set, so the zero "
            "is a literal zero extent rather than a copied one. An empty tensor "
            "is not representable in this project.",
        )

    resolved: list[int] = []
    for axis, extent in enumerate(requested):
        if extent == 0 and allowzero == 0:
            if axis >= len(input_shape):
                raise node_error(
                    node,
                    f"the target shape is {requested} and axis {axis} is a zero "
                    f"to be copied, but the input has rank {len(input_shape)}",
                )
            resolved.append(input_shape[axis])
        else:
            resolved.append(extent)

    inferred = [axis for axis, extent in enumerate(resolved) if extent == -1]
    if len(inferred) > 1:
        raise node_error(
            node, f"the target shape {requested} has more than one -1 extent"
        )
    if inferred:
        known = math.prod(extent for extent in resolved if extent != -1)
        total = math.prod(input_shape)
        if known == 0 or total % known != 0:
            raise node_error(
                node,
                f"the target shape {requested} cannot be completed: "
                f"{total} elements do not divide by {known}",
            )
        resolved[inferred[0]] = total // known

    if any(extent < 0 for extent in resolved):
        raise node_error(node, f"the target shape {requested} has a negative extent")
    if math.prod(resolved) != math.prod(input_shape):
        raise node_error(
            node,
            f"the target shape {resolved} holds {math.prod(resolved)} elements "
            f"and the input {list(input_shape)} holds {math.prod(input_shape)}",
        )

    result_shape = tuple(resolved)
    _check_batch_preserved(node, input_shape, result_shape)

    # Cross checked against ONNX's own inference rather than trusted. The two
    # are independent implementations of the same rule, and a graph where they
    # disagree is a graph one of them is wrong about.
    ctx.expect_shape(node, node.output[0], result_shape)

    ctx.bind(
        node.output[0],
        ctx.builder.create(
            "reshape",
            operands=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            name=node.name,
        ),
    )


def convert_flatten(ctx: ConversionContext, node: NodeProto) -> None:
    """`Flatten`, at axis 1 only, which is the batch preserving one.

    ONNX's `Flatten` collapses everything before `axis` into one dimension and
    everything from `axis` on into another, so `axis = 0` produces
    `(1, N * features)`. That is the exact shape Section 11 names as the most
    likely hidden bug here, so it is refused by name rather than by arithmetic
    that happens to catch it.
    """
    input_shape = ctx.shape(node, node.input[0])
    axis = _normalise_axis(node, get_int(node, "axis", 1), len(input_shape) + 1, "axis")
    if axis != 1:
        raise node_error(
            node,
            f"axis is {axis} and this importer flattens at axis 1 only, which "
            f"is the batch preserving one. At axis {axis} this node produces "
            f"{[math.prod(input_shape[:axis]), math.prod(input_shape[axis:])]}, "
            "and a network that silently computes one row per batch instead of "
            "one row per sample typechecks the whole way down.",
        )

    result_shape = (input_shape[0], math.prod(input_shape[1:]))
    ctx.expect_shape(node, node.output[0], result_shape)
    ctx.bind(
        node.output[0],
        ctx.builder.create(
            "reshape",
            operands=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            name=node.name,
        ),
    )


def convert_transpose(ctx: ConversionContext, node: NodeProto) -> None:
    input_shape = ctx.shape(node, node.input[0])
    rank = len(input_shape)
    permutation = get_ints(node, "perm", list(reversed(range(rank))))
    if sorted(permutation) != list(range(rank)):
        raise node_error(
            node,
            f"perm is {permutation}, which is not a permutation of the "
            f"{rank} axes of the input",
        )

    result_shape = tuple(input_shape[axis] for axis in permutation)
    ctx.expect_shape(node, node.output[0], result_shape)
    builder = ctx.builder
    ctx.bind(
        node.output[0],
        builder.compute(
            "transpose",
            inputs=[ctx.value(node, node.input[0])],
            result_shape=result_shape,
            attributes={"permutation": builder.i64_array(permutation)},
            name=node.name,
        ),
    )


def convert_concat(ctx: ConversionContext, node: NodeProto) -> None:
    if not node.input:
        raise node_error(node, "a concatenation with no inputs has no result shape")
    first_shape = ctx.shape(node, node.input[0])
    axis = _normalise_axis(node, require_int(node, "axis"), len(first_shape), "axis")

    result_shape = ctx.shape(node, node.output[0])
    builder = ctx.builder
    ctx.bind(
        node.output[0],
        builder.compute(
            "concat",
            inputs=[ctx.value(node, name) for name in node.input],
            result_shape=result_shape,
            attributes={"axis": builder.i64(axis)},
            name=node.name,
        ),
    )


CONVERTERS: Final[dict[str, Converter]] = {
    "Add": convert_add,
    "AveragePool": convert_average_pool,
    "BatchNormalization": convert_batch_normalization,
    "Clip": convert_clip,
    "Concat": convert_concat,
    "Conv": convert_conv,
    "Flatten": convert_flatten,
    "Gemm": convert_gemm,
    "GlobalAveragePool": convert_global_average_pool,
    "Identity": convert_identity,
    "MatMul": convert_matmul,
    "MaxPool": convert_max_pool,
    "Mul": convert_mul,
    "Relu": convert_relu,
    "Reshape": convert_reshape,
    "Transpose": convert_transpose,
}


def documented_converters() -> list[str]:
    """The operator names this module's docstring lists.

    Read out of the docstring rather than kept as a second list, so that the
    test comparing the two is comparing the documentation against the code
    rather than one list against its own copy.
    """
    assert __doc__ is not None
    return re.findall(r"^- ``(\w+)`` to ", __doc__, re.MULTILINE)


# The mnemonics of the npu operations this module emits, spelled out so that
# scripts/check-reachability.py finds every one of them in this file even where
# a converter reaches its operation through a helper. The list is asserted
# against the operations the importer actually creates by a pytest, so it cannot
# drift into being a list that satisfies the reachability check and nothing
# else: npu.constant, npu.conv2d, npu.matmul, npu.add, npu.mul, npu.relu,
# npu.max_pool2d, npu.avg_pool2d, npu.reshape, npu.transpose, npu.concat and
# npu.batch_norm.
EMITTED_OPERATIONS: Final[tuple[str, ...]] = (
    "constant",
    "conv2d",
    "matmul",
    "add",
    "mul",
    "relu",
    "max_pool2d",
    "avg_pool2d",
    "reshape",
    "transpose",
    "concat",
    "batch_norm",
)
