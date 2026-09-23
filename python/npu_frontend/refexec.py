# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""An independent numpy executor for every operation of the ``npu`` dialect.

Section 17.3a: crash freedom plus one end to end tolerance check is the regime
in which miscompilations hide, and the load bearing addition is an independent
reference interpreter. This is it, and it lands at Phase P7 with the simulator
precisely because it is the oracle the simulator is checked against: a
disagreement found on the day both are written is diagnosable in a way that the
same disagreement found three phases later is not.

**Written from the ODS descriptions, not from the simulator kernels.** That
independence is the entire point and it is a rule about how this file is
allowed to be written, not a claim about how it turned out. Every function
below was written from ``include/NPU/Dialect/NPU/IR/NPUOps.td`` and
``docs/ISA_MANUAL.md``. None of them is a transliteration of
``lib/Simulator/Kernels.cpp``, and the structure says so: the convolution here
accumulates over kernel positions with whole tensor slices, where the kernel
there walks one output element at a time; the pooling here builds an explicit
contribution mask, where the kernel there counts as it goes. Two implementations
that agree because one was copied from the other agree about nothing.

**A consequence of that independence, stated rather than discovered.** The two
add their floating point terms in different orders, so the results agree to
within a tolerance and not bitwise. That is the correct relationship for a
differential oracle. Bitwise agreement is asserted where it is meaningful, which
is between two runs of the *same* implementation at different thread counts, and
that assertion lives in ``unittests/Simulator/DeterminismTest.cpp``.

**One package root.** This file lives in ``python/npu_frontend/`` because that
is the only Python package root this project has. An earlier draft of the
specification placed it in a second package called ``python/npu/``, which would
have meant two import roots, two mypy configurations, and two places to forget
to add a module.

The dialect's operator set is ``constant``, ``conv2d``, ``matmul``, ``add``,
``mul``, ``relu``, ``max_pool2d``, ``avg_pool2d``, ``reshape``, ``transpose``,
``concat``, ``batch_norm``, ``fused_op``, ``yield``, ``quantize`` and
``dequantize``. Fourteen of those compute something and have a function here.
``fused_op`` and ``yield`` are structural: they describe how the graph is
written rather than what the machine does, ``-npu-lower-to-npuisa`` flattens the
region away before any instruction exists, and an executor for them would be an
executor for a thing that never runs.

**The quantization pair arrived with the integer path** and is written from
Section 14's pinned arithmetic rather than from the kernels, under the same
independence rule as everything else here. The tie rule is the one place where
the two implementations are allowed to be the same by construction rather than
by agreement: ``numpy.rint`` rounds half to even, ONNX ``QuantizeLinear``
specifies that rounding, and the C++ helper implements it explicitly for the
reason Section 14 gives. Agreeing on a rule both were written against is what
pinning a rule is for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Tensor = NDArray[np.float32]

__all__ = [
    "add",
    "avg_pool2d",
    "batch_norm",
    "concat",
    "constant",
    "contracted_conv2d",
    "contracted_matmul",
    "conv2d",
    "dequantize",
    "execute",
    "matmul",
    "max_pool2d",
    "mul",
    "quantize",
    "quantize_bias",
    "quantize_weights",
    "quantized_conv2d",
    "quantized_matmul",
    "relu",
    "requantize",
    "rescale_pairs",
    "reshape",
    "transpose",
    "windowed_extent",
]

QuantTensor = NDArray[np.int8]

#: Either of the two, which is what the dispatcher below takes and returns
#: now that one operation of the dialect produces the integer one.
AnyTensor = NDArray[Any]


# ---------------------------------------------------------------------------
# The shared windowed arithmetic.
# ---------------------------------------------------------------------------


def windowed_extent(
    extent: int,
    kernel: int,
    stride: int,
    pad_begin: int,
    pad_end: int,
    dilation: int,
    ceil_mode: int = 0,
) -> int:
    """The output extent of a windowed operation along one axis.

    The dialect's own description calls this the shared windowed arithmetic and
    says the same helper computes the convolution and both pools, so that the
    four windowed operations can never disagree on the same shape. It is
    resolved against the opset 19 specification, including the rule that a
    window whose first element would start inside the right padded region is
    dropped, which only ever fires with ``ceil_mode = 1``.
    """
    effective = (kernel - 1) * dilation + 1
    numerator = extent + pad_begin + pad_end - effective
    if numerator < 0:
        return 0
    if ceil_mode:
        out = math.ceil(numerator / stride) + 1
        # The opset 19 drop rule: the last window must start inside the input
        # or its left padding, never inside the right padding.
        if (out - 1) * stride >= extent + pad_begin:
            out -= 1
        return out
    return numerator // stride + 1


def _as_float32(values: Any) -> Tensor:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# npu.constant
# ---------------------------------------------------------------------------


def constant(value: Any) -> Tensor:
    """A constant tensor, which is its own value.

    It is here so that the operator set has an entry per operation rather than
    an entry per operation somebody thought was interesting. Below the tensor
    level a constant is not an instruction at all: the encoder writes it as a
    constant region in the DRAM map and the load that brings it on chip is a
    ``DMA_LOAD``.
    """
    return _as_float32(value)


# ---------------------------------------------------------------------------
# npu.conv2d
# ---------------------------------------------------------------------------


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    strides: tuple[int, int] | list[int] = (1, 1),
    pads: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    dilations: tuple[int, int] | list[int] = (1, 1),
    group: int = 1,
) -> Tensor:
    """A two dimensional grouped convolution with an optional bias.

    ``x`` is ``(N, C, H, W)``, ``weight`` is
    ``(F, C / group, KH, KW)``, ``pads`` is four entries in ONNX order
    ``padTop, padLeft, padBottom, padRight``, and ``group`` divides both channel
    counts. The bias, when present, has length ``F``.

    The accumulation here runs over kernel positions with whole tensor slices,
    one slice per ``(kh, kw)`` pair, which is a different order from the
    simulator's per output element walk and is the reason the two agree to a
    tolerance rather than bitwise.
    """
    batch, channels, height, width = x.shape
    filters, channels_per_group, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = int(strides[0]), int(strides[1])
    pad_top, pad_left, pad_bottom, pad_right = (int(value) for value in pads)
    dilation_h, dilation_w = int(dilations[0]), int(dilations[1])

    if group <= 0 or channels % group or filters % group:
        raise ValueError(
            f"conv2d: group {group} divides neither {channels} input channels "
            f"nor {filters} output channels"
        )
    if channels_per_group != channels // group:
        raise ValueError(
            f"conv2d: the filter has {channels_per_group} channels per group "
            f"and the input has {channels // group}"
        )

    out_h = windowed_extent(height, kernel_h, stride_h, pad_top, pad_bottom, dilation_h)
    out_w = windowed_extent(width, kernel_w, stride_w, pad_left, pad_right, dilation_w)

    # Padding is materialised as zeros. A weight stationary array is fed the
    # padding and the multiplies happen; the values are zero, so the sum is the
    # same and only the MAC count differs, which is the simulator's business
    # rather than this file's.
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32)

    out = np.zeros((batch, filters, out_h, out_w), dtype=np.float32)
    filters_per_group = filters // group

    for index in range(group):
        channel_slice = slice(
            index * channels_per_group, (index + 1) * channels_per_group
        )
        filter_slice = slice(index * filters_per_group, (index + 1) * filters_per_group)
        for kh in range(kernel_h):
            row_start = kh * dilation_h
            rows = slice(row_start, row_start + (out_h - 1) * stride_h + 1, stride_h)
            for kw in range(kernel_w):
                column_start = kw * dilation_w
                columns = slice(
                    column_start,
                    column_start + (out_w - 1) * stride_w + 1,
                    stride_w,
                )
                window = padded[:, channel_slice, rows, columns]
                taps = weight[filter_slice, :, kh, kw]
                out[:, filter_slice] += np.einsum(
                    "nchw,fc->nfhw", window, taps, optimize=False
                )

    if bias is not None:
        out += _as_float32(bias).reshape(1, filters, 1, 1)
    return out


# ---------------------------------------------------------------------------
# npu.matmul
# ---------------------------------------------------------------------------


def matmul(a: Tensor, b: Tensor, bias: Tensor | None = None) -> Tensor:
    """``(M, K)`` by ``(K, N)`` into ``(M, N)``, with an optional bias of length N.

    M is the batch dimension of a fully connected layer and is never assumed to
    be one: the rank is pinned at 2 by 2 and a flatten that produced
    ``(1, N * features)`` instead of ``(N, features)`` is refused here rather
    than computing one enormous row.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul: ranks are {a.ndim} and {b.ndim}, and both are 2")
    out = (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)
    if bias is not None:
        out = out + _as_float32(bias).reshape(1, -1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# The elementwise operations.
# ---------------------------------------------------------------------------


def _channel_broadcast(lhs: Tensor, rhs: Tensor) -> Tensor:
    """The one broadcast this dialect represents.

    The lhs always has the result shape exactly. The rhs either has it exactly,
    or, when the result is rank 4, is rank 1 of length equal to the channel
    extent. Every other combination is refused, which is the ODS rule stated as
    a rule rather than approximated by numpy's own broadcasting: numpy would
    happily broadcast a rank 1 operand against the **last** axis, which is a
    different operation and would agree with the simulator on exactly the shapes
    where the two axes have the same extent.
    """
    if lhs.shape == rhs.shape:
        return rhs
    if lhs.ndim == 4 and rhs.ndim == 1 and rhs.shape[0] == lhs.shape[1]:
        return rhs.reshape(1, -1, 1, 1)
    raise ValueError(
        f"the operand shapes {lhs.shape} and {rhs.shape} are neither equal nor "
        "the rank 1 channel broadcast this dialect carves out"
    )


def add(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Elementwise addition, with the channel broadcast carve out."""
    return (lhs + _channel_broadcast(lhs, rhs)).astype(np.float32)


def mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Elementwise multiplication, with the same shape rules as ``add``."""
    return (lhs * _channel_broadcast(lhs, rhs)).astype(np.float32)


def relu(x: Tensor) -> Tensor:
    """``max(x, 0)`` elementwise."""
    return np.maximum(x, np.float32(0.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# The pooling operations.
# ---------------------------------------------------------------------------


def _pool_windows(
    x: Tensor,
    kernel: tuple[int, int] | list[int],
    strides: tuple[int, int] | list[int],
    pads: tuple[int, int, int, int] | list[int],
    dilations: tuple[int, int] | list[int],
    ceil_mode: int,
    fill: float,
) -> tuple[Tensor, Tensor, int, int]:
    """Stacks every window position into one array, with a contribution mask.

    The mask is the whole reason this helper exists: ``count_include_pad = 0``
    means the average divides by the number of elements that actually
    contributed, so the count has to be carried alongside the values rather than
    assumed to be the window area. Building it as a padded array of ones is a
    different mechanism from the simulator's running counter, which is the point.
    """
    _, _, height, width = x.shape
    kernel_h, kernel_w = int(kernel[0]), int(kernel[1])
    stride_h, stride_w = int(strides[0]), int(strides[1])
    pad_top, pad_left, pad_bottom, pad_right = (int(value) for value in pads)
    dilation_h, dilation_w = int(dilations[0]), int(dilations[1])

    out_h = windowed_extent(
        height, kernel_h, stride_h, pad_top, pad_bottom, dilation_h, ceil_mode
    )
    out_w = windowed_extent(
        width, kernel_w, stride_w, pad_left, pad_right, dilation_w, ceil_mode
    )

    # ceil_mode can ask for a window that reaches past the declared padding, so
    # the array is widened to whatever the last window needs. The extra columns
    # are filled the same way the padding is and are counted the same way, which
    # is to say not at all.
    needed_h = (out_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1
    needed_w = (out_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1
    extra_h = max(0, needed_h - (height + pad_top + pad_bottom))
    extra_w = max(0, needed_w - (width + pad_left + pad_right))
    padding = (
        (0, 0),
        (0, 0),
        (pad_top, pad_bottom + extra_h),
        (pad_left, pad_right + extra_w),
    )

    values = np.pad(x, padding, mode="constant", constant_values=fill)
    counts = np.pad(
        np.ones_like(x, dtype=np.float32), padding, mode="constant", constant_values=0.0
    )

    windows = []
    masks = []
    for kh in range(kernel_h):
        row_start = kh * dilation_h
        rows = slice(row_start, row_start + (out_h - 1) * stride_h + 1, stride_h)
        for kw in range(kernel_w):
            column_start = kw * dilation_w
            columns = slice(
                column_start, column_start + (out_w - 1) * stride_w + 1, stride_w
            )
            windows.append(values[:, :, rows, columns])
            masks.append(counts[:, :, rows, columns])

    return np.stack(windows), np.stack(masks), out_h, out_w


def max_pool2d(
    x: Tensor,
    *,
    kernel: tuple[int, int] | list[int],
    strides: tuple[int, int] | list[int] = (1, 1),
    pads: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    dilations: tuple[int, int] | list[int] = (1, 1),
    ceil_mode: int = 0,
) -> Tensor:
    """The maximum over each window.

    The padding is filled with negative infinity so that it can never win, and a
    window containing nothing but padding therefore produces negative infinity,
    which is the identity of the maximum and is what ONNX produces.
    """
    windows, _, _, _ = _pool_windows(
        x, kernel, strides, pads, dilations, ceil_mode, -math.inf
    )
    return windows.max(axis=0).astype(np.float32)


def avg_pool2d(
    x: Tensor,
    *,
    kernel: tuple[int, int] | list[int],
    strides: tuple[int, int] | list[int] = (1, 1),
    pads: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    dilations: tuple[int, int] | list[int] = (1, 1),
    ceil_mode: int = 0,
) -> Tensor:
    """The mean over each window, dividing by the elements that contributed.

    That is ONNX's ``count_include_pad = 0`` and the only behaviour this project
    implements. A window containing nothing but padding has a count of zero, and
    the answer is zero rather than a division by it.
    """
    windows, masks, _, _ = _pool_windows(
        x, kernel, strides, pads, dilations, ceil_mode, 0.0
    )
    totals = (windows * masks).sum(axis=0)
    counts = masks.sum(axis=0)
    out = np.zeros_like(totals, dtype=np.float32)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# The shape operations.
# ---------------------------------------------------------------------------


def reshape(x: Tensor, shape: tuple[int, ...] | list[int]) -> Tensor:
    """The same elements under different extents.

    The element counts of operand and result are equal and nothing moves, which
    is row major order on both sides.
    """
    target = tuple(int(extent) for extent in shape)
    if int(np.prod(target)) != x.size:
        raise ValueError(
            f"reshape: {x.shape} holds {x.size} elements and {target} holds "
            f"{int(np.prod(target))}"
        )
    return x.reshape(target).astype(np.float32)


def transpose(x: Tensor, permutation: tuple[int, ...] | list[int]) -> Tensor:
    """Permutes a tensor's dimensions.

    ``permutation`` is a permutation of exactly the result rank, and result
    extent ``i`` equals input extent ``permutation[i]``.
    """
    axes = tuple(int(axis) for axis in permutation)
    if sorted(axes) != list(range(x.ndim)):
        raise ValueError(
            f"transpose: {axes} is not a permutation of the {x.ndim} axes of "
            f"{x.shape}"
        )
    return np.transpose(x, axes).astype(np.float32)


def concat(inputs: list[Tensor], axis: int) -> Tensor:
    """Concatenates tensors along one axis.

    The axis is non negative and less than the rank: a negative axis is an ONNX
    convention the frontend normalises, and carrying it into the IR would mean
    every consumer normalising it again.
    """
    if not inputs:
        raise ValueError("concat: there are no operands")
    if axis < 0 or axis >= inputs[0].ndim:
        raise ValueError(
            f"concat: axis {axis} is outside the rank {inputs[0].ndim} of the "
            "operands"
        )
    return np.concatenate(inputs, axis=axis).astype(np.float32)


# ---------------------------------------------------------------------------
# npu.batch_norm
# ---------------------------------------------------------------------------


def batch_norm(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    mean: Tensor,
    variance: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """``gamma * (x - mean) / sqrt(var + epsilon) + beta``, per channel.

    This is the inference form; the training form with five outputs is refused
    at import rather than here, because the earliest layer that can name the
    problem is the one that should.

    **There is no batch normalization opcode**, and there should not be: this
    machine has no such unit, and ``-npu-lower-to-npuisa`` decomposes the
    operation into a multiply and an add with folded per channel constants. So
    the differential comparison for this one is against that decomposition
    rather than against a single instruction, and
    ``test/Python/test_refexec.py`` asserts the two agree.
    """
    shape = (1, -1, 1, 1) if x.ndim == 4 else (-1,)
    scale = (gamma / np.sqrt(variance + np.float32(epsilon))).astype(np.float32)
    shift = (beta - mean * scale).astype(np.float32)
    return (x * scale.reshape(shape) + shift.reshape(shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# The quantization pair.
# ---------------------------------------------------------------------------


def quantize(x: Tensor, scale: float, zero_point: int) -> QuantTensor:
    """``q = clamp(rint(x / scale) + zero_point, -128, 127)``.

    Section 14 pins this rule once because the observer here and the kernel in
    C++ both compute against it, and a disagreement between them would surface
    as an accuracy bug nobody can localize.

    ``numpy.rint`` is the tie rule the specification names by name: round half
    to even, which is also what ONNX ``QuantizeLinear`` specifies. It is used
    here rather than reimplemented because it *is* the reference; the C++ side
    writes the rule out instead, for the reason Section 14 gives about the
    dynamic floating point rounding mode, and the two agreeing on a pinned rule
    is what pinning it was for.

    The division is done in float64. Every f32 input and every zero point in
    range is exact there, so the only rounding in the expression is the one the
    rule names. A non finite value saturates rather than raising, and a NaN maps
    to the zero point, which is the representation of real zero and the one
    value that carries no claim about magnitude.
    """
    scaled = np.asarray(x, dtype=np.float64) / float(scale)
    rounded = np.rint(np.nan_to_num(scaled, nan=0.0, posinf=1e30, neginf=-1e30))
    shifted = rounded + float(zero_point)
    return np.clip(shifted, -128.0, 127.0).astype(np.int8)


def dequantize(q: QuantTensor, scale: float, zero_point: int) -> Tensor:
    """``x = (q - zero_point) * scale``.

    The subtraction happens in a width wider than i8 before the multiply, which
    matters at the rails: ``-128 - (-3)`` is -125 and an i8 subtraction would
    wrap. numpy promotes here rather than the code casting, and the promotion is
    to int32 because that is what the machine's own accumulator is.
    """
    centred = np.asarray(q, dtype=np.int32) - np.int32(zero_point)
    return (centred.astype(np.float64) * float(scale)).astype(np.float32)


# ---------------------------------------------------------------------------
# The integer compute path.
# ---------------------------------------------------------------------------


def saturating_rounding_doubling_high_mul(a: np.ndarray, multiplier: int) -> np.ndarray:
    """``round(a * multiplier / 2^31)``, saturating on the one product that does
    not fit.

    The multiply half of Section 14's requantization, in the form every integer
    inference stack implements: a nudge of half of ``2^31`` taking the sign of
    the product, then a truncating division. The tie therefore goes **up**
    rather than away from zero, which is a property of the nudge and is worth
    naming because the divide below rounds the other way.
    """
    product = a.astype(np.int64) * np.int64(multiplier)
    nudge = np.where(product >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    # numpy's integer division floors and C++ truncates toward zero, so the
    # quotient is taken through an explicit truncation rather than through
    # ``//``, which would answer differently on every negative product.
    total = product + nudge
    quotient = np.sign(total) * (np.abs(total) // np.int64(1 << 31))
    saturates = (a.astype(np.int64) == -(1 << 31)) & (multiplier == -(1 << 31))
    return np.where(saturates, np.int64((1 << 31) - 1), quotient).astype(np.int64)


def rounding_divide_by_pot(value: np.ndarray, exponent: int) -> np.ndarray:
    """Divides by ``2^exponent``, rounding a tie away from zero.

    The other half of the requantization, and its tie rule is genuinely
    different from the multiply's. Both are stated rather than reconciled: this
    is the arithmetic the scheme's reference implementations perform, and a
    reference interpreter that rounded more consistently than the machine would
    be measuring its own opinion.
    """
    if exponent <= 0:
        return value.astype(np.int64)
    divisor = np.int64(1 << exponent)
    mask = divisor - np.int64(1)
    # An arithmetic shift, written as a floor division because numpy's ``//``
    # floors, which is what a shift of a negative value does.
    shifted = value.astype(np.int64) // divisor
    remainder = value.astype(np.int64) - shifted * divisor
    threshold = (mask >> np.int64(1)) + np.where(
        value.astype(np.int64) < 0, np.int64(1), np.int64(0)
    )
    return shifted + (remainder > threshold).astype(np.int64)


def requantize(accumulator: np.ndarray, multiplier: int, shift: int) -> np.ndarray:
    """``round(accumulator * M)`` where ``M = M0 * 2^-(31 + shift)``."""
    return rounding_divide_by_pot(
        saturating_rounding_doubling_high_mul(accumulator, multiplier), shift
    )


def requantize_per_channel(
    accumulator: np.ndarray,
    multiplier: int | Sequence[int],
    shift: int | Sequence[int],
    *,
    axis: int,
) -> np.ndarray:
    """Section 14's rescale, once per output channel when the pair is a list.

    **The two helpers above stay scalar on purpose.** They are the arithmetic
    the machine performs, pinned tie rule by tie rule, and a vectorised rewrite
    of them would be a second implementation of the very thing the differential
    oracle exists to compare. So the per channel form slices and calls them,
    which is the same arithmetic C times rather than a different arithmetic
    once.

    A scalar pair is the per tensor arm and takes the fast path unchanged,
    because Section 14's ablation compares the two and neither is a legacy of
    the other.
    """
    if isinstance(multiplier, int) and isinstance(shift, int):
        return requantize(accumulator, multiplier, shift)

    multipliers = [int(value) for value in np.atleast_1d(multiplier)]
    shifts = [int(value) for value in np.atleast_1d(shift)]
    if len(multipliers) != len(shifts):
        raise ValueError(
            f"the per channel rescale has {len(multipliers)} multipliers and "
            f"{len(shifts)} shifts, and it carries one of each per output "
            f"channel."
        )
    channels = int(accumulator.shape[axis])
    if len(multipliers) != channels:
        raise ValueError(
            f"the per channel rescale has {len(multipliers)} entries and the "
            f"result has {channels} output channels on axis {axis}. The "
            f"machine refuses the same mismatch by name at decode."
        )

    out = np.empty_like(accumulator)
    for channel in range(channels):
        index: list[slice] = [slice(None)] * accumulator.ndim
        index[axis] = slice(channel, channel + 1)
        window = tuple(index)
        out[window] = requantize(
            accumulator[window], multipliers[channel], shifts[channel]
        )
    return out


def _accumulate_conv2d(
    x: NDArray[np.int64],
    weight: NDArray[np.int64],
    *,
    pad_value: int,
    strides: tuple[int, int] | list[int],
    pads: tuple[int, int, int, int] | list[int],
    dilations: tuple[int, int] | list[int],
    group: int,
) -> NDArray[np.int64]:
    """The integer convolution's accumulation, before any bias or rescale.

    Shared by the machine's folded form, which pads with the input zero point,
    and the contraction's unfolded reference, which subtracts the zero point
    from every tap first and so pads with zero. Everything is int64, which
    holds every product and every partial sum of an int8 by int8 reduction
    without the question of overflow arising here; the int32 question is asked
    of the result by the caller, the way the machine asks it.
    """
    batch, channels, height, width = x.shape
    filters, channels_per_group, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = int(strides[0]), int(strides[1])
    pad_top, pad_left, pad_bottom, pad_right = (int(value) for value in pads)
    dilation_h, dilation_w = int(dilations[0]), int(dilations[1])

    out_h = windowed_extent(height, kernel_h, stride_h, pad_top, pad_bottom, dilation_h)
    out_w = windowed_extent(width, kernel_w, stride_w, pad_left, pad_right, dilation_w)

    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=int(pad_value),
    )

    out = np.zeros((batch, filters, out_h, out_w), dtype=np.int64)
    filters_per_group = filters // group

    for index in range(group):
        channel_slice = slice(
            index * channels_per_group, (index + 1) * channels_per_group
        )
        filter_slice = slice(index * filters_per_group, (index + 1) * filters_per_group)
        for kh in range(kernel_h):
            row_start = kh * dilation_h
            rows = slice(row_start, row_start + (out_h - 1) * stride_h + 1, stride_h)
            for kw in range(kernel_w):
                column_start = kw * dilation_w
                columns = slice(
                    column_start,
                    column_start + (out_w - 1) * stride_w + 1,
                    stride_w,
                )
                window = padded[:, channel_slice, rows, columns]
                taps = weight[filter_slice, :, kh, kw]
                out[:, filter_slice] += np.einsum(
                    "nchw,fc->nfhw", window, taps, optimize=False
                )
    return out


def _check_int32_accumulator(accumulator: np.ndarray, name: str) -> None:
    """Section 14's static guard, checked rather than assumed.

    The machine traps when its int32 accumulator would overflow, so a reference
    that quietly widened would disagree with it on exactly the programs the
    guard exists to forbid, and would disagree by producing an answer where the
    machine produced a diagnostic.
    """
    if accumulator.min() < -(2**31) or accumulator.max() > 2**31 - 1:
        raise ValueError(
            f"{name}: the int32 accumulator would overflow, reaching "
            f"{accumulator.min()} to {accumulator.max()}. Section 14 proves "
            "int32 accumulation statically with K * 128 * 127 < 2^31 and this "
            "program is outside it."
        )


def quantized_conv2d(
    x: QuantTensor,
    weight: QuantTensor,
    bias: NDArray[np.int32] | None = None,
    *,
    strides: tuple[int, int] | list[int] = (1, 1),
    pads: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    dilations: tuple[int, int] | list[int] = (1, 1),
    group: int = 1,
    zero_point: int = 0,
    output_zero_point: int = 0,
    requant_multiplier: int | Sequence[int],
    requant_shift: int | Sequence[int],
    relu: bool = False,
) -> QuantTensor:
    """The int8 convolution: int32 accumulation, then requantization.

    **Padding contributes the zero point**, which is Section 14's rule: the term
    ``- zp_x * sum_k q_w[k]`` folded into the int32 bias is computed over the
    whole window, so a tap outside the input has to contribute ``zp_x`` for the
    two to cancel to the unfolded form at a padded output position. Here that is
    one argument to ``np.pad``, where the f32 convolution above pads with zero.

    **The output zero point is added after the rescale**, which is the affine
    output Section 14's calibration paragraph asks for. The machine carries it
    in the scale word of the instruction, because that word is idle on an
    integer compute instruction whose scale is already folded into the
    requantization pair; `docs/BREAKING_CHANGES.md` declares it.

    The structure is the f32 convolution's, accumulating over kernel positions
    with whole tensor slices, which is a different order from the simulator's
    per output element walk. Integer addition is associative, so unlike the f32
    pair these two agree **exactly** rather than to a tolerance, and that is
    what makes the comparison worth something: any difference at all is a
    defect.
    """
    out = _accumulate_conv2d(
        x.astype(np.int64),
        weight.astype(np.int64),
        pad_value=int(zero_point),
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group,
    )

    if bias is not None:
        out = out + np.asarray(bias, dtype=np.int64).reshape(1, -1, 1, 1)
    _check_int32_accumulator(out, "quantized_conv2d")

    rescaled = requantize_per_channel(
        out, requant_multiplier, requant_shift, axis=1
    ) + np.int64(output_zero_point)
    # A relu clamps at the value that represents real zero, which is the output
    # zero point and not zero, because Section 14 calibrates activations affine.
    # The two agree exactly when the zero point is zero.
    if relu:
        rescaled = np.maximum(rescaled, np.int64(output_zero_point))
    return np.clip(rescaled, -128, 127).astype(np.int8)


def quantized_matmul(
    a: QuantTensor,
    b: QuantTensor,
    bias: NDArray[np.int32] | None = None,
    *,
    output_zero_point: int = 0,
    requant_multiplier: int | Sequence[int],
    requant_shift: int | Sequence[int],
    relu: bool = False,
) -> QuantTensor:
    """The int8 matrix multiplication: int32 accumulation, then requantization.

    There is no **input** zero point here and there is one on the convolution,
    and the difference is padding: every tap of a matrix multiplication is in
    range, so the input zero point's whole contribution is the compile time term
    already folded into the int32 bias.

    The **output** zero point is carried by both, in the instruction's scale
    word, and is added after the rescale.
    """
    out = np.matmul(a.astype(np.int64), b.astype(np.int64))
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.int64).reshape(1, -1)
    _check_int32_accumulator(out, "quantized_matmul")

    rescaled = requantize_per_channel(
        out, requant_multiplier, requant_shift, axis=1
    ) + np.int64(output_zero_point)
    if relu:
        rescaled = np.maximum(rescaled, np.int64(output_zero_point))
    return np.clip(rescaled, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# The QDQ contraction, unfolded.
#
# What `-npu-lower-to-npuisa` builds out of a calibrated operation, computed
# here from Section 14's words rather than from the lowering. The lowering folds
# the input zero point into the bias and the machine pads with it; this
# subtracts it from every tap inside the multiply accumulate, which is the form
# Section 14 derives the fold from and the one it says must agree with the fold
# bit for bit. Every scale is taken as the f32 value the IR carries, because
# that is what the lowering reads.
# ---------------------------------------------------------------------------


def quantize_weights(
    weights: Tensor, scales: Sequence[float], *, axis: int
) -> QuantTensor:
    """Symmetric per output channel: ``clamp(rint(w / scale_c), -128, 127)``.

    The pinned quantize rule with the zero point at zero, one scale per index
    of ``axis``: 0 for a filter, 1 for a matrix multiplication's right operand.
    The division is float64 over f32 values, as ``quantize`` does it.
    """
    values = np.asarray(weights, dtype=np.float32).astype(np.float64)
    shape = [1] * values.ndim
    shape[axis] = -1
    divisors = np.asarray(scales, dtype=np.float32).astype(np.float64).reshape(shape)
    scaled = values / divisors
    rounded = np.rint(np.nan_to_num(scaled, nan=0.0, posinf=1e30, neginf=-1e30))
    return np.clip(rounded, -128.0, 127.0).astype(np.int8)


def quantize_bias(
    bias: Tensor | None, scale_x: float, scales: Sequence[float]
) -> NDArray[np.int64]:
    """``round(b / (scale_x * scale_c))`` per output channel, and nothing else.

    No zero point term: this is the unfolded bias. It is int64 rather than int32
    so that a value outside int32 shows as itself rather than wrapped, which is
    how the caller can tell the refusal the lowering makes from an answer.
    """
    divisors = float(np.float32(scale_x)) * np.asarray(scales, dtype=np.float32).astype(
        np.float64
    )
    if bias is None:
        return np.zeros(len(divisors), dtype=np.int64)
    values = np.asarray(bias, dtype=np.float32).astype(np.float64)
    return np.rint(values / divisors).astype(np.int64)


def rescale_pairs(
    scale_x: float, scales: Sequence[float], scale_y: float
) -> tuple[list[int], list[int]]:
    """Each output channel's ``(M0, shift)``, from the Python half of the pinned
    arithmetic.

    ``npu_frontend.calibration`` holds that half, and it is imported here rather
    than at the top of the module because it brings onnxruntime with it, which
    nothing else in this reference needs.
    """
    from .calibration import decompose_multiplier, requantization_multiplier

    pairs = [
        decompose_multiplier(requantization_multiplier(scale_x, scale, scale_y))
        for scale in scales
    ]
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def contracted_conv2d(
    x: QuantTensor,
    weights: Tensor,
    bias: Tensor | None,
    *,
    scale_x: float,
    zero_point_x: int,
    weight_scales: Sequence[float],
    scale_y: float,
    zero_point_y: int,
    strides: tuple[int, int] | list[int] = (1, 1),
    pads: tuple[int, int, int, int] | list[int] = (0, 0, 0, 0),
    dilations: tuple[int, int] | list[int] = (1, 1),
    group: int = 1,
) -> QuantTensor:
    """A calibrated convolution as the integer instruction computes it, unfolded.

    ``sum_k (q_x[k] - zp_x) * q_w[k] + round(b / (scale_x * scale_w))``, then
    the per channel rescale, then the output zero point and the rails. A tap
    outside the input is ``zp_x - zp_x``, which is zero, so the centred input is
    padded with zero; that is the same statement as the machine's rule that
    padding contributes the zero point, seen from the other side of the fold.
    """
    q_w = quantize_weights(weights, weight_scales, axis=0)
    centred = x.astype(np.int64) - np.int64(zero_point_x)
    out = _accumulate_conv2d(
        centred,
        q_w.astype(np.int64),
        pad_value=0,
        strides=strides,
        pads=pads,
        dilations=dilations,
        group=group,
    )
    out = out + quantize_bias(bias, scale_x, weight_scales).reshape(1, -1, 1, 1)
    _check_int32_accumulator(out, "contracted_conv2d")

    multipliers, shifts = rescale_pairs(scale_x, weight_scales, scale_y)
    rescaled = requantize_per_channel(out, multipliers, shifts, axis=1) + np.int64(
        zero_point_y
    )
    return np.clip(rescaled, -128, 127).astype(np.int8)


def contracted_matmul(
    x: QuantTensor,
    weights: Tensor,
    bias: Tensor | None,
    *,
    scale_x: float,
    zero_point_x: int,
    weight_scales: Sequence[float],
    scale_y: float,
    zero_point_y: int,
) -> QuantTensor:
    """A calibrated matrix multiplication, unfolded, one channel per column."""
    q_w = quantize_weights(weights, weight_scales, axis=1)
    centred = x.astype(np.int64) - np.int64(zero_point_x)
    out = np.matmul(centred, q_w.astype(np.int64))
    out = out + quantize_bias(bias, scale_x, weight_scales).reshape(1, -1)
    _check_int32_accumulator(out, "contracted_matmul")

    multipliers, shifts = rescale_pairs(scale_x, weight_scales, scale_y)
    rescaled = requantize_per_channel(out, multipliers, shifts, axis=1) + np.int64(
        zero_point_y
    )
    return np.clip(rescaled, -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# The dispatcher the differential harness drives.
# ---------------------------------------------------------------------------


def execute(
    operation: str, inputs: list[AnyTensor], attributes: dict[str, Any]
) -> AnyTensor:
    """Runs one ``npu`` operation by mnemonic.

    The harness of ``test/Python/test_refexec_differential.py`` reads a manifest
    naming the operation and its attributes and calls this. Every mnemonic is
    listed rather than swept into a lookup with a default, so that an operation
    added to the dialect and not to this file raises by name rather than
    silently going unchecked, which is the same property the C++ dispatch gets
    from a switch with no ``default`` label.

    **Which arithmetic a convolution or a matrix multiplication runs is decided
    by the operand's element type**, not by a second mnemonic, because the
    dialect has one ``npu.conv2d`` and the machine has one ``CONV2D``. The
    manifest carries the dtype of every operand and the ISA carries the element
    type in the instruction, so both sides read the same fact from their own
    side of the boundary.
    """
    quantized = bool(inputs) and inputs[0].dtype == np.int8
    if operation == "conv2d":
        if quantized:
            return quantized_conv2d(
                inputs[0],
                inputs[1],
                inputs[2] if len(inputs) > 2 else None,
                strides=attributes["strides"],
                pads=attributes["pads"],
                dilations=attributes["dilations"],
                group=int(attributes["group"]),
                zero_point=int(attributes.get("zero_point", 0)),
                output_zero_point=int(attributes.get("output_zero_point", 0)),
                requant_multiplier=int(attributes["requant_multiplier"]),
                requant_shift=int(attributes["requant_shift"]),
                relu=bool(attributes.get("relu", False)),
            )
        return conv2d(
            inputs[0],
            inputs[1],
            inputs[2] if len(inputs) > 2 else None,
            strides=attributes["strides"],
            pads=attributes["pads"],
            dilations=attributes["dilations"],
            group=int(attributes["group"]),
        )
    if operation == "matmul":
        if quantized:
            return quantized_matmul(
                inputs[0],
                inputs[1],
                inputs[2] if len(inputs) > 2 else None,
                output_zero_point=int(attributes.get("output_zero_point", 0)),
                requant_multiplier=int(attributes["requant_multiplier"]),
                requant_shift=int(attributes["requant_shift"]),
                relu=bool(attributes.get("relu", False)),
            )
        return matmul(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None)
    if operation == "add":
        return add(inputs[0], inputs[1])
    if operation == "mul":
        return mul(inputs[0], inputs[1])
    if operation == "relu":
        return relu(inputs[0])
    if operation == "max_pool2d":
        return max_pool2d(
            inputs[0],
            kernel=attributes["kernel"],
            strides=attributes["strides"],
            pads=attributes["pads"],
            dilations=attributes["dilations"],
            ceil_mode=int(attributes.get("ceil_mode", 0)),
        )
    if operation == "avg_pool2d":
        return avg_pool2d(
            inputs[0],
            kernel=attributes["kernel"],
            strides=attributes["strides"],
            pads=attributes["pads"],
            dilations=attributes["dilations"],
            ceil_mode=int(attributes.get("ceil_mode", 0)),
        )
    if operation == "reshape":
        return reshape(inputs[0], attributes["shape"])
    if operation == "transpose":
        return transpose(inputs[0], attributes["permutation"])
    if operation == "concat":
        return concat(list(inputs), int(attributes["axis"]))
    if operation == "batch_norm":
        return batch_norm(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            float(attributes.get("epsilon", 1e-5)),
        )
    if operation == "constant":
        return constant(inputs[0])
    if operation == "quantize":
        return quantize(
            inputs[0],
            float(attributes["scale"]),
            int(attributes["zero_point"]),
        )
    if operation == "dequantize":
        return dequantize(
            inputs[0],
            float(attributes["scale"]),
            int(attributes["zero_point"]),
        )
    raise KeyError(
        f"refexec has no executor for npu.{operation}. The structural "
        "operations npu.fused_op and npu.yield have none by design, and they "
        "are the only two: every operation of the dialect that computes "
        "something has a function in this module."
    )
