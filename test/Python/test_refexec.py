# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The reference interpreter, against hand computed values.

``test_refexec_differential.py`` compares this interpreter with the simulator,
which catches a disagreement but says nothing about which of the two is wrong.
This file is what makes that answerable: it pins the reference against arithmetic
a reader can redo, so a differential failure is a question about the simulator
until one of these tests fails too.

The values here are computed in the comments, from the ODS descriptions. None of
them was captured from a run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from npu_frontend import refexec


def f32(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# The windowed arithmetic every shape below comes from.
# ---------------------------------------------------------------------------


def test_windowed_extent_matches_the_onnx_rule() -> None:
    # 8 wide, a 2 wide window at stride 2, no padding: four windows.
    assert refexec.windowed_extent(8, 2, 2, 0, 0, 1) == 4
    # A 3 wide window with one unit of padding on each side at unit stride
    # preserves the extent, which is the same convolution the suite uses
    # everywhere.
    assert refexec.windowed_extent(5, 3, 1, 1, 1, 1) == 5
    # Dilation 2 makes a 2 wide window reach 3, so 5 gives 3 outputs.
    assert refexec.windowed_extent(5, 2, 1, 0, 0, 2) == 3
    # Asymmetric padding: 2 above and 1 to the right of a 5 wide axis under a
    # 3 wide window is 5 + 2 + 0 - 3 + 1 = 5 one way and 5 + 0 + 1 - 3 + 1 = 4
    # the other.
    assert refexec.windowed_extent(5, 3, 1, 2, 0, 1) == 5
    assert refexec.windowed_extent(5, 3, 1, 0, 1, 1) == 4
    # ceil_mode rounds up, and the opset 19 drop rule removes a window that
    # would start inside the right padding. 5 under a 2 wide window at stride 3
    # gives 2 either way; 5 at stride 4 gives 2 with ceil and 1 without.
    assert refexec.windowed_extent(5, 2, 4, 0, 0, 1, ceil_mode=0) == 1
    assert refexec.windowed_extent(5, 2, 4, 0, 0, 1, ceil_mode=1) == 2


# ---------------------------------------------------------------------------
# The operations.
# ---------------------------------------------------------------------------


def test_conv2d_dense() -> None:
    # 1..9 under a 2 by 2 filter of ones: 1+2+4+5 = 12, 2+3+5+6 = 16,
    # 4+5+7+8 = 24, 5+6+8+9 = 28.
    x = f32([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
    w = f32([[[[1, 1], [1, 1]]]])
    out = refexec.conv2d(x, w)
    assert out.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(out, f32([[[[12, 16], [24, 28]]]]))


def test_conv2d_grouped() -> None:
    # Two groups of two channels, a 1 by 1 filter.
    #   f0 = 1 * c0 + 2 * c1, f1 = 3 * c2 + 4 * c3.
    x = f32(
        [
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
            ]
        ]
    )
    w = f32([[[[1]], [[2]]], [[[3]], [[4]]]])
    out = refexec.conv2d(x, w, group=2)
    expected = f32([[[[11, 14], [17, 20]], [[79, 86], [93, 100]]]])
    np.testing.assert_allclose(out, expected)


def test_conv2d_depthwise() -> None:
    # group == C, one filter tap per channel: scale each channel by its tap.
    x = f32([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]])
    w = f32([[[[2]]], [[[3]]], [[[4]]]])
    out = refexec.conv2d(x, w, group=3)
    expected = f32([[[[2, 4], [6, 8]], [[15, 18], [21, 24]], [[36, 40], [44, 48]]]])
    np.testing.assert_allclose(out, expected)


def test_conv2d_dilated() -> None:
    # Dilation 2 over 1..25 in a 5 by 5, a 2 by 2 filter of ones. The effective
    # kernel is 3, so the output is 3 by 3, and each output is the sum of the
    # four corners of a 3 by 3 patch.
    x = f32(np.arange(1, 26).reshape(1, 1, 5, 5))
    w = f32([[[[1, 1], [1, 1]]]])
    out = refexec.conv2d(x, w, dilations=(2, 2))
    expected = f32([[[[28, 32, 36], [48, 52, 56], [68, 72, 76]]]])
    np.testing.assert_allclose(out, expected)


def test_conv2d_asymmetric_padding() -> None:
    # One row above and one column to the right, over a 2 by 2 input.
    #   (0,0) = 1 + 2 = 3      (0,1) = 2
    #   (1,0) = 1 + 2 + 3 + 4  (1,1) = 2 + 4
    x = f32([[[[1, 2], [3, 4]]]])
    w = f32([[[[1, 1], [1, 1]]]])
    out = refexec.conv2d(x, w, pads=(1, 0, 0, 1))
    np.testing.assert_allclose(out, f32([[[[3, 2], [10, 6]]]]))


def test_conv2d_bias_is_per_output_channel() -> None:
    x = f32([[[[1, 1], [1, 1]]]])
    w = f32([[[[1]]], [[[2]]]])
    out = refexec.conv2d(x, w, f32([10, 20]))
    np.testing.assert_allclose(out, f32([[[[11, 11], [11, 11]], [[22, 22], [22, 22]]]]))


def test_conv2d_refuses_a_group_that_does_not_divide() -> None:
    x = f32(np.ones((1, 3, 2, 2)))
    w = f32(np.ones((2, 1, 1, 1)))
    with pytest.raises(ValueError, match="group"):
        refexec.conv2d(x, w, group=2)


def test_matmul() -> None:
    #   [1 2 3] [ 7  8]   [ 58  64]
    #   [4 5 6] [ 9 10] = [139 154]
    #           [11 12]
    a = f32([[1, 2, 3], [4, 5, 6]])
    b = f32([[7, 8], [9, 10], [11, 12]])
    np.testing.assert_allclose(refexec.matmul(a, b), f32([[58, 64], [139, 154]]))
    np.testing.assert_allclose(
        refexec.matmul(a, b, f32([1000, 2000])),
        f32([[1058, 2064], [1139, 2154]]),
    )


def test_elementwise_and_the_channel_broadcast() -> None:
    a = f32(np.arange(12).reshape(1, 3, 2, 2))
    b = f32([100, 200, 300])
    out = refexec.add(a, b)
    expected = f32(
        [[[[100, 101], [102, 103]], [[204, 205], [206, 207]], [[308, 309], [310, 311]]]]
    )
    np.testing.assert_allclose(out, expected)
    np.testing.assert_allclose(refexec.mul(a, a), a * a)


def test_the_channel_broadcast_is_the_only_one() -> None:
    # numpy would happily broadcast a rank 1 operand against the last axis. This
    # dialect carves out the channel axis and nothing else, so a rank 1 operand
    # whose length matches the last axis rather than the channel axis is
    # refused rather than quietly computing a different operation.
    a = f32(np.ones((1, 3, 2, 4)))
    with pytest.raises(ValueError, match="channel broadcast"):
        refexec.add(a, f32([1, 2, 3, 4]))


def test_relu() -> None:
    np.testing.assert_allclose(
        refexec.relu(f32([-1, 0, 1, -2.5, 5, -0.5])), f32([0, 0, 1, 0, 5, 0])
    )


def test_max_pool2d() -> None:
    x = f32(np.arange(1, 17).reshape(1, 1, 4, 4))
    out = refexec.max_pool2d(x, kernel=(2, 2), strides=(2, 2))
    np.testing.assert_allclose(out, f32([[[[6, 8], [14, 16]]]]))


def test_avg_pool2d() -> None:
    x = f32(np.arange(1, 17).reshape(1, 1, 4, 4))
    out = refexec.avg_pool2d(x, kernel=(2, 2), strides=(2, 2))
    np.testing.assert_allclose(out, f32([[[[3.5, 5.5], [11.5, 13.5]]]]))


def test_avg_pool2d_divides_by_the_contributing_count() -> None:
    # count_include_pad = 0: the corner window of a 3 by 3 over a padded 2 by 2
    # sees four real elements out of nine, and divides by four.
    x = f32([[[[1, 2], [3, 4]]]])
    out = refexec.avg_pool2d(x, kernel=(3, 3), strides=(1, 1), pads=(1, 1, 1, 1))
    assert out.shape == (1, 1, 2, 2)
    # Every window of a 3 by 3 over this input covers the whole 2 by 2, so every
    # answer is the mean of 1, 2, 3, 4.
    np.testing.assert_allclose(out, f32([[[[2.5, 2.5], [2.5, 2.5]]]]))


def test_the_all_padding_window() -> None:
    # A single element with two units of padding on every side and a 1 by 1
    # window: twenty four of the twenty five windows contain nothing at all.
    x = f32([[[[5]]]])
    average = refexec.avg_pool2d(x, kernel=(1, 1), strides=(1, 1), pads=(2, 2, 2, 2))
    assert average.shape == (1, 1, 5, 5)
    assert average[0, 0, 2, 2] == 5.0
    assert not np.isnan(average).any()
    assert average.sum() == 5.0

    maximum = refexec.max_pool2d(x, kernel=(1, 1), strides=(1, 1), pads=(2, 2, 2, 2))
    assert maximum[0, 0, 2, 2] == 5.0
    assert maximum[0, 0, 0, 0] == -math.inf


def test_reshape_and_transpose() -> None:
    x = f32([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(
        refexec.reshape(x, (3, 2)), f32([[1, 2], [3, 4], [5, 6]])
    )
    np.testing.assert_allclose(refexec.transpose(x, (0, 1)), x)
    np.testing.assert_allclose(refexec.transpose(x, (1, 0)), x.T)

    # The rank 4 NCHW to NHWC case, which is what the layout assignment pass
    # emits. result[h][w][c] = input[c][h][w].
    nchw = f32(np.arange(1, 13).reshape(1, 3, 2, 2))
    nhwc = refexec.transpose(nchw, (0, 2, 3, 1))
    assert nhwc.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(
        nhwc.reshape(-1), f32([1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12])
    )


def test_concat() -> None:
    a = f32([[1, 2], [3, 4]])
    b = f32([[5], [6]])
    c = f32([[7, 8, 9], [10, 11, 12]])
    np.testing.assert_allclose(
        refexec.concat([a, b, c], axis=1),
        f32([[1, 2, 5, 7, 8, 9], [3, 4, 6, 10, 11, 12]]),
    )


def test_batch_norm_agrees_with_its_decomposition() -> None:
    """The differential coverage batch normalization can have at this phase.

    There is no batch normalization opcode: this machine has no such unit and
    ``-npu-lower-to-npuisa`` decomposes the operation into a multiply and an add
    with folded per channel constants. So the comparison against the simulator
    that every other operation gets arrives at Phase P8 with the end to end
    pipeline. What can be checked here is the decomposition rule itself, which
    is what the lowering implements and what a folded constant has to satisfy.
    """
    rng = np.random.default_rng(20260822)
    x = f32(rng.standard_normal((2, 3, 4, 4)))
    gamma = f32([1.5, -2.0, 0.25])
    beta = f32([0.5, 1.0, -1.0])
    mean = f32([0.1, -0.2, 0.3])
    variance = f32([1.0, 4.0, 0.25])
    epsilon = 1e-5

    direct = refexec.batch_norm(x, gamma, beta, mean, variance, epsilon)

    # The decomposition the lowering emits: one per channel scale and one per
    # channel shift, applied as a mul and an add.
    scale = (gamma / np.sqrt(variance + np.float32(epsilon))).astype(np.float32)
    shift = (beta - mean * scale).astype(np.float32)
    decomposed = refexec.add(refexec.mul(x, scale), shift)

    np.testing.assert_allclose(direct, decomposed, rtol=1e-6, atol=1e-6)


def test_execute_refuses_an_operation_it_does_not_have() -> None:
    with pytest.raises(KeyError, match="P14"):
        refexec.execute("quantize", [f32([1.0])], {})
    with pytest.raises(KeyError, match="fused_op"):
        refexec.execute("fused_op", [f32([1.0])], {})
