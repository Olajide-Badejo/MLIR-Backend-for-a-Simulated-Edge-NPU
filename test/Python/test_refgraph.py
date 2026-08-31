# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""`refgraph`, which turns `refexec` from an operation oracle into a model one.

`test_end_to_end.py` uses it as an oracle over the whole suite. What this file
checks is the walker itself: that it reads properties rather than guessing them,
that its operand table is held to the IR rather than assumed, and that an
operation it does not know raises by name instead of being skipped.

The last one is the important one. A reference interpreter that silently ignores
an operation it does not recognise is worse than no reference at all: it agrees
with the simulator on every model that contains one, and the agreement is about
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from npu_frontend import refexec
from npu_frontend.refgraph import ExecutionError, execute_module

# A module written out in the dialect's custom assembly, which is what
# `npu-compile --emit npu` prints. It carries a windowed operation with real
# attributes, a channel broadcast, a constant, and a reshape whose extents come
# from its result type rather than from an attribute, so the four ways an
# operand or an attribute reaches the reference are all exercised.
MODULE = """
func.func @main(%x: tensor<1x2x4x4xf32>) -> tensor<1x8xf32> {
  %filter = npu.constant dense<0.5> : tensor<2x2x3x3xf32>
  %bias = npu.constant dense<[0.25, -0.5]> : tensor<2xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %conv = npu.conv2d ins(%x, %filter, %bias : tensor<1x2x4x4xf32>,
                                              tensor<2x2x3x3xf32>,
                                              tensor<2xf32>)
                     outs(%d0 : tensor<1x2x4x4xf32>)
                     {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                      dilations = array<i64: 1, 1>, group = 1 : i64}
                     -> tensor<1x2x4x4xf32>
  %scale = npu.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %scaled = npu.mul ins(%conv, %scale : tensor<1x2x4x4xf32>, tensor<2xf32>)
                    outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %d2 = tensor.empty() : tensor<1x2x2x2xf32>
  %pooled = npu.max_pool2d ins(%scaled : tensor<1x2x4x4xf32>)
                           outs(%d2 : tensor<1x2x2x2xf32>)
                           {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                            pads = array<i64: 0, 0, 0, 0>,
                            dilations = array<i64: 1, 1>}
                           -> tensor<1x2x2x2xf32>
  %flat = npu.reshape %pooled : tensor<1x2x2x2xf32> to tensor<1x8xf32>
  return %flat : tensor<1x8xf32>
}
"""


def reference(x: np.ndarray) -> np.ndarray:
    """The same computation, written out by hand against `refexec`.

    Not against `refgraph`. The point of this file is that the walker reads the
    IR correctly, and a hand written expected value is the only thing that can
    say so: comparing `refgraph` against `refgraph` would assert that it is
    consistent, which it would be even if it read every attribute wrong.
    """
    filter_ = np.full((2, 2, 3, 3), 0.5, dtype=np.float32)
    bias = np.array([0.25, -0.5], dtype=np.float32)
    conv = refexec.conv2d(
        x, filter_, bias, strides=(1, 1), pads=(1, 1, 1, 1), dilations=(1, 1), group=1
    )
    scaled = refexec.mul(conv, np.array([2.0, 3.0], dtype=np.float32))
    pooled = refexec.max_pool2d(
        scaled, kernel=(2, 2), strides=(2, 2), pads=(0, 0, 0, 0), dilations=(1, 1)
    )
    return refexec.reshape(pooled, (1, 8))


def test_the_walker_reproduces_a_hand_written_reference() -> None:
    rng = np.random.default_rng(20260831)
    x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    (produced,) = execute_module(MODULE, [x])
    np.testing.assert_allclose(produced, reference(x), rtol=0, atol=0)


def test_the_attributes_are_read_and_not_defaulted() -> None:
    """A walker that ignored `pads` would still produce a plausible tensor.

    It would produce a smaller one here, which the shape check catches, so the
    assertion is made against a case where the wrong attribute gives the right
    shape: the pooling stride. With `strides` defaulted to one the pool would
    be 3 by 3 rather than 2 by 2, so the shape moves; with `kernel` defaulted
    there is no default at all and the walker raises. What is left is `pads`,
    checked by the value at the corner, which padding changes and shape does
    not.
    """
    x = np.ones((1, 2, 4, 4), dtype=np.float32)
    (produced,) = execute_module(MODULE, [x])
    expected = reference(x)
    np.testing.assert_allclose(produced, expected, rtol=0, atol=0)

    # The corner of the padded convolution sums four taps rather than nine, so
    # it differs from the interior. A walker that dropped `pads` would produce
    # a uniform first channel here.
    assert not np.allclose(expected, expected.flat[0])


def test_an_operation_the_reference_does_not_know_raises_by_name() -> None:
    module = MODULE.replace(
        "%flat = npu.reshape %pooled : tensor<1x2x2x2xf32> to tensor<1x8xf32>",
        "%flat = npu.reshape %pooled : tensor<1x2x2x2xf32> to tensor<1x8xf32>\n"
        "  %unknown = npu.relu ins(%flat : tensor<1x8xf32>) "
        "outs(%flat : tensor<1x8xf32>) -> tensor<1x8xf32>",
    )
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    with pytest.raises(ExecutionError, match="destination operand is not a"):
        execute_module(module, [x])


def test_the_wrong_number_of_inputs_is_refused() -> None:
    with pytest.raises(ExecutionError, match="takes 1 arguments"):
        execute_module(MODULE, [])


def test_an_input_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ExecutionError, match="has shape"):
        execute_module(MODULE, [np.zeros((1, 2, 4, 5), dtype=np.float32)])


def test_a_module_that_is_not_this_projects_ir_is_refused() -> None:
    with pytest.raises(ExecutionError, match="reprint this module"):
        execute_module("func.func @main() { this is not IR }", [])


def test_a_function_by_another_name_is_refused() -> None:
    with pytest.raises(ExecutionError, match="functions named"):
        execute_module(
            MODULE, [np.zeros((1, 2, 4, 4), dtype=np.float32)], function_name="other"
        )
