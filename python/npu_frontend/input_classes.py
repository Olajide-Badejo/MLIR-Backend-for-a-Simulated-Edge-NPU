# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The five input classes of Section 17.4, each seeded so a failure reproduces.

One random input is not evidence. The matrix drives every cell with five
structurally different inputs, and each of them exists to reach a regime the
others cannot:

============  ===============================================================
``normal``    a standard normal draw. The ordinary case.
``zeros``     all zeros. Every reference value is exactly zero, so the
              relative bound is vacuous and only the absolute one is asserted.
``large_pos`` a constant large positive magnitude. Every ReLU passes, and the
              accumulations are long sums of same signed terms, which is where
              a summation order difference shows up largest.
``large_neg`` the same magnitude negative. Every ReLU is on its dead side, so
              a kernel that got the comparison backwards produces a graph of
              zeros where the reference has none, or the reverse.
``relu_knee`` a draw concentrated near zero, so values straddle the knee. The
              case where an off by one epsilon in a comparison changes an
              answer's sign rather than its last bits.
============  ===============================================================

**The seed is derived from the cell, not drawn from a counter.** A cell is a
model, a batch size and a class, and `class_seed` hashes exactly that triple.
So the input a failing cell used is reconstructible from the cell's own name
with no run log, which is what Section 17.4 means by "seeded so a failure
reproduces", and two cells never share a draw by accident. `zlib.crc32` is used
rather than `hash`, because `hash` on a string is salted per process and a seed
that changed between two runs would be the opposite of what this file is for.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: The five classes, in the order Section 17.4 lists them.
INPUT_CLASSES: Final[tuple[str, ...]] = (
    "normal",
    "zeros",
    "large_pos",
    "large_neg",
    "relu_knee",
)

#: The magnitude of the two constant classes.
#:
#: Ten rather than something larger. The suite's convolutions accumulate a few
#: hundred terms against weights of order 0.2, so ten is about fifty standard
#: deviations away from the normal class and unambiguously "large", while
#: staying far enough from f32's range that the comparison measures the
#: compiler rather than the float format. A magnitude of 1e30 would make every
#: cell a test of overflow behaviour, which is a real question and not this
#: matrix's question.
LARGE_MAGNITUDE: Final[float] = 10.0

#: The scale of the ``relu_knee`` draw.
#:
#: A thousandth of the normal class, so the values sit inside the band where a
#: ReLU's decision is made and both sides of the knee are populated. It is not
#: zero: a draw that collapsed to zero would be the ``zeros`` class under a
#: second name.
KNEE_SCALE: Final[float] = 1e-3


def class_seed(model: str, batch: int, input_class: str) -> int:
    """The seed for one cell, derived from the cell's own identity.

    Stable across processes, machines and Python versions, which `hash` is not:
    string hashing is salted per process by default, and a seed that changed
    between two runs of the same command would defeat the whole point.
    """
    if input_class not in INPUT_CLASSES:
        raise KeyError(
            f"{input_class!r} is not an input class. The five are "
            + ", ".join(INPUT_CLASSES)
            + "."
        )
    return zlib.crc32(f"{model}:{batch}:{input_class}".encode())


def make_input(
    input_class: str, shape: Sequence[int], *, seed: int
) -> NDArray[np.float32]:
    """One input tensor of the named class.

    The two constant classes are exactly constant rather than a draw around a
    constant, because their job is to put every activation on one side of every
    decision and a draw would put a few on the other.
    """
    extents = tuple(int(extent) for extent in shape)
    rng = np.random.default_rng(seed)

    if input_class == "normal":
        return rng.standard_normal(extents).astype(np.float32)
    if input_class == "zeros":
        return np.zeros(extents, dtype=np.float32)
    if input_class == "large_pos":
        return np.full(extents, LARGE_MAGNITUDE, dtype=np.float32)
    if input_class == "large_neg":
        return np.full(extents, -LARGE_MAGNITUDE, dtype=np.float32)
    if input_class == "relu_knee":
        return (rng.standard_normal(extents) * KNEE_SCALE).astype(np.float32)

    # Every class is listed rather than swept into a lookup with a default, so
    # a class added to INPUT_CLASSES and not to this function raises by name
    # instead of silently producing whatever the default was.
    raise KeyError(
        f"{input_class!r} is not an input class. The five are "
        + ", ".join(INPUT_CLASSES)
        + "."
    )


def make_inputs(
    input_class: str,
    shapes: Sequence[Sequence[int]],
    *,
    model: str,
    batch: int,
) -> list[NDArray[np.float32]]:
    """One tensor per graph input, all of the same class.

    Each argument gets its own draw, offset from the cell's seed by its
    position, so a graph with two inputs is not handed the same numbers twice.
    """
    base = class_seed(model, batch, input_class)
    return [
        make_input(input_class, shape, seed=base + index)
        for index, shape in enumerate(shapes)
    ]
