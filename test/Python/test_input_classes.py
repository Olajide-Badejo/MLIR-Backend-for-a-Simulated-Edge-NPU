# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The five input classes of Section 17.4, and the seed that reproduces a cell.

Small tests over a small module, and every one of them exists because the
alternative failure is silent. A class that collapsed to another class, a seed
that changed between runs, a draw that never crossed zero: none of those makes
a matrix go red, they make it go quietly weaker. That is exactly the shape of
D-0029, where a generator's one bit error left every value negative and
twenty four differential cases kept passing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from npu_frontend.input_classes import (
    INPUT_CLASSES,
    KNEE_SCALE,
    LARGE_MAGNITUDE,
    class_seed,
    make_input,
    make_inputs,
)

SHAPE = (2, 3, 4, 4)


def test_there_are_five_classes_and_they_are_the_documents_five() -> None:
    assert INPUT_CLASSES == (
        "normal",
        "zeros",
        "large_pos",
        "large_neg",
        "relu_knee",
    )


@pytest.mark.parametrize("input_class", INPUT_CLASSES)
def test_a_class_is_deterministic_from_its_seed(input_class: str) -> None:
    first = make_input(input_class, SHAPE, seed=17)
    second = make_input(input_class, SHAPE, seed=17)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    assert first.shape == SHAPE


def test_the_normal_class_straddles_zero() -> None:
    """The property D-0029 cost this project once already."""
    values = make_input("normal", (64, 64), seed=3)
    assert values.min() < 0.0 < values.max()


def test_the_zeros_class_is_exactly_zero() -> None:
    values = make_input("zeros", SHAPE, seed=3)
    assert np.array_equal(values, np.zeros(SHAPE, dtype=np.float32))


def test_the_two_constant_classes_are_constant_and_opposite() -> None:
    """Exactly constant, not a draw around a constant.

    Their job is to put every activation on one side of every decision, and a
    draw would put a few on the other, which is the class doing nothing that
    `normal` does not already do.
    """
    positive = make_input("large_pos", SHAPE, seed=3)
    negative = make_input("large_neg", SHAPE, seed=3)
    assert np.all(positive == np.float32(LARGE_MAGNITUDE))
    assert np.all(negative == np.float32(-LARGE_MAGNITUDE))
    np.testing.assert_array_equal(positive, -negative)


def test_the_knee_class_is_near_zero_and_is_not_the_zeros_class() -> None:
    values = make_input("relu_knee", (64, 64), seed=3)
    assert values.min() < 0.0 < values.max()
    assert np.abs(values).max() < 20 * KNEE_SCALE
    assert np.count_nonzero(values) == values.size


def test_two_classes_never_produce_the_same_array() -> None:
    """A class that collapsed onto another would weaken the matrix silently."""
    drawn = {
        name: make_input(name, SHAPE, seed=class_seed("m", 1, name))
        for name in INPUT_CLASSES
    }
    for first in INPUT_CLASSES:
        for second in INPUT_CLASSES:
            if first < second:
                assert not np.array_equal(
                    drawn[first], drawn[second]
                ), f"{first} and {second} produced the same array"


def test_the_seed_is_the_cells_own_identity() -> None:
    assert class_seed("lenet", 1, "normal") != class_seed("lenet", 4, "normal")
    assert class_seed("lenet", 1, "normal") != class_seed("lenet", 1, "zeros")
    assert class_seed("lenet", 1, "normal") != class_seed("resnet_block", 1, "normal")
    assert class_seed("lenet", 1, "normal") == class_seed("lenet", 1, "normal")


def test_the_seed_survives_a_new_process() -> None:
    """`hash` is salted per process and `zlib.crc32` is not.

    A seed that changed between two runs of the same command would defeat the
    whole point of deriving it from the cell, and the failure would look like a
    flaky test rather than like a seeding bug.

    The child's `PYTHONPATH` is built here rather than inherited. `pythonpath`
    in `pyproject.toml` puts this project's package root on *this* process's
    `sys.path` and does not export anything, so a child that inherited the
    environment would find `npu_frontend` only when the caller happened to have
    exported it. That is D-0030: this test passed under a developer's wrapper
    script and failed under `scripts/coverage.sh`, which is the same command
    with one variable fewer.
    """
    here = class_seed("lenet", 4, "relu_knee")
    package_root = str(Path(__file__).resolve().parents[2] / "python")
    existing = os.environ.get("PYTHONPATH", "")
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([package_root, *filter(None, [existing])]),
    )
    elsewhere = subprocess.run(
        [
            sys.executable,
            "-c",
            "from npu_frontend.input_classes import class_seed;"
            "print(class_seed('lenet', 4, 'relu_knee'))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert elsewhere.returncode == 0, elsewhere.stderr
    assert int(elsewhere.stdout.strip()) == here


def test_each_argument_of_a_graph_gets_its_own_draw() -> None:
    arrays = make_inputs("normal", [(8, 8), (8, 8)], model="m", batch=1)
    assert len(arrays) == 2
    assert not np.array_equal(arrays[0], arrays[1])


def test_a_class_that_is_not_a_class_raises_by_name() -> None:
    with pytest.raises(KeyError, match="relu_knee"):
        make_input("gaussian", SHAPE, seed=1)
    with pytest.raises(KeyError, match="relu_knee"):
        class_seed("m", 1, "gaussian")
