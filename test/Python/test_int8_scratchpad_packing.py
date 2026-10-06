# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The packed scratchpad sensitivity, and the split it rests on.

*Added at P14.* `experiments/int8_scratchpad_packing.py` splits a program's
scratchpad element accesses by element type, because `Stats` counts them
without one, and packs the int8 ones four to a word. The split is only worth
anything if it partitions what the machine counted, so that is what is held
here, on a quantized cell and on its fp32 twin, where nothing may pack.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from npu_frontend.results import RESULTS_DIR, load_result

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import int8_scratchpad_packing as packing  # noqa: E402
import roofline  # noqa: E402


def committed(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"{name} is not a committed cell in this checkout")
    return load_result(path)


def test_the_split_partitions_what_the_machine_counted() -> None:
    """Every scratchpad element the machine counted is in exactly one type.

    On the quantized cell the int8 elements are a real share of the traffic and
    the int32 and f32 ones are not zero either, because the bias, the rescale
    table and the f32 side of `QUANT` and `DEQUANT` are 32 bit; on the fp32
    twin there is no int8 element, so the packed counts are the recorded ones.
    """
    tool("npu-opt")
    quantized = committed("lenet-O2-default-n1-int8-normal")
    twin = committed("lenet-O2-default-n1-fp32-normal")
    with tempfile.TemporaryDirectory(prefix="npu-int8-packing-test-") as directory:
        compiler = roofline.Compiler(Path(directory))
        quantized_split = packing.split(compiler.allocated_ir(quantized))
        twin_split = packing.split(compiler.allocated_ir(twin))

    packing.check(quantized_split, quantized)
    packing.check(twin_split, twin)

    assert quantized_split.read_by_type["i8"] > 0
    assert quantized_split.read_by_type["i32"] > 0
    assert quantized_split.read_by_type["f32"] > 0
    assert quantized_split.packed_read < quantized_split.read
    assert quantized_split.packed_written < quantized_split.written
    # Four to a word at best, so never below a quarter of the int8 traffic.
    assert 4 * quantized_split.int8_read_packed >= quantized_split.read_by_type["i8"]

    assert "i8" not in twin_split.read_by_type
    assert "i8" not in twin_split.written_by_type
    assert twin_split.packed_read == twin_split.read
    assert twin_split.packed_written == twin_split.written


def test_a_split_that_does_not_partition_the_count_is_refused() -> None:
    """A split of a different program than the cell's is refused by name."""
    tool("npu-opt")
    quantized = committed("lenet-O2-default-n1-int8-normal")
    twin = committed("lenet-O2-default-n1-fp32-normal")
    with tempfile.TemporaryDirectory(prefix="npu-int8-packing-test-") as directory:
        twin_split = packing.split(
            roofline.Compiler(Path(directory)).allocated_ir(twin)
        )
    with pytest.raises(packing.PackingError, match="partition"):
        packing.check(twin_split, quantized)
