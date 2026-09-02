# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The Python mirror of the cost model agrees with the C++ header.

Section 5.5: the constants have exactly one home, ``CostModel.h``, and a Python
mirror that a test asserts equal to the header. This is that test. Two hand
copied sets of numbers is how a report starts lying, and the only thing standing
between this project and two hand copied sets of numbers is this file.

It parses the header rather than importing anything from it, and the header is
written to be parseable: every mirrored constant lives between two marker
comments, one per line, in the shape ``inline constexpr <type> <name> =
<literal>;``. A parser that had to understand C++ to read a table of constants
would be a worse mechanism than the drift it prevents.

The test fails in **both** directions on purpose. A constant added to the header
and not to the mirror fails, and so does one added to the mirror and not to the
header, because a mirror with an extra number in it is a number nothing checks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from npu_frontend import cost_model

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER = REPO_ROOT / "include" / "NPU" / "Simulator" / "CostModel.h"

BEGIN = "// NPU_COST_CONSTANTS_BEGIN"
END = "// NPU_COST_CONSTANTS_END"

CONSTANT_RE = re.compile(
    r"^inline\s+constexpr\s+(?:int64_t|double)\s+(?P<name>k\w+)\s*=\s*"
    r"(?P<value>-?[\d.]+)\s*;\s*$"
)


def header_constants() -> dict[str, float]:
    """Every constant between the markers, as a name to value mapping."""
    text = HEADER.read_text(encoding="utf-8")
    start = text.index(BEGIN) + len(BEGIN)
    stop = text.index(END)
    found: dict[str, float] = {}
    for line in text[start:stop].splitlines():
        match = CONSTANT_RE.match(line.strip())
        if match:
            found[match.group("name")] = float(match.group("value"))
    return found


def test_the_header_block_is_parseable() -> None:
    """The markers are there and the block is not empty.

    Without this, a header whose markers moved would make every assertion below
    pass over an empty table, which is the failure mode of every mechanical
    check: it stops checking rather than starting to fail.
    """
    assert HEADER.is_file(), HEADER
    found = header_constants()
    assert len(found) == len(cost_model.HEADER_NAMES), sorted(found)


def test_every_header_constant_is_mirrored() -> None:
    found = header_constants()
    missing = sorted(set(found) - set(cost_model.HEADER_NAMES))
    assert not missing, (
        f"these constants are in CostModel.h and not in the Python mirror: "
        f"{missing}. Add them to python/npu_frontend/cost_model.py in the same "
        f"commit as the header change."
    )


def test_every_mirrored_constant_is_in_the_header() -> None:
    found = header_constants()
    extra = sorted(set(cost_model.HEADER_NAMES) - set(found))
    assert not extra, (
        f"these constants are in the Python mirror and not in CostModel.h: "
        f"{extra}. A mirror with an extra number in it is a number nothing "
        f"checks."
    )


@pytest.mark.parametrize("header_name", sorted(cost_model.HEADER_NAMES))
def test_the_values_agree(header_name: str) -> None:
    found = header_constants()
    python_name = cost_model.HEADER_NAMES[header_name]
    assert found[header_name] == cost_model.VALUES[python_name], (
        f"{header_name} is {found[header_name]} in CostModel.h and "
        f"{python_name} is {cost_model.VALUES[python_name]} in the mirror"
    )


def test_the_mirrored_formulas_reproduce_the_headers() -> None:
    """The charges, not only the constants.

    Copying the numbers and paraphrasing the arithmetic would leave the mirror
    agreeing about every constant and disagreeing about every answer, which is
    the more expensive half of the same bug. These are the same cases
    ``CostModelTest.cpp`` asserts against the C++, written out again here so
    that the two suites are checking the same claims rather than adjacent ones.
    """
    assert cost_model.dma_cycles(64, 16, 1) == 4.0 + 64.0
    assert cost_model.dma_cycles(64, 16, 2) == 4.0 + 64.0 + 16.0 * 0.5

    charge = cost_model.gemm_charge(4, 40, 33, cost_model.PEAK_MACS_PER_CYCLE_F32)
    assert charge.macs == 4 * 40 * 33

    full = cost_model.gemm_charge(1024, 16, 16, cost_model.PEAK_MACS_PER_CYCLE_F32)
    assert full.utilization == 1.0

    narrow = cost_model.gemm_charge(1024, 1, 1, cost_model.PEAK_MACS_PER_CYCLE_F32)
    assert narrow.utilization == 1.0 / 256.0

    assert cost_model.overlap_fraction(100.0, 40.0, 140.0) == 0.0
    assert cost_model.overlap_fraction(100.0, 40.0, 100.0) == 1.0
    assert cost_model.overlap_fraction(0.0, 40.0, 40.0) == 0.0


def test_the_convolution_charge_composes_from_the_gemm_charge() -> None:
    """`conv2d_charge`, added to the mirror at P11, against the header's rule.

    The header states the mapping in words and the C++ implements it: per group
    the weight matrix is `(C / group) * kH * kW` by `O / group` and the streamed
    row count is `N * oH * oW`, and the whole convolution is the group charge
    multiplied by the group count with the occupancy terms left intensive. This
    asserts the Python does the same composition rather than an equivalent
    looking one.

    The depthwise case is the one that matters: with `group == C` each group
    presents a single column to a sixteen column array, and a mirror that had
    quietly divided by the group count instead of multiplying would agree about
    every dense convolution in the suite and be wrong about the one model that
    exists to expose it.
    """
    peak = cost_model.PEAK_MACS_PER_CYCLE_F32

    dense = cost_model.conv2d_charge(1, 6, 1, 28, 28, 5, 5, 1, peak)
    equivalent = cost_model.gemm_charge(1 * 28 * 28, 1 * 5 * 5, 6, peak)
    assert dense.macs == equivalent.macs
    assert dense.cycles == equivalent.cycles
    assert dense.utilization == equivalent.utilization
    assert dense.delta == equivalent.delta

    depthwise = cost_model.conv2d_charge(1, 8, 8, 8, 8, 3, 3, 8, peak)
    per_group = cost_model.gemm_charge(1 * 8 * 8, 1 * 3 * 3, 1, peak)
    assert depthwise.macs == per_group.macs * 8
    assert depthwise.cycles == per_group.cycles * 8
    # Intensive, and therefore not multiplied. A single column against a sixteen
    # column array is 9/16 of one row's worth of the array occupied.
    assert depthwise.utilization == per_group.utilization
    assert depthwise.utilization == pytest.approx((9 / 16) * (1 / 16))

    # A group count that does not divide the channels is refused with a zero
    # charge rather than producing a plausible wrong one.
    assert cost_model.conv2d_charge(1, 8, 8, 8, 8, 3, 3, 3, peak).macs == 0


@pytest.mark.parametrize(
    "cell",
    [
        "lenet-O2-default-n1-fp32-normal",
        "depthwise_separable-O2-default-n1-fp32-normal",
        "inception_block-O2-default-n1-fp32-normal",
        "dilated_stack-O2-default-n1-fp32-normal",
    ],
)
def test_the_convolution_mirror_reproduces_a_recorded_cell(cell: str) -> None:
    """And the same charge against the machine, on the programs the suite records.

    `test_the_mirror_reproduces_the_machines_numbers` above covers `gemm_charge`
    on one exported matmul. This covers `conv2d_charge` on every convolution of
    four real models at once, by walking the allocated IR and comparing the
    reconstructed totals with the ones the simulator recorded for the same cell.
    `utilization` and `delta` are the fields that make it a test of the charge
    rather than of the MAC count: `macs` would agree even if both occupancy terms
    had been dropped.
    """
    import sys
    import tempfile

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    import roofline
    from npu_frontend import npuisa_walk
    from npu_frontend.results import RESULTS_DIR, load_result

    tool("npu-opt")
    path = RESULTS_DIR / f"{cell}.json"
    if not path.is_file():
        pytest.skip(f"{cell} is not a committed cell in this checkout")
    result = load_result(path)

    with tempfile.TemporaryDirectory(prefix="npu-conv-mirror-") as directory:
        text = roofline.Compiler(Path(directory)).allocated_ir(result)
    operations = npuisa_walk.attribute_transfers(npuisa_walk.walk(text), text)

    # Raises naming every field that disagreed, including effective_macs,
    # utilization and delta.
    npuisa_walk.check_against_result(operations, result)

    walked = npuisa_walk.totals(operations)
    simulation = result["simulation"]
    assert walked.macs == int(simulation["macs"])
    assert walked.effective_macs == pytest.approx(
        float(simulation["effective_macs"]), rel=1e-12
    )
    assert walked.utilization == pytest.approx(
        float(simulation["utilization"]), rel=1e-12
    )
    assert walked.delta == pytest.approx(float(simulation["delta"]), rel=1e-12)


# ---------------------------------------------------------------------------
# The mirror against the machine, on a real program.
#
# Everything above compares the mirror with a header and with literals a reader
# can check. What it cannot catch is the mirror reproducing the header's
# constants and then charging with them differently, which is the more expensive
# half of the same bug and which was uncovered until D-0027. So one exported
# case is run through `npu-sim` and the numbers it prints are reconstructed from
# this file's own arithmetic.
#
# The case is a **narrow** matmul on purpose. A tile that fills the array has
# utilization exactly 1 and delta near 1, so a mirror that had dropped both
# terms would still agree; 5 by 19 by 3 folds into two tiles, neither of which
# fills the array, and every term is load bearing.
# ---------------------------------------------------------------------------

CASE = "matmul_narrow_bias"
CASE_M, CASE_K, CASE_N = 5, 19, 3


@pytest.fixture(scope="module")
def exported() -> Iterator[Path]:
    """The exported differential cases, which carry the program this test runs.

    **The lookup is `tools.tool` and no longer a copy of it**, for the reason
    D-0032 records: this module's own `build_directory()` skipped under
    `scripts/coverage.sh`, so the one test that checks the mirror against a real
    `npu-sim` run did not execute and the run reported success anyway.
    """
    exporter = tool("NPUSimulatorTests")
    # Resolved and discarded on purpose, so a missing simulator is one message
    # here rather than a failure inside the run below.
    tool("npu-sim")

    directory = Path(tempfile.mkdtemp(prefix="npu-costmodel-"))
    try:
        environment = dict(os.environ, NPU_DIFFERENTIAL_OUT=str(directory))
        completed = subprocess.run(
            [
                str(exporter),
                "--gtest_filter=Differential.TheCasesCanBeWrittenOutForTheReferenceInterpreter",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def run_simulator_stats(directory: Path, case: str) -> dict[str, float]:
    """Runs one exported program and returns the statistics it printed."""
    program = directory / f"{case}.nbin"
    assert program.is_file(), (
        f"{program} was not exported. This test names one case out of "
        f"DifferentialExport.cpp; if that case was renamed, rename it here too "
        f"rather than deleting the assertion."
    )
    command = [str(tool("npu-sim")), str(program)]
    operands = sorted(directory.glob(f"{case}.in*.bin"))
    for operand in operands:
        command += ["--input", str(operand)]
    command += ["--output", str(directory / f"{case}.cost.bin")]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    stats: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        name, _, value = line.partition(":")
        if value.strip():
            stats[name.strip()] = float(value)
    assert stats, completed.stdout
    return stats


def test_the_mirror_reproduces_the_machines_own_numbers(exported: Path) -> None:
    """The charges, checked against `npu-sim` rather than against literals."""
    stats = run_simulator_stats(exported, CASE)
    charge = cost_model.gemm_charge(
        CASE_M, CASE_K, CASE_N, cost_model.PEAK_MACS_PER_CYCLE_F32
    )

    # Raw, and the same number on both sides. Everything else in this test is a
    # float; this one is a count and is compared exactly.
    assert stats["macs"] == charge.macs == CASE_M * CASE_K * CASE_N
    assert stats["int8 macs"] == 0

    # The tolerance is the printed precision and nothing more. npu-sim prints
    # the occupancy terms at six decimal places and the cycle figures at four,
    # so a disagreement anywhere above half a unit in the last printed place is
    # a real one.
    assert stats["utilization"] == pytest.approx(charge.utilization, abs=5e-7)
    assert stats["delta"] == pytest.approx(charge.delta, abs=5e-7)
    assert stats["effective macs"] == pytest.approx(charge.effective_macs, abs=5e-5)

    # The compute timeline is the matmul's charge plus the issue overhead of the
    # two instructions on that port, the matmul and the HALT. Written out rather
    # than read back from the tool, because the point is to predict it.
    expected_compute = charge.cycles + 2 * cost_model.ISSUE_OVERHEAD_CYCLES
    assert stats["compute cycles"] == pytest.approx(expected_compute, abs=5e-5)

    # And the DMA timeline: three contiguous loads and one contiguous store,
    # each paying bytes over bandwidth, a descriptor, and the issue overhead.
    element_counts = [CASE_M * CASE_K, CASE_K * CASE_N, CASE_N, CASE_M * CASE_N]
    expected_dma = sum(
        cost_model.dma_cycles(4 * count, count, 1) + cost_model.ISSUE_OVERHEAD_CYCLES
        for count in element_counts
    )
    assert stats["dma cycles"] == pytest.approx(expected_dma, abs=5e-5)

    # The two timelines are independent, so the total is neither their sum nor
    # either one of them by accident: the matmul waits for its operands, and the
    # store waits for the matmul. Asserting the bracket is what stops this test
    # passing against a simulator that had quietly gone back to one port.
    assert stats["cycles"] > max(expected_dma, expected_compute)
    assert stats["cycles"] < expected_dma + expected_compute
    assert 0.0 < stats["overlap fraction"] < 1.0
