# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The simulator and the reference interpreter agree, on randomized inputs.

Section 17.3a's first oracle. The reference is ``npu_frontend.refexec``, written
in numpy from the ODS descriptions; the subject is ``npu-sim`` running a
``.nbin``. Neither has seen the other's arithmetic, so agreement between them is
evidence about the arithmetic rather than about one of them.

**How a case gets here.** ``unittests/Simulator/DifferentialExport.cpp`` writes,
for each case, the ``.nbin`` that carries out one ``npu`` operation, the raw f32
bytes of each operand, and a manifest naming the operation and its attributes.
This file loads the manifest, runs ``npu-sim`` over the program with those
inputs, runs ``refexec`` over the same numbers, and compares. The manifest is the
one thing both sides share, and it is deliberately small enough to read: it says
what the case is *supposed* to be, and a manifest that said the wrong thing
would make both sides wrong in the same way.

**The tolerance is the honest one.** The two implementations sum their floating
point terms in different orders, which is a property of their being independent
rather than a defect, so they agree to a tolerance and not bitwise. Bitwise
agreement is asserted where it is meaningful, which is between two runs of the
same implementation at different thread counts, and that lives in
``unittests/Simulator/DeterminismTest.cpp``.

**Proving this test can fail** is part of the Phase P7 gate: a deliberately
perturbed kernel has to make the comparison go red. The recipe and the observed
output are recorded in ``docs/PHASE_STATE.md``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from npu_frontend import refexec

from tools import tool

REPO_ROOT = Path(__file__).resolve().parents[2]

# The tolerance for a value produced by two different summation orders. It is
# generous relative to the magnitudes involved, which are inputs in [-1, 1) and
# reductions of at most a few hundred terms, and it is one number rather than a
# per case table so that a case cannot be quietly given room to disagree.
RTOL = 1e-5
ATOL = 1e-5


@pytest.fixture(scope="module")
def exported() -> Iterator[Path]:
    """Runs the exporter once and yields the directory it wrote.

    Module scoped because the export is a subprocess and twenty cases do not
    need twenty of them.

    **The lookup is `tools.tool` and no longer a copy of it.** This module used
    to carry its own `build_directory()`, which looked at `$NPU_BUILD_DIR` and
    `<repo>/build` and nothing else, and which **skipped** when it found
    neither. Under `scripts/coverage.sh` that skip fired on all four tests in
    this file, silently, and the coverage number was then measured from a run in
    which the reference interpreter differential had not executed at all. That
    is D-0032, and `test/Python/tools.py` is the one rule it produced.
    """
    exporter = tool("NPUSimulatorTests")
    # Resolved and discarded on purpose. `run_simulator` looks it up again per
    # case; doing it once here means a missing simulator is one message before
    # the first case rather than the same message twenty four times.
    tool("npu-sim")

    directory = Path(tempfile.mkdtemp(prefix="npu-differential-"))
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
        manifest = directory / "manifest.json"
        assert manifest.is_file(), completed.stdout + completed.stderr
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def load_cases(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    assert cases, "the exporter wrote an empty manifest"
    return cases


def read_f32(path: Path, shape: list[int]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    assert values.size == int(np.prod(shape)), (
        f"{path.name} holds {values.size} floats and its shape {shape} needs "
        f"{int(np.prod(shape))}"
    )
    return values.reshape(shape)


def run_simulator(directory: Path, case: dict) -> np.ndarray:
    output = directory / f"{case['name']}.sim.bin"
    command = [str(tool("npu-sim")), str(directory / case["program"])]
    for operand in case["inputs"]:
        command += ["--input", str(directory / operand["file"])]
    command += ["--output", str(output), "--quiet"]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, (
        f"{case['name']}: npu-sim exited {completed.returncode}\n"
        f"{completed.stdout}{completed.stderr}"
    )
    return read_f32(output, case["result_shape"])


def test_the_export_covers_every_operation_refexec_can_run(exported: Path) -> None:
    """Every executable operation of the dialect appears in the case set.

    Without this the differential suite could pass while silently covering four
    operations, which is the failure mode of every table driven test: it stops
    growing and nobody notices, because the tests it does have keep passing.

    ``batch_norm`` is the one executable operation absent by design. It has no
    opcode, it decomposes at lowering into a multiply and an add, and its
    comparison against the simulator arrives at Phase P8 with the end to end
    pipeline. ``test_refexec.py`` pins the decomposition rule in the meantime.
    """
    covered = {case["operation"] for case in load_cases(exported)}
    expected = {
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
    }
    assert expected <= covered, sorted(expected - covered)


def test_the_exported_inputs_straddle_zero(exported: Path) -> None:
    """The exported bytes exercise both signs, which is D-0029's guard.

    The generator in ``DifferentialExport.cpp`` has its own assertion about its
    range. This one is here as well rather than instead, because it checks the
    thing that matters, which is the numbers that actually reached the files,
    and it survives a rewrite of the C++ that keeps the comment and loses the
    property.

    The defect it exists to catch was one bit: a 33 bit shift instead of a 32
    bit one, which left every value in ``[-1, 0)``. Nothing looked wrong. The
    files were still random, still deterministic, still reproducible. What they
    were not was capable of exercising a relu, whose reference and simulated
    answers were both all zeros, or a maximum whose answer is positive.
    """
    cases = load_cases(exported)

    # Pooled across the whole export, because that is the claim: the generator
    # covers the interval it says it covers. Both ends and not merely both
    # signs, since a generator producing [-1, 0.01) would satisfy "something is
    # positive" and still be the same defect.
    everything = np.concatenate(
        [
            read_f32(exported / operand["file"], operand["shape"]).ravel()
            for case in cases
            for operand in case["inputs"]
        ]
    )
    assert everything.min() < -0.9, everything.min()
    assert everything.max() > 0.9, everything.max()

    # And per operand, for the operands large enough that one sign would be a
    # signal rather than a coincidence. The threshold is not decoration: the
    # three element bias of `matmul_narrow_bias` is all positive in this export,
    # which happens to a fair three element sample one time in four, and a rule
    # that called that a defect would be a rule somebody eventually deletes.
    # Sixteen elements puts the same coincidence at one in 32768.
    for case in cases:
        for operand in case["inputs"]:
            values = read_f32(exported / operand["file"], operand["shape"])
            if values.size < 16:
                continue
            assert values.min() < 0.0 < values.max(), (
                f"{case['name']}: {operand['file']} holds {values.size} values "
                f"of one sign, so this case exercises half the number line"
            )

    # And the case that goes vacuous first, checked as an answer rather than as
    # an input: a relu over negative numbers only is a comparison of zeros.
    relu = next(case for case in cases if case["operation"] == "relu")
    operands = [
        read_f32(exported / operand["file"], operand["shape"])
        for operand in relu["inputs"]
    ]
    reference = refexec.execute("relu", operands, {})
    assert np.count_nonzero(reference) > 0, (
        "the relu case's reference output is entirely zero, so comparing it "
        "against the simulator asserts nothing"
    )


def test_every_case_agrees(exported: Path) -> None:
    """The gate item: the two agree on every operation, on randomized inputs."""
    cases = load_cases(exported)
    disagreements: list[str] = []

    for case in cases:
        operands = [
            read_f32(exported / operand["file"], operand["shape"])
            for operand in case["inputs"]
        ]
        reference = refexec.execute(
            case["operation"], operands, dict(case["attributes"])
        )
        assert list(reference.shape) == case["result_shape"], (
            f"{case['name']}: the reference produced {reference.shape} and the "
            f"program declares {case['result_shape']}"
        )

        produced = run_simulator(exported, case)
        try:
            np.testing.assert_allclose(
                produced, reference, rtol=RTOL, atol=ATOL, err_msg=case["name"]
            )
        except AssertionError as failure:  # noqa: PERF203
            disagreements.append(f"{case['name']}: {failure}")

    assert not disagreements, "\n\n".join(disagreements)


def test_the_comparison_is_not_vacuous(exported: Path) -> None:
    """A perturbed reference makes the comparison fail.

    The gate asks for a deliberately perturbed kernel to make the differential
    comparison go red, and the full version of that is a C++ edit recorded in
    ``docs/PHASE_STATE.md``. This is the cheap standing version of the same
    claim, and it is here because a recipe in a document is not a thing CI runs:
    it perturbs the reference side by one part in a thousand and asserts the
    comparison notices. A tolerance quietly widened until everything passed
    would fail this test.
    """
    cases = load_cases(exported)
    checked = 0
    for case in cases:
        operands = [
            read_f32(exported / operand["file"], operand["shape"])
            for operand in case["inputs"]
        ]
        reference = refexec.execute(
            case["operation"], operands, dict(case["attributes"])
        )
        if not np.any(np.abs(reference) > 1e-3):
            # A case whose every output is near zero cannot be perturbed
            # relatively, and pretending otherwise would be the vacuous test
            # this one exists to prevent.
            continue
        perturbed = reference * np.float32(1.001)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(perturbed, reference, rtol=RTOL, atol=ATOL)
        checked += 1

    assert checked >= len(cases) // 2, (
        f"only {checked} of {len(cases)} cases had outputs large enough to "
        f"perturb, so this test checked less than it claims"
    )
