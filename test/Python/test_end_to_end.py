# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The end to end matrix of Section 17.4, at `-O0`.

ONNX in, a simulated answer out, checked against two oracles that do not know
about each other.

**Two oracles, because they localise differently.** `onnxruntime` runs the ONNX
graph, so a disagreement with it implicates the whole chain, importer included.
`refexec`, through `refgraph`, runs the `npu` IR the importer produced, so a
disagreement with it implicates only what is below the importer. A cell that
agrees with `refexec` and disagrees with `onnxruntime` has an importer bug, and
that is a diagnosis rather than the beginning of a search. Section 17.3a calls
the reference interpreter the load bearing part and Section 17.4 asks for the
external comparison; this file does both over the same cells.

**The matrix.** Seven models, two batch sizes, five input classes, at `-O0`,
which is the only level that exists at this phase. Section 17.4's full matrix
also sweeps levels and budgets; those axes arrive with the levels at P9 and are
absent here rather than faked, because a cell that compiled at a level the
compiler cannot emit would be a cell measuring nothing.

**Both bounds are asserted separately**, which is Section 17.4's rule and not a
style preference. `np.testing.assert_allclose` tests
`|a - b| <= atol + rtol * |b|`, one combined predicate in which a large `atol`
hides a failed relative bound and a large `rtol` hides a failed absolute one.
The two assertions below are two statements about the answer.

**The relative bound is taken against the reference's largest magnitude.** An
elementwise `|a - b| / |b|` is undefined where the reference is zero and
enormous where it is nearly zero, and every model here produces outputs
straddling zero, so an elementwise rule would be a rule about how close to zero
a value happened to land. The infinity norm of the reference is the scale of the
answer, and the error relative to it is the number a reader means by "relative
error" for a tensor.

**A reference that is exactly zero everywhere is a real case here and Section
17.4 predicted the wrong cell for it.** The specification expects the `zeros`
input class to produce an exactly zero reference and says the relative bound is
vacuous there. It is not: every model in this suite has biases, so a zero input
produces a nonzero answer and the relative bound bites normally. The class that
does produce an all zero reference is `large_neg` on `resnet_block`, where every
value is on the dead side of a closing ReLU. Where that happens the relative
bound is undefined and this file asserts something stronger instead: that the
simulated answer is exactly zero too, with no tolerance at all.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
import pytest
from npu_frontend import (
    MODELS,
    CompileResult,
    compile_model,
    generate_model,
    run_program,
)
from npu_frontend.input_classes import INPUT_CLASSES, make_inputs
from npu_frontend.refgraph import execute_module

#: The only level that exists at this phase.
LEVEL: Final[int] = 0

#: Section 17.4's two batch sizes.
BATCHES: Final[tuple[int, ...]] = (1, 4)

# --------------------------------------------------------------------------
# The tolerances.
#
# Measured on 2026-08-31 over the whole matrix, on this toolchain, and written
# down here because Section 17.4 requires the observed value beside the bound
# that was set from it.
#
#   against onnxruntime   worst absolute 4.77e-06   worst relative 8.08e-07
#   against refexec       worst absolute 4.77e-06   worst relative 4.17e-07
#
# Both worst cases are the `dilated_stack` and `depthwise_separable` cells at
# the two constant classes, whose answers have magnitudes in the tens and whose
# convolutions accumulate a few hundred same signed terms. That is where a
# summation order difference is largest, and a summation order difference is
# what these numbers are: the reference sums whole tensor slices per kernel
# position, the simulator walks one output element at a time, and neither is
# wrong.
#
# The bounds are roughly ten and six times the observed maxima. Not two, and
# the reason is stated rather than left as taste: this suite runs on at least
# two hosts, the developer machine and the CI container, and `onnxruntime`
# chooses its own vectorisation per host. A bound two times the observed value
# on one machine is a bound that goes red on another for a reason that is not a
# defect. Ten is far below the 1e-3 against 3e-8 that Section 17.4 names as the
# tolerance that cannot fail.
#
# Never loosened to make a cell pass. If a cell needs a wider bound, that is a
# finding: record the measured value and say why the bound moved.
# --------------------------------------------------------------------------

ABSOLUTE_TOLERANCE: Final[float] = 5e-5
RELATIVE_TOLERANCE: Final[float] = 5e-6


def _cross_product() -> list[tuple[str, int, str]]:
    """The matrix, computed rather than written out.

    `test_the_collected_cells_are_the_cross_product` recomputes this from the
    three registries and compares, so a cell filtered out of the list below
    fails rather than disappearing quietly.
    """
    return [
        (model, batch, input_class)
        for model in MODELS
        for batch in BATCHES
        for input_class in INPUT_CLASSES
    ]


CELLS: Final[list[tuple[str, int, str]]] = _cross_product()

#: The id a failure reports. Section 17.4: name the class in the test id so a
#: failure says which input broke, and the seed is derived from this triple, so
#: the id is also the reproduction recipe.
CELL_IDS: Final[list[str]] = [
    f"{model}-n{batch}-{input_class}" for model, batch, input_class in CELLS
]


class Suite:
    """The compiled suite, built once and reused across cells.

    Compilation and export are the expensive parts and neither depends on the
    input class, so a cache keyed on the model and the batch turns seventy
    cells into fourteen compilations.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._onnx: dict[tuple[str, int], Path] = {}
        self._programs: dict[tuple[str, int], CompileResult] = {}
        self._ir: dict[tuple[str, int], str] = {}
        self._sessions: dict[tuple[str, int], ort.InferenceSession] = {}

    def onnx(self, model: str, batch: int) -> Path:
        key = (model, batch)
        if key not in self._onnx:
            self._onnx[key] = generate_model(model, self._directory, batch=batch)
        return self._onnx[key]

    def program(self, model: str, batch: int) -> CompileResult:
        key = (model, batch)
        if key not in self._programs:
            self._programs[key] = compile_model(
                self.onnx(model, batch), level=LEVEL, emit="nbin"
            )
        return self._programs[key]

    def npu_ir(self, model: str, batch: int) -> str:
        key = (model, batch)
        if key not in self._ir:
            result = compile_model(self.onnx(model, batch), level=LEVEL, emit="npu")
            assert result.text is not None
            self._ir[key] = result.text
        return self._ir[key]

    def session(self, model: str, batch: int) -> ort.InferenceSession:
        key = (model, batch)
        if key not in self._sessions:
            self._sessions[key] = ort.InferenceSession(
                str(self.onnx(model, batch)), providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def onnxruntime_answer(
        self, model: str, batch: int, arrays: list[np.ndarray]
    ) -> list[np.ndarray]:
        session = self.session(model, batch)
        names = [entry.name for entry in session.get_inputs()]
        fed = dict(zip(names, arrays, strict=True))
        return [np.asarray(value, dtype=np.float32) for value in session.run(None, fed)]


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Suite]:
    yield Suite(tmp_path_factory.mktemp("end-to-end"))


def assert_both_bounds(
    produced: np.ndarray, reference: np.ndarray, cell: str, oracle: str
) -> None:
    """The two bounds of Section 17.4, asserted as two statements.

    Not one `assert_allclose`. `|a - b| <= atol + rtol * |b|` is a single
    predicate in which either bound can hide the other's failure, and the
    document is explicit that asserting an absolute bound and a relative bound
    is not the same thing.
    """
    assert produced.shape == reference.shape, (
        f"{cell}: {oracle} produced {reference.shape} and the simulator "
        f"{produced.shape}"
    )
    assert np.all(np.isfinite(produced)), (
        f"{cell}: the simulated answer holds a value that is not finite, which "
        f"no tolerance covers"
    )

    difference = np.abs(produced.astype(np.float64) - reference.astype(np.float64))
    absolute = float(difference.max())
    assert absolute <= ABSOLUTE_TOLERANCE, (
        f"{cell}: the absolute error against {oracle} is {absolute:.3e}, above "
        f"the bound of {ABSOLUTE_TOLERANCE:.3e}. This bound is not to be "
        f"widened to make the cell pass; record the measured value and say why "
        f"it moved."
    )

    scale = float(np.abs(reference.astype(np.float64)).max())
    if scale == 0.0:
        # The reference is zero everywhere, so a relative bound has nothing to
        # be relative to. The stronger statement is asserted instead: the
        # simulated answer is exactly zero as well, with no tolerance at all.
        assert np.array_equal(produced, np.zeros_like(produced)), (
            f"{cell}: {oracle} is exactly zero everywhere and the simulator is "
            f"not. A relative bound is vacuous here, so this is the assertion "
            f"that carries the cell."
        )
        return

    relative = absolute / scale
    assert relative <= RELATIVE_TOLERANCE, (
        f"{cell}: the relative error against {oracle} is {relative:.3e}, above "
        f"the bound of {RELATIVE_TOLERANCE:.3e}. The reference's largest "
        f"magnitude is {scale:.3e} and the absolute error is {absolute:.3e}."
    )


# ---------------------------------------------------------------------------
# The matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("model", "batch", "input_class"), CELLS, ids=CELL_IDS)
def test_the_simulated_answer_matches_onnxruntime(
    suite: Suite, model: str, batch: int, input_class: str
) -> None:
    """The gate item: every model matches onnxruntime at -O0.

    All five input classes, both batch sizes, both tolerances asserted
    separately. The drive is through the real `-O0` pipeline, because a test
    that ran a hardcoded pass list matching no optimization level would enforce
    nothing.
    """
    cell = f"{model}-n{batch}-{input_class}"
    program = suite.program(model, batch)
    assert program.binary is not None

    arrays = make_inputs(input_class, program.input_shapes, model=model, batch=batch)
    expected = suite.onnxruntime_answer(model, batch, arrays)
    answer = run_program(program.binary, arrays, program.output_shapes)

    assert len(answer.outputs) == len(expected)
    for produced, reference in zip(answer.outputs, expected, strict=True):
        assert_both_bounds(produced, reference, cell, "onnxruntime")


@pytest.mark.parametrize(("model", "batch", "input_class"), CELLS, ids=CELL_IDS)
def test_the_simulated_answer_matches_the_reference_interpreter(
    suite: Suite, model: str, batch: int, input_class: str
) -> None:
    """The other oracle, which localises a failure below the importer.

    `refexec` never saw the simulator's kernels and the simulator never saw
    `refexec`. What they share is the IR, so agreement here is evidence about
    lowering, allocation, encoding and the kernels, and nothing about the ONNX
    graph they both came from.
    """
    cell = f"{model}-n{batch}-{input_class}"
    program = suite.program(model, batch)
    assert program.binary is not None

    arrays = make_inputs(input_class, program.input_shapes, model=model, batch=batch)
    reference = execute_module(suite.npu_ir(model, batch), arrays)
    answer = run_program(program.binary, arrays, program.output_shapes)

    assert len(answer.outputs) == len(reference)
    for produced, expected in zip(answer.outputs, reference, strict=True):
        assert_both_bounds(produced, expected, cell, "refexec")


# ---------------------------------------------------------------------------
# The matrix's own shape.
# ---------------------------------------------------------------------------


def test_the_collected_cells_are_the_cross_product() -> None:
    """A silently dropped cell fails.

    The failure mode of a parametrized suite is that somebody filters the list
    and every remaining cell keeps passing. The cross product is recomputed
    here from the three registries and compared against the list the decorator
    actually used, so a filter applied to `CELLS` is a red test rather than a
    quieter matrix.
    """
    expected = [
        (model, batch, input_class)
        for model in MODELS
        for batch in BATCHES
        for input_class in INPUT_CLASSES
    ]
    assert CELLS == expected
    assert len(CELLS) == len(MODELS) * len(BATCHES) * len(INPUT_CLASSES)
    assert len(CELLS) == 70
    assert len(set(CELL_IDS)) == len(CELL_IDS)


def test_pytest_collects_every_cell_of_both_matrices() -> None:
    """What pytest collects, rather than what this file believes it declared.

    The check above compares one list against a recomputation of itself, which
    catches a filtered `CELLS` and not a `skipif` somebody added to the test.
    This one asks pytest, in its own process, and counts. Two matrices of
    seventy cells is a hundred and forty.
    """
    collected = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "slow or not slow",
            str(Path(__file__)),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr

    ids = [
        line.split("::", 1)[1] for line in collected.stdout.splitlines() if "::" in line
    ]
    for name in (
        "test_the_simulated_answer_matches_onnxruntime",
        "test_the_simulated_answer_matches_the_reference_interpreter",
    ):
        cells = {
            identifier.split("[", 1)[1].rstrip("]")
            for identifier in ids
            if identifier.startswith(f"{name}[")
        }
        assert cells == set(CELL_IDS), sorted(set(CELL_IDS) ^ cells)


def test_the_fast_subset_is_not_empty() -> None:
    """Section 17.4's second collection rule.

    **At this phase the fast subset is the whole matrix, and that is measured
    rather than assumed.** The rule exists so that a matrix too slow for an
    edit and rerun loop still has a subset that is not. The whole `-O0` matrix
    takes about eight seconds for one hundred and forty cells, including the
    exports, so there is nothing to carve out and no cell here is marked
    `slow`. Marking a cell that costs a tenth of a second as slow would be a
    label rather than a measurement.

    The marker and the CI step that runs it stay in place. They start doing
    work at P9 and P10, when three levels and two budgets multiply this matrix
    by six and the ablation cells arrive beside it.
    """
    marks = {
        mark.name
        for test in (
            test_the_simulated_answer_matches_onnxruntime,
            test_the_simulated_answer_matches_the_reference_interpreter,
        )
        for mark in test.pytestmark
    }
    assert "slow" not in marks
    assert CELL_IDS


# ---------------------------------------------------------------------------
# The two things this phase inherited by name.
# ---------------------------------------------------------------------------


def test_batch_norm_has_its_differential_case(suite: Suite) -> None:
    """P7 left this on P8's desk and this is it.

    `npu.batch_norm` has no opcode. `-npu-lower-to-npuisa` decomposes it into a
    multiply and an add over per channel constants computed at rewrite time, so
    comparing it against the simulator needs the whole pipeline: there is no
    single instruction to point a unit test at. `test_refexec.py` pins the
    decomposition rule at the tensor level and
    `test_refexec_differential.py::test_the_export_covers_every_operation_refexec_can_run`
    names it as the one executable operation absent from that suite by design.

    `conv_bn_relu_stack` is the model that carries it, and it carries it by
    construction: it is built with the ONNX API precisely so the
    `BatchNormalization` nodes survive export. The assertion is that
    `refexec.batch_norm`, which computes the closed form, agrees with what the
    machine computes after the decomposition.
    """
    model, batch = "conv_bn_relu_stack", 1
    ir_text = suite.npu_ir(model, batch)
    assert "npu.batch_norm" in ir_text, (
        "the model whose whole purpose is to hold an unfolded batch norm no "
        "longer holds one, so this differential case checks nothing"
    )

    program = suite.program(model, batch)
    assert program.binary is not None
    lowered = compile_model(suite.onnx(model, batch), level=LEVEL, emit="npuisa")
    assert lowered.text is not None
    assert "npuisa.batch_norm" not in lowered.text, (
        "the batch norm reached the instruction level, which would mean an "
        "opcode this machine does not have"
    )

    for input_class in INPUT_CLASSES:
        arrays = make_inputs(
            input_class, program.input_shapes, model=model, batch=batch
        )
        reference = execute_module(ir_text, arrays)
        answer = run_program(program.binary, arrays, program.output_shapes)
        for produced, expected in zip(answer.outputs, reference, strict=True):
            assert_both_bounds(
                produced,
                expected,
                f"batch_norm-{input_class}",
                "refexec.batch_norm through the decomposition",
            )


def test_the_instruction_count_is_the_simulators_field(suite: Suite) -> None:
    """Section 10.2, asserted rather than trusted.

    A regex over an IR dump is not an instruction count: it matches inside type
    strings and counts constants the encoder treats as data. The two numbers
    below are both wrong on purpose, and they are here to show by how much.
    """
    program = suite.program("lenet", 1)
    assert program.binary is not None
    arrays = make_inputs("normal", program.input_shapes, model="lenet", batch=1)
    answer = run_program(program.binary, arrays, program.output_shapes)

    assert answer.instructions == answer.stats["instructions"]
    assert answer.instructions > 0

    lowered = compile_model(suite.onnx("lenet", 1), level=LEVEL, emit="npuisa")
    assert lowered.text is not None
    npuisa_lines = sum(1 for line in lowered.text.splitlines() if "npuisa." in line)
    assert npuisa_lines != answer.instructions, (
        "counting lines of IR happened to give the right answer on this model, "
        "which would make this test look like a rule and act like a "
        "coincidence"
    )
