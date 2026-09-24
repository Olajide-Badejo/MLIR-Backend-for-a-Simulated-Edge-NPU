# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 14's two end to end bounds, on every model of the suite.

Each model is calibrated from its committed profile, compiled at `-O0` through
the QDQ contraction, encoded and run on the machine, and the answer is held to
two references that measure two different things:

1. **The numpy integer reference from the same profile, within one count of the
   output scale.** `refgraph` executes the same tensor level program with every
   contracted operation computed by `refexec`'s unfolded arithmetic, so this is
   the machine and the compiler against an independent reading of the same
   integer program. It is asserted on all five of Section 17.4's input classes,
   because exactness does not depend on the input.
2. **onnxruntime, within a per model accuracy budget.** This is the f32 graph
   the program approximates, so the distance is quantization error and the
   budget is how much of it each model is allowed. The budgets and the
   measurement they were set from are `npu_frontend.tolerances`'s, set after a
   committed prediction.

**One count of the output scale** is defined once, below: the scale of the
dequantize whose values reach the model's output without passing another
quantize, and the largest of them when there are several, which is
`inception_block`'s three branches. An f32 operation after it, an average pool
or a transpose, changes where a value lands between counts and not what a count
is.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from npu_frontend import compile_model, generate_model, refgraph, run_program
from npu_frontend.input_classes import INPUT_CLASSES, make_inputs
from npu_frontend.model_generator import MODELS
from npu_frontend.tolerances import (
    QUANTIZED_ACCURACY_BUDGETS,
    QUANTIZED_BUDGET_CLASS,
)
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES = REPO_ROOT / "experiments" / "calibration"

_DEFINITION = re.compile(r"^\s*(%[\w]+) = (npu\.[a-z_0-9]+|tensor\.empty)(.*)$")


def output_count(npu_text: str) -> float:
    """One count of the output's scale, read out of the tensor level program.

    Walks back from the function's results through every operation that is not
    a quantization, collecting the scale of each dequantize it reaches and
    stopping at a quantize, because a value that passed a quantize after that
    dequantize is counted in the quantize's scale by whatever dequantizes it
    next. The program is one block in static single assignment form, so a
    definition per value is the whole of the graph this needs.
    """
    definitions: dict[str, tuple[str, list[str], float | None]] = {}
    for line in npu_text.splitlines():
        found = _DEFINITION.match(line)
        if not found:
            continue
        body = found.group(3).split(" loc(")[0]
        scale = re.search(r"scale = ([^ ]+) : f32", body)
        definitions[found.group(1)] = (
            found.group(2),
            re.findall(r"%[\w]+", body),
            float(scale.group(1)) if scale else None,
        )
    returned = re.search(r"^\s*return (.*?) :", npu_text, re.MULTILINE)
    assert returned is not None, "the program returns nothing"

    pending = re.findall(r"%[\w]+", returned.group(1))
    visited: set[str] = set()
    scales: list[float] = []
    while pending:
        value = pending.pop()
        if value in visited or value not in definitions:
            continue
        visited.add(value)
        operation, operands, scale = definitions[value]
        if operation == "npu.dequantize":
            assert scale is not None
            scales.append(scale)
        elif operation != "npu.quantize":
            pending.extend(operands)
    assert scales, "no dequantize reaches the output, so this is not quantized"
    return max(scales)


def sqnr_db(reference: NDArray[Any], approximation: NDArray[Any]) -> float:
    """Section 16.1's primary accuracy metric, `20 * log10(|x| / |x - x_hat|)`."""
    signal = float(np.linalg.norm(reference.astype(np.float64)))
    noise = float(
        np.linalg.norm(reference.astype(np.float64) - approximation.astype(np.float64))
    )
    return float("inf") if noise == 0.0 else 20.0 * float(np.log10(signal / noise))


@dataclass
class Quantized:
    """One model, calibrated, compiled and ready to run."""

    name: str
    batch: int
    binary: bytes
    npu_text: str
    input_shapes: Any
    output_shapes: Any
    count: float
    session: ort.InferenceSession


@pytest.fixture(scope="module")
def quantized(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Quantized]]:
    """Every model of the suite, compiled once for this file."""
    directory = tmp_path_factory.mktemp("quantized-end-to-end")
    compiled: dict[str, Quantized] = {}
    for name in sorted(MODELS):
        onnx_path = generate_model(name, directory)
        program = compile_model(
            onnx_path,
            level=0,
            emit="nbin",
            calibrate=str(PROFILES / f"{name}.json"),
        )
        assert program.binary is not None
        compiled[name] = Quantized(
            name=name,
            batch=int(MODELS[name].input_shape[0]),
            binary=program.binary,
            npu_text=program.stages["npu"],
            input_shapes=program.input_shapes,
            output_shapes=program.output_shapes,
            count=output_count(program.stages["npu"]),
            session=ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            ),
        )
    yield compiled


def _run(model: Quantized, input_class: str) -> tuple[list[Any], list[NDArray[Any]]]:
    inputs = make_inputs(
        input_class, model.input_shapes, model=model.name, batch=model.batch
    )
    return inputs, run_program(model.binary, inputs, model.output_shapes).outputs


# ---------------------------------------------------------------------------
# The first bound: the numpy integer reference from the same profile.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_class", INPUT_CLASSES)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_machine_agrees_with_the_integer_reference(
    name: str, input_class: str, quantized: dict[str, Quantized]
) -> None:
    """Within one count of the output scale, on every class.

    **Measured at zero counts on every model and every class**, 2026-09-24:
    the two answers are the same bits, including on `conv_bn_relu_stack`,
    whose batch norm rounds in f32 between two quantized layers and was
    predicted to be the one place a value might land across a quantization
    boundary. The bound is Section 14's one count rather than zero, because
    zero on that model is a measurement and not a guarantee: a batch norm
    evaluated in a different order is allowed to put a value on the other side
    of a rounding.
    """
    model = quantized[name]
    inputs, (simulated,) = _run(model, input_class)
    (reference,) = refgraph.execute_module(model.npu_text, inputs)
    counts = (
        np.abs(simulated.astype(np.float64) - reference.astype(np.float64))
        / model.count
    )
    assert float(counts.max()) <= 1.0, (
        f"{name} on {input_class}: {float(counts.max()):.3f} counts from the "
        f"integer reference at a count of {model.count}"
    )


# ---------------------------------------------------------------------------
# The second bound: onnxruntime, within each model's budget.
# ---------------------------------------------------------------------------


def test_every_model_has_exactly_one_budget() -> None:
    """A model added to the suite without a budget is a red, not an unbounded model."""
    assert set(QUANTIZED_ACCURACY_BUDGETS) == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_model_is_within_its_accuracy_budget(
    name: str, quantized: dict[str, Quantized]
) -> None:
    """The SQNR floor and the largest error in counts, asserted separately.

    The budget and the measurement behind it are in `npu_frontend.tolerances`,
    on `normal`, which is the class the calibration draws come from.
    """
    floor_db, largest_counts = QUANTIZED_ACCURACY_BUDGETS[name]
    model = quantized[name]
    inputs, (simulated,) = _run(model, QUANTIZED_BUDGET_CLASS)
    names = [entry.name for entry in model.session.get_inputs()]
    (expected,) = model.session.run(None, dict(zip(names, inputs, strict=True)))

    ratio = sqnr_db(np.asarray(expected), simulated)
    assert ratio >= floor_db, (
        f"{name}: {ratio:.3f} dB against onnxruntime, below its floor of "
        f"{floor_db} dB"
    )
    worst = float(
        np.abs(
            simulated.astype(np.float64) - np.asarray(expected, dtype=np.float64)
        ).max()
    )
    assert worst / model.count <= largest_counts, (
        f"{name}: the largest error is {worst / model.count:.3f} counts of "
        f"{model.count}, above its bound of {largest_counts}"
    )
