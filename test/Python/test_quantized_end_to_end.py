# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 14's two end to end bounds, on every model of the suite and level.

Each model is calibrated from its committed profile, compiled at `-O0`, `-O1`
and `-O2` through the QDQ contraction, encoded and run on the machine, and the
answer is held to two references that measure two different things:

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

**The levels are also held to each other.** At `-O2` the calibration runs after
`-npu-fuse-bias`, `-npu-fold-batchnorm` and `-npu-fuse-ops`, so the two models
with a fold compile to a different integer program than at `-O0`, and the
other five to the same one. `-O1` has none of the three and is `-O0`'s program
on all seven. Which is which is asserted bit for bit, so that a change that
moved the calibration back, or let `-O1` drift, is a red rather than a number
that happens to stay inside a budget.
"""

from __future__ import annotations

import json
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
from npu_frontend.results import quant_boundary_crossings
from npu_frontend.tolerances import (
    QUANTIZED_ACCURACY_BUDGETS,
    QUANTIZED_ACCURACY_BUDGETS_AT_LEVEL,
    QUANTIZED_BUDGET_CLASS,
)
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES = REPO_ROOT / "experiments" / "calibration"

#: The levels a quantized compilation is measured at, which are all of them.
LEVELS: tuple[int, ...] = (0, 1, 2)

#: The models whose calibrated program `-O2` changes: a batch norm folded into
#: each convolution before its weights are scaled, and a bias add fused into a
#: convolution so that the relu after it fuses into the instruction.
CHANGED_AT_O2: frozenset[str] = frozenset({"conv_bn_relu_stack", "dilated_stack"})

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
    """One model at one level, calibrated, compiled and ready to run."""

    name: str
    level: int
    onnx_path: Path
    batch: int
    binary: bytes
    npu_text: str
    npuisa_text: str
    input_shapes: Any
    output_shapes: Any
    count: float
    session: ort.InferenceSession


@pytest.fixture(scope="module")
def quantized(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[tuple[str, int], Quantized]]:
    """Every model of the suite at every level, compiled once for this file."""
    directory = tmp_path_factory.mktemp("quantized-end-to-end")
    compiled: dict[tuple[str, int], Quantized] = {}
    for name in sorted(MODELS):
        onnx_path = Path(generate_model(name, directory))
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        for level in LEVELS:
            program = compile_model(
                onnx_path,
                level=level,
                emit="nbin",
                calibrate=str(PROFILES / f"{name}.json"),
            )
            assert program.binary is not None
            compiled[(name, level)] = Quantized(
                name=name,
                level=level,
                onnx_path=onnx_path,
                batch=int(MODELS[name].input_shape[0]),
                binary=program.binary,
                npu_text=program.stages["npu"],
                npuisa_text=program.stages["npuisa"],
                input_shapes=program.input_shapes,
                output_shapes=program.output_shapes,
                count=output_count(program.stages["npu"]),
                session=session,
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


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("input_class", INPUT_CLASSES)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_machine_agrees_with_the_integer_reference(
    name: str, input_class: str, level: int, quantized: dict[tuple[str, int], Quantized]
) -> None:
    """Within one count of the output scale, on every class and level.

    **Measured at zero counts on every model and every class**, 2026-09-24:
    the two answers are the same bits, including on `conv_bn_relu_stack`,
    whose batch norm rounds in f32 between two quantized layers and was
    predicted to be the one place a value might land across a quantization
    boundary. The bound is Section 14's one count rather than zero, because
    zero on that model is a measurement and not a guarantee: a batch norm
    evaluated in a different order is allowed to put a value on the other side
    of a rounding.

    **And zero at `-O1` and `-O2`**, measured 2026-09-29 with the calibration
    after the `-O2` folds: the reference reads the `-O2` tensor level program as
    it reads the `-O0` one, the folded filter and the fused bias included.
    """
    model = quantized[(name, level)]
    inputs, (simulated,) = _run(model, input_class)
    (reference,) = refgraph.execute_module(model.npu_text, inputs)
    counts = (
        np.abs(simulated.astype(np.float64) - reference.astype(np.float64))
        / model.count
    )
    assert float(counts.max()) <= 1.0, (
        f"{name} -O{level} on {input_class}: {float(counts.max()):.3f} counts "
        f"from the integer reference at a count of {model.count}"
    )


# ---------------------------------------------------------------------------
# The second bound: onnxruntime, within each model's budget.
# ---------------------------------------------------------------------------


def test_every_model_has_exactly_one_budget() -> None:
    """A model added to the suite without a budget is a red, not an unbounded model."""
    assert set(QUANTIZED_ACCURACY_BUDGETS) == set(MODELS)
    assert set(QUANTIZED_ACCURACY_BUDGETS_AT_LEVEL) == set(LEVELS)
    for level in LEVELS:
        assert set(QUANTIZED_ACCURACY_BUDGETS_AT_LEVEL[level]) == set(MODELS)


def test_no_level_is_allowed_more_error_than_minus_o_zero() -> None:
    """A budget tightens and never loosens, and that includes across levels.

    A higher level is allowed to be more accurate than `-O0` and to have a
    tighter budget for it, and is never allowed a wider one: an optimization
    that made a model less accurate is a finding, not a new budget.
    """
    for level in LEVELS:
        for name, (floor_db, counts) in QUANTIZED_ACCURACY_BUDGETS_AT_LEVEL[
            level
        ].items():
            base_floor, base_counts = QUANTIZED_ACCURACY_BUDGETS[name]
            assert floor_db >= base_floor, (name, level)
            assert counts <= base_counts, (name, level)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_model_is_within_its_accuracy_budget(
    name: str, level: int, quantized: dict[tuple[str, int], Quantized]
) -> None:
    """The SQNR floor and the largest error in counts, asserted separately.

    The budget and the measurement behind it are in `npu_frontend.tolerances`,
    on `normal`, which is the class the calibration draws come from, and the
    budget is the level's own.
    """
    floor_db, largest_counts = QUANTIZED_ACCURACY_BUDGETS_AT_LEVEL[level][name]
    model = quantized[(name, level)]
    inputs, (simulated,) = _run(model, QUANTIZED_BUDGET_CLASS)
    names = [entry.name for entry in model.session.get_inputs()]
    (expected,) = model.session.run(None, dict(zip(names, inputs, strict=True)))

    ratio = sqnr_db(np.asarray(expected), simulated)
    assert ratio >= floor_db, (
        f"{name} -O{level}: {ratio:.3f} dB against onnxruntime, below its floor "
        f"of {floor_db} dB"
    )
    worst = float(
        np.abs(
            simulated.astype(np.float64) - np.asarray(expected, dtype=np.float64)
        ).max()
    )
    assert worst / model.count <= largest_counts, (
        f"{name} -O{level}: the largest error is {worst / model.count:.3f} counts "
        f"of {model.count}, above its bound of {largest_counts}"
    )


def test_the_tight_budget_computes_what_the_default_budget_computes() -> None:
    """The budget axis, delegated to the recorded cells and checked there.

    *Added at P14 with the quantized cells, D-0072.* This file compiles every
    model at the default budget only, so the bounds above say nothing about the
    tight budget by themselves. The recorded quantized cells are what does:
    every model, level and budget must have one, and at the tight budget each
    must compute exactly what it computes at the default budget at the same
    batch, every field of its accuracy group equal. Then the default budget's
    bounds are the tight budget's too, checked rather than assumed, as
    `test_end_to_end.py` checks its own delegation of the same axis.
    """
    results_dir = REPO_ROOT / "experiments" / "results"
    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.glob("*-int8-*.json"))
    ]
    if not cells:
        pytest.skip("no quantized results recorded yet; run run_benchmarks.py")

    answers: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for cell in cells:
        if not cell["cell"]["quantized"] or cell["cell"]["ablated_pass"] is not None:
            continue
        key = (
            cell["cell"]["model"],
            int(cell["cell"]["opt_level"]),
            cell["cell"]["scratchpad_budget"],
            int(cell["cell"]["batch"]),
        )
        answers[key] = cell["accuracy"]

    missing = [
        (name, level, budget)
        for name in sorted(MODELS)
        for level in LEVELS
        for budget in ("default", "tight")
        if not any(key[:3] == (name, level, budget) for key in answers)
    ]
    assert not missing, (
        f"this file bounds the default budget only, on the understanding that "
        f"the recorded quantized cells carry the tight one, and these "
        f"combinations are not there: {missing}"
    )

    for (name, level, budget, batch), accuracy in sorted(answers.items()):
        if budget != "tight":
            continue
        default = answers.get((name, level, "default", batch))
        assert default is not None, (name, level, batch)
        assert accuracy == default, (
            f"{name} -O{level} at batch {batch}: the tight budget's answer differs "
            f"from the default budget's, so the bounds this file asserts at the "
            f"default budget do not cover it"
        )


# ---------------------------------------------------------------------------
# The levels against each other.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_class", INPUT_CLASSES)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_level_computes_what_minus_o_zero_computes_unless_a_fold_changed_it(
    name: str, input_class: str, quantized: dict[tuple[str, int], Quantized]
) -> None:
    """`-O1` is `-O0` on all seven, and `-O2` is `-O0` on the five without a fold.

    `-O1` runs constant folding and one canonicalization, and neither finds
    anything in a QDQ program whose value it could change. At `-O2` the
    calibration sees the folded programs, and on five models the folds and the
    fusion change nothing the calibration quantizes: the fused regions are put
    back into the block, and the one other difference, `inception_block`'s three
    input quantizes merged by CSE, computes what each of the three computed. On
    `conv_bn_relu_stack` and `dilated_stack` the program differs, and so must
    the answer, or the calibration is no longer where the level puts it.
    """
    _, (at_zero,) = _run(quantized[(name, 0)], input_class)
    _, (at_one,) = _run(quantized[(name, 1)], input_class)
    _, (at_two,) = _run(quantized[(name, 2)], input_class)
    assert np.array_equal(at_one.view(np.uint32), at_zero.view(np.uint32))
    if name in CHANGED_AT_O2:
        assert not np.array_equal(at_two.view(np.uint32), at_zero.view(np.uint32))
    else:
        assert np.array_equal(at_two.view(np.uint32), at_zero.view(np.uint32))


@pytest.mark.parametrize("name", sorted(MODELS))
def test_no_fused_region_survives_the_calibration(
    name: str, quantized: dict[tuple[str, int], Quantized]
) -> None:
    """Every calibrated operation of every model is outside a region at `-O2`.

    `-npu-fuse-ops` forms fifteen regions on these programs in fp32 and the
    calibration puts back every one around an operation its profile names,
    which on this suite is every one. A surviving region would be a pair sealed
    in where no neighbour's dequantize could fold with it.
    """
    assert "npu.fused_op" not in quantized[(name, 2)].npu_text


def _weight_scales(npu_text: str) -> list[NDArray[Any]]:
    """Each convolution's `weight_scales`, in program order."""
    found: list[NDArray[Any]] = []
    for line in npu_text.splitlines():
        if "npu.conv2d" not in line or "weight_scales" not in line:
            continue
        body = re.search(r"weight_scales = array<f32: ([^>]*)>", line)
        assert body is not None
        found.append(
            np.array([float(piece) for piece in body.group(1).split(",")], np.float32)
        )
    return found


def test_the_folded_filters_are_scaled_from_the_fold(
    quantized: dict[tuple[str, int], Quantized],
) -> None:
    """Section 14's argument for per channel weights, on the batch norm model.

    The batch norm fold multiplies each output channel's filter by its own
    factor, `gamma / sqrt(variance + epsilon)`, and at `-O2` the calibration
    scales the weights after it. So each channel's scale is `max |w'_c| / 127`
    of the folded filter, and this reproduces the fold's f32 arithmetic in
    numpy and holds every channel of both convolutions to it bit for bit.
    Measured on 2026-09-29, the largest channel scale over the smallest:

        convolution   channels   factor          before fold   after fold
        conv0         8          0.572 to 1.445  1.567         3.141
        conv1         8          0.561 to 2.229  1.483         3.254

    Per tensor weights would give the smallest channel 127 over that spread,
    about 40 levels after the fold where it had 81 and 86. Per channel weights
    lose nothing: the integer filters at `-O2` are the `-O0` ones times the
    sign of each channel's factor, on every one of the 792 elements, which is
    why the program's answer moves only by where the output is rounded.
    """
    import onnx
    from npu_frontend.onnx_importer import name_every_node
    from onnx import numpy_helper

    name = "conv_bn_relu_stack"
    before = _weight_scales(quantized[(name, 0)].npu_text)
    after = _weight_scales(quantized[(name, 2)].npu_text)

    model = onnx.load(str(quantized[(name, 2)].onnx_path))
    name_every_node(model.graph)
    initializers = {
        entry.name: numpy_helper.to_array(entry) for entry in model.graph.initializer
    }
    readers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for tensor in node.input:
            readers.setdefault(tensor, []).append(node)
    convolutions = [node for node in model.graph.node if node.op_type == "Conv"]
    assert len(convolutions) == len(before) == len(after) == 2

    f32 = np.float32
    for index, convolution in enumerate(convolutions):
        (norm,) = readers[convolution.output[0]]
        assert norm.op_type == "BatchNormalization"
        epsilon = next(
            (f32(entry.f) for entry in norm.attribute if entry.name == "epsilon"),
            f32(1e-5),
        )
        gamma, _, _, variance = (
            initializers[tensor].astype(f32) for tensor in norm.input[1:5]
        )
        factor = (
            gamma * (f32(1.0) / np.sqrt((variance + epsilon).astype(f32))).astype(f32)
        ).astype(f32)
        filters = initializers[convolution.input[1]].astype(f32)
        folded = (filters * factor.reshape(-1, 1, 1, 1)).astype(f32)
        expected = np.array(
            [f32(float(np.abs(channel).max()) / 127.0) for channel in folded],
            np.float32,
        )
        assert np.array_equal(after[index].view(np.uint32), expected.view(np.uint32))

        spread_before = float(before[index].max() / before[index].min())
        spread_after = float(after[index].max() / after[index].min())
        assert spread_after > 1.9 * spread_before, (spread_before, spread_after)


# ---------------------------------------------------------------------------
# The two result fields Section 14 gives meaning to, on the real programs.
# ---------------------------------------------------------------------------


#: The boundary crossings of each model's quantized program at `-O0`, and the
#: f32 `RELU` instructions left in it, counted from the ONNX graphs rather than
#: from a compilation. `experiments/predictions/p14-fused-relu-and-pair-fold.md`
#: carries the derivation. A relu that is a calibrated operation's only reader is
#: the instruction's activation; where its output is the next calibrated
#: operation's input, the pair between them folds away and the two integer
#: instructions meet with no crossing at all; everything else crosses into f32
#: and back, which is Section 14's I8 boundary around the pools, the adds, the
#: batch norms and the relus that do not follow a calibrated operation.
CROSSINGS_AT_O0: dict[str, tuple[int, int]] = {
    "conv_bn_relu_stack": (6, 2),
    "depthwise_separable": (2, 0),
    "dilated_stack": (4, 2),
    "inception_block": (6, 1),
    "lenet": (6, 0),
    "lenet_batched": (6, 0),
    "resnet_block": (2, 1),
}

#: The same at `-O2`, where the calibration runs after the folds and the
#: fusion, derived in `experiments/predictions/p14-quantized-o1-o2.md` from the
#: committed fp32 `-O2` programs. `conv_bn_relu_stack`'s relus now read their
#: convolutions, the batch norms folded into them, so both fuse and the pair
#: between the layers folds; `dilated_stack`'s `activated` reads a convolution
#: with its bias fused in and fuses; `inception_block` quantizes its shared
#: input once, CSE having merged the three identical quantizes. `-O1` has no
#: fold and no fusion and is `-O0`'s table.
CROSSINGS_AT_O2: dict[str, tuple[int, int]] = {
    **CROSSINGS_AT_O0,
    "conv_bn_relu_stack": (4, 0),
    "dilated_stack": (4, 1),
    "inception_block": (4, 1),
}

CROSSINGS_AT_LEVEL: dict[int, dict[str, tuple[int, int]]] = {
    0: CROSSINGS_AT_O0,
    1: CROSSINGS_AT_O0,
    2: CROSSINGS_AT_O2,
}


def test_every_model_has_its_crossings_counted() -> None:
    assert set(CROSSINGS_AT_LEVEL) == set(LEVELS)
    for table in CROSSINGS_AT_LEVEL.values():
        assert set(table) == set(MODELS)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_integer_fields_a_quantized_cell_records(
    name: str, level: int, quantized: dict[tuple[str, int], Quantized]
) -> None:
    """`int8_macs` from the machine and `quant_boundary_crossings` from the program.

    **Every multiply accumulate of a quantized model is an int8 one**, because
    every convolution and matrix multiplication in the seven contracts, so the
    machine's `int8_macs` equals its `macs` and is not zero.

    **The crossings are the table's**, and they are fewer than two per integer
    instruction wherever a fused relu's output is the next calibrated
    operation's input: the pair between the two folds away. That is the cost
    Section 14 asks to be measured rather than asserted, and what it counts is
    the I8 boundary Section 14 draws rather than an artefact of the lowering.
    The f32 `RELU` instructions are counted beside them, because a relu that
    stayed in f32 behind a calibrated operation would be a fusion that did not
    happen and would show there first.
    """
    model = quantized[(name, level)]
    inputs = make_inputs(
        QUANTIZED_BUDGET_CLASS, model.input_shapes, model=name, batch=model.batch
    )
    statistics = run_program(model.binary, inputs, model.output_shapes).stats
    assert statistics["int8_macs"] == statistics["macs"] > 0

    opcodes = re.findall(r"(npuisa\.[a-z_0-9]+) ins\(", model.npuisa_text)
    counts = {opcode: opcodes.count(opcode) for opcode in set(opcodes)}
    integer = len(
        re.findall(r"npuisa\.(?:conv2d|matmul) ins\([^)]*xi8,", model.npuisa_text)
    )
    crossings = quant_boundary_crossings(quantized=True, npuisa_op_counts=counts)
    expected_crossings, expected_relus = CROSSINGS_AT_LEVEL[level][name]
    assert crossings == {"quant_boundary_crossings": expected_crossings}
    assert counts.get("npuisa.relu", 0) == expected_relus
    assert integer > 0
