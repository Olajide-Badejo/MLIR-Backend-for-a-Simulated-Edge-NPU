# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The QDQ contraction, run on the machine and held to references that share no
code with it.

`-npu-lower-to-npuisa` turns a calibrated convolution or matrix multiplication
into one integer instruction. Four claims about that are made here, and each is
exact rather than a tolerance except the last, whose bound is Section 14's own:

1. **Hand computed.** The two functions of
   `test/Dialect/NPUISA/lowering-quantized.mlir`, compiled, encoded and run on
   the machine, against int8 results written by hand from the scales. The lit
   file checks the numbers the lowering writes; this checks what they compute.
2. **Folded against unfolded, bit for bit.** The program carries the fold the
   lowering wrote, the input zero point in the bias and in the padding;
   `refexec.contracted_conv2d` and `contracted_matmul` subtract the zero point
   inside the multiply accumulate instead. Section 14 says the two are equal
   over integers and asks for exactly this test over random int8 tensors.
3. **The pinned arithmetic, both halves.** The requantization table of every
   instruction in every model's quantized compilation, read out of the compiled
   program, against the Python decomposition of the same profile's scales. And
   every operation a profile covers does contract, so a silent fall back to f32
   on a real model is a red rather than a discovery.
4. **One whole model.** LeNet, compiled, encoded and simulated, against the
   numpy integer reference from the same profile, within one count of the
   output scale, which is Section 14's first end to end bound.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlir import ir
from npu_frontend import compile_model, generate_model, refexec, refgraph, run_program
from npu_frontend.builder import find_tool
from npu_frontend.calibration import decompose_multiplier, requantization_multiplier
from npu_frontend.input_classes import make_inputs
from npu_frontend.model_generator import MODELS
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES = REPO_ROOT / "experiments" / "calibration"


# ---------------------------------------------------------------------------
# Compiling and running a hand written integer program.
# ---------------------------------------------------------------------------


def _compile(module_text: str) -> tuple[str, bytes]:
    """The `-O0` level over a hand written module, as `npuisa` text and bytes."""
    lowered = subprocess.run(
        [str(find_tool("npu-opt")), "-", "--npu-O0"],
        input=module_text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lowered.returncode == 0, lowered.stderr
    with tempfile.TemporaryDirectory(prefix="npu-contraction-") as directory:
        target = Path(directory) / "program.nbin"
        translated = subprocess.run(
            [str(find_tool("npu-translate")), "-", "-o", str(target)],
            input=lowered.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
        assert translated.returncode == 0, translated.stderr
        return lowered.stdout, target.read_bytes()


def _run_integer(
    binary: bytes,
    inputs: Sequence[NDArray[np.int8]],
    output_shapes: Sequence[tuple[int, ...]],
) -> list[NDArray[np.int8]]:
    """`npu-sim` over raw int8 regions.

    `run_program` writes and reads f32, because a model's boundary is f32 by
    the importer's rule. A hand written integer program's boundary is i8, and
    the machine reads whatever bytes a region declares.
    """
    with tempfile.TemporaryDirectory(prefix="npu-contraction-") as directory:
        work = Path(directory)
        program = work / "program.nbin"
        program.write_bytes(binary)
        command = [str(find_tool("npu-sim")), str(program), "--quiet"]
        for index, array in enumerate(inputs):
            path = work / f"in{index}.bin"
            path.write_bytes(np.ascontiguousarray(array, dtype=np.int8).tobytes())
            command += ["--input", str(path)]
        outputs = [work / f"out{index}.bin" for index in range(len(output_shapes))]
        for path in outputs:
            command += ["--output", str(path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr or completed.stdout
        return [
            np.fromfile(path, dtype=np.int8).reshape(shape)
            for path, shape in zip(outputs, output_shapes, strict=True)
        ]


def _dense(values: NDArray[Any]) -> str:
    """An f32 array as an MLIR dense literal, exactly.

    Each element goes through the float64 of its f32 value, and `repr` of that
    is the shortest decimal that parses back to it, so the constant the compiler
    reads is the array the reference reads, bit for bit.
    """
    if values.ndim == 0:
        return repr(float(values))
    return "[" + ", ".join(_dense(row) for row in values) + "]"


def _f32(value: float) -> str:
    return repr(float(np.float32(value)))


def _shape(values: Sequence[int]) -> str:
    return "x".join(str(int(extent)) for extent in values)


# ---------------------------------------------------------------------------
# 1. The hand computed cases.
# ---------------------------------------------------------------------------

#: The convolution of `lowering-quantized.mlir`. Its working is in that file.
HAND_CONVOLUTION = """
func.func @main(%qx: tensor<2x1x1x2xi8>) -> tensor<2x2x1x3xi8> {
  %w = npu.constant dense<[[[[0.9375, -0.375]]], [[[0.625, -0.234375]]]]>
       : tensor<2x1x1x2xf32>
  %b = npu.constant dense<[1.0, -0.4]> : tensor<2xf32>
  %dx = npu.dequantize %qx {scale = 5.000000e-01 : f32, zero_point = 3 : i32}
        : tensor<2x1x1x2xi8> to tensor<2x1x1x2xf32>
  %d = tensor.empty() : tensor<2x2x1x3xf32>
  %y = npu.conv2d ins(%dx, %w, %b : tensor<2x1x1x2xf32>, tensor<2x1x1x2xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<2x2x1x3xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 1, 0, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64,
                   weight_scales = array<f32: 3.750000e-01, 1.562500e-01>}
       -> tensor<2x2x1x3xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = 5 : i32}
        : tensor<2x2x1x3xf32> to tensor<2x2x1x3xi8>
  return %qy : tensor<2x2x1x3xi8>
}
"""

#: The matrix multiplication of `lowering-quantized.mlir`.
HAND_MATMUL = """
func.func @main(%qx: tensor<2x3xi8>) -> tensor<2x2xi8> {
  %w = npu.constant dense<[[0.75, 0.65625], [-1.5, 0.21875], [1.875, -0.4375]]>
       : tensor<3x2xf32>
  %b = npu.constant dense<[0.5, -0.1]> : tensor<2xf32>
  %dx = npu.dequantize %qx {scale = 2.500000e-01 : f32, zero_point = -2 : i32}
        : tensor<2x3xi8> to tensor<2x3xf32>
  %d = tensor.empty() : tensor<2x2xf32>
  %y = npu.matmul ins(%dx, %w, %b : tensor<2x3xf32>, tensor<3x2xf32>,
                                    tensor<2xf32>)
                  outs(%d : tensor<2x2xf32>)
                  {weight_scales = array<f32: 7.500000e-01, 2.187500e-01>}
       -> tensor<2x2xf32>
  %qy = npu.quantize %y {scale = 2.500000e-01 : f32, zero_point = -128 : i32}
        : tensor<2x2xf32> to tensor<2x2xi8>
  return %qy : tensor<2x2xi8>
}
"""


def test_the_hand_computed_convolution_runs_to_the_hand_computed_integers() -> None:
    """Padding that contributes the input zero point, both rails, a tie in M.

    With the folded biases 2 and -11 and the pairs (1610612736, 0) and
    (1342177280, 1) of the lit file, one row [a, b] padded with zp_x = 3 on
    both sides gives, per channel, taps (3, a), (a, b), (b, 3):

      batch 0, [10, -4]
        channel 0, weights [2, -1]:  -2, 26, -9  times 0.75 with the nudge's
                                     tie going up: -1, 20, -7
        channel 1, weights [4, -2]: -19, 37, -33 times 0.625, then halved with
                                     a tie going away from zero: -6, 12, -11
      batch 1, [127, -128]
        channel 0: -119, 384, -257 to -89, 288, -193
        channel 1: -253, 753, -529 to -79, 236, -166

    and adding zp_y = 5 with the rails at -128 and 127. The two padded
    positions of batch 0 land inside the rails, so they are measured rather
    than clamped: padding with zero instead of zp_x would make channel 0's
    first tap pair (0, 10), an accumulator of -8 and an answer of -1 rather
    than 4. -33 is the double rounding worth reading: exact arithmetic gives
    -10.3125, and the machine's two steps give -21 and then -11.
    """
    _, binary = _compile(HAND_CONVOLUTION)
    x = np.array([[[[10, -4]]], [[[127, -128]]]], dtype=np.int8)
    (result,) = _run_integer(binary, [x], [(2, 2, 1, 3)])
    expected = np.array(
        [
            [[[4, 25, -2]], [[-1, 17, -6]]],
            [[[-84, 127, -128]], [[-74, 127, -128]]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(result, expected)


def test_the_hand_computed_matmul_runs_to_the_hand_computed_integers() -> None:
    """K != N, an output zero point of -128, and both rails.

    Folded biases 5 and 2, pairs (1610612736, 0) and (1879048192, 2):

      [20, -7, 100]    column 0: 239 times 0.75 is 179.25, 179, plus -128 = 51
                       column 1: -145 times 0.21875 is -31.7, -32, -160 -> -128
      [127, -128, 127] column 0: 642 times 0.75 is 481.5, 482, 354 -> 127
                       column 1: 1 times 0.21875 rounds to 0, which is -128
                       exactly, the value that represents real zero
    """
    _, binary = _compile(HAND_MATMUL)
    x = np.array([[20, -7, 100], [127, -128, 127]], dtype=np.int8)
    (result,) = _run_integer(binary, [x], [(2, 2)])
    np.testing.assert_array_equal(
        result, np.array([[51, -128], [127, -128]], dtype=np.int8)
    )


def test_what_the_lowering_leaves_in_the_qdq_form_runs_in_f32_on_both_sides() -> None:
    """Partial coverage, run rather than only printed.

    The hand computed convolution without its weight scales is an operation the
    calibrator wrapped and could not give per channel scales to. The lowering
    leaves it in the QDQ form, a DEQUANT, an f32 convolution and a QUANT, and
    the reference has to make the same decision or the two would be comparing
    an integer answer with an f32 one. Both run it in f32 between the pair, and
    they agree within one count: the f32 convolutions sum in different orders,
    so a result on a rounding boundary may land either side of it.
    """
    module = re.sub(r",\s*weight_scales = array<f32: [^>]*>", "", HAND_CONVOLUTION)
    assert "weight_scales" not in module
    text, binary = _compile(module)
    assert "npuisa.dequant" in text and "npuisa.quant " in text
    assert re.search(r"npuisa\.conv2d ins\([^)]*xf32, #npu\.scratchpad>", text)

    x = np.array([[[[10, -4]]], [[[127, -128]]]], dtype=np.int8)
    (simulated,) = _run_integer(binary, [x], [(2, 2, 1, 3)])
    (reference,) = refgraph.execute_module(module, [x])
    assert reference.dtype == np.int8
    difference = simulated.astype(np.int64) - reference.astype(np.int64)
    assert np.max(np.abs(difference)) <= 1


def test_the_hand_computed_program_is_one_integer_instruction() -> None:
    """What was compiled is the contraction and not something that happens to
    compute the same integers: one integer convolution, no DEQUANT of the input,
    no QUANT of the result, and no f32 buffer anywhere."""
    text, _ = _compile(HAND_CONVOLUTION)
    assert len(re.findall(r"npuisa\.conv2d ins\(", text)) == 1
    assert "npuisa.dequant" not in text
    assert "npuisa.quant " not in text
    assert "f32, #npu.scratchpad" not in text


# ---------------------------------------------------------------------------
# 2. Folded against unfolded, over seeded random int8 tensors.
# ---------------------------------------------------------------------------

#: The inputs drawn per case. Each is a full range int8 tensor, so both rails
#: of the input and every value between are reached.
DRAWS = 4


def _weights_and_scales(
    rng: np.random.Generator, shape: tuple[int, ...], axis: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """f32 weights and their symmetric per channel scales, the observer's rule.

    The channels are drawn at different magnitudes, so the scales differ and
    the table is the fourth operand rather than the scalar pair.
    """
    weights = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    spread = np.linspace(0.25, 2.0, shape[axis]).astype(np.float32)
    view = [1] * len(shape)
    view[axis] = -1
    weights = (weights * spread.reshape(view)).astype(np.float32)
    moved = np.moveaxis(weights, axis, 0).reshape(shape[axis], -1)
    scales = (np.abs(moved).max(axis=1).astype(np.float64) / 127.0).astype(np.float32)
    return weights, scales


def _conv_program(
    x_shape: tuple[int, int, int, int],
    weights: NDArray[np.float32],
    bias: NDArray[np.float32],
    scales: NDArray[np.float32],
    y_shape: tuple[int, ...],
    attributes: dict[str, Any],
    pair_x: tuple[float, int],
    pair_y: tuple[float, int],
) -> str:
    strides, pads, dilations, group = (
        attributes["strides"],
        attributes["pads"],
        attributes["dilations"],
        attributes["group"],
    )
    return f"""
func.func @main(%qx: tensor<{_shape(x_shape)}xi8>) -> tensor<{_shape(y_shape)}xi8> {{
  %w = npu.constant dense<{_dense(weights)}> : tensor<{_shape(weights.shape)}xf32>
  %b = npu.constant dense<{_dense(bias)}> : tensor<{_shape(bias.shape)}xf32>
  %dx = npu.dequantize %qx {{scale = {_f32(pair_x[0])} : f32, zero_point = {pair_x[1]} : i32}}
        : tensor<{_shape(x_shape)}xi8> to tensor<{_shape(x_shape)}xf32>
  %d = tensor.empty() : tensor<{_shape(y_shape)}xf32>
  %y = npu.conv2d ins(%dx, %w, %b : tensor<{_shape(x_shape)}xf32>,
                      tensor<{_shape(weights.shape)}xf32>, tensor<{_shape(bias.shape)}xf32>)
                  outs(%d : tensor<{_shape(y_shape)}xf32>)
                  {{strides = array<i64: {strides[0]}, {strides[1]}>,
                   pads = array<i64: {pads[0]}, {pads[1]}, {pads[2]}, {pads[3]}>,
                   dilations = array<i64: {dilations[0]}, {dilations[1]}>,
                   group = {group} : i64,
                   weight_scales = array<f32: {", ".join(_f32(s) for s in scales)}>}}
       -> tensor<{_shape(y_shape)}xf32>
  %qy = npu.quantize %y {{scale = {_f32(pair_y[0])} : f32, zero_point = {pair_y[1]} : i32}}
        : tensor<{_shape(y_shape)}xf32> to tensor<{_shape(y_shape)}xi8>
  return %qy : tensor<{_shape(y_shape)}xi8>
}}
"""


def _folded_against_unfolded_conv(
    seed: int,
    x_shape: tuple[int, int, int, int],
    filter_shape: tuple[int, int, int, int],
    attributes: dict[str, Any],
) -> None:
    rng = np.random.default_rng(seed)
    weights, scales = _weights_and_scales(rng, filter_shape, axis=0)
    bias = rng.uniform(-1.0, 1.0, size=filter_shape[0]).astype(np.float32)
    pair_x = (0.02, -17)
    pair_y = (0.1, 9)

    height = refexec.windowed_extent(
        x_shape[2],
        filter_shape[2],
        attributes["strides"][0],
        attributes["pads"][0],
        attributes["pads"][2],
        attributes["dilations"][0],
    )
    width = refexec.windowed_extent(
        x_shape[3],
        filter_shape[3],
        attributes["strides"][1],
        attributes["pads"][1],
        attributes["pads"][3],
        attributes["dilations"][1],
    )
    y_shape = (x_shape[0], filter_shape[0], height, width)

    text, binary = _compile(
        _conv_program(
            x_shape, weights, bias, scales, y_shape, attributes, pair_x, pair_y
        )
    )
    # The table is the fourth operand: this is the per channel arm.
    assert re.search(r"npuisa\.conv2d ins\((?:%[\w]+, ){3}%[\w]+ :", text)

    interior = 0
    for draw in range(DRAWS):
        x = rng.integers(-128, 128, size=x_shape, dtype=np.int64).astype(np.int8)
        (simulated,) = _run_integer(binary, [x], [y_shape])
        unfolded = refexec.contracted_conv2d(
            x,
            weights,
            bias,
            scale_x=pair_x[0],
            zero_point_x=pair_x[1],
            weight_scales=[float(s) for s in scales],
            scale_y=pair_y[0],
            zero_point_y=pair_y[1],
            **attributes,
        )
        np.testing.assert_array_equal(simulated, unfolded, err_msg=f"draw {draw}")
        interior += int(np.count_nonzero((unfolded > -128) & (unfolded < 127)))

    # Most results are inside the rails, so agreement is agreement about the
    # arithmetic and not two answers that both saturated.
    assert interior > DRAWS * int(np.prod(y_shape)) // 2


def test_a_padded_strided_convolution_folds_bit_for_bit() -> None:
    """Asymmetric padding on both axes and a stride of two, over a batch of
    two, so the folded term meets every kind of padded position."""
    _folded_against_unfolded_conv(
        seed=14001,
        x_shape=(2, 3, 9, 8),
        filter_shape=(4, 3, 3, 3),
        attributes={
            "strides": [2, 2],
            "pads": [1, 2, 0, 1],
            "dilations": [1, 1],
            "group": 1,
        },
    )


def test_a_depthwise_convolution_folds_bit_for_bit() -> None:
    """One filter per channel, laid out (C, 1, kH, kW) the way ONNX lays it
    out, so the output channel is axis 0 here as it is for a regular filter,
    dilated so that the padded taps are spread through the window."""
    _folded_against_unfolded_conv(
        seed=14002,
        x_shape=(1, 4, 7, 7),
        filter_shape=(4, 1, 3, 3),
        attributes={
            "strides": [1, 1],
            "pads": [2, 2, 2, 2],
            "dilations": [2, 2],
            "group": 4,
        },
    )


def test_a_matmul_whose_k_is_not_its_n_folds_bit_for_bit() -> None:
    """(3, 7) by (7, 5): K is 7 and N is 5, the shape D-0067 hid behind, so a
    channel read from the wrong axis would not even have the right count."""
    rng = np.random.default_rng(14003)
    weights, scales = _weights_and_scales(rng, (7, 5), axis=1)
    bias = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    pair_x, pair_y = (0.02, -17), (0.03, 9)
    text, binary = _compile(f"""
func.func @main(%qx: tensor<3x7xi8>) -> tensor<3x5xi8> {{
  %w = npu.constant dense<{_dense(weights)}> : tensor<7x5xf32>
  %b = npu.constant dense<{_dense(bias)}> : tensor<5xf32>
  %dx = npu.dequantize %qx {{scale = {_f32(pair_x[0])} : f32, zero_point = {pair_x[1]} : i32}}
        : tensor<3x7xi8> to tensor<3x7xf32>
  %d = tensor.empty() : tensor<3x5xf32>
  %y = npu.matmul ins(%dx, %w, %b : tensor<3x7xf32>, tensor<7x5xf32>, tensor<5xf32>)
                  outs(%d : tensor<3x5xf32>)
                  {{weight_scales = array<f32: {", ".join(_f32(s) for s in scales)}>}}
       -> tensor<3x5xf32>
  %qy = npu.quantize %y {{scale = {_f32(pair_y[0])} : f32, zero_point = {pair_y[1]} : i32}}
        : tensor<3x5xf32> to tensor<3x5xi8>
  return %qy : tensor<3x5xi8>
}}
""")
    assert re.search(r"npuisa\.matmul ins\((?:%[\w]+, ){3}%[\w]+ :", text)

    interior = 0
    for draw in range(DRAWS):
        x = rng.integers(-128, 128, size=(3, 7), dtype=np.int64).astype(np.int8)
        (simulated,) = _run_integer(binary, [x], [(3, 5)])
        unfolded = refexec.contracted_matmul(
            x,
            weights,
            bias,
            scale_x=pair_x[0],
            zero_point_x=pair_x[1],
            weight_scales=[float(s) for s in scales],
            scale_y=pair_y[0],
            zero_point_y=pair_y[1],
        )
        np.testing.assert_array_equal(simulated, unfolded, err_msg=f"draw {draw}")
        interior += int(np.count_nonzero((unfolded > -128) & (unfolded < 127)))
    assert interior > DRAWS * 15 // 2


# ---------------------------------------------------------------------------
# 3. The pinned arithmetic, over every committed profile.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    """Every model of the suite, exported once for this file."""
    directory = tmp_path_factory.mktemp("contraction-models")
    yield {name: generate_model(name, directory) for name in sorted(MODELS)}


def _integer_instructions(text: str) -> list[tuple[str, list[list[int]]]]:
    """Every integer compute instruction of a compiled program: the node name
    its location carries, and its rescale as rows of multipliers and shifts.

    The table is read out of the program rather than out of anything the
    compiler reports about it: the instruction's fourth operand is a buffer a
    `dma_load` fills from an `npuisa.const`, so the constant is found by
    following that operand back. An instruction without the operand is the
    per tensor arm, and its scalar pair is returned as a one column table.

    The Python bindings carry none of this project's dialects, so they cannot
    parse a memref whose memory space is `#npu.dram` or `#npu.scratchpad`, and
    those are the only two attributes of the `npu` dialect a compiled program
    holds. Each is spelled as the integer space it stands for before parsing.
    Nothing read below depends on which space a buffer is in.
    """
    generic = subprocess.run(
        [
            str(find_tool("npu-opt")),
            "-",
            "--mlir-print-op-generic",
            "--mlir-print-debuginfo",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    generic = generic.replace("#npu.dram", "1").replace("#npu.scratchpad", "2")

    context = ir.Context()
    context.allow_unregistered_dialects = True
    found: list[tuple[str, list[list[int]]]] = []
    with context, ir.Location.unknown(context=context):
        module = ir.Module.parse(generic)
        function = module.body.operations[0]
        block = function.regions[0].blocks[0]
        operations = [op.operation for op in block.operations]

        filled_from: dict[ir.Value, ir.Value] = {}
        constants: dict[ir.Value, ir.Attribute] = {}
        for operation in operations:
            if operation.name == "npuisa.dma_load":
                filled_from[operation.operands[1]] = operation.operands[0]
            if operation.name == "npuisa.const":
                constants[operation.results[0]] = operation.attributes["value"]

        for operation in operations:
            if operation.name not in ("npuisa.conv2d", "npuisa.matmul"):
                continue
            destination = ir.MemRefType(operation.operands[-1].type)
            if str(destination.element_type) != "i8":
                continue
            located = re.search(r'loc\("([^"]+)"', str(operation.location))
            assert located is not None, str(operation.location)
            if len(operation.operands) == 5:
                source = filled_from[operation.operands[3]]
                table = np.array(ir.DenseIntElementsAttr(constants[source]))
                rows = [list(map(int, table[0])), list(map(int, table[1]))]
            else:
                rows = [
                    [
                        int(
                            ir.IntegerAttr(
                                operation.attributes["requant_multiplier"]
                            ).value
                        )
                    ],
                    [int(ir.IntegerAttr(operation.attributes["requant_shift"]).value)],
                ]
            found.append((located.group(1), rows))
    return found


@pytest.mark.parametrize("name", sorted(MODELS))
def test_the_rescale_the_compiler_wrote_is_the_one_python_computes(
    name: str, models: dict[str, Path]
) -> None:
    """C++ against Python, bit for bit, over every channel of the profile.

    This is the third leg `docs/adr/0014-per-channel-weight-scales-as-an-
    attribute.md` promised: the profile reaches the IR as an attribute, and
    here the attribute reaches the instruction as the pair Python's half of the
    pinned arithmetic computes from the same profile. A disagreement in the last
    bit of one multiplier is a red here rather than an accuracy number nobody
    can explain.

    And **every operation the profile covers contracts**: a convolution or
    matrix multiplication whose weights the profile carries and that compiles
    as f32 anyway would be a quantized compilation measured as though it were
    quantized.
    """
    location = PROFILES / f"{name}.json"
    profile = json.loads(location.read_text(encoding="utf-8"))
    compiled = compile_model(
        models[name], level=0, emit="npuisa", calibrate=str(location)
    )
    assert compiled.text is not None
    instructions = _integer_instructions(compiled.text)

    activations = profile["activation_scales"]
    covered = {
        node: record
        for node, record in profile["nodes"].items()
        if len(record["inputs"]) >= 2
        and record["inputs"][1] in profile["weights"]
        and record["inputs"][0] in activations
        and record["outputs"][0] in activations
    }
    assert covered, f"{name}'s profile covers nothing"
    assert sorted(node for node, _ in instructions) == sorted(covered)

    channels = 0
    for node, rows in instructions:
        record = covered[node]
        scale_x = activations[record["inputs"][0]]["minmax"]["scale"]
        scale_y = activations[record["outputs"][0]]["minmax"]["scale"]
        scales = profile["weights"][record["inputs"][1]]["scales"]
        pairs = [
            decompose_multiplier(requantization_multiplier(scale_x, scale, scale_y))
            for scale in scales
        ]
        if len(rows[0]) == 1 and len(scales) > 1:
            # No table: every channel's pair is the scalar pair.
            assert {pair for pair in pairs} == {(rows[0][0], rows[1][0])}, node
        else:
            assert rows[0] == [pair[0] for pair in pairs], node
            assert rows[1] == [pair[1] for pair in pairs], node
        channels += len(scales)
    assert channels > 0


# ---------------------------------------------------------------------------
# 4. One whole model, against the numpy integer reference.
# ---------------------------------------------------------------------------


def _returned_output_scale(npu_text: str) -> float:
    """The scale of the dequantize the function returns, which is the output
    scale Section 14's bound is counted in."""
    returned = re.search(r"return (%[\w]+) ", npu_text)
    assert returned is not None
    defined = re.search(
        re.escape(returned.group(1))
        + r" = npu\.dequantize %[\w]+ \{scale = ([^ ]+) : f32",
        npu_text,
    )
    assert defined is not None
    return float(defined.group(1))


@pytest.mark.parametrize("input_class", ["normal", "relu_knee"])
def test_lenet_quantized_agrees_with_the_integer_reference(
    input_class: str, models: dict[str, Path]
) -> None:
    """Section 14's first end to end bound, on the first real program through
    the contraction: within one count of the output scale.

    The compiled program runs its five integer instructions on the machine. The
    reference is `refgraph` over the same tensor level IR, which executes each
    contracted operation through `refexec`'s unfolded arithmetic and every
    other operation in f32 between the dequantize and the quantize, exactly as
    the program is compiled. The profile both read is the committed one. The
    inputs are the evaluation classes, whose seeds are from a different
    namespace than the calibration draw's, so the model is not measured on what
    it was calibrated against.
    """
    location = PROFILES / "lenet.json"
    compiled = compile_model(
        models["lenet"], level=0, emit="nbin", calibrate=str(location)
    )
    assert compiled.binary is not None
    npu_text = compiled.stages["npu"]
    assert len(re.findall(r"npuisa\.(?:conv2d|matmul)", compiled.stages["npuisa"])) == 5

    inputs = make_inputs(input_class, compiled.input_shapes, model="lenet", batch=1)
    simulated = run_program(compiled.binary, inputs, compiled.output_shapes).outputs[0]
    (reference,) = refgraph.execute_module(npu_text, inputs)

    scale = _returned_output_scale(npu_text)
    counts = np.rint(
        (simulated.astype(np.float64) - reference.astype(np.float64)) / scale
    )
    assert np.max(np.abs(counts)) <= 1, counts
