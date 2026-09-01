# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The Section 12 passes, on IR the real ONNX importer produced.

*Added at P9.*

**Why this file exists beside the lit suite rather than instead of it.** The lit
files in `test/Transforms/` pin each pass's rules on IR written by hand, which is
where a guard's edge cases live and where a negative case can be stated exactly.
What they cannot show is that the shape a pass guards on is a shape a real model
actually imports to. `-npu-fuse-bias` is the case that makes the difference
load bearing: Section 11 leaves a channel shaped addend unexpanded *so that this
pass can match it*, and a hand written lit case would match a shape the importer
might have stopped producing years ago. So the fixtures below go through
`import_model` and the assertions are about what came out of it.

**The gate clause this file carries** is "`-npu-fuse-bias` fires on a real
imported model, which is what the Section 11 broadcast carve out exists to make
possible", and `test_fuse_bias_fires_on_a_real_imported_model` is it.

**The finding beside it, and what P9b did about it.** At P9 none of the seven
models in Section 15's suite contained a biasless `Conv` followed by a channel
shaped `Add`: every convolution in the suite carried its bias inline as a third
`Conv` input, which is what `torch.onnx.export` and this project's ONNX built
models both emit. The pass fired on an imported model and on no model *of the
suite*, so its Section 16.2 ablation row would have been a row of zeros that read
as a measurement of the pass rather than of the suite. P9b closed that:
`dilated_stack`'s `conv1` was already biasless and already followed by a `Relu`,
and it now carries a separate channel shaped bias `Add` between them.
`test_the_suite_gives_the_bias_fusion_exactly_one_target` is the assertion, and
it names the model rather than merely counting, so a second model quietly
acquiring or losing the shape is as loud as the first one gaining it.

**The `-sccp` row stays a row of zeros and that is a different kind of fact.**
Constant propagation needs a call graph to propagate across, an imported model is
one function, and no model change alters that: this compiler has no calls. So one
of the two zero rows P9 predicted was a gap in the suite and has been closed, and
the other is a true property of the programs this compiler compiles and is
reported as a measurement. `test_sccp_has_nothing_to_do_on_a_single_function`
holds that half, and the two are deliberately not written as one test: they would
go red for entirely different reasons.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnx
import pytest
from npu_frontend import (
    MODELS,
    ablatable_passes,
    compile_model,
    generate_model,
    run_program,
)
from npu_frontend.input_classes import make_inputs
from npu_frontend.onnx_importer import import_model
from onnx import helper
from onnx_fixtures import initializer, make_model, value

from tools import tool


def run_passes(text: str, *arguments: str) -> str:
    """One `npu-opt` invocation over `text`, or an assertion with its message."""
    completed = subprocess.run(
        [str(tool("npu-opt")), "-", *arguments],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"npu-opt {' '.join(arguments)} exited {completed.returncode}\n\n"
        f"{completed.stderr}"
    )
    return completed.stdout


def conv_then_channel_add() -> onnx.ModelProto:
    """A biasless `Conv` followed by an `Add` of a channel shaped initializer.

    The shape Section 11's carve out exists for, and the shape a separate bias
    add has. The addend is written as `(1, C, 1, 1)` rather than as `(C,)`
    because that is the spelling `p.reshape(1, -1, 1, 1)` produces in an exported
    graph, and the importer normalises it to rank 1; an importer that matched
    only a literally rank 1 initializer would expand this one and leave the
    fusion nothing to fire on.
    """
    rng = np.random.default_rng(20260901)
    return make_model(
        [
            helper.make_node(
                "Conv",
                ["x", "w"],
                ["c"],
                name="conv0",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node("Add", ["c", "bias"], ["y"], name="bias_add"),
        ],
        [value("x", (1, 3, 5, 5))],
        [value("y", (1, 4, 5, 5))],
        [
            initializer("w", rng.standard_normal((4, 3, 3, 3)).astype(np.float32)),
            initializer("bias", rng.standard_normal((1, 4, 1, 1)).astype(np.float32)),
        ],
    )


@pytest.fixture(scope="module")
def suite_models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    directory = tmp_path_factory.mktemp("transform-passes")
    yield {name: generate_model(name, directory) for name in MODELS}


# ---------------------------------------------------------------------------
# -npu-fuse-bias, and the carve out it was built for.
# ---------------------------------------------------------------------------


def test_fuse_bias_fires_on_a_real_imported_model() -> None:
    """P9's gate clause, through the importer rather than around it.

    ONNX in, `import_model` out, the pass over what it produced. The addend
    reaches the pass as a rank 1 constant because Section 11 leaves it
    unexpanded, and the convolution reaches it with two operands because the
    `Conv` node had two inputs; both halves are what the guard reads.
    """
    imported = import_model(conv_then_channel_add())
    assert re.search(r"npu\.add ins\(%\w+, %cst\w* :", imported), (
        "the importer did not produce the add this pass exists to fold, so "
        "the assertion below would pass for the wrong reason"
    )

    fused = run_passes(imported, "--npu-fuse-bias", "--canonicalize")

    assert "npu.add" not in fused, "the separate bias add survived the fusion"
    match = re.search(r"npu\.conv2d ins\(([^:]*):", fused)
    assert match is not None, "the convolution is gone, which is not fusion"
    assert match.group(1).count(",") == 2, (
        f"the convolution has {match.group(1).count(',') + 1} operands and a "
        f"fused bias makes three: input, filter, bias. Got {match.group(1)!r}"
    )


def test_fuse_bias_changes_no_answer_on_the_model_it_fires_on() -> None:
    """Exact, and measured rather than argued.

    The simulator's convolution kernel adds the bias to the same `f32`
    accumulator the unfused program stores and then adds to, so the two answers
    agree bit for bit. That is why `docs/BREAKING_CHANGES.md` attributes this
    phase's numerics movement to `-npu-fold-batchnorm` alone, and this is the
    measurement behind the attribution.
    """
    model = conv_then_channel_add()
    at_zero = compile_model(model, level=0, emit="nbin")
    at_two = compile_model(model, level=2, emit="nbin")
    assert at_zero.binary is not None and at_two.binary is not None

    rng = np.random.default_rng(20260901)
    x = rng.standard_normal(at_zero.input_shapes[0]).astype(np.float32)

    plain = run_program(at_zero.binary, [x], at_zero.output_shapes)
    fused = run_program(at_two.binary, [x], at_two.output_shapes)

    np.testing.assert_array_equal(
        fused.outputs[0],
        plain.outputs[0],
        err_msg=(
            "-O2 moved the answer on a model whose only optimization is the "
            "bias fusion, which is measured to be bit exact"
        ),
    )
    # And it is a saving rather than a rearrangement: one fewer instruction,
    # because the add became an operand.
    assert fused.instructions < plain.instructions


def test_the_suite_gives_the_bias_fusion_exactly_one_target(
    suite_models: dict[str, Path],
) -> None:
    """The inverse of the P9 assertion, and it names the model.

    At P9 this test's predecessor asserted the empty list: no model of Section
    15's suite held a biasless `Conv` followed by a channel shaped `Add`, so the
    pass's Section 16.2 ablation row would have been zeros for want of a target.
    `dilated_stack` carries the shape now and this asserts which model does.

    **Naming the model rather than asserting a non empty list** is the point. A
    count would stay green if `dilated_stack` lost the shape and some other model
    gained it, and those are two separate suite changes each of which should be
    read by somebody rather than absorbed.
    """
    fired_on = []
    for name in sorted(suite_models):
        before = compile_model(suite_models[name], level=0, emit="npu").text
        assert before is not None
        after = run_passes(before, "--npu-fuse-bias")
        if before.count("npu.conv2d") != after.count("npu.conv2d") or before.count(
            "npu.add"
        ) != after.count("npu.add"):
            fired_on.append(name)

    assert fired_on == ["dilated_stack"], (
        f"-npu-fuse-bias fires on {fired_on} and the suite is built for it to "
        f"fire on ['dilated_stack']. A shorter list is the P9 gap reopening and "
        f"a longer one is a model that gained the shape without anybody saying "
        f"so; both make the paragraph in this file's docstring out of date."
    )


def test_the_bias_fusion_is_a_saving_and_not_a_rearrangement(
    suite_models: dict[str, Path],
) -> None:
    """A target the pass declines to act on would satisfy the test above.

    So the saving is measured too, as a relation rather than as a constant: on
    `dilated_stack` the `-O2` program is exactly one instruction shorter than the
    `-O0` one, because the separate add became an operand of the convolution, and
    the two programs agree bit for bit. Both halves matter. The first says the
    ablation row P10 records will not be zero; the second says the row costs
    nothing in accuracy, which is what `docs/BREAKING_CHANGES.md` attributes to
    this pass and which was measured at P9 on a model built for the test rather
    than on a model of the suite.
    """
    path = suite_models["dilated_stack"]
    at_zero = compile_model(path, level=0, emit="nbin")
    at_two = compile_model(path, level=2, emit="nbin")
    assert at_zero.binary is not None and at_two.binary is not None

    arrays = make_inputs("normal", at_zero.input_shapes, model="dilated_stack", batch=1)
    plain = run_program(at_zero.binary, arrays, at_zero.output_shapes)
    fused = run_program(at_two.binary, arrays, at_two.output_shapes)

    assert fused.instructions == plain.instructions - 1, (
        f"-O2 is {plain.instructions - fused.instructions} instructions shorter "
        f"than -O0 on dilated_stack and the bias fusion accounts for exactly "
        f"one. A different number means another pass started firing on this "
        f"model, which is a measurement to take rather than a test to adjust."
    )
    np.testing.assert_array_equal(
        fused.outputs[0],
        plain.outputs[0],
        err_msg=(
            "-O2 moved dilated_stack's answer. The bias fusion is bit exact "
            "because the convolution kernel adds the bias to the same f32 "
            "accumulator the unfused program stores and then adds to."
        ),
    )


def test_sccp_has_nothing_to_do_on_a_single_function(
    suite_models: dict[str, Path],
) -> None:
    """The other zero row, which is a property rather than a gap.

    `-sccp` propagates constants across a call graph and an imported model is one
    function, so there is nothing for it to propagate across whatever the dialect
    can materialise. D-0033 made the dialect able to write an answer down and the
    pass still has nothing to write here, which is why Section 12 keeps it at
    `-O2` and P10 reports a row of zeros as the measurement that section asked
    for.

    This is deliberately a separate test from the bias fusion's. The two rows
    were both zero at P9 and only one of them was a gap; writing them as one
    parametrized assertion would have made closing the first look like it closed
    both.
    """
    for name in sorted(suite_models):
        before = compile_model(suite_models[name], level=0, emit="npu").text
        assert before is not None
        assert before.count("func.func") == 1, (
            f"{name} imports to more than one function, so the reason this row "
            f"is zero has changed and the claim above needs re-measuring rather "
            f"than repeating"
        )
        after = run_passes(before, "--sccp")
        assert after.count("npu.constant") == before.count("npu.constant"), (
            f"-sccp materialised a constant on {name}, which it cannot do "
            f"across a call graph that does not exist"
        )


# ---------------------------------------------------------------------------
# -npu-fold-batchnorm, on the model that exists to hold an unfolded batch norm.
# ---------------------------------------------------------------------------


def test_fold_batchnorm_fires_on_the_model_built_to_carry_one(
    suite_models: dict[str, Path],
) -> None:
    """`conv_bn_relu_stack` is built with the ONNX API precisely so its
    `BatchNormalization` nodes survive export, and this is the pass they exist
    for. Both of them fold, and the convolutions they fold into keep their bias
    operands rather than gaining a second one."""
    before = compile_model(suite_models["conv_bn_relu_stack"], level=0, emit="npu").text
    assert before is not None
    assert before.count("npu.batch_norm") == 2

    after = run_passes(
        before, "--npu-fuse-bias", "--npu-fold-batchnorm", "--canonicalize"
    )
    assert "npu.batch_norm" not in after
    assert after.count("npu.conv2d") == 2


def test_fold_batchnorm_is_the_only_pass_that_moves_a_number(
    suite_models: dict[str, Path],
) -> None:
    """The attribution `docs/BREAKING_CHANGES.md` makes, measured pass by pass.

    Each pass is run alone over the imported IR of the model that has most to
    fold, the result is lowered and simulated, and the answer is compared with
    the unoptimized one. Only the batch norm fold moves anything, and it moves
    4.47e-08.
    """
    model = suite_models["conv_bn_relu_stack"]
    base = compile_model(model, level=0, emit="nbin")
    assert base.binary is not None
    arrays = make_inputs(
        "normal", base.input_shapes, model="conv_bn_relu_stack", batch=1
    )
    reference = run_program(base.binary, arrays, base.output_shapes).outputs

    imported = compile_model(model, level=0, emit="npu").text
    assert imported is not None

    moved: dict[str, float] = {}
    for pass_name in (
        "npu-constant-fold",
        "canonicalize",
        "npu-fuse-bias",
        "npu-fold-batchnorm",
        "npu-fuse-ops",
        "cse",
        "sccp",
        "symbol-dce",
    ):
        optimized = run_passes(imported, f"--{pass_name}")
        answer = _lower_and_run(optimized, arrays, base.output_shapes)
        moved[pass_name] = max(
            float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
            for got, want in zip(answer.outputs, reference, strict=True)
        )

    assert moved["npu-fold-batchnorm"] > 0.0, (
        "the batch norm fold changed no bit on the model built to carry two "
        "batch norms, which means it did not fire"
    )
    assert moved["npu-fold-batchnorm"] < 1e-6, moved
    for pass_name, value_ in moved.items():
        if pass_name == "npu-fold-batchnorm":
            continue
        assert value_ == 0.0, (
            f"-{pass_name} moved the answer by {value_:.3e}. Only "
            f"-npu-fold-batchnorm is declared to move numbers at this phase, so "
            f"this is either a defect or a declaration that was not written."
        )


def _lower_and_run(
    npu_text: str, arrays: list[np.ndarray], output_shapes: tuple[tuple[int, ...], ...]
):
    """Lowers, encodes and runs one `npu` module.

    Returns the whole `SimulationResult`, so a caller wanting an instruction
    count reads `stats.instructions`, which Section 10.2 makes the only one in
    this project. Counting `npuisa.` in the IR would be the fallback that
    section forbids by name: it matches inside type strings and counts constants
    the encoder treats as data.
    """
    lowered = run_passes(
        npu_text,
        "--npu-lower-to-npuisa",
        "--npu-allocate-scratchpad",
        "--mlir-print-debuginfo",
    )
    translate = tool("npu-translate")
    completed = subprocess.run(
        [str(translate), "-", "-o", "-"],
        input=lowered.encode(),
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    return run_program(completed.stdout, arrays, output_shapes)


# ---------------------------------------------------------------------------
# -npu-fuse-ops, and the claim that it is numerically and structurally inert.
# ---------------------------------------------------------------------------


def test_fuse_ops_fires_on_the_suite_and_costs_nothing(
    suite_models: dict[str, Path],
) -> None:
    """Five of the seven models gain regions, and none of them gains or loses an
    instruction by it.

    Section 5.2 says fusion's benefit under this memory model is that the
    intermediate stays in the scratchpad, and an unfused chain already keeps it
    there: the only DMA producers are the boundary, the spiller and the double
    buffering pass. So the region is a statement about the IR, which P13's
    tiling reads, and not a saving. Measuring that rather than asserting it is
    what keeps the P10 ablation row honest.
    """
    fused_models = []
    for name in sorted(suite_models):
        compiled = compile_model(suite_models[name], level=0, emit="nbin")
        assert compiled.binary is not None
        before = compile_model(suite_models[name], level=0, emit="npu").text
        assert before is not None
        after = run_passes(before, "--npu-fuse-ops")
        if "npu.fused_op" not in after:
            continue
        fused_models.append(name)

        arrays = make_inputs(
            "normal",
            compiled.input_shapes,
            model=name,
            batch=compiled.input_shapes[0][0],
        )
        plain = run_program(compiled.binary, arrays, compiled.output_shapes)
        regioned = _lower_and_run(after, arrays, compiled.output_shapes)

        # Section 10.2: the count is the simulator's field, never a count of
        # lines of IR.
        assert plain.instructions == regioned.instructions, (
            f"{name}: forming a region changed the instruction count from "
            f"{plain.instructions} to {regioned.instructions}. The lowering "
            f"flattens the region, so the two programs are the same "
            f"instructions."
        )
        for got, want in zip(regioned.outputs, plain.outputs, strict=True):
            np.testing.assert_array_equal(got, want, err_msg=name)

    assert len(fused_models) >= 5, fused_models


# ---------------------------------------------------------------------------
# The pass names in the description are pass names npu-opt knows.
# ---------------------------------------------------------------------------


def test_every_ablatable_pass_is_a_pass_npu_opt_accepts() -> None:
    """Section 16.2 reads the ablation set out of the driver at run time, and
    then P10 will run each of those names as a command line argument. A name in
    the description that `npu-opt` does not accept would fail at that point,
    three phases after the table was written; this is the check that fails now
    instead.

    It also catches a `PassKind` and an `argument` that disagree, which is the
    one thing `test/Pipeline/opt-levels.mlir`'s diffs cannot: they compare a
    level against the list of those strings, so a pair that were wrong in the
    same way would agree with each other.
    """
    empty = (
        "func.func @f(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> }"
    )
    for level in (0, 1, 2):
        for name in ablatable_passes(level):
            run_passes(empty, f"--{name}")
