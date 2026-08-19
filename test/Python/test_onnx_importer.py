# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The ONNX importer, one converter at a time and one refusal at a time.

Section 17.1 asks for a pytest per importer operator in isolation. This file is
that, plus the negative cases, which are the larger half: an importer's job is
mostly to refuse things, and a converter with only positive tests is one nobody
has found the edge of.

Every test here goes through `import_model`, which round trips the module
through `npu-opt`. So every assertion in this file is an assertion about text
the real parser, the real verifiers and the real printer all agreed on, rather
than about a string this project wrote and this project read.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import onnx
import pytest
from npu_frontend import (
    CONVERTERS,
    PINNED_OPSET,
    ONNXImportError,
    attributes,
    import_model,
    op_mapping,
)
from npu_frontend.builder import _reject_discardable_attributes, verify
from npu_frontend.diagnostics import VerificationError
from npu_frontend.op_mapping import EMITTED_OPERATIONS, documented_converters
from onnx import helper
from onnx_fixtures import conv_model, initializer, make_model, value


def count(ir: str, mnemonic: str) -> int:
    return len(re.findall(rf"\bnpu\.{mnemonic}\b", ir))


# =============================================================================
# The registry and its documentation.
# =============================================================================


def test_the_module_docstring_lists_exactly_the_registered_converters() -> None:
    """Section 11 pins the docstring to the registry, and this is the pin.

    Both directions, so a converter added without a documentation line fails
    here rather than going undocumented, and a documentation line for a
    converter nobody wrote fails here rather than promising something.
    """
    assert sorted(documented_converters()) == sorted(CONVERTERS)


def test_the_quantization_pair_is_refused_by_name_rather_than_generically() -> None:
    model = make_model(
        [helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["y"], name="q0")],
        [value("x", (1, 4))],
        [helper.make_tensor_value_info("y", onnx.TensorProto.INT8, [1, 4])],
        [
            initializer("scale", np.array(0.05, dtype=np.float32)),
            initializer("zero", np.array(0, dtype=np.int8)),
        ],
    )
    with pytest.raises(ONNXImportError, match="quantization phase"):
        import_model(model)


def test_pad_is_refused_with_the_reason_rather_than_as_merely_unknown() -> None:
    model = make_model(
        [helper.make_node("Pad", ["x", "pads"], ["y"], name="pad0", mode="constant")],
        [value("x", (1, 3, 4, 4))],
        [value("y", (1, 3, 6, 6))],
        [initializer("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64))],
    )
    with pytest.raises(ONNXImportError, match="Pad is not in this project"):
        import_model(model)


def test_an_operator_with_no_converter_names_the_supported_set() -> None:
    model = make_model(
        [helper.make_node("Sigmoid", ["x"], ["y"], name="sig0")],
        [value("x", (1, 4))],
        [value("y", (1, 4))],
    )
    with pytest.raises(ONNXImportError) as raised:
        import_model(model)
    assert "Sigmoid" in str(raised.value)
    assert "AveragePool" in str(raised.value)


# =============================================================================
# The opset pin.
# =============================================================================


def test_a_model_at_another_opset_is_refused_naming_both_numbers() -> None:
    model = conv_model()
    model.opset_import[0].version = 17
    with pytest.raises(
        ONNXImportError, match="opset 17 and this project is pinned at 23"
    ):
        import_model(model)


def test_a_custom_operator_domain_is_refused() -> None:
    model = conv_model()
    model.opset_import.append(helper.make_opsetid("com.example", 1))
    with pytest.raises(ONNXImportError, match="com.example"):
        import_model(model)


def test_the_pin_is_the_number_the_decision_record_resolved() -> None:
    assert PINNED_OPSET == 23


# =============================================================================
# Conv.
# =============================================================================


def test_conv_imports_with_its_bias_as_an_operand() -> None:
    ir = import_model(conv_model())
    assert count(ir, "conv2d") == 1
    assert "npu.conv2d ins(%arg0, %cst, %cst_0" in ir
    assert "pads = array<i64: 1, 1, 1, 1>" in ir


def test_conv_without_a_bias_emits_two_input_operands() -> None:
    ir = import_model(conv_model(with_bias=False))
    match = re.search(r"npu\.conv2d ins\(([^:]*):", ir)
    assert match is not None
    assert match.group(1).count(",") == 1


def test_a_grouped_convolution_carries_its_group_attribute() -> None:
    ir = import_model(
        conv_model(
            input_shape=(1, 8, 8, 8),
            output_shape=(1, 8, 8, 8),
            weight_shape=(8, 1, 3, 3),
            group=8,
        )
    )
    assert "group = 8 : i64" in ir


def test_a_dilated_convolution_with_asymmetric_pads_keeps_the_onnx_pad_order() -> None:
    """pads is `[padTop, padLeft, padBottom, padRight]` from ONNX all the way in.

    Asymmetric on both axes so a transposition of the pair would change the
    result shape rather than being invisible. With `dilation = 2` the effective
    kernel is 5, so the height is `(8 + 1 + 2 - 5) + 1 = 7` and the width is
    `(8 + 2 + 0 - 5) + 1 = 6`.
    """
    ir = import_model(
        conv_model(
            input_shape=(1, 3, 8, 8),
            output_shape=(1, 4, 7, 6),
            pads=[1, 2, 2, 0],
            dilations=[2, 2],
        )
    )
    assert "pads = array<i64: 1, 2, 2, 0>" in ir
    assert "dilations = array<i64: 2, 2>" in ir
    assert "tensor<1x4x7x6xf32>" in ir


def test_auto_pad_is_refused_rather_than_resolved() -> None:
    """A graph `onnx.checker` accepts and this importer still refuses.

    `SAME_UPPER` with a 3 by 3 kernel at unit stride gives the same 8 by 8
    output as the explicit padding does, so ONNX is perfectly happy with it.
    Refusing anyway is the safe minimum: resolving the padding here would be a
    second implementation of a rule that has to agree with the runtime exactly.
    """
    model = conv_model(auto_pad="SAME_UPPER", pads=None)
    with pytest.raises(ONNXImportError, match="auto_pad is 'SAME_UPPER'"):
        import_model(model)


def test_a_kernel_shape_disagreeing_with_the_filter_is_refused() -> None:
    """ONNX infers from `kernel_shape` and never looks at the filter's extents.

    So a graph whose `kernel_shape` says 5 by 5 and whose filter is 3 by 3
    passes `onnx.checker` cleanly, with an output shape computed from an extent
    the weights do not have. Guessing which of the two is right is how a
    convolution ends up reading the wrong window, so the importer refuses.
    """
    model = conv_model(kernel_shape=[5, 5], output_shape=(1, 4, 6, 6))
    with pytest.raises(ONNXImportError, match="kernel_shape is \\[5, 5\\]"):
        import_model(model)


def test_a_non_constant_filter_is_refused_at_import() -> None:
    model = make_model(
        [
            helper.make_node(
                "Conv",
                ["x", "w"],
                ["y"],
                name="conv0",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        ],
        [value("x", (1, 3, 8, 8)), value("w", (4, 3, 3, 3))],
        [value("y", (1, 4, 8, 8))],
    )
    with pytest.raises(ONNXImportError, match="filter 'w' must be a constant"):
        import_model(model)


def test_an_arithmetically_impossible_convolution_is_refused_by_the_verifier() -> None:
    """The dialect diagnoses this, not the importer, and that is the design.

    Section 7.2's window arithmetic lives in `NPUShapeUtils.cpp` and nowhere
    else. Reimplementing it in Python to produce a friendlier message would be a
    second implementation that has to agree with the first, which is the thing
    that file exists to prevent. The importer's job here is to make sure the
    verifier gets asked.
    """
    model = conv_model(
        input_shape=(1, 3, 2, 2),
        output_shape=(1, 4, 0, 0),
        weight_shape=(4, 3, 3, 3),
        pads=[0, 0, 0, 0],
    )
    with pytest.raises(VerificationError, match="npu.conv2d"):
        import_model(model)


# =============================================================================
# Gemm and MatMul.
# =============================================================================


def test_gemm_with_transb_transposes_the_constant_at_import() -> None:
    """`nn.Linear` exports as `Gemm` with `transB = 1`, and the transpose is free.

    The weight is a constant, so transposing it once here costs nothing at run
    time and the emitted IR contains no transpose at all. An importer that
    emitted one would be putting a full pass over the weights into every
    inference for a fact known at compile time.
    """
    rng = np.random.default_rng(3)
    model = make_model(
        [
            helper.make_node(
                "Gemm",
                ["x", "w", "b"],
                ["y"],
                name="fc0",
                transB=1,
                alpha=1.0,
                beta=1.0,
            )
        ],
        [value("x", (4, 6))],
        [value("y", (4, 3))],
        [
            initializer("w", rng.standard_normal((3, 6)).astype(np.float32)),
            initializer("b", rng.standard_normal(3).astype(np.float32)),
        ],
    )
    ir = import_model(model)
    assert count(ir, "transpose") == 0
    assert (
        "npu.matmul ins(%arg0, %cst, %cst_0 : tensor<4x6xf32>, tensor<6x3xf32>, tensor<3xf32>)"
        in ir
    )


def test_gemm_with_a_scale_is_refused_quoting_the_scale() -> None:
    rng = np.random.default_rng(3)
    model = make_model(
        [helper.make_node("Gemm", ["x", "w"], ["y"], name="fc0", alpha=2.0)],
        [value("x", (4, 6))],
        [value("y", (4, 3))],
        [initializer("w", rng.standard_normal((6, 3)).astype(np.float32))],
    )
    with pytest.raises(ONNXImportError, match="alpha is 2.0"):
        import_model(model)


def test_gemm_with_transa_is_refused_rather_than_transposing_an_activation() -> None:
    rng = np.random.default_rng(3)
    model = make_model(
        [helper.make_node("Gemm", ["x", "w"], ["y"], name="fc0", transA=1)],
        [value("x", (6, 4))],
        [value("y", (4, 3))],
        [initializer("w", rng.standard_normal((6, 3)).astype(np.float32))],
    )
    with pytest.raises(ONNXImportError, match="transA is 1"):
        import_model(model)


def test_matmul_is_rank_two_by_rank_two() -> None:
    rng = np.random.default_rng(5)
    model = make_model(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="mm0")],
        [value("x", (4, 6))],
        [value("y", (4, 3))],
        [initializer("w", rng.standard_normal((6, 3)).astype(np.float32))],
    )
    assert count(import_model(model), "matmul") == 1


def test_a_batched_matmul_is_refused_naming_the_ranks() -> None:
    rng = np.random.default_rng(5)
    model = make_model(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="mm0")],
        [value("x", (2, 4, 6))],
        [value("y", (2, 4, 3))],
        [initializer("w", rng.standard_normal((6, 3)).astype(np.float32))],
    )
    with pytest.raises(ONNXImportError, match="ranks 3 and 2"):
        import_model(model)


# =============================================================================
# Add, Mul, and the broadcasting policy.
# =============================================================================


def _binary(op_type: str, rhs_shape: tuple[int, ...], shape=(1, 4, 3, 3)):
    rng = np.random.default_rng(11)
    return make_model(
        [helper.make_node(op_type, ["x", "k"], ["y"], name="bin0")],
        [value("x", shape)],
        [value("y", shape)],
        [initializer("k", rng.standard_normal(rhs_shape).astype(np.float32))],
    )


def test_same_shaped_operands_pass_straight_through() -> None:
    ir = import_model(_binary("Add", (1, 4, 3, 3)))
    assert "npu.add ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>)" in ir


@pytest.mark.parametrize("initializer_shape", [(4, 1, 1), (1, 4, 1, 1)])
def test_the_channel_carve_out_leaves_a_rank_one_addend(initializer_shape) -> None:
    """Section 11's carve out, on both shapes an ONNX per channel constant takes.

    `(1, C, 1, 1)` is the one that matters in practice: it is what the dynamo
    exporter writes for a bias or a scale written in PyTorch as
    `p.reshape(1, -1, 1, 1)`, so an importer that matched only a literally rank 1
    initializer would expand every one of them and the carve out would never
    fire on a single model in the suite. The emitted constant is rank 1 either
    way, which is what `-npu-fuse-bias` guards on.
    """
    ir = import_model(_binary("Add", initializer_shape))
    assert "npu.add ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<4xf32>)" in ir
    assert "tensor<1x4x3x3xf32>" in ir


def test_the_carve_out_reaches_mul_as_well_as_add() -> None:
    ir = import_model(_binary("Mul", (1, 4, 1, 1)))
    assert "npu.mul ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<4xf32>)" in ir


def test_a_literally_rank_one_initializer_is_a_width_broadcast_not_a_channel_one() -> (
    None
):
    """The bug this test exists for, found while writing the fixtures (D-0014).

    ONNX broadcasting aligns from the trailing axis, so an initializer of dims
    `[C]` against an `N x C x H x W` activation broadcasts over the **width**.
    A carve out that matched on "rank 1 and length equals the channel count"
    would import a per column constant as a per channel one on any model where
    the channel count and the width happen to be equal, which is a wrong answer
    that typechecks. Here the activation is `1 x 4 x 3 x 4`, so the channel
    count and the width are both 4 and the two readings are distinguishable
    only by knowing the rule.
    """
    ir = import_model(_binary("Add", (4,), shape=(1, 4, 3, 4)))
    assert "npu.add ins(%arg0, %cst : tensor<1x4x3x4xf32>, tensor<1x4x3x4xf32>)" in ir


def test_a_conv_followed_by_a_rank_one_add_imports_to_a_rank_one_addend() -> None:
    """The P3 gate item, and the thing `-npu-fuse-bias` is specified to match.

    An expanded addend would make the guard on a channel shaped constant addend
    never match on any model in the suite, so the pass's ablation row would be a
    row of zeros and the phase would look done while doing nothing.
    """
    rng = np.random.default_rng(13)
    model = make_model(
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
    ir = import_model(model)
    assert re.search(
        r"npu\.add ins\(%\w+, %cst\w* : tensor<1x4x5x5xf32>, tensor<4xf32>\)", ir
    )
    # And the convolution itself has no bias operand, so the fusion pass has
    # something to fuse rather than a convolution that is already complete.
    match = re.search(r"npu\.conv2d ins\(([^:]*):", ir)
    assert match is not None and match.group(1).count(",") == 1


def test_a_rank_one_operand_on_the_left_is_commuted_rather_than_refused() -> None:
    """Both operations are commutative, so there is one legal spelling.

    `npu.add` refuses a rank 1 lhs on purpose, which means the importer has to
    normalise rather than pass the node through as it found it. The result is
    that `-npu-fuse-bias` has one form to match instead of two.
    """
    rng = np.random.default_rng(17)
    model = make_model(
        [helper.make_node("Add", ["k", "x"], ["y"], name="bin0")],
        [value("x", (1, 4, 3, 3))],
        [value("y", (1, 4, 3, 3))],
        [initializer("k", rng.standard_normal((1, 4, 1, 1)).astype(np.float32))],
    )
    ir = import_model(model)
    assert "npu.add ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<4xf32>)" in ir


def test_a_broadcast_that_is_not_the_carve_out_is_materialised_same_shaped() -> None:
    """A scalar broadcast is expanded at import, so the emitted IR is same shaped.

    Section 11 expands every broadcast except the channel one. The point of
    expanding rather than representing is that `npu.add` then has exactly one
    broadcast to know about instead of a family of them.
    """
    ir = import_model(_binary("Mul", (1,)))
    assert "npu.mul ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>)" in ir


def test_a_broadcast_over_a_spatial_axis_is_expanded_not_carved_out() -> None:
    """A rank 1 constant of the width is a broadcast, and not the channel one.

    It is worth a test of its own because it is the shape a rule that matched
    "rank 1" without checking the length would have accepted, which would have
    emitted a rank 1 operand the verifier reads as a channel broadcast over the
    wrong axis.
    """
    ir = import_model(_binary("Add", (3,)))
    assert "npu.add ins(%arg0, %cst : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>)" in ir


def test_a_run_time_broadcast_is_refused_naming_both_shapes() -> None:
    model = make_model(
        [helper.make_node("Add", ["x", "k"], ["y"], name="bin0")],
        [value("x", (1, 4, 3, 3)), value("k", (1, 4, 1, 1))],
        [value("y", (1, 4, 3, 3))],
    )
    with pytest.raises(ONNXImportError, match=r"computed at run time"):
        import_model(model)


# =============================================================================
# Relu, Clip and Identity.
# =============================================================================


def test_relu_imports() -> None:
    model = make_model(
        [helper.make_node("Relu", ["x"], ["y"], name="relu0")],
        [value("x", (1, 4, 3, 3))],
        [value("y", (1, 4, 3, 3))],
    )
    assert count(import_model(model), "relu") == 1


def _clip(minimum=None, maximum=None, *, inputs=None):
    initializers = []
    node_inputs = ["x"]
    if inputs is not None:
        node_inputs = inputs
    else:
        if minimum is not None:
            node_inputs.append("lo")
            initializers.append(initializer("lo", np.array(minimum, dtype=np.float32)))
        if maximum is not None:
            node_inputs.append("hi")
            initializers.append(initializer("hi", np.array(maximum, dtype=np.float32)))
    return make_model(
        [helper.make_node("Clip", node_inputs, ["y"], name="clip0")],
        [value("x", (1, 4, 3, 3))],
        [value("y", (1, 4, 3, 3))],
        initializers,
    )


def test_clip_with_a_zero_lower_bound_and_no_upper_bound_is_a_relu() -> None:
    assert count(import_model(_clip(minimum=0.0)), "relu") == 1


def test_clip_with_an_infinite_upper_bound_is_a_relu() -> None:
    assert count(import_model(_clip(minimum=0.0, maximum=float("inf"))), "relu") == 1


def test_relu6_is_refused_rather_than_becoming_a_third_activation_case() -> None:
    with pytest.raises(ONNXImportError, match="minimum 0.0 and maximum 6.0"):
        import_model(_clip(minimum=0.0, maximum=6.0))


def test_a_clip_with_a_non_zero_lower_bound_is_refused_quoting_it() -> None:
    with pytest.raises(ONNXImportError, match="minimum -1.0"):
        import_model(_clip(minimum=-1.0))


def test_a_clip_whose_bound_input_is_present_but_empty_is_read_as_absent() -> None:
    """Since opset 11 the bounds are optional inputs, and an omitted one is `""`.

    A graph transform that removed the upper bound leaves the name empty rather
    than shortening the input list, so an importer that only checked the list
    length would read the empty name as a tensor and fail looking it up.
    """
    model = make_model(
        [helper.make_node("Clip", ["x", "lo", ""], ["y"], name="clip0")],
        [value("x", (1, 4, 3, 3))],
        [value("y", (1, 4, 3, 3))],
        [initializer("lo", np.array(0.0, dtype=np.float32))],
    )
    assert count(import_model(model), "relu") == 1


def test_a_clip_bound_that_is_an_empty_tensor_is_read_as_absent() -> None:
    """The other spelling of an omitted bound, and the reason `.size` is checked.

    A bound present as a zero element tensor is not a bound, and reading its
    first element would be an index error rather than a diagnostic.
    """
    model = make_model(
        [helper.make_node("Clip", ["x", "lo", "hi"], ["y"], name="clip0")],
        [value("x", (1, 4, 3, 3))],
        [value("y", (1, 4, 3, 3))],
        [
            initializer("lo", np.array(0.0, dtype=np.float32)),
            initializer("hi", np.zeros((0,), dtype=np.float32)),
        ],
    )
    assert count(import_model(model), "relu") == 1


def test_a_chain_of_identities_is_imported_past_rather_than_refused() -> None:
    """The P3 gate item: `Identity, Identity, Conv, BatchNormalization`.

    Exporters and graph transforms leave `Identity` nodes behind routinely, so
    an importer that rejected them would reject otherwise valid models for no
    reason. Binding rather than emitting means the chain costs nothing at all.
    """
    rng = np.random.default_rng(19)
    model = make_model(
        [
            helper.make_node("Identity", ["x"], ["i0"], name="ident0"),
            helper.make_node("Identity", ["i0"], ["i1"], name="ident1"),
            helper.make_node(
                "Conv",
                ["i1", "w"],
                ["c"],
                name="conv0",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node(
                "BatchNormalization",
                ["c", "s", "b", "m", "v"],
                ["y"],
                name="bn0",
                epsilon=1e-5,
            ),
        ],
        [value("x", (1, 3, 5, 5))],
        [value("y", (1, 4, 5, 5))],
        [
            initializer("w", rng.standard_normal((4, 3, 3, 3)).astype(np.float32)),
            initializer("s", (rng.random(4) + 0.5).astype(np.float32)),
            initializer("b", rng.standard_normal(4).astype(np.float32)),
            initializer("m", rng.standard_normal(4).astype(np.float32)),
            initializer("v", (rng.random(4) + 0.5).astype(np.float32)),
        ],
    )
    ir = import_model(model)
    assert count(ir, "conv2d") == 1
    assert count(ir, "batch_norm") == 1
    # The identities left nothing behind, and the convolution reads the function
    # argument directly rather than a copy of it.
    assert "npu.conv2d ins(%arg0" in ir


# =============================================================================
# The pools.
# =============================================================================


def _pool(op_type: str, output_shape=(1, 4, 4, 4), **attributes):
    return make_model(
        [helper.make_node(op_type, ["x"], ["y"], name="pool0", **attributes)],
        [value("x", (1, 4, 8, 8))],
        [value("y", output_shape)],
    )


def test_max_pool_imports_with_its_window() -> None:
    ir = import_model(
        _pool("MaxPool", kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0])
    )
    assert "kernel = array<i64: 2, 2>" in ir
    assert count(ir, "max_pool2d") == 1


def test_max_pool_asking_for_indices_is_refused() -> None:
    """The `Indices` output is left dangling, which is what a real graph does.

    It is not a graph output here on purpose. Making it one would refuse the
    model for its INT64 signature before the pooling converter ever ran, and the
    diagnostic would be about element types rather than about the thing that is
    actually wrong, which is that this project's pooling produces values only.
    """
    model = make_model(
        [
            helper.make_node(
                "MaxPool",
                ["x"],
                ["y", "idx"],
                name="pool0",
                kernel_shape=[2, 2],
                strides=[2, 2],
            )
        ],
        [value("x", (1, 4, 8, 8))],
        [value("y", (1, 4, 4, 4))],
    )
    with pytest.raises(ONNXImportError, match="Indices"):
        import_model(model)


def test_average_pool_with_count_include_pad_and_real_pads_is_refused() -> None:
    """The rule Section 11 states, on the case where the two settings disagree.

    This project's average pool divides by the number of elements that
    contributed. With a non zero pad, `count_include_pad = 1` divides by the
    window area instead, so the two compute different numbers and silently
    ignoring the attribute is what makes an importer disagree with ONNX.
    """
    with pytest.raises(ONNXImportError, match="count_include_pad is 1 with pads"):
        import_model(
            _pool(
                "AveragePool",
                output_shape=(1, 4, 5, 5),
                kernel_shape=[2, 2],
                strides=[2, 2],
                pads=[1, 1, 1, 1],
                count_include_pad=1,
            )
        )


def test_average_pool_with_count_include_pad_and_no_pads_is_accepted() -> None:
    """And the case where it is inert, which is every export this suite makes.

    The dynamo exporter writes `count_include_pad = 1` on every `AveragePool` it
    emits, pads or no pads. With every pad at zero no window overlaps the
    padding, the two settings produce bit identical results, and an
    unconditional refusal would reject a provably fine graph while proving
    nothing.
    """
    ir = import_model(
        _pool(
            "AveragePool",
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            count_include_pad=1,
        )
    )
    assert count(ir, "avg_pool2d") == 1


def test_average_pool_with_dilations_is_refused() -> None:
    """`dilations` arrived on `AveragePool` at opset 19 with a shape rule change.

    Section 11's safe minimum is refusal: an operator whose specification moved
    and whose converter did not is a silent wrong answer.
    """
    with pytest.raises(ONNXImportError, match=r"dilations is \[2, 2\]"):
        import_model(
            _pool(
                "AveragePool",
                output_shape=(1, 4, 3, 3),
                kernel_shape=[2, 2],
                strides=[2, 2],
                dilations=[2, 2],
            )
        )


def test_global_average_pool_becomes_a_full_extent_pool() -> None:
    ir = import_model(_pool("GlobalAveragePool", output_shape=(1, 4, 1, 1)))
    assert "kernel = array<i64: 8, 8>" in ir
    assert count(ir, "avg_pool2d") == 1


# =============================================================================
# BatchNormalization.
# =============================================================================


def _batch_norm(outputs=("y",), **attributes):
    rng = np.random.default_rng(23)
    output_infos = [value("y", (1, 4, 3, 3))]
    for extra in outputs[1:]:
        output_infos.append(value(extra, (4,)))
    return make_model(
        [
            helper.make_node(
                "BatchNormalization",
                ["x", "s", "b", "m", "v"],
                list(outputs),
                name="bn0",
                epsilon=1e-5,
                **attributes,
            )
        ],
        [value("x", (1, 4, 3, 3))],
        output_infos,
        [
            initializer("s", (rng.random(4) + 0.5).astype(np.float32)),
            initializer("b", rng.standard_normal(4).astype(np.float32)),
            initializer("m", rng.standard_normal(4).astype(np.float32)),
            initializer("v", (rng.random(4) + 0.5).astype(np.float32)),
        ],
    )


def test_batch_normalization_imports_with_its_epsilon() -> None:
    ir = import_model(_batch_norm())
    assert count(ir, "batch_norm") == 1
    assert "epsilon =" in ir


def test_the_training_form_is_refused_at_import() -> None:
    """Three outputs with `training_mode = 1`, which is a well formed ONNX graph.

    Both halves have to be set, because ONNX ties them together: three outputs
    at `training_mode = 0` and one output at `training_mode = 1` are each
    refused by the checker. That is why the importer asks one question about
    both facts rather than two questions, one of which could never fire.
    """
    with pytest.raises(ONNXImportError, match="3 outputs and training_mode 1"):
        import_model(_batch_norm(outputs=("y", "mean_out", "var_out"), training_mode=1))


def test_non_constant_batch_norm_parameters_are_refused_at_import() -> None:
    """The earliest layer that can name the problem is the one that should.

    At this level the diagnostic can say which ONNX node and which parameter. By
    the lowering it would be a message about an operand of an operation nobody
    can trace back to a model.
    """
    model = make_model(
        [
            helper.make_node(
                "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], name="bn0"
            )
        ],
        [
            value("x", (1, 4, 3, 3)),
            value("s", (4,)),
            value("b", (4,)),
            value("m", (4,)),
            value("v", (4,)),
        ],
        [value("y", (1, 4, 3, 3))],
    )
    with pytest.raises(ONNXImportError, match="the scale 's' must be a constant"):
        import_model(model)


# =============================================================================
# Reshape, Flatten, Transpose and Concat.
# =============================================================================


def _reshape(target, input_shape=(4, 8, 2, 2), output_shape=(4, 32), **attributes):
    return make_model(
        [
            helper.make_node(
                "Reshape", ["x", "shape"], ["y"], name="reshape0", **attributes
            )
        ],
        [value("x", input_shape)],
        [value("y", output_shape)],
        [initializer("shape", np.array(target, dtype=np.int64))],
        check=False,
    )


def test_a_flattening_reshape_keeps_the_batch() -> None:
    """Section 11 calls this the most likely hidden bug in the whole frontend.

    `(4, 8, 2, 2)` to `(4, 32)` is right; `(1, 128)` typechecks all the way down
    and computes one enormous row.
    """
    ir = import_model(_reshape([4, -1]))
    assert "npu.reshape %arg0 : tensor<4x8x2x2xf32> to tensor<4x32xf32>" in ir


def test_a_reshape_that_folds_the_batch_away_is_refused() -> None:
    with pytest.raises(ONNXImportError, match="folds the batch of 4"):
        import_model(_reshape([1, -1], output_shape=(1, 128)))


def test_a_zero_in_the_target_shape_copies_the_input_extent() -> None:
    """`allowzero = 0`, which is the default, and the behaviour since opset 14."""
    ir = import_model(_reshape([0, -1]))
    assert "to tensor<4x32xf32>" in ir


def test_allowzero_with_a_literal_zero_is_refused() -> None:
    with pytest.raises(ONNXImportError, match="literal zero extent"):
        import_model(_reshape([0, 32], output_shape=(0, 32), allowzero=1))


def test_allowzero_with_no_zero_in_the_shape_is_accepted() -> None:
    """Which is every `Reshape` the dynamo exporter emits.

    It writes `allowzero = 1` unconditionally, and with no zero in the target
    shape the two settings are identical, so the refusal is conditioned on a
    zero actually being present.
    """
    ir = import_model(_reshape([4, 32], allowzero=1))
    assert "to tensor<4x32xf32>" in ir


def test_a_target_shape_with_two_inferred_extents_is_refused() -> None:
    """Refused, and `onnx.checker` is the layer that gets there first.

    The importer keeps its own guard because it completes the shape itself and
    a single inferred extent is an assumption its arithmetic makes, but this
    test asserts the observable behaviour rather than which layer spoke: the
    graph is refused, as one `ONNXImportError`, with the message the layer that
    caught it produced.
    """
    with pytest.raises(ONNXImportError, match="-1"):
        import_model(_reshape([-1, -1]))


def test_flatten_at_axis_one_keeps_the_batch() -> None:
    model = make_model(
        [helper.make_node("Flatten", ["x"], ["y"], name="flat0", axis=1)],
        [value("x", (4, 8, 2, 2))],
        [value("y", (4, 32))],
    )
    ir = import_model(model)
    assert "npu.reshape %arg0 : tensor<4x8x2x2xf32> to tensor<4x32xf32>" in ir


def test_flatten_at_axis_zero_is_refused_by_name() -> None:
    """`axis = 0` produces `(1, N * features)` exactly, which is the named bug."""
    model = make_model(
        [helper.make_node("Flatten", ["x"], ["y"], name="flat0", axis=0)],
        [value("x", (4, 8, 2, 2))],
        [value("y", (1, 128))],
    )
    with pytest.raises(ONNXImportError, match="axis is 0 and this importer flattens"):
        import_model(model)


def test_transpose_imports_with_its_permutation() -> None:
    model = make_model(
        [helper.make_node("Transpose", ["x"], ["y"], name="perm0", perm=[0, 2, 3, 1])],
        [value("x", (1, 5, 3, 6))],
        [value("y", (1, 3, 6, 5))],
    )
    ir = import_model(model)
    assert "permutation = array<i64: 0, 2, 3, 1>" in ir
    assert "tensor<1x3x6x5xf32>" in ir


def test_a_perm_that_is_not_a_permutation_is_refused() -> None:
    """Refused, with `onnx.checker` again reaching it first.

    Same shape of assertion as the double `-1` case: the observable behaviour is
    one `ONNXImportError`, and the importer's own guard stays because it indexes
    the input shape with `perm` and a repeated index would read one axis twice.
    """
    model = make_model(
        [helper.make_node("Transpose", ["x"], ["y"], name="perm0", perm=[0, 1, 1, 2])],
        [value("x", (1, 5, 3, 6))],
        [value("y", (1, 5, 5, 3))],
        check=False,
    )
    with pytest.raises(ONNXImportError, match="perm"):
        import_model(model)


def test_the_importers_own_permutation_guard_fires_on_a_graph_onnx_accepts() -> None:
    """A `perm` shorter than the input's rank, which ONNX's inference allows.

    ONNX computes the output shape by indexing the input with `perm`, so a short
    `perm` produces a lower rank result and the checker is satisfied as long as
    the declared output agrees. This importer indexes the input the same way, so
    without the guard it would emit an `npu.transpose` whose permutation is not
    one, which the dialect verifier would then refuse with a message about the
    IR rather than about the model.
    """
    model = make_model(
        [helper.make_node("Transpose", ["x"], ["y"], name="perm0", perm=[0, 2, 1])],
        [value("x", (1, 5, 3, 6))],
        [value("y", (1, 3, 5))],
    )
    with pytest.raises(ONNXImportError, match="not a permutation"):
        import_model(model)


def test_concat_normalises_a_negative_axis() -> None:
    """A negative axis is an ONNX convention, and it stops here.

    `npu.concat` requires a non negative axis, because a convention every
    consumer has to normalise again is one that a consumer eventually forgets.
    """
    model = make_model(
        [helper.make_node("Concat", ["a", "b"], ["y"], name="cat0", axis=-3)],
        [value("a", (1, 4, 3, 3)), value("b", (1, 6, 3, 3))],
        [value("y", (1, 10, 3, 3))],
    )
    ir = import_model(model)
    assert "axis = 1 : i64" in ir
    assert count(ir, "concat") == 1


# =============================================================================
# The attribute reader.
# =============================================================================


def _node_with(*attributes: onnx.AttributeProto) -> onnx.NodeProto:
    """A bare node carrying exactly these attributes.

    The attribute reader is tested at its own level rather than through
    `import_model`, because `onnx.checker` refuses a mistyped attribute before
    any converter sees it. That does not make the reader's rules redundant: the
    checker only knows the types the ONNX specification declares, and every rule
    here is about what happens when a node reaches a converter that was written
    against a different shape of graph.
    """
    node = helper.make_node("Conv", ["x", "w"], ["y"], name="conv0")
    del node.attribute[:]
    node.attribute.extend(attributes)
    return node


def test_an_attribute_of_the_wrong_declared_type_is_refused_not_coerced() -> None:
    """Section 11 reads attributes by declared type. This is why.

    A `group` declared as INTS instead of INT read through `attr.i` returns
    zero, and a convolution with zero groups is a division by zero in the shape
    arithmetic rather than an error anybody can act on.
    """
    node = _node_with(helper.make_attribute("group", [1, 1]))
    with pytest.raises(ONNXImportError, match="declared INTS"):
        attributes.get_int(node, "group", 1)


def test_an_explicitly_empty_attribute_is_not_read_as_an_absent_one() -> None:
    """The case truthiness cannot distinguish, and the reason for that module.

    `pads = []` is a legal thing for a graph transform to leave behind. Read as
    absent it silently becomes four zeros; read as present it is an empty list,
    and the length check that follows says so.
    """
    empty = helper.make_attribute("pads", [], attr_type=onnx.AttributeProto.INTS)
    node = _node_with(empty)
    assert attributes.get_ints(node, "pads", [0, 0, 0, 0]) == []
    assert attributes.has_attribute(node, "pads")


def test_an_absent_attribute_returns_the_default() -> None:
    node = _node_with()
    assert attributes.get_ints(node, "pads", [0, 0, 0, 0]) == [0, 0, 0, 0]
    assert attributes.get_int(node, "group", 1) == 1
    assert attributes.get_float(node, "epsilon", 1e-5) == pytest.approx(1e-5)
    assert attributes.get_string(node, "auto_pad", "NOTSET") == "NOTSET"
    assert not attributes.has_attribute(node, "pads")


def test_a_string_attribute_is_decoded_rather_than_compared_as_bytes() -> None:
    """`auto_pad` is stored as bytes, and `b"NOTSET" != "NOTSET"`.

    Comparing without decoding makes the auto_pad check false on every model,
    including the ones it exists to refuse, and nothing about that failure is
    visible: the importer simply accepts everything.
    """
    node = _node_with(helper.make_attribute("auto_pad", "SAME_UPPER"))
    assert attributes.get_string(node, "auto_pad", "NOTSET") == "SAME_UPPER"


def test_a_required_attribute_that_is_absent_raises() -> None:
    node = _node_with()
    with pytest.raises(ONNXImportError, match="'kernel_shape' is required"):
        attributes.require_ints(node, "kernel_shape")


def test_an_attribute_type_this_reader_does_not_handle_raises() -> None:
    """GRAPH, and every other type an unsupported operator would carry.

    An unknown type raises rather than silently returning a default, which is
    Section 11's rule and is the difference between refusing a subgraph carrying
    operator and importing it as if it were empty.
    """
    graph = helper.make_graph([], "sub", [], [])
    node = _node_with(helper.make_attribute("body", graph))
    with pytest.raises(ONNXImportError, match="declared GRAPH"):
        attributes.get_ints(node, "body", [])


def test_the_reader_covers_all_six_types_section_eleven_names() -> None:
    """FLOATS and TENSOR too, which no converter currently asks for.

    They are here because the switch is specified over six types and a partial
    one raises a `KeyError` rather than a diagnostic the day a seventh arrives.
    Covering them is what makes that claim checked rather than asserted.
    """
    tensor = initializer("t", np.arange(4, dtype=np.float32))
    node = _node_with(
        helper.make_attribute("scales", [1.0, 2.0]),
        helper.make_attribute("value", tensor),
    )
    assert attributes.get_floats(node, "scales", []) == pytest.approx([1.0, 2.0])
    assert attributes.get_floats(node, "absent", [0.5]) == pytest.approx([0.5])
    read = attributes.get_tensor(node, "value")
    assert read is not None and list(read.dims) == [4]
    assert attributes.get_tensor(node, "absent") is None


# =============================================================================
# Destinations, locations, and the verification gate itself.
# =============================================================================


def test_every_compute_operation_has_a_tensor_empty_destination() -> None:
    """The P3 gate item, checked by counting rather than by reading.

    Every operation with an `outs(...)` clause takes a destination, and Section
    11 makes the importer materialise it. Counting `tensor.empty` against the
    number of `outs` clauses catches a converter that reused one destination for
    two operations, which would typecheck and would alias two live values.
    """
    ir = import_model(conv_model())
    outs = re.findall(r"outs\((%\w+) :", ir)
    empties = re.findall(r"(%\w+) = tensor\.empty\(\)", ir)
    assert outs, "the fixture must contain at least one destination passing operation"
    assert sorted(outs) == sorted(empties)
    assert len(set(outs)) == len(outs)


def test_every_emitted_operation_carries_the_onnx_node_name_as_a_location() -> None:
    """Section 11 requires a `NameLoc` per operation, and a pass drops it silently.

    The locations have to be printed as well as built: a location the round trip
    does not carry is one the next stage never sees, which would end the whole
    debug story at this boundary.
    """
    ir = import_model(conv_model())
    assert 'loc("conv0")' in ir
    assert 'loc("w")' in ir


def test_the_emitted_operation_list_matches_what_the_converters_produce() -> None:
    """`EMITTED_OPERATIONS` is what `check-reachability.py` reads, so it is pinned.

    Without this the list could drift into being a set of strings that satisfies
    the reachability grep and describes nothing, which would turn law 2's import
    layer into a comment.
    """
    source = Path(op_mapping.__file__).read_text(encoding="utf-8")
    for mnemonic in EMITTED_OPERATIONS:
        assert f'"{mnemonic}"' in source, mnemonic


def test_a_discardable_attribute_on_an_npu_operation_is_refused() -> None:
    """The failure mode the emission mechanism introduces, and its close.

    MLIR keeps an attribute whose name matches no inherent one as a discardable
    attribute rather than rejecting it, so a misspelled `strides` would parse,
    print and verify while the operation used its ODS default. This is the check
    that makes that impossible, tested on the exact text the generic printer
    produces for it.
    """
    generic = (
        '%1 = "npu.conv2d"(%arg0, %arg1, %0) <{dilations = array<i64: 1, 1>}> '
        "{strydes = array<i64: 9, 9>} : (tensor<1xf32>) -> tensor<1xf32>"
    )
    with pytest.raises(VerificationError, match="strydes"):
        _reject_discardable_attributes(generic)


def test_the_check_does_not_fire_on_a_clean_operation() -> None:
    generic = (
        '%1 = "npu.conv2d"(%arg0, %arg1, %0) <{dilations = array<i64: 1, 1>}> '
        ": (tensor<1xf32>) -> tensor<1xf32>"
    )
    _reject_discardable_attributes(generic)


def test_verify_rejects_ir_the_dialect_refuses() -> None:
    """The gate is real: bad IR in, exception out, never text.

    Written against `npu.add`'s channel broadcast rule, which is the one place
    the dialect and the frontend had to be brought into agreement at this phase.
    """
    module = """
    func.func @f(%a: tensor<1x8x4x4xf32>, %b: tensor<3xf32>,
                 %d: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
      %0 = npu.add ins(%a, %b : tensor<1x8x4x4xf32>, tensor<3xf32>)
                   outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
      return %0 : tensor<1x8x4x4xf32>
    }
    """
    with pytest.raises(VerificationError, match="channel extent 8"):
        verify(module)


def test_a_dynamic_input_extent_is_refused() -> None:
    model = conv_model()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    with pytest.raises(ONNXImportError, match="dynamic extent on axis 0"):
        import_model(model)


def test_an_integer_input_is_refused_until_the_quantization_phase() -> None:
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"], name="id0")],
        [helper.make_tensor_value_info("x", onnx.TensorProto.INT8, [1, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.INT8, [1, 4])],
    )
    with pytest.raises(ONNXImportError, match="element type INT8"):
        import_model(model)
