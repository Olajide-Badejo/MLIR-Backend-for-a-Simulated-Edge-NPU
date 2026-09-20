# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 14's calibration rules, against hand computed numbers.

Every case here is an exact arithmetic claim rather than a tolerance, for the
reason the integer kernel tests give: the rules are closed form, so a test that
allowed a tolerance would pass a rule that was slightly wrong, and slightly
wrong is exactly what a calibration bug looks like from the outside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from npu_frontend.calibration import (
    CALIB_METHODS,
    PROFILE_VERSION,
    QMAX,
    QMIN,
    CalibrationError,
    ChannelObservation,
    NodeRecord,
    Observation,
    accumulator_bound_holds,
    activation_scale,
    build_profile,
    calibration_seed,
    decompose_multiplier,
    extend_to_include_zero,
    max_reduction,
    method_range,
    read_profile,
    weight_scales,
    write_profile,
)
from npu_frontend.input_classes import class_seed
from npu_frontend.refexec import dequantize


def test_the_range_is_extended_to_include_real_zero() -> None:
    """The post ReLU case Section 14 names, with its own numbers.

    An observed range of [0.03, 4.03] does not contain real zero. Unextended,
    the affine rule would put the zero point outside the tensor's own range and
    every padded convolution would be biased, because this project's kernel rule
    is that padding contributes the zero point.
    """
    assert extend_to_include_zero(0.03, 4.03) == (0.0, 4.03)
    assert extend_to_include_zero(-2.0, -0.5) == (-2.0, 0.0)
    assert extend_to_include_zero(-1.0, 1.0) == (-1.0, 1.0)


def test_a_post_relu_tensor_gets_the_zero_point_at_the_bottom_rail() -> None:
    """min 0, max 4.08: the scale is 4.08 / 255 = 0.016 and the zero point -128.

    The whole int8 range is spent on the positive side, which is the point of an
    affine activation and is what the symmetric interim path could not do.
    """
    answer = activation_scale(0.0, 4.08)
    assert answer.scale == pytest.approx(0.016, abs=1e-12)
    assert answer.zero_point == -128
    assert not answer.degenerate


def test_the_zero_point_represents_real_zero_to_within_half_a_step() -> None:
    """Section 14 asks for this assertion rather than a clamp, on every tensor."""
    for low, high in ((0.03, 4.03), (-2.5, 7.5), (-1.0, 1.0), (-6.0, 0.0)):
        answer = activation_scale(low, high)
        represented = dequantize(
            np.array([answer.zero_point], dtype=np.int8),
            answer.scale,
            answer.zero_point,
        )
        assert abs(float(represented[0])) <= 0.5 * answer.scale


def test_a_symmetric_range_lands_the_zero_point_in_the_middle() -> None:
    """[-1, 1] gives a scale of 2/255, and the zero point lands at 0.

    Rounding `1 / (2/255)` is 127.5, and Python's round takes the even one, so
    the zero point is 128 - 128 = 0 rather than 1. The tie is stated here
    because this is the one place the affine rule has one.
    """
    answer = activation_scale(-1.0, 1.0)
    assert answer.scale == pytest.approx(2.0 / 255.0)
    assert answer.zero_point == 0


def test_only_an_all_zero_tensor_is_degenerate_once_zero_is_included() -> None:
    """The two range rules interact, and Section 14's word "first" decides how.

    A tensor constant at 3.0 has an observed range of no width, so the
    degenerate rule looks like it should fire. It does not, because the
    extension to include real zero runs first and turns the range into [0, 3],
    which is representable and is the better answer: 3.0 then lands on the top
    rail instead of on q = 3 with most of the range unused.

    So the degenerate case is not "the tensor was constant". It is "the tensor
    was all zeros", which is the only observation the extension leaves with no
    width, and a zero scale must never escape the calibrator because the
    verifier refuses one and that refusal must not be how a user finds out.
    """
    constant = activation_scale(3.0, 3.0)
    assert constant.scale == pytest.approx(3.0 / 255.0)
    assert constant.zero_point == -128
    assert not constant.degenerate

    zeros = activation_scale(0.0, 0.0)
    assert zeros.scale == 1.0
    assert zeros.zero_point == 0
    assert zeros.degenerate


def test_weight_scales_are_symmetric_and_per_channel() -> None:
    """`scale_c = max(|min_c|, |max_c|) / 127`, one per output channel."""
    assert weight_scales([1.27, 0.635, 2.54]) == pytest.approx([0.01, 0.005, 0.02])


def test_an_all_zero_weight_channel_takes_a_unit_scale() -> None:
    """The weight side of the degenerate rule, for the same reason."""
    assert weight_scales([0.0, 1.27]) == pytest.approx([1.0, 0.01])


def test_the_multiplier_decomposes_into_the_pinned_fixed_point_form() -> None:
    """`M = M0 * 2^-(31 + shift)` with `M0` in `[2^30, 2^31)`.

    Three hand computed cases. A half is `2^30` with no shift. Three quarters is
    `3 * 2^29`, which is 1610612736, also with no shift. A quarter needs one
    shift, because `0.25 * 2^31` is `2^29` and would be below the rail.
    """
    assert decompose_multiplier(0.5) == (1 << 30, 0)
    assert decompose_multiplier(0.75) == (1610612736, 0)
    assert decompose_multiplier(0.25) == (1 << 30, 1)


def test_every_decomposition_lands_inside_the_declared_rails() -> None:
    """The property the binary format's own check enforces, over a sweep."""
    for exponent in range(-20, 1):
        for step in (1.0, 1.3, 1.7):
            multiplier = step * (2.0**exponent)
            if multiplier >= 1.0:
                continue
            fixed, shift = decompose_multiplier(multiplier)
            assert (1 << 30) <= fixed < (1 << 31)
            assert 0 <= shift <= 31
            assert fixed * 2.0 ** (-(31 + shift)) == pytest.approx(multiplier, rel=1e-9)


def test_a_multiplier_that_is_not_positive_is_refused_by_name() -> None:
    with pytest.raises(CalibrationError, match="finite and strictly"):
        decompose_multiplier(0.0)
    with pytest.raises(CalibrationError, match="finite and strictly"):
        decompose_multiplier(float("nan"))


def test_a_multiplier_of_one_or_more_has_no_representation() -> None:
    """The shift is a right shift, so the pair cannot express a gain."""
    with pytest.raises(CalibrationError, match="right shift"):
        decompose_multiplier(1.5)


def test_the_accumulator_guard_is_exact_at_its_boundary() -> None:
    """`K * 128 * 127 < 2^31`, so the largest admissible K is 132104.

    132104 times 16256 is 2147482624, which is below 2^31 by 1024. One more
    reduction step is 2147498880, which is over.
    """
    assert max_reduction() == 132104
    assert accumulator_bound_holds(132104)
    assert not accumulator_bound_holds(132105)
    assert 132104 * 128 * 127 == 2147482624


def test_the_seed_is_stable_and_namespaced_away_from_the_input_classes() -> None:
    """A calibration draw is not an evaluation draw, and the seed says so."""
    assert calibration_seed("lenet", 1, 32) == calibration_seed("lenet", 1, 32)
    assert calibration_seed("lenet", 1, 32) != calibration_seed("lenet", 1, 8)
    assert calibration_seed("lenet", 1, 32) != class_seed("lenet", 1, "normal")


def test_minmax_takes_the_observed_range_whatever_the_histogram_says() -> None:
    observation = Observation(minimum=-1.0, maximum=3.0, counts=[10] * 2048)
    assert method_range(observation, "minmax") == (-1.0, 3.0)


def test_a_clipping_method_never_widens_the_observed_range() -> None:
    """The property that separates the three searches from min/max.

    A long tailed distribution: the mass in the first tenth of the range and a
    single count out at the far end. Every method must return a range inside the
    observed one, and at least one must be strictly inside, or the option is a
    no op with three spellings.
    """
    counts = [0] * 2048
    for index in range(200):
        counts[index] = 1000
    counts[2000] = 1
    observation = Observation(minimum=0.0, maximum=10.0, counts=counts)

    widths = {}
    for method in CALIB_METHODS:
        low, high = method_range(observation, method)
        assert low >= observation.minimum
        assert high <= observation.maximum
        widths[method] = high - low

    assert widths["minmax"] == pytest.approx(10.0)
    assert min(widths.values()) < widths["minmax"]


def test_a_two_sided_distribution_is_trimmed_at_both_ends() -> None:
    """The reason the searches trim by mass rather than clipping the top.

    The mass sits in the middle with one count at each end. A one sided search
    would keep the whole negative tail and spend the range defending it.
    """
    counts = [0] * 2048
    counts[0] = 1
    for index in range(900, 1100):
        counts[index] = 1000
    counts[2047] = 1
    observation = Observation(minimum=-10.0, maximum=10.0, counts=counts)

    low, high = method_range(observation, "percentile")
    assert low > observation.minimum
    assert high < observation.maximum


def test_a_degenerate_observation_falls_back_to_its_own_range() -> None:
    """No search runs on a distribution with no width: there is nothing to clip
    and the histogram is empty."""
    observation = Observation(minimum=2.0, maximum=2.0, counts=[])
    for method in CALIB_METHODS:
        assert method_range(observation, method) == (2.0, 2.0)


def test_an_unknown_method_is_refused_by_name() -> None:
    observation = Observation(minimum=0.0, maximum=1.0, counts=[1] * 2048)
    with pytest.raises(CalibrationError, match="is not a calibration method"):
        method_range(observation, "kl")


def test_a_profile_round_trips_and_carries_what_reproduces_it(tmp_path: Path) -> None:
    """The committed file's contract: the seed, the count, and the limitation."""
    profile = build_profile(
        model_name="lenet",
        batch=1,
        count=32,
        observations={"x": Observation(minimum=-1.0, maximum=1.0, counts=[1] * 2048)},
        weights={"w": ChannelObservation(axis=0, absolute_maxima=[1.27, 2.54])},
        nodes={
            "Conv_0": NodeRecord(op_type="Conv", inputs=["x", "w", "b"], outputs=["y"])
        },
    )
    assert profile["schema_version"] == PROFILE_VERSION
    assert profile["inputs"] == 32
    assert profile["seed"] == calibration_seed("lenet", 1, 32)
    assert "standard normal" in profile["input_distribution"]
    assert profile["weights"]["w"]["scales"] == pytest.approx([0.01, 0.02])
    assert set(profile["ranges"]["x"]) == set(CALIB_METHODS)
    assert profile["nodes"]["Conv_0"]["op_type"] == "Conv"
    assert profile["nodes"]["Conv_0"]["inputs"] == ["x", "w", "b"]

    path = write_profile(profile, tmp_path / "calibration" / "lenet.json")
    assert read_profile(path) == profile
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_a_profile_from_another_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": PROFILE_VERSION + 1}), "utf-8")
    with pytest.raises(CalibrationError, match="version"):
        read_profile(path)


def test_the_rails_are_the_int8_ones() -> None:
    """Stated once, here, so a change to either is a failing test rather than a
    silently different arithmetic."""
    assert (QMIN, QMAX) == (-128, 127)


def _one_convolution_model(directory: Path) -> Path:
    """A one node graph with an unnamed node, built without torch.

    Unnamed on purpose: what this fixture exists to check is that the observer
    and the importer agree about the name such a node gets.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.ones((2, 1, 3, 3), dtype=np.float32), name="w")
    bias = numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="b")
    node = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [node],
        "one_convolution",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializer=[weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )
    path = directory / "one_convolution.onnx"
    onnx.save(model, str(path))
    return path


def test_the_node_section_is_keyed_by_the_name_the_ir_will_carry(
    tmp_path: Path,
) -> None:
    """The join between a profile and an operation, and it has to be exact.

    The compiler sees the ONNX node name and nothing else, because that is what
    the importer puts in the location. An unnamed node gets a synthesised name
    from the importer's own rule, and the observer calls that same function, so
    the two cannot drift apart.
    """
    from npu_frontend.calibration import observe_nodes

    path = _one_convolution_model(tmp_path)
    nodes = observe_nodes(path)

    assert list(nodes) == ["Conv_0"]
    record = nodes["Conv_0"]
    assert record.op_type == "Conv"
    assert record.inputs == ["x", "w", "b"]
    assert record.outputs == ["y"]


def test_only_the_quantizable_operations_are_recorded(tmp_path: Path) -> None:
    """A profile that described a Relu would describe an operation the rewrite
    is documented not to touch."""
    import numpy as np
    import onnx
    from npu_frontend.calibration import QUANTIZABLE_OPS, observe_nodes
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(np.ones((2, 1, 3, 3), dtype=np.float32), name="w")
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "w"],
                ["c"],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )
    path = tmp_path / "conv_relu.onnx"
    onnx.save(model, str(path))

    nodes = observe_nodes(path)
    assert set(nodes) == {"Conv_0"}
    assert "Relu" not in QUANTIZABLE_OPS


def _conv_relu_model(directory: Path, *, dynamic_batch: bool = False) -> Path:
    """A two node graph, so there is an intermediate value to see inside."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = numpy_helper.from_array(
        np.array(
            [[[[0.5, -0.25], [0.125, 0.0]]], [[[1.0, -2.0], [0.25, 0.75]]]],
            dtype=np.float32,
        ),
        name="w",
    )
    batch = "n" if dynamic_batch else 1
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "w"],
                ["c"],
                pads=[0, 0, 0, 0],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
            ),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, 1, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, 2, 3, 3])],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
    )
    destination = directory / ("dynamic.onnx" if dynamic_batch else "conv_relu.onnx")
    onnx.save(model, str(destination))
    return destination


def test_promoting_intermediates_makes_every_produced_value_an_output(
    tmp_path: Path,
) -> None:
    """The mechanism the observer depends on.

    A tensor between two nodes is not a graph output, and a runtime returns the
    outputs it is asked for. Without the promotion the observer could see the
    model's answer and nothing else, which is every tensor it actually needs.
    """
    import onnx
    from npu_frontend.calibration import promote_intermediates

    model = onnx.load(str(_conv_relu_model(tmp_path)))
    assert [entry.name for entry in model.graph.output] == ["y"]

    promoted = promote_intermediates(model)
    names = {entry.name for entry in promoted.graph.output}
    assert {"c", "y"} <= names
    # The original is not mutated, which is the importer's rule too.
    assert [entry.name for entry in model.graph.output] == ["y"]


def test_the_observer_sees_inside_the_graph(tmp_path: Path) -> None:
    """Every intermediate gets a range and a histogram over that range."""
    from npu_frontend.calibration import observe

    path = _conv_relu_model(tmp_path)
    observations = observe(path, model_name="conv_relu", count=3, bins=64)

    assert {"c", "y"} <= set(observations)
    for name in ("c", "y"):
        entry = observations[name]
        assert entry.minimum <= entry.maximum
        assert len(entry.counts) == 64
        assert sum(entry.counts) > 0

    # The relu's output cannot be negative, and the convolution's can, which is
    # the cheapest possible check that these are the real tensors rather than
    # the same one twice.
    assert observations["y"].minimum >= 0.0
    assert observations["c"].minimum < 0.0


def test_the_observer_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    """Two runs of the same model at the same count see the same ranges.

    This is what makes a committed profile worth committing: the numbers in it
    are a function of the model, the seed and the count, and of nothing else.
    """
    from npu_frontend.calibration import observe

    path = _conv_relu_model(tmp_path)
    first = observe(path, model_name="conv_relu", count=2, bins=32)
    second = observe(path, model_name="conv_relu", count=2, bins=32)

    assert set(first) == set(second)
    for name in first:
        assert first[name].minimum == second[name].minimum
        assert first[name].maximum == second[name].maximum
        assert first[name].counts == second[name].counts


def test_the_weight_observer_reads_the_initializer_rather_than_running_it(
    tmp_path: Path,
) -> None:
    """Exact numbers, because a weight is a constant and needs no observation.

    The two output channels hold 0.5, -0.25, 0.125, 0 and 1, -2, 0.25, 0.75, so
    the absolute maxima are 0.5 and 2.0 and the symmetric scales are those over
    127.
    """
    from npu_frontend.calibration import observe_weights, weight_scales

    weights = observe_weights(_conv_relu_model(tmp_path))
    assert set(weights) == {"w"}
    assert weights["w"].axis == 0
    assert weights["w"].absolute_maxima == pytest.approx([0.5, 2.0])
    assert weight_scales(weights["w"].absolute_maxima) == pytest.approx(
        [0.5 / 127.0, 2.0 / 127.0]
    )


def test_a_profile_of_a_real_graph_carries_all_four_sections(tmp_path: Path) -> None:
    """The whole observer, end to end, into the file that gets committed."""
    from npu_frontend.calibration import (
        observe,
        observe_nodes,
        observe_weights,
    )

    path = _conv_relu_model(tmp_path)
    profile = build_profile(
        model_name="conv_relu",
        batch=1,
        count=2,
        observations=observe(path, model_name="conv_relu", count=2, bins=32),
        weights=observe_weights(path),
        nodes=observe_nodes(path),
    )

    assert profile["observed"]
    assert profile["ranges"]
    assert profile["weights"]["w"]["axis"] == 0
    assert profile["nodes"]["Conv_0"]["outputs"] == ["c"]

    written = write_profile(profile, tmp_path / "conv_relu.json")
    assert read_profile(written)["model"] == "conv_relu"


def test_a_dynamic_extent_is_refused_by_name(tmp_path: Path) -> None:
    """A calibration run has to feed the graph a concrete array, so a dynamic
    extent is a refusal rather than a guess at a batch size."""
    import onnx
    from npu_frontend.calibration import calibration_inputs

    model = onnx.load(str(_conv_relu_model(tmp_path, dynamic_batch=True)))
    with pytest.raises(CalibrationError, match="dynamic extent on axis"):
        calibration_inputs(model, model_name="conv_relu", count=1)


def test_the_profile_carries_the_derived_pair_beside_every_range() -> None:
    """What the compiler reads, and why it is here rather than there.

    The affine rule is arithmetic over observed data and it is pinned by
    Section 14. Deriving it again in the compiler would put the same rule in
    two places with nothing comparing them, which is the observer against
    kernel disagreement that section opens by warning about. So the profile
    carries the pair and `-npu-calibrate` reads it.

    The numbers here are the post ReLU case: a range of [0, 4.08] gives a scale
    of 0.016 and a zero point of -128, which is the same pair
    `test_a_post_relu_tensor_gets_the_zero_point_at_the_bottom_rail` asserts
    directly.
    """
    counts = [0] * 2048
    counts[2047] = 1000
    profile = build_profile(
        model_name="relu_like",
        batch=1,
        count=8,
        observations={"y": Observation(minimum=0.0, maximum=4.08, counts=counts)},
        weights={},
    )

    entry = profile["activation_scales"]["y"]["minmax"]
    assert entry["scale"] == pytest.approx(0.016, abs=1e-12)
    assert entry["zero_point"] == -128
    assert entry["degenerate"] is False
    assert set(profile["activation_scales"]["y"]) == set(CALIB_METHODS)


def test_a_constant_zero_tensor_is_marked_degenerate_in_the_profile() -> None:
    """The warning case, carried as a flag rather than left for the reader.

    A zero scale must never escape the calibrator, so the degenerate rule
    substitutes 1.0, and the profile says it did. A consumer that saw only the
    scale would have no way to tell a real unit scale from a substituted one.
    """
    profile = build_profile(
        model_name="dead",
        batch=1,
        count=8,
        observations={"z": Observation(minimum=0.0, maximum=0.0, counts=[])},
        weights={},
    )

    entry = profile["activation_scales"]["z"]["minmax"]
    assert entry["scale"] == 1.0
    assert entry["zero_point"] == 0
    assert entry["degenerate"] is True
