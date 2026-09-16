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
    )
    assert profile["schema_version"] == PROFILE_VERSION
    assert profile["inputs"] == 32
    assert profile["seed"] == calibration_seed("lenet", 1, 32)
    assert "standard normal" in profile["input_distribution"]
    assert profile["weights"]["w"]["scales"] == pytest.approx([0.01, 0.02])
    assert set(profile["ranges"]["x"]) == set(CALIB_METHODS)

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
