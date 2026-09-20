# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 14's calibration: the observation, the range rules, and the profile.

**What this module is and what `-npu-calibrate` is.** The observer runs the
model and writes down what it saw; the pass turns what was seen into the
integers an instruction carries. The split is not arbitrary. A range is a
property of the data, so choosing one from an observed distribution belongs
where the distribution is, in numpy, where it can be tested against hand
computed cases. A scale, a zero point and a rescaling pair are properties of the
machine, so deriving those belongs in the compiler. The profile is the interface
between the two, and it is a committed file rather than a pipe, because Section
14 asks for the profile to be committed per model with its seed and its input
count so that an accuracy number is reproducible rather than anecdotal.

**The calibration inputs are seeded synthetic standard normal draws and are not
samples from any data distribution.** Section 14 asks for that sentence to be
stated plainly rather than buried, because it is a real limitation of a suite
with no dataset: a calibrated range is only as representative as the inputs that
produced it, and these inputs represent nothing but themselves. What they do
give is reproducibility, which is the property the phase's numbers depend on.

**The calibration draw is deliberately not the evaluation draw.** The seed here
is derived in `npu_frontend.input_classes`' style and from a different namespace,
so a model calibrated at count 32 and then measured on the `normal` class is not
being measured on the inputs it was tuned to. Reusing `class_seed` would have
been one line shorter and would have made every accuracy number in the phase a
measurement of the calibrator against its own training set.

**The four methods live here, and the pass selects between them.** A profile
carries the range each of `minmax`, `percentile`, `mse` and `entropy` chooses,
computed from the same observed histogram, and `-npu-calibrate`'s
`calib-method` option says which one to read. The alternative, carrying the
histogram into the compiler and implementing three more search loops in C++, put
the numerics in the place with no numpy and no test harness for them, and made
the compiler responsible for a decision about data.

**The three searches differ in what they minimise, not in what they search
over.** Each considers the same short list of trim fractions, each trim is taken
**from both tails by mass**, and the range that survives is the one its own
criterion prefers: the fixed quantile for `percentile`, the estimated squared
error for `mse`, and the divergence from the observed distribution for
`entropy`. Trimming both tails matters rather than being tidy: the output of a
convolution before its activation is two sided, and a search that clipped only
the top would spend the int8 range defending a tail it had already kept.

**The output channel axis is 0 here, and Section 14 says 0 or 3.** Section 14
takes its axis from the reference specification, whose depthwise weights are
laid out `[1, kH, kW, C]`, so the output channel axis is 3 for that layout. This
project imports ONNX, where a convolution weight is `(M, C/group, kH, kW)` for
both the regular and the depthwise case, so the output channel axis is 0 in
both. The rule implemented is the one Section 14 means, "one scale per output
channel", and the axis is read from the layout in front of it rather than copied
from a document about a different one.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import ModelProto, numpy_helper

from npu_frontend.onnx_importer import name_every_node

#: The profile format's version, carried in every file this module writes.
#:
#: A profile is committed, so a reader years from now needs to know which rules
#: produced it. The version moves when the meaning of a field moves, which is
#: the same rule `Program::kVersion` follows one level down.
PROFILE_VERSION: Final[int] = 1

#: The four methods of Section 14, in the order that section lists them, with
#: `minmax` pinned as the default.
CALIB_METHODS: Final[tuple[str, ...]] = ("minmax", "percentile", "mse", "entropy")
DEFAULT_CALIB_METHOD: Final[str] = "minmax"

#: The input counts Section 14 asks to be measured, and the one a caller gets
#: when it does not choose. 32 rather than 8, because an unmotivated default of
#: 8 would make the count table's first row also the pinned configuration, and
#: that table exists to show what the count costs.
CALIBRATION_COUNTS: Final[tuple[int, ...]] = (8, 32, 128, 512)
DEFAULT_CALIBRATION_COUNT: Final[int] = 32

#: The histogram's bin count.
#:
#: 2048 rather than 256 or 8192. The three searches all read the shape of the
#: distribution's tails, and a tail resolved into eight bins is a tail whose
#: shape is the binning's rather than the data's. Above a few thousand the bins
#: are emptier than the estimate they feed is precise.
HISTOGRAM_BINS: Final[int] = 2048

#: The fractions of mass the three searches consider trimming, half from each
#: tail. Zero is in the list because "clip nothing" has to be allowed to win.
TRIM_FRACTIONS: Final[tuple[float, ...]] = (
    0.0,
    1e-5,
    1e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)

#: The quantile the `percentile` method keeps, as a percentage of the mass.
#:
#: 99.99 percent, which is the figure the published comparisons use for 8 bit
#: activation calibration. It is a constant with a source rather than a tuned
#: one, and the ablation measures whether it pays on this suite.
PERCENTILE: Final[float] = 99.99

#: The operations Section 14's QDQ rewrite quantizes, and no others. `ADD`,
#: `MUL`, `POOL_AVG`, `RELU` and `POOL_MAX` reject i8 operands by name at the
#: instruction level, which is the same boundary stated one level down.
QUANTIZABLE_OPS: Final[tuple[str, ...]] = ("Conv", "Gemm", "MatMul")

#: The int8 rails, which every rule here is expressed against.
QMIN: Final[int] = -128
QMAX: Final[int] = 127
#: The number of levels an int8 tensor has, which is what a range divides by.
QLEVELS: Final[int] = QMAX - QMIN + 1


class CalibrationError(Exception):
    """A profile that cannot be produced or cannot be read."""


def calibration_seed(model: str, batch: int, count: int) -> int:
    """The seed for one calibration run, derived from what identifies it.

    `zlib.crc32` over a namespaced string, which is `input_classes.class_seed`'s
    convention and is stable across processes, machines and interpreters where
    `hash` is not. The namespace differs from the input classes' so a calibration
    draw and an evaluation draw never coincide.
    """
    return zlib.crc32(f"calibration:{model}:{batch}:{count}".encode())


@dataclass(frozen=True)
class Observation:
    """What the observer saw of one tensor, before any rule is applied."""

    minimum: float
    maximum: float
    #: The histogram of the observed values over `[minimum, maximum]`.
    counts: list[int] = field(default_factory=list)

    @property
    def degenerate(self) -> bool:
        """Whether every value observed was the same one."""
        return self.maximum <= self.minimum


@dataclass(frozen=True)
class ChannelObservation:
    """The per output channel absolute maxima of one weight initializer."""

    axis: int
    absolute_maxima: list[float]


def _graph_input_shapes(model: ModelProto) -> dict[str, tuple[int, ...]]:
    """Every graph input's static shape, initializers excluded."""
    initializers = {entry.name for entry in model.graph.initializer}
    shapes: dict[str, tuple[int, ...]] = {}
    for entry in model.graph.input:
        if entry.name in initializers:
            continue
        extents: list[int] = []
        for index, dimension in enumerate(entry.type.tensor_type.shape.dim):
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                extents.append(int(dimension.dim_value))
                continue
            raise CalibrationError(
                f"the graph input {entry.name!r} has a dynamic extent on axis "
                f"{index}, and a calibration run has to feed it a concrete "
                f"array. The suite's models are exported with static shapes, "
                f"so a dynamic extent here means the model is not one of them."
            )
        shapes[entry.name] = tuple(extents)
    return shapes


def calibration_inputs(
    model: ModelProto, *, model_name: str, count: int, batch: int = 1
) -> list[dict[str, NDArray[np.float32]]]:
    """`count` seeded standard normal feeds, one dictionary per run."""
    shapes = _graph_input_shapes(model)
    base = calibration_seed(model_name, batch, count)
    feeds: list[dict[str, NDArray[np.float32]]] = []
    for draw in range(count):
        feed: dict[str, NDArray[np.float32]] = {}
        for index, (name, shape) in enumerate(sorted(shapes.items())):
            rng = np.random.default_rng(base + draw * 1000 + index)
            feed[name] = rng.standard_normal(shape).astype(np.float32)
        feeds.append(feed)
    return feeds


def promote_intermediates(model: ModelProto) -> ModelProto:
    """A copy of `model` whose every intermediate value is also a graph output.

    This is how the observer sees inside the graph at all: onnxruntime returns
    the outputs it is asked for, and a tensor between two nodes is not one of
    them. Shape inference runs first so the promoted values carry a type, which
    is what lets the runtime bind them.
    """
    inferred = onnx.shape_inference.infer_shapes(model, check_type=True)
    promoted = ModelProto()
    promoted.CopyFrom(inferred)

    existing = {entry.name for entry in promoted.graph.output}
    produced: list[str] = []
    for node in promoted.graph.node:
        for output in node.output:
            if output and output not in existing:
                produced.append(output)
                existing.add(output)

    known = {entry.name: entry for entry in promoted.graph.value_info}
    for name in produced:
        if name in known:
            promoted.graph.output.append(known[name])
            continue
        # A value shape inference could not type. It is still promoted, with an
        # untyped declaration, because a tensor the observer cannot see is a
        # tensor the calibrator has no range for, and a missing range is a
        # skipped operation rather than a silently wrong one.
        promoted.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    return promoted


def observe(
    model_path: str | Path,
    *,
    model_name: str,
    count: int = DEFAULT_CALIBRATION_COUNT,
    batch: int = 1,
    bins: int = HISTOGRAM_BINS,
) -> dict[str, Observation]:
    """Run the model over seeded inputs and return one observation per tensor.

    **Two passes over the draws, and the second one is the reason.** The first
    takes the running minimum and maximum, which is all `minmax` needs. The
    histogram the other three read has to be binned over a range, and a range
    still moving while the histogram filled would make the early draws land in
    bins meaning something different from the late ones. So the range is settled
    first and the histogram is filled against it.

    The runs are not kept between the passes. Holding every intermediate tensor
    of every draw would be the whole suite's activations in memory at once, so
    the model is run twice instead, which costs time this phase has and memory
    it does not.
    """
    model = onnx.load(str(model_path))
    feeds = calibration_inputs(model, model_name=model_name, count=count, batch=batch)
    promoted = promote_intermediates(model)
    session = ort.InferenceSession(
        promoted.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    names = [entry.name for entry in session.get_outputs()]

    minima: dict[str, float] = {}
    maxima: dict[str, float] = {}
    for feed in feeds:
        for name, value in zip(names, session.run(None, feed), strict=True):
            array = np.asarray(value, dtype=np.float32)
            if array.size == 0:
                continue
            low = float(array.min())
            high = float(array.max())
            minima[name] = min(minima.get(name, low), low)
            maxima[name] = max(maxima.get(name, high), high)

    edges: dict[str, NDArray[np.float64]] = {}
    totals: dict[str, NDArray[np.int64]] = {}
    for name, low in minima.items():
        if maxima[name] <= low:
            continue
        edges[name] = np.linspace(low, maxima[name], bins + 1, dtype=np.float64)
        totals[name] = np.zeros(bins, dtype=np.int64)

    if edges:
        for feed in feeds:
            for name, value in zip(names, session.run(None, feed), strict=True):
                if name not in edges:
                    continue
                array = np.asarray(value, dtype=np.float64)
                if array.size == 0:
                    continue
                counted, _ = np.histogram(array, bins=edges[name])
                totals[name] += counted

    return {
        name: Observation(
            minimum=low,
            maximum=maxima[name],
            counts=([int(entry) for entry in totals[name]] if name in totals else []),
        )
        for name, low in sorted(minima.items())
    }


def observe_weights(model_path: str | Path) -> dict[str, ChannelObservation]:
    """The per output channel absolute maxima of every weight initializer.

    Read from the initializers rather than from a run, because a weight is a
    constant: its range is exact and needs no observation at all. The axis is 0,
    which is where ONNX puts the output channel for both the regular and the
    depthwise case.
    """
    model = onnx.load(str(model_path))
    initializers = {entry.name: entry for entry in model.graph.initializer}
    weights: dict[str, ChannelObservation] = {}
    for node in model.graph.node:
        if node.op_type not in QUANTIZABLE_OPS:
            continue
        if len(node.input) < 2:
            continue
        name = node.input[1]
        if name not in initializers or name in weights:
            continue
        array = numpy_helper.to_array(initializers[name])
        if array.ndim == 0:
            continue
        flattened = array.reshape(array.shape[0], -1)
        weights[name] = ChannelObservation(
            axis=0,
            absolute_maxima=[float(np.abs(row).max()) for row in flattened],
        )
    return weights


@dataclass(frozen=True)
class NodeRecord:
    """One quantizable node, by the name the IR will carry for it."""

    op_type: str
    inputs: list[str]
    outputs: list[str]


def observe_nodes(model_path: str | Path) -> dict[str, NodeRecord]:
    """The quantizable nodes, keyed by the name their operation will carry.

    **This is what connects a profile to an operation.** Everything else in a
    profile is keyed by tensor name, because a tensor is what has a range. The
    compiler does not see tensor names: the importer gives every operation a
    `NameLoc` holding the ONNX **node** name, so the pass standing on an
    `npu.conv2d` knows which node it came from and nothing else. This section is
    the join, and it names the node's own inputs and outputs so the pass can ask
    for their ranges.

    The naming comes from the importer's own `name_every_node`, not from a
    second copy of the rule, so a model whose exporter left its nodes unnamed
    gets the same synthesised names on both sides.

    Only the operations the QDQ rewrite can quantize are recorded. A profile
    that also described every `Relu` would be describing operations the rewrite
    is documented not to touch, and Section 14's boundary is deliberate.
    """
    model = onnx.load(str(model_path))
    name_every_node(model.graph)
    records: dict[str, NodeRecord] = {}
    for node in model.graph.node:
        if node.op_type not in QUANTIZABLE_OPS:
            continue
        records[node.name] = NodeRecord(
            op_type=node.op_type,
            inputs=[entry for entry in node.input if entry],
            outputs=[entry for entry in node.output if entry],
        )
    return records


def _trimmed_bins(observation: Observation, trim: float) -> tuple[int, int]:
    """The half open bin span left after trimming `trim` of the mass, half from
    each tail."""
    counts = np.asarray(observation.counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0 or trim <= 0.0:
        return 0, counts.size
    cumulative = np.cumsum(counts) / total
    half = trim / 2.0
    low_candidates = np.nonzero(cumulative >= half)[0]
    high_candidates = np.nonzero(cumulative >= 1.0 - half)[0]
    low = int(low_candidates[0]) if low_candidates.size else 0
    high = int(high_candidates[0]) + 1 if high_candidates.size else counts.size
    if high <= low:
        return 0, counts.size
    return low, high


def _range_of_bins(
    observation: Observation, low: int, high: int
) -> tuple[float, float]:
    """A bin span back in the observation's own units."""
    bins = len(observation.counts)
    width = (observation.maximum - observation.minimum) / bins
    return (
        observation.minimum + width * low,
        observation.minimum + width * high,
    )


def _saturating_error(counts: NDArray[np.float64], low: int, high: int) -> float:
    """The squared error of clipping to `[low, high)` and quantizing inside it.

    Two terms, and both are paid in bin units, which is enough because every
    candidate is compared against every other in the same units. Inside the
    span, the error of collapsing the bins onto 256 uniform levels. Outside it,
    the error of every value saturating onto the rail it crossed.
    """
    inside = counts[low:high]
    error = 0.0
    if inside.sum() > 0:
        edges = np.linspace(0, inside.size, QLEVELS + 1)
        for level in range(QLEVELS):
            first = int(np.floor(edges[level]))
            last = int(np.ceil(edges[level + 1]))
            block = inside[first:last]
            if block.sum() <= 0:
                continue
            positions = np.arange(first, last, dtype=np.float64)
            centre = float((positions * block).sum() / block.sum())
            error += float((block * (positions - centre) ** 2).sum())

    below = counts[:low]
    if below.sum() > 0:
        positions = np.arange(0, low, dtype=np.float64)
        error += float((below * (positions - low) ** 2).sum())
    above = counts[high:]
    if above.sum() > 0:
        positions = np.arange(high, counts.size, dtype=np.float64)
        error += float((above * (positions - (high - 1)) ** 2).sum())
    return error


def _divergence(counts: NDArray[np.float64], low: int, high: int) -> float:
    """The divergence of the quantized distribution from the observed one.

    The published entropy calibration's criterion: the mass outside the span is
    folded onto the edge bins, the span is collapsed onto 256 levels and spread
    back over the bins that were occupied, and the two distributions are
    compared. A candidate that throws away a populated tail pays for it here.
    """
    reference = counts[low:high].astype(np.float64).copy()
    if reference.size == 0 or reference.sum() <= 0:
        return float("inf")
    reference[0] += counts[:low].sum()
    reference[-1] += counts[high:].sum()

    approximation = np.zeros(reference.size, dtype=np.float64)
    edges = np.linspace(0, reference.size, QLEVELS + 1)
    for level in range(QLEVELS):
        first = int(np.floor(edges[level]))
        last = int(np.ceil(edges[level + 1]))
        block = reference[first:last]
        occupied = float((block > 0).sum())
        if occupied <= 0:
            continue
        share = float(block.sum()) / occupied
        approximation[first:last] = np.where(block > 0, share, 0.0)

    if approximation.sum() <= 0:
        return float("inf")
    p = reference / reference.sum()
    q = approximation / approximation.sum()
    mask = (p > 0) & (q > 0)
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


def _searched_range(observation: Observation, criterion: str) -> tuple[float, float]:
    """The trim fraction whose range its own criterion prefers."""
    counts = np.asarray(observation.counts, dtype=np.float64)
    if counts.sum() <= 0:
        return observation.minimum, observation.maximum

    best: tuple[float, int, int] | None = None
    for trim in TRIM_FRACTIONS:
        low, high = _trimmed_bins(observation, trim)
        if criterion == "mse":
            score = _saturating_error(counts, low, high)
        else:
            score = _divergence(counts, low, high)
        if best is None or score < best[0]:
            best = (score, low, high)
    assert best is not None
    return _range_of_bins(observation, best[1], best[2])


def method_range(observation: Observation, method: str) -> tuple[float, float]:
    """The range one method chooses from one observation.

    Every method is named rather than swept out of a table with a default, so a
    method added to `CALIB_METHODS` and not here raises by name instead of
    quietly becoming min/max.
    """
    if method == "minmax":
        return observation.minimum, observation.maximum
    if method not in CALIB_METHODS:
        raise CalibrationError(
            f"{method!r} is not a calibration method. The four are "
            + ", ".join(CALIB_METHODS)
            + "."
        )
    if observation.degenerate or not observation.counts:
        return observation.minimum, observation.maximum
    if method == "percentile":
        low, high = _trimmed_bins(observation, 1.0 - PERCENTILE / 100.0)
        return _range_of_bins(observation, low, high)
    return _searched_range(observation, method)


def extend_to_include_zero(low: float, high: float) -> tuple[float, float]:
    """Section 14's first range rule, and it is a correctness requirement.

    The scheme needs real zero to be exactly representable so that zero padding
    is exact, and this project's kernel rule that padding contributes the zero
    point is only correct if it holds. A post ReLU tensor whose observed minimum
    is 0.03 puts real zero outside its own calibrated range, the zero point gets
    clamped, and every padded convolution is quietly biased.
    """
    return min(low, 0.0), max(high, 0.0)


@dataclass(frozen=True)
class ActivationScale:
    """The affine scale and zero point of one activation tensor."""

    scale: float
    zero_point: int
    degenerate: bool = False


def activation_scale(low: float, high: float) -> ActivationScale:
    """Section 14's affine activation rule, with its degenerate case.

    `scale = (max - min) / 255` and `zero_point = round(-min / scale) - 128`,
    after the range has been extended to include real zero. A degenerate range
    yields a scale of 1 and a zero point of 0 rather than a zero scale, because
    a zero scale is refused by the verifier and that refusal must not be how a
    user learns their tensor was constant.
    """
    low, high = extend_to_include_zero(low, high)
    if high <= low:
        return ActivationScale(scale=1.0, zero_point=0, degenerate=True)
    scale = (high - low) / float(QMAX - QMIN)
    zero_point = int(round(-low / scale)) + QMIN
    zero_point = max(QMIN, min(QMAX, zero_point))
    return ActivationScale(scale=scale, zero_point=zero_point)


def weight_scales(absolute_maxima: Sequence[float]) -> list[float]:
    """Section 14's symmetric per output channel weight rule.

    `scale_c = max(abs(min_c), abs(max_c)) / 127` with a zero point of zero. A
    channel whose weights are all zero takes a scale of 1 for the reason the
    degenerate activation does.
    """
    return [
        (float(maximum) / float(QMAX)) if maximum > 0.0 else 1.0
        for maximum in absolute_maxima
    ]


def decompose_multiplier(multiplier: float) -> tuple[int, int]:
    """`M` as `M0 * 2^-(31 + shift)` with `M0` in `[2^30, 2^31)`.

    The decomposition Section 14 pins, and the one the machine applies with a
    saturating doubling high multiply and a rounding divide by a power of two.
    The shift is non negative and at most 31, which is the range the binary
    format's own check accepts, so a multiplier of one or more is refused here
    rather than encoded into a field that cannot hold it.
    """
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise CalibrationError(
            f"the requantization multiplier must be finite and strictly "
            f"positive, and it is {multiplier}. A multiplier of zero is what a "
            f"missing calibration leaves behind, and it would produce an all "
            f"zero result rather than a diagnostic."
        )
    shift = 0
    scaled = float(multiplier)
    while scaled >= 1.0:
        scaled /= 2.0
        shift -= 1
    while scaled < 0.5:
        scaled *= 2.0
        shift += 1
    fixed = int(round(scaled * float(1 << 31)))
    if fixed == (1 << 31):
        fixed //= 2
        shift -= 1
    if shift < 0 or shift > 31:
        raise CalibrationError(
            f"the requantization shift for a multiplier of {multiplier} is "
            f"{shift}, which is outside the [0, 31] the binary format accepts. "
            f"The shift is a right shift, so a multiplier of one or more has no "
            f"representation in these two fields."
        )
    return fixed, shift


def accumulator_bound_holds(reduction: int) -> bool:
    """Section 14's static guard: `K * 128 * 127 < 2^31`."""
    return reduction * 128 * 127 < (1 << 31)


def max_reduction() -> int:
    """The largest reduction depth the int32 accumulator's guard admits."""
    return ((1 << 31) - 1) // (128 * 127)


def _activation_entry(low: float, high: float) -> dict[str, Any]:
    """One tensor's affine scale and zero point, as the profile records them."""
    answer = activation_scale(low, high)
    return {
        "scale": answer.scale,
        "zero_point": answer.zero_point,
        "degenerate": answer.degenerate,
    }


def build_profile(
    *,
    model_name: str,
    batch: int,
    count: int,
    observations: dict[str, Observation],
    weights: dict[str, ChannelObservation],
    nodes: dict[str, NodeRecord] | None = None,
) -> dict[str, Any]:
    """The committed profile: what was seen, and what each method makes of it.

    The raw observation is kept beside the four ranges rather than discarded,
    because a profile whose ranges cannot be rederived is a profile nobody can
    check. The histogram is not kept: it is large, it is reproducible from the
    seed and the count, and a committed file that grew by a megabyte a model
    would make Section 14's reproducibility argument expensive enough to lose an
    argument with.
    """
    return {
        "schema_version": PROFILE_VERSION,
        "model": model_name,
        "batch": batch,
        "inputs": count,
        "seed": calibration_seed(model_name, batch, count),
        "input_distribution": "synthetic standard normal, not sampled data",
        "nodes": {
            name: {
                "op_type": record.op_type,
                "inputs": record.inputs,
                "outputs": record.outputs,
            }
            for name, record in sorted((nodes or {}).items())
        },
        "observed": {
            name: {"min": observation.minimum, "max": observation.maximum}
            for name, observation in sorted(observations.items())
        },
        "ranges": {
            name: {
                method: list(method_range(observation, method))
                for method in CALIB_METHODS
            }
            for name, observation in sorted(observations.items())
        },
        # **The derived numbers live here rather than in the compiler**, and
        # that is the same decision the ranges were: the affine rule is
        # arithmetic over observed data, it is pinned by Section 14, and it is
        # held to hand computed cases in `test_calibration.py`. A compiler that
        # derived them again would be a second implementation of a rule with no
        # oracle comparing the two, which is exactly the disagreement Section
        # 14 opens by warning about. `-npu-calibrate` reads the pair for the
        # method it was asked for and writes it into the instruction.
        "activation_scales": {
            name: {
                method: _activation_entry(*method_range(observation, method))
                for method in CALIB_METHODS
            }
            for name, observation in sorted(observations.items())
        },
        "weights": {
            name: {
                "axis": entry.axis,
                "absolute_maxima": entry.absolute_maxima,
                "scales": weight_scales(entry.absolute_maxima),
            }
            for name, entry in sorted(weights.items())
        },
    }


def write_profile(profile: dict[str, Any], path: str | Path) -> Path:
    """Write a profile where the rest of the project writes committed JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def read_profile(path: str | Path) -> dict[str, Any]:
    """Read a profile and refuse one this module does not understand."""
    try:
        profile = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(
            f"the calibration profile {path} is unreadable: {exc}"
        ) from exc
    version = profile.get("schema_version")
    if version != PROFILE_VERSION:
        raise CalibrationError(
            f"the calibration profile {path} is version {version} and this "
            f"build reads version {PROFILE_VERSION}."
        )
    return dict(profile)
