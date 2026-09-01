# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The seeded model suite of Section 15: seven structurally distinct models.

Generated, never downloaded. Every model is built from one seed, so two runs
produce byte identical weights and a failure reproduces; `GENERATOR_VERSION` is
bumped whenever anything here changes what a model contains, because a result
manifest records it and a manifest that cannot distinguish two suites is a
manifest that cannot be trusted.

**Five are exported from PyTorch and two are built with the ONNX construction
API**, and the split is not a matter of taste. Section 15 gives the reason for
each hand built one:

- The **conv, batch norm, relu stack** exists to hold an *unfolded*
  `BatchNormalization`, and on a current PyTorch it cannot be exported. The
  dynamo exporter has been the default since 2.9 and `do_constant_folding` has
  been deprecated and permanently enabled since 2.6, so a `Conv` followed by a
  `BatchNorm` comes back with the batch norm already folded into the
  convolution. Probed on this toolchain on 2026-08-19, that is exactly what
  happens: the export of a two layer conv plus batch norm plus relu module
  contains one `Conv` and one `Relu` and no `BatchNormalization` at all. A model
  whose whole purpose is to give `-npu-fold-batchnorm` something to fold must
  not depend on an exporter's optimizer choosing to leave it alone.
- The **dilated convolution stack** needs asymmetric padding, and `nn.Conv2d`
  takes a symmetric padding argument only. Reaching asymmetric padding through
  `F.pad` exports a `Pad` node, which this importer refuses by name, so the
  model this suite needs is probably not exportable from PyTorch at all.

Neither escape hatch is `dynamo=False`. The legacy exporter is deprecated and
will be removed, and building a phase on a deprecated code path is how this
defect recurs a third time.

**Export settings are per model, never global.** Each spec below carries its own
options dictionary. A global flag change would move LeNet's exported graph,
which moves the golden tensors, which turns a model addition into a baseline
regression.

## What each model forces into the pipeline

| Model | Forces |
|---|---|
| `lenet` | the baseline and the regression anchor |
| `depthwise_separable` | grouped convolution, pointwise convolution, `group == C`, global average pooling |
| `resnet_block` | residual `Add`, identity shortcut, a per channel `Mul` on the residual branch |
| `inception_block` | `Concat`, parallel branches, branching topology |
| `conv_bn_relu_stack` | an unfolded `BatchNormalization`, plus `Identity`, `GlobalAveragePool`, `Flatten` and `MatMul` |
| `dilated_stack` | `dilation > 1`, asymmetric padding, a closing NCHW to NHWC `Transpose`, plus `Clip` and a separate channel shaped bias `Add` |
| `lenet_batched` | the batch path through the whole pipeline, at N = 4 |

The four extra operators on the two hand built models are there because Section
17.1 requires a suite model per importer operator and no model in Section 15's
table reaches `Identity`, `GlobalAveragePool`, `Flatten`, `MatMul` or `Clip`.
Each is placed where it is natural rather than appended: two `Identity` nodes at
the head of a graph is what a graph transform leaves behind, a global pool
followed by a flatten and a matrix multiply is the head of any small classifier,
and a `Clip` with a lower bound of zero and no upper bound is what a normaliser
turns a relu into.

**`dilated_stack` carries the separate bias add, and that is a shape rather than
an operator.** Every convolution the dynamo exporter writes carries its bias
inline as a third `Conv` input, so `add(conv(x, w), b)` appears nowhere in a
suite built only from exported graphs, and `-npu-fuse-bias` had a Section 16.2
ablation row of zeros for want of a target rather than for want of a saving.
`conv1` was already biasless and already followed by a `Relu`, so one `Add`
between them is the smallest change that gives the pass something to do on a
real model. It is not a synthetic node bolted on: a convolution followed by a
separate bias add is what a framework that keeps its bias as a parameter emits,
and it is the exact shape Section 11's broadcast carve out was written for.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import torch
from onnx import ModelProto, TensorProto, helper, numpy_helper
from torch import nn

from .onnx_importer import PINNED_OPSET

# Bumped whenever anything in this module changes what a model contains, so a
# result manifest naming a version identifies one suite and not a family of
# them.
GENERATOR_VERSION: Final[str] = "1.1.0"

# One seed for the whole suite. Every model derives its randomness from it and
# from nothing else, so no model's contents depend on the order the suite was
# generated in.
DEFAULT_SEED: Final[int] = 20260819

# The ONNX IR version the dynamo exporter writes on this toolchain. The hand
# built models are pinned to the same number so that the two halves of the suite
# do not differ in a field nobody meant to vary.
_IR_VERSION: Final[int] = 10


# =============================================================================
# The PyTorch modules.
# =============================================================================


class _LeNet(nn.Module):
    """A LeNet style classifier. The anchor model, and the regression baseline.

    The flatten is `reshape(batch, -1)` rather than `nn.Flatten()`, and the
    difference is visible in the export: `reshape` writes a target shape of
    `[N, -1]` and `nn.Flatten` writes the fully resolved `[N, features]`. Both
    are batch preserving, and taking the one that carries a `-1` means the
    importer's own shape completion is exercised by the anchor model rather than
    only by a unit test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class _DepthwiseSeparable(nn.Module):
    """A depthwise separable block: `group == C`, then a pointwise convolution.

    The closing pool is `nn.AvgPool2d` over the whole spatial extent rather than
    `nn.AdaptiveAvgPool2d((1, 1))`, and that is a finding rather than a
    preference. Probed on this toolchain, the dynamo exporter lowers every
    spelling of adaptive average pooling, including `x.mean(dim=(2, 3))`, to a
    `ReduceMean` node, which is not in this project's operator set. An average
    pool whose kernel is the input's spatial extent computes exactly the same
    thing, exports as `AveragePool`, and imports to the same `npu.avg_pool2d`
    that `GlobalAveragePool` maps to.
    """

    def __init__(self, spatial: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.pointwise = nn.Conv2d(8, 16, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.depthwise(x))
        x = torch.relu(self.pointwise(x))
        return self.pool(x)


class _ResNetBlock(nn.Module):
    """A residual block with an identity shortcut and a per channel scale.

    The scale is what puts an ONNX `Mul` into this suite at all, and it is a
    real structure rather than a bolt on: a scaled residual branch is what
    squeeze and excitation style blocks apply before the addition. It exports as
    one `Mul` node with a channel shaped initializer, which is the second
    operator the broadcasting carve out of Section 11 reaches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.rand(8) + 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = torch.relu(self.conv1(x))
        branch = self.conv2(branch)
        branch = branch * self.scale.reshape(1, -1, 1, 1)
        return torch.relu(x + branch)


class _InceptionBlock(nn.Module):
    """Three parallel branches into one `Concat`. Branching topology."""

    def __init__(self) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(8, 4, kernel_size=1)
        self.branch2 = nn.Conv2d(8, 6, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(8, 2, kernel_size=5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.branch1(x), self.branch2(x), self.branch3(x)]
        return torch.relu(torch.cat(branches, dim=1))


# =============================================================================
# Export.
# =============================================================================


def _export_torch(
    module: nn.Module,
    input_shape: tuple[int, ...],
    path: Path,
    options: Mapping[str, Any],
    seed: int,
) -> Path:
    torch.manual_seed(seed)
    # Re-seeding before the module is constructed is what makes the export
    # deterministic: the parameters are drawn at construction time, so a seed
    # set after it would leave the weights to whatever the global generator
    # happened to hold.
    module = module.eval()
    example = torch.randn(*input_shape, generator=torch.Generator().manual_seed(seed))
    torch.onnx.export(module, (example,), str(path), **dict(options))
    return path


def _seeded_module(factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return factory()


# =============================================================================
# The two models built with the ONNX construction API.
# =============================================================================


def _initializer(name: str, array: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name)


def _finish(graph: onnx.GraphProto, doc: str) -> ModelProto:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", PINNED_OPSET)],
        producer_name="npu-mlir model_generator",
        producer_version=GENERATOR_VERSION,
        doc_string=doc,
    )
    model.ir_version = _IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def with_batch(shape: tuple[int, ...], batch: int) -> tuple[int, ...]:
    """`shape` with its leading extent replaced by `batch`.

    Section 17.4's matrix sweeps the batch size over every model, and Section
    15's registry pins one batch per model. The two are reconciled here rather
    than by a second registry: a model is a structure, its batch is a parameter
    of an export, and the weights do not depend on it.
    """
    if batch <= 0:
        raise ValueError(f"a batch size is a positive integer, got {batch}")
    return (batch, *shape[1:])


def _build_conv_bn_relu_stack(seed: int, batch: int = 1) -> ModelProto:
    """Conv, BatchNorm, Relu twice, then a global pool, a flatten and a matmul.

    The batch norms are here by construction and stay unfolded, which is the
    whole point: `-npu-fold-batchnorm` is a flagship pass and this is the only
    model in the suite that gives it anything to fold.

    The two `Identity` nodes at the head are the shape a graph transform leaves
    behind, and the P3 gate asks specifically that an
    `Identity, Identity, Conv, BatchNormalization` sequence imports past every
    one of them.
    """
    rng = np.random.default_rng(seed)
    channels = 8
    nodes = []
    initializers = []

    nodes.append(helper.make_node("Identity", ["input"], ["ident0"], name="ident0"))
    nodes.append(helper.make_node("Identity", ["ident0"], ["ident1"], name="ident1"))

    current = "ident1"
    in_channels = 3
    for stage in range(2):
        weight = f"conv{stage}.weight"
        bias = f"conv{stage}.bias"
        initializers.append(
            _initializer(
                weight, rng.standard_normal((channels, in_channels, 3, 3)) * 0.2
            )
        )
        initializers.append(_initializer(bias, rng.standard_normal(channels) * 0.1))
        nodes.append(
            helper.make_node(
                "Conv",
                [current, weight, bias],
                [f"conv{stage}"],
                name=f"conv{stage}",
                kernel_shape=[3, 3],
                strides=[1, 1],
                pads=[1, 1, 1, 1],
                dilations=[1, 1],
                group=1,
            )
        )

        scale = f"bn{stage}.scale"
        shift = f"bn{stage}.bias"
        mean = f"bn{stage}.mean"
        variance = f"bn{stage}.var"
        initializers.append(_initializer(scale, rng.random(channels) + 0.5))
        initializers.append(_initializer(shift, rng.standard_normal(channels) * 0.1))
        initializers.append(_initializer(mean, rng.standard_normal(channels) * 0.1))
        # Strictly positive, so that the reciprocal square root the folding pass
        # computes is finite for every channel.
        initializers.append(_initializer(variance, rng.random(channels) + 0.25))
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                [f"conv{stage}", scale, shift, mean, variance],
                [f"bn{stage}"],
                name=f"bn{stage}",
                epsilon=1e-5,
            )
        )
        nodes.append(
            helper.make_node(
                "Relu", [f"bn{stage}"], [f"relu{stage}"], name=f"relu{stage}"
            )
        )
        current = f"relu{stage}"
        in_channels = channels

    nodes.append(
        helper.make_node("GlobalAveragePool", [current], ["pooled"], name="pooled")
    )
    nodes.append(helper.make_node("Flatten", ["pooled"], ["flat"], name="flat", axis=1))
    initializers.append(
        _initializer("head.weight", rng.standard_normal((channels, 4)) * 0.3)
    )
    nodes.append(
        helper.make_node("MatMul", ["flat", "head.weight"], ["output"], name="head")
    )

    graph = helper.make_graph(
        nodes,
        "conv_bn_relu_stack",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, 4])],
        initializer=initializers,
    )
    return _finish(
        graph,
        "Conv plus BatchNorm plus ReLU, built with the ONNX construction API so "
        "the BatchNormalization nodes are present by construction rather than "
        "at an exporter optimizer's discretion.",
    )


def _build_dilated_stack(seed: int, batch: int = 1) -> ModelProto:
    """Dilated convolutions with asymmetric padding, closing on a `Transpose`.

    The arithmetic is written out here rather than left to shape inference,
    because asymmetric padding is exactly where an off by one hides. With
    `effectiveKernel = dilation * (kernel - 1) + 1` and
    `extent = floor((input + padBegin + padEnd - effectiveKernel) / stride) + 1`:

        conv0 H: effective 5, (12 + 1 + 2 - 5) / 1 + 1 = 11
        conv0 W: effective 5, (14 + 2 + 1 - 5) / 1 + 1 = 13
        conv1 H: effective 7, (11 + 0 + 0 - 7) / 2 + 1 = 3
        conv1 W: effective 5, (13 + 1 + 1 - 5) / 2 + 1 = 6

    so the result before the permutation is `(1, 5, 3, 6)` and after
    `perm = [0, 2, 3, 1]` it is `(1, 3, 6, 5)`. The four extents are pairwise
    distinct on purpose: a permutation test on a tensor whose axes happen to be
    equal proves nothing, because every permutation of it has the same shape.

    The `Clip` is a lower bound of zero with no upper bound, which is the only
    form this importer accepts and is what a graph normaliser turns a relu into.

    **`conv1` has no `Conv` bias input and gains its bias from a separate `Add`,
    and that is the whole reason this model is the one that changed.** Section
    12's `-npu-fuse-bias` matches `add(conv(x, w), b)` with a channel shaped
    constant addend, which is the shape Section 11 leaves unexpanded precisely so
    the pass can match it. No exported graph in this suite produces it: the
    exporter writes the bias inline as a third `Conv` input. So the pass fired on
    a model built for it in the test suite and on nothing in Section 15's suite,
    and its Section 16.2 ablation row would have been a row of zeros that read as
    a measurement. One `Add` between `conv1` and the `Relu` fixes that, and the
    two convolutions now differ in more than their dilation: `conv0` carries its
    bias inline and `conv1` carries it separately, so this one model holds both
    spellings.

    **The addend is written as `(1, C, 1, 1)` rather than as `(C,)`**, for two
    reasons that agree. ONNX broadcasting aligns from the trailing axis, so a
    rank 1 initializer of length 5 would try to broadcast against the width of 6
    and `onnx.checker` refuses the graph outright. And `(1, C, 1, 1)` is the
    spelling `p.reshape(1, -1, 1, 1)` produces in an exported graph, which the
    importer normalises to the rank 1 constant `docs/adr/0005` describes. The
    pass therefore sees rank 1 without this file having to write rank 1.
    """
    rng = np.random.default_rng(seed)
    nodes = []
    initializers = []

    initializers.append(
        _initializer("conv0.weight", rng.standard_normal((7, 4, 3, 3)) * 0.2)
    )
    initializers.append(_initializer("conv0.bias", rng.standard_normal(7) * 0.1))
    nodes.append(
        helper.make_node(
            "Conv",
            ["input", "conv0.weight", "conv0.bias"],
            ["conv0"],
            name="conv0",
            kernel_shape=[3, 3],
            strides=[1, 1],
            # ONNX order: padTop, padLeft, padBottom, padRight. Asymmetric on
            # both axes, which is the shape nn.Conv2d cannot express.
            pads=[1, 2, 2, 1],
            dilations=[2, 2],
            group=1,
        )
    )

    initializers.append(_initializer("clip.min", np.array(0.0, dtype=np.float32)))
    nodes.append(
        helper.make_node("Clip", ["conv0", "clip.min"], ["clipped"], name="clipped")
    )

    initializers.append(
        _initializer("conv1.weight", rng.standard_normal((5, 7, 3, 3)) * 0.2)
    )
    nodes.append(
        helper.make_node(
            "Conv",
            ["clipped", "conv1.weight"],
            ["conv1"],
            name="conv1",
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[0, 1, 0, 1],
            dilations=[3, 2],
            group=1,
        )
    )
    # The separate bias add. It is drawn *after* `conv1.weight` deliberately:
    # every other tensor in this model is drawn from the same generator in the
    # same order as before, so appending here leaves conv0 and conv1's weights
    # bit identical and the only thing that moved is what this node adds.
    initializers.append(
        _initializer("conv1.bias", rng.standard_normal((1, 5, 1, 1)) * 0.1)
    )
    nodes.append(
        helper.make_node("Add", ["conv1", "conv1.bias"], ["biased"], name="biased")
    )

    nodes.append(helper.make_node("Relu", ["biased"], ["activated"], name="activated"))
    nodes.append(
        helper.make_node(
            "Transpose", ["activated"], ["output"], name="to_nhwc", perm=[0, 2, 3, 1]
        )
    )

    graph = helper.make_graph(
        nodes,
        "dilated_stack",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 4, 12, 14])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, 3, 6, 5])],
        initializer=initializers,
    )
    return _finish(
        graph,
        "Dilated convolutions with asymmetric padding and a closing NCHW to "
        "NHWC permutation, built with the ONNX construction API because "
        "nn.Conv2d takes a symmetric padding argument only.",
    )


# =============================================================================
# The registry.
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    """One model of the suite, and everything a test needs to check it."""

    name: str
    input_shape: tuple[int, ...]
    summary: str
    #: Exact ONNX node counts by operator type. A structural pytest asserts
    #: this against the generated file, so an exporter change that folds a node
    #: away or introduces one fails loudly instead of quietly moving what the
    #: suite covers.
    expected_nodes: Mapping[str, int]
    #: The tight scratchpad budget in bytes, measured at P8 and frozen.
    #:
    #: No default, deliberately: a model added without one is an error at
    #: import rather than a model quietly given somebody else's number. The
    #: measurement, the two deviations from Section 15's rule and the reason
    #: the fraction it asks for is inoperative until P13 are all in
    #: `docs/adr/0008-per-model-tight-scratchpad-budgets.md`. These numbers are
    #: frozen: a moved tight budget moves every tight budget cell in the
    #: project's history at once.
    tight_budget: int
    torch_factory: Callable[[], nn.Module] | None = None
    onnx_builder: Callable[[int, int], ModelProto] | None = None
    #: Export options, per model. Never shared, never global: a flag change
    #: applied to the whole suite would move the anchor model's graph, which
    #: moves the golden tensors, which turns a model addition into a baseline
    #: regression.
    export_options: Mapping[str, Any] = field(default_factory=dict)


def _torch_options(**overrides: Any) -> dict[str, Any]:
    """One model's export options. Returns a fresh dictionary every call.

    `dynamo=True` is passed explicitly rather than relied on as the default. It
    has been the default since torch 2.9, and writing it down is what makes the
    Section 15 rule that `dynamo=False` is never used visible in the source
    rather than only in a document.
    """
    options: dict[str, Any] = {
        "opset_version": PINNED_OPSET,
        "dynamo": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "verbose": False,
    }
    options.update(overrides)
    return options


MODELS: Final[dict[str, ModelSpec]] = {
    "lenet": ModelSpec(
        name="lenet",
        tight_budget=194624,
        input_shape=(1, 1, 28, 28),
        summary="LeNet style CNN. The baseline and the regression anchor.",
        expected_nodes={"Conv": 2, "Relu": 4, "MaxPool": 2, "Reshape": 1, "Gemm": 3},
        torch_factory=_LeNet,
        export_options=_torch_options(),
    ),
    "depthwise_separable": ModelSpec(
        name="depthwise_separable",
        tight_budget=8192,
        input_shape=(1, 8, 8, 8),
        summary=(
            "Depthwise separable block. Grouped convolution at group == C, a "
            "pointwise convolution, and global average pooling."
        ),
        expected_nodes={"Conv": 2, "Relu": 2, "AveragePool": 1},
        torch_factory=lambda: _DepthwiseSeparable(spatial=8),
        export_options=_torch_options(),
    ),
    "resnet_block": ModelSpec(
        name="resnet_block",
        tight_budget=6464,
        input_shape=(1, 8, 8, 8),
        summary=(
            "Small ResNet block. A residual Add over an identity shortcut, with "
            "a per channel Mul on the residual branch."
        ),
        expected_nodes={"Conv": 2, "Relu": 2, "Mul": 1, "Add": 1},
        torch_factory=_ResNetBlock,
        export_options=_torch_options(),
    ),
    "inception_block": ModelSpec(
        name="inception_block",
        tight_budget=6144,
        input_shape=(1, 8, 8, 8),
        summary="Small Inception block. Concat over three parallel branches.",
        expected_nodes={"Conv": 3, "Concat": 1, "Relu": 1},
        torch_factory=_InceptionBlock,
        export_options=_torch_options(),
    ),
    "conv_bn_relu_stack": ModelSpec(
        name="conv_bn_relu_stack",
        tight_budget=6464,
        input_shape=(1, 3, 8, 8),
        summary=(
            "Conv plus BatchNorm plus ReLU stack, built with the ONNX "
            "construction API so the BatchNormalization nodes survive."
        ),
        expected_nodes={
            "Identity": 2,
            "Conv": 2,
            "BatchNormalization": 2,
            "Relu": 2,
            "GlobalAveragePool": 1,
            "Flatten": 1,
            "MatMul": 1,
        },
        onnx_builder=_build_conv_bn_relu_stack,
    ),
    "dilated_stack": ModelSpec(
        name="dilated_stack",
        tight_budget=8064,
        input_shape=(1, 4, 12, 14),
        summary=(
            "Dilated convolution stack with asymmetric padding, a separate "
            "channel shaped bias add on the second convolution, and a closing "
            "NCHW to NHWC Transpose, built with the ONNX construction API."
        ),
        expected_nodes={"Conv": 2, "Add": 1, "Clip": 1, "Relu": 1, "Transpose": 1},
        onnx_builder=_build_dilated_stack,
    ),
    "lenet_batched": ModelSpec(
        name="lenet_batched",
        tight_budget=200832,
        input_shape=(4, 1, 28, 28),
        summary="LeNet at N = 4. The batch path through the whole pipeline.",
        expected_nodes={"Conv": 2, "Relu": 4, "MaxPool": 2, "Reshape": 1, "Gemm": 3},
        torch_factory=_LeNet,
        export_options=_torch_options(),
    ),
}

#: The input shape of every model, as one map. Section 15 requires the benchmark
#: harness's own `INPUT_SHAPES` to be asserted equal to the model registry by a
#: test; deriving it from the registry rather than writing it twice is the
#: stronger form of that, since the two cannot drift when there is only one.
INPUT_SHAPES: Final[dict[str, tuple[int, ...]]] = {
    name: spec.input_shape for name, spec in MODELS.items()
}

#: The tight scratchpad budget of every model, as one map.
#:
#: Measured at P8 and frozen. `docs/adr/0008-per-model-tight-scratchpad-budgets.md`
#: has the measurement, the two deviations from Section 15's rule, and the
#: reason the fraction that section asks for is inoperative until tiling lands
#: at P13. Derived from the registry rather than written a second time, for the
#: same reason `INPUT_SHAPES` is.
TIGHT_BUDGETS: Final[dict[str, int]] = {
    name: spec.tight_budget for name, spec in MODELS.items()
}

#: The budget the allocator uses when nobody names one, which is what the
#: default budget cells of every matrix in this project mean.
DEFAULT_BUDGET: Final[int] = 1048576


def generate_model(
    name: str,
    directory: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    batch: int | None = None,
) -> Path:
    """Build one model of the suite and write it as ONNX. Returns the path.

    `.onnx` files are never committed. They are regenerated from this seed,
    which is the arrangement Section 22 asks for and the reason `*.onnx` is in
    `.gitignore`.

    `batch` overrides the registry's batch size, which is what Section 17.4's
    matrix needs: it sweeps the batch over every model where Section 15 pins one
    per model. The weights do not depend on it, because every model draws them
    from `seed` at construction time and the batch only reaches the export's
    example input. A batch equal to the registry's writes the registry's own
    file rather than a second copy of it, so a caller sweeping `[1, 4]` over a
    directory exports eleven files and not fourteen.
    """
    spec = MODELS.get(name)
    if spec is None:
        raise KeyError(f"{name!r} is not in the model suite: {sorted(MODELS)}")

    effective = spec.input_shape[0] if batch is None else batch
    shape = with_batch(spec.input_shape, effective)

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    stem = name if effective == spec.input_shape[0] else f"{name}-n{effective}"
    path = target / f"{stem}.onnx"

    if spec.onnx_builder is not None:
        onnx.save(spec.onnx_builder(seed, effective), str(path))
        return path

    assert spec.torch_factory is not None
    module = _seeded_module(spec.torch_factory, seed)
    _export_torch(module, shape, path, spec.export_options, seed)
    return path


def generate_all(
    directory: str | os.PathLike[str],
    *,
    seed: int = DEFAULT_SEED,
    batch: int | None = None,
) -> dict[str, Path]:
    """Build every model of the suite into one directory."""
    return {
        name: generate_model(name, directory, seed=seed, batch=batch) for name in MODELS
    }


def node_counts(model: ModelProto) -> dict[str, int]:
    """How many nodes of each operator type a model has."""
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def first_weight(model: ModelProto) -> np.ndarray:
    """The first rank 4 float initializer, which is the first layer's filter.

    Used by the determinism test. It reads the first convolution's weights
    rather than a hash of the whole file because a file hash also covers the
    producer version and the node names, and a change to either of those is not
    a change to what the model computes.
    """
    for initializer in model.graph.initializer:
        if len(initializer.dims) == 4:
            return numpy_helper.to_array(initializer)
    raise ValueError(f"{model.graph.name} has no rank 4 initializer")
