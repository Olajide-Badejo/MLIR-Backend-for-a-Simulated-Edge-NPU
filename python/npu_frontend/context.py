# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The state one graph's conversion carries: names, shapes, and initializers.

Split out of `onnx_importer.py` so that `op_mapping.py` can be typed against it
without the two importing each other. It holds three maps and no policy:

- what SSA value an ONNX tensor name currently resolves to,
- what shape ONNX inferred for each name,
- what constant data each initializer holds, as numpy.

The one piece of behaviour here is that an initializer becomes an `npu.constant`
the first time something asks for it as a value, and never twice. Materialising
eagerly would emit a constant for every weight in the model whether or not the
graph uses it, and materialising twice would give the same weight two SSA values
and double what the allocator later sees.
"""

from __future__ import annotations

import numpy as np
from mlir import ir
from onnx import NodeProto

from .builder import ModuleBuilder
from .diagnostics import graph_error, node_error


class ConversionContext:
    """Everything a converter needs to look up, and nothing it needs to decide."""

    def __init__(
        self,
        builder: ModuleBuilder,
        shapes: dict[str, tuple[int, ...]],
        initializers: dict[str, np.ndarray],
    ) -> None:
        self.builder = builder
        self._shapes = shapes
        self._initializers = initializers
        self._values: dict[str, ir.Value] = {}
        self._materialised: dict[str, ir.Value] = {}

    # -- Names and values ----------------------------------------------------

    def bind(self, name: str, value: ir.Value) -> None:
        """Record that the ONNX tensor `name` is now this SSA value.

        Called by every converter for its output, and by the `Identity`
        converter for its input, which is how `Identity` emits no operation at
        all: it binds its output name to the value its input already had.
        """
        self._values[name] = value

    def value(self, node: NodeProto, name: str) -> ir.Value:
        """The SSA value for an ONNX tensor name, materialising a constant."""
        existing = self._values.get(name)
        if existing is not None:
            return existing

        materialised = self._materialised.get(name)
        if materialised is not None:
            return materialised

        array = self._initializers.get(name)
        if array is None:
            raise node_error(
                node,
                f"input {name!r} is neither a graph input, nor an initializer, "
                "nor the output of an earlier node. A graph whose nodes are not "
                "in topological order is refused rather than reordered, because "
                "reordering one is how an importer starts disagreeing with the "
                "runtime it is checked against.",
            )
        if array.dtype != np.float32:
            raise node_error(
                node,
                f"initializer {name!r} has dtype {array.dtype}, and this "
                "project's tensors are f32. The integer types arrive with the "
                "quantization phase and its operations.",
            )
        constant = self.builder.constant(array, name)
        self._materialised[name] = constant
        return constant

    def is_bound(self, name: str) -> bool:
        return name in self._values

    # -- Shapes and constants ------------------------------------------------

    def shape(self, node: NodeProto, name: str) -> tuple[int, ...]:
        shape = self._shapes.get(name)
        if shape is None:
            raise node_error(
                node,
                f"no shape was inferred for {name!r}. Every tensor in this "
                "project has a static shape, and a name ONNX shape inference "
                "could not resolve is one this importer cannot give a type.",
            )
        return shape

    def expect_shape(
        self, node: NodeProto, name: str, computed: tuple[int, ...]
    ) -> None:
        """Cross check a shape this importer computed against ONNX's inference.

        Used by the shape only converters, which are the ones where this
        importer does its own arithmetic rather than reading the answer out of
        the inferred value info. The two are independent implementations of the
        same rule, so a disagreement means one of them is wrong and the useful
        thing to do is say so loudly here rather than emit whichever was asked
        for last.

        A name ONNX inference did not resolve is not an error in this direction.
        Inference is allowed to give up on a graph the converter can still
        handle, and refusing that would be treating silence as disagreement.
        """
        inferred = self._shapes.get(name)
        if inferred is None:
            self._shapes[name] = computed
            return
        if inferred != computed:
            raise node_error(
                node,
                f"this importer computes the shape of {name!r} as "
                f"{list(computed)} and ONNX shape inference says "
                f"{list(inferred)}. Those are two independent implementations "
                "of the same rule and they disagree, so one of them is wrong.",
            )

    def graph_shape(self, name: str) -> tuple[int, ...]:
        shape = self._shapes.get(name)
        if shape is None:
            raise graph_error(f"no shape was inferred for the graph tensor {name!r}")
        return shape

    def initializer(self, name: str) -> np.ndarray | None:
        """The constant data behind a name, or None when it is not constant."""
        return self._initializers.get(name)

    def require_initializer(self, node: NodeProto, name: str, role: str) -> np.ndarray:
        array = self._initializers.get(name)
        if array is None:
            raise node_error(
                node,
                f"the {role} {name!r} must be a constant initializer, and in "
                "this graph it is computed at run time. The earliest layer that "
                "can name the problem is the one that should, so it is refused "
                "here rather than left for the lowering to fail on.",
            )
        return array
