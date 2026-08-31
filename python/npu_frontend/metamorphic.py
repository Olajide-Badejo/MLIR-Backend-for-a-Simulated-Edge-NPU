# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Semantics preserving rewrites of the *input model*, per Section 17.3a.

A metamorphic relation rewrites the model rather than the pipeline, compiles the
original and the variant at the same level, and asserts they agree. It is an
oracle that needs no external runtime, so it reaches operator combinations no
model in Section 15's suite contains: the technique found over 400 erroneous
compilation inputs across four industrial deep learning compilers.

**The rewrites act on ONNX, before the importer.** That is what makes them
independent of everything this project does. A rewrite applied to the `npu` IR
would be tested against the same importer that produced it, and the two would
agree about a misreading.

**Four of Section 17.3a's five are here. The fifth cannot be written.** Pad then
slice back needs an ONNX `Pad`, which Section 11 refuses by name and
`op_mapping.DEFERRED` explains, and an ONNX `Slice`, which has no converter at
all. A relation whose variant the importer refuses is not a relation this
project can check, and inventing a converter for two operators so that a test
could use them would be adding to the operator set to satisfy a test, which
Section 0.2's law 2 exists to prevent. It is recorded in `NOT_IMPLEMENTED`
below with that reason rather than quietly dropped, and it stays unavailable for
as long as Section 11's operator set does.

**Dead subgraph injection is here too**, and it is honestly labelled the way
Section 17.3a labels it: a straight line tensor graph has no unexecuted control
flow to prune, so what is injected is a subgraph whose results feed nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import onnx
from onnx import GraphProto, ModelProto, NodeProto, helper, numpy_helper

__all__ = [
    "DEAD_NODE_COUNT",
    "NOT_IMPLEMENTED",
    "RELATIONS",
    "NotApplicable",
    "Relation",
    "applicable_relations",
    "inject_dead_subgraph",
    "relation",
]


class NotApplicable(Exception):
    """This model has no place for this rewrite."""


@dataclass(frozen=True)
class Relation:
    """One semantics preserving rewrite of an input model."""

    name: str
    summary: str
    #: Rewrites the model, or raises `NotApplicable`.
    rewrite: Callable[[ModelProto], ModelProto]
    #: Whether the variant is expected to produce byte identical `npu` IR as
    #: well as an equal answer. True only where the rewrite is one the importer
    #: erases, which is a stronger claim than agreement and is worth asserting
    #: where it holds.
    erased_at_import: bool = False


#: Section 17.3a's fifth relation, and why it is not here.
NOT_IMPLEMENTED: Final[dict[str, str]] = {
    "pad_then_slice": (
        "needs an ONNX Pad and an ONNX Slice. Pad is refused by name by this "
        "importer, for the reason op_mapping.DEFERRED gives: its opset 18 "
        "optional axes input changes the length of pads and opset 19 added a "
        "wrap mode, so a converter that guessed would misread the padding "
        "rather than fail. Slice has no converter at all. Adding either so "
        "that a test could use it would be growing the operator set to satisfy "
        "a test, which is what law 2 exists to prevent. This relation returns "
        "when Section 11's operator set does"
    ),
}


# ---------------------------------------------------------------------------
# Helpers over the graph.
# ---------------------------------------------------------------------------


def _initializers(graph: GraphProto) -> dict[str, np.ndarray]:
    return {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}


def _shapes(model: ModelProto) -> dict[str, tuple[int, ...]]:
    """Every value's inferred shape. The rewrites need them to place a node."""
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes: dict[str, tuple[int, ...]] = {}
    for info in (
        *inferred.graph.input,
        *inferred.graph.output,
        *inferred.graph.value_info,
    ):
        tensor_type = info.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            continue
        if any(not dim.HasField("dim_value") for dim in tensor_type.shape.dim):
            continue
        shapes[info.name] = tuple(int(dim.dim_value) for dim in tensor_type.shape.dim)
    return shapes


def _finish(model: ModelProto) -> ModelProto:
    """Check the variant before handing it back.

    A rewrite that produced a graph `onnx.checker` refuses would surface later
    as an import failure, and the reader would be told about the importer
    rather than about the rewrite that broke the graph.
    """
    onnx.checker.check_model(model, full_check=True)
    return model


# ---------------------------------------------------------------------------
# The relations.
# ---------------------------------------------------------------------------


def insert_identities(model: ModelProto) -> ModelProto:
    """An `Identity` on every intermediate edge.

    The shape a graph transform leaves behind. It is erased at import, because
    `convert_identity` binds its output to its input rather than emitting an
    operation, so the variant's `npu` IR is byte identical to the original's.
    That is a stronger claim than agreement and the test asserts it as one.
    """
    variant = copy.deepcopy(model)
    graph = variant.graph
    outputs = {info.name for info in graph.output}

    rewritten: list[NodeProto] = []
    for node in graph.node:
        rewritten.append(node)
        for position, name in enumerate(node.output):
            if name in outputs:
                continue
            # The producer writes a new name and the Identity writes the old
            # one, so every consumer downstream keeps the operand it already
            # names and nothing else has to be rewritten.
            staged = f"{name}__pre_identity"
            node.output[position] = staged
            rewritten.append(
                helper.make_node("Identity", [staged], [name], name=f"{name}__identity")
            )

    del graph.node[:]
    graph.node.extend(rewritten)
    if len(graph.node) == len(model.graph.node):
        raise NotApplicable("this graph has no intermediate edge to interpose on")
    # The staged names are new values with no value_info, which is legal:
    # shape inference resolves them and the importer runs shape inference.
    del graph.value_info[:]
    return _finish(variant)


def insert_transpose_and_its_inverse(model: ModelProto) -> ModelProto:
    """A rank 4 permutation followed by the permutation that undoes it.

    Placed on the first rank 4 intermediate edge, which is where a layout pass
    would later want to act. It is *not* erased at import: two real
    `npu.transpose` operations appear, they move every element twice, and the
    answer must come back unchanged.
    """
    variant = copy.deepcopy(model)
    graph = variant.graph
    shapes = _shapes(model)
    outputs = {info.name for info in graph.output}

    forward = [0, 2, 3, 1]
    backward = [0, 3, 1, 2]

    for index, node in enumerate(graph.node):
        for position, name in enumerate(node.output):
            if name in outputs or len(shapes.get(name, ())) != 4:
                continue
            staged = f"{name}__pre_transpose"
            middle = f"{name}__nhwc"
            node.output[position] = staged
            graph.node.insert(
                index + 1,
                helper.make_node(
                    "Transpose",
                    [staged],
                    [middle],
                    name=f"{name}__to_nhwc",
                    perm=forward,
                ),
            )
            graph.node.insert(
                index + 2,
                helper.make_node(
                    "Transpose",
                    [middle],
                    [name],
                    name=f"{name}__to_nchw",
                    perm=backward,
                ),
            )
            del graph.value_info[:]
            return _finish(variant)

    raise NotApplicable("this graph has no rank 4 intermediate edge")


def split_convolution_into_channel_groups(model: ModelProto) -> ModelProto:
    """One convolution becomes two over halves of its output channels, plus a
    `Concat`.

    Section 17.3a names this relation as splitting a convolution into channel
    groups and concatenating. The split is over the **output** channels and the
    weight halves are sliced here, in numpy, at rewrite time. That matters: an
    ONNX `Slice` node would be needed to split at run time and this importer has
    no converter for one, so slicing the initializer is not a shortcut, it is
    the only form of this relation the operator set admits.

    Every output element of the variant is the same sum of the same products in
    the same order as in the original, so this one is exact rather than close.
    """
    variant = copy.deepcopy(model)
    graph = variant.graph
    weights = _initializers(graph)

    for index, node in enumerate(graph.node):
        if node.op_type != "Conv":
            continue
        group = next(
            (attribute.i for attribute in node.attribute if attribute.name == "group"),
            1,
        )
        if group != 1 or node.input[1] not in weights:
            continue
        filters = weights[node.input[1]]
        if filters.shape[0] < 2:
            continue
        cut = filters.shape[0] // 2

        bias = weights.get(node.input[2]) if len(node.input) > 2 else None
        halves = []
        for half, (start, stop) in enumerate(((0, cut), (cut, filters.shape[0]))):
            weight_name = f"{node.name}__split{half}.weight"
            graph.initializer.append(
                numpy_helper.from_array(
                    np.ascontiguousarray(filters[start:stop], dtype=np.float32),
                    weight_name,
                )
            )
            inputs = [node.input[0], weight_name]
            if bias is not None:
                bias_name = f"{node.name}__split{half}.bias"
                graph.initializer.append(
                    numpy_helper.from_array(
                        np.ascontiguousarray(bias[start:stop], dtype=np.float32),
                        bias_name,
                    )
                )
                inputs.append(bias_name)

            piece = helper.make_node(
                "Conv",
                inputs,
                [f"{node.name}__split{half}"],
                name=f"{node.name}__split{half}",
            )
            piece.attribute.extend(node.attribute)
            halves.append(piece)

        joined = helper.make_node(
            "Concat",
            [f"{node.name}__split0", f"{node.name}__split1"],
            [node.output[0]],
            name=f"{node.name}__rejoin",
            axis=1,
        )

        graph.node.remove(node)
        for offset, replacement in enumerate((*halves, joined)):
            graph.node.insert(index + offset, replacement)
        del graph.value_info[:]
        return _finish(variant)

    raise NotApplicable(
        "this graph has no ungrouped convolution with a constant filter of two "
        "or more output channels"
    )


def permute_independent_nodes(model: ModelProto) -> ModelProto:
    """A different, equally valid topological order for the same graph.

    Section 17.3a's "permuting independent parallel branches". The graph is
    unchanged as a graph; what changes is the order the importer walks it in,
    which is the order the allocator sees live ranges in and the order the
    encoder emits instructions in. A compiler whose answer depends on it has a
    defect that no single order can show.

    The rewrite finds the last adjacent pair with no dependency between them and
    swaps it. The last rather than the first, because a swap near the end moves
    live ranges that are still open across more of the program than one at the
    head does.

    **It applies to branching topologies only, and that is the relation rather
    than a limitation.** A straight line graph has exactly one topological
    order, so there is nothing to permute; the relation raises `NotApplicable`
    and the test records which models it reached. In Section 15's suite that is
    the Inception block, which is the model that exists to have parallel
    branches.
    """
    variant = copy.deepcopy(model)
    graph = variant.graph
    nodes = list(graph.node)

    for index in range(len(nodes) - 2, -1, -1):
        first, second = nodes[index], nodes[index + 1]
        if set(first.output) & set(second.input):
            continue
        if set(second.output) & set(first.input):
            continue
        nodes[index], nodes[index + 1] = second, first
        # A protobuf repeated field does not support item assignment, so the
        # reordered list is written back wholesale.
        reordered = [copy.deepcopy(node) for node in nodes]
        del graph.node[:]
        graph.node.extend(reordered)
        del graph.value_info[:]
        return _finish(variant)

    raise NotApplicable("every adjacent pair of nodes in this graph is dependent")


RELATIONS: Final[tuple[Relation, ...]] = (
    Relation(
        name="identity_insertion",
        summary="an Identity on every intermediate edge",
        rewrite=insert_identities,
        erased_at_import=True,
    ),
    Relation(
        name="transpose_and_inverse",
        summary="a rank 4 permutation followed by the one that undoes it",
        rewrite=insert_transpose_and_its_inverse,
    ),
    Relation(
        name="convolution_split",
        summary="one convolution as two over halves of its output channels, "
        "rejoined by a Concat",
        rewrite=split_convolution_into_channel_groups,
    ),
    Relation(
        name="node_order_permutation",
        summary="a different valid topological order for the same graph",
        rewrite=permute_independent_nodes,
        erased_at_import=False,
    ),
)


def relation(name: str) -> Relation:
    for entry in RELATIONS:
        if entry.name == name:
            return entry
    raise KeyError(
        f"{name!r} is not a relation. There are "
        + ", ".join(entry.name for entry in RELATIONS)
        + "."
    )


def applicable_relations(model: ModelProto) -> list[str]:
    """Which relations can act on this model. Used to report coverage."""
    names = []
    for entry in RELATIONS:
        try:
            entry.rewrite(model)
        except NotApplicable:
            continue
        names.append(entry.name)
    return names


# ---------------------------------------------------------------------------
# Dead subgraph injection.
# ---------------------------------------------------------------------------

#: How many nodes `inject_dead_subgraph` adds. A constant rather than a
#: measurement, because the test asserts the instruction count moved by exactly
#: this much at a level with no dead code elimination, and a number derived from
#: the thing under test would assert nothing.
DEAD_NODE_COUNT: Final[int] = 3


def inject_dead_subgraph(model: ModelProto) -> ModelProto:
    """Adds a subgraph whose results feed nothing.

    Section 17.3a labels this honestly and so does this docstring: a straight
    line tensor graph has no unexecuted control flow to prune, so equivalence
    modulo inputs cannot be reproduced literally. What is injected instead is
    real computation whose result is never read.

    It reads the graph's first input, which is a value that certainly exists,
    and computes three operations over it. Nothing consumes the last, and the
    graph's own outputs are untouched. Two things must then be true, and they
    are different claims:

    - the outputs are **bit identical**, because nothing the live graph computes
      depends on any of this;
    - at a level whose pipeline eliminates dead code, the instruction count is
      unchanged. At `-O0` there is no such pass and the count grows by exactly
      the three instructions this brought, which is the P8 form of the same
      check.

    It also stresses the allocator with interval sets no hand written test
    produces, which Section 17.3a names as the second reason to do it.
    """
    variant = copy.deepcopy(model)
    graph = variant.graph
    initializers = {tensor.name for tensor in graph.initializer}
    arguments = [info.name for info in graph.input if info.name not in initializers]
    if not arguments:
        raise NotApplicable("this graph has no argument to compute from")

    source = arguments[0]
    graph.node.extend(
        [
            helper.make_node("Relu", [source], ["__dead0"], name="__dead_relu"),
            helper.make_node(
                "Mul", ["__dead0", "__dead0"], ["__dead1"], name="__dead_mul"
            ),
            helper.make_node(
                "Add", ["__dead1", "__dead0"], ["__dead2"], name="__dead_add"
            ),
        ]
    )
    del graph.value_info[:]
    return _finish(variant)
