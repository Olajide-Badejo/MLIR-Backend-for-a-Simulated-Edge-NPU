# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The one exception this package raises, and the shape of its message.

Law 1 of this project is that there are no silent wrong answers, and the
frontend is where that law is mostly enforced by refusing things. Section 11
says the same in its own words: anything the importer does not support fails
loudly with the operation name and location, never as silent wrong IR.

A refusal is only useful if it says which node it is about. An importer that
reports "unsupported attribute" against a model with forty convolutions has
told the reader nothing they can act on, so every raise in this package goes
through one of the constructors below and every message therefore names the
node, its ONNX operator type, and the construct that was refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnx import NodeProto


class ONNXImportError(Exception):
    """An ONNX graph this importer refuses, or cannot represent.

    Raised for everything the frontend rejects: an unsupported operator, an
    attribute value outside what this project implements, a broadcast that is
    not one of the permitted shapes, a dtype the dialect has no type for. It is
    deliberately one class rather than a hierarchy, because every one of them
    has the same remedy from a caller's point of view, which is to look at the
    named node.

    It is **not** raised for arithmetic the `npu` dialect verifies itself. An
    impossible convolution extent is diagnosed by the dialect verifier through
    `npu-opt`, in the one place Section 7.2's arithmetic lives, and arrives here
    as a `VerificationError`.
    """


class VerificationError(Exception):
    """`npu-opt` rejected the module the importer built, or could not be run.

    Separate from `ONNXImportError` because it means something different. An
    `ONNXImportError` is a model this project does not accept and is usually the
    model's problem. A `VerificationError` is IR this package emitted that the
    dialect refused, which is this package's problem: either the converter built
    the wrong thing, or the shape arithmetic it copied out of ONNX's inference
    disagrees with the arithmetic the dialect computes for itself.

    The disagreement case is the valuable one and is why the round trip is not
    optional. ONNX shape inference and `NPUShapeUtils.cpp` are two independent
    implementations of the same pooling and convolution formulas, and a model
    where they differ is a model where one of them is wrong.
    """


def node_error(node: NodeProto, message: str) -> ONNXImportError:
    """An `ONNXImportError` naming a node, for a converter to raise.

    The name comes first because that is what a reader greps the ONNX graph for.
    An unnamed node still gets an identifier: `import_model` names every node
    before conversion starts, so the empty string never reaches here.
    """
    return ONNXImportError(f"{node.op_type} node {node.name!r}: {message}")


def graph_error(message: str) -> ONNXImportError:
    """An `ONNXImportError` about the graph rather than about one node.

    The opset, an input with a dynamic dimension, an initializer of a dtype this
    project has no type for. There is no node to name, so these say what they
    are about in the message instead.
    """
    return ONNXImportError(message)
