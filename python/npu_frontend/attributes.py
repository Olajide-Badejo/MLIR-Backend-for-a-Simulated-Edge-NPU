# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Reading ONNX node attributes by their declared type.

Section 11 is specific about this and the reason is a real bug rather than a
style preference:

> **Attributes are read by declared type.** Switch on `AttributeProto.type`,
> covering INT, INTS, FLOAT, FLOATS, STRING, and TENSOR, and return the default
> only when the attribute is genuinely absent. Testing truthiness cannot
> distinguish an absent attribute from a legitimately empty list, and an unknown
> type raises rather than silently returning a default.

The empty list is the case that bites. `Conv` with `pads = []` is a legal thing
for a graph transform to leave behind, and the natural spelling

    pads = node_attr(node, "pads") or [0, 0, 0, 0]

reads an explicitly empty `pads` as an absent one. Here that distinction is
mechanical: the attribute is either in `node.attribute` or it is not, and the
default is returned only in the second case.

An attribute present with a type the reader did not ask for is an error, not a
coercion. An `INTS` where an `INT` was expected means the graph is not the shape
this converter was written against, and quietly reading `attr.i` off it returns
zero.

There is an accessor for each of the six types, including the two no converter
currently reads, because the switch Section 11 specifies is over six types and a
partial switch would raise a `KeyError` rather than a diagnostic on the day a
converter asked for the seventh. Both are covered by tests in
`test_onnx_importer.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from onnx import AttributeProto

from .diagnostics import node_error

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnx import NodeProto, TensorProto

# The six types Section 11 names, mapped to the AttributeProto field that
# carries the value and to a readable name for a diagnostic. Every other type
# raises: GRAPH, SPARSE_TENSOR, TYPE_PROTO and the plural forms of all of them
# belong to operators this project does not support, so meeting one means the
# converter is reading a node it was not written for.
_FIELD: Final[dict[int, str]] = {
    AttributeProto.INT: "i",
    AttributeProto.INTS: "ints",
    AttributeProto.FLOAT: "f",
    AttributeProto.FLOATS: "floats",
    AttributeProto.STRING: "s",
    AttributeProto.TENSOR: "t",
}

_TYPE_NAME: Final[dict[int, str]] = {
    AttributeProto.INT: "INT",
    AttributeProto.INTS: "INTS",
    AttributeProto.FLOAT: "FLOAT",
    AttributeProto.FLOATS: "FLOATS",
    AttributeProto.STRING: "STRING",
    AttributeProto.TENSOR: "TENSOR",
}


def _describe_type(kind: int) -> str:
    """A readable name for an attribute type, including ones we refuse."""
    named = _TYPE_NAME.get(kind)
    if named is not None:
        return named
    try:
        return AttributeProto.AttributeType.Name(kind)
    except ValueError:
        return f"<unknown attribute type {kind}>"


def _lookup(node: NodeProto, name: str, expected: int) -> Any:
    """The raw value of one attribute, or None when it is genuinely absent.

    Returns `Any` because it returns whichever of six protobuf fields the
    declared type names, and onnx's generated classes carry no usable type
    information anyway. This is the boundary where `Any` is the honest
    annotation; the typed accessors below are what callers see.
    """
    for attribute in node.attribute:
        if attribute.name != name:
            continue
        if attribute.type != expected:
            raise node_error(
                node,
                f"attribute {name!r} is declared {_describe_type(attribute.type)} "
                f"but this converter reads it as {_describe_type(expected)}. "
                "Attributes are read by declared type rather than coerced, "
                "because a coercion here returns a plausible wrong value "
                "instead of an error.",
            )
        # No second branch for a type outside the six. The equality check above
        # already covers it: a GRAPH attribute asked for as INTS is refused as
        # "declared GRAPH", which names the actual problem, and a caller cannot
        # ask for GRAPH because there is no accessor that passes it.
        return getattr(attribute, _FIELD[attribute.type])
    return None


def get_int(node: NodeProto, name: str, default: int) -> int:
    value = _lookup(node, name, AttributeProto.INT)
    return default if value is None else int(value)


def require_int(node: NodeProto, name: str) -> int:
    value = _lookup(node, name, AttributeProto.INT)
    if value is None:
        raise node_error(node, f"attribute {name!r} is required and is absent")
    return int(value)


def get_ints(node: NodeProto, name: str, default: list[int]) -> list[int]:
    """An INTS attribute, or `default` when it is absent.

    An attribute present and empty returns the empty list, which is the whole
    point of this module. Callers check the length themselves, so an empty
    `pads` is refused by the length check with a message about `pads` rather
    than silently becoming four zeros.
    """
    value = _lookup(node, name, AttributeProto.INTS)
    return list(default) if value is None else [int(item) for item in value]


def require_ints(node: NodeProto, name: str) -> list[int]:
    value = _lookup(node, name, AttributeProto.INTS)
    if value is None:
        raise node_error(node, f"attribute {name!r} is required and is absent")
    return [int(item) for item in value]


def get_float(node: NodeProto, name: str, default: float) -> float:
    value = _lookup(node, name, AttributeProto.FLOAT)
    return default if value is None else float(value)


def get_floats(node: NodeProto, name: str, default: list[float]) -> list[float]:
    value = _lookup(node, name, AttributeProto.FLOATS)
    return list(default) if value is None else [float(item) for item in value]


def get_string(node: NodeProto, name: str, default: str) -> str:
    """A STRING attribute, decoded as UTF-8.

    ONNX stores strings as bytes. Decoding here rather than at every call site
    means a comparison against a `str` literal cannot silently be false because
    one side was `b"NOTSET"`, which is the failure that makes an `auto_pad`
    check pass on every model including the ones it should refuse.
    """
    value = _lookup(node, name, AttributeProto.STRING)
    if value is None:
        return default
    return bytes(value).decode("utf-8")


def get_tensor(node: NodeProto, name: str) -> TensorProto | None:
    value = _lookup(node, name, AttributeProto.TENSOR)
    if value is None:
        return None
    return value


def has_attribute(node: NodeProto, name: str) -> bool:
    """Whether the attribute is present at all, whatever its type or value."""
    return any(attribute.name == name for attribute in node.attribute)
