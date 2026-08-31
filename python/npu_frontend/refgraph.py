# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Executes a whole `npu` module with `refexec`, one operation at a time.

`refexec` is the independent reference interpreter of Section 17.3a and it
executes one operation. This file is what turns it into an oracle for a **model**:
it walks the `npu` dialect IR the importer produced and calls `refexec` for each
operation in order.

**Why this exists beside the onnxruntime comparison rather than instead of it.**
The two answer different questions and a phase that had only one of them would
be guessing which. onnxruntime runs the *ONNX graph*, so a disagreement with it
implicates the whole chain, importer included, and that is what Section 17.4's
matrix is for. `refexec` over the *imported IR* runs what the importer actually
produced, so a disagreement with it implicates only what is below the importer:
lowering, allocation, encoding, and the kernels. Agreeing with `refexec` and
disagreeing with onnxruntime localises a fault to the importer in one step, and
that is a diagnosis rather than a search.

**It reads the generic form, and the reason is a real constraint.** The `npu`
dialect is C++ only. The `mlir_core` package the LLVM build installs carries the
upstream dialects and nothing of this project, so a context here cannot parse
`npu.conv2d`'s custom assembly. The generic form it can parse, with
`allow_unregistered_dialects` set, which is the same arrangement `builder.py`
uses to *write* this IR. So this file asks `npu-opt` for the generic form and
walks that.

**The attributes come back typed, not scraped.** MLIR stores an operation's
inherent attributes as properties, and iterating an unregistered operation's
attribute map yields nothing, which is a trap worth naming because it looks like
an operation with no attributes rather than like a lookup that has to be done by
name. Looking one up by name goes through `Operation::getInherentAttr` and
returns the real `DenseI64ArrayAttr` or `IntegerAttr`. Nothing here parses text.

**Every operation is listed rather than swept into a lookup with a default.**
That is `refexec.execute`'s rule and it is this file's for the same reason: an
operation added to the dialect and not to these tables raises by name instead of
being executed as whatever the default was.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from mlir import ir
from numpy.typing import NDArray

from . import refexec
from .builder import find_tool

Tensor = NDArray[np.float32]

__all__ = ["ExecutionError", "execute_module", "generic_form"]


class ExecutionError(Exception):
    """The module holds something this reference cannot execute."""


@dataclass(frozen=True)
class _Destination:
    """A `tensor.empty` result: a place to write, never a value to read.

    It is a distinct type rather than a zero filled array so that reading one
    as an operand is an error with a name instead of an answer that is quietly
    wrong by exactly the destination's contents. The shape is carried so that
    the destination an operation was handed can be checked against the result
    it declares, which is what makes the operand table below verified against
    the IR rather than assumed.
    """

    shape: tuple[int, ...]


#: How many trailing operands of each operation are destinations.
#:
#: Read off `NPUOps.td`'s `arguments` lists. The compute operations of Section
#: 7.2 are destination passing and carry exactly one `$destination` as their
#: last operand; `constant` and `reshape` carry none, `reshape` because it is
#: `Pure` and takes its extents from its result type.
_DESTINATION_OPERANDS: Final[dict[str, int]] = {
    "constant": 0,
    "conv2d": 1,
    "matmul": 1,
    "add": 1,
    "mul": 1,
    "relu": 1,
    "max_pool2d": 1,
    "avg_pool2d": 1,
    "reshape": 0,
    "transpose": 1,
    "concat": 1,
    "batch_norm": 1,
}

#: Why an operation of the dialect is not in the table above.
#:
#: Listed rather than omitted, so that meeting one produces the sentence a
#: reader needs rather than "unknown operation".
#:
#: `fused_op` and `yield` were here until P9, when `-npu-fuse-ops` gave them a
#: producer. They are handled structurally now, by `_execute_fused` below, and
#: not through `refexec`: a region is not an operation with a kernel, it is a
#: nesting, and executing it means binding its block arguments and walking its
#: body. `yield` stays absent from both tables for the same reason a terminator
#: is not a computation.
_NOT_EXECUTABLE: Final[dict[str, str]] = {
    "yield": (
        "npu.yield is the terminator of an npu.fused_op region and is executed "
        "with the region rather than on its own"
    ),
    "quantize": "the quantization pair arrives with the integer path at P14",
    "dequantize": "the quantization pair arrives with the integer path at P14",
}


def _i64_array(attribute: ir.Attribute) -> list[int]:
    return [int(value) for value in ir.DenseI64ArrayAttr(attribute)]


def _integer(attribute: ir.Attribute) -> int:
    return int(ir.IntegerAttr(attribute).value)


def _real(attribute: ir.Attribute) -> float:
    return float(ir.FloatAttr(attribute).value)


#: The attributes each operation carries, with their readers and their ODS
#: defaults. The names are `NPUOps.td`'s and they are also the keys
#: `refexec.execute` reads, which is not a coincidence: one vocabulary for the
#: dialect, the reference and the differential manifest.
#:
#: The default is used when the attribute is absent. `group` and `ceil_mode`
#: are `DefaultValuedAttr` in the ODS, so an operation at the default may or
#: may not carry one depending on how it was built, and a reader that assumed
#: presence would fail on a hand written module rather than on a defect.
_ATTRIBUTES: Final[
    dict[str, tuple[tuple[str, Callable[[ir.Attribute], Any], Any], ...]]
] = {
    "constant": (),
    "conv2d": (
        ("strides", _i64_array, [1, 1]),
        ("pads", _i64_array, [0, 0, 0, 0]),
        ("dilations", _i64_array, [1, 1]),
        ("group", _integer, 1),
    ),
    "matmul": (),
    "add": (),
    "mul": (),
    "relu": (),
    "max_pool2d": (
        ("kernel", _i64_array, None),
        ("strides", _i64_array, [1, 1]),
        ("pads", _i64_array, [0, 0, 0, 0]),
        ("dilations", _i64_array, [1, 1]),
        ("ceil_mode", _integer, 0),
    ),
    "avg_pool2d": (
        ("kernel", _i64_array, None),
        ("strides", _i64_array, [1, 1]),
        ("pads", _i64_array, [0, 0, 0, 0]),
        ("dilations", _i64_array, [1, 1]),
        ("ceil_mode", _integer, 0),
    ),
    "reshape": (),
    "transpose": (("permutation", _i64_array, None),),
    "concat": (("axis", _integer, None),),
    "batch_norm": (("epsilon", _real, 1e-5),),
}


def generic_form(module_text: str) -> str:
    """The same module, printed generically, which is what can be parsed here."""
    tool = find_tool("npu-opt")
    completed = subprocess.run(
        [str(tool), "-", "--mlir-print-op-generic"],
        input=module_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExecutionError(
            "npu-opt could not reprint this module generically, which means it "
            "is not IR this project produced.\n\n" + completed.stderr.strip()
        )
    return completed.stdout


def _shape_of(value: ir.Value) -> tuple[int, ...]:
    return tuple(int(extent) for extent in ir.RankedTensorType(value.type).shape)


def _read_attributes(
    operation: ir.Operation, mnemonic: str, result_shape: tuple[int, ...]
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for name, reader, default in _ATTRIBUTES[mnemonic]:
        try:
            raw = operation.attributes[name]
        except KeyError:
            if default is None:
                raise ExecutionError(
                    f"npu.{mnemonic} carries no {name!r} attribute and the ODS "
                    "gives it no default, so there is nothing to execute it "
                    "with."
                ) from None
            attributes[name] = default
            continue
        attributes[name] = reader(raw)

    # `reshape` takes its extents from its result type rather than from an
    # attribute, which is what makes it `Pure`. The reference's signature wants
    # them as `shape`, so they are supplied from the type here.
    if mnemonic == "reshape":
        attributes["shape"] = list(result_shape)
    return attributes


def _execute_one(op: ir.OpView, values: dict[ir.Value, Any]) -> None:
    """Executes one operation and records its result in `values`.

    Split out of `execute_module` at P9, when `npu.fused_op` gave the walk a
    second place to happen. The alternative was to inline the region's body into
    the outer loop, which would have meant one loop that sometimes meant the
    function's block and sometimes a region's, and the two have different value
    scopes: `npu.fused_op` is `IsolatedFromAbove`, so its body sees its block
    arguments and nothing else.
    """
    operation = op.operation
    name = operation.name

    if name == "tensor.empty":
        values[op.results[0]] = _Destination(_shape_of(op.results[0]))
        return

    if not name.startswith("npu."):
        raise ExecutionError(
            f"{name} is not an operation this reference executes. The module "
            "is expected to hold the npu dialect, tensor.empty destinations "
            "and func.return, and nothing else."
        )

    mnemonic = name[len("npu.") :]
    if mnemonic in _NOT_EXECUTABLE:
        raise ExecutionError(
            f"{name} cannot be executed here: {_NOT_EXECUTABLE[mnemonic]}."
        )

    result_shape = _shape_of(op.results[0])

    if mnemonic == "fused_op":
        values[op.results[0]] = _execute_fused(op, values, result_shape)
        return

    if mnemonic not in _DESTINATION_OPERANDS:
        raise ExecutionError(
            f"{name} is in the module and not in this file's tables. An "
            "operation added to the dialect is added here in the same commit, "
            "or it goes unchecked by the reference."
        )

    if mnemonic == "constant":
        dense = ir.DenseElementsAttr(operation.attributes["value"])
        values[op.results[0]] = np.array(dense).astype(np.float32)
        return

    operands = list(operation.operands)
    destinations = _DESTINATION_OPERANDS[mnemonic]
    if destinations:
        for operand in operands[len(operands) - destinations :]:
            holder = values[operand]
            if not isinstance(holder, _Destination):
                raise ExecutionError(
                    f"{name}'s destination operand is not a tensor.empty. "
                    "Destination passing style puts the destination last, and "
                    "this file's operand table says how many trailing operands "
                    "that is; a mismatch means the table and the dialect have "
                    "drifted."
                )
            if holder.shape != result_shape:
                raise ExecutionError(
                    f"{name} writes a destination of shape {holder.shape} and "
                    f"declares a result of {result_shape}"
                )
        operands = operands[: len(operands) - destinations]

    arrays: list[Tensor] = []
    for operand in operands:
        value = values[operand]
        if isinstance(value, _Destination):
            raise ExecutionError(
                f"{name} reads a tensor.empty destination as a value operand, "
                "which has no contents to read"
            )
        arrays.append(value)

    attributes = _read_attributes(operation, mnemonic, result_shape)
    try:
        produced = refexec.execute(mnemonic, arrays, attributes)
    except (KeyError, ValueError) as failure:
        raise ExecutionError(
            f"{name}: the reference refused it: {failure}"
        ) from failure

    if tuple(produced.shape) != result_shape:
        raise ExecutionError(
            f"{name}: the reference produced {tuple(produced.shape)} and the "
            f"operation declares {result_shape}"
        )
    values[op.results[0]] = produced


def _execute_fused(
    op: ir.OpView, values: dict[ir.Value, Any], result_shape: tuple[int, ...]
) -> Tensor:
    """Executes one `npu.fused_op` region and returns what it yielded.

    *Added at P9, with `-npu-fuse-ops`.* The region is a nesting rather than an
    operation with a kernel, so it is executed here and not in `refexec`: the
    reference interpreter executes one `npu` operation from the ODS description,
    and there is no arithmetic in a region to describe.

    The operands bind to the block arguments one for one, destinations included,
    which is what the operation's own verifier requires. `IsolatedFromAbove`
    means the body sees those and nothing else, so the inner scope starts empty
    rather than inheriting the enclosing one; a region that read an outer value
    would not have verified.
    """
    body = op.operation.regions[0].blocks[0]
    inner: dict[ir.Value, Any] = {}
    for operand, argument in zip(op.operation.operands, body.arguments, strict=True):
        inner[argument] = values[operand]

    for inner_op in body.operations:
        if inner_op.operation.name == "npu.yield":
            yielded = inner[inner_op.operation.operands[0]]
            if isinstance(yielded, _Destination):
                raise ExecutionError(
                    "npu.fused_op yields a tensor.empty destination that no "
                    "operation in its body wrote to"
                )
            if tuple(yielded.shape) != result_shape:
                raise ExecutionError(
                    f"npu.fused_op yields {tuple(yielded.shape)} and declares a "
                    f"result of {result_shape}"
                )
            return yielded
        _execute_one(inner_op, inner)

    raise ExecutionError(
        "npu.fused_op's region has no npu.yield, so there is nothing to return "
        "from it. The operation's verifier requires one, so a region without it "
        "did not come through npu-opt."
    )


def execute_module(
    module_text: str,
    inputs: Sequence[np.ndarray],
    *,
    function_name: str = "main",
) -> list[Tensor]:
    """Runs one `npu` module over `inputs` and returns its results.

    `module_text` is the IR `npu-compile --emit npu` prints, in its custom form.
    """
    generic = generic_form(module_text)

    context = ir.Context()
    context.allow_unregistered_dialects = True
    with context, ir.Location.unknown(context=context):
        module = ir.Module.parse(generic)

        functions = [
            operation
            for operation in module.body.operations
            if operation.operation.name == "func.func"
        ]
        named = [
            function
            for function in functions
            if ir.StringAttr(function.operation.attributes["sym_name"]).value
            == function_name
        ]
        if len(named) != 1:
            raise ExecutionError(
                f"the module holds {len(named)} functions named "
                f"{function_name!r} and exactly one is expected. It holds "
                f"{len(functions)} functions in total."
            )

        block = named[0].regions[0].blocks[0]
        if len(block.arguments) != len(inputs):
            raise ExecutionError(
                f"{function_name} takes {len(block.arguments)} arguments and "
                f"{len(inputs)} were given"
            )

        values: dict[ir.Value, Any] = {}
        for argument, array in zip(block.arguments, inputs, strict=True):
            expected = _shape_of(argument)
            supplied = tuple(int(extent) for extent in np.shape(array))
            if expected != supplied:
                raise ExecutionError(
                    f"{function_name} argument {argument.arg_number} has shape "
                    f"{expected} and the array supplied has {supplied}"
                )
            values[argument] = np.asarray(array, dtype=np.float32)

        results: list[Tensor] = []
        for op in block.operations:
            if op.operation.name == "func.return":
                for operand in op.operation.operands:
                    value = values[operand]
                    if isinstance(value, _Destination):
                        raise ExecutionError(
                            "the function returns a tensor.empty destination "
                            "that no operation wrote to"
                        )
                    results.append(value)
                continue
            _execute_one(op, values)

    return results
