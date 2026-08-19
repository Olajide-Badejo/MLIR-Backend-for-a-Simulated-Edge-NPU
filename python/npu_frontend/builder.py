# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Building `npu` dialect IR from Python, and verifying it with `npu-opt`.

The `npu` dialect is C++ only: there are no Python bindings for it, and the
`mlir_core` package the LLVM build installs carries the upstream dialects and
nothing of this project. So this module builds the module on a context with
`allow_unregistered_dialects` set, creating every `npu` operation generically,
and then hands the printed module to `./build/bin/npu-opt`, which has the real
dialect registered and runs the real verifiers.

**The round trip is the contract, not a test time extra.** The text `npu-opt`
prints is what `import_model` returns, so no unverified IR ever leaves this
package, and every test in `test/Python` that asserts on IR is asserting on
output the real parser and the real printer both agreed on. The whole argument
is in `docs/adr/0004-frontend-ir-emission-mechanism.md`.

One failure mode is specific to this arrangement and is closed here rather than
left to review. MLIR promotes the inherent attributes of a registered operation
into its properties when it parses a generic form, and an attribute whose name
matches no inherent one is kept as a discardable attribute. `strydes` for
`strides` would therefore parse, print and verify, with the operation's real
`strides` taking its ODS default and the typo riding along unread. The generic
printer writes properties as `<{...}>` and discardables as a plain `{...}`, so
the two are distinguishable in text, and `_reject_discardable_attributes` below
refuses any `npu` operation that has one. No operation this package emits has a
legitimate discardable attribute, so the rule is total.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
from mlir import ir
from mlir.dialects import func as func_dialect
from mlir.dialects import tensor as tensor_dialect

from .diagnostics import VerificationError

REPO_ROOT = Path(__file__).resolve().parents[2]

# An npu operation in generic form, up to the end of its operand list. What
# follows is an optional properties dictionary, then an optional discardable
# one, then the type signature after a colon. Anchoring on the operand close
# paren rather than trying to match the whole operation keeps this a scanner
# over a known printer's output rather than a parser for MLIR.
_GENERIC_NPU_OP = re.compile(r'"(npu\.[a-z_0-9]+)"\([^)]*\)\s*')


def find_npu_opt() -> Path:
    """Locate the `npu-opt` binary, or say where it was looked for.

    Three places, in this order: `NPU_OPT` in the environment wins, so a
    developer with a build somewhere else sets one variable; then this
    repository's own `build/bin`, which is where `ninja -C build` puts it; then
    `PATH`, which is what the CI container has after its build step.

    There is deliberately no fourth branch returning None. This package cannot
    produce verified IR without the binary, and a caller that got unverified IR
    because the binary was missing would have no way to tell.
    """
    override = os.environ.get("NPU_OPT")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        raise VerificationError(
            f"NPU_OPT is set to {override!r}, which is not a file. Unset it or "
            "point it at a built npu-opt."
        )

    in_tree = REPO_ROOT / "build" / "bin" / "npu-opt"
    if in_tree.is_file():
        return in_tree

    on_path = shutil.which("npu-opt")
    if on_path:
        return Path(on_path)

    raise VerificationError(
        "npu-opt was not found, and this package cannot emit verified IR "
        "without it. Looked at $NPU_OPT (unset or not a file), "
        f"{in_tree} , and PATH. Build it with:\n\n"
        "    ninja -C build npu-opt\n"
    )


def _run_npu_opt(module_text: str, extra_args: Sequence[str] = ()) -> str:
    npu_opt = find_npu_opt()
    completed = subprocess.run(
        [str(npu_opt), "-", *extra_args],
        input=module_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise VerificationError(
            "npu-opt rejected the module the importer built. This is a defect "
            "in the frontend rather than in the model: either a converter "
            "emitted the wrong operation, or the shape ONNX inferred "
            "disagrees with the shape the dialect computes for itself.\n\n"
            f"npu-opt said:\n{completed.stderr.strip()}"
        )
    return completed.stdout


def _reject_discardable_attributes(generic_text: str) -> None:
    """Refuse any `npu` operation carrying a discardable attribute.

    See the module docstring. The scan walks the generic form and, for each
    `npu` operation, looks at what sits between the end of its operand list and
    the colon that starts its type signature. A properties dictionary is
    `<{...}>` and comes first; anything in a bare `{...}` after that is
    discardable and means an attribute name the importer wrote did not match an
    inherent one.
    """
    for match in _GENERIC_NPU_OP.finditer(generic_text):
        mnemonic = match.group(1)
        rest = generic_text[match.end() :]
        if rest.startswith("<{"):
            depth = 0
            for index, character in enumerate(rest):
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        rest = rest[index + 1 :].lstrip(">").lstrip()
                        break
        if not rest.startswith("{"):
            continue
        leftover = rest[: rest.find("}") + 1]
        raise VerificationError(
            f"{mnemonic} carries the discardable attributes {leftover}. An "
            "attribute whose name does not match an inherent one is kept as a "
            "discardable attribute rather than rejected, so the operation "
            "verifies while using its ODS default for the attribute the "
            "importer meant to set. Check the spelling against NPUOps.td."
        )


def verify(module_text: str) -> str:
    """Round trip a module through `npu-opt` and return the canonical form.

    Two invocations rather than one, and both are load bearing. The first asks
    for the default printer with debug information, whose output is the custom
    assembly of Section 5.3 with every location printed, and is what this
    function returns. The second asks for the generic form, which is the only
    place the properties and discardable dictionaries are distinguishable, and
    is what the attribute name check reads.

    **Debug information is on rather than optional**, because Section 11
    requires every operation the importer creates to carry a `NameLoc` with its
    ONNX node name and a lit test at the `npuisa` level asserts one survives
    that far. A location that is built and then not printed is a location the
    next stage in the pipeline never sees, so leaving it off would end the whole
    location story at this boundary.
    """
    canonical = _run_npu_opt(module_text, ["--mlir-print-debuginfo"])
    _reject_discardable_attributes(
        _run_npu_opt(module_text, ["--mlir-print-op-generic"])
    )
    return canonical


class ModuleBuilder:
    """A module under construction, with one function in it.

    Thin on purpose. It owns the MLIR context and the insertion point, converts
    numpy arrays to `DenseElementsAttr`, and creates unregistered operations.
    Everything about ONNX lives in the converters; everything about shapes lives
    in the dialect's own verifiers.
    """

    def __init__(self, function_name: str) -> None:
        self.context = ir.Context()
        self.context.allow_unregistered_dialects = True
        self._function_name = function_name
        # Two stacks rather than one. The outer holds the context and the
        # location and lives for the whole build; the inner holds the function's
        # insertion point and is closed when the function ends. They are
        # separate because the inner one is closed twice on the happy path,
        # once by end_function and once by __exit__, and ExitStack.close is
        # idempotent, which is exactly the property this needs.
        self._outer = ExitStack()
        self._insertion = ExitStack()

    def __enter__(self) -> ModuleBuilder:
        self._outer.enter_context(self.context)
        self._outer.enter_context(ir.Location.unknown(context=self.context))
        self.f32 = ir.F32Type.get()
        self.module = ir.Module.create()
        return self

    def __exit__(self, *exception: object) -> None:
        """Unwind everything, including a function that was never finished.

        The "including" is the whole reason this is an `ExitStack` rather than
        four paired calls, and it is a defect this project already hit (D-0013).
        A converter that raises part way through a graph leaves the function's
        `InsertionPoint` on MLIR's thread local stack, pointing into a module
        that is about to be freed. Nothing fails at the time; the process
        segfaults at interpreter shutdown, long after the test that caused it
        reported a clean expected failure.
        """
        self._insertion.close()
        self._outer.close()

    # -- Types and locations -------------------------------------------------

    def tensor_type(self, shape: Sequence[int]) -> ir.RankedTensorType:
        """A statically shaped f32 tensor with no layout encoding.

        No encoding means NCHW, which is the rule `NPUAttrs.td` states and the
        reason the importer never writes one: every graph it emits is in the
        default layout, and `-npu-assign-layout` is what introduces the other
        one later. Writing `#npu.layout<nchw>` everywhere would say the same
        thing in more characters and would make that later pass's diff noisier.
        """
        return ir.RankedTensorType.get(list(shape), self.f32)

    def named_loc(self, name: str) -> ir.Location:
        """A `NameLoc` carrying an ONNX node name.

        Section 11 requires every operation the importer creates to carry one,
        because a pass that drops locations turns the whole debug section of the
        report into blanks and the only way to notice is to have put something
        there in the first place.
        """
        return ir.Location.name(name, context=self.context)

    # -- The function --------------------------------------------------------

    def begin_function(
        self,
        argument_shapes: Sequence[Sequence[int]],
        result_shapes: Sequence[Sequence[int]],
    ) -> list[ir.Value]:
        argument_types = [self.tensor_type(shape) for shape in argument_shapes]
        result_types = [self.tensor_type(shape) for shape in result_shapes]
        with ir.InsertionPoint(self.module.body):
            self._function = func_dialect.FuncOp(
                self._function_name, (argument_types, result_types)
            )
            entry = self._function.add_entry_block()
        self._insertion.enter_context(ir.InsertionPoint(entry))
        return list(entry.arguments)

    def end_function(self, results: Sequence[ir.Value]) -> None:
        func_dialect.ReturnOp(list(results))
        self._insertion.close()

    # -- Operations ----------------------------------------------------------

    def constant(self, array: np.ndarray, name: str) -> ir.Value:
        """An `npu.constant` holding `array`.

        The array goes through `DenseElementsAttr.get`, which reads the buffer
        directly, so a weight tensor never round trips through decimal text and
        no bit of it is lost on the way in. That is the single biggest reason
        this package builds IR through the bindings rather than as a string.
        """
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        value = ir.DenseElementsAttr.get(
            contiguous, type=self.tensor_type(contiguous.shape)
        )
        operation = ir.Operation.create(
            "npu.constant",
            results=[self.tensor_type(contiguous.shape)],
            attributes={"value": value},
            loc=self.named_loc(name),
        )
        return operation.result

    def empty(self, shape: Sequence[int], name: str) -> ir.Value:
        """A `tensor.empty` of `shape`, to be somebody's destination operand.

        Section 11 puts this job on the importer: the compute operations of
        Section 7.2 carry a destination operand, so somebody has to create one,
        and `tensor.empty` is the right thing to create because it carries shape
        and element type and no value semantics at all, which is exactly what a
        destination passing operand needs.
        """
        return tensor_dialect.EmptyOp(
            list(shape), self.f32, loc=self.named_loc(name)
        ).result

    def create(
        self,
        mnemonic: str,
        *,
        operands: Sequence[ir.Value],
        result_shape: Sequence[int],
        attributes: dict[str, Any] | None = None,
        name: str,
    ) -> ir.Value:
        """One `npu` operation, created generically.

        `mnemonic` is the bare name, so `"conv2d"` rather than `"npu.conv2d"`,
        which keeps the dialect prefix in one place and keeps `op_mapping.py`
        reading as a list of the dialect's own operation names.
        """
        operation = ir.Operation.create(
            f"npu.{mnemonic}",
            results=[self.tensor_type(result_shape)],
            operands=list(operands),
            attributes=dict(attributes or {}),
            loc=self.named_loc(name),
        )
        return operation.result

    def compute(
        self,
        mnemonic: str,
        *,
        inputs: Sequence[ir.Value],
        result_shape: Sequence[int],
        attributes: dict[str, Any] | None = None,
        name: str,
    ) -> ir.Value:
        """A destination passing `npu` operation, with its `tensor.empty`.

        The destination is materialised immediately before the operation that
        consumes it, which is what Section 11 asks for and is also what keeps
        the two adjacent in the printed IR: a reader checking that every compute
        operation has a destination reads two consecutive lines rather than
        chasing an SSA value up the function.
        """
        destination = self.empty(result_shape, name)
        return self.create(
            mnemonic,
            operands=[*inputs, destination],
            result_shape=result_shape,
            attributes=attributes,
            name=name,
        )

    # -- Attribute helpers ---------------------------------------------------

    def i64_array(self, values: Sequence[int]) -> ir.Attribute:
        return ir.Attribute.parse(
            f"array<i64: {', '.join(str(int(value)) for value in values)}>",
            context=self.context,
        )

    def i64(self, value: int) -> ir.Attribute:
        return ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, self.context), int(value)
        )

    def f32_attr(self, value: float) -> ir.Attribute:
        return ir.FloatAttr.get(self.f32, float(value))

    def printed(self) -> str:
        """The module as text, with locations.

        `str(module)` prints without debug information, so the `NameLoc` every
        operation here carries would be dropped on the way to `npu-opt` and the
        round trip would hand back IR with file and line locations pointing at
        stdin. Asking for debug information is what makes the ONNX node names
        survive the boundary.
        """
        return self.module.operation.get_asm(enable_debug_info=True)
