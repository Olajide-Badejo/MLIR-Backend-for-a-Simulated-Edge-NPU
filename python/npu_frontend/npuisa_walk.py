# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""One walk over an allocated ``npuisa`` module, shared by everything at P11.

*Added at P11.* Both consumers Section 16 asks for need the same thing and
neither can get it from a result file: the roofline of Section 16.6 needs the
bound **per layer**, and the SCALE-Sim exporter of Section 16.3 needs the shapes
of every convolution and matmul plus a named account of what it could not
represent. `Stats` reports the program's totals, so a per layer answer has to
come from the program rather than from the recording of it.

Two walks would have been two sets of shapes to keep in agreement, and Section
5.5's reason for one cost model home is the same reason this is one walker.

## What it is, and what it is not

It reads the textual IR that ``compile_model(..., emit="npuisa")`` produced,
after allocation, and returns one record per operation in program order. The
charges it computes are the Python mirror's, which
``test/Python/test_cost_model_mirror.py`` holds equal to
``include/NPU/Simulator/CostModel.h``. So this module has no cost model of its
own; it decides which charge applies to which operation and nothing else.

**It is not a parser for arbitrary MLIR.** It reads the output of this project's
own printer, which is regular because a machine wrote it. Anything it does not
recognise **raises** and names the line, which is the only safe answer for a
walker whose output becomes a committed number: a walk that skipped an operation
it did not understand would report a MAC total that quietly excluded it, and
Section 16.4's rule that an absent number must never be indistinguishable from a
zero is exactly that failure written for a different tool.

## The self check is the point

`check_against_result` compares the walk's totals with the totals the simulator
recorded for the same cell: raw MACs exactly, DRAM bytes read and written
exactly, and the instruction count exactly. Those are counted quantities on both
sides, so they are compared for equality rather than within a band. A walk that
disagrees with the machine about how many multiplies the program performed has
no business writing a per layer bound, and every caller here runs the check
before it uses a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

from . import cost_model

#: Every tensor on this machine is f32 until the integer kernels of P14, and the
#: walker refuses anything else rather than assuming a width. Section 16.1's
#: `int8_macs` stays zero for the same reason.
ELEMENT_BYTES: Final[int] = 4

#: The operations that carry a systolic charge, which is to say the two the
#: SCALE-Sim topology of Section 16.3 can represent at all.
COMPUTE_OPS: Final[frozenset[str]] = frozenset({"conv2d", "matmul"})

#: The operations that move bytes between DRAM and the scratchpad.
TRANSFER_OPS: Final[frozenset[str]] = frozenset({"dma_load", "dma_store"})

#: The operations charged as elementwise, at the result's element count.
ELEMENTWISE_OPS: Final[frozenset[str]] = frozenset(
    {"add", "mul", "sub", "div", "max", "min", "relu", "reshape", "transpose", "concat"}
)

#: The windowed reductions, charged at the result's element count times the
#: window, per `pool` in `lib/Simulator/Kernels.cpp`.
POOL_OPS: Final[frozenset[str]] = frozenset({"pool_avg", "pool_max"})

#: `npuisa.const` names a constant that lives in DRAM. It is not an executed
#: instruction: the encoder puts it in the constant pool and the `dma_load` that
#: reads it is the instruction that costs anything. Listed rather than ignored,
#: so that the walk refuses a genuinely unknown operation instead of treating
#: every unknown one as free.
DECLARATION_OPS: Final[frozenset[str]] = frozenset({"const"})


class WalkError(Exception):
    """The IR is not something this walker can read, or the walk disagrees."""


# ---------------------------------------------------------------------------
# The types.
# ---------------------------------------------------------------------------

_MEMREF = re.compile(
    r"memref<(?P<extents>[0-9x]*)(?P<element>[a-z0-9]+)"
    r"(?:,\s*strided<\[(?P<strides>[^\]]*)\]>)?"
    r"(?:,\s*#npu\.(?P<space>scratchpad|dram))?>"
)

_LOCATION = re.compile(r'^#(loc\d*) = loc\("(?P<name>[^"]*)"\)')

_OP = re.compile(r"npuisa\.(?P<op>[a-z_0-9]+)")

_INT_ATTRIBUTE = re.compile(r"(?P<name>[a-z_]+) = (?P<value>-?\d+) : i64")
_ARRAY_ATTRIBUTE = re.compile(
    r"(?P<name>[a-z_]+) = array<i64:\s*(?P<values>[-0-9, ]*)>"
)

_TRAILING_LOCATION = re.compile(r"loc\(#(loc\d*)\)\s*$")


@dataclass(frozen=True)
class MemRef:
    """One operand or result type, as the walker needs it."""

    shape: tuple[int, ...]
    element: str
    space: str
    strides: tuple[int, ...] | None

    @property
    def elements(self) -> int:
        count = 1
        for extent in self.shape:
            count *= extent
        return count

    @property
    def byte_count(self) -> int:
        return self.elements * ELEMENT_BYTES

    @property
    def innermost_stride(self) -> int:
        """The stride the DMA's stride penalty is charged against.

        `transfer` in `lib/Simulator/Kernels.cpp` reads it off the source operand
        and defaults to one when the operand carries no strides, and this
        reproduces that rather than paraphrasing it.
        """
        if self.strides is None or not self.strides:
            return 1
        return self.strides[-1]


def _parse_memref(text: str) -> MemRef:
    match = _MEMREF.search(text)
    if match is None:
        raise WalkError(
            f"no memref type in {text!r}. Every operand of an allocated npuisa "
            f"module has one, so a line without one is a line this walker has "
            f"misread rather than an operand it may skip."
        )
    extents = tuple(
        int(part) for part in match.group("extents").split("x") if part.strip()
    )
    element = match.group("element")
    if element != "f32":
        raise WalkError(
            f"the element type is {element!r} and this walker reads f32. The "
            f"integer kernels arrive at P14 and a byte count computed at the "
            f"wrong width would be wrong in a direction nothing would notice."
        )
    strides_text = match.group("strides")
    strides = (
        tuple(int(part) for part in strides_text.split(",") if part.strip())
        if strides_text is not None
        else None
    )
    return MemRef(
        shape=extents,
        element=element,
        space=match.group("space") or "scratchpad",
        strides=strides,
    )


def _parse_memrefs(text: str) -> list[MemRef]:
    return [_parse_memref(found.group(0)) for found in _MEMREF.finditer(text)]


# ---------------------------------------------------------------------------
# One operation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """One `npuisa` operation, with the charge the cost model gives it."""

    position: int
    #: The mnemonic without the dialect prefix: `conv2d`, `relu`, `dma_load`.
    op: str
    #: The name the location carries, which is the ONNX node's, or a synthesised
    #: `<op>_<position>` where the location is unknown. Names are what a
    #: divergence table is read by, so a layer with no name would be a row a
    #: reader cannot attribute.
    name: str
    operands: tuple[MemRef, ...]
    result: MemRef
    attributes: dict[str, Any]

    #: The port the instruction issues on, `dma` or `compute`.
    port: str
    #: The kernel's own charge, before the issue overhead the machine adds.
    kernel_cycles: float
    #: Raw multiply accumulates. Never scaled by utilization; Section 5.5.
    macs: int
    effective_macs: float
    utilization: float
    delta: float
    dram_bytes_read: int
    dram_bytes_written: int

    #: The DRAM bytes attributed to this operation for the roofline's memory
    #: bound branch: its own traffic, plus the transfers that exist only to feed
    #: it or to drain it. Filled by `attribute_transfers`.
    attributed_dram_bytes: int = 0
    #: The cycles those transfers were charged, on the DMA timeline. Carried
    #: beside the bytes because the roofline compares a **time** against a bound
    #: in cycles, and the bytes alone cannot say what the machine charged to move
    #: them: `dma_cycles` adds a descriptor and a stride penalty on top of bytes
    #: over bandwidth, and both are part of what the layer cost.
    attributed_dma_cycles: float = 0.0
    #: On a transfer, the position of the operation its bytes were charged to, or
    #: its own position when nothing consumed or produced the buffer it moved.
    #: `None` on everything that is not a transfer.
    #:
    #: It exists because "were this transfer's bytes counted somewhere else" and
    #: "were they counted **inside a layer SCALE-Sim can see**" are different
    #: questions, and the SCALE-Sim divergence needs the second one: a load
    #: feeding a convolution is already inside that layer's charge, and one
    #: feeding a pooling operation is genuinely outside the comparison and is a
    #: named term of it. Deriving the answer from `attributed_dram_bytes` being
    #: zero would answer the first question and quietly get the second wrong.
    attributed_to: int | None = None

    @property
    def cycles(self) -> float:
        """What the machine charges, which includes the per instruction issue
        overhead of Section 5.5."""
        return self.kernel_cycles + cost_model.ISSUE_OVERHEAD_CYCLES

    @property
    def is_compute(self) -> bool:
        return self.op in COMPUTE_OPS


def _attribute_values(text: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for found in _ARRAY_ATTRIBUTE.finditer(text):
        attributes[found.group("name")] = tuple(
            int(part) for part in found.group("values").split(",") if part.strip()
        )
    for found in _INT_ATTRIBUTE.finditer(text):
        attributes[found.group("name")] = int(found.group("value"))
    return attributes


def _split_ins_outs(body: str) -> tuple[str, str]:
    """The `ins(...)` and `outs(...)` clauses, by balanced parentheses.

    A regular expression cannot do this correctly: `ins(` holds types that hold
    their own parentheses in the strided case, and a non greedy match stops at
    the first one. Counting is short and right, so it is what this does.
    """
    clauses: dict[str, str] = {}
    for keyword in ("ins", "outs"):
        start = body.find(f"{keyword}(")
        if start < 0:
            clauses[keyword] = ""
            continue
        cursor = start + len(keyword)
        depth = 0
        for index in range(cursor, len(body)):
            if body[index] == "(":
                depth += 1
            elif body[index] == ")":
                depth -= 1
                if depth == 0:
                    clauses[keyword] = body[cursor + 1 : index]
                    break
        else:
            raise WalkError(f"unbalanced {keyword}( in {body!r}")
    return clauses["ins"], clauses["outs"]


def _charge(
    op: str, operands: list[MemRef], result: MemRef, attributes: dict[str, Any]
) -> tuple[str, float, cost_model.ComputeCharge]:
    """The kernel's charge, dispatched on the mnemonic.

    Every branch mirrors one function in `lib/Simulator/Kernels.cpp` and takes
    its element counts from the same place that file does, which is the result
    shape everywhere except the pools.
    """
    peak = cost_model.PEAK_MACS_PER_CYCLE_F32

    if op == "conv2d":
        batch = result.shape[0]
        output_channels = result.shape[1]
        output_height = result.shape[2]
        output_width = result.shape[3]
        weights = operands[1]
        kernel_height = weights.shape[2]
        kernel_width = weights.shape[3]
        input_channels = operands[0].shape[1]
        group = int(attributes.get("group", 1))
        charge = cost_model.conv2d_charge(
            batch,
            output_channels,
            input_channels,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            group,
            peak,
        )
        return "compute", charge.cycles, charge

    if op == "matmul":
        rows = result.shape[0]
        columns = result.shape[1]
        reduction = operands[0].shape[1]
        charge = cost_model.gemm_charge(rows, reduction, columns, peak)
        return "compute", charge.cycles, charge

    empty = cost_model.ComputeCharge(0.0, 0, 0.0, 1.0, 1.0)

    if op in TRANSFER_OPS:
        source = operands[0]
        cycles = cost_model.dma_cycles(
            result.elements * ELEMENT_BYTES, result.elements, source.innermost_stride
        )
        return "dma", cycles, empty

    if op in POOL_OPS:
        window = attributes.get("kernel")
        if window is None or len(window) != 2:
            raise WalkError(f"{op} carries no two element kernel attribute")
        cycles = cost_model.elementwise_cycles(
            result.elements * int(window[0]) * int(window[1])
        )
        return "compute", cycles, empty

    if op in ELEMENTWISE_OPS:
        return "compute", cost_model.elementwise_cycles(result.elements), empty

    raise WalkError(
        f"npuisa.{op} has no charge in this walker. Section 16.4's rule that an "
        f"absent number must never be indistinguishable from a zero applies here "
        f"as well: an operation walked past silently would leave every total "
        f"this module produces quietly short of the machine's. Add it to one of "
        f"the sets at the top of this module, with the kernel it mirrors."
    )


# ---------------------------------------------------------------------------
# The walk.
# ---------------------------------------------------------------------------


def statements(npuisa_text: str) -> list[str]:
    """The lines of an allocated module that are `npuisa` operations, in order.

    **One rule, used by both passes over the text**, because the walk and the
    attribution have to agree about what an operation is and two copies of the
    rule would be two chances to disagree. It was two copies for one revision of
    this module and the disagreement was checked for; sharing the rule is better
    than checking it, so the check went and the sharing stayed.

    `npuisa.arg`, `npuisa.scratchpad_arena`, `npuisa.fragmentation_ratio` and the
    rest of the allocator's attributes carry the dialect prefix and are not
    operations. They are told apart by where the mnemonic sits: an operation
    either starts its statement or follows a single SSA result and an `=`.
    `npuisa.const` matches that shape and is still not an executed instruction,
    which is what `DECLARATION_OPS` is for.
    """
    found_statements: list[str] = []
    for raw in npuisa_text.splitlines():
        line = raw.strip()
        found = _OP.search(line)
        if found is None:
            continue
        if not (line.startswith("npuisa.") or re.match(r"^%\S+ = npuisa\.", line)):
            continue
        if found.group("op") in DECLARATION_OPS:
            continue
        found_statements.append(line)
    return found_statements


def walk(npuisa_text: str) -> list[Operation]:
    """Every `npuisa` operation of an allocated module, in program order."""
    names: dict[str, str] = {}
    for line in npuisa_text.splitlines():
        found = _LOCATION.match(line.strip())
        if found is not None:
            names[found.group(1)] = found.group("name")

    operations: list[Operation] = []
    for text in statements(npuisa_text):
        found = _OP.search(text)
        assert found is not None  # `statements` only yields lines that match
        op = found.group("op")

        location = _TRAILING_LOCATION.search(text)
        name = names.get(location.group(1), "") if location is not None else ""
        position = len(operations)

        if op in TRANSFER_OPS:
            types = _parse_memrefs(text)
            if len(types) != 2:
                raise WalkError(
                    f"npuisa.{op} names {len(types)} types and a transfer has "
                    f"exactly two, a source and a destination: {text!r}"
                )
            operands = [types[0]]
            result = types[1]
            attributes: dict[str, Any] = {}
        else:
            body = text
            ins, outs = _split_ins_outs(body)
            if not ins or not outs:
                raise WalkError(
                    f"npuisa.{op} has no ins/outs clause this walker could read: "
                    f"{text!r}"
                )
            operands = _parse_memrefs(ins)
            results = _parse_memrefs(outs)
            if len(results) != 1:
                raise WalkError(f"npuisa.{op} writes {len(results)} results: {text!r}")
            result = results[0]
            tail = body[body.find("outs(") :]
            brace = tail.find("{")
            attributes = _attribute_values(tail[brace:]) if brace >= 0 else {}

        port, kernel_cycles, charge = _charge(op, operands, result, attributes)

        read = 0
        written = 0
        if op == "dma_load":
            read = result.elements * ELEMENT_BYTES
        elif op == "dma_store":
            written = result.elements * ELEMENT_BYTES

        operations.append(
            Operation(
                position=position,
                op=op,
                name=name or f"{op}_{position}",
                operands=tuple(operands),
                result=result,
                attributes=attributes,
                port=port,
                kernel_cycles=kernel_cycles,
                macs=charge.macs,
                effective_macs=charge.effective_macs,
                utilization=charge.utilization,
                delta=charge.delta,
                dram_bytes_read=read,
                dram_bytes_written=written,
            )
        )

    if not operations:
        raise WalkError(
            "this module holds no npuisa operations. An empty walk producing an "
            "empty table is how a check stops checking rather than starts "
            "failing, so it is refused here."
        )
    return operations


# ---------------------------------------------------------------------------
# Attributing DRAM traffic to the layer it belongs to.
# ---------------------------------------------------------------------------


def attribute_transfers(
    operations: list[Operation], npuisa_text: str
) -> list[Operation]:
    """Charge each transfer's bytes to the operation it exists to serve.

    **This is an accounting decision and it is stated rather than assumed.** A
    convolution on this machine touches no DRAM: the allocator has already moved
    its operands into the scratchpad, so a per layer operational intensity read
    off the compute operation alone would be infinite for every layer, and the
    memory bound branch of Section 16.6, the independently binding half, would
    never bind anywhere. That would be a roofline that cannot fail.

    So a `dma_load` is charged to the **first** operation that reads the buffer
    it filled, and a `dma_store` to the operation that produced the buffer it
    drained. A transfer with no such consumer or producer keeps its bytes on
    itself and is reported by `unattributed_dram_bytes`, which is a number rather
    than a silence.

    The def use edges come from the SSA names in the printed IR, which are unique
    per buffer even where the allocator gave two buffers the same address. Two
    views at one offset are two names, so the attribution follows the program
    rather than the arena.

    **Aliases are followed, and finding out that they had to be is why this
    paragraph exists.** ADR 0005's broadcast mechanism gives a rank 1 channel
    operand the result's rank through a `memref.reinterpret_cast` with zero
    strides, so the buffer a `dma_load` filled and the buffer the `mul` reads
    have different SSA names. Without following the cast, every broadcast bias in
    the suite came out unattributed: 32 bytes on `resnet_block`, 128 on
    `conv_bn_relu_stack`, 20 on `dilated_stack`. The bytes were never lost, since
    `unattributed_dram_bytes` reported them, which is the whole reason that field
    is a number rather than a silence. They were charged to the transfer instead
    of to the layer the transfer exists for.
    """
    aliases: dict[str, str] = {}
    for line in npuisa_text.splitlines():
        found = re.match(
            r"^(%[A-Za-z0-9_]+) = memref\.(?:reinterpret_cast|subview)\s+"
            r"(%[A-Za-z0-9_]+)",
            line.strip(),
        )
        if found is not None:
            aliases[found.group(1)] = found.group(2)

    def resolve(value: str) -> str:
        """The buffer a name ultimately refers to, through any chain of casts."""
        seen: set[str] = set()
        while value in aliases and value not in seen:
            seen.add(value)
            value = aliases[value]
        return value

    lines = statements(npuisa_text)
    if len(lines) != len(operations):
        raise WalkError(
            f"the attribution was given {len(operations)} operations and the "
            f"text holds {len(lines)}, so the two are not the same program."
        )

    produced: dict[str, int] = {}
    consumed: dict[str, list[int]] = {}
    transfer_target: dict[int, str] = {}
    transfer_source: dict[int, str] = {}

    for index, statement in enumerate(lines):
        op = operations[index].op
        if op in TRANSFER_OPS:
            values = re.findall(r"%[A-Za-z0-9_]+", statement.split(" : ")[0])
            if len(values) != 2:
                raise WalkError(
                    f"npuisa.{op} names {len(values)} values: {statement!r}"
                )
            source, destination = resolve(values[0]), resolve(values[1])
            transfer_source[index] = source
            transfer_target[index] = destination
            if op == "dma_load":
                produced[destination] = index
            else:
                consumed.setdefault(source, []).append(index)
            continue
        ins, outs = _split_ins_outs(statement)
        for value in re.findall(r"%[A-Za-z0-9_]+", ins.split(" : ")[0]):
            consumed.setdefault(resolve(value), []).append(index)
        for value in re.findall(r"%[A-Za-z0-9_]+", outs.split(" : ")[0]):
            produced.setdefault(resolve(value), index)

    attributed = [0] * len(operations)
    charged = [0.0] * len(operations)
    goes_to: list[int | None] = [None] * len(operations)
    for index, operation in enumerate(operations):
        if operation.op == "dma_load":
            readers = [
                reader
                for reader in consumed.get(transfer_target[index], [])
                if operations[reader].op not in TRANSFER_OPS
            ]
            target = readers[0] if readers else index
            attributed[target] += operation.dram_bytes_read
            charged[target] += operation.cycles
            goes_to[index] = target
        elif operation.op == "dma_store":
            writer = produced.get(transfer_source[index])
            target = (
                writer
                if writer is not None and operations[writer].op not in TRANSFER_OPS
                else index
            )
            attributed[target] += operation.dram_bytes_written
            charged[target] += operation.cycles
            goes_to[index] = target

    return [
        Operation(
            **{
                **operation.__dict__,
                "attributed_dram_bytes": attributed[index],
                "attributed_dma_cycles": charged[index],
                "attributed_to": goes_to[index],
            }
        )
        for index, operation in enumerate(operations)
    ]


# ---------------------------------------------------------------------------
# The self check.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Totals:
    """What the walk says the whole program did."""

    instructions: int
    macs: int
    dram_bytes_read: int
    dram_bytes_written: int
    dma_cycles: float
    compute_cycles: float
    #: `cycles * peak` summed over the compute operations, which is what
    #: `Stats.effectiveMacs` is. Nothing in the energy path ever sees it.
    effective_macs: float = 0.0
    #: The MAC weighted means the machine reports, reconstructed the same way it
    #: reconstructs them, so a dropped occupancy term shows up here rather than
    #: nowhere: `macs` alone would agree even if `utilization` had been dropped.
    utilization: float = 1.0
    delta: float = 1.0
    unattributed_dram_bytes: int = 0
    layers: int = field(default=0)


def totals(operations: list[Operation]) -> Totals:
    """The walk's own totals, including the `HALT` the encoder appends.

    `HALT` is not in the IR and is in the program: `Stats.instructions` counts
    instructions executed and `HALT` is one of them. Adding it here is what makes
    the instruction count comparable rather than off by one, and it is added in
    one place so that no caller has to remember.
    """
    unattributed = sum(
        operation.attributed_dram_bytes
        for operation in operations
        if operation.op in TRANSFER_OPS
    )
    macs = sum(operation.macs for operation in operations)
    return Totals(
        instructions=len(operations) + 1,
        macs=macs,
        effective_macs=sum(operation.effective_macs for operation in operations),
        utilization=(
            sum(operation.utilization * operation.macs for operation in operations)
            / macs
            if macs
            else 1.0
        ),
        delta=(
            sum(operation.delta * operation.macs for operation in operations) / macs
            if macs
            else 1.0
        ),
        dram_bytes_read=sum(operation.dram_bytes_read for operation in operations),
        dram_bytes_written=sum(
            operation.dram_bytes_written for operation in operations
        ),
        dma_cycles=sum(
            operation.cycles for operation in operations if operation.port == "dma"
        ),
        compute_cycles=sum(
            operation.cycles for operation in operations if operation.port == "compute"
        )
        + cost_model.ISSUE_OVERHEAD_CYCLES,
        unattributed_dram_bytes=unattributed,
        layers=sum(1 for operation in operations if operation.is_compute),
    )


def check_against_result(operations: list[Operation], result: dict[str, Any]) -> None:
    """The walk agrees with the machine, on the counted quantities, exactly.

    Raw MACs, DRAM bytes read, DRAM bytes written and the instruction count are
    counts on both sides. They are compared for equality and not within a band,
    because a band on a count is a band that hides the disagreement it was added
    to tolerate.

    Cycles are **not** compared here, and the reason is the two timelines of
    Section 5.5: the machine's total is the later of the DMA and compute
    timelines at `HALT`, which is neither the sum of the per instruction charges
    nor either one of them, and reconstructing the interleaving from the IR would
    be a second scheduler.
    """
    walked = totals(operations)
    simulation = result["simulation"]
    disagreements: list[str] = []
    for label, mine, theirs in (
        ("macs", walked.macs, int(simulation["macs"])),
        (
            "dram_bytes_read",
            walked.dram_bytes_read,
            int(simulation["dram_bytes_read"]),
        ),
        (
            "dram_bytes_written",
            walked.dram_bytes_written,
            int(simulation["dram_bytes_written"]),
        ),
        ("instruction_count", walked.instructions, int(result["instruction_count"])),
    ):
        if mine != theirs:
            disagreements.append(
                f"{label}: the walk says {mine}, the cell says {theirs}"
            )

    # And the three derived figures, which are what actually exercise
    # `conv2d_charge`: the counts above would agree even if the occupancy terms
    # were dropped, because `macs` does not carry them. These are doubles summed
    # in program order on both sides, so they should be bit identical and are
    # compared within a band derived from that rather than assumed to be: at most
    # a few hundred additions at a unit roundoff of 2.22e-16 bounds the relative
    # error near 1e-13, and 1e-12 is one order above it.
    macs = walked.macs
    if macs > 0:
        for label, mine_float, theirs_float in (
            (
                "effective_macs",
                walked.effective_macs,
                float(simulation["effective_macs"]),
            ),
            ("utilization", walked.utilization, float(simulation["utilization"])),
            ("delta", walked.delta, float(simulation["delta"])),
        ):
            scale = max(abs(mine_float), abs(theirs_float), 1.0)
            if abs(mine_float - theirs_float) > 1e-12 * scale:
                disagreements.append(
                    f"{label}: the walk says {mine_float!r}, the cell says "
                    f"{theirs_float!r}"
                )

    if disagreements:
        raise WalkError(
            f"the walk of {result['cell']['name']} disagrees with the numbers the "
            f"simulator recorded for the same cell:\n  "
            + "\n  ".join(disagreements)
            + "\n\nThese are counts on both sides, so this is a real disagreement "
            "rather than a rounding one. A per layer figure derived from a walk "
            "that cannot reproduce the program's totals is not a measurement."
        )
