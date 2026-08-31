#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Enforces law 2 of this project: no operator representing imported computation
# may exist in the `npu` dialect unless it is importable from ONNX, lowerable to
# `npuisa`, encodable, simulatable, and exercised by at least one model in the
# benchmark suite. An operator that cannot satisfy all five is deleted, and the
# one alternative to deleting it is a dated entry in docs/EXEMPTIONS.md naming
# the phase that resolves it.
#
# The check classifies before it judges. An operation marked as imported
# computation must satisfy all five layers; a structural operation, meaning a
# region terminator or an operation a pass creates rather than the importer,
# corresponds to no ONNX node and so is held to the four that are not
# importability.
#
# **The classification is read out of the ODS description**, not out of a table
# this script keeps. That is the whole point: an operation cannot be quietly
# reclassified to make the check pass, because doing so changes NPUOps.td and
# changes docs/DIALECT_REFERENCE.md, and both are reviewed.
#
#   python scripts/check-reachability.py                 the full check
#   python scripts/check-reachability.py --skip-models   without the model layer
#
# --skip-models exists for the lint job, which runs before any model is built.
# It is not a weaker check of the same thing: it is the same check with the one
# layer that needs artifacts left out, and it says so in its own output rather
# than reporting a pass that covered less than the reader thinks.

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OPS_TD = REPO_ROOT / "include" / "NPU" / "Dialect" / "NPU" / "IR" / "NPUOps.td"
EXEMPTIONS = REPO_ROOT / "docs" / "EXEMPTIONS.md"

# The generated opcode list of Section 9.4. It is written by
# `ninja -C build npu-isa-doc` from include/NPU/Encoding/NPUISADescription.td
# and committed, and scripts/check-isa-staleness.sh keeps it honest.
ISA_OPCODES = REPO_ROOT / "docs" / "ISA_OPCODES.json"

# The five layers of law 2, in the order the pipeline meets them.
LAYERS = ("import", "lowering", "encoding", "simulation", "model")

# The layer a structural operation is not held to, because it corresponds to no
# ONNX node and so importability is a requirement it can never meet.
STRUCTURAL_EXCLUDES = ("import",)

# An operation definition in the ODS source. The mnemonic is the quoted string
# after the operation class name, which is the first thing on the line for every
# definition in this dialect.
#
#   def NPU_AddOp : NPU_ElementwiseBinaryOp<"add"> {
#   def NPU_Conv2DOp : NPU_ComputeOp<"conv2d",
#
# The record name is captured too, because a diagnostic that names only the
# mnemonic sends the reader searching for a string that appears in a dozen
# places.
OP_DEF_RE = re.compile(
    r"^def\s+(?P<record>NPU_\w+)\s*:\s*NPU_\w+<\s*\"(?P<mnemonic>[\w_]+)\"",
    re.MULTILINE,
)

# The classification line inside an operation's description.
CLASSIFICATION_RE = re.compile(
    r"^\s*Reachability:\s*(?P<kind>imported computation|structural)\s*\.\s*$",
    re.MULTILINE,
)

# One entry of the EXEMPT block of docs/EXEMPTIONS.md. Lines beginning with a
# hash are comments, including the column header, so they are skipped.
EXEMPT_ENTRY_RE = re.compile(
    r"^(?P<op>npu\.[\w_]+)\s+(?P<layer>\w+)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<phase>P\d+)\s+(?P<reason>.+?)\s*$"
)


@dataclass
class Operation:
    """One operation of the npu dialect, as the ODS source describes it."""

    record: str
    mnemonic: str
    classification: str | None = None

    @property
    def name(self) -> str:
        return f"npu.{self.mnemonic}"

    @property
    def required_layers(self) -> tuple[str, ...]:
        if self.classification == "structural":
            return tuple(layer for layer in LAYERS if layer not in STRUCTURAL_EXCLUDES)
        return LAYERS


@dataclass
class Exemption:
    op: str
    layer: str
    date: str
    phase: str
    reason: str


@dataclass
class Findings:
    """What the check found, kept as data so the report is written once."""

    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # op name -> the layers it is missing.
    missing: dict[str, list[str]] = field(default_factory=dict)


def parse_operations(source: str) -> list[Operation]:
    """Reads the operation list and each operation's classification.

    The classification belongs to the operation whose definition most recently
    opened, which is what makes a missing classification detectable: the next
    definition starts and the previous one still has none.
    """
    operations: list[Operation] = []
    matches = list(OP_DEF_RE.finditer(source))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[start:end]

        operation = Operation(
            record=match.group("record"), mnemonic=match.group("mnemonic")
        )
        classification = CLASSIFICATION_RE.search(body)
        if classification:
            kind = classification.group("kind")
            operation.classification = (
                "structural" if kind == "structural" else "imported"
            )
        operations.append(operation)

    return operations


def parse_exemptions(text: str) -> list[Exemption]:
    """Reads the EXEMPT block of docs/EXEMPTIONS.md.

    The block is the fenced code block that follows the `## EXEMPT` heading. It
    is parsed positionally rather than by a general markdown reader, because the
    file's own documentation describes the shape and a parser that accepted more
    than that shape would accept an entry the file says is invalid.
    """
    exemptions: list[Exemption] = []

    heading = re.search(r"^##\s+EXEMPT\s*$", text, re.MULTILINE)
    if not heading:
        return exemptions

    fence = re.search(
        r"^```\s*$(?P<block>.*?)^```\s*$",
        text[heading.end() :],
        re.MULTILINE | re.DOTALL,
    )
    if not fence:
        return exemptions

    for line in fence.group("block").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entry = EXEMPT_ENTRY_RE.match(stripped)
        if entry:
            exemptions.append(Exemption(**entry.groupdict()))

    return exemptions


# Where each layer's evidence lives, once the phase that builds it has landed.
# A layer whose home does not exist yet is not decidable, and the difference
# between "not decidable" and "absent" is the difference between a check that is
# not yet applicable and a check that failed.
#
# The paths are the ones the repository layout names. When a phase creates one,
# this table needs no edit: the layer becomes decidable on the day its directory
# appears, which is the day the check should start asking about it.
LAYER_HOMES: dict[str, tuple[Path, ...]] = {
    "import": (REPO_ROOT / "python" / "npu_frontend" / "op_mapping.py",),
    "lowering": (
        REPO_ROOT
        / "lib"
        / "Dialect"
        / "NPUISA"
        / "Transforms"
        / "LowerNPUToNPUISA.cpp",
    ),
    "encoding": (ISA_OPCODES,),
    # *Changed at P8.* The simulation layer used to be a substring search over
    # lib/Simulator/Simulator.cpp's operation table, which is a comment. It is
    # now decided from the same generated description the encoding layer reads.
    # See `simulatable_operations`.
    "simulation": (ISA_OPCODES,),
    # Written by scripts/build-model-ir.py, which the CI step runs immediately
    # before this check. The directory holds the `npu` and `npuisa` level IR of
    # every model at both batch sizes, and the IR is what step 3 of Section 17.5
    # asks about: an .onnx file holds ONNX operator names, so searching one for
    # `npu.batch_norm` would find nothing whatever the truth was.
    "model": (REPO_ROOT / "experiments" / "models",),
}


def layer_decidable(layer: str) -> bool:
    """Whether the phase that builds this layer has landed."""
    return all(home.exists() for home in LAYER_HOMES[layer])


def encodable_operations() -> set[str]:
    """The `npu` mnemonics the ISA description accounts for.

    *Changed at P6.* Every other layer is decided by looking for the operation's
    mnemonic in a source file, which is coarse on purpose: the job is to make
    adding an operator without wiring it up impossible to do silently, not to
    prove the wiring is correct. The encoding layer used to be decided the same
    way, and Section 9.4 names that as the thing to replace, because the encoder
    reads `npuisa` operations and the mnemonic it would be searched for is the
    `npu` one. Grepping for `max_pool2d` in a file that says `PoolMaxOp` finds
    nothing true either way.

    So the encoding layer is decided from the generated opcode list instead. An
    operation is encodable when some opcode names it as a source, or when the
    description records that it reaches the encoder by elimination, which is
    what `npu.fused_op`, `npu.yield` and `npu.constant` do. Both halves come out
    of one description file that the staleness gate keeps honest, so this answer
    cannot drift from the instruction set the way a substring search could.
    """
    data = json.loads(ISA_OPCODES.read_text(encoding="utf-8"))
    names: set[str] = set()
    for opcode in data.get("opcodes", []):
        names.update(opcode.get("sources", []))
    for entry in data.get("eliminated_sources", []):
        names.add(entry["source"])
    return names


def simulatable_operations() -> set[str]:
    """The `npu` mnemonics the simulator has kernels for.

    *Changed at P8, and this is the fifth thing P7 left on this phase's desk.*
    The simulation layer used to be a substring search over
    `lib/Simulator/Simulator.cpp`, which carries an operation table as a
    comment. That was the weakest of the four layers and the file said so: a
    mnemonic appearing inside an unrelated word would have satisfied it, and
    the evidence was a comment rather than anything the compiler enforces.

    It is decided from the generated description now, the way P6 made the
    encoding layer decidable. An operation is simulatable when it reaches the
    encoder by elimination, so no instruction ever executes it, or when every
    opcode naming it as a source is one the simulator needs a kernel for.

    **The part that makes this stronger is enforced elsewhere, and saying where
    is the point.** This function establishes that an operation reaches a
    computation opcode. That the computation opcode *has* a kernel is the
    compiler's guarantee rather than this script's: the dispatch skeleton
    generated from the same description expands to a missing identifier, and to
    a static assertion that the kernel table and the ISA description disagree
    about how many opcodes there are, when one is absent. P7 demonstrated that
    by appending a sixteenth opcode and watching the build fail in four places,
    each naming it. So the two halves of "simulatable" are checked by the two
    mechanisms that can check them, and neither is a comment.

    **What this is not.** It is close in shape to `encodable_operations`, and
    the difference is one clause: an operation reaching only a control opcode,
    which the simulator sequences rather than computes, would be encodable and
    not simulatable. No operation is in that position today. The distinction is
    thin and it is real, and stating its thinness is better than implying two
    independent checks where there are one and a half.
    """
    data = json.loads(ISA_OPCODES.read_text(encoding="utf-8"))

    eliminated = {entry["source"] for entry in data.get("eliminated_sources", [])}

    # source -> whether every opcode reaching it needs a kernel.
    computed: dict[str, bool] = {}
    for opcode in data.get("opcodes", []):
        needs_kernel = bool(opcode.get("needs_kernel", False))
        for source in opcode.get("sources", []):
            computed[source] = computed.get(source, True) and needs_kernel

    return eliminated | {name for name, ok in computed.items() if ok}


def layer_present(operation: Operation, layer: str) -> bool | None:
    """Whether one layer exists for one operation.

    Returns None when the layer is not yet decidable, meaning the phase that
    builds it has not landed and there is nothing to look in. Reporting that as
    a note rather than as a pass is what keeps this script from claiming to have
    checked something it did not.

    Once a layer is decidable, the evidence is the operation's mnemonic
    appearing in that layer's source. That is a coarse test and it is meant to
    be: its job is to make adding an operator without wiring it up impossible to
    do silently, not to prove the wiring is correct. Correctness is what the
    end to end tests are for.
    """
    if not layer_decidable(layer):
        return None

    # Two layers are answered from the generated opcode list rather than by
    # searching a source file. See `encodable_operations` and
    # `simulatable_operations` for why each is.
    if layer == "encoding":
        return operation.mnemonic in encodable_operations()
    if layer == "simulation":
        return operation.mnemonic in simulatable_operations()

    # The model layer looks for the operation as it is spelled in the IR, so
    # `npu.relu` and not `relu`. The dotted form is what a `.mlir` file holds,
    # and searching for the bare mnemonic would match inside another operation's
    # name: `add` is a substring of `batch_norm`'s neighbours and of half the
    # SSA value names a pass ever generates.
    if layer == "model":
        needle = operation.name
    else:
        needle = operation.mnemonic
    for home in LAYER_HOMES[layer]:
        if home.is_dir():
            sources = sorted(p for p in home.rglob("*") if p.is_file())
        else:
            sources = [home]
        for source in sources:
            try:
                if needle in source.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except OSError:
                continue
    return False


def check(skip_models: bool) -> Findings:
    findings = Findings()

    if not OPS_TD.exists():
        findings.errors.append(
            f"{OPS_TD.relative_to(REPO_ROOT)} does not exist, so there is no "
            "operation list to check. This script activates with the npu "
            "dialect."
        )
        return findings

    operations = parse_operations(OPS_TD.read_text(encoding="utf-8"))
    if not operations:
        findings.errors.append(
            f"no operation definitions were found in "
            f"{OPS_TD.relative_to(REPO_ROOT)}. Either the dialect is empty or "
            "this script's definition pattern no longer matches the source, "
            "and the second is worse than the first because it reports an "
            "empty dialect as compliant."
        )
        return findings

    exemptions = parse_exemptions(
        EXEMPTIONS.read_text(encoding="utf-8") if EXEMPTIONS.exists() else ""
    )
    exempt_pairs = {(entry.op, entry.layer) for entry in exemptions}
    known = {operation.name for operation in operations}

    # ---- Step 1. Classification. -------------------------------------------
    #
    # Every operation carries one, read from the ODS description. This is the
    # part of the check that is fully decidable today, and it is what
    # --skip-models exists to run.
    for operation in operations:
        if operation.classification is None:
            findings.errors.append(
                f"{operation.name} ({operation.record}) carries no "
                "classification. Add a line reading exactly "
                "'Reachability: imported computation.' or "
                "'Reachability: structural.' to its description in "
                f"{OPS_TD.relative_to(REPO_ROOT)}. The classification lives in "
                "the description so that reclassifying an operation cannot be "
                "done without changing the dialect source and the generated "
                "reference."
            )

    # ---- Step 2. Structural rules over the exemption block. ----------------
    #
    # An exemption naming an operation that does not exist, or a layer that is
    # not one of the five, is a stale entry, and a stale entry weakens a law
    # rather than recording history.
    for entry in exemptions:
        if entry.op not in known:
            findings.errors.append(
                f"{EXEMPTIONS.relative_to(REPO_ROOT)} exempts {entry.op}, "
                "which is not an operation of this dialect. An exemption for an "
                "operation that does not exist is a stale entry, and the file's "
                "own rule is that an entry is deleted when its gap closes."
            )
        if entry.layer not in LAYERS:
            findings.errors.append(
                f"{EXEMPTIONS.relative_to(REPO_ROOT)} exempts {entry.op} from "
                f"the layer '{entry.layer}', which is not one of "
                f"{', '.join(LAYERS)}."
            )

    # A structural operation exempted from importability is exempted from a
    # requirement it was never held to, which the exemptions file says
    # explicitly is not an exemption.
    for operation in operations:
        if operation.classification != "structural":
            continue
        for excluded in STRUCTURAL_EXCLUDES:
            if (operation.name, excluded) in exempt_pairs:
                findings.errors.append(
                    f"{operation.name} is structural and is therefore not held "
                    f"to the '{excluded}' layer at all, but "
                    f"{EXEMPTIONS.relative_to(REPO_ROOT)} carries an exemption "
                    "for it. That records as a temporary gap something that is "
                    "a permanent and intended property. Delete the entry."
                )

    # ---- Step 3. The layers themselves. ------------------------------------
    for operation in operations:
        for layer in operation.required_layers:
            if layer == "model" and skip_models:
                continue
            present = layer_present(operation, layer)
            if present is False and (operation.name, layer) not in exempt_pairs:
                findings.missing.setdefault(operation.name, []).append(layer)

    # ---- The notes, which are the honest part. -----------------------------
    imported = [op for op in operations if op.classification == "imported"]
    structural = [op for op in operations if op.classification == "structural"]

    findings.notes.append(
        f"{len(operations)} operations: {len(imported)} imported computation, "
        f"{len(structural)} structural."
    )
    findings.notes.append(
        "structural: " + ", ".join(sorted(op.name for op in structural))
    )
    if skip_models:
        findings.notes.append(
            "--skip-models: the model layer was not checked. This mode runs in "
            "the lint job, before any model is built."
        )
    considered = [layer for layer in LAYERS if layer != "model" or not skip_models]
    decidable = [layer for layer in considered if layer_decidable(layer)]
    undecidable = [layer for layer in considered if not layer_decidable(layer)]

    if decidable:
        findings.notes.append("layers checked: " + ", ".join(decidable) + ".")
    if undecidable:
        findings.notes.append(
            "layers not yet decidable: "
            + ", ".join(undecidable)
            + ". The phase that builds each of these has not landed, so there "
            "is nothing to look in and this run did not check them. Each "
            "becomes decidable on the day its source appears, and the full "
            "check is passable at the phase where the last of them does."
        )
    if not decidable:
        findings.notes.append(
            "what this run did check: that every operation carries a "
            "classification, and the structural rules over the exemption "
            "block. That is what is checkable without the layers themselves, "
            "and it is the whole of what this result claims."
        )
    if exemptions:
        findings.notes.append(f"{len(exemptions)} exemptions in force.")
    else:
        findings.notes.append("no exemptions in force.")

    return findings


def report(findings: Findings, skip_models: bool) -> int:
    mode = "--skip-models" if skip_models else "full"
    print(f"check-reachability ({mode})")
    print()

    for note in findings.notes:
        print(f"  note: {note}")
    print()

    if findings.missing:
        print("  Operations missing a required layer:")
        print()
        width = max(len(name) for name in findings.missing)
        for name in sorted(findings.missing):
            layers = ", ".join(findings.missing[name])
            print(f"    {name:<{width}}  missing: {layers}")
        print()
        print(
            "  Each of these either gains the missing layer or is deleted. The "
            "one alternative is a dated entry in docs/EXEMPTIONS.md naming the "
            "phase that resolves it."
        )
        print()

    for error in findings.errors:
        print(f"  error: {error}")
    if findings.errors:
        print()

    failed = bool(findings.errors or findings.missing)
    print("check-reachability: FAIL" if failed else "check-reachability: pass")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enforces law 2: every operation of the npu dialect representing "
            "imported computation is importable, lowerable, encodable, "
            "simulatable, and exercised by a model."
        )
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help=(
            "skip the model layer, for the lint job, which runs before any "
            "model is built"
        ),
    )
    args = parser.parse_args()
    return report(check(args.skip_models), args.skip_models)


if __name__ == "__main__":
    sys.exit(main())
