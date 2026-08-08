"""Fail if an op in the npu dialect cannot be reached end to end.

UPGRADE_SPEC_V3.md section 5.2 states the rule this enforces:

    No operator may exist in the npu dialect unless it is reachable end to end:
    importable from ONNX, lowerable to npuisa, encodable, simulatable, and
    exercised by at least one model in the benchmark suite. An operator that
    cannot satisfy this must be deleted from the dialect or must have a tracked,
    dated exemption in docs/DESIGN_DECISIONS.md stating when it will be
    completed.

The rule exists because three ops (transpose, concat, batch_norm) were defined
in ODS with verifiers and round trip tests, and had no conversion pattern at
all, so any graph containing one failed to compile. That stayed invisible for
twelve phases because LeNet contains none of them. A check that runs on every
push would have caught it the day it was introduced.

Layers checked per op, in pipeline order:

  importer   a converter in python/npu_frontend/op_mapping.py builds the op
  lowering   a conversion pattern in LowerNPUToNPUISA.cpp consumes it
  encoder    InstructionEncoder.cpp turns its npuisa form into an Instruction
  simulator  Simulator.cpp has a case for the resulting opcode
  model      the op appears in the imported IR of a benchmark suite model

The last one needs torch and onnxruntime, so it is skipped with a warning when
they are unavailable rather than reported as a failure. Everything else is
static analysis of the sources and needs no build.

Exit codes: 0 all reachable or exempt, 1 a gap with no exemption, 2 the checker
itself could not run (a source file moved, or an op has no NPUISA_EQUIVALENT
entry).
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

OPS_TD = REPO / "include" / "NPU" / "Dialect" / "NPU" / "IR" / "NPUOps.td"
LOWERING = REPO / "lib" / "Dialect" / "NPUISA" / "Transforms" / "LowerNPUToNPUISA.cpp"
ENCODER = REPO / "lib" / "Encoding" / "InstructionEncoder.cpp"
SIMULATOR = REPO / "lib" / "Simulator" / "Simulator.cpp"
IMPORTER = REPO / "python" / "npu_frontend" / "op_mapping.py"
DECISIONS = REPO / "docs" / "DESIGN_DECISIONS.md"

EXEMPT_BEGIN = "<!-- REACHABILITY-EXEMPT-BEGIN -->"
EXEMPT_END = "<!-- REACHABILITY-EXEMPT-END -->"

# How each npu op is represented once it is past the tensor level. The npuisa op
# class name, and the Opcode the simulator switches on.
#
# This table is hand maintained on purpose, and the checker fails if an op in
# NPUOps.td is missing from it. Adding an op to the dialect should force a
# conscious answer to "what does this become in the ISA", rather than the
# checker silently inferring something and passing.
#
# None means the op has no instruction level form yet, which is itself a gap.
NPUISA_EQUIVALENT: dict[str, tuple[str | None, str | None]] = {
    # npu op mnemonic:  (npuisa op class, Opcode)
    "constant": ("ConstOp", None),  # data in the .nbin, not an instruction
    "relu": ("ReluOp", "Relu"),
    "add": ("AddOp", "Add"),
    "mul": ("MulOp", "Mul"),
    "conv2d": ("Conv2DOp", "Conv2D"),
    "matmul": ("MatMulOp", "MatMul"),
    "max_pool2d": ("PoolMaxOp", "PoolMax"),
    "avg_pool2d": ("PoolAvgOp", "PoolAvg"),
    "reshape": ("ReshapeOp", "Reshape"),
    "transpose": (None, None),
    "concat": (None, None),
    "batch_norm": (None, None),
}

LAYERS = ["importer", "lowering", "encoder", "simulator", "model"]


class CheckerError(RuntimeError):
    """The checker could not run, as distinct from finding a gap."""


def parse_ops() -> list[tuple[str, str]]:
    """Return (cpp class name, mnemonic) for every op defined in NPUOps.td.

    Matches both the direct form,

        def NPU_Conv2DOp : NPU_Op<"conv2d", [Pure]> {

    and ops built through a helper class,

        def NPU_MaxPool2DOp : NPU_PoolOp<"max_pool2d"> {
    """
    text = OPS_TD.read_text()
    pattern = re.compile(r'def\s+NPU_(\w+)Op\s*:\s*NPU_\w+?<\s*"([a-z_0-9]+)"')
    ops = [(m.group(1), m.group(2)) for m in pattern.finditer(text)]
    if not ops:
        raise CheckerError(f"no ops parsed out of {OPS_TD}; has the syntax changed?")
    return ops


def parse_exemptions() -> dict[str, tuple[str, str]]:
    """Return mnemonic -> (date, reason) from the exemption block.

    An exemption needs a date. "We will get to it" is not a plan, and the whole
    point of the block is that it stays uncomfortable to look at.
    """
    if not DECISIONS.exists():
        raise CheckerError(f"{DECISIONS} not found")
    text = DECISIONS.read_text()
    if EXEMPT_BEGIN not in text:
        return {}
    body = text.split(EXEMPT_BEGIN, 1)[1].split(EXEMPT_END, 1)[0]

    exemptions: dict[str, tuple[str, str]] = {}
    row = re.compile(
        r"^\|\s*`npu\.([a-z_0-9]+)`\s*\|(.+?)\|\s*(\d{4}-\d{2}-\d{2})\s*\|"
    )
    for line in body.splitlines():
        m = row.match(line.strip())
        if m:
            exemptions[m.group(1)] = (m.group(3), m.group(2).strip())
    return exemptions


def reachability(op_class: str, mnemonic: str, model_ops: set[str] | None) -> dict:
    """Return layer -> True (present), False (missing), or None (not checked)."""
    if mnemonic not in NPUISA_EQUIVALENT:
        raise CheckerError(
            f"npu.{mnemonic} is defined in NPUOps.td but has no entry in "
            "NPUISA_EQUIVALENT in this script. Add one, choosing deliberately "
            "what it lowers to, or (None, None) if nothing yet."
        )
    isa_class, opcode = NPUISA_EQUIVALENT[mnemonic]

    found = {}
    found["importer"] = f'"npu.{mnemonic}"' in IMPORTER.read_text()
    found["lowering"] = f"npu::{op_class}Op" in LOWERING.read_text()
    found["encoder"] = bool(isa_class) and f"npuisa::{isa_class}" in ENCODER.read_text()

    if opcode:
        found["simulator"] = f"Opcode::{opcode}" in SIMULATOR.read_text()
    elif mnemonic == "constant":
        # Constants are copied into DRAM before the instruction loop rather than
        # executed, so there is no opcode to switch on.
        found["simulator"] = "program.constants" in SIMULATOR.read_text()
    else:
        found["simulator"] = False

    found["model"] = None if model_ops is None else (mnemonic in model_ops)
    return found


def ops_used_by_models() -> set[str]:
    """Every npu op appearing in the imported IR of every benchmark model."""
    sys.path.insert(0, str(REPO / "python"))
    from npu_frontend import compile as npu_compile
    from npu_frontend import model_generator

    used: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(model_generator.MODELS):
            onnx_path = model_generator.export(name, Path(tmp) / f"{name}.onnx")
            text = npu_compile.compile_model(onnx_path, opt_level=0, emit="import")
            used.update(re.findall(r"\bnpu\.([a-z_0-9]+)", str(text)))
    return used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="skip the benchmark model check, which needs torch and onnxruntime",
    )
    args = parser.parse_args(argv)

    try:
        ops = parse_ops()
        exemptions = parse_exemptions()
    except CheckerError as exc:
        print(f"check-reachability: {exc}", file=sys.stderr)
        return 2

    model_ops: set[str] | None = None
    if args.skip_models:
        print("note: skipping the benchmark model layer (--skip-models)\n")
    else:
        try:
            model_ops = ops_used_by_models()
        except Exception as exc:
            print(
                f"note: skipping the benchmark model layer, could not import or "
                f"run the frontend ({exc.__class__.__name__}: {exc})\n"
            )

    try:
        table = {m: reachability(c, m, model_ops) for c, m in ops}
    except CheckerError as exc:
        print(f"check-reachability: {exc}", file=sys.stderr)
        return 2

    width = max(len(m) for m in table) + 6
    header = "op".ljust(width) + "".join(layer.ljust(11) for layer in LAYERS)
    print(header)
    print("-" * len(header))

    unreachable: dict[str, list[str]] = {}
    for mnemonic, found in sorted(table.items()):
        cells = []
        missing = []
        for layer in LAYERS:
            state = found[layer]
            if state is None:
                cells.append("skip")
            elif state:
                cells.append("ok")
            else:
                cells.append("MISSING")
                missing.append(layer)
        row = f"npu.{mnemonic}".ljust(width) + "".join(c.ljust(11) for c in cells)
        if missing:
            exempt = exemptions.get(mnemonic)
            row += f"  exempt until {exempt[0]}" if exempt else "  UNEXEMPTED"
            unreachable[mnemonic] = missing
        print(row)

    print()
    failures = []
    today = date.today().isoformat()
    for mnemonic, missing in sorted(unreachable.items()):
        exempt = exemptions.get(mnemonic)
        if exempt is None:
            failures.append(
                f"npu.{mnemonic} is unreachable ({', '.join(missing)}) and has no "
                f"exemption. Either implement the missing layers, delete the op "
                f"from the dialect, or add a dated row to the "
                f"{EXEMPT_BEGIN} block in docs/DESIGN_DECISIONS.md."
            )
        elif exempt[0] < today:
            failures.append(
                f"npu.{mnemonic} is unreachable ({', '.join(missing)}) and its "
                f"exemption expired on {exempt[0]}. Implement it, delete it, or "
                f"move the date and say why."
            )

    # Only worth saying when every layer was actually checked. With the model
    # layer skipped, an op whose sole remaining gap is "not exercised by any
    # model" looks fully reachable, and telling someone to delete a still valid
    # exemption would be worse than staying quiet.
    if model_ops is not None:
        for mnemonic in sorted(set(exemptions) - set(unreachable)):
            print(
                f"note: npu.{mnemonic} is exempt but is now fully reachable. "
                "Remove its row from the exemption block."
            )

    if failures:
        print()
        for message in failures:
            print(f"FAIL: {message}")
        print(f"\ncheck-reachability: {len(failures)} unreachable op(s)")
        return 1

    exempt_count = len(unreachable)
    if exempt_count:
        print(
            f"check-reachability: all ops reachable or exempt "
            f"({exempt_count} exempt, all with unexpired dates)"
        )
    else:
        print("check-reachability: every op is reachable end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
