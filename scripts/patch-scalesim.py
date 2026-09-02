#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Make the pinned SCALE-Sim v3 run under numpy 2. See D-0044.

    python scripts/patch-scalesim.py            # apply, idempotently
    python scripts/patch-scalesim.py --check    # report without writing

**Why this exists.** SCALE-Sim v3 at the sha this project pins does not run under
numpy 2, on any upstream branch, on its own shipped example. Three expressions
convert a numpy array to a Python integer with the builtin `int`, which numpy 2
removed for arrays of rank one or more:

    TypeError: only 0-dimensional arrays can be converted to Python scalars

`~/npu-venv` has numpy 2.5.1, pinned in `requirements-lock.txt` and carried by
every recorded result in `experiments/results/`. Moving it to satisfy an external
tool would invalidate every number this project has measured, which is a far
larger cost than the one this script pays. Ubuntu 26.04 ships CPython 3.14 only
and no numpy 1.x wheel exists for it, so a second interpreter was not available
either. `docs/DEFECT_LOG.md` D-0044 carries the whole decision.

**What it changes, and what it does not.** Each edit replaces `int(max(x))` with
`int(np.max(np.asarray(x)))`, which is the same value under numpy 1 and is the
same value the author's own commented out line above one of them takes. It
changes no cycle arithmetic, no dataflow, and no report. `numpy` is already
imported in both files.

**How a reader knows what ran.** The exporter records the sha256 of the installed
SCALE-Sim tree in every result manifest, beside the upstream git sha. So the
manifest says both which commit the tool came from and that the tree that ran was
not byte identical to it, rather than showing a sha that does not describe the
code that produced the number.

This script is deliberately not run automatically by anything. Modifying an
installed dependency is a decision, and a decision that happens as a side effect
of running a benchmark is a decision nobody made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Each entry is one file and the exact expression to replace in it. Exact rather
#: than a regular expression, because a pattern that matched more than these
#: three would be a patch whose extent nobody had checked.
EDITS: list[tuple[str, str, str]] = [
    (
        "memory/read_buffer.py",
        "self.last_prefetch_cycle = int(max(response_cycles_arr))",
        "self.last_prefetch_cycle = int(np.max(np.asarray(response_cycles_arr)))",
    ),
    (
        "memory/double_buffered_scratchpad_mem.py",
        "self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], "
        "ofmap_stalls[0]))",
        "self.stall_cycles += int(np.max(np.asarray([ifmap_stalls[0], "
        "filter_stalls[0], ofmap_stalls[0]])))",
    ),
    (
        "memory/double_buffered_scratchpad_mem.py",
        "self.total_cycles = int(max(ofmap_serviced_cycles))",
        "self.total_cycles = int(np.max(np.asarray(ofmap_serviced_cycles)))",
    ),
]


def installed_root() -> Path:
    try:
        import scalesim  # type: ignore[import-untyped]
    except ImportError:
        print(
            "scalesim is not installed in this interpreter. Install it from the "
            "pinned sha recorded in docs/adr/0003-resolved-tool-matrix.md first.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    return Path(scalesim.__file__).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patch-scalesim.py")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether each edit is applied, and change nothing",
    )
    arguments = parser.parse_args(argv)

    root = installed_root()
    print(f"scalesim: {root}")

    pending = 0
    missing = 0
    for relative, old, new in EDITS:
        path = root / relative
        if not path.is_file():
            print(f"  MISSING FILE {relative}")
            missing += 1
            continue
        text = path.read_text(encoding="utf-8")
        if new in text:
            print(f"  applied      {relative}: {old[:48]}...")
            continue
        if old not in text:
            # Neither form is present, which means upstream changed the line.
            # That is a reason to look rather than to write something into a file
            # whose contents this script no longer recognises.
            print(
                f"  UNRECOGNISED {relative}: neither the original nor the "
                f"patched form of {old[:48]}... is in this file. Upstream moved; "
                f"do not force this."
            )
            missing += 1
            continue
        pending += 1
        if arguments.check:
            print(f"  needs patch  {relative}: {old[:48]}...")
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"  patched      {relative}: {old[:48]}...")

    if missing:
        print(
            f"\n{missing} edit(s) could not be placed. See docs/DEFECT_LOG.md "
            f"D-0044 before changing anything here.",
            file=sys.stderr,
        )
        return 2
    if arguments.check and pending:
        print(f"\n{pending} edit(s) not applied.", file=sys.stderr)
        return 1
    print("\nscalesim: every edit is in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
