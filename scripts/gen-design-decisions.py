#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Generates docs/DESIGN_DECISIONS.md as an index over docs/adr/NNNN-*.md.
#
# Section 20 of the build specification makes DESIGN_DECISIONS.md a generated
# index rather than a hand maintained file, because a flat file cannot express
# supersession and supersession is what the prime directive is about. The
# records are the source of truth; this script only reads them.
#
#   python scripts/gen-design-decisions.py           write the index
#   python scripts/gen-design-decisions.py --check   fail if it is stale
#
# The output is deterministic: records are ordered by their number, and nothing
# in the generated text depends on the filesystem's iteration order or on the
# clock. Two runs on the same input produce the same bytes, which is what lets
# --check be a staleness gate rather than a coin toss.

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_DIR = REPO_ROOT / "docs" / "adr"
INDEX_PATH = REPO_ROOT / "docs" / "DESIGN_DECISIONS.md"

# NNNN-short-title.md, four digits, per Section 20.
FILENAME_RE = re.compile(r"^(\d{4})-(.+)\.md$")

# The first ATX heading of a record. The number is repeated there in the
# `1. Title` form, and it is stripped so the index prints the title once.
TITLE_RE = re.compile(r"^#\s+(?:\d+\.\s*)?(.+?)\s*$")

# The status and date bullets of the standard form. Bold markers are allowed
# either side of the colon so a record may write `**Status:**` or `Status:`.
FIELD_RE = re.compile(
    r"^\s*[-*]\s*\**(?P<key>Status|Date)\**\s*:\s*\**(?P<value>.+?)\**\s*$",
    re.IGNORECASE,
)

HEADER = """<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

<!--
GENERATED FILE. DO NOT EDIT BY HAND.

This index is generated from the architecture decision records in docs/adr/.
Every edit belongs in the record itself; an edit made here is lost the next
time the generator runs. Regenerate with:

    python scripts/gen-design-decisions.py

and check it for staleness with:

    python scripts/gen-design-decisions.py --check
-->

# Design decisions

*Diataxis type: explanation (index).*

This is the generated index over the architecture decision records in
[`docs/adr/`](adr/). Each record is written once, in the standard form of
Section 20 of the build specification: title, status, context, decision,
consequences. A decision that changed is not edited in place; the old record
is marked superseded and a new one is written, so the reasoning that was true
at the time stays readable.

The reachability `EXEMPT` block is **not** here. It lives in
[`EXEMPTIONS.md`](EXEMPTIONS.md), because an exemption list is live, machine
parsed and frequently edited, and a generated index is none of those.
"""

TABLE_HEADER = """
| Number | Title | Status | Date |
|---|---|---|---|
"""

EMPTY_NOTE = """
No architecture decision records exist yet.
"""


@dataclass(frozen=True)
class Record:
    """One architecture decision record, as parsed from its file."""

    number: int
    slug: str
    filename: str
    title: str
    status: str
    date: str

    @property
    def sort_key(self) -> tuple[int, str]:
        # The number orders the index. The slug only breaks a tie between two
        # files claiming the same number, which is a mistake rather than a
        # supported arrangement, but it keeps the output deterministic anyway.
        return (self.number, self.slug)


def parse_record(path: Path) -> Record:
    """Read one record file and pull out its number, title, status and date."""
    match = FILENAME_RE.match(path.name)
    if match is None:  # pragma: no cover - excluded by the caller's glob
        raise ValueError(f"{path.name}: not in the NNNN-short-title.md form")

    number = int(match.group(1))
    slug = match.group(2)

    title = ""
    status = ""
    date = ""

    for line in path.read_text(encoding="utf-8").splitlines():
        if not title:
            heading = TITLE_RE.match(line)
            if heading is not None:
                title = heading.group(1)
                continue
        field = FIELD_RE.match(line)
        if field is not None:
            key = field.group("key").lower()
            value = field.group("value").strip()
            if key == "status" and not status:
                status = value
            elif key == "date" and not date:
                date = value

    missing = [
        name
        for name, value in (("title", title), ("status", status), ("date", date))
        if not value
    ]
    if missing:
        raise ValueError(f"{path.name}: no {', '.join(missing)} found in the record")

    return Record(
        number=number,
        slug=slug,
        filename=path.name,
        title=title,
        status=status,
        date=date,
    )


def collect(adr_dir: Path) -> list[Record]:
    """Parse every record in adr_dir, ordered by number."""
    records = [
        parse_record(path) for path in sorted(adr_dir.glob("[0-9][0-9][0-9][0-9]-*.md"))
    ]
    return sorted(records, key=lambda record: record.sort_key)


def render(records: list[Record]) -> str:
    """Render the whole index. Pure: same records in, same bytes out."""
    if not records:
        return HEADER + EMPTY_NOTE

    rows = "".join(
        f"| {record.number:04d} "
        f"| [{record.title}](adr/{record.filename}) "
        f"| {record.status} "
        f"| {record.date} |\n"
        for record in records
    )
    return HEADER + TABLE_HEADER + rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero if the committed index differs from the generated one",
    )
    args = parser.parse_args(argv)

    if not ADR_DIR.is_dir():
        print(f"gen-design-decisions: no such directory: {ADR_DIR}", file=sys.stderr)
        return 1

    try:
        rendered = render(collect(ADR_DIR))
    except ValueError as exc:
        print(f"gen-design-decisions: {exc}", file=sys.stderr)
        return 1

    if args.check:
        current = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if current != rendered:
            print(
                "gen-design-decisions: docs/DESIGN_DECISIONS.md is stale. "
                "Run: python scripts/gen-design-decisions.py",
                file=sys.stderr,
            )
            return 1
        print(f"gen-design-decisions: index is up to date ({INDEX_PATH})")
        return 0

    INDEX_PATH.write_text(rendered, encoding="utf-8")
    print(f"gen-design-decisions: wrote {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
