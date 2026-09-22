# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The freshness stamp of `experiments/models/`.

`scripts/build-model-ir.py` writes the stamp and `scripts/check-reachability.py`
reads it, so that the model layer of law 2 cannot be answered from an artifact
built before the change under test.

## Why this exists

`experiments/models/` is a gitignored build artifact. Nothing in the repository
records which sources it was built from, so a check that reads it reports on
whatever happens to be on disk. CI hides that by running the sweep and the check
in one step, which makes the order a habit of one workflow file rather than a
property of the tools: run the check alone on a developer machine after changing
the frontend and it answers from the previous build, silently and green. That is
how D-0067 reached CI, and D-0040 and D-0063 are the same family of defect, an
answer computed from an input nobody checked was current.

## What the stamp holds

A sha256 over the sources that decide the artifact's content, in the same
construction `npu_frontend.results.content_hash` uses for a result cell: the
file list is sorted, and each file contributes its repository relative path as
well as its bytes, so a rename with identical content changes the fingerprint.

The input set is not the same as a result cell's, and the difference is the
point. A result cell is stale when the compiler or the cost model moves. The
model IR is stale when the compiler moves, when the frontend that drives it
moves, when a calibration profile moves, because the quantized compilation reads
one, or when the sweep script itself moves, because it decides which models at
which batch sizes and which levels are written at all. The cost model constants
are in the result cell's hash and not in this one: they charge a compiled
program and do not change a line of the IR.

## Why a content hash and not a timestamp

An mtime comparison answers a question close to the one that matters but not the
same one. `git checkout` rewrites mtimes on files whose content it did not
change, which would call a current artifact stale, and a file edited back to its
original bytes would count as a change that produced no difference. The content
hash answers exactly the question asked, which is whether this artifact was
built from the sources in the tree now, and it is the vocabulary this project
already uses for staleness.

## Why the standard library alone

`check-reachability.py` runs in the lint job, which has no MLIR bindings, so it
cannot import `npu_frontend` even to reach one function. This module is imported
by both scripts and needs nothing beyond the standard library.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

#: The stamp lives inside the artifact it describes, so that deleting the
#: artifact deletes the claim that it is current. `build-model-ir.py` clears the
#: directory before a full sweep, which takes the previous stamp with it.
STAMP_NAME: Final[str] = "provenance.json"

#: Bumped when the covered set below changes. A stamp written under an older
#: version is refused rather than compared, because comparing two fingerprints
#: taken over different inputs answers nothing.
STAMP_VERSION: Final[int] = 1

#: The suffixes that count inside a source tree. The same tuple
#: `npu_frontend.results.CONTENT_HASH_SUFFIXES` uses, and a test pins the two
#: equal: a `.pyc` or an editor backup must not mark the artifact stale.
CODE_SUFFIXES: Final[tuple[str, ...]] = (".cpp", ".h", ".td", ".py", ".inc")

#: The trees whose contents decide what the sweep writes, each with the suffixes
#: that count inside it. The first three are
#: `npu_frontend.results.CONTENT_HASH_SOURCES`; the fourth is the calibration
#: profiles, which only this artifact depends on.
SOURCE_TREES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("lib", CODE_SUFFIXES),
    ("include", CODE_SUFFIXES),
    ("python/npu_frontend", CODE_SUFFIXES),
    ("experiments/calibration", (".json",)),
)

#: The sweep script itself. It decides which models, which batch sizes and which
#: levels are written, so a change to it changes the artifact without touching
#: any tree above.
SOURCE_FILES: Final[tuple[str, ...]] = ("scripts/build-model-ir.py",)


def covered_files(repo_root: Path) -> list[Path]:
    """Every file the fingerprint is taken over, sorted, absolute.

    A tree that does not exist contributes nothing rather than raising, which is
    what lets this run at a phase before `experiments/calibration/` is written.
    A missing input is not the same as a stale artifact and is not reported as
    one.
    """
    covered: list[Path] = []
    for relative, suffixes in SOURCE_TREES:
        root = repo_root / relative
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if "__pycache__" in path.parts:
                continue
            covered.append(path)
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if path.is_file():
            covered.append(path)
    return sorted(covered)


def fingerprint(repo_root: Path) -> str:
    """The sha256 of the sources the model IR is built from."""
    digest = hashlib.sha256()
    digest.update(f"model-ir-provenance v{STAMP_VERSION}".encode())
    for path in covered_files(repo_root):
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stamp_path(models_dir: Path) -> Path:
    """Where the stamp of one artifact directory lives."""
    return models_dir / STAMP_NAME


def write_stamp(models_dir: Path, repo_root: Path) -> Path:
    """Records what the sweep just built from, atomically through a rename.

    Called only after a full sweep. A partial rebuild calls `clear_stamp`
    instead: a stamp written after `--model lenet` would certify six models that
    nobody rebuilt.
    """
    target = stamp_path(models_dir)
    document: dict[str, Any] = {
        "stamp_version": STAMP_VERSION,
        "fingerprint": fingerprint(repo_root),
        "files_covered": len(covered_files(repo_root)),
        "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return target


def clear_stamp(models_dir: Path) -> None:
    """Removes the stamp, so that a partial rebuild certifies nothing."""
    stamp_path(models_dir).unlink(missing_ok=True)


def read_stamp(models_dir: Path) -> dict[str, Any] | None:
    """The stamp, or None when there is none to read or it will not parse.

    An unreadable stamp is treated as an absent one rather than raising. Both
    mean the artifact cannot be shown to be current, and the caller says so in
    words either way.
    """
    source = stamp_path(models_dir)
    if not source.is_file():
        return None
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def staleness(models_dir: Path, repo_root: Path) -> str | None:
    """None when the artifact matches the tree, or the reason it does not.

    The reason is the whole message a caller prints. It names the command that
    fixes it, because that is the first thing a reader of a red gate needs.
    """
    rebuild = "Run `python scripts/build-model-ir.py` and check again."
    shown = models_dir.name
    try:
        shown = models_dir.relative_to(repo_root).as_posix()
    except ValueError:
        pass

    stamp = read_stamp(models_dir)
    if stamp is None:
        return (
            f"{shown}/ carries no readable {STAMP_NAME}, so there is no record "
            "of which sources it was built from and no way to tell a current "
            "artifact from one built before the change under test. The "
            f"directory is gitignored, so nothing else notices. {rebuild}"
        )

    version = stamp.get("stamp_version")
    if version != STAMP_VERSION:
        return (
            f"{shown}/{STAMP_NAME} is stamp version {version!r} and this check "
            f"writes version {STAMP_VERSION}. The two fingerprints are taken "
            "over different inputs, so comparing them would answer nothing. "
            f"{rebuild}"
        )

    recorded = stamp.get("fingerprint")
    current = fingerprint(repo_root)
    if recorded != current:
        written = stamp.get("written_utc", "an unrecorded time")
        return (
            f"{shown}/ was built at {written} from sources whose fingerprint "
            f"was {str(recorded)[:12]}, and the tree's fingerprint is now "
            f"{current[:12]}. The model layer would be answered from an "
            "artifact that predates the change under test, which is the defect "
            f"D-0067 was. {rebuild}"
        )
    return None
