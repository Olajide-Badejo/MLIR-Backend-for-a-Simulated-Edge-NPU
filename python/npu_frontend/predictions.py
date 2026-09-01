# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Section 17.8's registered report protocol, in a directory and a parser.

Law 4 of Section 0.2 is honest measurement, and it is the strongest honesty
claim this project makes and was the least enforced of the four. The other three
have mechanisms: the negative test rule, the reachability check, the
traceability tests. This is law 4's, and it is deliberately the same shape as a
registered report [R55]: **the prediction is written down and committed before
the measurement, and the ordering is proved by a test rather than remembered.**

**What the proof is.** A result file names the prediction it is evidence for
through `prediction_id`, and the commit that prediction landed in through
`prediction_sha`. `git merge-base --is-ancestor <prediction_sha>
<manifest.git_sha>` then either holds or does not, and a prediction written after
the measurement cannot make it hold. That is the whole mechanism, and it works
because git already refuses to let a commit claim an ancestor it does not have.

**Why the mechanism lands at P10 and its first user is at P11.** Section 17.8
says it in one sentence: P11's gate requires a prediction to already exist and to
already be an ancestor, and a mechanism that arrives in the same phase as its
first consumer has never been tested when it matters. So the directory, the two
fields, the ancestor test and at least one prediction taken through the whole
path are P10's, and P11 finds a path that is known to work.

**The format is machine parsed on purpose.** A prediction whose direction and
magnitude bracket are only in prose is a prediction that can be reinterpreted
after the fact, which is the exact failure the protocol exists to prevent. The
header block below carries the five things Section 17.8 names, this module
refuses a file missing any of them, and the prose underneath is where the
reasoning goes.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: Where the predictions live. Section 6's tree names this directory.
PREDICTIONS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "experiments" / "predictions"
)

#: The five things Section 17.8 requires of every entry, plus the two this
#: project adds.
#:
#: `id` is what a result's `prediction_id` names, and it is required to equal the
#: file stem so that a reader with an id can find the file without a search.
#: `written` is the date the prose was written, which is not the same as the date
#: the commit landed and is not what the ancestor test uses; the test uses git,
#: because a date in a file is a claim and a commit is a fact.
REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "written",
    "result field",
    "direction",
    "magnitude bracket",
    "answered at",
)

#: The two sections every entry carries. The falsification section is the one
#: that makes the rest a prediction rather than a hope: Section 17.8 asks for
#: "what observation would falsify it", and an entry without one cannot be wrong
#: and therefore cannot be evidence of anything.
REQUIRED_SECTIONS: Final[tuple[str, ...]] = (
    "## Hypothesis",
    "## What would falsify it",
)

_FIELD = re.compile(r"^- \*\*([^:*]+):\*\*\s+(.+?)\s*$", re.MULTILINE)
_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class PredictionError(Exception):
    """A prediction entry is not one this project can read or check."""


@dataclass(frozen=True)
class Prediction:
    """One entry of `experiments/predictions/`."""

    identifier: str
    path: Path
    fields: dict[str, str]
    body: str

    @property
    def result_fields(self) -> list[str]:
        """The result schema fields this prediction is about."""
        return [name.strip("` ") for name in self.fields["result field"].split(",")]

    @property
    def answered_at(self) -> str:
        """The phase whose measurement answers it, as the entry states it."""
        return self.fields["answered at"]


def parse_prediction(path: str | Path) -> Prediction:
    """One entry, or a refusal naming what is missing."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    fields = {key.strip().lower(): value.strip() for key, value in _FIELD.findall(text)}

    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        raise PredictionError(
            f"{source.name} is missing {missing} from its header block. Section "
            f"17.8 requires a hypothesis, a predicted direction, a magnitude "
            f"bracket, the exact result field it concerns, and what observation "
            f"would falsify it. A prediction with no magnitude bracket is one "
            f"that any outcome confirms."
        )
    absent = [name for name in REQUIRED_SECTIONS if name not in text]
    if absent:
        raise PredictionError(f"{source.name} is missing the sections {absent}.")

    identifier = fields["id"]
    if not _ID.match(identifier):
        raise PredictionError(
            f"{source.name} declares the id {identifier!r}, which is not lower "
            f"case words joined by single hyphens. The id goes into a result "
            f"file's prediction_id and into a file name, so it is held to a "
            f"shape both can carry."
        )
    if identifier != source.stem:
        raise PredictionError(
            f"{source.name} declares the id {identifier!r}. The id and the file "
            f"stem have to agree, so that a reader holding a result's "
            f"prediction_id can find the entry without searching for it."
        )
    return Prediction(identifier=identifier, path=source, fields=fields, body=text)


def load_predictions(directory: str | Path | None = None) -> dict[str, Prediction]:
    """Every entry, keyed by id. An unreadable entry raises rather than being skipped."""
    root = Path(directory) if directory is not None else PREDICTIONS_DIR
    entries = sorted(root.glob("*.md"))
    if not entries:
        raise PredictionError(
            f"{root} holds no predictions. The mechanism of Section 17.8 lands "
            f"with at least one real prediction taken through the whole path, "
            f"because a mechanism with no user has never been tested."
        )
    return {
        prediction.identifier: prediction
        for prediction in (parse_prediction(entry) for entry in entries)
    }


# ---------------------------------------------------------------------------
# The ancestor test's own machinery.
# ---------------------------------------------------------------------------


def commit_exists(sha: str, *, repository: str | Path | None = None) -> bool:
    """Whether the sha names a commit in this repository.

    Section 16.1's rule for `manifest.git_sha`: a committed number tracing to a
    commit that does not exist is the exact failure the check prevents.
    """
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(repository or PREDICTIONS_DIR.parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def is_ancestor(
    earlier: str, later: str, *, repository: str | Path | None = None
) -> bool:
    """Whether `earlier` is an ancestor of `later`, as git decides it.

    Not "as a date decides it". A timestamp in a file is a claim the person
    writing the file controls; an ancestry relation is a fact about a graph
    nobody can rewrite without rewriting every sha downstream of it. That
    difference is the whole reason Section 17.8 specifies this test rather than
    a `written` field comparison.
    """
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", earlier, later],
        cwd=str(repository or PREDICTIONS_DIR.parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def head_sha(*, repository: str | Path | None = None) -> str:
    """The commit a result recorded now would carry in its manifest."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository or PREDICTIONS_DIR.parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def landing_sha(identifier: str, *, repository: str | Path | None = None) -> str | None:
    """The commit that added the entry, which is what a result records.

    Read from git rather than written into the file, and that is the point:
    an entry cannot state the sha of the commit that contains it, so a `sha`
    field in the header would have to be filled in by a later commit and would
    then be a claim rather than a fact.
    """
    root = Path(repository) if repository is not None else PREDICTIONS_DIR.parents[1]
    completed = subprocess.run(
        [
            "git",
            "log",
            "--format=%H",
            "--diff-filter=A",
            "--",
            f"experiments/predictions/{identifier}.md",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line]
    return lines[-1] if lines else None
