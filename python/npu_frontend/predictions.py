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


def _git(
    arguments: list[str],
    *,
    repository: str | Path | None = None,
    answers: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """Runs one git command and refuses to read a fatal as an answer.

    **This is D-0042, and it is the second time this project folded a nonzero
    exit into a boolean.** git distinguishes the two cases and this module did
    not. `git cat-file -e` exits **1** when the object is genuinely absent and
    **128** when it could not look; `git merge-base --is-ancestor` exits 1 for
    "no" and 128 for the same reason. Reading any nonzero as the answer turns "I
    could not ask" into "the answer is no", which is the fault Section 16.1
    forbids for result fields, appearing in a git query instead of in a field.

    `answers` is the set of exit codes that are genuinely answers. Anything else
    raises, with git's own stderr in the message, because git explains itself
    better than a paraphrase and its message carries the fix.

    The case that produced this: a workspace owned by one uid with a container
    running as another, where git refuses the repository as dubiously owned until
    `safe.directory` names it. Every call below then exits 128, and
    `rev-parse --is-shallow-repository` additionally prints its fatal to stdout,
    so a caller comparing stdout against `"true"` concluded the repository was
    not shallow and carried on.
    """
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(repository or PREDICTIONS_DIR.parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode in answers:
        return completed
    raise PredictionError(
        f"`git {' '.join(arguments)}` exited {completed.returncode}, which is "
        f"not an answer to the question it was asked. Exit codes "
        f"{list(answers)} are answers here; anything else means git could not "
        f"look, and reading it as an answer would turn a question this project "
        f"could not ask into a finding it did not make.\n\n"
        f"git said:\n{completed.stderr.strip() or '(nothing on stderr)'}"
    )


def repository_is_shallow(*, repository: str | Path | None = None) -> bool:
    """Whether this checkout has had its history truncated.

    *Added at P10 after D-0041.* `actions/checkout` defaults to `fetch-depth: 1`,
    so a CI checkout holds exactly one commit and every question about history
    has no answer available. Asking git directly is the only reliable test: a
    commit count, the presence of `.git/shallow` and the resolvability of
    `HEAD~1` each look normal in some shape of truncated checkout and not in
    others, and the `pull_request` trigger's synthetic merge commit is one of the
    shapes that confuses all three.

    **Goes through `_git`, which is D-0042.** This function used to compare
    stdout against `"true"` and treat everything else as False. Under a dubious
    ownership refusal git exits 128 and prints its fatal, so the comparison was
    False and this reported a repository it could not read at all as one with
    full history, which let `require_full_history` pass and produced a
    downstream message blaming the shas.
    """
    return (
        _git(
            ["rev-parse", "--is-shallow-repository"], repository=repository
        ).stdout.strip()
        == "true"
    )


def require_full_history(what: str, *, repository: str | Path | None = None) -> None:
    """Refuses, once and readably, when the answer needs history there is none of.

    **This exists because the alternative was eight wrong answers**, which is
    D-0041. Every function here can be asked something a shallow checkout cannot
    answer, and each of them answered anyway: `commit_exists` returned False for
    a commit that exists and is merely not present, `is_ancestor` returned False
    for "cannot tell", and `landing_sha` returned the graft commit, which is a
    plausible looking sha that is not the answer. Those are three spellings of
    the fault this project forbids everywhere else, an absent measurement that
    cannot be told apart from a measured one.

    So the question is refused before it is asked, with the reason and the fix in
    the message, because the reader of a red log needs to know that the checkout
    is the problem rather than the sha.
    """
    if not repository_is_shallow(repository=repository):
        return
    raise PredictionError(
        f"{what} needs this repository's history, and this checkout is shallow. "
        f"git reports it as a shallow repository, so the commits the question is "
        f"about are not here to be found.\n\n"
        f"This is not a fault in the shas being asked about. In CI it means the "
        f"job's checkout is at the `actions/checkout` default of "
        f"`fetch-depth: 1`; the jobs that run this suite set `fetch-depth: 0` "
        f"for exactly this reason, and `docs/DEFECT_LOG.md` D-0041 records why. "
        f"Locally it means a `git clone --depth`, and `git fetch --unshallow` "
        f"fixes it.\n\n"
        f"Law 3 of Section 0.2 is traceability and law 4's mechanism is an "
        f"ancestry relation. Neither can be checked against history that was "
        f"never fetched, and reporting them as violated rather than as "
        f"uncheckable would be this project asserting something it cannot see."
    )


def commit_exists(sha: str, *, repository: str | Path | None = None) -> bool:
    """Whether the sha names a commit **in this checkout**.

    Section 16.1's rule for `manifest.git_sha`: a committed number tracing to a
    commit that does not exist is the exact failure the check prevents.

    **False means "not here", which is not the same as "does not exist".** In a
    shallow checkout every historical sha is absent and this returns False for
    all of them. A caller that means "does this commit exist at all" calls
    `require_full_history` first, and every caller in this project does.

    **Exit 1 is absence and exit 128 is a refusal to look**, which is D-0042.
    Only the first is an answer; the second raises through `_git`.

    **`rev-parse --verify --quiet` rather than `cat-file -e`, and the difference
    is the whole point.** Measured on 2026-09-02:

    ```
    cat-file  -e <present>^{commit}   exit 0
    cat-file  -e <absent>^{commit}    exit 128  fatal: Not a valid object name
    cat-file  -e <present>^{commit}   exit 128  fatal: dubious ownership
    rev-parse --verify -q <present>^{commit}   exit 0
    rev-parse --verify -q <absent>^{commit}    exit 1
    rev-parse --verify -q <present>^{commit}   exit 128  in an unreadable repository
    ```

    `cat-file -e` with a `^{commit}` peel returns **128 for an absent object as
    well as for an unreadable repository**, so it cannot tell the two apart at
    all and the distinction this function exists to make is not available through
    it. That is why the first attempt at D-0042's fix still could not separate
    them: the exit codes were being read correctly and the probe was wrong.

    `rev-parse --verify --quiet` also returns 1 for a sha that resolves to
    something that is not a commit, a blob for instance, which is the right
    answer to "is this a commit" and one more thing `cat-file -e <sha>` would
    have said yes to.
    """
    return (
        _git(
            ["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
            repository=repository,
            answers=(0, 1),
        ).returncode
        == 0
    )


def is_ancestor(
    earlier: str, later: str, *, repository: str | Path | None = None
) -> bool:
    """Whether `earlier` is an ancestor of `later`, as git decides it.

    Not "as a date decides it". A timestamp in a file is a claim the person
    writing the file controls; an ancestry relation is a fact about a graph
    nobody can rewrite without rewriting every sha downstream of it. That
    difference is the whole reason Section 17.8 specifies this test rather than
    a `written` field comparison.

    **A reference that does not resolve raises rather than returning False**,
    which is D-0041. `git merge-base --is-ancestor` exits nonzero both for "no"
    and for "I have never heard of that commit", and collapsing the two made a
    shallow checkout report the ancestor relation as violated when it was merely
    unobservable. Those want opposite responses: one is a prediction written
    after its measurement, which is a serious finding, and the other is a
    checkout option.
    """
    for reference in (earlier, later):
        if not commit_exists(reference, repository=repository):
            require_full_history(
                f"deciding whether {earlier[:12]} precedes {later[:12]}",
                repository=repository,
            )
            raise PredictionError(
                f"{reference} is not a commit in this repository, and the "
                f"repository is not shallow, so it is genuinely absent rather "
                f"than merely unfetched. An ancestry relation cannot be decided "
                f"against a commit that does not exist."
            )
    return (
        _git(
            ["merge-base", "--is-ancestor", earlier, later],
            repository=repository,
            answers=(0, 1),
        ).returncode
        == 0
    )


def head_sha(*, repository: str | Path | None = None) -> str:
    """The commit a result recorded now would carry in its manifest.

    Raises rather than returning an empty string when git cannot answer, D-0042.
    An empty sha flowing into a comparison is a comparison that quietly means
    something else.
    """
    return _git(["rev-parse", "HEAD"], repository=repository).stdout.strip()


def landing_sha(identifier: str, *, repository: str | Path | None = None) -> str | None:
    """The commit that added the entry, which is what a result records.

    Read from git rather than written into the file, and that is the point:
    an entry cannot state the sha of the commit that contains it, so a `sha`
    field in the header would have to be filled in by a later commit and would
    then be a claim rather than a fact.

    **Refuses in a shallow checkout rather than answering wrongly**, which is the
    half of D-0041 that no checkout option fixes. `git log --diff-filter=A`
    against a truncated history attributes every file to the graft commit,
    because that commit has no parent to have differed from, so this returned the
    checkout's own tip and it looked exactly like a real answer. A result then
    naming that sha as `prediction_sha` would satisfy the ancestor test while
    recording a provenance link to the wrong commit, which is worse than the
    failure it replaced. Measured on a `pull_request` merge ref at depth 1: this
    returned the merge commit for an entry that landed six commits earlier.
    """
    require_full_history(
        f"finding the commit that added the prediction {identifier}",
        repository=repository,
    )
    completed = _git(
        [
            "log",
            "--format=%H",
            "--diff-filter=A",
            "--",
            f"experiments/predictions/{identifier}.md",
        ],
        repository=repository,
    )
    lines = [line for line in completed.stdout.splitlines() if line]
    return lines[-1] if lines else None
