# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Law 4's mechanism, and the ancestor test that is the whole of it.

Section 0.2's fourth law is honest measurement. Section 17.8 says it is the
strongest honesty claim this document makes and was the least enforced of the
four, and gives it this mechanism: a prediction is written down and committed
**before** the measurement, a result names it and names the commit it landed in,
and a test asserts the one is an ancestor of the other.

**The load bearing test in this file is the ancestor one.** Everything else here
is format checking, which keeps the entries readable and machine parsed; the
ancestor test is what makes the ordering a fact rather than a claim. A date in a
file is written by the person writing the file. An ancestry relation in git
cannot be forged without rewriting every sha downstream of it, which would change
`manifest.git_sha` in every result and be visible.

**The mechanism is tested on both sides**, which is the rule Section 19.1 states
for CI and which applies just as well to a test: a check that has only ever
passed is a check nobody has seen work. So the ancestor relation is asserted for
every result that names a prediction, and separately shown to *refuse* a sha that
is not an ancestor.

**In a shallow checkout these tests fail rather than skip, and that is
deliberate.** D-0041 made them refuse with a readable reason instead of reporting
every historical sha as missing, and the obvious next step, turning the refusal
into `pytest.skip`, is the wrong one. Law 3 and law 4 are two of the four binding
laws of Section 0.2. A checkout that cannot check them has not satisfied them,
and a green run that quietly skipped both would be the two strongest honesty
claims in the project switching themselves off in the environment where nobody is
watching. Section 19.0's rule that silence and success must not look alike is the
same rule; a skip here is silence. The fix is to fetch the history, which is one
line in the workflow and is already there.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from npu_frontend.predictions import (
    PREDICTIONS_DIR,
    REQUIRED_FIELDS,
    PredictionError,
    commit_exists,
    head_sha,
    is_ancestor,
    landing_sha,
    load_predictions,
    parse_prediction,
    repository_is_shallow,
    require_full_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def committed_results() -> list[Path]:
    """Every result file in the tree, which may legitimately be none yet."""
    return sorted(RESULTS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# The entries.
# ---------------------------------------------------------------------------


def test_the_directory_holds_at_least_one_prediction() -> None:
    """Section 17.8: the mechanism lands with a user, not empty.

    A directory, two schema fields and a test, with nothing in the directory,
    would be a mechanism whose first exercise happens at P11 and whose first
    failure is therefore found by the phase that depends on it.
    """
    predictions = load_predictions()
    assert predictions
    assert "p10-ablation-deltas" in predictions


def test_every_entry_carries_the_five_things_section_17_8_names() -> None:
    """Parsed rather than eyeballed, over whatever the directory holds today."""
    for identifier, prediction in load_predictions().items():
        for name in REQUIRED_FIELDS:
            assert prediction.fields[name], f"{identifier} has an empty {name!r}"
        assert prediction.result_fields, f"{identifier} names no result field"
        assert "## Hypothesis" in prediction.body
        assert "## What would falsify it" in prediction.body


def test_an_entry_without_a_magnitude_bracket_is_refused(tmp_path: Path) -> None:
    """The format check, shown failing, because a check that has only passed is not one.

    The magnitude bracket is the field this test drops on purpose. Section 17.8
    lists five things and this is the one whose absence is invisible in prose: an
    entry that predicts a direction and no magnitude is confirmed by any outcome
    in that direction, however small, which is the failure the protocol exists to
    prevent.
    """
    entry = tmp_path / "p99-no-bracket.md"
    entry.write_text(
        "\n".join(
            [
                "# Prediction: something",
                "",
                "- **id:** p99-no-bracket",
                "- **written:** 2026-09-01",
                "- **result field:** `simulated_cycles`",
                "- **direction:** up",
                "- **answered at:** P99",
                "",
                "## Hypothesis",
                "",
                "It goes up.",
                "",
                "## What would falsify it",
                "",
                "It goes down.",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(PredictionError) as failure:
        parse_prediction(entry)
    assert "magnitude bracket" in str(failure.value)
    assert "any outcome confirms" in str(failure.value)


def test_an_id_that_does_not_match_the_file_name_is_refused(tmp_path: Path) -> None:
    """A reader holding a result's prediction_id has to be able to find the entry."""
    entry = tmp_path / "p99-named-one-thing.md"
    entry.write_text(
        "\n".join(
            [
                "# Prediction: something",
                "",
                "- **id:** p99-named-another",
                "- **written:** 2026-09-01",
                "- **result field:** `simulated_cycles`",
                "- **direction:** up",
                "- **magnitude bracket:** 1 to 2",
                "- **answered at:** P99",
                "",
                "## Hypothesis",
                "",
                "It goes up.",
                "",
                "## What would falsify it",
                "",
                "It goes down.",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(PredictionError) as failure:
        parse_prediction(entry)
    assert "have to agree" in str(failure.value)


def test_an_empty_directory_is_refused_rather_than_read_as_no_predictions(
    tmp_path: Path,
) -> None:
    with pytest.raises(PredictionError) as failure:
        load_predictions(tmp_path)
    assert "has never been tested" in str(failure.value)


def test_every_entry_landed_in_a_commit_this_repository_has() -> None:
    """The sha a result would record for it, resolved.

    Read from git rather than from a field in the file, and that is deliberate:
    an entry cannot state the sha of the commit that contains it, so a `sha`
    field would have to be filled in afterwards and would be a claim.
    """
    require_full_history("resolving the commit each prediction landed in")
    for identifier in load_predictions():
        sha = landing_sha(identifier)
        assert sha, (
            f"{identifier} is not in any commit, so no result can name a "
            f"prediction_sha for it. Commit the entry before recording the "
            f"measurement, which is the whole of Section 17.8."
        )
        assert commit_exists(sha)


# ---------------------------------------------------------------------------
# The ancestor test.
# ---------------------------------------------------------------------------


def test_every_result_that_names_a_prediction_predates_it() -> None:
    """Section 17.8's ancestor test, over every committed result.

    Vacuous while `experiments/results/` is empty, which it is until the first
    benchmark run is committed. That is not a weakness of the test: the next test
    asserts the directory is not empty once it exists, and
    `test_at_least_one_committed_result_names_a_prediction` asserts the path has
    actually been walked.
    """
    require_full_history("checking that every prediction predates its measurement")
    predictions = load_predictions()
    for path in committed_results():
        cell = json.loads(path.read_text(encoding="utf-8"))
        identifier = cell["prediction_id"]
        sha = cell["prediction_sha"]
        if identifier is None:
            assert sha is None, (
                f"{path.name} names no prediction and carries a prediction_sha "
                f"anyway, which is a sha for nothing"
            )
            continue

        assert identifier in predictions, (
            f"{path.name} names the prediction {identifier!r} and "
            f"experiments/predictions/ has no such entry"
        )
        assert sha, f"{path.name} names {identifier!r} with no prediction_sha"
        assert commit_exists(sha), (
            f"{path.name} records the prediction_sha {sha}, which is not a "
            f"commit in this repository"
        )
        measured_at = cell["manifest"]["git_sha"]
        assert commit_exists(measured_at), (
            f"{path.name} was measured at {measured_at}, which is not a commit "
            f"in this repository. Section 16.1: a committed number tracing to a "
            f"commit that does not exist is the exact failure this prevents."
        )
        assert is_ancestor(sha, measured_at), (
            f"{path.name} names the prediction {identifier!r}, which landed in "
            f"{sha[:12]}, and was measured at {measured_at[:12]}, which does not "
            f"have it as an ancestor. The prediction does not provably predate "
            f"the measurement it predicts, which is the one thing Section 17.8's "
            f"mechanism exists to make impossible."
        )


def test_the_ancestor_check_refuses_a_sha_that_is_not_an_ancestor() -> None:
    """The mechanism shown failing, on this repository's own history.

    The empty tree object's commit is not used, because there is not one. What is
    used is the relation itself, in both directions between two real commits:
    HEAD is not an ancestor of its own parent. If `is_ancestor` were returning
    true unconditionally, every assertion in the test above would pass over any
    pair of shas at all, and the mechanism would be decorative.
    """
    require_full_history("checking that the ancestor relation refuses as well as holds")
    head = head_sha()
    parent = f"{head}~1"
    assert is_ancestor(parent, head), "a parent is an ancestor of its child"
    assert not is_ancestor(head, parent), (
        "a child is not an ancestor of its parent, and if this passes the "
        "ancestor test above is asserting nothing"
    )


def test_the_ancestor_check_refuses_to_guess_in_a_shallow_checkout(
    tmp_path: Path,
) -> None:
    """D-0041, and it is the half of it that no checkout option fixes.

    `git merge-base --is-ancestor` exits nonzero both for "no" and for "I have
    never heard of that commit", and the first version of `is_ancestor` collapsed
    the two into False. In a shallow checkout that turned "this cannot be
    observed" into "the prediction does not predate its measurement", which is a
    serious finding reported where there was none.

    `landing_sha` was worse, because it did not fail at all. `git log
    --diff-filter=A` against a truncated history attributes every file to the
    graft commit, so it returned the checkout's own tip, which looks exactly like
    a real answer and would have been recorded as a result's `prediction_sha`.

    The reproduction is a real shallow clone rather than a mock, because what is
    under test is what git does rather than what this project believes git does.
    It is the shape `actions/checkout` produces at its default `fetch-depth: 1`,
    which is how the defect was found.
    """
    shallow = tmp_path / "shallow"
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", f"file://{REPO_ROOT}", str(shallow)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:  # pragma: no cover
        pytest.skip(f"could not make a shallow clone here: {completed.stderr}")
    assert repository_is_shallow(repository=shallow)

    with pytest.raises(PredictionError) as guard:
        require_full_history("anything at all", repository=shallow)
    assert "fetch-depth: 0" in str(guard.value)
    assert "D-0041" in str(guard.value)

    with pytest.raises(PredictionError):
        is_ancestor(
            "f92de427d1f315d9d6621c44516e54f886f18a9c",
            "d4210f352957b95e69185003bb1b960a2a3286be",
            repository=shallow,
        )

    with pytest.raises(PredictionError) as landing:
        landing_sha("p10-ablation-deltas", repository=shallow)
    assert "shallow" in str(landing.value)

    # The same clone, deepened, answering the same questions. This is the
    # contrast that makes the refusal meaningful: the guard is about the depth of
    # the checkout rather than about the questions being asked.
    #
    # Deepening this clone rather than asserting anything about the repository
    # the suite is running in, because that repository is shallow in exactly the
    # situation this test is about, and a test that asserted otherwise would fail
    # in the environment it exists to describe. It did, once, before this comment.
    deepened = subprocess.run(
        ["git", "fetch", "--unshallow"],
        cwd=shallow,
        capture_output=True,
        text=True,
        check=False,
    )
    if deepened.returncode != 0:  # pragma: no cover
        pytest.skip(f"could not unshallow the fixture: {deepened.stderr}")

    assert not repository_is_shallow(repository=shallow)
    require_full_history("the same question, now answerable", repository=shallow)
    assert landing_sha("p10-ablation-deltas", repository=shallow) is not None


def test_at_least_one_committed_result_names_a_prediction() -> None:
    """The path is known to have been walked, not merely to be available.

    Section 17.8's reason for landing the mechanism at P10 is that a mechanism
    arriving with its first consumer has never been tested when it matters. That
    argument is only worth something if P10 actually walks it, so this asserts
    the directory, the two schema fields and the ancestor test have all been
    exercised by a real recorded measurement rather than by a fixture.

    `test_every_result_that_names_a_prediction_predates_it` is vacuously green
    over an empty results directory. This one is not, and that is the division of
    labour between the two.
    """
    results = committed_results()
    if not results:
        pytest.skip("no results recorded yet; run experiments/run_benchmarks.py")

    naming = []
    for path in results:
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell["prediction_id"] is not None:
            naming.append(cell)

    assert naming, (
        "no committed result names a prediction, so the P10 path has been built "
        "and never walked, which is the exact situation Section 17.8 landed the "
        "mechanism a phase early to avoid"
    )
    assert {cell["prediction_id"] for cell in naming} == {"p10-ablation-deltas"}
    assert len({cell["prediction_sha"] for cell in naming}) == 1

    # Every ablation row is evidence for it, and so is each unablated `-O2` cell
    # its deltas are taken against. Counted rather than asserted loosely, so that
    # a wiring change which quietly stopped tagging the baselines is red.
    assert len(naming) == 126
    assert sum(1 for cell in naming if cell["cell"]["ablated_pass"] is not None) == 112


def test_a_prediction_landing_commit_is_an_ancestor_of_head() -> None:
    """What a result recorded right now would be asserting.

    Run at every commit rather than only where a result exists, so that an entry
    added on a branch that was never merged is caught here rather than at the
    moment a result names it.
    """
    require_full_history("checking each prediction is an ancestor of HEAD")
    head = head_sha()
    for identifier in load_predictions():
        sha = landing_sha(identifier)
        assert sha is not None
        assert is_ancestor(sha, head), (
            f"{identifier} landed in {sha[:12]}, which is not an ancestor of "
            f"HEAD, so a result measured now could not legitimately name it"
        )


# ---------------------------------------------------------------------------
# What the directory says about the phases.
# ---------------------------------------------------------------------------


def test_the_p11_divergence_prediction_is_here_and_unanswered() -> None:
    """Section 17.8's reason for landing the mechanism a phase early.

    P11's gate requires the SCALE-Sim divergence prediction to already exist and
    to already be an ancestor of the commit that records the first SCALE-Sim
    number. It is here, at P10, before `experiments/scalesim_export.py` exists at
    all, which is the strongest form of that claim available: no code in this
    repository could have produced a number to write it from.
    """
    predictions = load_predictions()
    assert "p11-scalesim-divergence" in predictions
    entry = predictions["p11-scalesim-divergence"]
    assert entry.answered_at == "P11"
    assert "scalesim_cycles" in entry.result_fields
    assert not (REPO_ROOT / "experiments" / "scalesim_export.py").exists(), (
        "the exporter exists, so this prediction is no longer provably written "
        "before anything could produce the number it predicts. That is not a "
        "reason to weaken this test; it is a reason to check that the entry "
        "landed strictly before the exporter did."
    )


def test_the_entries_are_where_section_6_puts_them() -> None:
    assert PREDICTIONS_DIR == REPO_ROOT / "experiments" / "predictions"
    assert PREDICTIONS_DIR.is_dir()
