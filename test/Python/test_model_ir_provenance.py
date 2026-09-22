# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The freshness stamp of `experiments/models/`, and the refusal it feeds.

`scripts/check-reachability.py` answers the model layer of law 2 out of a
gitignored build artifact. Until P14 the artifact was kept current by running
the sweep before the check, which is an ordering written into one CI workflow
file and nowhere else: run the check alone after changing the frontend and it
answers from the previous build, green, having checked the wrong thing. D-0067
is what that costs.

What is tested here is the whole of the replacement. The fingerprint moves for
every input that decides the artifact and stands still for the ones that do
not, an artifact with no stamp or a stamp from another version is refused
rather than compared, and the refusal reaches the check as an error with the
model layer reported as unchecked instead of answered.

The fake repository is built in a temporary directory rather than measured
against the real one, so that a test of staleness cannot itself be made stale
by an unrelated edit to `lib/`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import model_ir_provenance as provenance  # noqa: E402


def fake_repo(root: Path) -> Path:
    """A tree with one file in each covered input and two in none of them."""
    (root / "lib" / "Dialect").mkdir(parents=True)
    (root / "lib" / "Dialect" / "Ops.cpp").write_text("int lowering = 1;\n")
    (root / "include" / "NPU").mkdir(parents=True)
    (root / "include" / "NPU" / "Ops.td").write_text("def Op;\n")
    (root / "python" / "npu_frontend").mkdir(parents=True)
    (root / "python" / "npu_frontend" / "compile.py").write_text("LEVELS = 3\n")
    (root / "experiments" / "calibration").mkdir(parents=True)
    (root / "experiments" / "calibration" / "lenet.json").write_text('{"a": 1}\n')
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "build-model-ir.py").write_text("BATCHES = (1, 4)\n")

    # Neither of these decides a line of the IR, and a fingerprint that moved
    # for them would cry stale at every documentation edit.
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "PASSES.md").write_text("# passes\n")
    (root / "lib" / "notes.txt").write_text("scratch\n")

    models = root / "experiments" / "models"
    models.mkdir(parents=True)
    (models / "lenet-O0.npu.mlir").write_text("npu.conv2d\n")
    return models


def test_the_fingerprint_is_the_same_twice_over_an_unchanged_tree(
    tmp_path: Path,
) -> None:
    fake_repo(tmp_path)
    assert provenance.fingerprint(tmp_path) == provenance.fingerprint(tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "lib/Dialect/Ops.cpp",
        "include/NPU/Ops.td",
        "python/npu_frontend/compile.py",
        "experiments/calibration/lenet.json",
        "scripts/build-model-ir.py",
    ],
)
def test_every_input_that_decides_the_artifact_moves_the_fingerprint(
    tmp_path: Path, relative: str
) -> None:
    """The five, named one at a time so a failure says which one stopped counting.

    The calibration profile and the sweep script are the two this project's
    result hash does not cover, and they are here because the artifact depends
    on both: the quantized compilation reads a profile, and the sweep decides
    which models at which levels exist at all.
    """
    fake_repo(tmp_path)
    before = provenance.fingerprint(tmp_path)
    target = tmp_path / relative
    target.write_text(target.read_text() + "# changed\n")
    assert provenance.fingerprint(tmp_path) != before


@pytest.mark.parametrize("relative", ["docs/PASSES.md", "lib/notes.txt"])
def test_a_file_that_decides_nothing_leaves_the_fingerprint_alone(
    tmp_path: Path, relative: str
) -> None:
    fake_repo(tmp_path)
    before = provenance.fingerprint(tmp_path)
    (tmp_path / relative).write_text("rewritten\n")
    assert provenance.fingerprint(tmp_path) == before


def test_a_rename_with_identical_content_moves_the_fingerprint(tmp_path: Path) -> None:
    """Each file contributes its path as well as its bytes, deliberately.

    A pass moved from one file to another with no edit is a different compiler
    to the build system and to `#include`, and an artifact built before the
    move is not current.
    """
    fake_repo(tmp_path)
    before = provenance.fingerprint(tmp_path)
    source = tmp_path / "lib" / "Dialect" / "Ops.cpp"
    source.rename(source.with_name("Lowering.cpp"))
    assert provenance.fingerprint(tmp_path) != before


def test_a_compiled_python_file_is_not_an_input(tmp_path: Path) -> None:
    fake_repo(tmp_path)
    before = provenance.fingerprint(tmp_path)
    cache = tmp_path / "python" / "npu_frontend" / "__pycache__"
    cache.mkdir()
    (cache / "compile.cpython-314.py").write_text("stale bytecode stand in\n")
    assert provenance.fingerprint(tmp_path) == before


def test_a_missing_input_tree_is_not_reported_as_a_change(tmp_path: Path) -> None:
    """An input that does not exist yet is not the same as one that moved.

    `experiments/calibration/` arrives at P14 and the check has to run before
    it, so an absent tree contributes nothing rather than raising.
    """
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    profile = tmp_path / "experiments" / "calibration" / "lenet.json"
    assert profile in provenance.covered_files(tmp_path)

    profile.unlink()
    (tmp_path / "experiments" / "calibration").rmdir()
    assert profile not in provenance.covered_files(tmp_path)
    # Deleting an input is a change like any other, so the artifact is stale,
    # and the point is that the walk answered rather than raised.
    assert provenance.staleness(models, tmp_path) is not None


def test_a_stamped_artifact_matching_the_tree_is_not_stale(tmp_path: Path) -> None:
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    assert provenance.staleness(models, tmp_path) is None

    stamp = provenance.read_stamp(models)
    assert stamp is not None
    assert stamp["stamp_version"] == provenance.STAMP_VERSION
    assert stamp["files_covered"] == len(provenance.covered_files(tmp_path))
    assert stamp["written_utc"].endswith("Z")


def test_a_source_changed_after_the_sweep_makes_the_artifact_stale(
    tmp_path: Path,
) -> None:
    """The defect itself: the artifact predates the change under test.

    The message has to carry both fingerprints and the command that fixes it,
    because the reader of this red has a directory they cannot see into and an
    ordering they did not know existed.
    """
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    frontend = tmp_path / "python" / "npu_frontend" / "compile.py"
    frontend.write_text("LEVELS = 4\n")

    reason = provenance.staleness(models, tmp_path)
    assert reason is not None
    assert "D-0067" in reason
    assert "scripts/build-model-ir.py" in reason
    assert provenance.fingerprint(tmp_path)[:12] in reason


def test_an_artifact_with_no_stamp_is_refused_rather_than_read(tmp_path: Path) -> None:
    models = fake_repo(tmp_path)
    reason = provenance.staleness(models, tmp_path)
    assert reason is not None
    assert provenance.STAMP_NAME in reason
    assert "scripts/build-model-ir.py" in reason


def test_a_stamp_that_will_not_parse_is_treated_as_no_stamp(tmp_path: Path) -> None:
    models = fake_repo(tmp_path)
    provenance.stamp_path(models).write_text("{ this is not json\n")
    assert provenance.read_stamp(models) is None
    reason = provenance.staleness(models, tmp_path)
    assert reason is not None
    assert "no readable" in reason


def test_a_stamp_that_is_a_list_is_treated_as_no_stamp(tmp_path: Path) -> None:
    models = fake_repo(tmp_path)
    provenance.stamp_path(models).write_text("[]\n")
    assert provenance.read_stamp(models) is None


def test_a_stamp_from_another_version_is_refused_rather_than_compared(
    tmp_path: Path,
) -> None:
    """Two fingerprints over different input sets compare to nothing.

    Answering "stale" would be right by accident and answering "current" would
    be wrong, so the version mismatch is its own refusal with its own words.
    """
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    stamp = json.loads(provenance.stamp_path(models).read_text())
    stamp["stamp_version"] = provenance.STAMP_VERSION + 1
    provenance.stamp_path(models).write_text(json.dumps(stamp))

    reason = provenance.staleness(models, tmp_path)
    assert reason is not None
    assert "stamp version" in reason


def test_clearing_the_stamp_leaves_nothing_certified(tmp_path: Path) -> None:
    """What `--model` does, because one model is not the sweep the stamp claims."""
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    provenance.clear_stamp(models)
    assert not provenance.stamp_path(models).exists()
    assert provenance.staleness(models, tmp_path) is not None

    # Clearing what is not there is the ordinary case on a first sweep.
    provenance.clear_stamp(models)


def test_the_stamp_is_written_through_a_rename(tmp_path: Path) -> None:
    """The project's rule for every artifact a reader may open mid write."""
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    leftovers = list(models.glob("*.tmp"))
    assert leftovers == []


def test_the_shared_half_of_the_input_set_matches_the_result_hash(
    tmp_path: Path,
) -> None:
    """One construction, two input sets, and the overlap is pinned here.

    `npu_frontend.results.content_hash` keys a result cell's staleness on the
    compiler sources; this keys an artifact's on the same sources plus the two
    that only it depends on. If the result hash widens its suffixes and this
    does not, the two answers drift apart with nothing to notice.
    """
    from npu_frontend import results

    assert provenance.CODE_SUFFIXES == results.CONTENT_HASH_SUFFIXES
    code_trees = tuple(
        tree
        for tree, suffixes in provenance.SOURCE_TREES
        if suffixes == provenance.CODE_SUFFIXES
    )
    assert code_trees == results.CONTENT_HASH_SOURCES


# ---------------------------------------------------------------------------
# The refusal as check-reachability.py performs it.
# ---------------------------------------------------------------------------


def reachability() -> ModuleType:
    """`scripts/check-reachability.py`, whose name is not an identifier."""
    path = REPO_ROOT / "scripts" / "check-reachability.py"
    spec = importlib.util.spec_from_file_location("check_reachability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_reachability"] = module
    spec.loader.exec_module(module)
    return module


def test_the_check_refuses_the_model_layer_when_the_artifact_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error is raised and the layer is not answered, which is the point.

    A stale artifact is worse than a missing one. Missing is reported as a
    layer this run did not check; stale answers, and answers about a build
    nobody looked at.
    """
    module = reachability()
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)
    (tmp_path / "lib" / "Dialect" / "Ops.cpp").write_text("int lowering = 2;\n")

    monkeypatch.setattr(module, "MODELS_DIR", models)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(module.LAYER_HOMES, "model", (models,))

    assert module.model_layer_staleness(True) is None
    reason = module.model_layer_staleness(False)
    assert reason is not None and "D-0067" in reason

    findings = module.check(False)
    assert any("D-0067" in error for error in findings.errors)
    assert not any("model" in layers for layers in findings.missing.values())
    assert any("was not checked" in note for note in findings.notes)
    for note in findings.notes:
        if note.startswith("layers checked"):
            assert "model" not in note


def test_the_check_says_which_build_answered_when_the_artifact_is_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass that names the fingerprint it read is a pass a reader can audit."""
    module = reachability()
    models = fake_repo(tmp_path)
    provenance.write_stamp(models, tmp_path)

    monkeypatch.setattr(module, "MODELS_DIR", models)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(module.LAYER_HOMES, "model", (models,))

    assert module.model_layer_staleness(False) is None
    findings = module.check(False)
    fingerprint = provenance.fingerprint(tmp_path)[:12]
    assert any(fingerprint in note for note in findings.notes)


def test_the_lint_job_mode_still_runs_with_no_artifact_at_all() -> None:
    """The one shape this change could break, run the way the lint job runs it.

    `--skip-models` executes in a job with no MLIR bindings and no
    `experiments/models/`, so the freshness module it now imports has to be the
    standard library alone and the staleness path has to stay untouched.
    """
    completed = subprocess.run(
        [sys.executable, "scripts/check-reachability.py", "--skip-models"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "check-reachability: pass" in completed.stdout
    assert provenance.STAMP_NAME not in completed.stdout
