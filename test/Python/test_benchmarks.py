"""A recorded benchmark result must not outlive the code that produced it.

Every number in the README and both reports traces back to a file in
experiments/results/. The resume logic used to accept any file that parsed as
JSON, which made staleness permanent: results generated three commits earlier
were reused forever unless someone remembered --force, so the published numbers
quietly decoupled from the compiler. These tests pin the manifest check that
replaced it. See docs/ASSESSMENT.md section 4.2.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))

import run_benchmarks  # noqa: E402


def _write(path: Path, manifest: dict) -> Path:
    path.write_text(json.dumps({"model": "lenet", "manifest": manifest}))
    return path


def _current_manifest() -> dict:
    return {
        "git_sha": run_benchmarks.git_sha(),
        "cost_model": dict(run_benchmarks.COST_MODEL),
    }


def test_result_from_this_commit_is_reusable(tmp_path):
    path = _write(tmp_path / "r.json", _current_manifest())
    assert run_benchmarks.staleness(path) is None


def test_result_from_an_unknown_commit_is_stale(tmp_path):
    """A sha this repository has never seen cannot be vouched for."""
    manifest = _current_manifest()
    manifest["git_sha"] = "8095dbec" + "0" * 32
    path = _write(tmp_path / "r.json", manifest)

    why = run_benchmarks.staleness(path)
    assert why is not None
    # The reason has to name both shas, so the operator can see how far behind
    # the result is rather than just being told to rerun.
    assert "8095dbec" in why
    assert run_benchmarks.git_sha()[:8] in why


def test_result_from_a_commit_predating_compiler_changes_is_stale(tmp_path):
    """The root commit predates every line of the compiler, so nothing from it
    can still be current."""
    root = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-list", "--max-parents=0", "HEAD"], text=True
    ).split()[0]
    manifest = _current_manifest()
    manifest["git_sha"] = root
    path = _write(tmp_path / "r.json", manifest)

    why = run_benchmarks.staleness(path)
    assert why is not None
    assert "compiler input" in why


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def test_documentation_only_commits_do_not_count_as_compiler_changes():
    """The rule is "did anything that moves a number change", not "did HEAD
    move".

    This repository's history contains several commits that only edit the
    README. If RESULT_INPUTS were too broad, those would invalidate every
    recorded result and the harness would regenerate the whole suite after a
    typo fix. This walks recent history for such a commit and asserts the
    filter sees through it.
    """
    shas = _git("rev-list", "-40", "HEAD").split()
    for sha in shas:
        touched_all = _git("diff", "--name-only", f"{sha}~1", sha).split()
        if not touched_all:
            continue
        touched_inputs = _git(
            "diff", "--name-only", f"{sha}~1", sha, "--", *run_benchmarks.RESULT_INPUTS
        ).split()
        if not touched_inputs:
            assert touched_all, f"{sha} changed nothing at all"
            return
    pytest.skip("no documentation only commit in the last 40 commits")


def test_result_with_different_cost_model_is_stale(tmp_path):
    manifest = _current_manifest()
    manifest["cost_model"]["macs_per_cycle"] = 128
    path = _write(tmp_path / "r.json", manifest)

    why = run_benchmarks.staleness(path)
    assert why is not None
    assert "cost model" in why


def test_result_without_a_manifest_is_stale(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"model": "lenet", "simulated_cycles": 12710}))
    assert run_benchmarks.staleness(path) == "no manifest block"


def test_unparseable_result_is_stale(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{ this is not json")
    why = run_benchmarks.staleness(path)
    assert why is not None
    assert "unreadable" in why


def test_committed_results_are_current():
    """The results in the repository must match HEAD.

    This is the test that keeps the published numbers honest. It fails when
    results are committed from one commit and the code moves under them, which
    is exactly the state the repository was found in.
    """
    results = sorted((REPO / "experiments" / "results").glob("*.json"))
    if not results:
        pytest.skip("no recorded results in the working tree")
    stale = {p.name: run_benchmarks.staleness(p) for p in results}
    stale = {name: why for name, why in stale.items() if why is not None}
    assert not stale, f"stale results, rerun experiments/run_benchmarks.py: {stale}"


def test_no_result_traces_to_a_missing_commit():
    """Every result's git_sha must be a commit that exists in this repository.

    The published numbers were once generated at git_sha 8095dbec, which is not
    a commit here and never was, so nobody could reproduce them or even say what
    code produced them. staleness() compares the recorded sha against HEAD but
    says nothing about whether it is real, so a fabricated or rebased away sha
    passes every other check in this file.
    """
    results = sorted((REPO / "experiments" / "results").glob("*.json"))
    if not results:
        pytest.skip("no recorded results in the working tree")
    missing = {}
    for path in results:
        sha = json.loads(path.read_text())["manifest"]["git_sha"]
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=REPO,
            capture_output=True,
        )
        if proc.returncode != 0:
            missing[path.name] = sha
    assert not missing, f"results tracing to commits that do not exist: {missing}"


def test_macros_match_the_committed_results():
    """The tex the report includes must equal the results it claims to come from.

    report/generated/ was gitignored, so nothing enforced spec 23.1: the PDFs
    could cite numbers no committed result contained, and did. The directory is
    tracked now, and this test is what makes tracking it mean something.
    """
    macros_path = REPO / "report" / "generated" / "macros.tex"
    if not macros_path.exists():
        pytest.skip("report/generated/macros.tex not built")
    macros = dict(
        re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", macros_path.read_text())
    )

    rows = [
        json.loads(p.read_text())
        for p in sorted((REPO / "experiments" / "results").glob("*.json"))
    ]
    if not rows:
        pytest.skip("no recorded results in the working tree")

    shas = {r["manifest"]["git_sha"][:12] for r in rows}
    assert len(shas) == 1, f"results disagree on git_sha: {shas}"
    assert macros["GitSha"] == shas.pop(), (
        f"macros.tex cites GitSha {macros['GitSha']} but the results were "
        f"generated at a different commit"
    )

    default = {
        r["opt_level"]: r
        for r in rows
        if r["model"] == "lenet" and r["scratchpad_budget"] == 1048576
    }
    expected = {
        "OaInstr": default[0]["instruction_count"],
        "OcInstr": default[2]["instruction_count"],
        "OaCycles": default[0]["simulated_cycles"],
        "OcCycles": default[2]["simulated_cycles"],
        "OaDram": default[0]["dram_bytes_total"] // 1024,
        "ObDram": default[1]["dram_bytes_total"] // 1024,
    }
    for name, value in expected.items():
        assert macros[name] == str(value), (
            f"macros.tex has {name}={macros[name]} but the committed result "
            f"says {value}"
        )


def test_cost_model_constants_match_the_cpp_header():
    """The Python mirror of the cost model must not drift from the C++ source.

    run_benchmarks.COST_MODEL is a hand copy of the defaults in CostModel.h and
    is recorded in every manifest, so if the two disagree the manifests record a
    cost model that never ran.
    """
    header = (REPO / "include" / "NPU" / "Simulator" / "CostModel.h").read_text()
    expected = {
        "macs_per_cycle": "macsPerCycle",
        "dram_bytes_per_cycle": "dramBytesPerCycle",
        "lanes": "lanes",
        "issue_overhead": "issueOverhead",
    }
    for python_name, cpp_name in expected.items():
        marker = f"int64_t {cpp_name} = "
        line = next(ln for ln in header.splitlines() if marker in ln)
        value = int(line.split("=", 1)[1].split(";")[0].strip())
        assert run_benchmarks.COST_MODEL[python_name] == value, (
            f"{python_name} is {run_benchmarks.COST_MODEL[python_name]} in "
            f"run_benchmarks.py but {cpp_name} is {value} in CostModel.h"
        )
