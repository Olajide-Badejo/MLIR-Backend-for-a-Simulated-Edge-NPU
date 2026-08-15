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


def _committed_results() -> list[dict]:
    return [
        json.loads(p.read_text())
        for p in sorted((REPO / "experiments" / "results").glob("*.json"))
    ]


def test_instruction_count_comes_from_the_simulator():
    """instruction_count must be the simulator's number, not the IR regex sum.

    The harness used to set it to sum(npuisa_op_counts.values()), a regex over
    the final IR dump that matches inside type strings such as !npuisa.buffer
    and counts npuisa.const, which is DRAM data rather than an instruction. For
    LeNet that inflated 21 to 70. See docs/ASSESSMENT.md section 4.3.
    """
    rows = _committed_results()
    if not rows:
        pytest.skip("no recorded results in the working tree")
    for r in rows:
        regex_sum = sum(r["npuisa_op_counts"].values())
        assert r["instruction_count"] != regex_sum, (
            f"{r['model']} -O{r['opt_level']} at {r['scratchpad_budget']}: "
            f"instruction_count {r['instruction_count']} equals the regex sum, "
            f"so it came from the IR dump rather than the simulator"
        )
        assert r["instruction_count"] < regex_sum, (
            "the simulator count should be the smaller of the two, since the "
            "regex over counts"
        )


def test_results_agree_with_the_recorded_baseline():
    """The results and test/baseline/baseline.json must not disagree.

    Both were committed, saying 91 / 82 / 70 and 28 / 25 / 21 for the same
    cells, which is ASSESSMENT 13.4 item 4: two contradictory instruction counts
    committed side by side. Whichever was right, committing both was the defect.
    """
    baseline_path = REPO / "test" / "baseline" / "baseline.json"
    if not baseline_path.exists():
        pytest.skip("no recorded baseline")
    cells = json.loads(baseline_path.read_text())["cells"]
    rows = _committed_results()
    if not rows:
        pytest.skip("no recorded results in the working tree")

    for r in rows:
        tag = "default" if r["scratchpad_budget"] == 1048576 else "tight"
        name = f"{r['model']}_O{r['opt_level']}_{tag}"
        if name not in cells:
            continue
        assert r["instruction_count"] == cells[name]["instructions"], (
            f"{name}: results say {r['instruction_count']} but the recorded "
            f"baseline says {cells[name]['instructions']}"
        )


def test_count_ops_is_not_an_instruction_count():
    """Document why count_ops is unfit as a scalar, by exercising both faults."""
    dump = (
        "%0 = npuisa.dma_load %x : (tensor<4xf32>) -> !npuisa.buffer<tensor<4xf32>>\n"
        "%1 = npuisa.const dense<1.0> : tensor<4xf32>\n"
        "%2 = npuisa.relu %0 : (!npuisa.buffer<tensor<4xf32>>)"
        " -> !npuisa.buffer<tensor<4xf32>>\n"
    )
    counts = run_benchmarks.count_ops(dump, "npuisa")
    # It counts the type string as though it were an op.
    assert counts.get("npuisa.buffer", 0) == 3
    # And it counts constant data as though it were an instruction.
    assert counts.get("npuisa.const", 0) == 1
    # So the sum is far above the two real instructions in this dump.
    assert sum(counts.values()) > 2


def test_readme_table_matches_the_results():
    """The hand written README table must equal the default budget results.

    The table is the most read number in the repository and is not generated,
    so nothing stopped it from drifting. It did: it published the regex count
    for a month while the recorded baseline said otherwise.
    """
    rows = _committed_results()
    if not rows:
        pytest.skip("no recorded results in the working tree")
    default = {
        r["opt_level"]: r for r in rows
        if r["model"] == "lenet" and r["scratchpad_budget"] == 1048576
    }
    readme = (REPO / "README.md").read_text()
    line = next(
        (ln for ln in readme.splitlines() if ln.strip().startswith("| Instructions")),
        None,
    )
    assert line is not None, "no Instructions row found in the README table"
    figures = [int(m) for m in re.findall(r"\d+", line)]
    assert figures == [
        default[0]["instruction_count"],
        default[1]["instruction_count"],
        default[2]["instruction_count"],
    ], f"README Instructions row is {figures} but the results say otherwise"


def _expected_pipeline(row: dict) -> list[str]:
    from npu_frontend.compile import _passes_for_level

    return [
        run_benchmarks.pass_flag_name(f) for f in _passes_for_level(row["opt_level"])
    ] + ["npu-lower-to-npuisa", "npu-allocate-scratchpad"]


def test_every_pass_in_the_pipeline_has_a_record():
    """`passes` must cover the cell's pipeline exactly, in order, with no gaps.

    The pipeline is read from _passes_for_level at assert time rather than
    hardcoded here, so adding a pass to a level without instrumenting it fails
    this test instead of quietly producing a table that omits it.
    """
    rows = _committed_results()
    if not rows:
        pytest.skip("no recorded results in the working tree")
    for r in rows:
        assert "passes" in r, f"{r['model']} -O{r['opt_level']} has no passes array"
        recorded = [p["name"] for p in r["passes"]]
        assert recorded == _expected_pipeline(r), (
            f"{r['model']} -O{r['opt_level']} at {r['scratchpad_budget']}: "
            f"recorded {recorded}, pipeline is {_expected_pipeline(r)}"
        )
        positions = [p["position"] for p in r["passes"]]
        assert positions == list(
            range(len(recorded))
        ), f"positions are {positions}, expected 0..{len(recorded) - 1}"


def test_every_pass_has_a_wall_clock():
    """A pass with no timing is an error, never a zero.

    A zero would read as a free pass, which is a claim about the compiler that
    no measurement supports.
    """
    rows = _committed_results()
    if not rows:
        pytest.skip("no recorded results in the working tree")
    for r in rows:
        assert r.get("pass_timing_source") == "--mlir-timing", (
            f"{r['model']} -O{r['opt_level']}: pass_timing_source is "
            f"{r.get('pass_timing_source')!r}, so a reader cannot tell whether "
            f"the numbers were measured"
        )
        for p in r["passes"]:
            assert p["wall_ms"] > 0, (
                f"{r['model']} -O{r['opt_level']} pass {p['name']} at position "
                f"{p['position']} recorded wall_ms {p['wall_ms']}"
            )
            assert p["ops_before_total"] == sum(p["ops_before"].values())
            assert p["ops_after_total"] == sum(p["ops_after"].values())


def test_o0_has_no_optimization_passes():
    """The negative case: -O0 records only the two lowering passes.

    If the enumeration were hardcoded rather than read from _passes_for_level,
    -O0 would carry optimization passes it never ran and every other test here
    would still pass.
    """
    rows = [r for r in _committed_results() if r["opt_level"] == 0]
    if not rows:
        pytest.skip("no -O0 results in the working tree")
    for r in rows:
        assert [p["name"] for p in r["passes"]] == [
            "npu-lower-to-npuisa",
            "npu-allocate-scratchpad",
        ], f"-O0 at {r['scratchpad_budget']} recorded {[p['name'] for p in r['passes']]}"


def test_op_stats_parser_pins_the_format():
    """The op stats parser must decode a real sample and raise on a broken one.

    npu-opt's print-op-stats output is an external format. If it shifts, an
    empty histogram would record every pass as changing nothing, which looks
    like data rather than like a failure.
    """
    sample = (
        '{\n  "builtin.module" : 1,\n  "func.func" : 1,\n'
        '  "npu.conv2d" : 2,\n  "npu.relu" : 4\n}'
    )
    assert run_benchmarks.parse_op_stats(sample) == {
        "builtin.module": 1,
        "func.func": 1,
        "npu.conv2d": 2,
        "npu.relu": 4,
    }

    # The older text form, a truncated object, and an empty histogram all raise.
    with pytest.raises(RuntimeError):
        run_benchmarks.parse_op_stats("Operations encountered:\n  npu.conv2d , 2\n")
    with pytest.raises(RuntimeError):
        run_benchmarks.parse_op_stats("{ not json at all ")
    with pytest.raises(RuntimeError):
        run_benchmarks.parse_op_stats("{}")


def test_pass_timing_parser_raises_on_a_missing_pass():
    """A pipeline pass absent from the timing output must raise, not be zeroed."""
    two = ["-canonicalize", "-symbol-dce"]
    only_one = json.dumps(
        [
            {"wall": {"duration": 0.001}, "name": "Parser", "passes": [{}]},
            {"wall": {"duration": 0.002}, "name": "Canonicalizer", "passes": [{}]},
            {"wall": {"duration": 0.003}, "name": "Total"},
        ]
    )
    with pytest.raises(RuntimeError, match="pipeline has"):
        run_benchmarks.parse_pass_timings(only_one, two)

    mismatched = json.dumps(
        [
            {"wall": {"duration": 0.002}, "name": "Canonicalizer", "passes": [{}]},
            {"wall": {"duration": 0.002}, "name": "SomeOtherPass", "passes": [{}]},
        ]
    )
    with pytest.raises(RuntimeError, match="disagree"):
        run_benchmarks.parse_pass_timings(mismatched, two)


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
