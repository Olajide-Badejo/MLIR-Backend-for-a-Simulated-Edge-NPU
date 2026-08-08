"""Record and re-check the regression baseline that gates every upgrade phase.

This is the safety net. Before any upgrade work starts, `--record` captures what
the repository does today: which tests exist and pass, and what the compiler
produces for every LeNet cell in the optimization level times budget grid.
Afterwards, `--check` re-runs the identical measurements and diffs them against
the recording, exiting nonzero on any drift.

Some things are deliberately recorded but never treated as drift, because they
change for reasons that are not behaviour changes: the git sha, which moves with
every commit; the tool versions, which move when the machine is updated; and
newly added tests, since every upgrade phase adds some. All are reported as
notes. Pass --strict-tools to make a tool version change fatal.

What does count as drift in a test suite is a test disappearing, a test that
used to pass no longer passing, or anything failing at all.

The instruction count comes from the simulator's own `stats.instructions`, not
from a regex over the IR dump. The regex counts `npuisa.const`, which is data
rather than an instruction, and matches inside type strings such as
`!npuisa.buffer`.

Usage is through scripts/regression-baseline.sh, which sets up the environment.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from npu_frontend import compile as npu_compile  # noqa: E402
from npu_frontend import model_generator  # noqa: E402

BASELINE_DIR = REPO / "test" / "baseline"
BASELINE_JSON = BASELINE_DIR / "baseline.json"
GOLDEN_DIR = BASELINE_DIR / "golden"

LLVM_TAG = "llvmorg-22.1.8"
DEFAULT_BUDGET = 1048576
TIGHT_BUDGET = 143360  # 140 KB: below the LeNet peak, so weights spill
BUDGETS = {"default": DEFAULT_BUDGET, "tight": TIGHT_BUDGET}
LEVELS = [0, 1, 2]
SEED = 0

# Golden tensors are fp32 network outputs. A change that moves one by more than
# this must be justified in docs/BREAKING_CHANGES.md.
GOLDEN_ATOL = 1e-6
# The end to end error against onnxruntime is deterministic for a fixed binary
# and input, so any movement at all is worth reporting.
ERROR_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _first_line(cmd: list[str]) -> str:
    """Report a tool as "resolved path: version".

    The path matters as much as the version here. This machine has a pip
    installed cmake in ~/.local/bin that shadows /usr/bin/cmake in login shells
    but not in non-login ones, so the same command can be two different
    binaries depending on how the script was invoked. Recording the path makes
    that visible instead of surfacing as an unexplained version change.
    """
    resolved = shutil.which(cmd[0]) or cmd[0]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        version = (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        version = "unknown"
    return f"{resolved}: {version}"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def tool_versions() -> dict:
    import onnx
    import torch

    return {
        "llvm_tag": LLVM_TAG,
        "cmake": _first_line(["cmake", "--version"]),
        "ninja": _first_line(["ninja", "--version"]),
        "cxx": _first_line(["c++", "--version"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
    }


def cost_model_constants(repo: Path) -> dict:
    """Read the analytical cost model defaults out of the C++ header.

    Parsed rather than duplicated so a constant cannot be changed in the header
    without the baseline noticing. This is what makes the deliberate
    perturbation in the Phase U0 gate detectable.
    """
    header = repo / "include" / "NPU" / "Simulator" / "CostModel.h"
    text = header.read_text()
    constants = {}
    for field in ("macsPerCycle", "dramBytesPerCycle", "lanes", "issueOverhead"):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"int64_t {field}"):
                value = stripped.split("=", 1)[1].split(";")[0].strip()
                constants[field] = int(value)
                break
        else:
            raise RuntimeError(f"cost model constant {field} not found in {header}")
    return constants


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------


def parse_junit(path: Path) -> dict:
    """Read a JUnit or XUnit XML report into a suite summary.

    lit, GoogleTest, and pytest all emit this format natively, so one parser
    covers all four suites and the recorded shape is uniform.
    """
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    tests = []
    passed = failed = skipped = 0
    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname") or ""
            name = case.get("name") or ""
            full = f"{classname}::{name}" if classname else name
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
                failed += 1
            elif case.find("skipped") is not None:
                status = "skipped"
                skipped += 1
            else:
                status = "passed"
                passed += 1
            tests.append({"name": full, "status": status})
    tests.sort(key=lambda t: t["name"])
    return {"passed": passed, "failed": failed, "skipped": skipped, "tests": tests}


def run_lit(build_dir: Path, lit: Path, workdir: Path) -> dict:
    xml = workdir / "lit.xml"
    subprocess.run(
        [str(lit), "-sv", "--xunit-xml-output", str(xml), str(build_dir / "test")],
        cwd=str(build_dir / "test"),
        capture_output=True,
        text=True,
    )
    if not xml.exists():
        raise RuntimeError("lit produced no XML report")
    return parse_junit(xml)


def run_gtest(binary: Path, workdir: Path) -> dict:
    xml = workdir / f"{binary.name}.xml"
    subprocess.run(
        [str(binary), f"--gtest_output=xml:{xml}"], capture_output=True, text=True
    )
    if not xml.exists():
        raise RuntimeError(f"{binary.name} produced no XML report")
    return parse_junit(xml)


def run_pytest(repo: Path, workdir: Path) -> dict:
    xml = workdir / "pytest.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "test/Python", "-q", f"--junitxml={xml}"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if not xml.exists():
        raise RuntimeError("pytest produced no XML report")
    return parse_junit(xml)


def run_dash_lint(repo: Path) -> dict:
    proc = subprocess.run(
        ["bash", "scripts/dash-lint.sh"], cwd=str(repo), capture_output=True, text=True
    )
    ok = proc.returncode == 0
    return {
        "passed": 1 if ok else 0,
        "failed": 0 if ok else 1,
        "skipped": 0,
        "tests": [{"name": "dash-lint", "status": "passed" if ok else "failed"}],
    }


# ---------------------------------------------------------------------------
# Compiler cells
# ---------------------------------------------------------------------------


def cell_name(model: str, level: int, budget_tag: str) -> str:
    return f"{model}_O{level}_{budget_tag}"


def run_cell(
    model: str, level: int, budget_tag: str, bin_dir: Path, workdir: Path
) -> tuple[dict, np.ndarray]:
    """Compile and simulate one cell, returning its metrics and its output."""
    budget = BUDGETS[budget_tag]
    tag = cell_name(model, level, budget_tag)
    onnx_path = model_generator.export(model, workdir / f"{tag}.onnx", seed=SEED)

    nbin = workdir / f"{tag}.nbin"
    npu_compile.compile_model(
        onnx_path,
        opt_level=level,
        emit="nbin",
        output=nbin,
        budget=budget,
        bin_dir=bin_dir,
    )

    shape = model_generator.MODELS[model][1]
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(shape).astype(np.float32)

    in_path = workdir / f"{tag}.in.bin"
    in_path.write_bytes(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    out_path = workdir / f"{tag}.out.bin"
    stats_path = workdir / f"{tag}.stats.json"
    subprocess.run(
        [
            str(bin_dir / "npu-sim"),
            str(nbin),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "--stats",
            str(stats_path),
        ],
        check=True,
        capture_output=True,
    )
    stats = json.loads(stats_path.read_text())

    reference = ort.InferenceSession(str(onnx_path)).run(None, {"input": x})[0]
    simulated = np.frombuffer(out_path.read_bytes(), dtype=np.float32)
    simulated = simulated.reshape(reference.shape)

    metrics = {
        "model": model,
        "opt_level": level,
        "budget_tag": budget_tag,
        "scratchpad_budget": budget,
        # From the simulator, not a regex over the IR (ASSESSMENT 4.3).
        "instructions": int(stats["instructions"]),
        "simulated_cycles": int(stats["cycles"]),
        "dram_bytes_read": int(stats["dram_bytes_read"]),
        "dram_bytes_written": int(stats["dram_bytes_written"]),
        "max_abs_error_vs_onnxruntime": float(np.max(np.abs(simulated - reference))),
        "note": "cycles and DRAM traffic are simulated estimates, not measurements",
    }
    return metrics, simulated


def collect_cells(bin_dir: Path, workdir: Path) -> tuple[dict, dict]:
    cells: dict[str, dict] = {}
    goldens: dict[str, np.ndarray] = {}
    for model in sorted(model_generator.MODELS):
        for level in LEVELS:
            for budget_tag in BUDGETS:
                tag = cell_name(model, level, budget_tag)
                print(f"  cell {tag}", flush=True)
                metrics, output = run_cell(model, level, budget_tag, bin_dir, workdir)
                cells[tag] = metrics
                goldens[tag] = output
    return cells, goldens


# ---------------------------------------------------------------------------
# Build, record, check
# ---------------------------------------------------------------------------


def build(build_dir: Path, jobs: int) -> None:
    # Parallelism is capped because this machine's WSL guest has a hard 12 GB
    # ceiling. See docs/BUILD.md.
    subprocess.run(
        ["ninja", "-C", str(build_dir), f"-j{jobs}"], check=True, cwd=str(REPO)
    )


def measure(build_dir: Path, lit: Path, workdir: Path) -> tuple[dict, dict]:
    bin_dir = build_dir / "bin"
    print("running lit", flush=True)
    suites = {"lit": run_lit(build_dir, lit, workdir)}
    print("running NPUEncodingTests", flush=True)
    suites["gtest_encoding"] = run_gtest(bin_dir / "NPUEncodingTests", workdir)
    print("running NPUSimulatorTests", flush=True)
    suites["gtest_simulator"] = run_gtest(bin_dir / "NPUSimulatorTests", workdir)
    print("running pytest", flush=True)
    suites["pytest"] = run_pytest(REPO, workdir)
    print("running dash-lint", flush=True)
    suites["dash_lint"] = run_dash_lint(REPO)

    print("compiling and simulating benchmark cells", flush=True)
    cells, goldens = collect_cells(bin_dir, workdir)

    snapshot = {
        "schema": 1,
        "git_sha": git_sha(),
        "recorded_utc": datetime.now(UTC).isoformat(),
        "tools": tool_versions(),
        "cost_model": cost_model_constants(REPO),
        "suites": suites,
        "cells": cells,
    }
    return snapshot, goldens


def record(snapshot: dict, goldens: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for tag, array in goldens.items():
        np.save(GOLDEN_DIR / f"{tag}.npy", array)
    BASELINE_JSON.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"\nrecorded baseline at {BASELINE_JSON}")
    print(f"recorded {len(goldens)} golden tensors in {GOLDEN_DIR}")


def check(snapshot: dict, goldens: dict, strict_tools: bool) -> int:
    if not BASELINE_JSON.exists():
        print(f"no baseline recorded at {BASELINE_JSON}; run without --check first")
        return 2
    old = json.loads(BASELINE_JSON.read_text())
    drift: list[str] = []
    notes: list[str] = []

    if old["git_sha"] != snapshot["git_sha"]:
        notes.append(
            f"git sha moved {old['git_sha'][:8]} to {snapshot['git_sha'][:8]} "
            "(expected; not drift)"
        )

    for tool, was in old["tools"].items():
        now = snapshot["tools"].get(tool, "missing")
        if was != now:
            message = f"tool {tool}: {was!r} to {now!r}"
            (drift if strict_tools else notes).append(message)

    for name, was in old["cost_model"].items():
        now = snapshot["cost_model"].get(name)
        if was != now:
            drift.append(f"cost model constant {name}: {was} to {now}")

    # What is drift in a test suite: a test that existed before has disappeared,
    # a test that passed before no longer does, or anything is failing. What is
    # not drift: new tests. Every upgrade phase adds tests, so a rising pass
    # count is the expected shape of progress and is reported as a note.
    for suite, was in old["suites"].items():
        now = snapshot["suites"].get(suite)
        if now is None:
            drift.append(f"suite {suite} disappeared")
            continue
        if now["failed"]:
            drift.append(f"suite {suite} has {now['failed']} failing test(s)")
        was_names = {t["name"] for t in was["tests"]}
        now_names = {t["name"] for t in now["tests"]}
        for gone in sorted(was_names - now_names):
            drift.append(f"suite {suite} lost test {gone}")
        added = sorted(now_names - was_names)
        if added:
            notes.append(
                f"suite {suite} gained {len(added)} test(s): {', '.join(added)}"
            )
        was_status = {t["name"]: t["status"] for t in was["tests"]}
        for test in now["tests"]:
            before = was_status.get(test["name"])
            if before is not None and before != test["status"]:
                drift.append(
                    f"suite {suite} test {test['name']}: {before} to {test['status']}"
                )
        for field in ("passed", "failed", "skipped"):
            if was[field] != now[field]:
                notes.append(
                    f"suite {suite} {field} count: {was[field]} to {now[field]}"
                )
    for suite in snapshot["suites"]:
        if suite not in old["suites"]:
            notes.append(f"suite {suite} is new")

    integer_metrics = (
        "instructions",
        "simulated_cycles",
        "dram_bytes_read",
        "dram_bytes_written",
    )
    for tag, was in old["cells"].items():
        now = snapshot["cells"].get(tag)
        if now is None:
            drift.append(f"cell {tag} disappeared")
            continue
        for metric in integer_metrics:
            if was[metric] != now[metric]:
                delta = now[metric] - was[metric]
                drift.append(
                    f"cell {tag} {metric}: {was[metric]} to {now[metric]} ({delta:+d})"
                )
        before = was["max_abs_error_vs_onnxruntime"]
        after = now["max_abs_error_vs_onnxruntime"]
        if abs(before - after) > ERROR_ATOL:
            drift.append(
                f"cell {tag} max_abs_error_vs_onnxruntime: {before:.6g} to {after:.6g}"
            )
    for tag in snapshot["cells"]:
        if tag not in old["cells"]:
            notes.append(f"cell {tag} is new")

    for tag, array in goldens.items():
        path = GOLDEN_DIR / f"{tag}.npy"
        if not path.exists():
            notes.append(f"golden {tag}.npy is new")
            continue
        was_array = np.load(path)
        if was_array.shape != array.shape:
            drift.append(f"golden {tag} shape: {was_array.shape} to {array.shape}")
            continue
        worst = float(np.max(np.abs(was_array - array)))
        if worst > GOLDEN_ATOL:
            drift.append(f"golden {tag} moved by {worst:.6g} (limit {GOLDEN_ATOL:g})")

    print()
    for note in notes:
        print(f"note:  {note}")
    if drift:
        print()
        for item in drift:
            print(f"DRIFT: {item}")
        print(f"\nbaseline check FAILED: {len(drift)} item(s) moved")
        return 1
    print("\nbaseline check clean: no drift")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-measure and diff against the recorded baseline instead of recording",
    )
    parser.add_argument("--build-dir", default=str(REPO / "build"))
    parser.add_argument(
        "--lit",
        default=str(Path.home() / "llvm-project" / "build" / "bin" / "llvm-lit"),
    )
    parser.add_argument("--no-build", action="store_true", help="skip the ninja build")
    parser.add_argument(
        "--jobs", type=int, default=6, help="build parallelism (keep at or below 6)"
    )
    parser.add_argument(
        "--strict-tools",
        action="store_true",
        help="treat a tool version change as drift rather than a note",
    )
    parser.add_argument(
        "--keep",
        help="keep intermediate artifacts in this directory instead of a temp dir",
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).resolve()
    lit = Path(args.lit)
    if not lit.exists():
        print(f"lit not found at {lit}; pass --lit", file=sys.stderr)
        return 2

    if not args.no_build:
        print(f"building with -j{args.jobs}", flush=True)
        build(build_dir, args.jobs)

    if args.keep:
        workdir = Path(args.keep).resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        snapshot, goldens = measure(build_dir, lit, workdir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot, goldens = measure(build_dir, lit, Path(tmp))

    total_failed = sum(s["failed"] for s in snapshot["suites"].values())
    total_passed = sum(s["passed"] for s in snapshot["suites"].values())
    print(f"\n{total_passed} passed, {total_failed} failed across all suites")

    if args.check:
        return check(snapshot, goldens, args.strict_tools)

    if total_failed:
        print("refusing to record a baseline with failing tests", file=sys.stderr)
        return 1
    record(snapshot, goldens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
