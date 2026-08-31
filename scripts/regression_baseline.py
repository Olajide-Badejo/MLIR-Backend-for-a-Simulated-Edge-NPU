#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# The regression baseline of Section 17.6.
#
#   python scripts/regression_baseline.py            record the baseline
#   python scripts/regression_baseline.py --check    re-run and diff against it
#
# `scripts/regression-baseline.sh` is the entry point that builds first. This
# file is the whole of the measuring and the diffing, and it is Python because
# the cells need the frontend and the diff needs to read JSON rather than a log.
#
# WHAT IS RECORDED, AND WHY EACH PART IS THERE
#
# Two halves. The **suites** half records the pass and fail count of every suite
# and the list of test names in each, which is what turns a silently deleted
# test into a red run. The **cells** half compiles every model at every level
# and budget and records what the simulator measured, which is what turns a
# performance change into a red run rather than a number nobody compared.
#
# EVERY NUMBER COMES FROM A MACHINE READABLE SOURCE
#
# lit writes JSON with `--output`, GoogleTest writes JSON with
# `--gtest_output=json:`, pytest writes JUnit XML with `--junitxml`, and the
# simulator writes JSON with `--json-stats`. Nothing here parses a summary line
# out of a log. That is Section 16.2's rule and it earns its keep here more than
# anywhere: a baseline that read "0 tests" out of a changed format would report
# no drift forever.
#
# THE SCHEMA IS VERSIONED BECAUSE IT DOES NOT ARRIVE COMPLETE
#
# Section 17.6 is explicit. Per level fields arrive at P9, when levels above
# `-O0` first exist, and energy fields arrive at P11, when Accelergy lands. A P8
# baseline that claimed energy would be a baseline recording a number no phase
# had computed. So `absent_fields` names them and the phase that adds them,
# `--check` compares only the fields present in both, and an unrecognised
# `schema_version` is a loud failure rather than a guess.

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "test" / "baseline"
BASELINE_PATH = BASELINE_DIR / "baseline.json"
GOLDEN_DIR = BASELINE_DIR / "golden"

#: Bumped whenever the recorded shape changes. `--check` refuses a version it
#: does not know rather than guessing what a missing field meant.
SCHEMA_VERSION: Final[int] = 1

#: The fields Section 17.6 names and this phase cannot compute, with the phase
#: that adds each. Recorded in the file itself, so a reader of a P8 baseline at
#: P14 is told why the field is missing rather than left to infer it.
ABSENT_FIELDS: Final[dict[str, str]] = {
    "energy": (
        "P11, when Accelergy lands. A P8 baseline that claimed energy would be "
        "recording a number no phase had computed."
    ),
    "per_level": (
        "P9, when -O1 and -O2 first exist. At P8 the only level the compiler "
        "can emit is -O0, and a baseline that claimed a level the compiler "
        "cannot emit is a baseline nobody can re-record."
    ),
}

#: The GoogleTest binaries, in the order the CI job runs them.
GTEST_BINARIES: Final[tuple[str, ...]] = (
    "NPUInterfaceTests",
    "NPUTilingTests",
    "NPUAllocatorTests",
    "NPUEncodingTests",
    "NPUSimulatorTests",
)

#: The golden tolerance at this phase, which is zero.
#:
#: Section 17.6 sets the tolerance per phase class and P8 is in neither list,
#: because P8 is the phase that *creates* the golden set rather than one that
#: moves it. Within one build the simulator is deterministic, which
#: `unittests/Simulator/DeterminismTest.cpp` asserts across thread counts, so a
#: re-run that moved a bit moved it for a reason. The phases that intend to move
#: numbers are P9, P13 and P14, and each carries the declare then re-record step
#: of ground rule 7.
#:
#: The one caveat, stated rather than left to be discovered: this is a bound
#: between two runs of the *same* build. A different compiler or a different
#: host may contract a multiply and an add differently, so a baseline recorded
#: on one machine and checked on another can differ in the last bits without
#: anything being wrong. `--check` is a developer and orchestrator command for
#: that reason and is not a CI step at this phase.
GOLDEN_TOLERANCE: Final[float] = 0.0


class BaselineError(Exception):
    """Something the baseline cannot proceed without."""


# ---------------------------------------------------------------------------
# Locating the pieces.
# ---------------------------------------------------------------------------


def build_directory() -> Path:
    override = os.environ.get("NPU_BUILD_DIR")
    directory = Path(override) if override else REPO_ROOT / "build"
    if not (directory / "CMakeCache.txt").is_file():
        raise BaselineError(
            f"{directory} is not a configured build directory. Configure and "
            f"build first, or set NPU_BUILD_DIR."
        )
    return directory


def cache_value(name: str) -> str:
    cache = build_directory() / "CMakeCache.txt"
    pattern = re.compile(rf"^{re.escape(name)}:[^=]*=(.*)$")
    for line in cache.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return ""


def find_lit() -> Path:
    """The lit runner, from the same places the configure step looked.

    Four, in order: `LIT` in the environment; `LLVM_EXTERNAL_LIT` out of this
    build's cache, which is what CI sets; `llvm-lit` or `lit` on `PATH`; and the
    tools directory beside `LLVM_DIR`, which is where an LLVM build tree keeps
    it. A build tree configured the ordinary way records no
    `LLVM_EXTERNAL_LIT` at all, which is why the last branch exists.
    """
    override = os.environ.get("LIT")
    if override and Path(override).is_file():
        return Path(override)

    from_cache = cache_value("LLVM_EXTERNAL_LIT")
    if from_cache and Path(from_cache).is_file():
        return Path(from_cache)

    for name in ("llvm-lit", "lit"):
        found = shutil.which(name)
        if found:
            return Path(found)

    llvm_dir = cache_value("LLVM_DIR")
    if llvm_dir:
        candidate = Path(llvm_dir).parents[2] / "bin" / "llvm-lit"
        if candidate.is_file():
            return candidate

    raise BaselineError(
        "lit was not found. Looked at $LIT, LLVM_EXTERNAL_LIT in the build "
        "cache, llvm-lit and lit on PATH, and the bin directory beside "
        "LLVM_DIR."
    )


# ---------------------------------------------------------------------------
# The suites.
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    tests: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "tests": sorted(self.tests),
        }


def run_lit_suite(work: Path) -> SuiteResult:
    output = work / "lit.json"
    completed = subprocess.run(
        [str(find_lit()), "-q", f"--output={output}", str(build_directory() / "test")],
        capture_output=True,
        text=True,
        check=False,
    )
    if not output.is_file():
        raise BaselineError(
            "lit wrote no result file, so there is nothing to record.\n"
            + completed.stdout
            + completed.stderr
        )
    data = json.loads(output.read_text(encoding="utf-8"))

    result = SuiteResult()
    for test in data["tests"]:
        result.tests.append(test["name"])
        code = test["code"]
        if code in ("PASS", "XFAIL"):
            result.passed += 1
        elif code in ("UNSUPPORTED", "SKIPPED"):
            result.skipped += 1
        else:
            result.failed += 1
    return result


def run_gtest_binary(name: str, work: Path) -> SuiteResult | None:
    binary = build_directory() / "bin" / name
    if not binary.is_file():
        return None

    output = work / f"{name}.json"
    subprocess.run(
        [str(binary), f"--gtest_output=json:{output}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if not output.is_file():
        raise BaselineError(f"{name} wrote no JSON result, so it cannot be recorded")
    data = json.loads(output.read_text(encoding="utf-8"))

    result = SuiteResult()
    for suite in data.get("testsuites", []):
        for test in suite.get("testsuite", []):
            full = f"{suite['name']}.{test['name']}"
            result.tests.append(full)
            if test.get("result") == "SKIPPED" or test.get("status") == "NOTRUN":
                result.skipped += 1
            elif test.get("failures"):
                result.failed += 1
            else:
                result.passed += 1
    return result


def run_pytest(work: Path) -> SuiteResult:
    output = work / "pytest.xml"
    environment = dict(os.environ)
    bindings = cache_value("MLIR_PYTHON_PACKAGES_DIR")
    if bindings:
        environment["MLIR_PYTHON_PACKAGES_DIR"] = bindings

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test/Python",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "slow or not slow",
            f"--junitxml={output}",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if not output.is_file():
        raise BaselineError("pytest wrote no JUnit XML, so it cannot be recorded")

    result = SuiteResult()
    root = ElementTree.parse(output).getroot()
    for suite in root:
        for case in suite:
            result.tests.append(f"{case.get('classname')}::{case.get('name')}")
            kinds = {child.tag for child in case}
            if "skipped" in kinds:
                result.skipped += 1
            elif kinds & {"failure", "error"}:
                result.failed += 1
            else:
                result.passed += 1
    return result


def run_dash_lint() -> SuiteResult:
    """The dash lint and its self test, which Section 17.6 names beside the suites.

    It has no machine readable output and needs none: it is two commands and
    each either succeeds or does not. The names are the invocations, so a
    baseline reader sees which two ran.
    """
    result = SuiteResult()
    for arguments, name in (
        ([], "dash-lint.sh"),
        (["--self-test"], "dash-lint.sh --self-test"),
    ):
        completed = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts" / "dash-lint.sh"), *arguments],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        result.tests.append(name)
        if completed.returncode == 0:
            result.passed += 1
        else:
            result.failed += 1
    return result


def collect_suites(work: Path) -> dict[str, dict[str, Any]]:
    suites: dict[str, SuiteResult] = {"check-npu": run_lit_suite(work)}
    for name in GTEST_BINARIES:
        found = run_gtest_binary(name, work)
        if found is not None:
            suites[name] = found
    suites["pytest"] = run_pytest(work)
    suites["dash-lint"] = run_dash_lint()
    return {name: result.as_json() for name, result in suites.items()}


# ---------------------------------------------------------------------------
# The cells.
# ---------------------------------------------------------------------------


def _frontend() -> Any:
    """Imports the frontend, with this repository's package root on the path.

    Imported inside a function rather than at module scope so that `--help` and
    the argument parsing work in an environment without the MLIR bindings, and
    so the error a missing binding produces names this script's requirement
    rather than arriving as an import failure at line one.
    """
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import npu_frontend
    except ImportError as failure:  # pragma: no cover
        raise BaselineError(
            "the frontend could not be imported, so the cells cannot be "
            "measured. It needs the MLIR Python bindings on PYTHONPATH; see "
            "docs/BUILD.md.\n\n" + str(failure)
        ) from failure
    return npu_frontend


def collect_cells(work: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per model, level and budget, plus the golden tensors.

    The level axis holds one value at this phase and is written out anyway, so
    that a P9 baseline differs from a P8 one by rows rather than by shape.
    """
    frontend = _frontend()

    import numpy as np
    import onnxruntime as ort
    from npu_frontend.input_classes import make_inputs
    from npu_frontend.model_generator import DEFAULT_BUDGET, TIGHT_BUDGETS

    levels = frontend.implemented_levels()
    cells: list[dict[str, Any]] = []
    goldens: dict[str, Any] = {}

    for name, spec in frontend.MODELS.items():
        batch = spec.input_shape[0]
        onnx_path = frontend.generate_model(name, work)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_names = [entry.name for entry in session.get_inputs()]

        for level in levels:
            for budget_name, budget in (
                ("default", None),
                ("tight", TIGHT_BUDGETS[name]),
            ):
                program = frontend.compile_model(
                    onnx_path, level=level, emit="nbin", budget=budget
                )
                arrays = make_inputs(
                    "normal", program.input_shapes, model=name, batch=batch
                )
                expected = session.run(
                    None, dict(zip(input_names, arrays, strict=True))
                )
                answer = frontend.run_program(
                    program.binary, arrays, program.output_shapes
                )

                worst = 0.0
                for produced, reference in zip(answer.outputs, expected, strict=True):
                    worst = max(
                        worst,
                        float(
                            np.abs(
                                produced.astype(np.float64)
                                - np.asarray(reference, dtype=np.float64)
                            ).max()
                        ),
                    )

                statistics = answer.stats
                cells.append(
                    {
                        "model": name,
                        "level": level,
                        "budget": budget_name,
                        "budget_bytes": (
                            budget if budget is not None else DEFAULT_BUDGET
                        ),
                        "batch": batch,
                        # Section 10.2: the only instruction count in this
                        # project, taken as a field and never counted.
                        "instructions": statistics["instructions"],
                        "cycles": statistics["cycles"],
                        "dma_cycles": statistics["dma_cycles"],
                        "compute_cycles": statistics["compute_cycles"],
                        "dram_bytes_read": statistics["dram_bytes_read"],
                        "dram_bytes_written": statistics["dram_bytes_written"],
                        "macs": statistics["macs"],
                        "max_abs_error_vs_onnxruntime": worst,
                    }
                )

                # The golden tensors are per level and not per budget, because a
                # spill is a DMA round trip and the arithmetic is untouched.
                # `test_tight_budgets.py` asserts that rather than assuming it.
                if budget_name == "default":
                    for index, produced in enumerate(answer.outputs):
                        goldens[f"{name}-O{level}-out{index}"] = produced

    return cells, goldens


# ---------------------------------------------------------------------------
# The manifest.
# ---------------------------------------------------------------------------


def tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    npu_opt = build_directory() / "bin" / "npu-opt"
    if npu_opt.is_file():
        completed = subprocess.run(
            [str(npu_opt), "--version"], capture_output=True, text=True, check=False
        )
        for line in completed.stdout.splitlines():
            if "LLVM version" in line:
                versions["llvm"] = line.strip()
                break

    sys.path.insert(0, str(REPO_ROOT / "python"))
    for module in ("numpy", "onnx", "onnxruntime", "torch"):
        try:
            imported = __import__(module)
        except ImportError:
            continue
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    return versions


def git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def measure() -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="npu-baseline-") as directory:
        work = Path(directory)
        suites = collect_suites(work)
        cells, goldens = collect_cells(work)

    frontend = _frontend()
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
        "generator_version": frontend.GENERATOR_VERSION,
        "tool_versions": tool_versions(),
        "absent_fields": ABSENT_FIELDS,
        "suites": suites,
        "cells": cells,
    }
    return baseline, goldens


# ---------------------------------------------------------------------------
# Recording.
# ---------------------------------------------------------------------------


def write(baseline: dict[str, Any], goldens: dict[str, Any]) -> None:
    import numpy as np

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Every `.npy` that is no longer produced is removed, so the directory is
    # the golden set rather than the golden set plus whatever a previous shape
    # of the suite left behind.
    for stale in GOLDEN_DIR.glob("*.npy"):
        if stale.stem not in goldens:
            stale.unlink()
    for name, array in goldens.items():
        np.save(GOLDEN_DIR / f"{name}.npy", array)

    BASELINE_PATH.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"regression-baseline: wrote {BASELINE_PATH}")
    print(f"regression-baseline: wrote {len(goldens)} golden tensors to {GOLDEN_DIR}")


# ---------------------------------------------------------------------------
# Checking.
# ---------------------------------------------------------------------------


def cell_key(cell: dict[str, Any]) -> str:
    return f"{cell['model']}-O{cell['level']}-{cell['budget']}"


def compare(
    recorded: dict[str, Any], current: dict[str, Any], goldens: dict[str, Any]
) -> list[str]:
    """Every difference, as one line each. Empty means no drift."""
    import numpy as np

    drift: list[str] = []

    # Section 17.6: fail loudly on a schema_version this script does not
    # recognise rather than guessing what a field meant.
    version = recorded.get("schema_version")
    if version != SCHEMA_VERSION:
        raise BaselineError(
            f"the recorded baseline is schema_version {version!r} and this "
            f"script writes and reads {SCHEMA_VERSION}. A version this script "
            f"does not recognise is refused rather than guessed at: a field "
            f"that appeared in a later version would otherwise be read as a "
            f"regression from zero."
        )

    if recorded.get("generator_version") != current.get("generator_version"):
        drift.append(
            f"GENERATOR_VERSION: {recorded.get('generator_version')} -> "
            f"{current.get('generator_version')}. The model suite changed, so "
            f"every cell below is measuring a different suite."
        )

    # ---- suites ----------------------------------------------------------
    before, after = recorded["suites"], current["suites"]
    for name in sorted(set(before) | set(after)):
        if name not in after:
            drift.append(f"suite {name}: recorded and no longer runs")
            continue
        if name not in before:
            drift.append(f"suite {name}: runs and was not in the baseline")
            continue
        for field_name in ("passed", "failed", "skipped"):
            if before[name][field_name] != after[name][field_name]:
                drift.append(
                    f"suite {name}: {field_name} {before[name][field_name]} -> "
                    f"{after[name][field_name]}"
                )
        missing = sorted(set(before[name]["tests"]) - set(after[name]["tests"]))
        added = sorted(set(after[name]["tests"]) - set(before[name]["tests"]))
        for test in missing:
            drift.append(f"suite {name}: test gone: {test}")
        for test in added:
            drift.append(f"suite {name}: test added: {test}")

    # ---- cells -----------------------------------------------------------
    #
    # Only the fields present in both are compared, which is what lets a P8
    # baseline still be readable once P9 and P11 add theirs.
    recorded_cells = {cell_key(cell): cell for cell in recorded["cells"]}
    current_cells = {cell_key(cell): cell for cell in current["cells"]}
    for key in sorted(set(recorded_cells) | set(current_cells)):
        if key not in current_cells:
            drift.append(f"cell {key}: recorded and no longer produced")
            continue
        if key not in recorded_cells:
            drift.append(f"cell {key}: produced and not in the baseline")
            continue
        old, new = recorded_cells[key], current_cells[key]
        for field_name in sorted(set(old) & set(new)):
            if field_name in ("model", "level", "budget"):
                continue
            if old[field_name] != new[field_name]:
                drift.append(
                    f"cell {key}: {field_name} {old[field_name]} -> "
                    f"{new[field_name]}"
                )

    # ---- goldens ---------------------------------------------------------
    on_disk = {path.stem for path in GOLDEN_DIR.glob("*.npy")}
    for name in sorted(on_disk | set(goldens)):
        if name not in goldens:
            drift.append(f"golden {name}: recorded and no longer produced")
            continue
        if name not in on_disk:
            drift.append(f"golden {name}: produced and not recorded")
            continue
        expected = np.load(GOLDEN_DIR / f"{name}.npy")
        produced = goldens[name]
        if expected.shape != produced.shape:
            drift.append(f"golden {name}: shape {expected.shape} -> {produced.shape}")
            continue
        worst = float(np.abs(produced - expected).max())
        if worst > GOLDEN_TOLERANCE:
            drift.append(
                f"golden {name}: largest movement {worst:.6e}, above the "
                f"tolerance of {GOLDEN_TOLERANCE:.6e}"
            )

    return drift


def check() -> int:
    if not BASELINE_PATH.is_file():
        print(
            f"regression-baseline: {BASELINE_PATH} does not exist. Record one "
            f"first with scripts/regression-baseline.sh.",
            file=sys.stderr,
        )
        return 2

    recorded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current, goldens = measure()
    drift = compare(recorded, current, goldens)

    print("regression-baseline --check")
    print()
    print(f"  recorded at {recorded['git_sha'][:12]}, checked at {git_sha()[:12]}")
    print(f"  {len(current['cells'])} cells, {len(goldens)} golden tensors")
    for name, result in sorted(current["suites"].items()):
        print(
            f"  suite {name:20} {result['passed']:>4} passed  "
            f"{result['failed']:>3} failed  {result['skipped']:>3} skipped"
        )
    print()

    if not drift:
        print("regression-baseline: no drift.")
        return 0

    print(f"  {len(drift)} differences:")
    print()
    for line in drift:
        print(f"    {line}")
    print()
    print(
        "regression-baseline: FAIL. An optimization that moves a cycle count "
        "is not necessarily wrong, but it must never move silently. Re-record "
        "in its own commit, after declaring the intended movement in "
        "docs/BREAKING_CHANGES.md."
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regression_baseline.py",
        description=(
            "Record the regression baseline of Section 17.6, or re-run "
            "everything and diff against it."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run everything and fail on any drift, instead of recording",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.check:
            return check()
        baseline, goldens = measure()
        write(baseline, goldens)
        failed = {
            name: result["failed"]
            for name, result in baseline["suites"].items()
            if result["failed"]
        }
        if failed:
            print(
                f"regression-baseline: WARNING, the baseline was recorded with "
                f"failing suites: {failed}. A baseline recorded from a red "
                f"tree records what is broken as if it were correct.",
                file=sys.stderr,
            )
            return 1
        return 0
    except BaselineError as failure:
        print(f"regression-baseline: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
