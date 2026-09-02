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
# Section 17.6 is explicit. Per level fields arrived at P9, when levels above
# `-O0` first existed, and energy fields arrive at P11, when Accelergy lands. A
# baseline that claimed energy before then would be recording a number no phase
# had computed. So `absent_fields` names what is still missing and the phase
# that adds it, `--check` compares only the fields present in both, and an
# unrecognised `schema_version` is a loud failure rather than a guess.
#
# THE TWO BANDS, BECAUSE P9 IS WHERE THEY BECOME DIFFERENT NUMBERS
#
# `GOLDEN_TOLERANCE` is zero and bounds a **re-run against the recorded run at
# the same level**, which is a reproducibility question with no acceptable
# slack. Section 17.6's 1e-6 for P9 bounds a different quantity, **how far a
# level's answer sits from `-O0`'s**, which is recorded per cell as
# `max_abs_movement_vs_o0`. Confusing the two would turn the safety net off in
# the phase that most needs it.

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
#:
#: **2 at P9**, which is the bump Section 17.6 asks for when the per level
#: fields arrive. What changed: `per_level` left `absent_fields`, the manifest
#: gained a `levels` list, every cell gained `max_abs_movement_vs_o0`, and the
#: cell and golden sets grew from one level to three. A P8 baseline is refused
#: rather than read, which is the point of the version: at version 1 a reader
#: would find fourteen cells where there are forty two and report twenty eight
#: of them as regressions from nothing.
#: **3 at P11**, which is the bump Section 17.6 asks for when the energy fields
#: arrive, declared in `docs/BREAKING_CHANGES.md` before the commit that caused
#: it. What changed: `energy` left `absent_fields`, every cell gained
#: `energy_pj_per_inference`, and the manifest gained `technology_node`,
#: `registered_estimators` and `energy_per_action_pj`. A version 2 baseline is
#: refused rather than read.
SCHEMA_VERSION: Final[int] = 3

#: The fields Section 17.6 names and this phase cannot compute, with the phase
#: that adds each. Recorded in the file itself, so a reader of a P9 baseline at
#: P14 is told why the field is missing rather than left to infer it.
#:
#: `per_level` was here at P8 and left at P9 with the levels themselves.
#: `energy` was here from P8 and left at P11 with Accelergy.
#:
#: **Empty is a legitimate state and is not the same as absent.** The key stays
#: in the recorded file carrying `{}`, which says "this baseline claims every
#: field the schema has" rather than "nobody wrote this down".
ABSENT_FIELDS: Final[dict[str, str]] = {}

#: The technology node the recorded energy is at, from the energy module's one
#: home rather than restated here.
#:
#: **Why the baseline records the per action table and not just the totals.**
#: `--check` re-runs everything and compares, and it runs in CI, where Accelergy
#: is not installed. Recomputing the energy there would either need the tool or
#: silently skip the field. So the coefficients are recorded **once**, at record
#: time, and `--check` recomputes each cell's energy from the current action
#: counts against the **recorded** coefficients. That makes the energy field a
#: drift check on the counts, which is the thing that can move, and makes a
#: coefficient change visible as a manifest difference rather than as a hundred
#: cell differences with no cause.
ENERGY_COMPONENTS: Final[tuple[str, ...]] = ("mac_array", "scratchpad", "main_memory")

#: The GoogleTest binaries, in the order the CI job runs them.
GTEST_BINARIES: Final[tuple[str, ...]] = (
    "NPUInterfaceTests",
    "NPUTilingTests",
    "NPUAllocatorTests",
    "NPUEncodingTests",
    "NPUSimulatorTests",
)

#: The golden tolerance, which is zero, and stays zero at P9.
#:
#: Section 17.6 puts P9 in the "within 1e-6, with the movement justified in
#: writing" class, and this constant is **not** that number. The distinction is
#: worth stating because getting it backwards would turn the safety net off in
#: the phase that most needs it.
#:
#: `--check` compares a re-run against the recorded run **at the same level**.
#: Within one build the simulator is deterministic, which
#: `unittests/Simulator/DeterminismTest.cpp` asserts across thread counts, so a
#: re-run that moved a bit moved it for a reason and there is no band inside
#: which that is acceptable. The 1e-6 of Section 17.6 bounds a different
#: quantity: how far a level's answer may sit from `-O0`'s. That one is recorded
#: per cell as `max_abs_movement_vs_o0`, declared in
#: `docs/BREAKING_CHANGES.md`, and asserted by `test_regression_baseline.py`.
#: Two bands, two questions, and this is the one about reproducibility.
#:
#: The one caveat, stated rather than left to be discovered: this is a bound
#: between two runs of the *same* build. A different compiler or a different
#: host may contract a multiply and an add differently, so a baseline recorded
#: on one machine and checked on another can differ in the last bits without
#: anything being wrong.
#:
#: **P8 and P9 made that caveat the reason `--check` was not a CI step**, and
#: P9b takes the other route: the step is switched on with the tolerance left at
#: zero, and if the container disagrees with the developer machine then the
#: disagreement is the measurement this project did not have. Widening the band
#: first and measuring afterwards would have thrown away the only number the
#: question is about. What the first red run has to be able to say is *how far*,
#: on how many elements, and in what unit, which is what the drift line for a
#: golden was rewritten to carry.
#:
#: **And it held.** On the step's first CI runs the whole numeric half of the
#: file reproduced bit for bit under clang in the container against a baseline
#: recorded under gcc on WSL2: every cell, every golden tensor, every suite
#: count. This constant stays at zero on evidence. The one field that did move
#: is below, and it is not this compiler's.
GOLDEN_TOLERANCE: Final[float] = 0.0

#: The one cell field that is **not** compared for equality, and why it is the
#: only one.
#:
#: `max_abs_error_vs_onnxruntime` is the distance between this compiler's answer
#: and `onnxruntime`'s for the same graph and the same input. It has two ends,
#: and only one of them belongs to this project.
#:
#: **This compiler's end is pinned bit for bit already.** The golden tensors do
#: it at `GOLDEN_TOLERANCE` of zero for every default budget cell, and
#: `test/Python/test_tight_budgets.py` does it for the rest by asserting that a
#: tight budget answer is bit identical to the default budget one. So given a
#: green golden comparison, **this field cannot move because of anything this
#: compiler did**. Everything it can still report is a change at the other end.
#:
#: **And the other end moves per host.** P9b measured it. Two CI runs that
#: landed on different GitHub runner hardware reported eighteen cells with a
#: different distance, three models at every level and both budgets, moving
#: between 1e-8 and 1e-7 **in both directions**, with no golden tensor and no
#: cycle count moving at all. `onnxruntime` dispatches its CPU kernels on what
#: the host supports. `npu_frontend.tolerances` has said so in prose since P8;
#: this is the measurement behind the prose. D-0039.
#:
#: So equality here was asserting that two machines choose the same
#: vectorisation, and buying no coverage of this project in exchange. What is
#: asserted instead is the field's one load bearing meaning, Section 17.4's end
#: to end band: the answer is still inside the tolerance the matrix enforces.
#: The recorded value stays in the file as what the recording host measured,
#: which is the status `tool_versions` already has and for the same reason.
ORACLE_FIELD: Final[str] = "max_abs_error_vs_onnxruntime"


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

    **A failure prints the child's output, and that is D-0038.** Every other
    suite here writes a machine readable file, so a failing test reaches the
    drift report by name and the reader knows what broke. This one has no such
    file, so its whole contribution is a count, and the first CI run of the
    `--check` step reported `suite dash-lint: passed 2 -> 0` with nothing about
    why. The why was a `SyntaxError`: the linter did not parse under the
    container's interpreter, and it had been printing that to a pipe nobody
    read. Two lines of stderr turn a diagnosis into a glance, which is the
    standard the golden drift lines were rewritten to meet in the same phase.
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
            print(
                f"regression-baseline: {name} exited {completed.returncode}. "
                f"Its output follows, because a count of zero says nothing a "
                f"reader can act on:",
                file=sys.stderr,
            )
            for stream in (completed.stdout, completed.stderr):
                for line in stream.rstrip().splitlines():
                    print(f"  {line}", file=sys.stderr)
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

    *Three levels since P9.* The axis was written out at P8 with one value in
    it, which is why a P9 baseline differs from a P8 one by rows rather than by
    shape: forty two cells where there were fourteen, twenty one golden tensors
    where there were seven, and one new field per cell.

    That field is `max_abs_movement_vs_o0`, and it is the per level number
    Section 17.6 and P9's gate ask for. It is the largest absolute distance
    between this cell's answer and the same model's answer at `-O0` under the
    same budget, so it is zero by construction at `-O0` and is the phase's own
    numerics claim everywhere else. Recording it per cell rather than as one
    headline is what lets a later phase see *which* cell moved when the headline
    changes.
    """
    frontend = _frontend()

    import numpy as np
    import onnxruntime as ort
    from npu_frontend.input_classes import make_inputs
    from npu_frontend.model_generator import DEFAULT_BUDGET, TIGHT_BUDGETS

    levels = frontend.implemented_levels()
    cells: list[dict[str, Any]] = []
    goldens: dict[str, Any] = {}
    # The `-O0` answer per model and budget, kept so that every higher level's
    # movement is measured against it. `implemented_levels()` is ordered and
    # starts at zero, which the assertion below states rather than assumes,
    # because a reordering would silently make every movement zero.
    at_zero: dict[tuple[str, str], list[Any]] = {}
    if levels and levels[0] != 0:
        raise BaselineError(
            "the compiler's implemented levels do not start at -O0, so there is "
            "no unoptimized answer to measure this phase's movement against. "
            f"They are {levels}."
        )

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

                if level == 0:
                    at_zero[(name, budget_name)] = [
                        array.copy() for array in answer.outputs
                    ]
                    movement = 0.0
                else:
                    movement = max(
                        float(
                            np.abs(
                                produced.astype(np.float64)
                                - unoptimized.astype(np.float64)
                            ).max()
                        )
                        for produced, unoptimized in zip(
                            answer.outputs, at_zero[(name, budget_name)], strict=True
                        )
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
                        # The scratchpad port's traffic, recorded from P11
                        # because it is what Accelergy's scratchpad action
                        # counts are and the energy field below is computed
                        # from it. `effective_macs` is deliberately not here:
                        # Section 5.5 keeps it out of the energy path and the
                        # cheapest way to keep it out is not to record it.
                        "scratchpad_elements_read": statistics[
                            "scratchpad_elements_read"
                        ],
                        "scratchpad_elements_written": statistics[
                            "scratchpad_elements_written"
                        ],
                        "max_abs_error_vs_onnxruntime": worst,
                        # Section 17.6's per level field and P9's own gate
                        # clause: how far this level's answer sits from the
                        # unoptimized one. Zero at -O0 by construction.
                        "max_abs_movement_vs_o0": movement,
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


def energy_tables(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], str, list[str]]:
    """Accelergy's per action coefficients, one table per distinct budget.

    Run once, at record time, and recorded in the baseline so that `--check` can
    recompute every cell's energy without the tool. See `ENERGY_COMPONENTS` for
    why that is the arrangement rather than re-running Accelergy in CI.
    """
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    # Imported here rather than at module scope, and by path rather than as a
    # package, because `experiments/` is a directory of scripts and not an
    # importable package: `--check` must work without it on `sys.path` and
    # without Accelergy installed at all.
    import accelergy_energy  # type: ignore[import-not-found]

    tables: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="npu-baseline-energy-") as directory:
        estimator = accelergy_energy.Estimator(Path(directory))
        for cell in cells:
            budget = str(int(cell["budget_bytes"]))
            if budget in tables:
                continue
            estimate = estimator.estimate(
                {
                    "cell": {
                        "name": f"baseline-{budget}",
                        "scratchpad_budget_bytes": int(cell["budget_bytes"]),
                    },
                    "simulation": {
                        "macs": int(cell["macs"]),
                        "scratchpad_elements_read": int(
                            cell["scratchpad_elements_read"]
                        ),
                        "scratchpad_elements_written": int(
                            cell["scratchpad_elements_written"]
                        ),
                        "dram_bytes_read": int(cell["dram_bytes_read"]),
                        "dram_bytes_written": int(cell["dram_bytes_written"]),
                    },
                }
            )
            tables[budget] = {
                component: dict(estimate.energy_per_action_pj[component])
                for component in ENERGY_COMPONENTS
            }
        return tables, accelergy_energy.TECHNOLOGY_NODE, estimator.registered


def energy_of(cell: dict[str, Any], tables: dict[str, Any]) -> float:
    """One cell's picojoules per inference, from its counts and one table.

    The counts are this project's own and the coefficients are Accelergy's, which
    is Section 16.4's honest division: only the coefficients are external, and
    every counting bug in the simulator propagates straight into this number.

    `macs` is **raw**. Section 5.5 forbids the energy path from seeing
    `effective_macs`, and the baseline does not record it, so the wrong figure is
    not merely unused here but unavailable.
    """
    table = tables[str(int(cell["budget_bytes"]))]
    # A DRAM cannot fetch part of a word, so a partial access is paid in full.
    # `accelergy_energy.dram_accesses` is the one home for that rounding and its
    # docstring carries the reason; this restates the arithmetic rather than
    # importing the module, because `--check` must not need it installed.
    dram_access_bytes = 8

    def accesses(byte_count: int) -> int:
        return -(-byte_count // dram_access_bytes)

    return (
        table["mac_array"]["mac"] * int(cell["macs"])
        + table["scratchpad"]["read"] * int(cell["scratchpad_elements_read"])
        + table["scratchpad"]["write"] * int(cell["scratchpad_elements_written"])
        + table["main_memory"]["read"] * accesses(int(cell["dram_bytes_read"]))
        + table["main_memory"]["write"] * accesses(int(cell["dram_bytes_written"]))
    )


def measure(
    energy_from: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Everything the baseline records.

    `energy_from` is the coefficient table a previously recorded baseline
    carries. `--check` passes it so the energy is recomputed against the same
    coefficients rather than against a tool that may not be installed; recording
    passes nothing and runs Accelergy.
    """
    with tempfile.TemporaryDirectory(prefix="npu-baseline-") as directory:
        work = Path(directory)
        suites = collect_suites(work)
        cells, goldens = collect_cells(work)

    if energy_from is None:
        tables, node, estimators = energy_tables(cells)
    else:
        tables = energy_from["energy_per_action_pj"]
        node = energy_from["technology_node"]
        estimators = energy_from["registered_estimators"]
    for cell in cells:
        cell["energy_pj_per_inference"] = energy_of(cell, tables)

    frontend = _frontend()
    baseline = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
        "generator_version": frontend.GENERATOR_VERSION,
        "tool_versions": tool_versions(),
        "absent_fields": ABSENT_FIELDS,
        # Section 16.4's pinned node, and the estimator list Section 16.1 asks
        # for: two runs at the same Accelergy sha with different plug ins are two
        # different measurements.
        "technology_node": node,
        "registered_estimators": estimators,
        "energy_per_action_pj": tables,
        # The levels this baseline covers, as a field rather than as something a
        # reader infers by grouping the cells. *Added at P9.* It is what makes
        # "the baseline records -O0 only" and "the baseline records all three"
        # different statements a test can check, and the P8 form of that check
        # was reading the cells to find out.
        "levels": sorted({int(cell["level"]) for cell in cells}),
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


def oracle_bound() -> float:
    """Section 17.4's absolute end to end band, imported and never restated.

    It lives in `npu_frontend.tolerances` with the measurement it was set from
    and the argument for its size, and `test/Python/test_end_to_end.py` imports
    the same constant. Two copies of a tolerance is the duplication D-0032's fix
    made a test hunt for, and a tolerance is the worst thing to have two of:
    the copies agree until somebody widens one.
    """
    return float(_frontend().ABSOLUTE_TOLERANCE)


def oracle_notes(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """The oracle distances that moved and stayed inside the band.

    Reported rather than ignored. A field this script has stopped comparing for
    equality is a check that was switched off, and this project's rule is that a
    step which is off says so in its own output. Silence and a green run must
    not look the same, so the movement is printed with its magnitude and the
    reader gets to disagree with the decision.
    """
    bound = oracle_bound()
    recorded_cells = {cell_key(cell): cell for cell in recorded["cells"]}
    notes: list[str] = []
    for cell in current["cells"]:
        key = cell_key(cell)
        old = recorded_cells.get(key)
        if old is None or ORACLE_FIELD not in old or ORACLE_FIELD not in cell:
            continue
        was, now = float(old[ORACLE_FIELD]), float(cell[ORACLE_FIELD])
        if was == now or now > bound:
            continue
        notes.append(
            f"cell {key}: {ORACLE_FIELD} {was:.6e} -> {now:.6e} "
            f"({'further from' if now > was else 'closer to'} the oracle by "
            f"{abs(now - was):.3e}), inside the band of {bound:.6e}"
        )
    return notes


def compare(
    recorded: dict[str, Any], current: dict[str, Any], goldens: dict[str, Any]
) -> list[str]:
    """Every difference, as one line each. Empty means no drift."""
    import numpy as np

    drift: list[str] = []
    bound = oracle_bound()

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

    if recorded.get("levels") != current.get("levels"):
        drift.append(
            f"levels: {recorded.get('levels')} -> {current.get('levels')}. The "
            f"compiler builds a different set of optimization levels than the "
            f"one this baseline was recorded from, so the cells below are not "
            f"the same cells."
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
            # The distance to onnxruntime is bounded rather than fixed, for the
            # reason ORACLE_FIELD carries. A value outside the band is drift; a
            # value inside it that moved is a note, reported by oracle_notes.
            if field_name == ORACLE_FIELD:
                if float(new[field_name]) > bound:
                    drift.append(
                        f"cell {key}: {field_name} {new[field_name]:.6e} is "
                        f"outside the end to end band of {bound:.6e}. The "
                        f"baseline recorded {float(old[field_name]):.6e} on the "
                        f"recording host. This field is not compared for "
                        f"equality, because its other end is onnxruntime and "
                        f"that moves per host, so a value out here is the "
                        f"answer having genuinely left Section 17.4's "
                        f"tolerance rather than a runner difference"
                    )
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
        difference = np.abs(produced.astype(np.float64) - expected.astype(np.float64))
        worst = float(difference.max())
        if worst > GOLDEN_TOLERANCE:
            # **This line is written for a reader who has only the log.** Since
            # P9b `--check` is a CI step, and the run that matters most is the
            # first one in the container, where the question is whether a
            # baseline recorded under gcc on a developer machine reproduces bit
            # for bit under clang on a hosted runner. "largest movement 4.7e-08"
            # does not answer it: one element moved in the last bit and every
            # element moved by the same amount are the same sentence and
            # opposite findings. So the line carries how many elements moved,
            # where the worst one is, both values at that index, and the
            # movement in units in the last place at that scale, which is the
            # unit that separates an arithmetic difference from a bug.
            differing = int(np.count_nonzero(difference))
            index = np.unravel_index(int(difference.argmax()), difference.shape)
            was = float(expected[index])
            now = float(produced[index])
            ulp = float(np.spacing(np.float32(max(abs(was), abs(now)))))
            in_ulps = f"{worst / ulp:.1f} ulps" if ulp > 0.0 else "unmeasurable ulps"
            drift.append(
                f"golden {name}: {differing} of {difference.size} elements "
                f"differ, largest movement {worst:.6e} at index "
                f"{tuple(int(value) for value in index)}, where the baseline "
                f"records {was!r} and this run produced {now!r}, which is "
                f"{in_ulps} at that scale, against a tolerance of "
                f"{GOLDEN_TOLERANCE:.6e}"
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
    # The recorded coefficients, so that `--check` needs no Accelergy. A
    # coefficient that moved shows up as a manifest difference rather than as a
    # hundred unexplained cell differences, which is the more readable failure
    # and the one that names its own cause.
    current, goldens = measure(energy_from=recorded)
    drift = compare(recorded, current, goldens)
    notes = oracle_notes(recorded, current)

    print("regression-baseline --check")
    print()
    print(f"  recorded at {recorded['git_sha'][:12]}, checked at {git_sha()[:12]}")
    print(
        f"  {len(current['cells'])} cells, {len(goldens)} golden tensors, "
        f"levels {', '.join(f'-O{level}' for level in current['levels'])}"
    )
    worst = max(
        (float(cell["max_abs_movement_vs_o0"]) for cell in current["cells"]),
        default=0.0,
    )
    print(f"  largest movement against -O0: {worst:.3e}")
    for name, result in sorted(current["suites"].items()):
        print(
            f"  suite {name:20} {result['passed']:>4} passed  "
            f"{result['failed']:>3} failed  {result['skipped']:>3} skipped"
        )
    print()

    if notes:
        print(
            f"  {len(notes)} oracle distances moved and are inside the band. "
            f"Not drift, and not silence either:"
        )
        print()
        for line in notes:
            print(f"    {line}")
        print()
        print(
            "  max_abs_error_vs_onnxruntime is bounded rather than fixed, "
            "because its other end is onnxruntime and onnxruntime dispatches "
            "its CPU kernels on what the host supports. This compiler's end is "
            "pinned bit for bit by the golden tensors above, at a tolerance of "
            "zero, so a moved distance with green goldens is the oracle "
            "moving. See D-0039."
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
