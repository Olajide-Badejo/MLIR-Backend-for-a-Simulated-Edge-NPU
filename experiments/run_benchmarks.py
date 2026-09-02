# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The benchmark suite of Sections 16.1 and 16.2, one JSON per cell.

    python experiments/run_benchmarks.py
    python experiments/run_benchmarks.py --models lenet --force
    python experiments/run_benchmarks.py --budget-minutes 90

## The cells, computed and never written out

**Benchmark cells:** 7 models times 3 levels times 3 budget and batch
combinations, which is **63**. **Ablation cells:** the ablatable `-O2` set, read
from the driver at run time, times 7 models times 2 budgets, which is 8 times 7
times 2 and therefore **112**. **175 in total.**

Two of those numbers differ from Section 2's arithmetic and both differences are
recorded rather than reconciled away.

**Eight ablatable passes, not eleven.** Section 2 multiplies 11 by 7 by 2 to
reach 154 ablation cells. Three of Section 12's eleven,
`-npu-assign-layout`, `-npu-tile-to-scratchpad` and `-npu-double-buffer`, arrive
at P13 and no `-O` level names them yet, because a level that named a pass
nothing implements would give the ablation table a row it could not fill. The
eight is never written down here: `ablatable_passes(2)` reads it out of the
compiler, which is Section 16.2's rule and exists so that a pass added to `-O2`
and marked ablatable is swept the day it lands rather than the day somebody
remembers this file.

**The tight budget is not an axis the batch axis is free to cross, and this is a
P10 decision recorded in `docs/adr/0010`.** Section 17.4's matrix sweeps budget
and batch independently, which reads as 2 times 2. It is not, and the reason is
in ADR 0008's own definition: a tight budget is the smallest budget at which
**that program** allocates, measured from its allocated peak. A model at batch 4
is a different program from the same model at batch 1, and the peak measured on
2026-09-01 says how different:

| Model | Peak at its declared batch | Peak at batch 4 | Tight budget allocates at batch 4 |
|---|---|---|---|
| `lenet` | 194592 | 200800 | no |
| `depthwise_separable` | 8192 | 32768 | no |
| `resnet_block` | 8480 | 26912 | no |
| `inception_block` | 6848 | 24576 | no |
| `conv_bn_relu_stack` | 6432 | 18720 | no |
| `dilated_stack` | 8036 | 32080 | no |
| `lenet_batched` | 200800 | 200800 | yes, it is declared at batch 4 |

Six of the seven are declared at batch 1 and none of them allocates at batch 4
under its recorded tight budget. **That is not fragmentation and not an allocator
limitation**; it is the mechanism ADR 0008 already names, a single buffer larger
than the whole budget, whose remedy is tiling rather than spilling and whose
phase is P13.

So **a cell that names the tight budget runs at the model's declared batch**, and
a cell that names the default budget runs at both. Nothing in ADR 0008 moves and
no recorded constant changes: extending the tight budgets to a second batch would
be a re-measurement, and the phase state hands that to P13, which is also the
phase that makes a smaller budget reachable at all.

**The measured cost per cell is printed at the end of every run**, and that
figure is what replaces Section 2's 15 second planning number.

## The suite runs serially, on purpose

Every cell carries a wall clock with a confidence interval. Cells competing for
cores measure the contention rather than the compiler, so parallelising this
suite would corrupt the one group of fields it exists to produce. That is a
deliberate departure from `scripts/regression-baseline.sh`, which parallelises
freely because it records no timing at all.

The budget has room for it. Serial is what the measurement wants and the measured
runtime is the evidence that it costs nothing worth having.

## What each cell does

One instrumented compile through the real `-O` pipeline, repeated
`n_trials = 10` times for the timing object, then encode, simulate, and compare
against `onnxruntime`. Section 16.1 is explicit that the ten repetitions are of
**the whole instrumented pipeline**, each recompiling the same input through the
same single pass manager, and that this is not ten runs of one pass in isolation,
which Section 16.2 forbids.

Every trial loads its pass statistics with the expectation the level's own
description produces, so **a pass present in the pipeline and absent from the
JSON raises** and takes the run down. On an ablation cell the expectation is the
level's list minus the named pass, which is how a removal that silently did not
happen is caught by measurement rather than trusted to a flag.

## The gates this file enforces

- **The 90 minute budget of Section 2.** `--budget-minutes` defaults to 90 and
  the run **fails** when the measured suite runtime exceeds it. That is what
  "measured and enforced as a gate" means, and it is stated the same way here and
  in Section 2.
- **An ablation whose numerics move beyond tolerance fails the run.** Section
  16.2: it means the removed pass is load bearing for correctness rather than for
  performance, which is a finding rather than a table entry. The band is
  `npu_frontend.tolerances.ABSOLUTE_TOLERANCE`, imported and never restated.
- **A pass in the pipeline and absent from the statistics raises.**
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mlir_python_packages_dir() -> Path:
    """Where the MLIR Python bindings live.

    The same three step resolution `test/Python/conftest.py` uses, repeated
    rather than imported because a conftest is pytest's file and an experiment is
    not run under pytest. `experiments/allocator_fragmentation.py` carries the
    same block for the same reason.
    """
    override = os.environ.get("MLIR_PYTHON_PACKAGES_DIR")
    if override:
        return Path(override)

    cache = REPO_ROOT / "build" / "CMakeCache.txt"
    if cache.is_file():
        pattern = re.compile(r"^MLIR_PYTHON_PACKAGES_DIR:[^=]*=(.*)$")
        for line in cache.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if match and match.group(1).strip():
                return Path(match.group(1).strip())

    return (
        Path.home()
        / "llvm-project"
        / "build"
        / "tools"
        / "mlir"
        / "python_packages"
        / "mlir_core"
    )


sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from npu_frontend import cost_model  # noqa: E402
from npu_frontend.compile import (  # noqa: E402
    ablatable_passes,
    compile_model,
    describe_pipeline,
    implemented_levels,
    run_program,
)
from npu_frontend.input_classes import make_inputs  # noqa: E402
from npu_frontend.model_generator import (  # noqa: E402
    DEFAULT_BUDGET,
    GENERATOR_VERSION,
    MODELS,
    TIGHT_BUDGETS,
    generate_model,
)
from npu_frontend.pass_stats import (  # noqa: E402
    cross_check_against_mlir_timing,
    expected_passes,
    load_pass_stats,
)
from npu_frontend.predictions import landing_sha  # noqa: E402
from npu_frontend.results import (  # noqa: E402
    RANKING_METRIC,
    RESULTS_DIR,
    SCHEMA_VERSION,
    TRIALS,
    UNITS_CONVENTION,
    CellKey,
    ResultSchemaError,
    content_hash,
    host_manifest,
    load_result,
    null,
    timing_object,
    write_result,
)
from npu_frontend.tolerances import ABSOLUTE_TOLERANCE  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "experiments"))

import accelergy_energy  # noqa: E402
import roofline  # noqa: E402
import scalesim_export  # noqa: E402

#: Section 2's budget, in minutes, and the gate this file enforces.
DEFAULT_BUDGET_MINUTES = 90.0

#: The default seed for the run order. Recorded in every manifest, so a run is
#: reproducible in its ordering as well as in its numbers.
DEFAULT_RUN_ORDER_SEED = 20260901

#: The one input class the benchmark cells use. Section 17.4's five are the
#: pytest matrix's axis; a benchmark cell measures one input and says which.
INPUT_CLASS = "normal"

#: The prediction the `-O2` batch 1 cells and every ablation cell are evidence
#: for. Read as an id and resolved to a sha through git, because an entry cannot
#: state the sha of the commit that contains it.
ABLATION_PREDICTION = "p10-ablation-deltas"

#: The allocator's own attributes, read back off the allocated function.
_ATTRIBUTE = re.compile(r"npuisa\.(\w+) = ([-0-9.e+]+)")


class BenchmarkError(Exception):
    """The run cannot proceed, or has found something that fails it."""


# ---------------------------------------------------------------------------
# The two metrics Section 16.1 records and forbids ranking on.
# ---------------------------------------------------------------------------


def _top1_agreement(produced: list[np.ndarray], oracle: list[Any]) -> float:
    """The fraction of rows where the two answers pick the same largest element.

    Over the last axis, averaged across everything in front of it, which is the
    only reading that means anything for a tensor that is not a logit vector.

    Recorded and **not** used to rank, per Section 16.1. On this suite it is
    expected to sit at or very near 1.0, and that is the point being made rather
    than a result: the models are seeded and never trained, so the comparison is
    between two computations of the same untrained function and agreement is
    close to guaranteed. A metric that cannot separate configurations is a metric
    that must not order them.
    """
    matches = 0
    rows = 0
    for got, expected in zip(produced, oracle, strict=True):
        reference = np.asarray(expected, dtype=np.float64)
        wide = got.astype(np.float64)
        if wide.ndim == 0 or wide.shape[-1] < 2:
            # A single element last axis has exactly one argmax, so agreement is
            # one by construction and would inflate the average with a row that
            # asserted nothing.
            continue
        got_flat = wide.reshape(-1, wide.shape[-1])
        reference_flat = reference.reshape(-1, reference.shape[-1])
        matches += int(
            np.count_nonzero(
                np.argmax(got_flat, axis=-1) == np.argmax(reference_flat, axis=-1)
            )
        )
        rows += int(got_flat.shape[0])
    return float(matches) / rows if rows else 1.0


def _cosine_similarity(produced: list[np.ndarray], oracle: list[Any]) -> float:
    """Cosine similarity over every output flattened and concatenated.

    Recorded and not used to rank, for the reason above: on this suite it parks
    at four nines and has no resolution.
    """
    got = np.concatenate([array.astype(np.float64).ravel() for array in produced])
    reference = np.concatenate(
        [np.asarray(array, dtype=np.float64).ravel() for array in oracle]
    )
    denominator = float(np.linalg.norm(got)) * float(np.linalg.norm(reference))
    if denominator == 0.0:
        # Both answers are the zero vector, which happens on the classes that put
        # every value on a ReLU's dead side. Two identical zero vectors are as
        # similar as two vectors get, and the alternative, a division by zero
        # recorded as a null, would say the measurement failed when it did not.
        return 1.0
    return float(np.dot(got, reference)) / denominator


# ---------------------------------------------------------------------------
# The cell set.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One cell of the suite, with everything needed to run it."""

    key: CellKey
    budget_bytes: int

    @property
    def name(self) -> str:
        return self.key.name


#: Section 17.4's batch axis.
BATCHES = (1, 4)


def declared_batch(model: str) -> int:
    """The batch the model's own shape declares, which its tight budget was measured at.

    One for six of the seven and four for `lenet_batched`, which exists to be the
    batched one. This is the batch every tight budget cell runs at, per ADR 0010.
    """
    return int(MODELS[model].input_shape[0])


def budgets_and_batches(model: str) -> list[tuple[str, int, int]]:
    """The budget and batch combinations one model is measured at.

    `(budget name, budget bytes, batch)`. The default budget at both batches, the
    tight budget at the model's declared batch only. ADR 0010 and the module
    docstring carry the reason and the measurement behind it.
    """
    combinations = [("default", DEFAULT_BUDGET, batch) for batch in BATCHES]
    combinations.append(("tight", TIGHT_BUDGETS[model], declared_batch(model)))
    return combinations


def planned_cells(models: list[str] | None = None) -> list[Cell]:
    """The cell set, computed from the registries and the compiler.

    Neither the level list nor the ablatable set is written here. The levels come
    from `implemented_levels()` and the ablatable set from `ablatable_passes(2)`,
    both of which are the compiler describing itself, so this function's answer
    moves when the compiler does.
    """
    names = list(models) if models else list(MODELS)
    unknown = [name for name in names if name not in MODELS]
    if unknown:
        raise BenchmarkError(f"{unknown} are not models of the suite: {sorted(MODELS)}")

    levels = implemented_levels()
    cells: list[Cell] = []

    for model in names:
        for budget_name, budget_bytes, batch in budgets_and_batches(model):
            for level in levels:
                cells.append(
                    Cell(
                        key=CellKey(
                            model=model,
                            opt_level=level,
                            scratchpad_budget=budget_name,
                            batch=batch,
                            input_class=INPUT_CLASS,
                        ),
                        budget_bytes=budget_bytes,
                    )
                )

        # Section 16.2: an ablation row for every ablatable `-O2` pass **at every
        # budget**, because passes can behave oppositely at a tight budget and a
        # table that only reports the generous one hides it.
        #
        # At the model's declared batch for both budgets, which is what makes
        # every ablation row's `baseline_cell` a cell this suite also measured. A
        # row whose baseline is not in the run would be a subtraction with one
        # operand, and `fill_deltas` refuses it.
        batch = declared_batch(model)
        for budget_name, budget_bytes in (
            ("default", DEFAULT_BUDGET),
            ("tight", TIGHT_BUDGETS[model]),
        ):
            for ablated in ablatable_passes(2):
                cells.append(
                    Cell(
                        key=CellKey(
                            model=model,
                            opt_level=2,
                            scratchpad_budget=budget_name,
                            batch=batch,
                            input_class=INPUT_CLASS,
                            ablated_pass=ablated,
                        ),
                        budget_bytes=budget_bytes,
                    )
                )
    return cells


# ---------------------------------------------------------------------------
# The manifest's run half.
# ---------------------------------------------------------------------------


def git_sha() -> str:
    """The commit every cell of this run records as `manifest.git_sha`.

    **Raises rather than recording the string `unknown`**, which is D-0042. That
    fallback would put a manifest into every committed file naming no commit, and
    law 3 of Section 0.2 is that every published number traces to one that
    resolves. A run that cannot say which commit it measured has not produced
    results worth keeping, and failing before the first cell costs nothing beside
    discovering it after 175 of them.
    """
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        raise BenchmarkError(
            f"`git rev-parse HEAD` exited {completed.returncode}, so this run "
            f"cannot record which commit it measured. Recording `unknown` would "
            f"write a file that traces to nothing, which is the one thing law 3 "
            f"of Section 0.2 forbids.\n\n"
            f"git said:\n{completed.stderr.strip() or '(nothing on stderr)'}"
        )
    return sha


def llvm_tag() -> str:
    from npu_frontend.builder import find_tool

    completed = subprocess.run(
        [str(find_tool("npu-opt")), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if "LLVM version" in line:
            return line.strip()
    return "unknown"


def tool_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("numpy", "onnx", "onnxruntime", "torch"):
        try:
            imported = __import__(module)
        except ImportError:  # pragma: no cover
            continue
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    versions["llvm"] = llvm_tag()
    return versions


# ---------------------------------------------------------------------------
# Running one cell.
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """The `-O0` answer a cell's movement and SQNR are measured against.

    Computed on demand and cached per model, budget and batch. It has to be on
    demand because the run order is randomized under a seed, so a cell can be
    reached before its own reference has been.
    """

    outputs: list[np.ndarray]


class Runner:
    """The suite, with the caches that stop it recompiling what it has."""

    def __init__(self, work: Path, hash_of_sources: str) -> None:
        self._work = work
        self._hash = hash_of_sources
        self._onnx: dict[tuple[str, int], Path] = {}
        self._sessions: dict[tuple[str, int], ort.InferenceSession] = {}
        self._references: dict[tuple[str, str, int], Reference] = {}
        self._tool_versions = tool_versions()
        self._git_sha = git_sha()
        # `landing_sha` refuses in a shallow checkout rather than returning the
        # graft commit, which is D-0041. A truncated history therefore reaches
        # the caller as one readable refusal naming the checkout, instead of as a
        # `prediction_sha` that is wrong and looks right.
        self._prediction_sha = landing_sha(ABLATION_PREDICTION)
        if self._prediction_sha is None:
            raise BenchmarkError(
                f"the prediction {ABLATION_PREDICTION!r} is not in any commit, "
                f"so no cell can name a prediction_sha for it. Section 17.8: the "
                f"prediction is committed before the measurement, and that "
                f"ordering is the whole mechanism. This checkout has its history, "
                f"so the entry is genuinely uncommitted rather than unfetched."
            )
        #: The worst gap between the two clocks over the whole run, reported at
        #: the end so a run says how well its own cross check held rather than
        #: only that it did.
        self.worst_timing_gap_ms = 0.0
        self.worst_timing_gap_pass = ""
        #: The allocated IR of every cell this runner measured, by cell name.
        #: The P11 groups are filled from it after the run rather than during it,
        #: for the same reason the deltas are: the run order is randomized.
        self.allocated_ir: dict[str, str] = {}
        #: The commit the divergence prediction landed in, resolved once. It
        #: refuses in a shallow checkout rather than returning the graft commit,
        #: which is D-0041.
        self.divergence_sha = landing_sha(DIVERGENCE_PREDICTION)
        if self.divergence_sha is None:
            raise BenchmarkError(
                f"the prediction {DIVERGENCE_PREDICTION!r} is not in any commit, "
                f"so no cell can name a prediction_sha for the SCALE-Sim "
                f"numbers it now carries."
            )

    # ---- the pieces a cell needs ----------------------------------------

    def onnx(self, model: str, batch: int) -> Path:
        key = (model, batch)
        if key not in self._onnx:
            self._onnx[key] = generate_model(model, self._work, batch=batch)
        return self._onnx[key]

    def session(self, model: str, batch: int) -> ort.InferenceSession:
        key = (model, batch)
        if key not in self._sessions:
            self._sessions[key] = ort.InferenceSession(
                str(self.onnx(model, batch)), providers=["CPUExecutionProvider"]
            )
        return self._sessions[key]

    def reference(
        self, model: str, budget: str, budget_bytes: int, batch: int
    ) -> Reference:
        """The unablated `-O0` answer for the same model, budget and batch."""
        key = (model, budget, batch)
        if key not in self._references:
            program = compile_model(
                self.onnx(model, batch),
                level=0,
                emit="nbin",
                budget=budget_bytes,
            )
            assert program.binary is not None
            arrays = make_inputs(
                INPUT_CLASS, program.input_shapes, model=model, batch=batch
            )
            answer = run_program(program.binary, arrays, program.output_shapes)
            self._references[key] = Reference(
                outputs=[array.copy() for array in answer.outputs]
            )
        return self._references[key]

    # ---- one cell --------------------------------------------------------

    def run(self, cell: Cell, position: int, seed: int) -> dict[str, Any]:
        key = cell.key
        onnx = self.onnx(key.model, key.batch)
        expected = expected_passes(key.opt_level, ablated=key.ablated_pass)

        stats_path = self._work / f"{cell.name}.stats.json"
        compile_samples: list[float] = []
        pass_samples: dict[int, list[float]] = {}
        records = None
        program = None

        # Section 16.1: `n_trials >= 10` is ten repetitions of the whole
        # instrumented pipeline, each recompiling the same input through the same
        # single pass manager. Ten compilations, not one compilation and ten
        # readings of it, and not one pass run ten times.
        for trial in range(TRIALS):
            started = time.perf_counter()
            program = compile_model(
                onnx,
                level=key.opt_level,
                emit="nbin",
                budget=cell.budget_bytes,
                ablate=key.ablated_pass,
                pass_stats_json=stats_path,
                mlir_timing=(trial == 0),
            )
            compile_samples.append((time.perf_counter() - started) * 1000.0)

            # The expectation is the level's list minus the ablated pass, so an
            # ablation that did not happen is a raise naming the pass rather than
            # a row of zeros that reads as a pass with no effect.
            records = load_pass_stats(stats_path, expected=expected)
            for record in records:
                pass_samples.setdefault(record.position, []).append(record.wall_ms)

            if trial == 0:
                check = cross_check_against_mlir_timing(
                    records, program.mlir_timing_text
                )
                if check.worst_gap_ms > self.worst_timing_gap_ms:
                    self.worst_timing_gap_ms = check.worst_gap_ms
                    worst = max(check.rows, key=lambda row: row[3])
                    self.worst_timing_gap_pass = f"{worst[0]} in {cell.name}"

        assert program is not None and records is not None
        assert program.binary is not None

        arrays = make_inputs(
            INPUT_CLASS, program.input_shapes, model=key.model, batch=key.batch
        )
        answer = run_program(program.binary, arrays, program.output_shapes)
        statistics = answer.stats

        names = [
            entry.name for entry in self.session(key.model, key.batch).get_inputs()
        ]
        oracle = self.session(key.model, key.batch).run(
            None, dict(zip(names, arrays, strict=True))
        )

        accuracy = self._accuracy(cell, answer.outputs, oracle)

        allocation = self._allocator_attributes(program.stages["npuisa"])
        npuisa_counts = {
            name: count
            for name, count in records[-1].ops_after.items()
            if name.startswith("npuisa.")
        }
        # Kept so the P11 groups can be filled after the run without recompiling.
        # The roofline and the SCALE-Sim export both walk this text, and a second
        # compilation to get it back would be a second program to argue was the
        # same one.
        self.allocated_ir[cell.name] = program.stages["npuisa"]

        return self._assemble(
            cell=cell,
            position=position,
            seed=seed,
            records=records,
            compile_samples=compile_samples,
            pass_samples=pass_samples,
            statistics=statistics,
            npuisa_counts=npuisa_counts,
            allocation=allocation,
            accuracy=accuracy,
        )

    # ---- the parts of a cell --------------------------------------------

    def _accuracy(
        self, cell: Cell, produced: list[np.ndarray], oracle: list[Any]
    ) -> dict[str, Any]:
        key = cell.key
        worst_absolute = 0.0
        worst_relative = 0.0
        absolute_sum = 0.0
        absolute_count = 0
        for got, expected in zip(produced, oracle, strict=True):
            difference = np.abs(
                got.astype(np.float64) - np.asarray(expected, dtype=np.float64)
            )
            worst_absolute = max(worst_absolute, float(difference.max()))
            absolute_sum += float(difference.sum())
            absolute_count += int(difference.size)
            scale = float(np.abs(np.asarray(expected, dtype=np.float64)).max())
            if scale > 0.0:
                worst_relative = max(worst_relative, float(difference.max()) / scale)

        reference = self.reference(
            key.model, key.scratchpad_budget, cell.budget_bytes, key.batch
        )
        movement = 0.0
        noise = 0.0
        signal = 0.0
        for got, base in zip(produced, reference.outputs, strict=True):
            wide = got.astype(np.float64)
            reference_wide = base.astype(np.float64)
            movement = max(movement, float(np.abs(wide - reference_wide).max()))
            noise += float(np.square(wide - reference_wide).sum())
            signal += float(np.square(reference_wide).sum())

        accuracy: dict[str, Any] = {
            "max_abs_error_vs_onnxruntime": worst_absolute,
            "max_rel_error_vs_onnxruntime": worst_relative,
            "mean_abs_error_vs_onnxruntime": absolute_sum / max(absolute_count, 1),
            "max_abs_movement_vs_o0": movement,
            "top1_agreement_vs_onnxruntime": _top1_agreement(produced, oracle),
            "cosine_similarity_vs_onnxruntime": _cosine_similarity(produced, oracle),
            "ranking_metric": RANKING_METRIC,
        }

        is_the_reference = key.opt_level == 0 and key.ablated_pass is None
        if is_the_reference:
            accuracy.update(null("sqnr_db_vs_fp32_simulated"))
        elif noise == 0.0:
            accuracy.update(null("sqnr_db_vs_fp32_simulated", cause="identical"))
        else:
            accuracy["sqnr_db_vs_fp32_simulated"] = 10.0 * float(
                np.log10(signal / noise)
            )

        accuracy.update(null("per_layer_sqnr_db"))
        accuracy.update(null("max_abs_error_vs_fp32_simulated"))
        return accuracy

    @staticmethod
    def _allocator_attributes(npuisa_text: str) -> dict[str, float]:
        found = dict(_ATTRIBUTE.findall(npuisa_text))
        required = {"fragmentation_ratio", "spill_count"}
        if not required.issubset(found):
            raise BenchmarkError(
                "the allocated function is missing an attribute this schema "
                f"records: {sorted(required - set(found))}. Section 16.1 makes a "
                "missing number an error rather than a zero, and these two are "
                "fields this phase can fill rather than fields it defers."
            )
        return {name: float(value) for name, value in found.items()}

    def _assemble(
        self,
        *,
        cell: Cell,
        position: int,
        seed: int,
        records: list[Any],
        compile_samples: list[float],
        pass_samples: dict[int, list[float]],
        statistics: dict[str, Any],
        npuisa_counts: dict[str, int],
        allocation: dict[str, float],
        accuracy: dict[str, Any],
    ) -> dict[str, Any]:
        key = cell.key
        dram_total = int(statistics["dram_bytes_read"]) + int(
            statistics["dram_bytes_written"]
        )

        simulation: dict[str, Any] = {
            "simulated_cycles": float(statistics["cycles"]),
            "dram_bytes_read": int(statistics["dram_bytes_read"]),
            "dram_bytes_written": int(statistics["dram_bytes_written"]),
            "dram_bytes_total": dram_total,
            "macs": int(statistics["macs"]),
            "effective_macs": float(statistics["effective_macs"]),
            "utilization": float(statistics["utilization"]),
            "delta": float(statistics["delta"]),
            # The scratchpad port's traffic, which is what Accelergy's scratchpad
            # action counts are. Recorded from P11 with the energy path.
            "scratchpad_elements_read": int(statistics["scratchpad_elements_read"]),
            "scratchpad_elements_written": int(
                statistics["scratchpad_elements_written"]
            ),
            "int8_macs": int(statistics["int8_macs"]),
            "spill_count": int(allocation["spill_count"]),
            "fragmentation_ratio": float(allocation["fragmentation_ratio"]),
            "overlap_fraction": float(statistics["overlap_fraction"]),
        }
        simulation.update(null("tiling_choices"))
        simulation.update(null("quant_boundary_crossings"))

        roofline: dict[str, Any] = {}
        for field in (
            "roofline_bound_cycles",
            "roofline_bound_cycles_per_layer",
            "operational_intensity",
            "roofline_verdict",
        ):
            roofline.update(null(field))

        normalized: dict[str, Any] = {
            "macs_per_dram_byte": (
                float(statistics["macs"]) / dram_total if dram_total else 0.0
            ),
            "units_convention": UNITS_CONVENTION,
        }
        normalized.update(null("energy_pj_per_inference"))
        normalized.update(null("edp"))

        external: dict[str, Any] = {}
        for field in (
            "scalesim_cycles",
            "scalesim_cycles_per_layer",
            "scalesim_skipped",
            "scalesim_approximations",
            "scalesim_covered_cycle_fraction",
            "scalesim_covered_op_fraction",
            "energy_pj",
            "energy_pj_per_component",
            "area_mm2",
            "area_mm2_per_component",
            "technology_node",
        ):
            external.update(null(field))

        passes = [
            {
                "name": record.name,
                "pass_name": record.pass_name,
                "position": record.position,
                "anchor_op": record.anchor_op,
                "ops_before": record.ops_before,
                "ops_after": record.ops_after,
                "ops_before_total": record.ops_before_total,
                "ops_after_total": record.ops_after_total,
                "timing": timing_object(pass_samples[record.position]),
                "pass_timing_source": record.pass_timing_source,
            }
            for record in records
        ]

        manifest: dict[str, Any] = {
            "git_sha": self._git_sha,
            "llvm_tag": self._tool_versions["llvm"],
            "generator_version": GENERATOR_VERSION,
            "cost_model_constants": dict(cost_model.VALUES),
            "run_order_seed": seed,
            "run_order_position": position,
            "timestamp": datetime.now(UTC).isoformat(),
            "tool_versions": self._tool_versions,
        }
        manifest.update(host_manifest())
        manifest.update(null("calibration_methodology_version"))
        manifest.update(null("technology_node"))
        manifest.update(null("tool_shas"))
        manifest.update(null("registered_estimators"))

        # The cells that are evidence for `p10-ablation-deltas`: every ablation
        # row, and the unablated `-O2` cells its deltas are taken against. Every
        # other cell carries null for both fields, which Section 16.1 calls
        # legitimate and common in those words.
        names_prediction = key.ablated_pass is not None or (
            key.opt_level == 2 and key.batch == declared_batch(key.model)
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "cell": key.as_json(cell.budget_bytes),
            "content_hash": self._hash,
            "instruction_count": int(statistics["instructions"]),
            "npuisa_op_counts": npuisa_counts,
            "timing": {
                "compile_ms": timing_object(compile_samples),
                "passes_total_ms": timing_object(
                    [sum(values) for values in zip(*pass_samples.values(), strict=True)]
                ),
            },
            "passes": passes,
            "simulation": simulation,
            "roofline": roofline,
            "prediction_id": ABLATION_PREDICTION if names_prediction else None,
            "prediction_sha": self._prediction_sha if names_prediction else None,
            "normalized": normalized,
            "external": external,
            "accuracy": accuracy,
            "deltas": None,
            "manifest": manifest,
        }


# ---------------------------------------------------------------------------
# The deltas, which need every cell before any of them can be written.
# ---------------------------------------------------------------------------


def fill_deltas(results: dict[str, dict[str, Any]]) -> None:
    """Section 16.1: deltas stored alongside the absolutes, so nobody has to trust
    the subtraction.

    Filled after the run rather than during it, because the run order is
    randomized and an ablation cell can be reached before its baseline. A delta
    computed against a baseline that had not run yet would be the reason the
    randomization exists undoing the thing it exists for.
    """
    for name, result in results.items():
        baseline_name = result["cell"]["baseline_cell"]
        if baseline_name is None:
            continue
        baseline = results.get(baseline_name)
        if baseline is None:
            raise BenchmarkError(
                f"cell {name} names the baseline {baseline_name!r} and it is not "
                f"in this run. An ablation row whose baseline is missing is a "
                f"subtraction with one operand."
            )
        result["deltas"] = {
            "instruction_count": result["instruction_count"]
            - baseline["instruction_count"],
            "simulated_cycles": result["simulation"]["simulated_cycles"]
            - baseline["simulation"]["simulated_cycles"],
            "dram_bytes_total": result["simulation"]["dram_bytes_total"]
            - baseline["simulation"]["dram_bytes_total"],
            "macs": result["simulation"]["macs"] - baseline["simulation"]["macs"],
        }


# ---------------------------------------------------------------------------
# The P11 groups, which need the program rather than only its statistics.
# ---------------------------------------------------------------------------


#: The prediction the SCALE-Sim fields are evidence for. Every cell carries them,
#: so every cell names it.
DIVERGENCE_PREDICTION = "p11-scalesim-divergence"


def opt_out_external(result: dict[str, Any]) -> None:
    """Record the P11 fields as null with the reason Section 16.4 asks for.

    Not the same thing as the tools being absent. A missing tool fails loudly
    naming the dependency, which is what the exporters do; this is the explicit
    opt out, and it says which one it is in the reason string so nobody reads a
    declined measurement as a failed one.
    """
    for field in (
        "roofline_bound_cycles",
        "roofline_bound_cycles_per_layer",
        "operational_intensity",
        "roofline_verdict",
    ):
        result["roofline"].pop(field, None)
        result["roofline"].pop(f"{field}_null_reason", None)
        result["roofline"].update(null(field, cause="opted_out"))
    for field in (
        "scalesim_cycles",
        "scalesim_cycles_per_layer",
        "scalesim_skipped",
        "scalesim_approximations",
        "scalesim_covered_cycle_fraction",
        "scalesim_covered_op_fraction",
        "energy_pj",
        "energy_pj_per_component",
        "area_mm2",
        "area_mm2_per_component",
        "technology_node",
    ):
        result["external"].pop(field, None)
        result["external"].pop(f"{field}_null_reason", None)
        result["external"].update(null(field, cause="opted_out"))
    for field in ("energy_pj_per_inference", "edp"):
        result["normalized"].pop(field, None)
        result["normalized"].pop(f"{field}_null_reason", None)
        result["normalized"].update(null(field, cause="opted_out"))
    for field in ("technology_node", "tool_shas", "registered_estimators"):
        result["manifest"].pop(field, None)
        result["manifest"].pop(f"{field}_null_reason", None)
        result["manifest"].update(null(field, cause="opted_out"))


def fill_external(
    result: dict[str, Any],
    npuisa_text: str,
    *,
    estimator: accelergy_energy.Estimator,
    directory: Path,
    prediction_sha: str,
) -> list[str]:
    """The roofline, SCALE-Sim and Accelergy groups for one cell.

    Returns the roofline violations, which Section 16.6 makes a failure of the
    run rather than a note.

    **Filling a field means deleting its null reason**, because
    `validate_result` refuses a field carrying both and a half done fill is
    therefore red rather than quietly partial. Each group is replaced whole for
    that reason: updating the value and leaving the sibling behind is the exact
    failure the validator exists to catch.
    """
    answer = roofline.roofline_for(result, npuisa_text)
    result["roofline"] = answer.as_schema()

    divergence = scalesim_export.divergence_for(result, npuisa_text, directory)
    energy = estimator.energy(result)

    external = divergence.as_schema()
    external.update(
        {
            key: value
            for key, value in energy.as_schema(
                float(result["simulation"]["simulated_cycles"])
            ).items()
            if key not in ("energy_pj_per_inference", "edp")
        }
    )
    result["external"] = external

    result["normalized"] = {
        "macs_per_dram_byte": result["normalized"]["macs_per_dram_byte"],
        "units_convention": UNITS_CONVENTION,
        "energy_pj_per_inference": energy.energy_pj,
        "edp": energy.energy_pj
        * float(result["simulation"]["simulated_cycles"])
        * accelergy_energy.GLOBAL_CYCLE_SECONDS,
    }

    manifest = result["manifest"]
    for field in ("technology_node", "tool_shas", "registered_estimators"):
        manifest.pop(field, None)
        manifest.pop(f"{field}_null_reason", None)
    manifest["technology_node"] = accelergy_energy.TECHNOLOGY_NODE
    manifest["tool_shas"] = external_tool_shas()
    manifest["registered_estimators"] = estimator.registered

    # Every cell now carries a SCALE-Sim number, so every cell is evidence for
    # the divergence prediction. A cell that already names the P10 ablation
    # prediction keeps it: `prediction_id` holds one entry, and the ablation
    # deltas are the claim those cells were recorded to answer.
    if result["prediction_id"] is None:
        result["prediction_id"] = DIVERGENCE_PREDICTION
        result["prediction_sha"] = prediction_sha

    return divergence_findings(divergence) + answer.violations


def divergence_findings(divergence: scalesim_export.CellDivergence) -> list[str]:
    """What the SCALE-Sim comparison fails the run for, which is nothing.

    **Section 16.3 pre-registers divergence bands and this does not enforce
    them.** A band being exceeded is a finding to write up in `docs/NUMBERS.md`;
    it is not a reason to fail a measurement run, whose job is to record what
    happened. The prediction of Section 17.8 is answered by reading the recorded
    numbers, not by a run refusing to record them.

    **A coverage threshold was here for one revision and was wrong.** It failed
    the run on `scalesim_covered_cycle_fraction` above 0.9, on the grounds that
    the divergence prediction names that as evidence the exporter is representing
    operations it has no systolic representation for. `lenet` measures 0.9548 and
    the exporter is representing nothing it should not: the fraction is high
    because a covered layer's cycles are the later of its compute and the DMA
    that feeds it, on both sides of the comparison, and `lenet`'s three matmuls
    are dominated by loading a 400 by 120 weight matrix. `scalesim_covered_op_fraction`
    is 0.21 on the same cell and is the figure the prediction's clause was
    reaching for. Both are recorded; `docs/NUMBERS.md` answers the clause with
    both and with the reason. A check that fails a run for a number the exporter
    is right about is worse than no check.

    Kept as a function rather than deleted, because the SCALE-Sim comparison is
    where a future genuinely run failing condition would go, and a caller that
    already collects findings is easier to add one to than one that does not.
    """
    del divergence
    return []


def external_tool_shas() -> dict[str, str]:
    """The tool provenance Section 16.1 asks the manifest to carry.

    Git shas rather than installed metadata, because Accelergy is not on PyPI and
    SCALE-Sim has no tagged releases, so reading installed metadata would record
    nothing useful for two of the three. Read from the clones this project
    installed from, and the installed SCALE-Sim tree's own hash beside its sha,
    because `scripts/patch-scalesim.py` means the sha does not describe the code
    that ran. D-0044.
    """
    shas: dict[str, str] = {}
    root = Path(os.environ.get("NPU_EXTERNAL_DIR", str(Path.home() / "npu-external")))
    for name in (
        "scale-sim-v2",
        "accelergy",
        "accelergy-library-plug-in",
        "accelergy-aladdin-plug-in",
        "accelergy-cacti-plug-in",
        "accelergy-table-based-plug-ins",
    ):
        clone = root / name
        if not (clone / ".git").exists():
            raise BenchmarkError(
                f"the clone of {name} is not at {clone}, so this run cannot "
                f"record which commit of it produced the numbers. Section 16.1 "
                f"records these by git sha, and a manifest that omitted one "
                f"would carry an external figure traceable to nothing. Set "
                f"NPU_EXTERNAL_DIR if the clones live elsewhere."
            )
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(clone),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            raise BenchmarkError(
                f"`git rev-parse HEAD` in {clone} exited "
                f"{completed.returncode}.\n\n{completed.stderr.strip()}"
            )
        shas[name] = completed.stdout.strip()

    shas["scalesim_installed_tree_sha256"] = scalesim_export.installed_tree_sha256()
    return shas


def budget_verdict(elapsed_seconds: float, budget_minutes: float) -> str | None:
    """The Section 2 gate, as a function so a test can reach it without a run.

    Returns the failure message, or `None` when the suite is inside its budget.
    Section 2 and P10's gate state this the same way in both places: the suite
    runtime is measured against the budget and **the run fails if it exceeds
    it**. That is what "enforced as a gate" means, and a budget that only printed
    a warning would be Section 19.0's silence and success looking alike, applied
    to a number instead of to a step.
    """
    if elapsed_seconds <= budget_minutes * 60.0:
        return None
    return (
        f"the suite took {elapsed_seconds / 60.0:.2f} minutes against a budget "
        f"of {budget_minutes:.0f}. Section 2 states the budget as measured and "
        f"enforced as a gate, and this is the enforcement. Do not widen the "
        f"budget to make this pass: the measured cost per cell is printed above "
        f"and the question is what got slower."
    )


def check_ablation_numerics(results: dict[str, dict[str, Any]]) -> list[str]:
    """Section 16.2: an ablation whose numerics move beyond tolerance fails the run.

    It means the removed pass is load bearing for correctness rather than for
    performance, and attributing that to a performance table would be the exact
    misreading the ablation exists to prevent.
    """
    findings: list[str] = []
    for name, result in results.items():
        if result["cell"]["ablated_pass"] is None:
            continue
        distance = result["accuracy"]["max_abs_error_vs_onnxruntime"]
        if distance > ABSOLUTE_TOLERANCE:
            findings.append(
                f"cell {name}: removing {result['cell']['ablated_pass']} moved "
                f"the answer to {distance:.6e} from onnxruntime, outside the "
                f"end to end band of {ABSOLUTE_TOLERANCE:.6e}. Section 16.2 "
                f"makes that a finding and fails the run: the pass is load "
                f"bearing for correctness rather than for performance."
            )
    return findings


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_benchmarks.py",
        description=(
            "Run the benchmark suite of Sections 16.1 and 16.2, one JSON per "
            "cell under experiments/results/."
        ),
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="where the cells are written. Defaults to experiments/results/.",
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=DEFAULT_BUDGET_MINUTES,
        help=(
            "Section 2's suite runtime budget. The run FAILS when the measured "
            "runtime exceeds it, which is what enforced as a gate means."
        ),
    )
    parser.add_argument(
        "--run-order-seed",
        type=int,
        default=DEFAULT_RUN_ORDER_SEED,
        help=(
            "the seed the cell execution order is shuffled under, recorded in "
            "every manifest. Section 16.2 asks for the randomization because "
            "ordering effects, thermal drift and warm cache state bias a "
            "measurement."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo cells that exist and validate, instead of skipping them",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "do not run the roofline, SCALE-Sim or Accelergy, and record their "
            "fields as null with a reason saying this flag was used. Section "
            "16.4: a missing external tool fails loudly naming the dependency, "
            "and an explicit opt out records a null plus a reason. This is the "
            "second of those and never a way to make the first quiet."
        ),
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "reuse a result whose content_hash no longer matches the sources. "
            "For deliberate reuse, and never to make a test pass."
        ),
    )
    arguments = parser.parse_args(argv)

    try:
        return _run(arguments)
    except (BenchmarkError, ResultSchemaError) as failure:
        print(f"run-benchmarks: {failure}", file=sys.stderr)
        return 2


def _run(arguments: argparse.Namespace) -> int:
    import tempfile

    cells = planned_cells(arguments.models)
    current_hash = content_hash(tool_versions=tool_versions())
    results_dir = Path(arguments.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Section 18: the planned cell count and a projected wall clock, printed up
    # front. The projection is the last measured per cell cost when there is one,
    # and Section 2's planning figure when there is not, and it says which.
    per_cell_guess, source = _projection(results_dir)
    ablatable = ablatable_passes(2)
    ablation_cells = sum(1 for cell in cells if cell.key.ablated_pass is not None)
    print(
        f"run-benchmarks: {len(cells)} cells planned, "
        f"{len(cells) - ablation_cells} benchmark and {ablation_cells} ablation "
        f"over {len(ablatable)} ablatable passes read from the driver at run time"
    )
    print(
        f"run-benchmarks: projected {len(cells) * per_cell_guess / 60.0:.1f} "
        f"minutes at {per_cell_guess:.1f} s per cell ({source}), against a "
        f"budget of {arguments.budget_minutes:.0f} minutes"
    )
    print(f"run-benchmarks: run order seed {arguments.run_order_seed}")

    order = list(cells)
    random.Random(arguments.run_order_seed).shuffle(order)

    results: dict[str, dict[str, Any]] = {}
    reused = 0
    measured = 0
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="npu-benchmarks-") as directory:
        runner = Runner(Path(directory), current_hash)
        for position, cell in enumerate(
            tqdm(order, unit="cell", desc="cells", dynamic_ncols=True)
        ):
            existing = results_dir / f"{cell.name}.json"
            if existing.is_file() and not arguments.force:
                recorded = load_result(existing)
                fresh = recorded["content_hash"] == current_hash
                if fresh or arguments.allow_stale:
                    results[cell.name] = recorded
                    reused += 1
                    continue
            results[cell.name] = runner.run(cell, position, arguments.run_order_seed)
            measured += 1

        # Section 16.6, 16.3 and 16.4, filled after the loop rather than inside
        # it, for the same reason the deltas are: the cell that a figure needs
        # may not have run yet. Inside the temporary directory, because the
        # external tools want somewhere to write and their traces are large.
        external_findings: list[str] = []
        if arguments.skip_external:
            for result in results.values():
                opt_out_external(result)
            print(
                "run-benchmarks: --skip-external, so the roofline, SCALE-Sim and "
                "Accelergy fields are null with a reason rather than absent. "
                "Section 16.4 asks for the opt out to be recorded rather than to "
                "look like a tool that could not be found."
            )
        else:
            estimator = accelergy_energy.Estimator(Path(directory) / "accelergy")
            for index, (name, result) in enumerate(
                tqdm(
                    sorted(results.items()),
                    unit="cell",
                    desc="external",
                    dynamic_ncols=True,
                )
            ):
                text = runner.allocated_ir.get(name)
                if text is None:
                    raise BenchmarkError(
                        f"cell {name} was reused from disk, so this run does not "
                        f"have its program and cannot fill the P11 fields from "
                        f"it. Re-record the whole suite with --force: Section "
                        f"16.1's staleness rule and results_to_tex.py both "
                        f"require the committed cells to come from one run."
                    )
                external_findings += fill_external(
                    result,
                    text,
                    estimator=estimator,
                    directory=Path(directory) / "scalesim" / f"cell{index}",
                    prediction_sha=runner.divergence_sha,
                )

    elapsed = time.perf_counter() - started

    fill_deltas(results)
    for result in results.values():
        write_result(result, results_dir)

    findings = check_ablation_numerics(results) + external_findings

    print()
    print(f"run-benchmarks: {len(results)} cells, {measured} measured, {reused} reused")
    print(
        f"run-benchmarks: worst instrumentation against --mlir-timing gap "
        f"{runner.worst_timing_gap_ms:.4f} ms"
        + (
            f", at {runner.worst_timing_gap_pass}"
            if runner.worst_timing_gap_pass
            else ""
        )
    )

    # The number that replaces Section 2's 15 second planning figure. Printed
    # over the cells this run actually measured, because averaging in the reused
    # ones would report a cost nobody paid.
    if measured:
        per_cell = elapsed / measured
        print(
            f"run-benchmarks: measured {elapsed / 60.0:.2f} minutes for "
            f"{measured} cells, {per_cell:.2f} seconds per cell. That per cell "
            f"cost is the measured figure Section 2's 15 second planning number "
            f"is replaced by."
        )
        # **Beside the results directory and not inside it.** Section 16.1 says
        # `experiments/results/` holds one JSON per cell, and this file is not a
        # cell. A non cell JSON in there is a trap for every consumer that globs
        # the directory, which is all of them; the schema validator hit it within
        # a minute of the first run, and `results_to_tex.py` would have been next.
        runtime_path(results_dir).write_text(
            json.dumps(
                {
                    "cells_measured": measured,
                    "cells_total": len(results),
                    "suite_seconds": elapsed,
                    "seconds_per_cell": per_cell,
                    "budget_minutes": arguments.budget_minutes,
                    "run_order_seed": arguments.run_order_seed,
                    "worst_timing_gap_ms": runner.worst_timing_gap_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    if findings:
        print()
        for line in findings:
            print(f"run-benchmarks: {line}", file=sys.stderr)
        print(
            "run-benchmarks: FAIL. An ablation that moves the numerics is a "
            "finding rather than a row.",
            file=sys.stderr,
        )
        return 1

    # Section 2, and P10's gate says it the same way in both places: the suite
    # runtime is measured against the budget and **the run fails if it exceeds
    # it**. Checked last, so that a run over budget still leaves its results and
    # its measured cost behind for the person who has to decide what to do.
    over_budget = budget_verdict(elapsed, arguments.budget_minutes)
    if over_budget is not None:
        print(f"run-benchmarks: FAIL. {over_budget}", file=sys.stderr)
        return 1

    print(
        f"run-benchmarks: inside the budget, {elapsed / 60.0:.2f} minutes "
        f"against {arguments.budget_minutes:.0f}."
    )
    return 0


def runtime_path(results_dir: Path) -> Path:
    """Where a run records what it cost, beside the cells rather than among them.

    `<results>/../<name>-runtime.json`, so that a caller pointing `--results` at
    a temporary directory gets its runtime file beside it rather than in the
    repository.
    """
    return results_dir.parent / f"{results_dir.name}-runtime.json"


def _projection(results_dir: Path) -> tuple[float, str]:
    """The per cell cost to project from, and where the figure came from.

    Section 18 wants a projected wall clock printed up front. Section 2 wants its
    own 15 second planning figure replaced by a measurement. Both are served by
    projecting from the last run's measurement when there is one and saying so,
    rather than by carrying the planning figure forever.
    """
    recorded = runtime_path(results_dir)
    if recorded.is_file():
        try:
            document = json.loads(recorded.read_text(encoding="utf-8"))
            return float(document["seconds_per_cell"]), "measured by the last run"
        except (json.JSONDecodeError, KeyError, ValueError):  # pragma: no cover
            pass
    return 15.0, "Section 2's planning figure, not yet measured here"


def describe() -> dict[str, Any]:
    """What the suite is, for a test that wants the shape without running it."""
    return {
        "cells": [cell.name for cell in planned_cells()],
        "ablatable": ablatable_passes(2),
        "levels": implemented_levels(),
        "pipeline": describe_pipeline(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
