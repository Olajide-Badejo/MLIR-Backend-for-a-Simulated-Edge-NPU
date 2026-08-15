"""Benchmark the compiler across models, optimization levels, and scratchpad
budgets, writing one JSON result per cell.

Each result carries a manifest (git sha, LLVM tag, tool versions, cost model
constants, timestamp) so every number in the report traces back to a recorded
experiment. Results that already exist and validate are skipped; pass --force to
redo them. Writes are atomic (temp file then rename).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from npu_frontend import compile as npu_compile  # noqa: E402
from npu_frontend import model_generator  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
LLVM_TAG = "llvmorg-22.1.8"
# Documented cost model constants, mirroring the C++ CostModel defaults.
COST_MODEL = {
    "macs_per_cycle": 256,
    "dram_bytes_per_cycle": 16,
    "lanes": 16,
    "issue_overhead": 1,
}
DEFAULT_BUDGET = 1048576
TIGHT_BUDGET = 143360  # 140 KB: below the LeNet peak so weights spill

INPUT_SHAPES = {"lenet": (1, 1, 28, 28)}


# npu-opt's timing output names each pass by its C++ class display name, not by
# the command line flag that selected it, so the two have to be related
# explicitly. A pass in the pipeline with no entry here is an error rather than
# an unnamed row: the point of the mapping is to catch a pipeline change that
# nobody taught this file about.
PASS_DISPLAY_NAMES = {
    "canonicalize": "Canonicalizer",
    "npu-fuse-ops": "NPUFuseOps",
    "symbol-dce": "SymbolDCE",
    "npu-lower-to-npuisa": "NPULowerToNPUISA",
    "npu-allocate-scratchpad": "NPUAllocateScratchpad",
}

# Entries --mlir-timing emits that are not passes.
TIMING_INFRASTRUCTURE = {"Parser", "Output", "Rest", "Total"}

PASS_TIMING_SOURCE = "--mlir-timing"

# An ablated pipeline must still match onnxruntime as closely as the full one.
# This is the end to end tolerance phase U1 set, just above the 2.98e-8 the
# pipeline actually achieves.
ABLATION_ERROR_TOLERANCE = 1e-5


def bin_dir() -> Path:
    return Path(os.environ.get("NPU_BIN", str(REPO / "build" / "bin")))


def cpu_model() -> str:
    """The CPU the wall clock numbers were measured on.

    Per pass wall clock is a real measurement rather than a simulated estimate,
    so it means nothing without the machine it was measured on.
    """
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def pass_flag_name(flag: str) -> str:
    """'-npu-allocate-scratchpad=budget=1048576' -> 'npu-allocate-scratchpad'."""
    return flag.lstrip("-").split("=", 1)[0]


def pipeline_for(level: int, budget: int) -> list[str]:
    """Every npu-opt pass a cell runs, optimization then lowering, in order.

    Taken from _passes_for_level at run time rather than hardcoded, so a pass
    added to a level is instrumented without editing this file. The ONNX import
    stage is deliberately absent: it is not an MLIR pass.
    """
    return [
        *npu_compile._passes_for_level(level),
        "-npu-lower-to-npuisa",
        f"-npu-allocate-scratchpad=budget={budget}",
    ]


def parse_op_stats(text: str) -> dict[str, int]:
    """Parse the JSON histogram npu-opt's print-op-stats pass emits.

    This is the compiler's own count of the ops in the module, which is what
    spec 12.1 item 1 asks for. It is not count_ops(), a regex over the printed
    IR whose docstring explains why that is unfit.

    A format shift raises. Returning an empty histogram instead would record
    every pass as having changed nothing, which is a plausible looking lie.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(
            f"no JSON object in print-op-stats output; got {text[:200]!r}"
        )
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"print-op-stats output is not JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"print-op-stats returned no ops; got {data!r}")
    return {str(k): int(v) for k, v in data.items()}


def op_stats(bd: Path, mlir_path: Path) -> dict[str, int]:
    """Ask npu-opt what ops a module holds."""
    proc = subprocess.run(
        [
            str(bd / "npu-opt"),
            str(mlir_path),
            "--pass-pipeline=builtin.module(print-op-stats{json=true})",
            "-o",
            os.devnull,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_op_stats(proc.stderr or proc.stdout)


def parse_pass_timings(text: str, pipeline: list[str]) -> list[float]:
    """Milliseconds per pass, from npu-opt's --mlir-timing JSON.

    The whole pipeline is run once and timed, so these are what the passes cost
    in sequence rather than in isolation. The JSON is a tree: pipelines nest
    their passes, so the leaves in document order are the passes in pipeline
    order, once the Parser, Output, Rest, and Total entries are dropped.

    A pass that appears in the pipeline but not in the timing output raises. It
    must never be recorded as zero, which would read as a free pass.
    """
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise RuntimeError(f"no JSON array in --mlir-timing output; got {text[:200]!r}")
    try:
        tree = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"--mlir-timing output is not JSON: {exc}") from exc

    leaves: list[tuple[str, float]] = []

    def walk(nodes: list) -> None:
        for node in nodes:
            if not node or "name" not in node:
                continue
            children = [c for c in node.get("passes", []) if c and "name" in c]
            if children:
                walk(children)
            else:
                leaves.append((node["name"], float(node["wall"]["duration"])))

    walk(tree)
    measured = [(n, d) for n, d in leaves if n not in TIMING_INFRASTRUCTURE]

    if len(measured) != len(pipeline):
        raise RuntimeError(
            f"--mlir-timing reported {len(measured)} passes "
            f"({[n for n, _ in measured]}) but the pipeline has "
            f"{len(pipeline)} ({pipeline})"
        )

    out = []
    for (display, seconds), flag in zip(measured, pipeline):
        name = pass_flag_name(flag)
        expected = PASS_DISPLAY_NAMES.get(name)
        if expected is None:
            raise RuntimeError(
                f"no timing display name known for pass {name!r}; add it to "
                f"PASS_DISPLAY_NAMES so the pipeline stays instrumented"
            )
        if display != expected:
            raise RuntimeError(
                f"--mlir-timing named {display!r} where {expected!r} was expected "
                f"for {flag!r}; the pipeline and the timing output disagree"
            )
        out.append(seconds * 1000.0)
    return out


def pass_timings(bd: Path, mlir_path: Path, pipeline: list[str]) -> list[float]:
    proc = subprocess.run(
        [
            str(bd / "npu-opt"),
            str(mlir_path),
            *pipeline,
            "--mlir-timing",
            "--mlir-output-format=json",
            "-o",
            os.devnull,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_pass_timings(proc.stderr or proc.stdout, pipeline)


def per_pass_records(
    bd: Path, import_ir: str, level: int, budget: int, tmp: Path
) -> list[dict]:
    """Op counts before and after each pass, plus its wall clock.

    Each pass is run on its own, fed the previous pass's output, so the before
    and after histograms bracket exactly one pass. The timings come from a
    single run of the whole pipeline, because timing a pass in isolation would
    measure a different thing from what it costs in sequence.
    """
    pipeline = pipeline_for(level, budget)

    source = tmp / "pipeline_input.mlir"
    source.write_text(import_ir)
    timings = pass_timings(bd, source, pipeline)

    records = []
    current = source
    for position, flag in enumerate(pipeline):
        before = op_stats(bd, current)
        nxt = tmp / f"after_{position}.mlir"
        subprocess.run(
            [str(bd / "npu-opt"), str(current), flag, "-o", str(nxt)],
            capture_output=True,
            text=True,
            check=True,
        )
        after = op_stats(bd, nxt)
        records.append(
            {
                "name": pass_flag_name(flag),
                "position": position,
                "ops_before": before,
                "ops_after": after,
                "ops_before_total": sum(before.values()),
                "ops_after_total": sum(after.values()),
                "wall_ms": timings[position],
            }
        )
        current = nxt
    return records


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def manifest(seed: int) -> dict:
    return {
        "git_sha": git_sha(),
        "llvm_tag": LLVM_TAG,
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cost_model": COST_MODEL,
        # The .onnx files are not committed because they are regenerated
        # deterministically. That only holds if the seed and the generator that
        # consumed it travel with the result, so both are recorded here.
        "seed": seed,
        "model_generator_version": model_generator.GENERATOR_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        # Per pass wall clock is a measurement, not a simulated estimate, so it
        # is only meaningful with the machine attached. It does not transfer to
        # another machine and must not be compared across them.
        "cpu_model": cpu_model(),
        "wall_clock_note": (
            "compile_ms and passes[].wall_ms are wall clock measured on the "
            "machine named in cpu_model; they are not portable and not "
            "simulated estimates. Every other performance figure here is a "
            "simulated estimate from the analytical cost model."
        ),
    }


def count_ops(mlir_text: str, dialect: str) -> dict:
    """Histogram of textual `dialect.name` occurrences in an IR dump.

    This is a regex over text, not a semantic count, and it is deliberately kept
    as one because the per op histogram is useful as a histogram. What it is not
    is an instruction count, for two reasons that both inflate it:

    - it matches inside type strings, so `!npuisa.buffer<...>` is counted as an
      occurrence of an op named `npuisa.buffer`;
    - it counts `npuisa.const`, which the encoder emits as DRAM data rather than
      as an instruction.

    For LeNet the difference is 91 / 82 / 70 from this function against the
    simulator's true 28 / 25 / 21. Take `instruction_count` from the simulator.
    """
    return dict(Counter(re.findall(rf"{dialect}\.[a-z_0-9]+", mlir_text)))


def simulate(nbin: Path, x: np.ndarray) -> tuple[np.ndarray, dict]:
    npu_sim = bin_dir() / "npu-sim"
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.bin"
        inp.write_bytes(np.ascontiguousarray(x, dtype=np.float32).tobytes())
        out = Path(tmp) / "out.bin"
        stats = Path(tmp) / "stats.json"
        subprocess.run(
            [
                str(npu_sim),
                str(nbin),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--stats",
                str(stats),
            ],
            check=True,
            capture_output=True,
        )
        y = np.frombuffer(out.read_bytes(), dtype=np.float32)
        return y, json.loads(stats.read_text())


def benchmark(model: str, level: int, budget: int, seed: int) -> dict:
    bd = bin_dir()
    with tempfile.TemporaryDirectory() as tmp:
        onnx = model_generator.export(model, Path(tmp) / f"{model}.onnx", seed=seed)

        t0 = time.perf_counter()
        isa = npu_compile.compile_model(
            onnx, opt_level=level, emit="npuisa", budget=budget, bin_dir=bd
        )
        compile_ms = (time.perf_counter() - t0) * 1000
        isa_ops = count_ops(isa, "npuisa")

        # Per pass measurement, on the same model. The import stage output is
        # the pipeline's input: it is the IR before any pass has run.
        import_ir = npu_compile.compile_model(
            onnx, opt_level=level, emit="import", bin_dir=bd
        )
        passes = per_pass_records(bd, import_ir, level, budget, Path(tmp))

        nbin = Path(tmp) / f"{model}.nbin"
        npu_compile.compile_model(
            onnx, opt_level=level, emit="nbin", output=nbin, budget=budget, bin_dir=bd
        )

        rng = np.random.default_rng(seed)
        x = rng.standard_normal(INPUT_SHAPES[model]).astype(np.float32)
        y_sim, stats = simulate(nbin, x)

        # The instruction count is the simulator's own, not a regex over the IR.
        # A missing field is an error rather than a fall back to the regex: the
        # regex answer is wrong by a factor of three and silently reinstating it
        # is how the wrong number got published in the first place.
        if "instructions" not in stats:
            raise RuntimeError(
                f"npu-sim stats for {model} -O{level} at budget {budget} carry no "
                f"'instructions' field; got keys {sorted(stats)}. Refusing to "
                f"fall back to the IR regex, which counts type strings and "
                f"npuisa.const and is not an instruction count."
            )

        ref = ort.InferenceSession(str(onnx)).run(None, {"input": x})[0]
        y_sim = y_sim.reshape(ref.shape)
        max_abs_error = float(np.max(np.abs(y_sim - ref)))

    return {
        "model": model,
        "opt_level": level,
        "scratchpad_budget": budget,
        "npuisa_op_counts": isa_ops,
        "instruction_count": int(stats["instructions"]),
        "simulated_cycles": stats["cycles"],
        "dram_bytes_read": stats["dram_bytes_read"],
        "dram_bytes_written": stats["dram_bytes_written"],
        "dram_bytes_total": stats["dram_bytes_read"] + stats["dram_bytes_written"],
        "max_abs_error_vs_onnxruntime": max_abs_error,
        "compile_ms": compile_ms,
        "passes": passes,
        "pass_timing_source": PASS_TIMING_SOURCE,
        "note": "simulated estimates, not measurements",
        "manifest": manifest(seed),
    }


def ablatable_passes() -> list[str]:
    """The distinct passes an -O2 ablation removes, in first appearance order.

    Read from _passes_for_level(2) at run time, never hardcoded. Hardcoding the
    four strings would mean the ablation table silently stops covering a pass
    the day one is added, which is the failure this whole phase exists to stop.

    Distinct, because -canonicalize appears twice at -O2 and the question an
    ablation asks is "is this pass worth having", not "is this position worth
    having". Removing every occurrence is what answers that. See
    docs/DESIGN_DECISIONS.md.
    """
    seen: list[str] = []
    for flag in npu_compile._passes_for_level(2):
        name = pass_flag_name(flag)
        if name not in seen:
            seen.append(name)
    return seen


def ablation(model: str, pass_name: str, budget: int, seed: int) -> dict:
    """Compile with the -O2 pipeline minus every occurrence of one pass.

    Encoded, simulated, and checked against onnxruntime exactly as a normal cell
    is. The numerics check is the point as much as the performance delta: if
    removing a pass changes the answer, the pass is load bearing for correctness
    rather than for performance, and that is a finding rather than a row.
    """
    bd = bin_dir()
    full = npu_compile._passes_for_level(2)
    kept = [f for f in full if pass_flag_name(f) != pass_name]
    if len(kept) == len(full):
        raise RuntimeError(
            f"{pass_name!r} is not in the -O2 pipeline {full}; nothing was ablated"
        )

    lowering = [
        "-npu-lower-to-npuisa",
        f"-npu-allocate-scratchpad=budget={budget}",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        onnx = model_generator.export(model, Path(tmp) / f"{model}.onnx", seed=seed)

        t0 = time.perf_counter()
        text = npu_compile.onnx_importer.import_model(onnx)
        text = npu_compile._run_opt(bd / "npu-opt", text, kept)
        isa = npu_compile._run_opt(bd / "npu-opt", text, lowering)
        compile_ms = (time.perf_counter() - t0) * 1000

        isa_path = Path(tmp) / "program.isa.mlir"
        isa_path.write_text(isa)
        nbin = Path(tmp) / f"{model}.nbin"
        proc = subprocess.run(
            [str(bd / "npu-translate"), str(isa_path), "-o", str(nbin)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ablating {pass_name} at budget {budget} produced a program "
                f"npu-translate refused:\n{proc.stderr}"
            )

        rng = np.random.default_rng(seed)
        x = rng.standard_normal(INPUT_SHAPES[model]).astype(np.float32)
        y_sim, stats = simulate(nbin, x)

        if "instructions" not in stats:
            raise RuntimeError(
                f"npu-sim stats for the {pass_name} ablation carry no "
                f"'instructions' field; got keys {sorted(stats)}"
            )

        ref = ort.InferenceSession(str(onnx)).run(None, {"input": x})[0]
        y_sim = y_sim.reshape(ref.shape)
        max_abs_error = float(np.max(np.abs(y_sim - ref)))

    # A removed optimization must not move the answer. If it does, the pass is
    # doing something other than optimizing and the table would be reporting a
    # performance delta for a correctness change.
    if max_abs_error > ABLATION_ERROR_TOLERANCE:
        raise RuntimeError(
            f"ablating {pass_name} at budget {budget} changed the numerics: "
            f"max absolute error {max_abs_error:.3e} exceeds "
            f"{ABLATION_ERROR_TOLERANCE:.3e}. That pass is load bearing for "
            f"correctness, not for performance, which is a finding rather than "
            f"an ablation row."
        )

    baseline_name = result_path(model, 2, budget).name
    baseline = json.loads((RESULTS / baseline_name).read_text())

    absolute = {
        "instruction_count": int(stats["instructions"]),
        "simulated_cycles": stats["cycles"],
        "dram_bytes_read": stats["dram_bytes_read"],
        "dram_bytes_written": stats["dram_bytes_written"],
        "dram_bytes_total": stats["dram_bytes_read"] + stats["dram_bytes_written"],
        "compile_ms": compile_ms,
    }

    return {
        "model": model,
        "opt_level": 2,
        "ablated_pass": pass_name,
        "pipeline_without": [pass_flag_name(f) for f in kept],
        "scratchpad_budget": budget,
        # Named so a reader can recompute every delta below from two committed
        # files rather than trusting the subtraction.
        "baseline_cell": baseline_name,
        **absolute,
        "delta_instruction_count": absolute["instruction_count"]
        - baseline["instruction_count"],
        "delta_simulated_cycles": absolute["simulated_cycles"]
        - baseline["simulated_cycles"],
        "delta_dram_bytes_total": absolute["dram_bytes_total"]
        - baseline["dram_bytes_total"],
        "delta_compile_ms": absolute["compile_ms"] - baseline["compile_ms"],
        "max_abs_error_vs_onnxruntime": max_abs_error,
        "note": "simulated estimates, not measurements; deltas are ablated minus full -O2",
        "manifest": manifest(seed),
    }


def ablation_path(model: str, pass_name: str, budget: int) -> Path:
    """Same convention as result_path, so staleness() covers these too."""
    tag = "default" if budget == DEFAULT_BUDGET else f"{budget}b"
    return RESULTS / f"{model}_O2_ablate_{pass_name}_{tag}.json"


def result_path(model: str, level: int, budget: int) -> Path:
    tag = "default" if budget == DEFAULT_BUDGET else f"{budget}b"
    return RESULTS / f"{model}_O{level}_{tag}.json"


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# Paths whose contents can change a benchmark number. A commit that only edits
# the README or a report does not invalidate a recorded result, and treating it
# as though it did would mean no result was ever reusable.
RESULT_INPUTS = [
    "CMakeLists.txt",
    "include",
    "lib",
    "tools",
    "python/npu_frontend",
    "experiments/run_benchmarks.py",
]


def _changed_since(sha: str) -> list[str] | None:
    """Files under RESULT_INPUTS that differ between ``sha`` and the worktree.

    Returns None if ``sha`` is not a commit this repository knows about.
    Uncommitted edits count, because a result produced from a dirty tree is not
    reproducible from any commit.
    """
    try:
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except Exception:
        return None

    changed: set[str] = set()
    for args in (
        ["diff", "--name-only", sha, "HEAD", "--"],
        ["status", "--porcelain", "--"],
    ):
        out = subprocess.check_output(
            ["git", "-C", str(REPO), *args, *RESULT_INPUTS], text=True
        )
        for line in out.splitlines():
            name = line[3:] if args[0] == "status" else line
            if name.strip():
                changed.add(name.strip())
    return sorted(changed)


def staleness(path: Path) -> str | None:
    """Return why a recorded result cannot be reused, or None if it can be.

    The old version returned True for anything that parsed as JSON, which made
    staleness permanent: a result generated three commits earlier was reused
    forever unless someone remembered --force. Since every number in the README
    and both reports traces back to these files, that quietly decoupled the
    published numbers from the compiler that produced them.

    A note on the rule. UPGRADE_SPEC_V3.md section 8.1 item 4 asks for "stale if
    its manifest.git_sha differs from the current sha". Taken literally that is
    unusable, because a result is committed by the very commit that follows its
    generation, so it would be born stale and every run would regenerate
    everything. What the rule is actually protecting is that a published number
    came from the code currently in the tree. So the check is: same sha, or no
    change since that sha to any path that can move a number. Recorded in
    docs/DESIGN_DECISIONS.md.
    """
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return f"unreadable ({exc.__class__.__name__})"

    recorded = data.get("manifest")
    if not isinstance(recorded, dict):
        return "no manifest block"

    if recorded.get("cost_model") != COST_MODEL:
        return f"cost model changed: {recorded.get('cost_model')} to {COST_MODEL}"

    current = git_sha()
    was = str(recorded.get("git_sha", ""))
    if current == "unknown":
        return None
    if was == current:
        return None

    changed = _changed_since(was)
    if changed is None:
        return f"generated at unknown commit {was[:8]}, HEAD is {current[:8]}"
    if changed:
        shown = ", ".join(changed[:4]) + (" ..." if len(changed) > 4 else "")
        return (
            f"generated at {was[:8]}, HEAD is {current[:8]}, and "
            f"{len(changed)} compiler input(s) changed since: {shown}"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark suite.")
    parser.add_argument("--force", action="store_true", help="redo existing results")
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="reuse results whose manifest does not match HEAD or the current "
        "cost model, instead of regenerating them",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    models = list(model_generator.MODELS)
    levels = [0, 1, 2]
    budgets = [DEFAULT_BUDGET, TIGHT_BUDGET]
    cells = list(product(models, levels, budgets))
    print(f"planned {len(cells)} cells: {models} x O{levels} x budgets {budgets}")

    try:
        from tqdm import tqdm

        iterator = tqdm(cells, desc="benchmarks")
    except ImportError:
        iterator = cells

    for model, level, budget in iterator:
        path = result_path(model, level, budget)
        if path.exists() and not args.force:
            why = staleness(path)
            if why is None:
                continue
            if args.allow_stale:
                print(f"reusing stale {path.name}: {why}")
                continue
            print(f"regenerating {path.name}: {why}")
        write_atomic(path, benchmark(model, level, budget, args.seed))

    # Ablations run after the full cells, because each one records deltas
    # against the -O2 result for its own budget and needs that file present and
    # current. Both budgets, because the passes behave oppositely at the tight
    # one and a table covering only the generous budget would hide it.
    ablations = list(product(models, ablatable_passes(), budgets))
    print(
        f"planned {len(ablations)} ablation cells: "
        f"{models} x {ablatable_passes()} x budgets {budgets}"
    )
    try:
        from tqdm import tqdm as _tqdm

        ablation_iterator = _tqdm(ablations, desc="ablations")
    except ImportError:
        ablation_iterator = ablations

    for model, pass_name, budget in ablation_iterator:
        path = ablation_path(model, pass_name, budget)
        if path.exists() and not args.force:
            why = staleness(path)
            if why is None:
                continue
            if args.allow_stale:
                print(f"reusing stale {path.name}: {why}")
                continue
            print(f"regenerating {path.name}: {why}")
        write_atomic(path, ablation(model, pass_name, budget, args.seed))

    print(f"results in {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
