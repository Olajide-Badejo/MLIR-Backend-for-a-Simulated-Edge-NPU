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


def bin_dir() -> Path:
    return Path(os.environ.get("NPU_BIN", str(REPO / "build" / "bin")))


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
        "note": "simulated estimates, not measurements",
        "manifest": manifest(seed),
    }


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

    print(f"results in {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
