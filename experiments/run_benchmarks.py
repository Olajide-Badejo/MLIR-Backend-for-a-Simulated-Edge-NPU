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


def manifest() -> dict:
    return {
        "git_sha": git_sha(),
        "llvm_tag": LLVM_TAG,
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cost_model": COST_MODEL,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def count_ops(mlir_text: str, dialect: str) -> dict:
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

        ref = ort.InferenceSession(str(onnx)).run(None, {"input": x})[0]
        y_sim = y_sim.reshape(ref.shape)
        max_abs_error = float(np.max(np.abs(y_sim - ref)))

    return {
        "model": model,
        "opt_level": level,
        "scratchpad_budget": budget,
        "npuisa_op_counts": isa_ops,
        "instruction_count": int(sum(isa_ops.values())),
        "simulated_cycles": stats["cycles"],
        "dram_bytes_read": stats["dram_bytes_read"],
        "dram_bytes_written": stats["dram_bytes_written"],
        "dram_bytes_total": stats["dram_bytes_read"] + stats["dram_bytes_written"],
        "max_abs_error_vs_onnxruntime": max_abs_error,
        "compile_ms": compile_ms,
        "note": "simulated estimates, not measurements",
        "manifest": manifest(),
    }


def result_path(model: str, level: int, budget: int) -> Path:
    tag = "default" if budget == DEFAULT_BUDGET else f"{budget}b"
    return RESULTS / f"{model}_O{level}_{tag}.json"


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def valid(path: Path) -> bool:
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark suite.")
    parser.add_argument("--force", action="store_true", help="redo existing results")
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
        if path.exists() and valid(path) and not args.force:
            continue
        write_atomic(path, benchmark(model, level, budget, args.seed))

    print(f"results in {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
