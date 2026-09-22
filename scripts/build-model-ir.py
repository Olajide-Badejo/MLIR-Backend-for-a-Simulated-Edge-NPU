#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Builds every model of Section 15's suite and writes its IR into
# experiments/models/, which is the directory scripts/check-reachability.py
# reads for the model layer of law 2.
#
#   python scripts/build-model-ir.py                the whole suite
#   python scripts/build-model-ir.py --model lenet  one of them
#   python scripts/build-model-ir.py --list         what it would write
#
# WHY THE IR AND NOT THE .onnx
#
# Step 3 of Section 17.5 asks that each operation appear in at least one
# generated benchmark model's **IR**. An `.onnx` file holds ONNX operator names,
# so searching one for `npu.batch_norm` would find nothing whatever the truth
# was. The `npu` level IR is where the dialect's own operations appear, and the
# `npuisa` level IR is written beside it because it is what the instruction
# stream is encoded from and because a reader diagnosing a missing layer wants
# both.
#
# NOTHING HERE IS COMMITTED
#
# experiments/models/ is a build artifact regenerated from a seed, like every
# `.onnx` in this project. The CI step that runs the full reachability check
# runs this first, in the same step, so that the check has something to look in
# and so that switching the check on is one activation rather than two.

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# This script and check-reachability.py both live in scripts/, which is the
# directory Python puts on sys.path when either is run, so the shared module
# needs no path handling. Its docstring holds why the freshness of this
# artifact is a content hash written here rather than an ordering convention in
# the CI workflow.
import model_ir_provenance

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "experiments" / "models"
PROFILE_DIR = REPO_ROOT / "experiments" / "calibration"

#: Both batch sizes of Section 17.4's matrix. The model layer would be
#: satisfied by one, and both are written because the batched path reaches the
#: same operations through different shapes and a reader looking at this
#: directory should see what the matrix actually compiles.
BATCHES = (1, 4)

#: Section 17.5's step 3 asks that each operation appear in at least one
#: generated benchmark model's IR, and since P9 that means every level rather
#: than one: `npu.fused_op` and `npu.yield` exist only in a `-O2` pipeline's
#: output, so a sweep at `-O0` alone would leave two operations of the dialect
#: with no model to appear in and the reachability check red. The levels come
#: from the compiler's own description; see `main` below.


def _frontend() -> object:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import npu_frontend
    except ImportError as failure:  # pragma: no cover
        print(
            "build-model-ir: the frontend could not be imported. It needs the "
            "MLIR Python bindings on PYTHONPATH; see docs/BUILD.md.\n\n"
            f"{failure}",
            file=sys.stderr,
        )
        raise SystemExit(2) from failure
    return npu_frontend


def stem(name: str, batch: int, registry_batch: int, level: int) -> str:
    base = name if batch == registry_batch else f"{name}-n{batch}"
    return f"{base}-O{level}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-model-ir.py",
        description=(
            "Generate every model of the suite and write its npu and npuisa "
            "IR into experiments/models/, for the model layer of law 2."
        ),
    )
    parser.add_argument("--model", help="build one model rather than the suite")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what would be written and exit",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help=(
            "do not clear experiments/models/ first. The default is to clear "
            "it, because a stale file left behind from a model that no longer "
            "exists would satisfy the reachability check for an operation "
            "nothing reaches any more."
        ),
    )
    arguments = parser.parse_args(argv)

    frontend = _frontend()
    models = frontend.MODELS  # type: ignore[attr-defined]
    names = [arguments.model] if arguments.model else list(models)
    for name in names:
        if name not in models:
            print(
                f"build-model-ir: {name!r} is not in the suite: "
                + ", ".join(sorted(models)),
                file=sys.stderr,
            )
            return 2

    # Every level the compiler can build, read from the compiler rather than
    # written down here. *Changed at P9.* Until `-O1` and `-O2` existed the
    # sweep was `-O0` only and the filenames carried no level; now an operation
    # that only a `-O2` pipeline can create, which is `npu.fused_op` and its
    # terminator, has a model IR file to appear in. That is what closed the two
    # exemptions `docs/EXEMPTIONS.md` carried from P8.
    levels = frontend.implemented_levels()  # type: ignore[attr-defined]

    if arguments.list:
        for name in names:
            registry = models[name].input_shape[0]
            for batch in BATCHES:
                for level in levels:
                    base = stem(name, batch, registry, level)
                    print(f"experiments/models/{base}.npu.mlir")
                    print(f"experiments/models/{base}.npuisa.mlir")
            print(f"experiments/models/{name}-quantized.npu.mlir")
            print(f"experiments/models/{name}-quantized.npuisa.mlir")
        if arguments.model:
            print(
                f"experiments/models/{model_ir_provenance.STAMP_NAME} would be "
                "removed, because one model is not the sweep the stamp claims"
            )
        else:
            print(f"experiments/models/{model_ir_provenance.STAMP_NAME}")
        return 0

    if MODELS_DIR.exists() and not arguments.keep and not arguments.model:
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # **One quantized compilation per model, which is what the model layer of
    # law 2 is missing without it.** Section 17.5's step 3 asks that every
    # operation appear in a generated benchmark model's IR, "including one
    # quantized compilation for the quantization operations", and `npu.quantize`
    # and `npu.dequantize` appear in no fp32 model because the suite is fp32 by
    # construction. So the sweep runs `-npu-calibrate` once per model against
    # the committed profile, at the registry batch, and writes the result
    # beside the others. It is one compilation rather than the whole cross
    # product because the requirement is one.
    written = 0
    for name in names:
        profile = PROFILE_DIR / f"{name}.json"
        if not profile.exists():
            print(
                f"build-model-ir: {profile} is missing, so no quantized "
                f"compilation can be written for {name}. Run "
                f"scripts/build-calibration-profiles.py first.",
                file=sys.stderr,
            )
            return 2

    for name in names:
        registry = models[name].input_shape[0]
        for batch in BATCHES:
            onnx_path = frontend.generate_model(  # type: ignore[attr-defined]
                name, MODELS_DIR, batch=batch
            )
            for level in levels:
                base = stem(name, batch, registry, level)
                for emit, suffix in (("npu", "npu.mlir"), ("npuisa", "npuisa.mlir")):
                    result = frontend.compile_model(  # type: ignore[attr-defined]
                        onnx_path, level=level, emit=emit
                    )
                    (MODELS_DIR / f"{base}.{suffix}").write_text(
                        result.text, encoding="utf-8"
                    )
                    written += 1

    for name in names:
        registry = int(models[name].input_shape[0])
        onnx_path = frontend.generate_model(  # type: ignore[attr-defined]
            name, MODELS_DIR, batch=registry
        )
        for emit, suffix in (("npu", "npu.mlir"), ("npuisa", "npuisa.mlir")):
            result = frontend.compile_model(  # type: ignore[attr-defined]
                onnx_path,
                level=0,
                emit=emit,
                calibrate=str(PROFILE_DIR / f"{name}.json"),
            )
            (MODELS_DIR / f"{name}-quantized.{suffix}").write_text(
                result.text, encoding="utf-8"
            )
            written += 1

    print(f"build-model-ir: wrote {written} IR files to {MODELS_DIR}")

    # **The freshness stamp, and why one model removes it rather than writing
    # one.** The stamp says that everything in this directory was built from
    # the sources hashed into it, and after `--model lenet` that is false of
    # every other model there. Removing it makes the next reachability check
    # refuse, which is the honest answer: the directory is part current and
    # part whatever it held before, and no fingerprint describes that.
    if arguments.model:
        model_ir_provenance.clear_stamp(MODELS_DIR)
        print(
            "build-model-ir: this was one model rather than the suite, so "
            f"{model_ir_provenance.STAMP_NAME} was removed and "
            "check-reachability will refuse until the whole sweep runs."
        )
    else:
        model_ir_provenance.write_stamp(MODELS_DIR, REPO_ROOT)
        stamp = model_ir_provenance.read_stamp(MODELS_DIR) or {}
        print(
            f"build-model-ir: wrote {model_ir_provenance.STAMP_NAME} at "
            f"fingerprint {str(stamp.get('fingerprint'))[:12]} over "
            f"{stamp.get('files_covered')} source files."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
