#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Builds a calibration profile for every model of Section 15's suite and writes
# it into experiments/calibration/, which -npu-calibrate reads.
#
#   python scripts/build-calibration-profiles.py                the whole suite
#   python scripts/build-calibration-profiles.py --model lenet  one of them
#   python scripts/build-calibration-profiles.py --count 128    a wider sweep
#   python scripts/build-calibration-profiles.py --list         what it writes
#
# THESE ARE COMMITTED, UNLIKE THE MODELS THEY COME FROM
#
# `experiments/models/` is a build artifact regenerated from a seed and is in
# `.gitignore`. A profile is the opposite and Section 14 says so: it is
# committed per model with its seed and its input count, because that is what
# makes an accuracy number reproducible rather than anecdotal. A reader who
# wants to know what a quantized number was measured against opens the profile
# rather than rerunning an observer and hoping.
#
# The `.onnx` the observer runs is still a build artifact. It is regenerated
# here into a temporary directory from the same seed the rest of the project
# uses, so this script writes exactly one thing: the profile.

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "experiments" / "calibration"


def _frontend() -> object:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    try:
        import npu_frontend
    except ImportError as failure:  # pragma: no cover
        print(
            f"build-calibration-profiles: cannot import npu_frontend: {failure}",
            file=sys.stderr,
        )
        raise SystemExit(2) from failure
    return npu_frontend


def main(argv: Sequence[str] | None = None) -> int:
    from npu_frontend.calibration import (  # noqa: PLC0415
        DEFAULT_CALIBRATION_COUNT,
    )

    parser = argparse.ArgumentParser(
        description="Write a calibration profile for each model of the suite."
    )
    parser.add_argument("--model", help="one model rather than the whole suite")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_CALIBRATION_COUNT,
        help="how many seeded inputs the observer runs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print what would be written and write nothing",
    )
    arguments = parser.parse_args(argv)

    frontend = _frontend()
    models = frontend.MODELS  # type: ignore[attr-defined]
    if arguments.model and arguments.model not in models:
        print(
            f"build-calibration-profiles: {arguments.model!r} is not a model. "
            f"The suite is {', '.join(sorted(models))}.",
            file=sys.stderr,
        )
        return 2

    names = [arguments.model] if arguments.model else sorted(models)
    if arguments.list:
        for name in names:
            print(PROFILE_DIR / f"{name}.json")
        return 0

    from npu_frontend.calibration import (  # noqa: PLC0415
        build_profile,
        observe,
        observe_nodes,
        observe_weights,
        write_profile,
    )
    from npu_frontend.model_generator import generate_model  # noqa: PLC0415

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        batch = int(models[name].input_shape[0])
        with tempfile.TemporaryDirectory() as directory:
            model_path = generate_model(name, directory, batch=batch)
            profile = build_profile(
                model_name=name,
                batch=batch,
                count=arguments.count,
                observations=observe(
                    model_path, model_name=name, count=arguments.count, batch=batch
                ),
                weights=observe_weights(model_path),
                nodes=observe_nodes(model_path),
            )
        written = write_profile(profile, PROFILE_DIR / f"{name}.json")
        print(
            f"build-calibration-profiles: {written.relative_to(REPO_ROOT)} "
            f"at {arguments.count} inputs, seed {profile['seed']}, "
            f"{len(profile['nodes'])} quantizable operations"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
