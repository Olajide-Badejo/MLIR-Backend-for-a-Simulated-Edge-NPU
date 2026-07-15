"""Generate the seeded ONNX test models into this directory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from npu_frontend import model_generator  # noqa: E402


def main() -> None:
    out = Path(__file__).resolve().parent
    for name in model_generator.MODELS:
        path = model_generator.export(name, out / f"{name}.onnx", seed=0)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
