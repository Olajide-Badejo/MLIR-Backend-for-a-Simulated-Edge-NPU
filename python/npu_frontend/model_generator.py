"""Generate small, seeded PyTorch models and export them to ONNX.

The models are defined here and generated on demand, never downloaded, so the
test suite has no network dependency and every run is reproducible from the seed.
The core model is a LeNet style convolutional network sized for 1x1x28x28 input.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class LeNet(nn.Module):
    """A LeNet style CNN for 28x28 single channel input.

    Layer shapes for a 1x1x28x28 input:
      conv1 1->6  5x5 valid  -> 6x24x24, relu, maxpool 2 -> 6x12x12
      conv2 6->16 5x5 valid  -> 16x8x8,  relu, maxpool 2 -> 16x4x4
      flatten 256, fc1 256->120 relu, fc2 120->84 relu, fc3 84->10
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Registry of test models: name -> (module factory, input shape).
MODELS = {
    "lenet": (LeNet, (1, 1, 28, 28)),
}

# Bump this whenever a model definition or the export call changes in a way that
# would produce different weights or a different graph from the same seed. The
# exported .onnx files are not committed, so a recorded benchmark result is only
# reproducible if the seed and this version travel with it. Both go into the
# result manifest; see experiments/run_benchmarks.py.
GENERATOR_VERSION = 1


def build(name: str, seed: int = 0) -> tuple[nn.Module, tuple[int, ...]]:
    """Construct a seeded model and its input shape."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known models: {sorted(MODELS)}")
    factory, shape = MODELS[name]
    torch.manual_seed(seed)
    model = factory()
    model.eval()
    return model, shape


def export(name: str, path: str | Path, seed: int = 0, opset: int = 17) -> Path:
    """Build a seeded model and export it to ONNX at ``path``."""
    model, shape = build(name, seed=seed)
    dummy = torch.randn(*shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a seeded test model to ONNX.")
    parser.add_argument("name", choices=sorted(MODELS), help="model to export")
    parser.add_argument("output", help="output .onnx path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = export(args.name, args.output, seed=args.seed)
    print(f"wrote {out}")
