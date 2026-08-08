"""The compiler must refuse a batch size it cannot compute correctly.

The simulator's convolution kernel hardcodes batch index 0 and its pooling
kernel never sees the batch dimension at all, so a model with N greater than 1
returns correct data for the first image and whatever the scratchpad happened to
hold for the rest. Nothing used to reject it: the importer accepted it, the
verifiers accepted it, the allocator sized buffers for the full batch, and the
simulator quietly produced garbage.

Until phase U6 adds real batch loops, the answer is to refuse. These tests pin
that refusal down at the importer, which is the earliest layer that can see the
problem. The matching verifier level tests are in
test/Dialect/NPU/invalid.mlir, and the numerical proof of the underlying bug is
recorded in docs/ASSESSMENT.md section 2.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from npu_frontend import model_generator, onnx_importer
from npu_frontend.op_mapping import UnsupportedOpError


def _export_lenet_with_batch(path: Path, batch: int) -> Path:
    """Export the seeded LeNet at an arbitrary static batch size."""
    model, shape = model_generator.build("lenet", seed=0)
    dummy = torch.randn(batch, *shape[1:])
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    return path


@pytest.mark.parametrize("batch", [2, 4])
def test_batch_greater_than_one_is_rejected(tmp_path, batch):
    onnx_path = _export_lenet_with_batch(tmp_path / f"lenet_b{batch}.onnx", batch)

    with pytest.raises(UnsupportedOpError) as excinfo:
        onnx_importer.import_model(onnx_path)

    message = str(excinfo.value)
    # The diagnostic has to name the offending construct, not just complain.
    assert f"batch size {batch}" in message
    assert str((batch, 1, 28, 28)) in message
    # It also has to say this is a tracked limitation rather than a design
    # choice, so a reader knows whether to work around it or wait for it.
    assert "U6" in message


def test_batch_one_still_imports(tmp_path):
    """The guard must not be so broad that it rejects the supported case."""
    onnx_path = _export_lenet_with_batch(tmp_path / "lenet_b1.onnx", 1)
    text = onnx_importer.import_model(onnx_path)
    assert "npu.conv2d" in text


def test_rank_four_weight_is_not_mistaken_for_a_batch(tmp_path):
    """A conv weight is OIHW, so its dimension 0 is output channels, not batch.

    LeNet's first weight is 6x1x5x5. A guard written as "reject any rank 4
    tensor whose leading dimension is not 1" would reject every real model, so
    this pins the guard to activations only.
    """
    onnx_path = _export_lenet_with_batch(tmp_path / "lenet_w.onnx", 1)
    text = onnx_importer.import_model(onnx_path)
    assert "6x1x5x5" in text
