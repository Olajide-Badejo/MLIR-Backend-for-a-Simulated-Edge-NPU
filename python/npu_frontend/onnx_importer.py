"""Import an ONNX model into npu dialect MLIR.

The importer runs onnx.checker and onnx.shape_inference first, then walks the
graph in topological order building npu ops. Weights become npu.constant. The
result is textual MLIR; run it through npu-opt to verify it against the dialect.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from mlir import ir
from mlir.dialects import func as func_dialect
from onnx import numpy_helper

from . import (
    _mlir,  # noqa: F401  (side effect: puts mlir on the path)
    op_mapping,
)
from .op_mapping import UnsupportedOpError


class _Context:
    """Shared state threaded through the per node converters."""

    def __init__(self, shapes, initializers):
        self.values: dict[str, ir.Value] = {}
        self._shapes = shapes
        self._initializers = initializers

    def shape(self, name):
        if name not in self._shapes:
            raise UnsupportedOpError(f"no inferred shape for tensor {name!r}")
        return self._shapes[name]

    def init(self, name):
        return self._initializers.get(name)


def _collect_shapes(graph) -> dict[str, tuple]:
    shapes: dict[str, tuple] = {}
    for coll in (graph.input, graph.output, graph.value_info):
        for vi in coll:
            dims = tuple(d.dim_value for d in vi.type.tensor_type.shape.dim)
            shapes[vi.name] = dims
    return shapes


def import_model(path: str | Path) -> str:
    """Return npu dialect MLIR text for the ONNX model at ``path``."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    initializers = {
        init.name: numpy_helper.to_array(init).astype(np.float32)
        for init in graph.initializer
    }
    shapes = _collect_shapes(graph)
    for name, arr in initializers.items():
        shapes.setdefault(name, tuple(arr.shape))

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.create()

        graph_inputs = [i for i in graph.input if i.name not in initializers]
        in_types = [op_mapping.tensor_type(shapes[i.name]) for i in graph_inputs]
        out_types = [op_mapping.tensor_type(shapes[o.name]) for o in graph.output]
        ftype = ir.FunctionType.get(in_types, out_types)

        with ir.InsertionPoint(module.body):
            fn = func_dialect.FuncOp("main", ftype)
        block = fn.add_entry_block()

        conv_ctx = _Context(shapes, initializers)
        for info, arg in zip(graph_inputs, block.arguments, strict=True):
            conv_ctx.values[info.name] = arg

        with ir.InsertionPoint(block):
            for name, arr in initializers.items():
                conv_ctx.values[name] = op_mapping.make_constant(arr)
            for node in graph.node:
                op_mapping.convert_node(node, conv_ctx)
            results = [conv_ctx.values[o.name] for o in graph.output]
            func_dialect.ReturnOp(results)

    return str(module)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Import an ONNX model to npu MLIR.")
    parser.add_argument("model", help="input .onnx file")
    parser.add_argument("-o", "--output", help="output .mlir file (default stdout)")
    args = parser.parse_args()
    text = import_model(args.model)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
