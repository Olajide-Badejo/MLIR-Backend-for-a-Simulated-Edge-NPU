# ONNX frontend

The frontend turns a trained ONNX model into `npu` dialect MLIR. It lives in
`python/npu_frontend/` and is built on the MLIR Python bindings produced by the one
time LLVM/MLIR build.

## Pipeline

1. `onnx.checker.check_model` rejects a malformed model up front.
2. `onnx.shape_inference.infer_shapes` gives every intermediate tensor a static
   shape, which the importer turns into `tensor<...xf32>` types.
3. The graph is walked in topological order. Initializers (weights, biases, batch
   norm parameters) become `npu.constant`. Each node is dispatched to a converter
   in `op_mapping.py` that builds the matching `npu` op.
4. The result is textual MLIR. Run it through `npu-opt` to verify it against the
   dialect verifiers; the frontend itself builds ops in generic form because the
   `npu` dialect is not registered in the Python context.

Anything unsupported fails loudly with an `UnsupportedOpError` that names the op
and node. The frontend never emits silently wrong IR.

## Supported ops (opset 17)

| ONNX op | npu op | Notes |
|---|---|---|
| Conv | `npu.conv2d` | NCHW, optional bias, strides/pads/dilations/group |
| Gemm | `npu.matmul` | `transB` folds into a transposed weight constant; alpha, beta must be 1, transA unset |
| Relu | `npu.relu` | |
| MaxPool | `npu.max_pool2d` | kernel_shape, strides, pads |
| AveragePool | `npu.avg_pool2d` | kernel_shape, strides, pads |
| Reshape | `npu.reshape` | result shape from shape inference |
| Flatten | `npu.reshape` | expressed as a reshape to the flattened shape |

MatMul, Add, BatchNormalization, GlobalAveragePool, Concat, and Clip are part of
the documented target subset and are added as the test models that exercise them
are introduced.

## Usage

```bash
# Generate a seeded model, then import it.
python -m npu_frontend.model_generator lenet lenet.onnx --seed 0
python -m npu_frontend.onnx_importer lenet.onnx -o lenet.mlir
npu-opt lenet.mlir            # verifies and pretty prints
```

The MLIR bindings are located through the `MLIR_PYTHON_PACKAGES` environment
variable, defaulting to the build tree path in `docs/BUILD.md`.
