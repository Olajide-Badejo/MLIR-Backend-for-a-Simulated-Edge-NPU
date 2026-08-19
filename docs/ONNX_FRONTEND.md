<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# The ONNX frontend

*Diataxis type: reference, with a how to section at the end.*

This is the contract of `python/npu_frontend`: which ONNX operators it accepts,
what each one becomes, and what it refuses. It carries no narrative
justification, which belongs in the architecture decision records it points at,
and no commands except in the final section, which is the how to.

The frontend targets **opset 23**, resolved by measurement at P0 and recorded in
[`adr/0002-onnx-opset-pin.md`](adr/0002-onnx-opset-pin.md). A model declaring
any other opset is refused, naming the opset it found.

## The pipeline through this package

```
.onnx
  -> onnx.checker.check_model(full_check=True)
  -> opset check
  -> onnx.shape_inference.infer_shapes(check_type=True, strict_mode=True)
  -> per node conversion, through the registry in op_mapping.py
  -> MLIR module, built through the Python bindings as unregistered operations
  -> npu-opt, which parses, verifies, and prints
  -> text, returned
```

The last two steps are not optional and there is no mode that skips them. The
text `npu-opt` printed is what `import_model` returns, so unverified IR never
leaves the package. The mechanism and its one hazard are in
[`adr/0004-frontend-ir-emission-mechanism.md`](adr/0004-frontend-ir-emission-mechanism.md).

`./build/bin/npu-opt` is therefore a **runtime** dependency of this package, not
a test one. It is located from `NPU_OPT` in the environment, then from
`build/bin/npu-opt` under the repository root, then from `PATH`. When none of
the three resolves, `import_model` raises and names all three places it looked.

## Converters

Sixteen, one per accepted ONNX operator. `op_mapping.py`'s module docstring
carries the same list and a test asserts the two agree in both directions.

| ONNX operator | Becomes | Notes |
|---|---|---|
| `Add` | `npu.add` | through the broadcasting policy below |
| `AveragePool` | `npu.avg_pool2d` | `count_include_pad` and `dilations` rules below |
| `BatchNormalization` | `npu.batch_norm` | inference form only, constant parameters only |
| `Clip` | `npu.relu` | only for the bounds that are a relu; see below |
| `Concat` | `npu.concat` | a negative axis is normalised to non negative |
| `Conv` | `npu.conv2d` | rank 4, grouped, optional bias operand |
| `Flatten` | `npu.reshape` | `axis = 1` only; see the batch rule below |
| `Gemm` | `npu.matmul` | optional bias operand; `transB` folded into the constant |
| `GlobalAveragePool` | `npu.avg_pool2d` | kernel equal to the input's spatial extent |
| `Identity` | nothing | binds its output name to its input value |
| `MatMul` | `npu.matmul` | rank 2 by rank 2 |
| `MaxPool` | `npu.max_pool2d` | values only, never the `Indices` output |
| `Mul` | `npu.mul` | through the broadcasting policy below |
| `Relu` | `npu.relu` | |
| `Reshape` | `npu.reshape` | `allowzero` handled; see the batch rule below |
| `Transpose` | `npu.transpose` | `perm` must be a permutation of the input's axes |

`QuantizeLinear`, `DequantizeLinear` and `Pad` are refused with a reason more
specific than "unsupported": the first two arrive with their integer kernels and
calibrated models at the quantization phase, and `Pad` is not in the operator set
and is not planned. Every other operator is refused with a message naming the
node and listing the supported set.

## The rules, in the order they bite

### Attributes are read by declared type

Every attribute read switches on `AttributeProto.type` and returns the default
**only** when the attribute is genuinely absent from the node. An attribute
present with a type the converter did not ask for raises rather than being
coerced.

The case this exists for is the empty list. `pads = []` is a legal thing for a
graph transform to leave behind, and testing truthiness cannot distinguish it
from an absent `pads`. Read as absent it silently becomes four zeros; read as
present it is the wrong length and says so.

### Broadcasting, and the one carve out

Stated once, in `_binary_operands`, and applied to `Add` and `Mul`.

1. **Identical shapes pass through.** Both operands already have the result
   shape, and nothing happens.
2. **An initializer that broadcasts is materialised at import time**, as a same
   shaped constant, so the emitted IR needs no broadcasting operation.
3. **One carve out.** A constant that broadcasts against a rank 4 activation
   over the channel axis is **not** expanded. It is emitted as a rank 1
   `npu.constant` of length C and passed as the right hand operand.
4. **Anything else raises**, naming the node and both shapes. In particular a
   broadcast between two values computed at run time is refused: this project
   materialises a broadcast or refuses it, and does not emit one.

The carve out is load bearing rather than an optimisation.
`-npu-fuse-bias` folds `add(conv(x, w), b)` into the convolution's bias operand
and guards on a channel shaped constant addend. An expanded addend makes that
guard unmatchable on every model in the suite, so the pass's ablation row would
be a row of zeros and the phase would look done while doing nothing. Expansion
also inflates DRAM traffic by the expansion factor, which would move every
published byte count for a reason that is an importer artefact.

**Which initializer shapes are the carve out, and which are not.** ONNX
broadcasting aligns from the trailing axis, so against an `N x C x H x W`
activation the shapes that broadcast over the channel axis are `[C, 1, 1]` and
`[1, C, 1, 1]`. A literally rank 1 initializer of dims `[C]` broadcasts over the
**width** and is expanded like any other broadcast; matching it as a channel
constant would import a per column vector as a per channel one whenever the
channel count and the width happen to be equal. `[1, C, 1, 1]` is what the
dynamo exporter writes for a bias or a scale spelled `p.reshape(1, -1, 1, 1)` in
PyTorch, so this is the shape the carve out meets in practice.

**The rank 1 operand is always the right hand one.** `npu.add` and `npu.mul`
refuse a rank 1 left hand operand, so a channel broadcast has exactly one
spelling in the IR and the fusion pass has one form to match. Both operations
are commutative, so a node whose constant arrived on the left is commuted, and
the commutation is not observable in the result. The dialect side of this is
[`adr/0005-channel-broadcast-on-add-and-mul.md`](adr/0005-channel-broadcast-on-add-and-mul.md).

### `Clip` is a relu or it is refused

Since opset 11 the bounds are optional **inputs**, not attributes, so they are
read from the initializer list. An input name that is present but empty, and an
input that is a zero element tensor, both mean the bound was omitted.

A lower bound of 0 with an absent or infinite upper bound becomes `npu.relu`.
Every other pair raises, quoting the actual minimum and maximum. There is no
relu6 case: a general bounded activation belongs in the `npu.fused_op` region,
and a third activation enum case would be a migration to undo.

### `AveragePool`

`dilations` other than `[1, 1]` is **refused**. It arrived at opset 19 together
with a change to the output shape formula, and an operator whose specification
moved and whose converter did not is a silent wrong answer.

`count_include_pad = 1` is **refused when any pad is non zero**, and accepted
when every pad is zero. The condition is the content of the rule rather than a
softening of it. This project's average pool divides by the number of elements
that actually contributed, which is the `count_include_pad = 0` behaviour, and
the two settings disagree exactly when a window overlaps the padded region. With
every pad at zero no window ever does and the two produce bit identical results.
The distinction matters in practice because the dynamo exporter writes
`count_include_pad = 1` on every `AveragePool` it emits, pads or no pads, so an
unconditional refusal would make every average pool this suite exports
unimportable while proving nothing.

### `MaxPool`

The optional second output, `Indices`, is refused: this project's pooling
produces values only, and an importer that dropped that output would silently
change what the graph computes. `storage_order = 1` is refused for the same
reason, since it only describes the indices.

### `BatchNormalization`

The training form is refused, on the output count and the `training_mode`
attribute together, because ONNX ties the two facts to each other. Non constant
parameters are refused at import rather than left for the lowering to diagnose:
the earliest layer that can name the ONNX node is the one that should.

`momentum` is not read. It affects the running statistics update, which happens
during training and not during inference, so it has no effect on what an
inference graph computes.

### `Reshape` and `Flatten` preserve the batch

A reshape from rank 4 to rank 2 must keep dimension 0, and `Flatten` is accepted
at `axis = 1` only. Both refusals name the node and quote the shape that would
have been produced.

Section 11 calls this the most likely hidden bug in the whole frontend and the
reason is that nothing else catches it. A `(N, C, H, W)` activation reshaped to
`(1, N * C * H * W)` typechecks, feeds a `matmul` that typechecks, and computes
one enormous row instead of one row per sample. Every shape in the graph is
consistent; only the meaning is wrong.

`allowzero` arrived at opset 14, before this project's pin, so it is handled
rather than migrated. At 0, the default, a zero in the target shape copies the
corresponding input extent. At 1 a zero is a literal zero extent, which is an
empty tensor and is refused. The dynamo exporter writes `allowzero = 1`
unconditionally and with no zero in the target shape the two settings are
identical, so the refusal is conditioned on a zero actually being present.

### `Gemm`

`alpha` or `beta` other than 1.0 is refused, quoting the values: folding a scale
into the weights at import would change the constants the report publishes for a
reason invisible in the model. `transA = 1` is refused, because it would need a
transpose of the activation and this importer will not insert a full pass over
the data silently.

`transB = 1` is accepted and the weight is transposed **at import**. The weight
is a constant, so this costs nothing at run time and the emitted IR contains no
transpose at all. This is the shape `nn.Linear` exports as.

### Destinations and locations

Every `npu` compute operation carries a destination operand, and the importer
materialises it: a `tensor.empty` of the inferred result type, emitted
immediately before the operation that consumes it. `npu.constant` and
`npu.reshape` take no destination, because neither is a compute operation.

Every operation the importer creates carries a `NameLoc` with the ONNX node
name, and the returned text is printed with debug information so the locations
survive the round trip. A node the exporter left unnamed is given a
deterministic name derived from its operator type and its index in the graph, so
two imports of the same model produce identical locations.

### Shapes are cross checked, not trusted

For the shape only converters, which do their own arithmetic, the shape this
importer computes is compared against the shape ONNX inferred and a disagreement
raises. The two are independent implementations of the same rule, so a graph
where they differ is a graph one of them is wrong about.

Everywhere else the shape comes from ONNX's inference and the arithmetic is
checked by the `npu` dialect's own verifiers through `npu-opt`. Convolution and
pooling extents are therefore diagnosed by `NPUShapeUtils.cpp`, the one place
Section 7.2's formula lives, rather than by a second implementation in Python
that would then have to agree with it.

## Diagnostics

Two exception types, and the difference between them is which side is at fault.

- **`ONNXImportError`**: a model this project does not accept. Every message
  names the node and its operator type, or says what graph level fact it is
  about. Refusals from `onnx.checker` and from ONNX shape inference are
  re-raised as this type, with the original message kept in full, so a caller
  has one exception type to catch.
- **`VerificationError`**: IR this package emitted that `npu-opt` refused, or
  `npu-opt` not being found. This one means the frontend is at fault, not the
  model.

## How to use it

Import a model that already exists:

```python
from npu_frontend import import_model_file

print(import_model_file("model.onnx"))
```

Generate one of the suite's models first, then import it:

```python
from pathlib import Path
from npu_frontend import generate_model, import_model_file

path = generate_model("resnet_block", Path("/tmp/models"))
print(import_model_file(path))
```

From a shell, with the MLIR Python bindings on the path:

```bash
export PYTHONPATH="$HOME/llvm-project/build/tools/mlir/python_packages/mlir_core:$PWD/python"
python -c 'from npu_frontend import generate_model, import_model_file; \
           print(import_model_file(generate_model("lenet", "/tmp/models")))'
```

Point it at an `npu-opt` outside this repository:

```bash
export NPU_OPT=/somewhere/else/bin/npu-opt
```

Run the tests, which wire the bindings path themselves through
`test/Python/conftest.py` and need no `PYTHONPATH`:

```bash
python -m pytest test/Python -q
```
