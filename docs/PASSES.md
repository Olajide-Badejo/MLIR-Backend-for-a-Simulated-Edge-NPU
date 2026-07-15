# Optimization passes

Each pass is shown with a before and after example. Run any of these yourself with
`npu-opt <file> -<pass-name>`.

## Constant folding and canonicalization (`-canonicalize`)

The ops carry `Pure` traits and folders, so the built in canonicalizer folds
constant subgraphs and applies the dialect's canonicalization patterns (relu
idempotence, reshape identity, reshape of reshape).

Before:

```mlir
%0 = npu.constant {value = dense<[1.0, 2.0]> : tensor<2xf32>} : tensor<2xf32>
%1 = npu.constant {value = dense<[3.0, 4.0]> : tensor<2xf32>} : tensor<2xf32>
%2 = npu.add %0, %1 : tensor<2xf32>
```

After:

```mlir
%0 = npu.constant {value = dense<[4.0, 6.0]> : tensor<2xf32>} : tensor<2xf32>
```

## BatchNorm folding (`-npu-fold-batchnorm`)

At inference the batch norm parameters are constants, so `bn(conv(x, W, b))` with
no activation between the two is another convolution with rescaled weights and
bias. The pass does the tensor arithmetic at compile time.

Before:

```mlir
%c = npu.conv2d %x, %w {...} : (...) -> tensor<1x2x2x2xf32>
%y = npu.batch_norm %c, %scale, %offset, %mean, %var {epsilon = 1.0e+00 : f32} : (...)
```

After (weights and bias absorb the normalization; the batch_norm is gone):

```mlir
%w2 = npu.constant ... : tensor<2x1x1x1xf32>
%b2 = npu.constant ... : tensor<2xf32>
%y  = npu.conv2d %x, %w2, %b2 {...} : (...) -> tensor<1x2x2x2xf32>
```

## Operator fusion (`-npu-fuse-ops`)

A trailing relu is folded into a producing conv or matmul by setting the fused
activation, so the intermediate stays in scratchpad rather than round tripping to
DRAM.

Before:

```mlir
%0 = npu.conv2d %x, %w, %b {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]} : (...)
%1 = npu.relu %0 : tensor<1x4x8x8xf32>
```

After:

```mlir
%0 = npu.conv2d %x, %w, %b {activation = 1 : i32, ...} : (...)
```

## Dead code elimination (`-canonicalize`, `-symbol-dce`)

Because every op is `Pure`, an op whose results are unused is removed by the
canonicalizer, and unused private functions are removed by symbol-dce. No custom
DCE pass is needed.

## Lowering and allocation

`-npu-lower-to-npuisa` converts the tensor level to the instruction level (see
[ARCHITECTURE.md](ARCHITECTURE.md)), and `-npu-allocate-scratchpad` assigns byte
offsets and spills to DRAM when the working set exceeds the budget (see
[ISA_MANUAL.md](ISA_MANUAL.md)).

## What the levels buy

The benchmark suite records this. At the 1 MB budget on the LeNet model, moving
from O0 to O1 removes the dead transposed Gemm weight constants (the instruction
count drops and DRAM traffic roughly halves), and O2 fusion folds the activations
(the instruction count and simulated cycles drop further). The exact numbers live
in `experiments/results/` and are simulated estimates.
