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
count drops from 28 to 25 and DRAM traffic roughly halves, 339 KB to 176 KB), and
O2 fusion folds the activations (25 to 21 instructions, 13,008 to 12,710 simulated
cycles). The exact numbers live in `experiments/results/` and are simulated
estimates.

## What each pass buys, measured

The level numbers above are cumulative and cannot say what one pass contributes.
`run_benchmarks.py` records a leave one out ablation for every `-O2` pass at both
budgets: the full pipeline minus that pass, compiled, encoded, simulated, and
checked against onnxruntime. The deltas below are ablated minus full `-O2`, so a
positive number is what removing the pass costs. They regenerate with the results
and are plotted in `docs/images/ablations.png`.

| Pass | 1 MB: instrs, cycles | 140 KB: instrs, cycles, DRAM |
|---|---|---|
| `-canonicalize` (all occurrences) | +0, +0 | +0, +0, +0 |
| `-npu-fuse-ops` | **+4, +298** | +2, **-96**, **-6.1 KB** |
| `-symbol-dce` | +0, +0 | +0, +0, +0 |

Three things these numbers say that the level table cannot.

**Canonicalization is redundant at `-O2`, and load bearing at `-O1`.** Removing
both of its occurrences from `-O2` produces a byte identical program.
`-npu-fuse-ops` drives its patterns with `applyPatternsGreedily`, whose fixed
point loop folds constants and erases dead ops as a side effect, so it already
does everything canonicalization contributed here. At `-O1`, where it is the only
pass, it is responsible for the entire DRAM halving. A pass can be worth having
at one level and redundant at another, and only the ablation shows it.

**Fusion is counterproductive under memory pressure.** At the tight budget,
removing it *saves* 96 simulated cycles and 6.1 KB of DRAM traffic. Folding an
activation into its producer extends that value's live range, and under a budget
that already forces spilling, a longer live range buys a spill and a reload that
cost more than the fused instruction saved. Fusion is a win at 1 MB and a loss at
140 KB, which is an argument for making it budget aware rather than
unconditional.

**`-symbol-dce` earns nothing measurable on this model.** It removes unused
private functions and LeNet has none after import. It is cheap and correct to
keep, but no number here justifies it; that is worth stating rather than implying
otherwise.
