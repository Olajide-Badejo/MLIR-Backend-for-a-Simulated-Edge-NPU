<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: the quantized models at -O1 and -O2, with the calibration after the folds

- **id:** p14-quantized-o1-o2
- **written:** 2026-09-29
- **result field:** `sqnr_db_vs_onnxruntime` and the largest error in counts on
  the `normal` class at `-O1` and `-O2`, `quant_boundary_crossings` and the f32
  `RELU` count per model and level, the integer reference on all five classes,
  the fourteen quantized goldens `test/baseline/golden/int8/<model>-O1-out0`
  and `-O2-out0`, and `conv_bn_relu_stack`'s per channel weight scales before
  and after the batch norm fold
- **direction:** `-O1` identical to `-O0` on all seven; `-O2` identical to `-O0`
  on five, and up or unchanged on the two whose program changes, down on none
- **magnitude bracket:** `conv_bn_relu_stack` up by 0.5 to 6 dB,
  `dilated_stack` up by 0.3 to 5 dB; crossings 32 at `-O0` to 28 at `-O2`, f32
  relus 6 to 3, per model in the table below; the fold widens the channel scale
  spread of the batch norm model by about a factor of two
- **answered at:** P14, the record that follows the commit that extends the
  quantized end to end test to `-O1` and `-O2`

*Written and committed after the calibration moved (7e313eb) and before any
quantized program has been compiled at `-O1`, or at `-O2` with the pass in its
new position. The inputs to the reasoning are committed: the fp32 `-O2` tensor
level programs in `experiments/models/`, which are what the calibration now
sees, since it runs after the passes that made them and before any that would
change them further; the profiles' `graph` sections and ranges; the `-O0`
accuracy table in `npu_frontend.tolerances`; and the note of 2026-09-24 in the
engineering log that reproduced the fold in numpy.*

## Hypothesis

### 1. What the calibration sees at -O2

The committed fp32 `-O2` programs show every convolution or matrix
multiplication whose only reader is a relu inside an `npu.fused_op`, and the
calibration puts each back into the block, since the profile names all of
them. So the question per model is only what the folds changed:

| Model | Changed by `-O2` before the calibration | So the calibrated program is |
|---|---|---|
| `conv_bn_relu_stack` | both batch norms folded into their convolutions, which now write `bn0` and `bn1` | different from `-O0`'s: each convolution's result is the batch norm's, a relu reads it, so both relus fuse and the pair between the two layers folds |
| `depthwise_separable` | nothing | `-O0`'s |
| `dilated_stack` | the constant add `biased` fused into `conv1` as its bias | different: `conv1` writes `biased`, which the relu `activated` reads, so that relu fuses. `clipped` after `conv0` is a `Clip` in the graph and does not fuse, at any level |
| `inception_block` | nothing | `-O0`'s, and the three identical input quantizes meet the CSE that now runs after them |
| `lenet`, `lenet_batched` | nothing | `-O0`'s |
| `resnet_block` | nothing | `-O0`'s |

**`-O1` runs no fold and no fusion**, so the calibration still runs first
there, and constant folding and one canonicalization find nothing in a QDQ
program that they could change the value of.

### 2. The crossings and the f32 relus

| Model | `-O0` and `-O1` | `-O2` | Why `-O2` differs |
|---|---|---|---|
| `conv_bn_relu_stack` | 6, 2 | **4, 0** | the two relus fuse and the pair between the layers folds; the pool and the head keep theirs |
| `depthwise_separable` | 2, 0 | 2, 0 | |
| `dilated_stack` | 4, 2 | **4, 1** | `activated` fuses; the pair count is the same because nothing after it is calibrated |
| `inception_block` | 6, 1 | **4, 1** | one `QUANT` of the shared input where there were three |
| `lenet` | 6, 0 | 6, 0 | |
| `lenet_batched` | 6, 0 | 6, 0 | |
| `resnet_block` | 2, 1 | 2, 1 | |
| total | 32, 6 | **28, 3** | |

No `npu.fused_op` survives in any of the seven quantized `-O2` programs, and
every convolution and matrix multiplication contracts at both levels.

### 3. Bit identity

- **`-O1`: every model's output bit identical to its `-O0` output**, on every
  input class, so the seven `-O1` goldens are byte identical to the `-O0`
  quantized goldens. They are added anyway, because the golden layout is per
  level and a later change that separated the two levels should show as a
  moved tensor rather than a missing one.
- **`-O2`: five models bit identical to `-O0`**, `depthwise_separable`,
  `inception_block`, `lenet`, `lenet_batched` and `resnet_block`, for the
  reason in section 1: the same integer program, and the merged quantize of
  `inception_block` computes what each of the three computed. Their `-O2`
  goldens are byte identical to their `-O0` ones.
- **`-O2`: `conv_bn_relu_stack` and `dilated_stack` differ**, and their `-O2`
  goldens are new tensors.

### 4. The accuracy of the two that move

**`conv_bn_relu_stack`**, 33.80 dB and 2.24 counts at `-O0`. At `-O0` each
convolution's result is rounded at its own scale, 0.0324 and 0.0376, and then
the batch norm multiplies that rounding by its per channel factor, 0.56 to
2.23; the first layer's result is rounded a second time at the relu's scale,
0.0205, on its way into the second convolution. At `-O2` each layer is rounded
once, at the relu output's scale, 0.0205 and 0.0235, and the pair between them
folds. The weights are the same integers up to the sign of each channel's
factor, because a per channel symmetric scale moves with the channel. The
pool, the flatten quantize at 0.00427 and the head are unchanged, and the
global average pool attenuates the convolution layers' independent noise
before the head sees it, which is why the bracket is wide at the bottom:
**up by 0.5 to 6 dB**. The output count is the head's scale at both levels,
0.00274, so the counts compare directly: **at most 2.24, predicted 1.2 to
2.2**, inside the budget of 3.

**`dilated_stack`**, 34.40 dB and 1.10 counts at `-O0`. At `-O0` the last
layer is rounded at `conv1`'s scale, 0.0419, and the bias add and the relu run
in f32 after it. At `-O2` it is rounded once, at `activated`'s scale, 0.0192,
with the bias inside the instruction's accumulator. Everything before it is the
same: the input quantize, `conv0`'s output at 0.0419, `clipped` in f32 and
`conv1`'s input at 0.0229. **Up by 0.3 to 5 dB.**

**Its count bound is the one at risk, and the reason is the unit rather than
the error.** The output's count is the scale of the dequantize the output comes
from, which shrinks from 0.0419 to 0.0192, a factor of 0.459. At `-O0` the
largest error was 0.0461, of which the final rounding contributed at most
0.0210, so at least 0.025 came from before it and does not change. At `-O2`
the largest error is predicted at 0.030 to 0.045 absolute, below `-O0`'s, and
that is **1.6 to 2.4 counts of the new unit**, against a budget of 2. If it is
above 2, the `-O0` budget does not hold at `-O2` for a program that is more
accurate in every absolute sense, and that is reported as a finding rather
than met by a wider budget.

**The other five are `-O0`'s numbers to the last digit**, and so is every
model at `-O1`: 33.80, 52.68, 34.40, 36.29, 45.24, 46.76 and 47.83 dB.

### 5. The integer reference

Zero counts on all 35 model and class pairs at `-O1` and at `-O2`, as at
`-O0`: the reference executes the same tensor level program the lowering
reads, with the fused relus and the folded bias computed by the same unfolded
arithmetic.

### 6. The channel scale spread before and after the fold

The `weight_scales` of `conv_bn_relu_stack`'s two convolutions, read out of the
compiled programs: at `-O0` from the unfolded filter, where they equal the
profile's entries bit for bit, and at `-O2` from the folded one. The
engineering log of 2026-09-24 reproduced the fold in numpy and put the
factor at 0.56 to 2.23 and the spread, the largest channel scale over the
smallest, at about 1.5 before and about 3.2 after.

- The `-O2` scales equal `max |w'_c| / 127` of that numpy reproduction of the
  folded filter, bit for bit on every channel of both convolutions.
- The spread roughly doubles on each convolution, to between 2.5 and 4.5, and
  per tensor weights would give the smallest channel 127 / spread levels, 28 to
  51 where it had about 85. Per channel weights, the default, lose nothing to
  it, which is Section 14's argument and the reason the scales come from the
  constant.

### 7. The goldens and fp32

Fourteen quantized goldens arrive, seven per level, and none of the seven
`-O0` quantized goldens moves. No fp32 cell field, fp32 golden byte or fp32
binary moves: the calibration runs only when a profile is named, and the
move of 7e313eb left all 217 planned cells byte identical.

## What would falsify it

- Any `-O1` output differing from its `-O0` output by any bit, or any of the
  five `-O2` models named identical differing by any bit.
- Either moving model's SQNR below its `-O0` figure, or outside its bracket.
- `conv_bn_relu_stack` above 2.24 counts.
- A crossing or f32 `RELU` count other than the table's, or a surviving
  `npu.fused_op`, or a convolution or matrix multiplication left in f32.
- Any disagreement with the integer reference, on any model, class or level.
- An `-O2` weight scale on `conv_bn_relu_stack` that differs from the numpy
  fold's by any bit, or a spread that does not grow.
- Any `-O0` quantized golden, fp32 golden or fp32 cell field moving.
