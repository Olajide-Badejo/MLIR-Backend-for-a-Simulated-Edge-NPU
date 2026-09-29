<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: what fusing the relu into the integer instruction, and folding an identical pair, moves

- **id:** p14-fused-relu-and-pair-fold
- **written:** 2026-09-29
- **result field:** `sqnr_db_vs_onnxruntime` and the largest error in counts on
  the `normal` class at `-O0`, `quant_boundary_crossings` per model, the seven
  quantized goldens under `test/baseline/golden/int8/`
- **direction:** up or unchanged on every model and down on none; crossings
  down or unchanged on every model
- **magnitude bracket:** three models bit identical; `depthwise_separable` up
  by 1 to 6 dB, `lenet` and `lenet_batched` up by 0.5 to 5 dB and within 1.5 dB
  of each other, `resnet_block` up by 0 to 3 dB; crossings 44 to 32 over the
  seven, per model in the table below; four goldens move and three do not
- **answered at:** P14, Checkpoint B, the record that follows the commit that
  fuses the relu

*Written and committed before the relu fusion or the pair fold exists in any
form, and before any program with either has been compiled. What the
repository already records is an input to the reasoning rather than a
prediction: the -O0 accuracy table of `p14-quantized-accuracy.md`, the
committed profiles' scales, and the structure of the seven graphs.*

## Hypothesis

### 1. Which models can move at all

The relu fusion changes a program only where a calibrated convolution or
matrix multiplication has a relu as its **only** reader in the graph the
calibrator sees, and the pair fold only where that relu's output is itself the
input of a calibrated operation. At `-O0` that is a question about the ONNX
graphs, answered before anything is compiled:

| Model | A relu reads a calibrated operation | The relu's output feeds a calibrated operation |
|---|---|---|
| `conv_bn_relu_stack` | no, a batch norm does | no |
| `depthwise_separable` | yes, both convolutions | yes, the first into the second |
| `dilated_stack` | no, a clip and an add do | no |
| `inception_block` | no, the concat does | no |
| `lenet` | yes, four of its five calibrated operations | yes, `relu_2` and `relu_3` into the next matrix multiplication; the two convolution relus go into max pools |
| `lenet_batched` | the same as `lenet` | the same |
| `resnet_block` | yes, the first convolution | yes, into the second |

**So three models compile to the same program they compile to today**, byte
for byte, and their goldens and accuracy numbers do not move by a bit. The
other four move.

### 2. The crossings, counted from the graphs

Today every integer instruction has one `QUANT` before it and one `DEQUANT`
after it, 2 crossings each. A fused relu moves the output quantize past the
relu; a pair fold then removes the `DEQUANT` of that output and the `QUANT` of
the next input together, where the relu's output is the next calibrated
operation's input. A relu into a max pool keeps both, because `POOL_MAX`
refuses i8 by Section 14's own boundary.

| Model | Today | Predicted | The `RELU` instructions removed |
|---|---|---|---|
| `conv_bn_relu_stack` | 6 | 6 | 0 |
| `depthwise_separable` | 4 | **2** | 2 |
| `dilated_stack` | 4 | 4 | 0 |
| `inception_block` | 6 | 6 | 0 |
| `lenet` | 10 | **6** | 4 |
| `lenet_batched` | 10 | **6** | 4 |
| `resnet_block` | 4 | **2** | 1 |
| total | 44 | **32** | 11 |

### 3. The accuracy, and why it can only go up

Today a relu after a calibrated operation costs two roundings on the elements
it keeps: the operation's output rounded to its own scale `s_c`, then, where
the relu's output is quantized again, rounded to the relu output's scale
`s_r`. Fused, it costs one rounding, at `s_r`. The committed profiles put
`s_r / s_c` between 0.44 and 0.55 on every one of the eleven relus, because the
relu's range is `[0, max]` of a range that was near symmetric, so the removed
rounding is the coarse one and carried about four times the fine one's noise
power. Where the relu fed a max pool, the pool's output range is the relu's
exactly and its quantize becomes a rounding of values already on its grid,
which is exact.

Nothing else in the arithmetic changes: the weights, the biases, the input
quantize and the output rounding of the last layer are the same numbers. So no
model can get worse except by the one route that changes the result of a
single element: an element whose two roundings happened to cancel. That is a
per element effect and not a per model one, and over thousands of elements it
does not reverse the sign of a four to one reduction in one noise term.

The size depends on what share of each model's noise the removed roundings
were, which item 3 did not measure separately. The brackets are therefore
wide, and ordered by how much of each model's quantized path is a fused relu:

| Model | Today | Predicted change |
|---|---|---|
| `depthwise_separable` | 46.17 dB | up by 1 to 6 dB: both of its layers fuse, and its first relu's double rounding goes entirely |
| `lenet` | 43.89 dB | up by 0.5 to 5 dB: four of five layers fuse; the last layer's own rounding, which dominated its largest error, does not change |
| `lenet_batched` | 42.97 dB | up by 0.5 to 5 dB, and within 1.5 dB of `lenet` |
| `resnet_block` | 46.14 dB | up by 0 to 3 dB: one layer of two fuses, beside an f32 residual path |
| the other three | | unchanged to the last digit |

**The largest error in counts stays within each model's current budget**: 3,
1, 2, 2, 2, 2 and 3 counts in the order of the budget table.

### 4. The integer reference

Zero counts on all 35 model and class pairs, as today: the reference makes the
same decision about the relu that the lowering makes and computes the fused
instruction by the same unfolded arithmetic with the clamp at the output zero
point.

### 5. The fused form against the unfused one, on the machine

Over random draws, the fused instruction and the program that rounds to `s_c`,
dequantizes, runs the relu in f32 and quantizes to `s_r` differ only by the
unfused path's extra rounding, so by at most
`floor(1 + 2^-n_r + (1/2 + 2^-n_c) * s_c / s_r + 256 * s_c / s_r * 2^-24)`
counts per element, where `n_r` and `n_c` are the two shifts; the derivation
is in the test's docstring. At a ratio near 2 that is **2 counts**. I predict
the largest difference observed is 1 or 2 counts and never more than the bound.

### 6. The pair fold

A dequantize then a quantize with bitwise equal scales and zero points returns
every one of the 256 int8 values unchanged, on the machine and in the numpy
reference, for every scale and zero point the committed profiles hold and at
the boundary scales, so removing the pair moves no value. A pair whose scale
or zero point differs is left alone.

### 7. fp32

Nothing here reaches an fp32 compilation: the fusion needs `weight_scales`, the
fold needs a quantize, and the fusion pass's new guard needs `weight_scales`
too. So the 42 baseline cells and the 21 fp32 goldens do not move by a bit.

## What would falsify it

- Any model's SQNR below today's figure, or any of the three unchanged models'
  outputs differing from today's by any bit.
- A model outside its bracket, or `lenet` and `lenet_batched` more than 1.5 dB
  apart.
- A crossing count other than the table's, or a `RELU` count other than it.
- Any disagreement with the integer reference, on any model and class.
- A fused against unfused difference above the derived bound, or one of 0
  everywhere, which would mean the test compared a program with itself.
- A pair fold that moves any value, or that fires on a mismatched pair.
- Any fp32 cell field or fp32 golden byte moving.
