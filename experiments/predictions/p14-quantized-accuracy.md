<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: how far each quantized model lands from onnxruntime, and from its own integer reference

- **id:** p14-quantized-accuracy
- **written:** 2026-09-24
- **result field:** `sqnr_db_vs_fp32_simulated`, `max_abs_error_vs_onnxruntime`
- **direction:** the signal to quantization noise ratio against onnxruntime
  falls with the number of quantized layers in series, so `lenet` and
  `lenet_batched` are the lowest of the seven and the two models whose output
  also carries an unquantized path or passes a single quantized layer,
  `resnet_block` and `inception_block`, are the highest
- **magnitude bracket:** on the `normal` class, at `-O0`, per output channel
  weights, `minmax`, 32 calibration inputs: every model between 22 and 50 dB
  against onnxruntime; `lenet` between 25 and 35 dB and within 1.5 dB of
  `lenet_batched`; the largest absolute error against onnxruntime between 1 and
  10 counts of the output's own scale on every model; and against the numpy
  integer reference from the same profile, 0 counts on six models and at most 1
  on `conv_bn_relu_stack`, on all five input classes
- **answered at:** P14, Checkpoint B's item 3, in the commit that adds the seven
  model end to end test

*Written and committed before any quantized model has been compared with
onnxruntime. What the repository already records is an input to the reasoning
rather than a prediction: quantized LeNet agrees with the numpy integer
reference exactly on all five input classes, measured at the contraction's
boundary. No accuracy against onnxruntime has been measured for any model at
any granularity, and none from the interim per tensor execution will be.*

## Hypothesis

### 1. The noise sources, counted per model

The quantized program is exact integer arithmetic between the points where a
value is rounded, so its error against the f32 answer is the sum of those
roundings carried to the output. Three kinds, with their size relative to the
tensor they round:

| Rounding | Why it is that size | Relative noise power |
|---|---|---|
| a two sided activation, a convolution's output or the model's input | `minmax` over 32 draws puts the range near plus and minus 4.5 standard deviations, so a step is about 0.035 of one, and uniform rounding noise is a twelfth of a step squared | about -40 dB |
| a one sided activation after a ReLU | the range is `[0, max]`, half as wide at the same maximum, and the signal keeps half its power | about -43 dB |
| a weight, per output channel | the seeded initializers are uniform, so the largest magnitude is the bound and the step is the bound over 127 | about -48 dB |

Counting them along each model's longest quantized path, assuming each layer
passes relative noise through at a gain near one, which is what an untrained
network with fan in scaled initializers does:

| Model | Quantized layers in series | Sources | Predicted SQNR |
|---|---|---|---|
| `lenet` | 5 | 15 | about 30 dB |
| `lenet_batched` | 5 | 15 | about 30 dB, within 1.5 dB of `lenet` |
| `conv_bn_relu_stack` | 3 | 9 | about 32 dB |
| `depthwise_separable` | 2 | 6 | about 34 dB |
| `dilated_stack` | 2 | 6 | about 34 dB |
| `inception_block` | 1, three in parallel | 3 per branch | about 36 dB |
| `resnet_block` | 2, beside an unquantized residual path | 6 on the convolution path only | about 40 dB |

`resnet_block` is highest because its residual path is f32: the output carries
the exact input beside the quantized branch, so the signal grows and the noise
does not.

### 2. The largest error in counts

An SQNR near 30 to 40 dB puts the noise's standard deviation at roughly a half
to one count of an output scale that spans the tensor's calibrated range, and
the largest of a few hundred to a few thousand elements lands a few standard
deviations out: between 1 and 10 counts.

### 3. Against the integer reference

The reference computes every integer instruction by the unfolded form and every
f32 operation with `refexec`. Where the only f32 operations are exact ones, a
ReLU, a max pool, a reshape, a transpose or a concat, the two answers are the
same bits. Where an f32 operation rounds, it can land a value on the other side
of a later quantization boundary, and that is one count. `conv_bn_relu_stack` is
the one model with rounding f32 arithmetic, its batch norm, **between** two
quantized layers; the average pools and adds of the others come after the last
quantize and move the answer by units in the last place, which is no count at
all.

### 4. What is not predicted

The four other input classes against onnxruntime. The two constant classes sit
far outside the range the calibration draws covered, so the input quantize
saturates by design and the number would measure the saturation. They are
measured and recorded, not predicted and not budgeted.

## What would falsify it

- Any model below 22 dB or above 50 dB on the `normal` class, or LeNet outside
  25 to 35 dB, or `lenet` and `lenet_batched` more than 1.5 dB apart.
- `lenet` or `lenet_batched` not the lowest pair, or `resnet_block` and
  `inception_block` not the two highest.
- A largest error outside 1 to 10 counts on any model.
- Any disagreement with the integer reference beyond 1 count anywhere, or any
  disagreement at all on the six models predicted exact.
