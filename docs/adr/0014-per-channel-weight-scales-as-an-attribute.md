<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 14. Carry per output channel weight scales as a tensor level attribute

- **Status:** Accepted
- **Date:** 2026-09-22
- **Diataxis type:** explanation

## Context

Section 14 makes weight scales per output channel. The instruction level can
express that: a quantized compute instruction takes a fourth operand holding
one requantization multiplier and one shift per output channel. The question
this record answers is how the scales get from the calibration profile to the
contraction that builds that operand.

**The QDQ form cannot carry them, and that is the finding underneath this
decision.** `npu.quantize` and `npu.dequantize` each carry a single `scale`
attribute, by Section 7.2 and by the shape of an exported QDQ graph. So the
tensor level expresses **per tensor activation quantization exactly and per
channel weight quantization not at all**. `-npu-calibrate` therefore wraps
activations and leaves weights in f32, and something else has to carry the
weight half.

## Decision

An optional `weight_scales` attribute on `npu.conv2d` and `npu.matmul`, a
`DenseF32ArrayAttr` of one scale per output channel, written by
`-npu-calibrate` from the committed profile and read by the contraction in the
lowering.

Three rules hold it honest, each with a negative test:

- it may appear only where the operation is actually quantized, which at this
  level means its data operand is the result of an `npu.dequantize`;
- its length equals the output channel count on the reference axis, which is
  the filter's axis 0 for a convolution and the right operand's axis 1 for a
  matrix multiplication;
- every scale is finite and strictly positive.

## The alternatives, and why they lost

**Quantize the weights per tensor in the QDQ form.** This is the one the
representation invites, because it needs no new attribute: take one scale for
the whole weight tensor and wrap it like an activation. It was rejected
because it throws away the granularity the phase gate exists to measure.
Section 14 asks for an ablation of per channel against per tensor, and a
compiler whose only representable form is per tensor cannot run one arm of it.
The suite contains a depthwise separable block and a Conv plus BatchNorm plus
ReLU stack precisely because they are the structures per tensor scaling
handles worst.

**Give the contraction a profile path of its own.** The lowering would read
the same JSON the pass read and look the operation up again. It was rejected
because it puts a file dependency where IR belongs: a lowering that reads a
file is a lowering whose output depends on something the IR does not carry, so
the same module compiled twice with the file moved would produce two different
programs and nothing in the IR would say why. It also duplicates the join from
operation to profile entry, which is the fragile part, in a second place.

## Consequences

The profile stays the single source of truth and the attribute is derived from
it, which is asserted end to end rather than assumed: a test compiles a real
model and checks that every attribute in the IR is one of the profile's own
scale lists and that every quantizable operation the profile names has one.
When the contraction lands, the third leg joins it, so a divergence between
the profile and the requantization values that reach the instruction is a red
rather than a discovery.

An fp32 compilation carries no such attribute, which is checked from both
ends: a test asserts it is absent, and the verifier refuses one whose operand
did not come from a dequantize.

The tiling interface passes the attribute through unchanged. A tile that keeps
its output channels carries the right scales; one that splits the channel axis
carries the wrong number and is refused by the length rule, by name. That is
deliberate: slicing them would be writing the quantized tiling path before
anything exercises it, and dropping them would silently turn a quantized
convolution into an unquantized tile.

## Revision, 2026-09-29

**The attribute is now computed by `-npu-calibrate` from the constant the
operation holds, not copied from the profile.** Everything else here stands:
the attribute, its three rules, and the lowering reading it and never a file.

The reason is the one this record's context gives for per channel weights at
all. At `-O2` the batch norm fold multiplies each output channel of a filter by
its own factor, and calibration is to run after the fold there, which the
owner ruled the same day, so the profile's entry, which describes the
initializer, describes weights the machine no longer has; on `conv_bn_relu_stack` it would clamp 32 of the 792 folded weights. Only
scales taken from the folded constant describe what the machine quantizes.

**The profile stops being the single source of truth for the weights and
becomes their oracle.** The rule is evaluated the way the observer evaluates
it, and at `-O0`, where nothing has touched a filter, the two agree bit for bit
on all 22 operations and 556 channels, which a test holds. The consequence
above, a divergence between the profile and the instruction being a red, holds
at `-O0` exactly as before, and it is the level at which the two describe the
same weights. The per tensor arm no longer needs the profile's absolute maxima
to tell a channel of zeros from a real one, because it reads the zeros.
