<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 11. Reject adaptive rounding

- **Status:** Rejected
- **Date:** 2026-09-22
- **Diataxis type:** explanation

## Context

Post training quantization rounds a weight to the nearest representable value.
Adaptive rounding asks whether nearest is the right choice per weight, and
shows that it often is not: rounding some weights away from nearest lowers the
task loss, because what matters is the error of the layer's *output* rather
than the error of each weight in isolation. The technique learns a per weight
rounding decision by optimising a continuous relaxation of it, and the
quantization white paper this project already cites reports it as one of the
larger post training gains available at low bit widths [R16].

Section 14 names it as a technique to consider and asks for the decision to be
recorded either way. This is that record.

## Decision

**Not built.** The quantized path rounds half to even, as Section 14's pinned
arithmetic states, and this project does not learn a rounding.

## Why

**It needs a training loop, and training is out of scope by Section 0.4.**
Adaptive rounding is not a closed form correction applied to a calibrated
range. It optimises a per weight parameter against a task loss over a set of
calibration images, on the order of a thousand of them, with an optimiser and
a schedule [R16]. That is a training loop by every property that makes a
training loop expensive to own: it has hyperparameters, it needs a loss, it
needs enough data to be meaningful, and its result is not reproducible from a
seed and a range the way this project's profiles are.

**This project has no dataset, and that is the harder half.** The suite is
seven generated models driven by seeded synthetic standard normal inputs. The
calibration methodology in `docs/PASSES.md` says plainly that these inputs
represent nothing but themselves. A technique whose entire benefit is measured
as task loss on real data cannot be evaluated here at all: running it against
synthetic draws would produce a number, and the number would be a measurement
of the draw rather than of the technique. Reporting such a figure would be
worse than not building it.

**What would be left is an unmeasurable feature.** This project's standard is
that a technique is built and then measured, which is exactly the standard
Section 14 applies to per channel weights: build it, then measure whether it
pays on this suite, and an ablation showing near zero on one model and a large
gain on another is a real result. Adaptive rounding cannot be held to that
standard here, so building it would add a code path with no evidence behind
it.

## What is built instead

Per output channel weight scales, which the same literature places in the same
range of benefit for 8 bit post training quantization and which **is**
measurable on this suite [R16][R17]. The suite contains a depthwise separable
block and a Conv plus BatchNorm plus ReLU stack, which are the structures per
tensor scaling handles worst, so the ablation has something to find. The
instruction carries the per channel rescale in its fourth operand and the
machine applies it.

## When to revisit

If this project ever acquires a real dataset and a task metric, the decision
changes shape, because the objection is the absence of both rather than the
technique. The trigger to state plainly: **a labelled evaluation set and an
accuracy metric on it**. Until then the honest position is that the technique
is known, understood, and not evaluable here.

## Consequences

The phase reports no adaptive rounding number, and `docs/NUMBERS.md` carries
none. A reviewer asking why finds this file rather than silence, which is what
Section 14 asks for: a cited refusal is stronger than an omission.
