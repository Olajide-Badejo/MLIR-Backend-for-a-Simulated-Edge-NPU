<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 13. Hold cross layer equalization in reserve

- **Status:** Deferred, with the trigger stated below
- **Date:** 2026-09-22
- **Diataxis type:** explanation

## Context

Cross layer equalization exploits a property of a ReLU network: scaling one
layer's output channel down and the next layer's matching input channel up by
the same factor leaves the function unchanged. Choosing those factors to even
out the per channel weight ranges makes a tensor that per tensor quantization
handled badly into one it handles well [R16].

The structure it helps is the one this suite deliberately contains. Depthwise
convolutions have wildly different ranges per channel, and batch norm folding
is a documented cause of the same spread [R17], so the depthwise separable
block and the Conv plus BatchNorm plus ReLU stack are the two models where it
would show.

## Decision

**Not built now, and not rejected either.** It is held in reserve behind a
stated trigger, which is a different status from the two records beside this
one: those are refused on grounds that will not change without a dataset, and
this one is a question of whether it is still needed.

## Why not now

**Its published gain and per channel quantization's published gain land in the
same place**, and per channel is already built. Both address per channel range
spread, and the literature reports them as alternatives more than as a stack:
equalization exists largely to make *per tensor* quantization survive the
spread, and the straightforward answer to that spread is to scale per channel,
which the machine now does in its fourth operand [R16].

So building it now would be building a second solution to a problem the first
one may already have solved, and measuring it would be measuring the residue
of a fix rather than a gap. That is the wrong order: measure first.

**It is also a graph rewrite with a correctness obligation**, not a
calibration detail. It changes weights in two layers at once and relies on the
function being invariant under the pair of scalings, which holds for a
positively homogeneous activation and fails quietly where it does not. Taking
that obligation on before knowing whether it buys anything is the trade this
project keeps refusing elsewhere.

## The trigger

Build it when, and only when, **per layer SQNR still looks poor after per
channel weights have landed and been measured.** Concretely, the condition to
check at Checkpoint C, when the per channel against per tensor ablation runs:

- per channel quantization is in effect, and
- a layer of `depthwise_separable` or `conv_bn_relu_stack` still shows a
  signal to quantization noise ratio materially worse than its neighbours, or
  the model's accuracy budget is still missed with per channel on.

If per channel closes the gap on those two models, equalization has nothing
left to do on this suite and this record becomes a rejection with a
measurement behind it, which is a better record than either a rejection
without one or a feature nobody needed.

## Consequences

Checkpoint C's ablation is what settles this, so the ablation is not only a
number for the report: it is the decision procedure for this file. Whoever
runs it should come back here and either close this record with the
measurement or open the work with it.
