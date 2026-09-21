<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 12. Reject empirical bias correction

- **Status:** Rejected
- **Date:** 2026-09-22
- **Diataxis type:** explanation

## Context

Quantizing a weight tensor perturbs it, and the perturbation is not zero mean
in general. When it is not, the layer's output acquires a systematic shift
rather than only noise, and the shift accumulates through depth. Bias
correction removes it by measuring the mean output error and folding the
negative of it into the layer's bias, which costs nothing at inference time
because the bias is already there.

Two ways to get the correction are described in the literature this project
cites [R16]. The empirical one runs the float model and the quantized model
over data and takes the difference of the means. The analytic one estimates
the same quantity from the weight perturbation and an assumption about the
input distribution, typically that it is a zero mean Gaussian after a batch
norm.

Section 14 names bias correction as a technique to consider and asks for the
decision to be recorded. This is that record.

## Decision

**Not built**, in either form. The int32 bias carries what Section 14 puts in
it: the quantized bias itself, and the folded input zero point term
`- zp_x * sum_k q_w[k]`. It carries no correction term.

## Why the empirical form is out

**It needs paired activations over a real dataset.** The measurement is the
mean difference between the float and the quantized model's output at each
layer, over data drawn from the distribution the model will see [R16]. This
suite has no such data. Its inputs are seeded synthetic standard normal draws,
stated as a limitation in `docs/PASSES.md`, and a bias correction computed
against them would correct the model for a distribution nobody will run it on.

That is worse than it sounds, because the correction is **applied** rather
than merely reported. A wrong accuracy number is a wrong number; a bias folded
into every layer from the wrong distribution is a model that has been changed
on the strength of it, and every later measurement inherits the change. This
project's standard is to measure before changing, and here the measurement is
not available.

## Why the analytic form is also out, which is the less obvious half

The analytic form needs no data, so the argument above does not reach it. It
is rejected for a different reason: **its assumption is not satisfied here and
this project would not be able to tell.** The estimate rests on the input to
each layer being approximately zero mean Gaussian, which holds after a batch
norm and holds poorly after a ReLU, and the correction it produces is only as
good as that assumption. Checking whether it holds on a given model needs the
activation statistics the empirical form needs, so the analytic form is
data free at the point of use and not data free at the point where anyone
could decide it was working.

There is also a smaller, concrete objection. This project's `-O2` pipeline
folds batch norm into the preceding convolution, so by the time anything is
quantized the batch norms are gone and the very structure the assumption leans
on has been optimised away [R17].

## What is built instead

Two things that address the same failure without needing data. **Per output
channel weight scales**, which remove most of the perturbation that bias
correction would otherwise be correcting for, and which this suite can measure
because the depthwise separable block and the Conv plus BatchNorm plus ReLU
stack are exactly the structures that expose it [R16][R17]. And **the affine
activation quantization of Section 14**, whose range is extended to include
real zero, so the zero point represents real zero exactly and a padded
convolution is not biased by its own padding.

## When to revisit

The same trigger as the previous record: **a labelled evaluation set and an
accuracy metric on it** would make the empirical form both computable and
checkable, and would also make the analytic form's assumption testable. The
objection is the absence of data rather than the technique.

## Consequences

The quantized bias is exactly what Section 14 says it is, which keeps the
kernel, the numpy reference and the encoder agreeing about one definition. A
reviewer asking about bias correction finds this file.
