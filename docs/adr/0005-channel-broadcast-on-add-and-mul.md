<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 5. `npu.add` and `npu.mul` take a rank 1 channel operand, and the importer normalises to it

- **Status:** Accepted
- **Date:** 2026-08-19
- **Diataxis type:** explanation

## Context

Section 11 states the frontend's broadcasting policy in three parts. Identical
shapes pass through. An initializer that broadcasts against the other operand
is expanded into a same shaped constant at import time. And then one carve out,
which the specification calls load bearing:

> A rank 1 initializer of length `C` broadcasting against a rank 4 activation
> over the channel axis is left **unexpanded**, as a rank 1 constant, with the
> broadcast recorded on the consuming operation.

The reason given is `-npu-fuse-bias`, which Section 12 defines as folding
`add(conv(x, w), b)` into the convolution's bias operand and which guards on a
channel shaped constant addend. Expand that addend into a full `N x C x H x W`
constant and the guard never matches on any model in the suite, the pass's
ablation row is a row of zeros, and, in the specification's words, the phase
looks done while doing nothing. Section 15 then puts the same carve out on a
second operator: the small ResNet block carries a rank 1 per channel scale on
the residual branch, which exports as an ONNX `Mul`.

P1 read that carve out differently, and wrote the reading into `NPUOps.td`:
`npu.add` and `npu.mul` require both operands to have the result shape exactly,
with the comment that the carve out "is expressed as a bias operand on the
consuming convolution rather than as a broadcasting add". That is a defensible
reading of "recorded on the consuming operation", and its stated reason is good
in itself: one fact should not have two representations.

Building the frontend against it showed that it does not survive contact with
either of the two cases the specification names.

**It defeats its own purpose on the `Add` case.** If the importer folds a
`Conv` followed by a rank 1 `Add` straight into the convolution's bias operand,
then `-npu-fuse-bias` has nothing left to fuse. The pass would be structurally
unfireable on every model in the suite, which is precisely the outcome Section
11 introduced the carve out to prevent. The constant does stay rank 1, so the
DRAM traffic argument is satisfied, but the ablation argument, which is the one
the specification leads with, is not.

**It cannot express the `Mul` case at all.** A per channel scale has no bias
operand to be recorded on. There is no operand of any `npu` operation that a
rank 1 scale on a residual branch can be folded into. Under P1's rule the only
remaining option is to expand it to `1 x C x 1 x 1` or to the full activation
shape, which is exactly what Section 11 says to leave alone, and which inflates
one constant per residual block by the spatial extent.

So the two operands of this decision are: a merged dialect rule with a written
justification, and two specification requirements that rule makes
unimplementable. The dialect is the thing that moves.

There is also a structural argument that this is a smaller change than it
looks. `npu.batch_norm` already takes four rank 1 parameter tensors of length C
against a rank 4 input, and P4 already has to decompose it into a multiply and
an add over the channel axis. Per channel arithmetic against a rank 4
activation is therefore not a new concept in this dialect; it is the concept
`npu.batch_norm` is built out of, and this record only gives it a second
spelling that the importer can reach.

## Decision

**Relax `npu.add` and `npu.mul` to accept a rank 1 right hand operand whose
length equals the result's channel extent, and nothing else.** Precisely:

- The left hand operand always has the result shape exactly.
- The right hand operand has the result shape exactly, **or** the result is
  rank 4 and the right hand operand is rank 1 of length equal to the result's
  channel extent under the result's layout.
- Every other shape combination is refused by the verifier, with a message
  naming which operand and quoting both types.

The asymmetry is deliberate and is the answer to P1's objection. A rank 1
operand may only ever be the right hand one, so `add(bias, conv)` and
`add(conv, bias)` do not both exist as legal IR: there is exactly one spelling
of a channel broadcast, and `-npu-fuse-bias` therefore has one form to match
rather than two. **The importer is what guarantees that.** Both operations are
commutative, so when the rank 1 operand arrives on the left the importer
commutes the node and the commutation is not observable in the result.

The channel extent is read through `getChannelExtent`, which reads the layout
encoding, so the rule stays correct when `-npu-assign-layout` moves a tensor to
NHWC and the channel axis moves from 1 to 3.

## Consequences

**The importer implements the carve out as Section 11 wrote it.** A `Conv`
followed by a rank 1 `Add` imports to `npu.add` with a rank 1 addend, and
`-npu-fuse-bias` at P6 has the form it is specified to match. A per channel
`Mul` imports to `npu.mul` with a rank 1 rhs. Neither constant is expanded, so
neither inflates the byte counts the report publishes.

**The carve out is defined by what an initializer broadcasts as, not by the
literal rank it was stored with, and the importer normalises to rank 1.** This
matters because it is not optional: the dynamo exporter emits a per channel
scale written in PyTorch as `scale.reshape(1, -1, 1, 1)` as an initializer of
dims `[1, 8, 1, 1]`, not `[8]`, and a separate bias add comes out the same way.
An importer that matched only a literally rank 1 initializer would expand every
one of them and the carve out would never fire on a single model in the suite.
So the importer accepts an initializer of shape `(C,)`, `(1, C, 1, 1)` or
`(C, 1, 1)` against a rank 4 activation of channel extent C, and emits it as a
rank 1 `npu.constant` of length C. Shape `(1, C, 1, 1)` with C equal to 1 is
also a scalar broadcast; the tie is broken towards the carve out, which is
harmless because the two produce identical values and the rank 1 form is the
one the fusion pass matches.

**P4 gains a lowering obligation and P7 gains a kernel obligation**, both
already implied by `npu.batch_norm`: an add or a multiply whose rhs is rank 1
lowers to a per channel broadcast, and the simulator kernel reads the addend
with a channel stride of one and a spatial stride of zero. Named here so that
neither phase discovers it from a verifier failure.

**What this does not do.** It does not make `npu.add` a general broadcasting
operation. Rank 1 against rank 2, a rank 1 lhs, a rank 1 operand whose length
is not the channel extent, and every numpy style broadcast are all still
refused, by the verifier and separately by the importer, which expands what it
can and names the node when it cannot. The negative cases are in
`test/Dialect/NPU/invalid.mlir` because a relaxation with only positive tests is
a relaxation nobody has measured the edge of.

Recorded as defect D-0012, because the dialect rule and the frontend
requirement disagreed in the tree for the length of two phases and a reader
comparing `NPUOps.td` at P1 against Section 11 would have found the conflict
sitting there.
