<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Breaking changes

*Diataxis type: reference.*

This file records every **deliberate** regression of the recorded baseline, and
it records each one **before** the commit that causes it.

The prime directive of the build specification is that once a behaviour is in
the baseline it does not change silently, and that if it must change I say so in
writing first. This file is where that writing goes. From Phase P8 the
repository carries a baseline of test names and counts, instruction counts,
simulated cycles, DRAM bytes and golden output tensors, and every phase gate
re-runs it. A gate that finds the baseline moved fails, and the only thing that
turns that failure into a pass is an entry here that predicted the movement.

The ordering is the whole mechanism. An entry written after the number moved is
an explanation; an entry written before it moved is a decision. Git commit order
is what tells the two apart, so the entry lands in its own commit, strictly
before the commit that changes the behaviour.

Two things do not belong here. A number that moved by accident is a defect and
goes in `DEFECT_LOG.md` until it is understood. A user visible change that does
not regress the baseline is a changelog line and goes in `CHANGELOG.md`. Some
changes are both, and then they are written in both places rather than in
whichever one was closer to hand.

A baseline field that did not exist yet cannot have regressed. The baseline
grows across phases with a `schema_version` bump each time, and the arrival of a
new field is not a breaking change.

## Entry form

Each entry names the date, the phase, which baseline fields move and in which
direction, roughly how far, why the regression is worth taking, and the commit
that causes it once it exists.

## Entries

### 2026-09-01, Phase P9: `-O1` and `-O2` arrive, and `-O2` moves the last bits

**Written before the commit that causes it.** The commit that registers the two
levels is the next one; this entry is what makes it a decision rather than an
explanation.

**What moves, and what does not.**

- **No `-O0` cell moves.** Not one instruction count, cycle count, DRAM byte
  count or golden tensor. `-O0`'s pipeline is unchanged and the two lowering
  fixes that landed ahead of this entry, D-0034 and D-0035, are both no
  operations on the IR `-O0` produces. That is stated first because it is the
  claim a reader most needs and the one easiest to lose in a phase that adds
  forty two cells where there were fourteen.
- **`-O1` matches `-O0` exactly on every model in the suite**, in every field
  including the goldens. `-O1` is constant folding and canonicalization, and no
  model in Section 15's suite has a constant subgraph to fold or an operation
  nothing reads. That is a measurement rather than a disappointment: the passes
  are proven to work by the dead subgraph injection, which is a graph
  constructed to have something for them to remove.
- **`-O2` moves numbers on one model.** `conv_bn_relu_stack` is the only model in
  the suite that carries an unfolded batch norm, `-npu-fold-batchnorm` folds
  both of them, and the answer moves.

**The largest observed movement, and the mechanism.**

Measured over all forty two cells on 2026-09-01. The two that move are
`conv_bn_relu_stack` at `-O2`, and they move identically at both budgets because
the fold happens above the allocator:

| Field | at `-O0` | at `-O2` |
|---|---|---|
| `max_abs_movement_vs_o0` | 0 | **4.47e-08** |
| `instructions` | 23 | 15 |
| `cycles` | 1372.50 | 1160.50 |
| `dram_bytes_read` | 4256 | 4128 |

**4.47e-08 is the largest movement at any level, on any model, at either
budget**, and it is within the 1e-6 band Section 17.6 sets for this phase. Every
other cell of the forty two moves by exactly zero, and every `-O0` cell matches
the P8 baseline field for field.

**The mechanism, named rather than attributed to "fusion".** Before the fold the
machine computes a convolution and then scales its result per channel; after it
the machine convolves with pre scaled weights, so every product in the reduction
is scaled instead of the sum being scaled once at the end. The two are equal in
exact arithmetic and differ in the last bits of `f32`. The pass computes its
constants as `invStd = 1 / sqrt(variance + epsilon)`, `scale = gamma * invStd`,
`shift = beta - mean * scale`, and that order is written down in `docs/PASSES.md`
because it is observable.

**The other two `-O2` passes were measured and move nothing**, which is why the
attribution above is to one pass rather than to the level.
`-npu-fuse-bias` is bit exact because the simulator's convolution kernel adds
the bias to the same `f32` accumulator the unfused program stores and then adds
to. `-npu-fuse-ops` is bit exact because `-npu-lower-to-npuisa` flattens the
region into the instruction stream the unfused chain produced.
`test/Python/test_transform_passes.py::test_fold_batchnorm_is_the_only_pass_that_moves_a_number`
runs each of the eight ablatable passes alone and asserts it.

**Why the regression is worth taking.** Folding a batch norm into the
convolution before it is the single largest structural saving available at this
phase: it removes eight of `conv_bn_relu_stack`'s twenty three instructions and
15 percent of its cycles. Refusing it to keep a bit would be refusing the phase.

**Two changes to `-npu-lower-to-npuisa` landed ahead of this entry rather than
inside it**, and they belong in this record because both were found by running
the levels and both would otherwise look like unexplained movements to a later
reader. Neither moves a number: both are no operations on the IR `-O0`
produces, which is why they are not declarations. D-0034 gives two operations
that share a destination two buffers, which `-cse` made reachable in one step.
D-0035 puts a constant's transfer, and a destination's allocation, where the
data is used rather than where `-canonicalize` and `-npu-fuse-ops` had hoisted
them; without it `-O1` was 37 percent slower than `-O0` on LeNet and `-O2` could
not place LeNet's tight budget cell at all. `docs/DEFECT_LOG.md` carries both.

**What is not a regression, and is here so a reader does not go looking.** The
baseline's suite counts move a great deal at P9, because the test suite grew:
`check-npu` from 20 to 25, pytest from 495 to well over eight hundred, with the
end to end matrix multiplied by three by the level axis. The suite's composition
changing is not the numbers changing, and Section 17.6 draws that distinction in
the two halves of the baseline file. The `schema_version` bump from 1 to 2 is
likewise not a regression: a field that did not exist cannot have moved.

**The causing commit** is `feat(pipeline): -O1 and -O2, and the two exemptions
-npu-fuse-ops closes`. The baseline is re-recorded two commits later, in a commit
that touches `test/baseline/` and nothing else.
