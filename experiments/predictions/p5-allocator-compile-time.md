<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: the allocator's compile time curve and its fragmentation ratios

- **id:** p5-allocator-compile-time
- **written:** 2026-08-20
- **result field:** `fragmentation_ratio`, `spill_count`
- **direction:** the offset assignment's quadratic scan dominates the compile time curve, and `pack` comes out at or below `interval` on every model
- **magnitude bracket:** the exponent between consecutive sizes is between 1.5 and 2.0, the pass takes under 100 milliseconds at 5000 operations, and `pack` stays at or below a ratio of 1.05 on every model
- **answered at:** P5

*The header block above was added at P10, when Section 17.8's mechanism landed
and gave this directory a machine parsed format. **Nothing below it has been
edited.** A prediction rewritten after the measurement is not one, and this
project's rule for this directory is that an entry that turned out wrong stays as
it was written. The block restates in a checkable shape what the prose below
already says, so that this entry is held to the same five requirements as every
entry written since.*

*This entry predates the result schema, so no result file names it. Its numbers
were answered in `docs/ENGINEERING_LOG.md` at P5 rather than in
`experiments/results/`, which is why `prediction_id` is null in every cell for
it rather than absent by oversight.*

## Hypothesis

*Written 2026-08-20, before either measurement was taken. Ground rule 15: the
commit order is the evidence, and a prediction written after the measurement is
worth nothing. A prediction that turns out wrong and is reported honestly is
stronger evidence of understanding than two numbers that happen to agree, so
what follows is what I actually expect rather than what would look best.*

Two experiments land at P5 and both produce numbers the report will interpret,
so both are predicted here, in one file, committed strictly before the commit
that records the first number.

### 1. The compile time curve

`experiments/compile_time_benchmark.py` measures `-npu-allocate-scratchpad` on a
synthetic straight line function at 500, 1000, 2000 and 5000 operations, and
reports the pass's own wall time out of `--mlir-timing`.

**What I expect, and why.** The pass has three parts with different growth.
Liveness walks the block once per buffer and follows each buffer's users, which
is linear in the total number of uses. The sweep line is one sort and one walk,
so O(n log n). Offset assignment is the part I expect to dominate: for each
buffer it scans **every already placed buffer** to find the ones whose live
ranges overlap, which is quadratic in the buffer count as it is written. There
is no interval tree in there and I did not put one in, because Section 13.1 asks
for the sweep line and says nothing about the placement's data structure, and an
unmeasured optimization is how a phase turns into two.

So:

- **The exponent between consecutive sizes is between 1.5 and 2.0**, rising
  across the curve, and closer to 2.0 between 2000 and 5000 than between 500 and
  1000. Below 1.3 would mean the quadratic scan is not dominating and I have
  mismodelled the constant factors. Above 2.1 would mean something is
  superquadratic and would be a defect worth a root cause, because nothing in
  the pass should be.
- **The pass takes under 100 milliseconds at 5000 operations** on this machine,
  and under 2 milliseconds at 500. If 5000 takes more than a second, the
  quadratic term has a constant factor large enough to matter for real models
  and the placement needs the interval tree after all.
- **The total wall time is dominated by parsing at every size**, and the ratio
  of total to pass time falls as the size grows.
- **No spilling happens at any size.** The chain's peak is two buffers of 256
  bytes and the default budget is a mebibyte, so the spill loop runs zero
  rounds. If it runs any, the benchmark is measuring the wrong thing.

### 2. The fragmentation ratio per model

`experiments/allocator_fragmentation.py` compiles each of the seven suite models
through `-npu-lower-to-npuisa` and `-npu-allocate-scratchpad` under both offset
assignment strategies at the default budget, and reads
`npuisa.scratchpad_bytes` divided by `npuisa.scratchpad_peak_bytes` off the
function.

**What I expect.** The ratio is 1.0 exactly when the placement achieves the
sweep line peak, which is the lower bound. A pure chain of operations, where
each buffer dies as the next is born, has no fragmentation to suffer under
either strategy: whichever order you place them in, at most two are ever live.

- **`pack` is less than or equal to `interval` on every model.** Greedy by size
  is not optimal in general, and a program where the interval scheme wins is
  constructible, but I do not expect one in this suite. If `interval` wins on
  any model that is a finding and goes in the engineering log rather than being
  smoothed over.
- **`lenet` and `lenet_batched` come out at exactly 1.00 under both**, because
  they are chains.
- **At least one of `resnet_block`, `inception_block` and `dilated_stack` shows
  `interval` at 1.10 or worse**, because all three have a value that stays live
  across a branch or a concatenation while other buffers are born and die around
  it, which is the shape that leaves a hole in definition order.
- **`pack` stays at or below 1.05 on every model.** Above that would mean the
  greedy by size algorithm is leaving real holes on real programs, which would
  be worth reporting because it is the algorithm TFLite Micro ships.
- **The absolute `npuisa.scratchpad_bytes` is under 1 MiB for every model**, so
  no model spills at the default budget and every ratio is a placement result
  rather than a spilling result. `lenet` I expect in the low hundreds of
  kilobytes.

## What would falsify it

*Added at P10 as a heading over what the prose already said, for the same reason
the header block was: Section 17.8 requires every entry to name what observation
would falsify it, and every bullet above carries one. Collected here so the
parser can find them under the heading the format asks for.*

An exponent below 1.3 or above 2.1 between consecutive sizes; the pass taking
more than a second at 5000 operations; the spill loop running at all; `interval`
beating `pack` on any model; `lenet` or `lenet_batched` coming out above 1.00;
none of `resnet_block`, `inception_block` or `dilated_stack` reaching 1.10 under
`interval`; `pack` above 1.05 anywhere; or `npuisa.scratchpad_bytes` reaching a
mebibyte on any model.

## How this gets checked

The measured tables go in `docs/ENGINEERING_LOG.md` in a later commit, with the
predictions above quoted beside them and each one marked as met or not. A
prediction that was wrong stays in this file unedited.
