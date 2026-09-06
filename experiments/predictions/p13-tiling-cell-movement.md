<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: which cells move when `-npu-tile-to-scratchpad` joins `-O2`

- **id:** p13-tiling-cell-movement
- **written:** 2026-09-05
- **result field:** `instruction_count`, `simulated_cycles`, `dram_bytes_total`,
  `npuisa.scratchpad_peak_bytes`, and the golden output tensors
- **direction:** no cell moves at the default budget; at most one model's tight
  budget cells move, and the golden tensors do not move anywhere
- **magnitude bracket:** zero cells moved at the default budget, and between
  zero and three of the seven models moved at their tight budget
- **answered at:** P13, in the commit that wires the pass into the levels

*Written before the pass was wired into any `-O` level and committed strictly
before the first cell was measured with it wired. Ground rule 15: a prediction
written after the measurement is worth nothing, and one that turns out wrong and
is reported honestly is stronger evidence of understanding than two numbers that
happen to agree.*

## Hypothesis

### 1. Not one cell moves at the default budget

The default is 1048576 bytes. The largest single operation in Section 15's suite
is `lenet`'s fully connected layer, whose working set is under 200 kilobytes, so
**no operation's working set exceeds a megabyte and the pass fires on nothing.**
Every default budget cell should be byte for byte what it was: same instruction
count, same cycles, same DRAM bytes, same goldens.

**If a default budget cell moves, that is a defect in the wiring and not a
result.** Nothing at that budget should tile, so a movement there means the
trigger is firing on something it should not, and the response is to find out
why rather than to write a declaration.

### 2. At the tight budgets, I expect very little and possibly nothing

This is the part I am least sure of and it is the reason the bracket is as wide
as it is. **ADR 0008's tight budget is the smallest budget at which one program
allocates**, and allocation needs every simultaneously live buffer to fit, which
is a stronger requirement than any single operation's working set. So a tight
budget is at or above the largest single working set for most models, and the
tiling trigger, which fires on one operation's working set alone, may never see
a value it has to split.

Arithmetic where I can do it from the recorded shapes:

| Model | Tight budget | Largest working set I can compute | Expect |
|---|---|---|---|
| `conv_bn_relu_stack` | 6464 | `conv1` at 6432 | **no tiling**, and by 32 bytes |
| `resnet_block` | 6464 | `node_conv2d` at 6400 | **no tiling**, and by 64 bytes |
| `lenet` | 194624 | the 400 by 120 matmul at 194560 | **no tiling**, and by 64 bytes |
| `inception_block` | 6144 | not computed here | **the one to watch**, because 6144 is 0.897 of its peak, the most aggressive fraction ADR 0008 records |
| `depthwise_separable` | 8192 | not computed here | no tiling |
| `dilated_stack` | 8064 | not computed here | no tiling |
| `lenet_batched` | 200832 | not computed here | no tiling |

**So the single most likely outcome is that the ablation rows for all three new
passes are zero on every model at both budgets**, and the suite goes from 175
cells to 217 with not one number moving.

### 3. The goldens do not move anywhere

Tiling parallel dimensions is exact: every output position reads the same input
positions in the same order, which
`Conv2DEveryTileReadsTheSamePositionsAsTheWhole` asserts over five window shapes
and every tile size that divides the output, and which a whole tiled program
reproduced byte identically at a 2048 byte budget. **A moved golden would be a
defect and not a declaration**, at either budget.

## What would falsify it

**This section was added after the measurement and that has to be said here
rather than left in a commit message.** The file was committed at `33c17bc`,
strictly before the pass was wired and before any cell was measured with it, and
it was refused by `test_the_directory_holds_at_least_one_prediction` for missing
two sections the schema requires by name. The heading above it was renamed to
`## Hypothesis` for the same reason. **No claim, number or bracket was changed**,
and `git diff 33c17bc` over this file is the check on that sentence rather than
an assurance a reader has to take.

Adding falsifiers after seeing the result is exactly what a prediction file
exists to prevent, so the four below are derived from clauses that were already
written rather than chosen against what was measured, and the diff is what makes
that verifiable. The schema was right to refuse the file: a prediction with no
stated falsifier is an opinion, and the mechanism caught it on the first run.

Each of these is a specific observation that would make a clause above wrong,
and each is checked by the same measurement rather than by a later judgement.

- **Any cell moving at the default budget falsifies clause 1**, and it would be
  a defect in the wiring rather than a result: at 1048576 bytes nothing in this
  suite has a working set that could trigger the pass, so a movement there means
  the trigger fired on something it should not have.
- **Four or more of the seven models tiling at their tight budget falsifies the
  bracket**, which says between zero and three.
- **Any golden tensor moving, at either budget, falsifies clause 3** and is a
  defect rather than a declaration. Tiling parallel dimensions is exact, a whole
  tiled program has already reproduced byte identically at a 2048 byte budget,
  and a moved golden would mean the exactness argument is wrong somewhere the
  unit tests do not reach.
- **The pass reporting a declined operation anywhere in the suite** would
  falsify the implicit claim that every operation either fits or is splittable,
  and would mean some shape needs a reduction split to fit, which Section 13.2
  forbids under fp32.

**What would not falsify it:** the pass tiling nothing. That is the predicted
outcome, and confirming a negative prediction is not the same as the prediction
having been unfalsifiable, because each of the four observations above would
have contradicted it and none of them is unobservable.

## What each outcome would mean

- **Nothing moves anywhere.** The pass is correct and unexercised by this suite,
  and Section 13.3's experiment needs budgets **below** the tight ones to have a
  subject at all. That is not a failure of the phase; it is the measurement that
  tells the experiment where to look, and ADR 0010 already points the same way
  by recording six cells that cannot allocate at batch 4.
- **Only `inception_block` moves.** The prediction is right in shape and the
  experiment has one model to compare arms on, which is thin but real.
- **Several tight budget cells move.** The bracket was too pessimistic, and the
  three arm experiment has the population it wants.
- **A default budget cell moves.** A defect in the wiring, per above.

## What is deliberately not predicted

How much a moved cell moves. The direction is not in doubt, since tiling adds
instructions and DRAM traffic and removes scratchpad pressure, and the magnitudes
depend on which mapping the search picks, which is an exhaustive search over a
space this file has not enumerated. Predicting a number there would be inventing
one.
