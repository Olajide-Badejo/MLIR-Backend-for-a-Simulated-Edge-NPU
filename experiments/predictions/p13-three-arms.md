<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: which of Section 13.3's three arms wins, per model and per budget

- **id:** p13-three-arms
- **written:** 2026-09-06
- **result field:** `instruction_count`, `simulated_cycles`, `dram_bytes_total`, `spill_count`, `scratchpad_peak_bytes`, `placed`
- **direction:** tiling beats spilling on every model whose over budget operand
  is a function argument or a constant, and loses to spilling on every model
  whose over budget operand is an on chip producer; recompute changes nothing at
  any budget in the swept range
- **magnitude bracket:** the cycle difference between arm one and arm two is
  under 2000 cycles on every model at every swept budget; the crossover budget,
  where one exists, is within 512 bytes of the budget at which the untiled
  program first spills; and the two spill heuristics differ on at most one model
  at one budget
- **answered at:** P13, Section 13.3, in the commit that runs the three arms

*`placed` is the field that carries whether the allocator produced a program at
all, and it is in the list because Section 13.3 says a program an arm cannot
place is reported as exactly that rather than as a slower one. A comparison that
had nowhere to put "there is no program here" would have to put a number there.*

*Written and committed before the experiment script exists and before any arm
has been run at any swept budget. Ground rule 15. What this file may not do is
pretend to know less than the repository already records: `docs/NUMBERS.md` and
D-0056 carry the two tight budget cells where tiling already fires, and those
are inputs to the reasoning below rather than predictions. **What is predicted
is the swept range, the two heuristics, the halo arm and the crossover**, none of
which has been measured.*

## Hypothesis

### 0. What is already measured, and is therefore not predicted here

At the two published budgets, with `-O2` as it stands, `docs/NUMBERS.md` records
`inception_block` at its tight budget going from 3799.0 cycles with three spills
and 21936 DRAM bytes to 3395.0 with none and 12720, and `resnet_block` going
from 17 instructions and 2018.0 cycles to 21 and 2660.0. Those two numbers are
the reason the mechanism below is stated as confidently as it is. They are not
what this entry is answered against.

### 1. The mechanism that decides which arm wins, stated once

**Under the per slice convention a slice of a DRAM value becomes a transfer and
a slice of a scratchpad value becomes a view.** So tiling relieves an operand
that lives off chip, which is a function argument or an `npuisa.const`, and does
not relieve an on chip producer, which stays whole resident as the base the tile
views are taken of.

That gives a prediction per model, from which operand is over budget:

| Model | The operand the budget runs out on | Arm predicted to win | Why |
|---|---|---|---|
| `conv_bn_relu_stack` | a function argument | **tile** | the slices become transfers and the whole value never has to be resident |
| `lenet` | a matmul weight, an `npuisa.const` | **tile** | same, and the weight is the largest single value in the program |
| `lenet_batched` | the same weight at batch 4 | **tile** | same |
| `depthwise_separable` | a function argument | **tile** | same, and only with `-npu-fuse-ops` ablated, because fusion hides both compute operations |
| `dilated_stack` | a function argument | **tile** | same, and only with fusion ablated |
| `inception_block` | one argument read by three branches | **tile** | tiling relieves the argument and the three spills go with it |
| `resnet_block` | the block's own residual, on chip | **spill** | tiling adds the tile buffers on top of a residency it cannot remove, so the peak does not fall and the transfers are paid for nothing |

**`resnet_block` is the one model where I predict tiling loses**, and it is the
case worth the entry: the pass fires, the program is correct, the peak does not
move, and the cost is four instructions and 642 cycles of extra transfer. An
experiment that reported only the wins would be reporting the mechanism
backwards.

### 2. Arm one, spill, under both heuristics

`longest-range` takes the longest live range crossing the pressure peak;
`cost` takes the smallest `bytes * (1 + reloads)`. **I predict the two choose the
same victim on every model at every swept budget except at most one**, because
the candidate set at the peak is three to five buffers in these programs and the
buffer with the longest range across the peak is usually also the largest, so the
two keys agree.

Where they differ, I predict `cost` wins on cycles, because its rule is the one
that counts reloads and a reload is a transfer on the port that decides the
makespan.

### 3. Arm two, tile, in two configurations

At `-O2` fusion hides 30 of the 44 convolutions and matrix multiplications, and
two models have none left visible. So the arm is reported with `-npu-fuse-ops`
present and ablated, and **I predict the ablated configuration tiles on all seven
models at their tight budgets and the present one on two**, which is the fourteen
against forty four the pass already counts.

**The gap between the two configurations is the measured cost of the fusion and
tiling conflict**, and I predict it is the larger of the two effects this arm
reports: bigger than the difference between the spill heuristics, and bigger than
anything arm three does.

### 4. Arm three, recompute, changes nothing in the swept range

The halo boolean only bites when the tiling splits a **spatial** axis under a
window wider than one. The search prefers output channel splits, which have no
halo at all, and `test/Transforms/tile-to-scratchpad.mlir` records the shape of
the boundary: at a 2048 byte budget `halo=cache` tiles four ways, and it takes
768 bytes before `halo=cache` declines and `halo=recompute` fits in 720.

**So I predict arm three is a no operation at every budget in the swept range**,
6000 to 6464 on `resnet_block` and `conv_bn_relu_stack` and 4000 to 6000 on
`inception_block`, and that it becomes visible only below 2048 bytes, which is
below every budget this experiment sweeps. If that holds, the honest report is
that arm three has no subject in this suite and the reason is the search's
preference for the channel axis, not that recompute is worthless.

### 5. The crossover, per model

**Tiling overtakes spilling at the budget where the untiled program first has to
spill**, and not before, because above that budget the untiled program pays
nothing and tiling pays transfers. Below it, spilling pays a `dma_store` plus one
`dma_load` per later reader while tiling pays one transfer per slice, and on the
models where tiling relieves a DRAM operand the second is the smaller bill.

I predict the crossover is within 512 bytes of the first spilling budget on
`conv_bn_relu_stack` and `inception_block`, and that **`resnet_block` has no
crossover at any budget at which its program places**.

### 6. A program an arm cannot place

At the bottom of the swept range I expect at least one model and one arm to stop
placing altogether. That is reported as not placing, with the allocator's own
message, and never as a large cycle count. **I predict this happens to arm one
before it happens to arm two**, on the models where tiling relieves the DRAM
operand, because spilling cannot reduce a peak that one operation's working set
sets on its own.

## What would falsify it

- **Tiling losing on `conv_bn_relu_stack`, `lenet`, `lenet_batched`,
  `depthwise_separable`, `dilated_stack` or `inception_block` at a budget where
  both arms place** falsifies the mechanism in clause 1, and would mean the DRAM
  and scratchpad distinction is not what decides the trade.
- **Tiling winning on `resnet_block` at any swept budget** falsifies the other
  half of the same clause.
- **A cycle difference above 2000 between arm one and arm two on any model at any
  swept budget** falsifies the bracket.
- **The two spill heuristics differing on two or more models** falsifies clause 2.
- **`halo=recompute` changing any counted field at any budget in the swept
  range** falsifies clause 4, and would be the more interesting outcome of the
  two: it would mean the search takes a spatial split somewhere I expect a
  channel split.
- **A crossover further than 512 bytes from the first spilling budget** falsifies
  clause 5.
- **Arm two failing to place where arm one places, on a model whose over budget
  operand is in DRAM**, falsifies clause 6.

**What would not falsify it:** arm three doing nothing. That is the predicted
outcome, and every clause above is an observation this experiment makes rather
than a judgement it defers.

## What each outcome would mean

- **The mechanism holds on six models and fails on `resnet_block` alone.** The
  per slice convention is the right explanation and Section 13.3 has an answer
  with a rule in it rather than a table.
- **The mechanism fails on a second model.** The convention is not sufficient and
  the next question is what else is resident across the tiled operation, which is
  the follow up D-0056 already names: a tiling pass that consults the allocator.
- **Arm three is not a no operation.** The search is taking spatial splits
  somewhere, and the halo cost becomes a third axis of the experiment rather than
  a boundary case.
- **No crossover anywhere.** Spilling is the better arm on this machine at every
  budget a program places at, and the tiling pass is a correctness feature for
  budgets below the spilling floor rather than a performance one.

## What is deliberately not predicted

The absolute cycle counts at the swept budgets. They depend on which mapping the
exhaustive search picks at each budget, and the search's scoring is the two port
makespan over a space this file has not enumerated. Predicting a number there
would be inventing one.
