<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 7. The dataflow is weight stationary, and it is pinned

- **Status:** Accepted
- **Date:** 2026-08-31
- **Diataxis type:** explanation

## Context

Section 5.5 opens the cost model with a sentence that is easy to read past: a 16
by 16 array is not a cost model until the dataflow is named. The array size on
its own says how many multipliers there are. It says nothing about which operand
stays resident, which streams, what a tile boundary costs, or what the pipeline
fill is, and every one of those is a term in the charge the simulator applies.
A cost model that quotes 256 MACs per cycle and leaves the dataflow unstated is
quoting a peak it cannot justify spending.

The taxonomy this sits inside is Chen and others' [R11][R12] and the survey that
generalised it [R13]: output stationary, weight stationary, row stationary, no
local reuse. Two of the four are candidates for a machine of this shape.

**Weight stationary** holds a tile of the weight matrix in the array and streams
activation rows past it. The reuse is on the weights: one weight tile is loaded
and then multiplied against every streamed row. The cost consequences are two,
and both are things the simulator can see from an instruction's shapes alone.
The array is occupied in proportion to how much of it the tile covers, which is
the spatial `utilization` term. And the fill of the weight pipeline is amortised
over the streamed rows, which is the temporal `delta` term.

**Row stationary**, Eyeriss's contribution, maps a row of the convolution onto a
row of processing elements and reuses filter rows, activation rows and partial
sums together. It is the better dataflow by the energy measurements in the paper
that introduced it, which is exactly why it is tempting to claim.

## Decision

**The dataflow is weight stationary, and it is an assumption rather than a
measurement.** `include/NPU/Simulator/CostModel.h` is its one home, and the two
terms above are written into `gemmCharge`:

    utilization = (rows / kArrayDim) * (columns / kArrayDim)
    delta       = m / (m + kWeightPreloadCycles)
    cycles      = tileMacs / (utilization * delta * peak)

`kWeightPreloadCycles` is 16, which is the array's depth, and it is the whole
content of the temporal term: under a weight stationary dataflow a weight tile
is pushed into the array before any activation row can be streamed against it,
and the push is as deep as the array.

**Row stationary is rejected, and the rejection is stated rather than left as an
unexplained absence.** Row stationary requires a register file per processing
element and a two dimensional interconnect between them [R11][R12]. This machine
has one flat scratchpad and no per element storage at all, which is a fact about
the machine Section 5.1 describes rather than a simplification of it. Claiming a
dataflow the hardware cannot express would make every cycle count this project
publishes a number about a machine that does not exist. The report says the same
thing in the same words, because a rejection recorded only in a decision record
is a rejection the reader of the report never sees.

**It matches where the project is going.** Phase P18's Gemmini target [R14] is
configured weight stationary, so the pinned assumption and the eventual hardware
comparison are the same assumption, and a divergence measured there is a
divergence about the model rather than about which dataflow each side assumed.

## Consequences

**The depthwise case is the one that shows the pin doing work.** With
`group == C` each group presents a single column to a sixteen column array, so
the spatial term is a small fraction of one and the cycle charge is far above
what the MAC count alone suggests. `Convolution.DepthwiseConvolution` in
`unittests/Simulator/SimulatorTest.cpp` asserts exactly that: utilization below
0.01 and `effectiveMacs` strictly above `macs`. Under a dataflow that reused
along a different axis the same layer would be charged differently, which is the
sense in which naming the dataflow is what makes the number mean anything.

**`macs` stays raw and nothing scales it.** The utilization term describes how
long the array was busy, not how many multiplies happened. `Stats::macs` is the
count, `effectiveMacs`, `utilization` and `delta` sit beside it, and the energy
path of Phase P11 consumes the raw count only. A utilization scaled action count
handed to Accelergy would overstate the energy of precisely the layers the
evaluation cares most about.

**What would move this decision.** Two things, and neither is a preference.
Giving the machine per element register files and a 2D interconnect would make
row stationary expressible, at which point this record is superseded rather than
edited. And a measurement, from SCALE-Sim or from the Gemmini target, showing
that the fill and occupancy terms above misprice the suite by more than the
uncertainty already claimed for them would move the constants inside the pin
without moving the pin itself. Until one of those happens the number stays an
assumption with a stated uncertainty, and the report says so wherever it quotes
one.

**What this does not decide.** It does not decide the tiling order, which is
Phase P13's and is scored against this same cost function rather than against a
second heuristic one. It does not decide anything about the kernels: the
convolution kernel is direct and stays direct, Section 10.3 rejects an im2col
plus GEMM restructuring of it outright, and mapping a convolution onto
`gemmCharge` is how the cost of the array is computed rather than a claim about
how the arithmetic is performed. The two are separate on purpose.
