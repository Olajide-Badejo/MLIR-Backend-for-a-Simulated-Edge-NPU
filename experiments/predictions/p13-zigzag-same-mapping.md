<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: what ZigZag says about the mappings this compiler chose

- **id:** p13-zigzag-same-mapping
- **written:** 2026-09-06
- **result field:** `zigzag_latency_cycles`, `simulated_cycles`
- **direction:** this project reads **above** ZigZag on every tiled layer
- **magnitude bracket:** between 1.2 and 6 times per layer, with a geometric mean
  over the tiled layers between 1.5 and 4; the bounded exploration finishes in
  under ten minutes of wall clock and under two gigabytes of peak resident memory
- **answered at:** P13, Section 16.5, in the commit that runs the comparison

*The two fields are compared per tiled layer, under the one mapping
`npu.tiling_choice` records, and never as two whole model totals from two
mappers. Section 16.5 asks for cost under the same mapping and that is the whole
of what this entry is about.*

*Written and committed before the exporter exists and before ZigZag has been run
once against anything in this repository. Section 16.5 asks for cost under the
**same** mapping rather than two totals from two mappers, so what is predicted
below is a comparison of one arrangement, not a contest between two searches.*

## Hypothesis

### 1. The direction, and the reason is this machine's DMA charge

**Tiling multiplies transfers while leaving the MAC count alone**, which is why a
tiled program here is DMA bound: `docs/PASSES.md` records 1524 DMA cycles against
596 of compute on the hand written two tile convolution. This project charges a
transfer as bytes over bandwidth **plus a descriptor per transfer**, so splitting
one load into four costs four descriptors, and the makespan of a tiled layer is
its transfer timeline rather than its array time.

ZigZag scores a mapping against a memory hierarchy with reuse, and its latency
for a fixed mapping is the maximum over the levels of that hierarchy rather than
a per transfer descriptor sum. **So under the same mapping I expect ZigZag to
read below this project on every tiled layer**, and by more on the layers with
the most tiles, because the descriptor term grows with the tile count and
ZigZag's does not.

### 2. The size, and why the bracket is as wide as it is

The comparison this project already has against SCALE-Sim disagrees by as much as
8.65 times on `dilated_stack`'s `conv1` and 4.34 times the other way on
`inception_block`'s 1 by 1, and `docs/NUMBERS.md` records that the two tools
still disagree about the compute time of the same MAC count after the stalls are
removed. **A second external tool has no reason to land closer than the first**,
so the bracket is 1.2 to 6 per layer rather than something tighter, and the
geometric mean is the figure worth reporting because a single layer's ratio is
the noisiest number in this comparison.

**The bracket is one sided on purpose.** A ratio below 1.0 anywhere is ZigZag
reading above this project, which is the direction clause 1 says will not happen,
and one layer doing it is a fact to record while two would be the clause being
wrong.

### 3. The mapping exports without inventing anything

`npu.tiling_choice` is recorded on every tile the pass emits and carries the
chosen mapping. **I predict it carries enough to write both a ZigZag mapping and
a Timeloop mapping without a free parameter**: the tile extents per axis and the
order the pass unrolled them in. If it does not, the missing piece is named in
the report rather than filled in with a default, because a default would make the
comparison a comparison of two mappings again.

### 4. The exploration is bounded and cheap, because there is nothing to search

The comparison fixes the mapping, so ZigZag's search space is one point per
layer. **I predict the whole bounded run finishes in under ten minutes of wall
clock and under two gigabytes of peak resident memory**, over the tiled layers of
`resnet_block` and `inception_block` at their tight budgets plus every layer the
swept range tiles.

**If it does not fit, that step stops and records why.** The WSL configuration is
not a variable in this project and nothing here may propose changing it.

### 5. If ZigZag finds a better tiling, that is a defect entry and not a retune

Section 16.5 forbids tuning this project's search to match an external tool, and
P11 obeyed the same rule when it left D-0045 open. **I predict ZigZag's own
search finds a mapping it scores better than the one this compiler chose on at
least one layer**, because this compiler's search scores on a two port makespan
that charges a descriptor per transfer and ZigZag's does not, so the two are
optimising different objectives and would be expected to disagree about the
optimum. That is a defect entry with a reproduction, quantified, and the search
stays exactly as it is.

## What would falsify it

- **ZigZag reading above this project on two or more tiled layers** falsifies the
  direction in clause 1.
- **Any tiled layer outside the 1.2 to 6 bracket**, in either direction,
  falsifies the bracket, and a geometric mean outside 1.5 to 4 falsifies it
  separately.
- **`npu.tiling_choice` not carrying enough to write the mapping without a free
  parameter** falsifies clause 3, and the missing field is the finding.
- **The bounded run exceeding ten minutes or two gigabytes** falsifies clause 4.
- **ZigZag's own search agreeing with this compiler's chosen mapping on every
  layer** falsifies clause 5, and would be the more surprising outcome: it would
  say the descriptor term does not change which mapping is best, only what it
  costs.

**What would not falsify it:** ZigZag refusing a layer shape it does not model.
That is a coverage statement about the tool and it is reported as coverage, the
way the SCALE-Sim comparison reports pooling, rather than as a divergence.

## What each outcome would mean

- **This project above ZigZag by 1.5 to 4 times on average.** The descriptor per
  transfer is the largest single difference between the two cost models on a
  tiled program, and that is a statement about this machine's model rather than
  about the tiling.
- **The ratio near one.** The descriptor term is not what separates them and the
  disagreement with SCALE-Sim has a different cause than the one this entry
  assumes, which would make the open question `docs/NUMBERS.md` records sharper
  rather than answered.
- **ZigZag above this project anywhere.** Something in this project's charge is
  optimistic on a tiled layer, and the first place to look is the fill, which is
  charged once per fold here and which the tiling changes the fold count of.

## What is deliberately not predicted

**The energy comparison.** ZigZag reports energy as well as latency, and this
project's energy comes from Accelergy at 45nm with an fp32 MAC coefficient that
already fails Section 16.4's order of magnitude check at 10.71 times. Comparing
energy under the same mapping would measure that gap rather than the mapping, so
the comparison is latency only and the reason is written here rather than
discovered later.
