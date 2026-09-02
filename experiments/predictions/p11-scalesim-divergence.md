<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: how far this cost model will sit from SCALE-Sim, and why

- **id:** p11-scalesim-divergence
- **written:** 2026-09-01
- **result field:** `scalesim_cycles`, `simulated_cycles`, `scalesim_covered_cycle_fraction`
- **direction:** this project's analytical cycle count comes out **above** SCALE-Sim's on the layers SCALE-Sim covers, and the gap is dominated by named modelling differences rather than by error
- **magnitude bracket:** per layer divergence under 10 percent on dense compute bound layers, 10 to 25 percent on the 1 by 1 convolutions, and a whole model divergence of 5 to 20 percent once the skipped operations are excluded from both sides
- **answered at:** P11

*This entry lands at P10 and is answered at P11, and the split is the whole
point of the mechanism. Section 17.8: P11's gate requires the divergence
prediction to already exist and to already be an ancestor of the commit that
records the first SCALE-Sim number, and a prediction written in the same phase as
the measurement it predicts is a prediction nobody can prove predates it.
Committing it here, before `experiments/scalesim_export.py` exists at all, is the
strongest form of that claim available: on 2026-09-01 there is no code in this
repository that could have produced a SCALE-Sim number to write this from.*

## Hypothesis

### 1. The direction, and it is not the obvious one

The obvious guess is that an analytical model reads pessimistic because it lacks
DMA and compute overlap. **That is not the first order effect**, and Section
16.3 says so from the literature rather than from taste. This project's simulator
already models the overlap explicitly, on two ports, and records
`overlap_fraction` per run. So the overlap is not where the gap comes from here.

Three mechanisms are named ahead of the measurement, in the order I expect them
to matter:

1. **SCALE-Sim assumes optimistically high bandwidth on 1 by 1 convolutions**,
   so its counts come out low there [R39]. `inception_block` and `resnet_block`
   are the models in this suite with 1 by 1 convolutions, so I expect the widest
   per layer gaps on those two, and I expect them in the direction of this
   project reading higher.
2. **SCALE-Sim does not model pooling at all**, which is worth roughly 23
   percent of cycles on layers that have it [R39]. In this project pooling is a
   skipped operation on the export side, so it leaves the comparison entirely
   rather than biasing it, and its contribution appears as a coverage fraction
   below one rather than as a divergence. **This is the reason
   `scalesim_covered_cycle_fraction` has to be printed beside every agreement
   figure**: without it a high agreement on `lenet` would be an agreement over
   whatever survived the export.
3. **Array fragmentation.** The array is 16 by 16 and several layers in this
   suite have channel counts that are not multiples of 16, so both models charge
   for positions that do nothing. I expect the two to disagree about how much,
   because this project's `utilization` and `delta` are analytic factors and
   SCALE-Sim's is a mapping.

### 2. The numbers

Section 16.3 pre-registers the bands and this entry adopts them rather than
inventing softer ones:

- **under 10 percent per layer on dense compute bound layers is expected**, and
  is what the published error bands for analytical models of this class report:
  under 1 percent at 16 by 16 and 7 percent averaged over array sizes [R39], 3.9
  percent against cycle accurate RTL [R44], 0.1 to 8 percent for the sparse
  analytical model [R45];
- **10 to 25 percent is a finding requiring an explanation**, and I expect the 1
  by 1 convolutions to land here;
- **above 25 percent is a defect requiring a root cause**, and I predict no layer
  reaches it at the default budget with both ports enabled.

**The single port cells are excluded from that band**, deliberately and in
advance. Section 16.3 says to version the band against the cost model constants
because single port cells will legitimately exceed it, and a band that quietly
covered them would be a band this project would have had to widen after seeing
the answer.

**Whole model:** 5 to 20 percent, computed only over the covered layers.

**Coverage:** I expect `scalesim_covered_cycle_fraction` between 0.5 and 0.85 on
the convolutional models and **below 0.3 on `lenet`**, which is the model with
the most elementwise and pooling work relative to its convolutions.

### 3. Rank fidelity holds where absolute error does not

The cost model exists to make compiler decisions, so what matters is whether it
orders candidates the way the reference does. I predict **Kendall tau above 0.8**
between the two rankings across the grid, and pairwise comparison accuracy above
0.85, even on the models where the absolute divergence lands in the 10 to 25
percent band. A cost model can be numerically poor and decision optimal [R46],
and the reason to predict this separately is that the two genuinely come apart.

## What would falsify it

- **This project reading systematically *below* SCALE-Sim** on the covered
  layers. That would invert the first mechanism and would mean the overlap model
  is crediting more hiding than the hardware could deliver.
- **Any layer above 25 percent at the default budget with both ports on.**
  Section 16.3 makes that a defect with a root cause, not a table entry.
- **The widest gaps landing somewhere other than the 1 by 1 convolutions**, which
  would mean mechanism 1 is not the dominant term and the account has to be
  rewritten from the decomposition rather than from this file.
- **A decomposition that does not sum.** Section 16.3 requires the divergence to
  read as pooling gap plus elementwise gap plus dilation approximation plus
  double buffering plus fragmentation plus residual. If the residual dominates,
  the named terms were the wrong terms.
- **Kendall tau below 0.8**, which would mean the cost model is not merely
  imprecise but is ordering candidates differently from the reference, and that
  is a stronger finding than any absolute error here.
- **A coverage fraction above 0.9 on any model**, which would mean the exporter
  is representing operations it has no systolic representation for rather than
  skipping them, and every agreement figure beside it would be measuring the
  wrong thing.

## How this gets answered

P11 records `scalesim_cycles` per cell with its per layer breakdown, the
`skipped` and `approximations` lists, and both coverage fractions. Those cells
name this entry in `prediction_id` and name this file's landing commit in
`prediction_sha`, and `test/Python/test_predictions.py` asserts that commit is an
ancestor of the commit each cell was measured at. Section 23's P11 gate requires
this file quoted verbatim beside the measurement with each claim answered. It is
not edited after the measurement, whatever the measurement says.
