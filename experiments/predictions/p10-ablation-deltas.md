<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Prediction: which of the eight ablatable passes has a row that is not zero

- **id:** p10-ablation-deltas
- **written:** 2026-09-01
- **result field:** `instruction_count`, `simulated_cycles`, `max_abs_error_vs_onnxruntime`
- **direction:** three of the eight ablation rows move `instruction_count` upward when the pass is removed and five are exactly zero on every model at every budget
- **magnitude bracket:** the whole suite's `-O2` instruction count rises by between 1 and 30 instructions when the three are removed one at a time, and by 0 for the other five
- **answered at:** P10

*Written before `experiments/run_benchmarks.py` was run for the first time, and
committed strictly before the first result file. Ground rule 15: a prediction
written after the measurement is worth nothing, and a prediction that turns out
wrong and is reported honestly is stronger evidence of understanding than two
numbers that happen to agree. What follows is what I actually expect.*

**What is already known and is therefore not predicted here**, because a
prediction of something already measured is not a prediction. P9b measured that
`dilated_stack` is thirteen instructions at `-O0` and twelve at `-O2`, that the
saving is `-npu-fuse-bias`'s, and that `-O1` is exactly `-O0` on every model.
`-sccp`'s row being zero is likewise already argued from the structure and
recorded in `docs/PASSES.md`. Those three are stated below as expectations the
ablation table has to reproduce rather than as claims this file is putting at
risk. Everything else here is genuinely unmeasured on 2026-09-01.

## Hypothesis

### 1. Exactly three rows are not zero

`-npu-fuse-bias`, `-npu-fold-batchnorm` and `-canonicalize`. The other five,
`-npu-constant-fold`, `-npu-fuse-ops`, `-cse`, `-sccp` and `-symbol-dce`, produce
an identical instruction count with and without the pass, on all seven models, at
both budgets.

The reasoning per pass, because a bare list is not a prediction:

- **`-npu-fuse-bias` moves it by exactly one instruction, on `dilated_stack`
  alone.** That is the P9b measurement and this is the ablation table having to
  reproduce it from the other direction. If any other model shows a nonzero row
  for this pass, the P9b claim that the suite gives the bias fusion exactly one
  target was wrong.
- **`-npu-fold-batchnorm` moves it upward on `conv_bn_relu_stack` and nowhere
  else**, because that model is the only one built to keep unfolded
  `BatchNormalization` nodes through export. An unfolded batch norm is decomposed
  at lowering into a multiply and an add over per channel constants, so removing
  the fold turns each folded batch norm into two instructions plus their
  constants' transfers. I expect **between 4 and 24 extra instructions** on that
  model.
- **`-canonicalize` moves it upward on at least `conv_bn_relu_stack`**, because
  the second canonicalization exists to remove the dead parameter constants the
  batch norm fold leaves behind, and a dead `npu.constant` becomes an
  `npuisa.const` and a `dma_load` in the instruction stream. I expect **between 4
  and 16 extra instructions** across the suite. Whether it moves any other model
  is the part of this I am least sure of.
- **`-npu-fuse-ops` is exactly zero everywhere**, and this is the sharpest claim
  in the file, because `docs/PASSES.md` states it as a property rather than a
  measurement: forming an `npu.fused_op` region is numerically inert and the
  lowering flattens it back into the same instructions. If this row is not zero,
  either the claim is wrong or the lowering treats a fused region differently
  from its unfused form, and either is a finding worth more than this prediction.
- **`-cse` is zero on all seven.** These models are generated from a seed and
  none of them contains two identical operations over identical operands. I would
  guess `inception_block` is the likeliest to surprise me, because it is the one
  with parallel branches over the same input.
- **`-sccp` is zero on all seven**, which is structural rather than empirical:
  it propagates constants through the call graph and there is one function and no
  calls, so there is nothing for it to cross. Recorded in `docs/PASSES.md` under
  both passes and asserted by `test_sccp_has_nothing_to_do_on_a_single_function`.
- **`-symbol-dce` is zero on all seven.** It removes a private symbol nobody
  calls, and every module here has one public function.
- **`-npu-constant-fold` is zero on all seven.** It evaluates elementwise `npu`
  operations over constant operands, and no model in this suite has an elementwise
  operation both of whose operands are constants. That is the same reason the
  `-O1` level, whose whole content is this pass and one canonicalization, is
  exactly `-O0`.

### 2. No ablation moves the numerics beyond tolerance, with one asymmetry

Every ablation row's `max_abs_error_vs_onnxruntime` stays inside the Section 17.4
absolute band of 5e-5. Section 16.2 makes the opposite a finding and a failed
run, because a pass whose removal breaks correctness is load bearing for
correctness rather than for performance.

The asymmetry: `-npu-fold-batchnorm` is the one pass in this pipeline that moves
numbers, so **ablating it should move the `-O2` answer back toward `-O0`'s**. I
predict `max_abs_movement_vs_o0` falls to exactly 0.0 on `conv_bn_relu_stack`
when the fold is removed, and that no other ablation moves that field at all.

### 3. The rows are identical at both budgets

Every ablation row's `instruction_count` and `simulated_cycles` are the same at
the default budget and at the tight one, for all seven models and all eight
passes, because the per model tight budgets of ADR 0008 are the smallest budget
at which the program still places, so nothing spills at either.

### 4. The suite total

Summed over the seven models at `-O2` and batch 1, the default budget, the total
instruction count of the unablated suite is **between 100 and 130**, and no
single ablation raises it by more than 30.

## What would falsify it

Any of these, and each is recorded as a finding with its number rather than
smoothed over:

- **A fourth pass with a nonzero row**, or one of the three named coming out
  zero. Either means the mechanism named above for that pass is wrong.
- **`-npu-fuse-ops` not exactly zero.** This is the one I would most want to be
  wrong about, because it would mean the lowering does not flatten a fused region
  into the same instructions its unfused form produces, which is a claim
  `docs/PASSES.md` makes in prose today.
- **`-npu-fuse-bias` nonzero on a model other than `dilated_stack`**, which
  would contradict `test_the_suite_gives_the_bias_fusion_exactly_one_target`.
- **Any cell's `max_abs_error_vs_onnxruntime` outside 5e-5 on an ablation row**,
  which Section 16.2 makes a failed run rather than a table entry.
- **A row that differs between the two budgets**, which would mean something
  spills at the tight budget and the per model tight budgets of ADR 0008 are not
  what that record says they are.
- **An extra instruction count outside the brackets above**: below 4 or above 24
  for the batch norm fold, below 4 or above 16 for canonicalization, or a suite
  total outside 100 to 130.

## How this gets answered

The measured table goes into `docs/NUMBERS.md` with this file quoted beside it
and every claim above marked met or not. The result cells that carry it name it
in `prediction_id` and name this file's landing commit in `prediction_sha`, and
`test/Python/test_predictions.py` asserts the landing commit is an ancestor of
the commit each of those cells was measured at. This file is not edited after the
measurement, whatever the measurement says.
