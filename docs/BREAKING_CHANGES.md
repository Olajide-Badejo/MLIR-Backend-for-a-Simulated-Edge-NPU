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

### 2026-10-06, Phase P14: the quantized cells, 63 of them, and the suite at 280

**Written before the commits that cause it.** Three commits follow this one and
a `record:` commit after them; each is named here once it exists.

**What changes.**

- **The benchmark suite gains 63 quantized cells**, the mirror of the fp32
  benchmark grid the driver runs under ADR 0010, as ruled on 2026-09-30: seven
  models at three levels, at the default budget at both batches and at the
  tight budget at each model's declared batch. Each is the model calibrated
  from its committed profile, compiled through the QDQ contraction, and
  measured as an fp32 cell is, with the fields only a quantized cell fills:
  `quant_boundary_crossings`, `simulated_cycles_without_int8_packing`,
  `max_abs_error_vs_fp32_simulated` and `sqnr_db_vs_fp32_simulated` against its
  fp32 twin at the same model, level, budget and batch, and the manifest's
  `calibration_methodology_version`. Ablation cells stay fp32 only. **The suite
  is 280 cells**: 63 fp32 benchmark cells, 63 int8 ones and 154 fp32 ablation
  cells.
- **The analysis reads an int8 program.** The `npuisa` walker, the roofline and
  the SCALE-Sim export were written for f32 programs and refuse any other
  element type. They learn the int8 and int32 widths, the int8 peak an integer
  contraction is charged at, and the `QUANT` and `DEQUANT` instructions; the
  SCALE-Sim decomposition gains a named `int8_packing` term, because SCALE-Sim
  models no packing and an int8 layer's divergence would otherwise sit inside
  the fragmentation term. Every one of these is a mirror of what the simulator
  already charges, and none of them changes what an f32 program gets.
- **Every hardcoded count of the suite moves from 217 to 280 in one commit**,
  derived from the driver: the tests, the harness's docstrings, and the
  documents that state the suite's size.

**Which baseline fields move.**

- **`experiments/results/`: 63 files arrive.** The 217 fp32 cells are
  re-recorded in the same serial run and **every field each of them carries is
  asserted unchanged against `d2caa05`**, apart from what a re-record always
  moves, the timing objects, the timestamps, the git sha and the content hash,
  **and `manifest.run_order_position`**, which moves on nearly every fp32 cell:
  Section 16.2's seeded shuffle runs over the whole suite, and a suite of 280
  shuffles into a different order than a suite of 217. It records where a cell
  ran in that run and nothing the cell measured.
- **`report/generated/macros.tex`**: the cell count, the results sha and
  content hash, and macros for the quantized cells beside the fp32 ones.
- **`experiments/results-runtime.json`**: the run's own count and timing.
- **`test/baseline/baseline.json`**: suites only, which grow. Its 42 cells are
  fp32 and do not move, and neither does a golden tensor.
- **Not baseline fields, and declared here anyway because they are recorded
  numbers**: the 63 quantized cells' own figures, predicted before any of them
  is recorded.

**The prediction** is `experiments/predictions/p14-quantized-cells.md`,
committed before any of these commits exists.

**The commits that cause it:** named here once they exist.

### 2026-09-30, Phase P14: the cost model's INT8 terms, an int8 MAC coefficient and the throughput assumption made separable

**Written before the commits that cause it.** Three commits follow this one and
a `record:` commit after them; each is named here once it exists.

**What changes.**

- **An int8 multiply accumulate is charged its own energy.** The MAC array
  gains a second action, `int8_mac`, whose coefficient Accelergy answers from
  the same `Aladdin_table` plug in the fp32 MAC's comes from, at the same
  pinned 45 nm and 1 ns: an 8 bit integer multiplier and a 32 bit integer
  adder, because the machine multiplies int8 by int8 and accumulates in int32,
  as Section 14 requires. `macs` stays raw and stays one count: the array's
  action counts become `mac`, the fp32 MACs, and `int8_mac`, the simulator's
  own `int8_macs`, so every MAC is charged exactly once.
- **The int8 datapath adds no area.** The int8 peak already rests on four int8
  multiplies packed into each fp32 lane, and a lane that packs them is the fp32
  lane, so the two subcomponents carry an area scale of zero and the array's
  area is the one P11 recorded. The area a machine with separate int8 units
  would add is measured and recorded beside it, so the assumption has a number
  on it.
- **The cycle charge does not change.** An integer `CONV2D` or `MATMUL` is
  charged at `kPeakMacsPerCycleI8`, as it has been since the integer kernels
  landed. What changes is that the assumption becomes separable: the simulator
  gains an option that charges every int8 MAC at the fp32 peak instead, and a
  quantized cell records the cycles its own program takes under it, so that the
  int8 throughput assumption's share of a cycle win is the difference between
  the two figures and the rest is the DMA traffic reduction and the crossings.
  Section 14's gate asks for exactly that separation.
- **`include/NPU/Simulator/CostModel.h` gains `kI8MacsPerLane`, 4**, the
  packing factor the int8 peak has been written as a multiple of, pinned in
  `FrozenConstants`, mirrored in `npu_frontend.cost_model`, and asserted to be
  the ratio of the two peaks.
- **An int8 element at the scratchpad port stays one 32 bit access**, as an f32
  element is. Packing four to a word would lower the int8 scratchpad energy by
  up to four times and is not assumed, which is the direction that does not
  flatter the result.

**Which baseline fields move.**

- **`test/baseline/baseline.json`: `energy_per_action_pj` gains
  `mac_array.int8_mac` at each of its seven budgets.** No existing coefficient
  moves, and no cell field moves: every baseline cell is fp32 and performs no
  int8 MAC.
- **`experiments/results/*.json`, `schema_version` 2 to 3.** `simulation` gains
  `simulated_cycles_without_int8_packing`, null on every fp32 cell with its
  reason, and the manifest's `cost_model_constants` gains `I8_MACS_PER_LANE`.
  The 217 cells are re-recorded in one run, because a version 2 file is refused
  rather than read, and **every field each of them carried before is asserted
  unchanged, field by field**, apart from what a re-record always moves: the
  timing objects, the timestamps, the git sha and the content hash.
- **No fp32 cycle, byte, MAC, energy or area figure moves, and no golden
  tensor**, fp32 or quantized.
- **Not baseline fields, and declared here anyway because they are recorded
  numbers**: the cycles, the cycles without packing and the energy of the
  quantized programs, per model at the three levels, over the 63 configurations
  item 5 will record as cells.
- **Test counts and names**, which grow.

**The prediction** is `experiments/predictions/p14-int8-cost-terms.md`,
committed before any of these commits exists.

**The commits that cause it:** named here once they exist.

**Measured, after the commits that cause it.** *Added in the docs commit before
the baseline's record; everything above is the declaration as committed in
`5217d0c`, unchanged.*

- **The commits.** The simulator's option is `d383a48`, the energy path
  `b9b6288`, the schema `387a641`, and the 217 cells' record `408cf17`. Five
  more landed that the declaration did not foresee, and the correction is here
  rather than above: D-0070's entry `d71478e`, its fix `7c75ab8`, its audit
  `31f91a0` and its second fix `655295c`, and `c3eadc9`, a null reason's
  wording. And there are two records, not one: the 217 cells in `408cf17`, and
  the baseline in the `record:` commit after the one carrying this paragraph,
  because the baseline records the suites and the suites read the cells.
- **What the declaration covered, checked against the record.** It names
  schema 3 and the re-record of all 217 committed result files, with every
  field asserted unchanged apart from the timing objects, the timestamps, the
  git sha and the content hash. Leaf by leaf, the 217 files at `408cf17`
  against their parents differ in exactly those and in `schema_version`: 16
  leaf patterns, `content_hash`, `manifest.git_sha`, `manifest.timestamp`,
  `schema_version`, the four statistics of `timing.compile_ms` and of
  `timing.passes_total_ms`, and the same four of every `passes[*].timing`;
  12236 leaf differences in all. Three leaves arrived on every file, the two
  the declaration names and the null's reason. **It did not name two files the
  record also moved**, `report/generated/macros.tex`, whose results sha,
  schema version and content hash follow the cells, and
  `experiments/results-runtime.json`, the run's own timing.
- **Every field the 217 cells carried is unchanged**, compared field by field
  at the record against the files it replaced: every cycle, byte, MAC, energy,
  area, divergence, bound, pass count, oracle distance and layer name. What
  moved is what a re-record always moves, and what arrived is what this entry
  names: `schema_version` 3, `simulated_cycles_without_int8_packing` as a null
  with its reason on all 217, `I8_MACS_PER_LANE` in every manifest. **It took a
  defect fix to be true**: the first comparison found 170 layer names moved
  over 45 cells, D-0070, which `ff36b1f` had caused and nothing had checked,
  and the walker was fixed before anything was recorded.
- **The baseline's check before its record found suites only**, at
  `408cf17`: the simulator suite 80 to 82 and the pytest suite 1445 to 1450,
  and not one cell or golden line; `655295c` adds one pytest case after it.
  Its coefficient table gains `mac_array.int8_mac` at the record.
- **The int8 MAC is 1.0025 pJ**, 4.36 times the published 0.23, inside an order
  of magnitude; the fp32 MAC is untouched at 49.286. Separate int8 units would
  have added 0.691 mm2, which the zero area scale assumes away.
- **The quantized numbers, which are not baseline fields**, measured over the 63
  configurations item 5 will record: the cycles did not move; the packing's
  share of the cycle win is 0.26 on LeNet at batch 1 and 0.93 to 1.57 on the
  other models at the default budget, and 0.30 to 0.52 where an fp32 twin
  spills at a tight budget; the quantized energy is 0.20 to 0.63 of fp32 at the
  default budget and 0.07 to 0.29 at the tight one. The prediction's
  adjudication is in `docs/ENGINEERING_LOG.md` under 2026-09-30.

### 2026-09-29, Phase P14: the quantized goldens arrive at `-O1` and `-O2`, with the calibration after the `-O2` folds

**Written before the commits that cause it.** Two commits follow this one: the
harness learns to write a quantized golden at every level, and a `record:`
commit writes them. Each is named here once it exists.

**What changes.** `scripts/regression_baseline.py` compiles each model
calibrated from its committed profile at `-O0`, `-O1` and `-O2`, where it
compiled it at `-O0` only, and keeps each answer under
`test/baseline/golden/int8/<model>-O<level>-out0`. Twenty one quantized goldens
where there were seven. The quantized end to end test is extended to the same
three levels with both of its bounds, in a commit before the harness change
that moves no recorded field.

**Why.** The owner's ruling of 2026-09-29 puts quantized cells at all three
levels, and 7e313eb moved the calibration after `-npu-fuse-bias`,
`-npu-fold-batchnorm` and `-npu-fuse-ops` at `-O2`, which makes `-O2` a
different integer program from `-O0` on the two models with a fold. A level
whose arithmetic can differ from `-O0`'s and has no golden is a level whose
arithmetic can move silently.

**Which baseline fields move.**

- **Fourteen golden tensors arrive**, and the check before the record reports
  each as produced and not recorded, which is the category a moved golden is
  counted in. Measured in the working tree after 7e313eb, before this entry:
  - at `-O1` all seven are byte identical to their `-O0` quantized goldens,
    because `-O1` runs no fold and no fusion and the calibration still runs
    first there. They are recorded anyway, because the layout is per level and
    a later change that separated the two levels should show as a moved tensor
    rather than a missing one;
  - at `-O2` five are byte identical to their `-O0` quantized goldens,
    `depthwise_separable`, `inception_block`, `lenet`, `lenet_batched` and
    `resnet_block`, and two are new tensors: `conv_bn_relu_stack`, whose batch
    norms are folded before its weights are scaled, and `dilated_stack`, whose
    bias add is fused into its second convolution so that the relu after it
    fuses too.
- **No `-O0` quantized golden, no fp32 golden and no cell field moves.** The
  harness change adds compilations beside the existing ones and changes none of
  them, and 7e313eb left all 217 planned fp32 cells byte identical, debug
  section included.
- **Not baseline fields, and declared here anyway because they are recorded
  numbers**: the quantized end to end budgets gain a per level table in
  `npu_frontend.tolerances`, identical to the `-O0` one at `-O1` and tightened
  at `-O2` for the two models that move, never loosened; and
  `quant_boundary_crossings` on a quantized compilation is 28 at `-O2` over the
  seven where it is 32 at `-O0` and `-O1`, which no committed cell carries yet
  because the quantized cells are item 5.
- **Test counts and names**, which grow. One test is renamed rather than kept:
  the one that asserted the quantized golden set is exactly one `-O0` tensor
  per model cannot hold once the set is per level, and it becomes the per level
  assertion.

**The prediction** is `experiments/predictions/p14-quantized-o1-o2.md`,
committed in b8e661d before any quantized program was compiled at `-O1`, or at
`-O2` with the calibration in its new position.

**The commits that cause it:** named here once they exist.

**Measured, after the commits that cause it.** *Added in the docs commit before
the record that answers this entry; everything above is the declaration as
committed in `222f70d`, unchanged.*

- **The commits.** The end to end test at three levels is `acd142c` and moved
  no recorded field; the harness change is `a6d80b8`; the tensors are written by
  the `record:` commit after the one carrying this paragraph.
- **The check before the record found what this entry names and nothing
  else**, from a quiet start at `a6d80b8`: the fourteen quantized goldens
  produced and not recorded, seven at `-O1` and seven at `-O2`, and not one
  cell line, fp32 golden line or `-O0` quantized golden line. 42 cells and the
  28 recorded tensors identical, the largest movement against `-O0` 4.470e-08.
- **The tensors are what the declaration said.** All seven `-O1` answers and the
  five named `-O2` answers are bit identical to the `-O0` quantized ones, which
  `test_quantized_end_to_end.py` asserts on all five input classes, and
  `conv_bn_relu_stack` and `dilated_stack` are new at `-O2`.
- **Test names moved more than the declaration said**, and the correction is
  here rather than above. Beside the growth and the one rename, the end to end
  test's parametrized cases gained a level in their identifiers, so the check
  reports 49 old identifiers gone and 197 pytest identifiers added, 1297 to 1445
  passed, and four lit files added. The renamed golden test lands with the
  record, because it asserts the tensors the record writes.
- **The numbers that are not baseline fields.** At `-O2`, `conv_bn_relu_stack`
  is 34.75 dB and 1.58 counts and `dilated_stack` 36.52 dB and 1.71 counts on
  the `normal` class; their `-O2` floors tightened to 33 and 35 dB and no count
  bound moved. `quant_boundary_crossings` is 28 at `-O2` over the seven, per
  model 4, 2, 4, 4, 6, 6 and 2.

### 2026-09-29, Phase P14: a relu that reads a calibrated operation is fused into its integer instruction, and an identical dequantize and quantize pair is removed

**Written before the commits that cause it.** Two commits follow this one and
cause the movement between them, the pair fold first and the fusion second; each
is named here once it exists.

**What changes.** Four things, the last of them only a guard:

- **`-npu-calibrate` quantizes a relu's output rather than the operation's**
  where a calibrated convolution or matrix multiplication has a relu as its only
  reader: the output quantize and dequantize go after the relu, with the scale
  and zero point the profile holds for the relu's output tensor. Which tensor
  that is comes from a new `graph` section in the profile, every node of the
  ONNX graph with its type, inputs and outputs, and the seven committed profiles
  are regenerated with it. No other byte of any profile moves.
- **The contraction takes dequantize, operation, relu, quantize** and emits one
  integer instruction carrying `relu`, which encodes to the `activation` field
  `CONV2D` and `MATMUL` have had since version 1. The machine clamps at the
  output zero point, the value that represents real zero. The standalone `RELU`
  keeps refusing i8, as Section 14 lists.
- **`npu.quantize` folds `quantize(dequantize(x))` to `x`** when the two scales
  are bitwise equal, the two zero points are equal, and 255 times the scale is a
  finite f32, which are the conditions under which the pair returns every one of
  the 256 values unchanged. `-npu-calibrate` applies it to the pairs it forms, so
  it holds at every level, and `-canonicalize` applies it wherever it runs.
- **`-npu-fuse-ops` leaves a calibrated operation's relu alone**, because the
  contraction now fuses it into the instruction itself and a region around a
  calibrated operation is one its verifier refuses.

**Why the movement is worth taking.** Without the fusion an integer program
dequantizes to f32 between every layer, because the relu sits there in f32.
The gate's DMA traffic reduction would then be measured on a program no INT8
NPU runs, and `quant_boundary_crossings` would count an artefact of this
lowering rather than the I8 rejection boundary Section 14 draws around `ADD`,
`MUL`, `RELU` and the pools. The accuracy moves in the right direction for the
same reason: one rounding at the relu output's scale replaces two, the first of
them at a scale about twice as coarse.

**Which baseline fields move.**

- **No fp32 field and no fp32 golden, by construction and by measurement.** The
  fusion needs `weight_scales`, which only `-npu-calibrate` writes; the fold
  needs a quantize, which no fp32 compilation contains; the fusion pass's guard
  needs `weight_scales` too. The baseline check before the record is the
  measurement: 42 cells and 21 fp32 goldens identical.
- **The quantized goldens of `depthwise_separable`, `lenet`, `lenet_batched`
  and `resnet_block` at `-O0`**, re-recorded in a `record:` commit of their own.
  The other three models have no relu reading a calibrated operation and their
  goldens do not move by a byte.
- **Not baseline fields, and declared here anyway because they are recorded
  numbers**: the `-O0` accuracy table of item 3, re-measured with the old one
  kept beside it, and the budgets in `npu_frontend.tolerances`, which may
  tighten from the new observations and may not loosen; and
  `quant_boundary_crossings` on a quantized compilation, 44 to 32 over the
  seven by the prediction, which no committed cell carries yet because the
  quantized cells are item 5.
- **Test counts and names**, which only ever grow.

**`Program::kVersion` does not move.** The `activation` field is in the record
from version 1 and its check has always admitted `relu` on `CONV2D` and
`MATMUL`. No byte of any existing program moves, because nothing set the field
before.

**The prediction** is `experiments/predictions/p14-fused-relu-and-pair-fold.md`,
committed before either change existed.

**The commits that cause it:** named here once they exist.

**Measured, after the commits that cause it.** *Added at the record that
answers this entry; everything above is the declaration as committed in
`d7369cd`, unchanged.*

- **The commits.** Four followed this entry, not the two it said, and the
  correction is here rather than above: the fusion needed the instruction's
  attribute and the profile's graph section first. `4c2c56f` is the pair fold,
  with no pair yet to fold; `8e37021` the attribute, which nothing set yet;
  `e718c72` the graph section, which nothing read yet. Each was measured to move
  nothing, all 98 model IR files byte identical across the first two, and the
  third changed no C++. The fusion is `69d7651`.
- **What moved is what this entry names and nothing else.** Of the 98 model IR
  files, the quantized `npu` and `npuisa` files of `depthwise_separable`,
  `lenet`, `lenet_batched` and `resnet_block` changed, and no other. The
  baseline check before the record found those four quantized goldens moved,
  each by about one count of its output scale, and not one cell line or fp32
  golden line: 42 cells and 21 fp32 tensors identical, the largest movement
  against `-O0` 4.470e-08 as before.
- **The accuracy table**, on the `normal` class at `-O0`, went to 52.68, 45.24,
  46.76 and 47.83 dB on the four from 46.17, 43.89, 42.97 and 46.14, and the
  other three are bit identical; the four budgets tightened and none loosened.
  The prediction's adjudication is in `docs/ENGINEERING_LOG.md` under
  2026-09-29.
- **`quant_boundary_crossings`** on a quantized compilation went from 44 to 32
  over the seven, 6, 2, 4, 6, 6, 6 and 2 per model.
- **`Program::kVersion`** is 2, unmoved.

### 2026-09-20, Phase P14: a quantized compute instruction gains a fourth operand, the per output channel rescale

**Written before the commit that causes it.** The commit that adds the operand
is the next one, and this entry is what makes it a decision rather than an
explanation.

**What changes, in one sentence.** `CONV2D` and `MATMUL` accept a **fourth
operand**, an `i32` buffer in the scratchpad holding one requantization
multiplier and one shift per output channel, and it is meaningful only when the
result element type is an integer one.

**Why, and it is the owner's decision of 2026-09-20.** Section 14 opens with the
granularity decision: weights get one symmetric scale per output channel, and
the P14 gate asks for the ablation that measures it. The rescale of output
channel `c` is `M_c = (scale_x * scale_w_c) / scale_y`, so a weight scale that
varies per channel makes the multiplier vary per channel by construction, and
the two scalar fields `requantMultiplier` and `requantShift` express exactly one
channel between them. Unlike the output zero point, which was a homeless number
that fitted in an idle word, this is `F` numbers and no vector in the record is
idle. The three ways out were an operand, per tensor weights, and a wider
record; the second contradicts Section 14's own granularity paragraph and makes
the gate's ablation unmeasurable, and the third is forbidden in this phase by
name, because a version bump invalidates `test_binary_stability` and every seed
in the fuzz corpus in the commit that introduces quantization.

**Which baseline fields move: none, and that is a measurement rather than a
hope.** The operand list has always been length prefixed, `putCount(out,
instruction.operands.size())` on the way out and `operands.resize(operandCount)`
on the way in, so an instruction with two or three operands writes exactly the
bytes it writes today and reads back the same. `Program::kVersion` stays **2**.
No cell of the 217 can reach the new operand, because no pass in any `-O` level
emits an integer instruction yet. The 21 fp32 golden tensors are untouched by
construction: an f32 instruction never carries this operand and is refused if it
does.

**What does change is a declared interface**, which is why this entry exists at
all rather than a changelog line. The two opcodes' `maxOperands` goes from 3 to
4 in `include/NPU/Encoding/NPUISADescription.td`, their integer operand profile
gains a slot, and `docs/ISA_MANUAL.md` and `docs/ISA_OPCODES.json` are
regenerated in the same commit with `check-isa-staleness.sh` clean.

**The prediction, before the measurement.** The malformed corpus is run at the
parent and at the change and every verdict compared case by case, as it was for
the output zero point. **I predict that the only cases that can flip are those
whose mutated operand count is exactly four on `CONV2D` or `MATMUL`, that such a
case flips from `arity` to the check that refuses the new operand rather than
becoming accepted, and that no case flips from refused to accepted.** If a case
does become accepted, the widening admitted a program the format should still
refuse and the change is wrong rather than the prediction.

**The per tensor path stays and stays tested.** Section 14's own ablation
compares per channel against per tensor, so both arms are selectable: an
instruction without the fourth operand rescales with the two scalar fields
exactly as it does today, and that path keeps its hand computed tests. **No
accuracy number measured under the interim per tensor execution is published as
the phase's result.**

**The commit that causes it:** the next one, named in this entry once it exists.

### 2026-09-16, Phase P14: an integer compute instruction carries its output zero point in the `scale` field

**Written before the commit that causes it.** The commit that changes the
meaning of the field is the next one but two, and this entry is what makes it a
decision rather than an explanation. The one in between adds the instrument that
measures it, and adds no behaviour.

**What changes, in one sentence.** On `CONV2D` and `MATMUL`, and **only when the
result element type is an integer one**, the `scale` word of `Instruction` stops
being required to hold zero and starts holding the output zero point.

**Why, and it is the owner's decision of 2026-09-07.** Checkpoint A found that a
quantized convolution needs two zero points and `Instruction` carries one. The
input's is needed at run time, because Section 14's rule is that padding
contributes `zp_x` and because the `- zp_x * sum_k q_w[k]` term folded into the
int32 bias is taken over the whole window; so the `zeroPoint` word is the
input's and cannot be anything else. The interim implementation therefore made
quantized compute results **symmetric**, which is consistent with Section 14's
pinned arithmetic paragraph and **inconsistent with its calibration paragraph**,
where activations are affine. Symmetric output costs the negative half of the
int8 range on every post ReLU tensor in the suite, which is most of them. The
owner settled it: the activations stay affine as Section 14 specifies, and the
output zero point goes in the `scale` word.

**Why that word is the place, rather than a new one.** On an integer compute
instruction the scale is **already** carried, folded into `requantMultiplier`
and `requantShift`: `M = (scale_x * scale_w) / scale_y` is the whole of the
rescale and the fixed point pair is how the machine applies it. So the f32 word
is idle on exactly the instructions that need somewhere to put a zero point, and
using it costs no layout change. `QUANT` and `DEQUANT` are untouched: they
declare `FieldScale`, they use it as a scale, and nothing about them moves.

**Which baseline fields move: none, and the measurement says so rather than the
prose.** No cell of the 217 contains an integer instruction, because no pass in
any `-O` level emits one yet, so no recorded number can reach this change.
`regression-baseline --check` reports no drift at the parent and is expected to
report none at the change, with the 21 fp32 golden tensors byte identical.

**So why this file rather than the changelog.** Because it changes the meaning of
a field in a **declared interface**. The ISA description, `docs/ISA_MANUAL.md`
and `docs/ISA_OPCODES.json` all state what the `scale` word means, and a reader
of a `.nbin` decides what a number is by reading them. That is the same class as
P13's entry for checks 8 and 9, which moved no cell either and changed what the
validator accepts. A change that alters how a committed file is to be read
belongs where changes are declared in advance.

**No byte of any existing program moves.** `Program::kVersion` stays 2, the
layout is untouched, the fuzz corpus is not reseeded, and the field was already
physically present on every instruction. An f32 program is accepted or refused
exactly as before, because the new rule is gated on the result element type and
the f32 path keeps the rule it has always had: the word holds zero.

**What becomes newly legal, and what becomes newly refused.** At an integer
result on those two opcodes, a `scale` word holding an integral value inside the
i8 range becomes legal where only zero was legal before, which is strictly a
widening; zero stays legal and is the symmetric case. Two values that were
already refused are still refused and are refused **by name** now: a non
integral value, because a zero point is an integer, and a value outside
`[-128, 127]`, because it is an i8 zero point. Each gets its own unit test.

**The prediction on the malformed corpus, written before it is run.** **No case
flips its verdict and no case flips its check name.** Every case in the corpus is
a mutation of `chainProgram`, whose second instruction is an `f32` `RELU`, and
the cases that reach an integer result element type do it by setting the opcode
to `QUANT`, which declares `FieldScale` and is not part of this change. The one
case that writes a scale onto an opcode that does not quantize writes it onto
that `f32` `RELU`, which still refuses it with the same check and the same
message. The corpus is run at the parent and at the change and every difference
is listed; a flip that this paragraph did not predict is a finding.

**Where the old behaviour is recorded.** The symmetric path never shipped and no
published number was measured under it, so nothing is restated in `NUMBERS.md`.
What it cost is an argument rather than a measurement, and `PHASE_STATE.md` and
the engineering log say the owner decided rather than pretending the question
was never open.

**Why the change is worth making.** The alternative is a machine that cannot
represent an affine activation, on a suite whose tensors are mostly one sided
after a ReLU, measured in a phase whose gate is accuracy per model. Shipping the
symmetric path would have put a representational choice inside every accuracy
number the phase reports, where no reader could see it.

### 2026-09-06, Phase P13: `-npu-double-buffer` starts firing, and declines the prefetches that would not place

**Written before the commit that causes it.** The commit that changes the pass
is the next one; this entry is what makes a pass that has fired on nothing since
it landed start moving recorded fields a decision rather than an explanation.

**Why, in one paragraph.** D-0054 recorded that the pass fires on nothing this
compiler emits, for a reason that is not the overlap being worthless: every
argument load sits in the entry block beside the other argument loads, where the
hoist correctly stops at another transfer, so the one transfer with a
computation before it is the load of a weight, and its source is an
`npuisa.const` that `prologueOf` would not let move with it. Admitting
`npuisa::ConstOp` to the prologue makes the pass fire on one to four transfers
per model. It also makes **five of the seven models stop placing at their ADR
0008 tight budgets**, because a prefetched weight is resident across the
computation it hides under, and those budgets are frozen. **A prefetch that
cannot be placed is not a prefetch**, so the pass now asks what the doubling
costs before it commits to it.

**The rule, exactly, because a pass that declines has to say what it declines
on.** Before committing a hoist, the pass takes the live intervals the allocator
itself collects, moves the definition of every allocation that would travel with
the transfer to the point the transfer is moving to, and runs the allocator's own
`assignOffsets` over the result at the same budget, the same strategy and the
same alignment. A set that does not place is a decline, counted as
`would-not-fit` beside `not-hoisted`, so a pass that answered no reads
differently from one that never asked.

**The sweep line peak was tried as that rule first and is not enough**, and the
measurement is why this entry is worth reading. Section 13.1 says the peak is a
lower bound on any placement and that the allocator's spill trigger is therefore
"offset assignment failed" and never "peak exceeded budget". Under a peak rule
`dilated_stack` at its tight budget of 8064 accepted a prefetch that left a peak
of 8028, which fits, and the arena then needed 8084 and **the cell did not
compile**. 56 bytes of alignment is the difference between a lower bound and an
answer, and a change that takes a cell away is not one this phase ships.

**What the pass answers now, measured at this tree per model, as prefetched /
not hoisted / would not fit.**

| Model | at the default 1048576 | at its tight budget |
|---|---|---|
| `lenet` | 4 / 7 / 0 | 3 / 7 / 1 |
| `depthwise_separable` | 1 / 4 / 0 | 1 / 4 / 0 |
| `resnet_block` | 2 / 4 / 0 | **0** / 6 / 1 |
| `inception_block` | 2 / 5 / 0 | 1 / 8 / 0 |
| `conv_bn_relu_stack` | 2 / 4 / 0 | 1 / 4 / 1 |
| `dilated_stack` | 1 / 4 / 0 | **0** / 4 / 1 |
| `lenet_batched` | 4 / 7 / 0 | 3 / 7 / 1 |

**Which fields this predicts will move.**

| Field | Prediction |
|---|---|
| `npuisa_op_counts` | **moves on every `-O2` cell where the pass hoists at least one transfer.** `npuisa.dma_load` falls by the number prefetched, and `npuisa.dma_load_async` and `npuisa.await` appear with that count. It is the same transfer named differently |
| `simulation.fragmentation_ratio` | **moves on six of the seven default budget baselines**, every model but `depthwise_separable`, because the prefetched destination is live longer and the arena is packed differently. Sweep line peak and high water mark, measured: `lenet` 194560 to 194896 and 194592 to 194960; `resnet_block` 8480 to 8512 and 8480 to 8544; `inception_block` 6728 unchanged and 6848 to 6856; `conv_bn_relu_stack` 6432 to 6560 in both; `dilated_stack` 8008 to 8028 and 8036 to 8084; `lenet_batched` 200800 to 201136 and 200800 to 201168 |
| the same, at the **tight** budgets | **does not move on any model.** A tight budget is the smallest at which the program places, so a hoist that raises the peak does not place and is declined, and only a hoist that costs nothing survives there. That is the decline rule doing exactly what it was written for, and it is the sharpest prediction in this entry |
| `instruction_count` | **does not move anywhere.** `npuisa.await` encodes to nothing and `npuisa.dma_load_async` encodes to `DMA_LOAD`, because the wait is a property of the dialect level and not of the machine. A function does not get longer from a reorder |
| `simulated_cycles`, `compute_cycles`, `dma_cycles`, `overlap_fraction` | **do not move anywhere.** Section 5.5 starts an instruction at the later of its port becoming free and its last operand becoming ready, which is a dataflow schedule rather than a program order one, so reordering two instructions charged to different ports moves no start time. `docs/PASSES.md` already carries that measured on a hand written tiled convolution: 2116 cycles, 1524 DMA and 596 compute, before and after |
| `dram_bytes_read`, `dram_bytes_written`, `macs`, `scratchpad_elements_read`, `scratchpad_elements_written`, `spill_count` | **do not move anywhere.** No transfer is added or removed and no spill count moves in the measurement above |
| the 21 golden tensors | **byte identical.** A reorder of a transfer and a computation changes no arithmetic, so the numerics move by exactly zero rather than by less than a tolerance |
| `max_abs_error_vs_onnxruntime` | **does not move**, for the same reason |
| the 14 `-ablate-npu-double-buffer` cells | **do not move at all**, because the pass is not in those pipelines and the tiling coupling is unchanged |
| `content_hash`, and `npuResultsContentHash` with it | moves on all 217 cells, because it is a hash over the compiler sources and they moved. That is provenance rather than a measurement, and it moves on every code commit |

**What does not move, and this list is the point of the entry.**

- **No bound, tolerance or threshold.** `GOLDEN_TOLERANCE` stays zero,
  `TIMING_GAP_FRACTION` stays 0.5, and the coverage thresholds stay where they
  are.
- **ADR 0008's suite tight budgets are not re-measured and do not move.** They
  are the reason the pass declines rather than the thing that gives way.
- **No cost model constant**, and no file under `include/NPU/Simulator`,
  `lib/CostModel` or `python/npu_frontend/cost_model.py`.
- **`Program::kVersion` stays 2** and the corpus is not reseeded.

**Why the regression is worth taking.** The alternative is a pass that has been
in `-O2` since the wiring commit and has never once fired, whose ablation row is
entirely the tiling coupling, and whose only measurement is a negative about the
pair. Section 13.3 asks what overlapping a transfer with a computation is worth
on this machine, and a pass that never overlaps anything cannot answer it. What
this entry buys is a row that measures the pass.

**A default budget cell moving is not the wiring defect the tiling entry
named.** That rule was about `-npu-tile-to-scratchpad`, where nothing is over
budget at the default budget, so a default budget cell moving would have meant
the search firing where it had nothing to do. Here the default budget is where
the pass has the most room, and a default budget cell that moves is the pass
working.

**Outcome, written after the re-record and kept apart from the predictions
above, which are not edited.** The causing commit is `4e9816d` and the record is
the commit after it. All 217 cells were re-run once, serially, on a quiet
machine, and every field of every cell was diffed against the recorded one.

| Prediction | Verdict |
|---|---|
| `npuisa_op_counts` moves wherever a transfer is prefetched | **met**, on **138** cells. `lenet-O2-default-n1-fp32-normal` reads `npuisa.dma_load` 11 to 7 with `npuisa.dma_load_async` 4 and `npuisa.await` 4 beside it, which is the same eleven transfers named differently |
| the fragmentation ratio moves on six of the seven default budget baselines | **met**, and **56** default budget cells move it once each baseline's eleven ablation rows are counted with it |
| it does not move on any tight budget cell | **wrong, on one cell of the 217.** `conv_bn_relu_stack-O2-tight-n1-fp32-normal-ablate-npu-fold-batchnorm` goes from 1.1834 to **1.0**, which is the arena becoming exactly the peak rather than a cost. The reasoning behind the clause was about the seven **baselines**: a tight budget is the smallest at which **that** program places, so a hoist that raises its peak cannot place and is declined. An ablation row is a different program run at the baseline's budget, so it has slack there, and this one had enough room for a prefetch that happens to pack better than the program without it |
| `instruction_count` moves nowhere | **met**, on all 217 |
| `simulated_cycles`, `compute_cycles`, `dma_cycles` and `overlap_fraction` move nowhere | **met**, on all 217 |
| traffic in both directions, MACs, scratchpad elements and spill counts move nowhere | **met**, on all 217 |
| the 21 golden tensors are byte identical | **met.** `git status` on `test/baseline/golden` is empty at the re-recorded tree |
| `max_abs_error_vs_onnxruntime` does not move | **met**, on all 217 |
| the 14 `-ablate-npu-double-buffer` cells do not move | **met.** Not one of them moved a counted field; each moved `content_hash` alone, which every cell did |
| `content_hash` moves on all 217 | **met** |

**Nothing outside the predicted set moved**, which is the clause this entry
exists to make checkable. Over instructions, cycles, compute and DMA cycles,
traffic in both directions, MACs, scratchpad elements read and written, spill
count, spill DMA count, the fragmentation ratio outside those 57 cells and the
oracle distance, the diff over all 217 cells is empty.

**The one wrong clause is worth more than the ten right ones**, and it is the
reason a prediction is written per field rather than per pass: the mechanism
behind it was sound and its scope was not, and only running it over the ablation
rows could have said so.

### 2026-09-01, Interphase P9b: `dilated_stack` gains the separate bias add, and every one of its cells moves

**Written before the commit that causes it.** The commit that changes the model
suite is the next one; this entry is what makes it a decision rather than an
explanation.

**What moves, and it is a model change rather than a compiler change.** That
distinction is the whole entry, so it goes first. Nothing about the compiler's
arithmetic moves here. A model in Section 15's suite gains one node, so it
computes a different function, so its outputs are different numbers. Every field
below moves because the program moved, and the fields that measure the
*compiler* against itself, `max_abs_movement_vs_o0` and
`max_abs_error_vs_onnxruntime`, do not move at all.

**Why the suite is changing.** P9 recorded the finding and left the decision:
`-npu-fuse-bias` fires on no model of Section 15's suite, because every
convolution in it carries its bias inline as a third `Conv` input, which is what
`torch.onnx.export` and this project's ONNX built models both emit. Section
16.2's ablation table at P10 would therefore have carried a row of zeros for that
pass, and a reader would have concluded that folding a bias into a convolution is
worth nothing on this workload. It is not worth nothing; there was nothing to
fold. `dilated_stack`'s `conv1` was already biasless and already followed by a
`Relu`, so one `Add` of a channel shaped initializer between them is the smallest
change that turns that row into a measurement.

**`GENERATOR_VERSION` moves from `1.0.0` to `1.1.0`**, which is what the manifest
field is for: a version that could not distinguish two suites is a version that
cannot be trusted. `--check` compares it before it compares a cell, and a moved
suite version is reported as its own line rather than as forty two puzzling
ones.

**The cells that move are the six `dilated_stack` cells and no others.** Three
levels times two budgets. Measured on 2026-09-01:

| Field | `-O0` and `-O1`, both budgets | `-O2`, both budgets |
|---|---|---|
| `instructions` | 11 to **13** | 11 to **12** |
| `cycles` | 1234.0625 to **1243.6875** | **1234.0625, unchanged** |
| `dma_cycles` | 674 to **743.25** | 674 to **743.25** |
| `compute_cycles` | 710.8125 to **720.4375** | **710.8125, unchanged** |
| `dram_bytes_read` | 4984 to **5004** | 4984 to **5004** |
| `dram_bytes_written` | 360, unchanged | 360, unchanged |
| `macs` | 41706, unchanged | 41706, unchanged |
| `max_abs_error_vs_onnxruntime` | 5.960464e-07, unchanged | 5.960464e-07, unchanged |
| `max_abs_movement_vs_o0` | 0.0, unchanged | **0.0, unchanged** |

**Read the `-O2` column, because it is the point of the change.** The twenty
extra DRAM bytes are the bias itself and they are read at every level, since the
bytes have to arrive whichever operation consumes them. Everything else the added
node costs is gone at `-O2`: the separate add becomes an operand of the
convolution, the instruction count is one lower than at `-O0`, and the cycle
count and the compute cycle count come back to **exactly** the numbers this model
had before the node existed. That is `-npu-fuse-bias` paying for itself on a
model of the suite, which is the thing P9 could not show.

**And `max_abs_movement_vs_o0` stays at 0.0 at `-O2`.** The fusion is bit exact,
because the simulator's convolution kernel adds the bias to the same `f32`
accumulator the unfused program stores and then adds to. P9 asserted that on a
model built for the test; this is the first time it is measured on a model of the
suite.

**Three golden tensors move, by 4.597557e-02 each.** `dilated_stack-O0-out0`,
`dilated_stack-O1-out0` and `dilated_stack-O2-out0`, all by the same amount and
all for the same reason: the model now adds a bias, so its output is the old
output plus a bias. **This is not a numerics movement and it is not inside any
tolerance band**, and confusing the two would be the worst available reading of
this entry. Section 17.6's 1e-6 band bounds how far a *level's* answer sits from
`-O0`'s on the *same* program, and that quantity is still exactly zero on all six
cells. A golden tensor is the answer of a particular program, and this is a
different program.

**The suite counts move too, and that is composition rather than regression.**
pytest goes from 864 to 867. One test is deleted,
`test_no_suite_model_gives_the_bias_fusion_anything_to_do`, which asserted the
gap so the claim could not go stale and which has done its job. Four arrive:
`test_the_suite_gives_the_bias_fusion_exactly_one_target`, which is its inverse
and which names the model rather than counting;
`test_the_bias_fusion_is_a_saving_and_not_a_rearrangement`, which measures the
one instruction and the bit equality;
`test_sccp_has_nothing_to_do_on_a_single_function`, which holds the other zero
row apart from this one; and
`test_the_dilated_stack_carries_a_separate_channel_shaped_bias_add`, which pins
the model's new shape where the model is built.

**What does not move, measured rather than assumed.**

- **No other model's cells move.** The `Add` and its initializer are appended
  after `conv1.weight` in the same generator draw order, so every other tensor in
  `dilated_stack` is bit identical and no other model shares a draw with it.
- **The tight budget of `dilated_stack` does not move.** Re-measured on
  2026-09-01 by the sweep `docs/adr/0008` describes: the allocated peak at the
  default budget is **8036 bytes** at all three levels, unchanged, and the
  smallest budget that allocates is still **8064**. The added buffers are the
  twenty byte bias constant and a 360 byte destination, both live late in the
  program where the pressure is a little over five kilobytes; the peak is set by
  `conv0`, which this change does not touch. So `docs/adr/0008`'s frozen constant
  stands and the model still spills nothing at it, which
  `test_the_tight_budget_spills_what_the_record_says` asserts.
- **`-O1` is still exactly `-O0` on this model**, in every field including the
  goldens.

**Why the regression is worth taking.** Section 16.2 asks P10 for a leave one out
ablation over the eight ablatable passes, and the value of that table is the
reasons behind its numbers. A row of zeros caused by a suite that cannot reach a
pass and a row of zeros caused by a pass that is worth nothing look identical in
a table and are opposite findings. This change costs six cells and three goldens
and buys a table whose rows mean what they say.

**The causing commit** is `feat(models): dilated_stack carries the separate bias
add -npu-fuse-bias exists for`. The baseline is re-recorded in the commit after
it, touching `test/baseline/` and nothing else.

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
runs each ablatable pass alone and asserts it. That was eight passes when this
entry was written and is eleven from P13, because the test sweeps the set the
driver reports rather than a list written beside it.

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

## 2026-09-02, P11: the energy fields, and two action counts the schema did not carry

**This entry declares a schema movement and not a numeric regression, and it says
so first because the distinction is the one this file exists to draw.** No golden
tensor moves. No instruction count, cycle count, DRAM byte count or MAC count
moves. Nothing this project has ever measured changes value. What changes is the
shape of two recorded files, and the rule at the top of this page says a field
that did not exist cannot have regressed. It is written here anyway, before the
commit that causes it, because the `schema_version` refusal that follows is loud
and a reader meeting it deserves to find the reason where this project promised
to put it.

### What moves

**`test/baseline/baseline.json`, `schema_version` 2 to 3.**

- `energy` leaves `absent_fields`. It has been there since P8 saying "P11, when
  Accelergy lands", and Accelergy has landed.
- Every cell gains `energy_pj_per_inference`, computed from that cell's own
  action counts and Accelergy's per action coefficients at 45 nm.
- The manifest gains `technology_node` and `registered_estimators`, because two
  runs at the same Accelergy sha with different estimation plug ins are two
  different measurements and the baseline has to be able to say which one it is.

**`experiments/results/*.json`, `schema_version` 1 to 2.**

- `simulation` gains `scratchpad_elements_read` and
  `scratchpad_elements_written`. These are counts the simulator has produced
  since P7 and `npu-sim` has printed since P7, and the schema did not carry them
  because until P11 nothing consumed them. Accelergy's scratchpad action counts
  are exactly these two, so recording them is what makes an energy figure
  reconstructible from a committed cell rather than only reproducible by
  re-running the simulator.
- The P11 fields stop being null: the roofline group, the SCALE-Sim group, the
  energy and area group, `technology_node`, `tool_shas` and
  `registered_estimators`. Each loses its `_null_reason` sibling in the same
  write, because the validator refuses a field carrying both and a half done fill
  is red.

### Why the two new fields rather than recomputing them

The alternative was to leave the schema alone and have the energy path re-run the
simulator for every cell it wanted a scratchpad count for. That would make the
energy figure a thing this project can produce rather than a thing a reader can
check, and law 3 of Section 0.2 is that every published number traces to a
committed file. A field that has to be recomputed to be read is not recorded.

### What the movement costs

One re-record of all 175 cells and of the baseline, in one run each. The whole
suite has to be re-recorded in a single run rather than cell by cell, because
`experiments/results_to_tex.py` refuses to generate a table from cells measured
at more than one commit, and it refuses for the right reason: a table whose rows
come from different builds is a table nobody can reproduce.

### The order these commits land in

1. This entry, in its own commit, touching `docs/BREAKING_CHANGES.md` and nothing
   else.
2. The commit that causes the movement: the schema fields, the energy path, and
   the wiring that fills them.
3. The re-record, in its own commit, touching `experiments/results/`,
   `test/baseline/` and the generated table and nothing else.

That is Section 17.6's declare then re-record, and the reason it is three commits
rather than two is that `git log` is the only thing that can tell a decision from
an explanation.

### 2026-09-05, Phase P13: checks 8 and 9 gain region scoped coverage on the DRAM side, so that a tiled result can be read

**Written before the commit that causes it.** The validator commit and the
compiler commit that follows it are the next two; this entry is what makes the
change to two **declared** checks a decision rather than an explanation.

**Why, in one paragraph.** `Program::kVersion` went to 2 so that a buffer could
be written in pieces, and the entry below promised that checks 8 and 9 would
move from "one written count per address" to "written ranges per buffer". **Only
the format half of that landed.** The validator kept the single span rule, so a
tiled result assembled in DRAM is written by one store per tile and then refused
the moment anything reads it: `operand-extent: operand 0 reads 2048 bytes from
10944 and the buffer written there ends at 11968`. D-0052 has the reproduction
and the three measurements that settle its scope. This entry is the other half
of the bump, decided by the owner, and it needs **no further version bump**
because no encoded byte moves: what changes is what the validator makes of bytes
the format already carries.

**The rule, exactly, because a declared check is worth stating precisely.**

- **The scratchpad side does not change.** Buffers there have no identity, the
  arena is one run of offsets, and the no merge rule is the only thing that can
  catch an over read that runs off the end of one buffer and into the next.
  `test/Encoding/tiled-assembly-in-scratchpad.mlir` stays a refusal and stays
  the reason.
- **The DRAM side gains region scoped coverage, and only inside a declared spill
  slot.** `program.spillSlots` carries an offset, an element type and a shape per
  slot, so each slot has its own extent and its own identity. For a read whose
  address lies inside one slot, the validator accepts when **every byte the read
  addresses lies inside that one slot** and **every one of those bytes has been
  written**. A read that reaches into the next slot is refused for leaving its
  region; a read of interior bytes no write covered is refused for reading what
  nothing wrote.
- **The bytes are computed exactly from the strides on both sides, run by run**,
  not from an element count laid down as one contiguous span. That closes the
  asymmetry D-0052 measured, where a strided tile write recorded 1024 bytes as a
  run while the matching read addressed a reach of 1920.
- **Inputs and constants stay defined whole before the first instruction, and
  outputs stay never read.** Those three region kinds are untouched.

**What moves.**

| Thing | From | To |
|---|---|---|
| ISA check 8, `operand-defined` | one written span per address, in every space | unchanged on the scratchpad; exact byte coverage inside one declared spill slot on the DRAM side |
| ISA check 9, `operand-extent` | the read's span fits the one span written at its address | unchanged on the scratchpad; every addressed byte inside one slot and covered, on the DRAM side |
| `include/NPU/Encoding/NPUISADescription.td` | the two check texts above | the texts that say which side changed |
| `docs/ISA_MANUAL.md`, `docs/ISA_OPCODES.json` | generated from the old text | regenerated, `check-isa-staleness.sh` clean |
| `-npu-tile-to-scratchpad`'s decline rule | every user of the result is `func.return` | a DRAM assembled result may be read, whole or by slices |
| `test/Dialect/NPUISA/dma-boundaries.mlir` | Section 8's count without an assembly that re-enters the scratchpad | with one, entering once |

**Which cells this predicts will move, and by how much.** Measured at the tree
that had the passes wired and the decline rule not yet written, where tiling
really fired and the encoder refused the programs:

| Cell | Prediction |
|---|---|
| `resnet_block-O2-tight-n1-fp32-normal` | one convolution tiles into two. Instructions rise from 17; the allocator's peak stays 6432 and the spill count stays 1 |
| `inception_block-O2-tight-n1-fp32-normal` | two convolutions tile into four. Instructions rise from 22; the peak stays 6144 and **the spill count is predicted to fall from 3 to 0**, because tiling is what relieves the pressure that was spilling |
| the nine other `-O2` tight ablation rows on each of those two models | move with their baselines |
| `-ablate-npu-tile-to-scratchpad` on both | **must not move**, and must still read 17 / 2018.0 / 1 and 22 / 3799.0 / 3 to the cycle. That is the gate clause and this entry does not license it to move |
| `-ablate-npu-double-buffer` on both | **not predicted to move**, because ablating it relaxes the tiling search and nothing tiles without the prefetch's contribution |
| **every default budget cell, on all seven models** | **must not move.** Nothing is over budget at the default budget, so a default budget cell that moves is a wiring defect and not this declaration |
| the other five models at either budget | must not move |

**What does not move, and this list is the point of the entry.**

- **Not one golden tensor byte.** Tiling over parallel dimensions splits no
  reduction and reassociates no `f32` sum, so the tiled program computes the same
  bytes. **This is the P13 gate's first clause becoming evidence**: until now the
  goldens were byte identical because nothing consumed the tiling interface.
- **`Program::kVersion` stays 2** and no encoded byte moves, so
  `test_binary_stability` is untouched and the corpus is not reseeded. What a
  corpus seed can do is change **verdict**, which is the declared effect and is
  listed seed by seed in the validator commit.
- **No cost model constant**, and no file under `include/NPU/Simulator`,
  `lib/CostModel` or `python/npu_frontend/cost_model.py`.
- **No bound, tolerance or threshold.** `GOLDEN_TOLERANCE` stays zero,
  `TIMING_GAP_FRACTION` stays 0.5, the coverage thresholds stay where they are,
  and ADR 0008's suite tight budgets are **not** re-measured here.

**Why the regression is worth taking.** The alternative is a compiler whose
tiling pass declines every operation it could tile, which is what P13 shipped
one commit ago and recorded as D-0052. Section 13.3's tiling arm has no subject
without this, so the phase's reason to exist is what the change buys. The
narrower alternative, relaxing the no merge rule everywhere, was considered at
D-0050 and refused: it would give up refusing a read of two exactly adjacent
buffers as one, in the space where buffers have no identity. **Scoping the
relaxation to a declared region is what makes it a completion of the version 2
decision rather than a weakening of it.**

**Outcome, written after the fact and kept apart from the predictions above,
which are not edited.** The validator half landed and is `20fc6c1`. **The
compiler half did not, so none of the cells above moved**, and the entry stands
as a declaration whose causing commit was held back rather than as one that
failed to predict.

Compiling all 168 cells at the tree that used the fix measured what the entry
did not ask about: tiling now produces a valid program everywhere except one
cell, `resnet_block` at its tight budget with `-npu-fuse-ops` ablated, where the
allocator refuses at a sweep line peak of 7456 bytes against 6464. Four other
cells improve, including a peak of 4640 against 6432 on `conv_bn_relu_stack` and
three spills removed from `inception_block`. **No rule inside the tiling pass
separates the four from the one**, because the deciding quantity is the
program's sweep line peak and the pass sees one operation. D-0056 carries the
measurement, the two rules that were tried and failed, and the three ways
forward.

**And then the compiler half did land, one checkpoint later, so the rest of this
outcome is what it moved.** D-0056's answer was that the allocator was refusing
a legal spill: its rule refused a buffer any view was taken of, and what a
reload cannot serve is only a view that is **written through**. With that rule
narrowed, the cell that would not place places, at a peak of 6144 bytes against
the 6432 the untiled program needs.

**31 cells moved and not one of them is a default budget cell**, which is the
line this entry drew and which held. The prediction above named `resnet_block`
and `inception_block` and said the other five models must not move. **That last
clause is wrong**, and the reason is one the prediction did not think through:
ablating `-npu-fuse-ops` un-hides the convolutions fusion was covering, so the
tiling pass sees them on every model, and eight further tight budget ablation
rows move with it. The eight are `conv_bn_relu_stack` ablating
`npu-fold-batchnorm` and `npu-fuse-ops`, `depthwise_separable` and `lenet` and
`lenet_batched` ablating `npu-fuse-ops`, and `dilated_stack` ablating
`npu-fuse-bias` and `npu-fuse-ops`. The prediction is not edited; this is the
adjudication.

**The direction is not one sided and that is the finding rather than the
disappointment.** `inception_block` at its tight budget goes from 3799.0 cycles
with 3 spills and 21936 DRAM bytes to **3395.0 with none and 12720**.
`resnet_block` goes from 17 instructions and 2018.0 cycles to **21 and 2660.0**.
Tiling helps one model and costs four, which is the trade Section 13.3 exists to
quantify.

**What did not move, and it is the whole list this entry promised.** No default
budget cell. No golden tensor byte. No `max_abs_error_vs_onnxruntime`. The
tiling disabled ablation cells still read 17 / 2018.0 / 1 and 22 / 3799.0 / 3 to
the cycle, which is the gate clause; what moved there is the row's delta,
because the baseline moved, which is an ablation row becoming a measurement.

### 2026-09-05, Phase P13: `Program::kVersion` goes to 2, so that a buffer can be written in pieces

**Written before the commit that causes it.** The commits that change the format
are the next ones; this entry is what makes the bump a decision rather than an
explanation. It is the first version bump this format has had.

**Why, in one paragraph.** `-npu-tile-to-scratchpad` splits an operation that
does not fit the scratchpad into tiles, and every tile writes a piece of one
buffer. The binary cannot express that, and D-0050 records the three refusals it
takes to find out: a sub region of a DRAM argument has no address the encoder can
name; `Instruction` carries `resultShape` and no `resultStrides`, so a strided
write is not representable; and ISA checks 8 and 9 ask whether a consumer's need
fits **the count written to the buffer it reads**, which assumes a buffer is
written whole by one instruction. The third is the binding one: a **contiguous**
channel tile, which needs no strides at all, is refused by the same rule.

**What moves.**

| Thing | From | To |
|---|---|---|
| `Program::kVersion` | 1 | **2** |
| the `.nbin` header's `version` word | 1 | 2, in every file this build writes |
| `Instruction` | `resultShape` only | `resultShape` **and `resultStrides`** |
| ISA check 8, `operand-defined` | one written count per address | written **ranges** per buffer |
| ISA check 9, `operand-extent` | the same | the same |
| `fuzz/corpus/*.nbin` | version 1 seeds | regenerated at version 2 |
| `FrozenConstants.TheFormatsNumbers` | asserts `kVersion == 1` | asserts 2, in the same commit |
| `docs/ISA_MANUAL.md` | "currently 1" and the generated check table | 2 and the regenerated table |

**What does not move, and this list is the point of the entry.**

- **Not one simulated number.** No cycle count, no DRAM byte count, no
  instruction count, no MAC count, no utilization, no energy or area figure. The
  version word is a header field; nothing downstream of it reads differently.
- **Not one golden tensor byte.** `test/baseline/golden` is expected to be
  untouched by every commit in this sequence, and a moved golden here would be a
  defect rather than a declared movement.
- **No cost model constant**, and no file under `include/NPU/Simulator` or
  `python/npu_frontend/cost_model.py`.
- **Nothing about P14's claim.** The format's own version policy says the
  element types are present from version one "together with `requantMultiplier`
  and `requantShift`, and those specific fields **and nothing broader** are what
  let Phase P14 land without bumping `kVersion`". Those six fields are untouched
  here. **P14 still bumps nothing**, and its gate's clause that
  `Program::kVersion` is unmoved is a statement about what P14 does, which
  remains true. What this bump changes is the number that clause is measured
  from, and the record says so here rather than leaving P14 to discover it.

**Why the baseline moves at all, and in which direction.** The only baseline
movement this sequence causes is **composition**: new tests for the new field and
for the range tracking rules, so the suite counts and the recorded test names
grow. That is not a regression and would not need this entry on its own. The
entry exists because the format's own version policy in `docs/ISA_MANUAL.md`
says a bump invalidates the seed corpus, and regenerating committed artifacts is
a deliberate movement of things this repository has promised to keep stable.

**Why the regression is worth taking.** Without it, tiling cannot be lowered at
all. Measured on a two tile convolution, the arrangement the bump enables takes
the sweep line peak from **4224 bytes to 1728**, where 4224 is the untiled
working set to the byte; without it a tiled program either does not encode or
splits instructions while leaving the peak exactly where it was. Section 13.3's
experiment, which is the reason this phase exists, has no subject in either case.

**What is deliberately not done.** The version is bumped once, to 2, and the
layout change is the single field the write model needs. No field is added
speculatively against a later phase, which is the discipline that kept the
version at 1 through six phases and is the reason P14 costs nothing.

**The order these commits land in.**

1. This entry, in its own commit, touching `docs/BREAKING_CHANGES.md` and the
   version policy prose in `docs/ISA_MANUAL.md`, and nothing else.
2. The format change: `resultStrides` written and read, checks 8 and 9 tracking
   ranges, the frozen version test moving with it in the same commit, and the
   generated artifacts regenerated so the staleness gate stays green.
3. `dramAddressOf` learning the view chain walk, which needs no format change
   and is separated from the one that does.
4. The corpus reseed and the malformed corpus extension, in their own commit.
5. The baseline re-record, in its own commit, after all of it.

That is Section 17.6's declare then re-record, and the reason the format change
and the address resolution are separate commits is that only one of them is a
format change and a reader should not have to untangle which.
