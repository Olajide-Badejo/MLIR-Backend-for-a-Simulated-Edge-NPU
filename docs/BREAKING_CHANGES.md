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
