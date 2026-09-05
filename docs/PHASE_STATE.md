<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Phase state

*Diataxis type: reference.*

Ground rule 17: this file is updated at the end of **every** session, including
a session that achieved nothing, and it carries four things: the current phase,
the status of its gate, the open questions, and the exact next command. This
build spans dozens of sessions, and reconstructing where it stood from `git log`
costs more than writing these lines did.

**Last updated:** 2026-09-04.

## Current phase

**P13, tiling, double buffering and layout. Incomplete, and this handoff says so
first.** Branch `phase/p13-tiling`, cut from `main` at `2f59429`, which is the
P12 merge. Six commits. **Not pushed.** **The gate is not met**, three of the
four deliverables have not started, and the reason the phase stopped where it did
is the first commit.

| Commit | Subject |
|---|---|
| `b2b63df` | `test(simulator): the weight preload is already charged once per fold, and D-0045 is not a defect` |
| `6d662f2` | `build(external): zigzag joins the external tools, on the terms the other two are on` |
| `f7b32ba` | `feat(dialect): the halo arithmetic P1 declined to write, over the parallel dimensions` |
| `d8532a0` | `record: the baseline at the tip, and it moves fifteen test names and nothing else` |
| `e8910f2` | `feat(passes): -npu-tile-to-scratchpad, implemented and deliberately in no level yet` |
| tip | `docs: the flake is a conditional bound, and the lowering is what tiling waits on` |

**The commit order carries meaning and the first commit is the phase.** P13 was
briefed to fix D-0045 under the full declare then re-record governance, on the
stated understanding that fixing it changes the cost model. `b2b63df` is the
measurement that says there is nothing there to fix: the charge already does what
the entry says it does not, and the entry's own reproduction quotes a cell the
committed results contradict. Everything the governance sequence would have
governed is therefore not done, deliberately, and the rest of the phase was
re planned around that.

`f7b32ba` is the deliverable that did land and it is the one the rest of the
phase blocks on: the halo arithmetic for tiling every windowed operation in the
dialect, which P1 declined to write and named the reason for. It lands **before**
the pass that consumes it, which is the ordering the phase's own commit rule
asks for.

## Gate status

**Not met.** Clause by clause, with what stands and what has not started.

| Clause | Status |
|---|---|
| Goldens byte identical for the tiling work, exactly | **stands, and it is not yet evidence.** All 21 golden tensors are byte identical and `git status` on `test/baseline/golden` is empty. Nothing consumes the tiling interface yet, so nothing could have moved them. The claim becomes evidence at the commit that adds `-npu-tile-to-scratchpad` |
| Any movement from layout or double buffering inside 1e-6, declared in `docs/BREAKING_CHANGES.md` before the causing commit | **not reached.** Neither pass exists. No entry was written, and writing one before the pass would be declaring a movement nobody has measured |
| No `scf` operation reaches the lowering, asserted by a lit test | **met for the pass, not yet for a level.** `-npu-tile-to-scratchpad` asserts it about its own function and `test/Transforms/tile-to-scratchpad.mlir` asserts it from outside with a `NOSCF` prefix. It becomes a statement about the lowering when a level runs the pass |
| A tiling disabled ablation row reproduces the previous spilling numbers to the cycle | **not started**, and the numbers it will have to reproduce are recorded below so the next session does not have to re-derive them |
| The tight budget question answered per model with all three arms of Section 13.3 | **not started.** This is the phase's reason to exist and none of the three arms has been run |
| The layout delta reported whichever way it went, with the DMA stride term shown to carry it | **not started** |
| The ZigZag comparison shown next to its prediction, compared under the same mapping | **not started.** The tool is installed, pinned, recorded and wired into the external tools policy; there is no mapping to export yet, because mappings are what the tiling pass produces |

**Two clauses of the brief around the gate are answered rather than pending, and
both are answered in the negative.** The D-0045 governance sequence does not run,
because no charge moves. The Section 2 carve out is **not** re-derived to 217
cells, because no ablatable pass landed and a re-derivation to a cell count the
suite does not have would be worse than the stale one that is at least honest
about which phase it belongs to.

## D-0048, which is the phase

**D-0045 named a mechanism the cost model does not have, and quoted a cell it
does not match.** The full entry with both reproductions is in
`docs/DEFECT_LOG.md`; D-0045 itself is marked withdrawn with its body left
exactly as P11 wrote it, because this project's audit trail is worth more than a
tidy file.

**The arithmetic.** D-0045 says `gemmCharge` computes
`delta = m / (m + kWeightPreloadCycles)` once per instruction and applies it to
every tile, so the fill is amortised across the whole layer however many times
the array is refilled. The premise is true and the conclusion does not follow.
`FrozenConstants.TheCostModelsNumbers` already asserts the f32 peak is the
array's area, so `utilization * peak` is exactly `rows * columns` for any tile
and the tile's charge reduces to `m / delta`, which is
`m + kWeightPreloadCycles`. With `T` folds the instruction is charged `T` times
that, which is the fill counted once per refill. **Applying the same fraction to
every tile is not the same operation as counting the fill once.** Verified over
343 shape combinations: the charge equals the explicit per fold accounting on all
343, and differs from the once per instruction accounting on every shape with
more than one fold by exactly `(folds - 1) * 16`.

**The reproduction, and it is a budget confusion.** D-0045 names
`resnet_block-O2-default-n1-fp32-normal`, layer `node_conv2d`, and quotes
SCALE-Sim at 1465 cycles and utilization 0.098. The committed result for that
cell and layer says 549 and 0.2623 with **zero stall cycles**, and has since P11.
1465 and 0.0983 is the same layer at the **tight** budget, where SCALE-Sim
reports **916 stall cycles**, and 1465 minus 916 is 549. Both pairs reconcile
against the same MAC count, which is why `check_the_same_arithmetic` passes on
both and the crossing was invisible.

**This project's 478 is a DMA bound figure**, not a compute one. That layer's
`analytical_compute_cycles` is 404, of which 400 is the array and 4 the issue
overhead. So the entry compared 478 cycles of mostly transfer against 1465 of
which 916 is SCALE-Sim waiting on memory.

**The stalls are a tight budget phenomenon and they are recorded here because
they are the number the next session needs.** Of the 550 layer rows in the
committed suite, **66 carry SCALE-Sim stall cycles and all 66 are tight budget
cells**; the 308 default budget rows carry none. Median divergence with stalls is
-72.42 percent against +11.59 percent without. Suite total stall: 107206 cycles.

**What that says about the decomposition, which is the transferable part.** The
stall term enters `decompose()` twice with opposite signs:
`array_fragmentation` is `analytical_compute - (matched_total - stalls)` and
`double_buffering` is `max(0, dma - compute) - stalls`. They cancel exactly in
the total. `docs/NUMBERS.md` leads on those two terms being nearly equal and
opposite at +442289 and -435825 and calls it the most useful thing the comparison
produced. It still is, and the sign structure is part of it: **the cancellation
is partly a property of how the terms are written and not only of the physics.**
Neither term is wrong; both subtract the same stalls so memory time is charged
once rather than twice, which is the double count `decompose`'s own comment
records fixing.

**What is still open is the question D-0045 was reaching for.** With the stalls
removed on SCALE-Sim's side the two tools still disagree about the compute time
of the same MAC count, by 8.65 times on `dilated_stack`'s `conv1`, 7.63 on its
`conv0`, 6.14 on `inception_block`'s 5 by 5, and 4.34 times in the **other**
direction on `inception_block`'s 1 by 1. The dilation approximation has its own
term and its own second SCALE-Sim run, so it is accounted for separately and is
not the answer. **Whatever the mechanism is, it is not the weight preload**,
because both tools charge that once per fold.

**Why the frozen constants test could not have caught it, which is the lesson.**
`FrozenConstants.TheCostModelsNumbers` pins `kWeightPreloadCycles` at 16.0 and
says nothing about where the 16 is charged, so **the accounting was never under
any assertion at all**. That is D-0047's shape one layer over: a property nobody
could observe from inside the artefact, and two readers who never compared notes.
The fix is two assertions in the P9 pattern,
`CostModel.TheWeightPreloadIsChargedOncePerFold` and
`test_the_weight_preload_is_charged_once_per_fold`, and both assert the per fold
accounting **and assert it apart from** the once per instruction accounting,
because the two agree whenever there is exactly one fold and every shape small
enough to check by hand has exactly one fold.

## What P13 has, in one place

- **The cost model is untouched.** `git diff main..HEAD` touches no file under
  `lib/Simulator/`, `include/NPU/Simulator/` or `python/npu_frontend/cost_model.py`.
  No `docs/BREAKING_CHANGES.md` entry, no band re-versioning, no re-record for a
  numeric reason.
- **The `TilingInterface` generation half is complete for the operations that
  matter**, over the parallel dimensions only: convolution, both pools, matmul.
  A tile that splits the reduction is declined, which is Section 13.2's rule
  living in the interface rather than in the pass. Transpose, concat and
  batch_norm still have the introspection half only, and none of the three is
  what a scratchpad budget runs out on.
- **The exactness property is asserted rather than described.** For every output
  position of every tile, the window touches the same input positions it touched
  untiled, and the positions lying outside the input are the same ones. The
  second half is what an average pool's divisor depends on.
  `Conv2DEveryTileReadsTheSamePositionsAsTheWhole` checks both over five window
  shapes, every tile size that divides the output, and every offset.
- **`-npu-tile-to-scratchpad` is implemented and is in no `-O` level**, which is
  the state its own commit argues for at length. It fires only when an
  operation's working set exceeds the budget, enumerates the mapping space
  exhaustively with capacity pruning, scores on Section 5.5's two port makespan
  through the simulator's own `gemmCharge` and `dmaCycles`, records the chosen
  mapping on every tile as `npu.tiling_choice`, declines rather than splitting an
  fp32 reduction, and leaves no `scf` behind. Six lit cases in the P9 pattern
  across two files, including Section 13.3's halo boolean as a measured contrast
  rather than a claim: `halo=cache` tiles four ways at a 2048 byte budget and
  declines at 768, where `halo=recompute` splits the rows and fits in 720.
- **The cost model has its own library now and did not change.**
  `lib/CostModel`, built as `NPUCostModel`, linking nothing. Section 5.5 requires
  the tiling pass to score against the one home, and before the split the only
  way to reach `gemmCharge` was to link the whole simulator into `npu-opt`.
- **ZigZag is installed, pinned and wired**, and unused. `zigzag-dse` 3.8.5,
  imports as `zigzag`, four seconds of wall clock, nothing in
  `requirements-lock.txt` moved. Recorded in ADR 0003 by version rather than by
  git sha, which is the exception that document already carves out for a package
  index install.
- **The suite is still 175 cells and the ablatable set is still 8.** Nothing in
  this branch puts a pass into an `-O` level.

## What tiling is waiting on, which is the next piece and is two changes

**`-npu-lower-to-npuisa` has no pattern for `tensor.extract_slice` or
`tensor.insert_slice`**, which is what a tiled program is stitched from, so a
tiled function does not lower at all. `npu-opt --npu-tile-to-scratchpad
--npu-lower-to-npuisa` reports `failed to legalize unresolved materialization`
on the destination argument. That is the first change and it is the smaller one.

**The second decides whether tiling is worth anything, and it is now a
measurement rather than an argument.** The conversion today loads each DRAM
function argument into the scratchpad **once, whole**, and records it in
`LoweringState::argumentBuffers` so every consumer reads the same resident
buffer. Under that arrangement a tiled program's slices are views of buffers
that are **already resident**. Measured by writing the same two tile convolution
both ways by hand and running `-npu-allocate-scratchpad` over each:

| Arrangement | Sweep line peak |
|---|---|
| whole arguments resident, tiles as views of them | **4224 bytes** |
| the slice is what enters the scratchpad | **1728 bytes** |

**4224 is the untiled working set to the byte**, which is the whole point: under
the first arrangement the allocator sees the same three buffers it always saw and
the peak cannot move, so tiling would split one compute instruction into several
and report instructions moving, cycles moving and pressure standing still. Under
the second, each tile's operands are their own allocations and are dead after the
tile that used them, so the second tile reuses the first tile's offsets and the
peak is one tile's working set rather than the whole layer's. **A 2.44 times
reduction on a two tile example**, and it is the arrangement rather than the tile
count that decides it.

So **the slice of a DRAM value has to be what enters the scratchpad**. That is
consistent with Section 8, which counts one `dma_load` per DRAM value entering
the scratchpad and under tiling the values are the slices rather than the whole
arguments, and Section 8 already names tiling as one of exactly three permitted
DMA producers. It is a real change to the conversion and to
`test/Dialect/NPUISA/dma-boundaries.mlir`, which pins that invariant at exactly
that point.

**There is a third thing in the way and it was found by trying it rather than by
reading.** A strided `dma_load` whose source is a `memref.subview` of a DRAM
argument **parses, verifies and allocates cleanly**, and then `npu-translate`
refuses it:

```
error: this DRAM buffer has no address in the DRAM map. The map holds the
function's arguments, the npuisa.const results, and the allocator's
npuisa.spill_slot allocations, and nothing else may live off chip
```

**That question is now answered and the answer is the second one.** The address
lookup is the small part and it is not the blocker. **The binary format cannot
express a buffer written in pieces at all**, which is what a tiled program does
by construction, and the full account with all four refusals is **D-0050**.

Three facts, in the order they bite:

- `Operand` already carries `address`, `shape` **and a stride per dimension**, so
  a DRAM sub region is `base + byteOffset` with the parent's strides and needs no
  new field. Teaching `dramAddressOf` the view chain walk that
  `scratchpadAddressOf` already does through `npuisa::computeBufferRange` is the
  authorised change, and it is necessary but nowhere near sufficient.
- `Instruction` has `resultShape` and **no `resultStrides`**. `setResult` builds a
  full `Operand`, strides included, and copies four of its five fields. Every
  spatially tiled convolution needs the fifth.
- **The validation model assumes a buffer is written whole by one instruction.**
  ISA checks 8 and 9, `operand-defined` and `operand-extent`, ask whether a
  consumer's need fits the count written to the buffer it reads. **A contiguous
  channel tile is refused by that rule too**, which is the measurement that
  settles the scope: the blocker is the write model and not the layout.

**P13 stopped here rather than proceeding**, because the fix needs a
`Program::kVersion` bump: it reseeds the fuzz corpus, re-records the binary
stability test, and moves the baseline P14's gate is written around, in the phase
immediately before it. Checks 8 and 9 are **declared** ISA checks besides,
mirrored into `docs/ISA_MANUAL.md` and `docs/ISA_OPCODES.json` and kept in step
by `check-isa-staleness.sh`, so changing what they mean changes the declared ISA
rather than an implementation detail. That is an owner decision rather than a
phase's. **Nothing miscompiles in the meantime**: the encoder's validator refuses
and says so.

**The lowering patterns were deliberately not written.** A compiler that emits
programs its own encoder refuses is a worse state than one that does not emit
them, and the tiling pass being in no `-O` level already keeps the suite green.

## Where a tiled result is assembled, and why it is DRAM

**Decided. A tiled operation's result is assembled in DRAM, not in the
scratchpad, and Stage B is deleted from the plan.** Checks 8 and 9 keep their
present meaning, `WrittenSpans` keeps its no merge rule and the comment that
explains it, and nothing about the declared ISA moves.

**The fork.** A tiled program's assembled intermediate would be N disjoint
writes covering one scratchpad buffer, then a read of all of it. Checks 8 and 9
refuse that, by design rather than by accident: `WrittenSpans` records one span
per written address and deliberately does not merge adjacent ones, because
merging two adjacent buffers into one range would let an over read that runs off
the end of the first and into the second pass validation, "which is precisely
the case the rule exists to catch". **A tiled assembly and that over read are
not distinguishable by addresses**, and addresses are all a validator has: the
binary carries region identity for DRAM, in the declared input, output, constant
and spill regions, and none at all for the scratchpad, which is one arena of
offsets.

The relaxation that would have permitted it was found and is recorded so the
option is legible rather than forgotten: a read is satisfied when every byte is
covered **and no covering write extends beyond the read's end**. It accepts the
tiled assembly and still refuses a read past what was written, and what it gives
up is refusing a read of two exactly adjacent buffers as one.

**Why the other branch was taken.** Four reasons and the last is about who may
decide.

- **Tiling exists because the value does not fit on chip.** A result that had to
  be split has no business being reassembled in the memory it did not fit in.
- **The DRAM round trip is a cost the program already accepted** when it tiled.
  A value that did not fit was going to make that trip.
- **Per slice stores mirror the per slice loads** the authorised convention
  already established. One rule in both directions is easier to hold than one
  rule and an exception.
- **It keeps a declared check where the other branch spends one.** Every byte in
  the refused read was written by the program itself, so the relaxation would
  have cost a diagnostic rather than opened a memory safety hole; it would still
  have been a weakening of semantics the ISA description declares and
  `docs/ISA_MANUAL.md` and `docs/ISA_OPCODES.json` mirror. **Declining to change
  spec declared semantics is not the same class of decision as changing them**,
  which is why this fork was decidable where the `kVersion` bump was not.

**What follows for the lowering.** Tiles are stored to DRAM as they are produced
and the next layer loads the slices it needs, in both directions, which is the
per slice convention applied symmetrically. Nothing is ever written in pieces and
read whole, so no scratchpad buffer is ever assembled from tiles.

**And it lands where the experiment wants it.** Section 13.3 measures spilling
against tiling against recompute, and the DRAM traffic a tiled assembly makes is
exactly the quantity those arms compare. Putting the assembly in DRAM does not
hide a cost from the experiment; it puts the cost in the column the experiment
reads.

`test/Encoding/tiled-assembly-in-scratchpad.mlir` is the negative test that keeps
this a decision. It is the refused program, asserting the refusal by name, so
that a later reader finds out that a scratchpad assembled tiled result is
rejected on purpose rather than discovering it as a puzzle.

## The numbers the next session needs and should not re-derive

**The tight budget spilling numbers a tiling disabled ablation row has to
reproduce to the cycle**, read out of the committed cells at `-O2`:

| Model | instructions, default to tight | cycles, default to tight | spills, buffers and DMA |
|---|---|---|---|
| `lenet` | 25 to 25 | 17766.25 to 17766.25 | 0, 0 |
| `depthwise_separable` | 12 to 12 | 1324 to 1324 | 0, 0 |
| `resnet_block` | **14 to 17** | **1626 to 2018.0** | **1, 3** |
| `inception_block` | **14 to 22** | **2398.5 to 3799.0** | **3, 8** |
| `conv_bn_relu_stack` | 15 to 15 | 1160.5 to 1160.5 | 0, 0 |
| `dilated_stack` | 12 to 12 | 1234.0625 to 1234.0625 | 0, 0 |
| `lenet_batched` | 25 to 25 | 20000 to 20000 | 0, 0 |

**Only two models spill at their tight budget**, which is ADR 0008's own finding
and is the whole population the three arm experiment has to say something about
at a budget below the peak. ADR 0010 records that the remaining five do not
allocate at batch 4 under their recorded tight budget at all, and names tiling as
the remedy, so P13 is also the phase that can make those six cells exist.

**Where the counts are hardcoded and will have to move together** when three
ablatable passes land and the suite goes from 175 to 217:
`test/Python/test_benchmarks.py` at the assertions for 8, 63, 112, 175 and the
`checked == 112` and `cells_total == 175` ones; `docs/NUMBERS.md`'s cell count
table; `report/generated/macros.tex` through `results_to_tex.py`;
`test/Pipeline/opt-levels.mlir`'s flat `-O2` pass argument list and its
`DESCRIBE` sequence; `test/Pipeline/pass-stats.mlir`'s ordered name list;
`docs/PASSES.md`'s pass table and its "eight ablatable" paragraph. **No
arithmetic in `run_benchmarks.py` changes**: it reads the ablatable set out of
the driver at run time and the 63 and 112 are computed, never written down.

## The Section 2 carve out, for the owner

**Unchanged from P12 and repeated because it is still open.** The measured figure
is **1.17 seconds per cell**, over 175 cells in 3.43 minutes, serially, on the
14700K under WSL2. Section 2 still says 15 seconds and 238 cells while the
repository says 1.17 and 175, and a reader comparing them finds a disagreement
rather than a correction. The suggested replacement paragraph is in P12's handoff
below and it is unchanged.

**The ablatable set is still 8 and the suite is still 175 cells.** P12's handoff
said the arithmetic needs re-deriving to 11 and 217 in the commit that adds the
three passes rather than after it. That commit has not happened, so the
re-derivation has not either, and doing it in advance would put a cell count in a
document that the suite would contradict.

## Verification output

Every command run at the tip of this branch, from `/home/elijah/npu-mlir-v2`, in
`~/npu-venv`.

**Two whole categories were deliberately not run, and the reason is the same for
both.** `run_benchmarks.py`, `roofline.py`, `scalesim_export.py`,
`accelergy_energy.py`, `kernel_threads.py` and `compile_time_benchmark.py` all
measure the compiled suite, and **this branch changes nothing the suite
measures**: no `-O` level gained a pass, no charge moved, the cell count is P12's
175 and the committed results are P12's. Re-running them would produce P12's
numbers with a new timestamp and would spend the one thing a measurement of a
load sensitive bound cannot get back, which is a quiet machine.
`regression-baseline --check` is the gate that would have caught it if that were
wrong, and it reports no drift over 42 cells and 21 golden tensors.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build-ndebug -j6` | clean, no warnings |
| `ninja -C build check-npu` | 26 discovered, 26 passed |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | **20 passed**. 12 at P12, plus the eight tiled implementation tests |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | **56 passed**, 1 skipped. 55 at P12, plus the per fold assertion |
| `build-ndebug/bin/NPUSimulatorTests` | 56 passed, 1 skipped |
| `build-ndebug/bin/NPUEncodingTests` | 76 passed, 1 skipped |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **1082 passed, 18 skipped**. 1076 at P12, plus the six parametrized cases of the mirror's per fold assertion |
| `mypy` | no issues found in 26 source files |
| `black --check .` | 66 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 478 of 478 files |
| `pre-commit run` | all twelve hooks passed, on every commit of this branch |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date |
| `python scripts/patch-scalesim.py --check` | every edit in place, exit 0 |
| `bash scripts/regression-baseline.sh --check` | **no drift**, 42 cells, 21 golden tensors, exit 0 |
| `bash scripts/coverage.sh 85 93 16 58` | C++ **86.1** PASS against 85, branch 75.7; per tree **93.4313 / 16.1191 / 74.5156** PASS, exit 0 |
| the whole suite in the CI shape, now four differences rather than three | **1069 passed, 31 skipped, 0 failed**, mypy clean under `--python-executable /usr/bin/python3` |
| `regression-baseline --check` in the CI shape | **no drift**, with both environments named, the count difference printed, and three oracle distances reported as inside D-0039's band rather than as silence |
| the same environment with `NPU_EXTERNAL_TOOLS=1` | the guards **fail** naming the variable rather than skipping, and the message now lists **all three** tools |
| `git status --short` | empty |
| `git log -p main..HEAD` grepped for tooling and authorship traces | 0 matches, case insensitive with word boundaries |
| the same diff grepped for em and en dashes | 0 matches |
| `git diff main..HEAD` over `lib/Simulator`, `include/NPU/Simulator` and the Python mirror | **empty**, which is this branch's claim about the cost model rather than a statement about it |

**The CI shape has a fourth difference from P13 and the recipe above carries
it.** `zigzag` is a third module the image does not have, so the shim's meta path
finder blocks three roots rather than two. The prediction for the suite row was
written before the run: 1069 passed and 31 skipped, which is P12's 1063 and 31
plus the six new mirror cases, since those need no external tool and run in both
shapes and the one renamed test moves neither count. **The run measured exactly
that.**

**The branch is 1349 insertions and 38 deletions over 11 files**, of which 448
are the tiled implementations and 498 are the tests that check them.

**C++ coverage moved from 86.4 to 86.1 and branch from 76.8 to 75.7**, both still
above the 85 the gate checks, and the margin is now 1.1 points rather than 1.4.
It is named here rather than left as an unexplained dip. The 448 new lines are
mostly covered, and what is not is the decline paths: `constantsOf` returning
nothing on a dynamic offset, and `windowSlice` refusing a tile whose every window
lies in the padding. **Neither is reachable from a test**, because every
iteration domain in this dialect is static and the verifier's rule that a pad is
smaller than its kernel makes the second impossible for a well formed operation.
They are guards against a caller this dialect does not currently have, and the
alternative to an uncovered guard there is an unguarded assumption.

## What P12 measured, and still holds

`docs/NUMBERS.md` is the ledger. **Every figure below is P12's and nothing on
this branch re-measured any of them**, because no ablatable pass landed and the
suite is the same 175 cells. Five things worth repeating here.

- **The suite is 1.17 seconds per cell, 3.43 minutes for 175 cells**, against 90.
  It was 1.27 at P11 with the same tools inside the same suite, and the
  difference is the kernel. The factor in hand went from twenty four to twenty
  six. **This is a host wall clock and nothing else.**
- **Thread scaling is 0.86 to 3.17 times, geometric mean 2.10**, byte identical
  at every thread count on every model. `depthwise_separable` is the row below
  one and it stays reported: 12800 multiply accumulates inside a process that
  takes longer than that to start has no arithmetic left for a thread to win.
- **The fitted exponent is 1.1038 against an n log n reference of 1.1365 at the
  same sizes**, so the curve is **below** n log n rather than merely near it.
- **The inertness proof is a diff and not a claim.** 95614 leaf fields over 175
  cells, seven movers, all wall clock or provenance, zero forbidden. 21 golden
  tensors byte identical. That claim would have been true at P11 for the
  uninteresting reason that the kernel was serial at both ends; it is worth
  something here because the kernel really went parallel in between.
- **The worst `--mlir-timing` gap was 0.1856 ms against D-0043's 0.2000 bound**,
  measured with nothing else on the machine. P11's two quiet runs measured 0.1577
  and 0.1177. Inside the bound, and closer to it than either. Recorded as an
  observation and not as a trend, because one run is not one.


## D-0047, which was P12's phase

**The convolution kernel was never compiled with OpenMP, in any build, in any
environment, from P7 to P12.** `lib/Simulator/CMakeLists.txt` had one line of
OpenMP wiring, `target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)`.
`add_mlir_library` compiles that library's sources in an object library called
`obj.NPUSimulator`, so a usage requirement attached to `NPUSimulator` reaches
everything that **links** it and never reaches what it is **made of**.
`-fopenmp` landed on `npu-sim`, on `NPUSimulatorTests`, on every consumer, and on
none of the kernels.

**The determinism test is what it cost.** `DeterminismTest.cpp` links
NPUSimulator, so it did receive `-fopenmp`, so its `_OPENMP` was defined, so it
printed a thread count of 28, called `omp_set_num_threads(1)` and then
`omp_set_num_threads(28)`, and compared two single threaded runs. It passed for
three phases while asserting nothing, under a comment header saying in as many
words that a test which silently becomes vacuous is worse than no test. **This
file said at P11 that the determinism assertion "asserts at full strength
everywhere now".** It did not, and it never had.

**How it was found:** by measuring before changing anything. Seven models, 1
thread against 28, and a table of speedups between 0.98 and 1.03. That is exactly
what small models look like and these models are small, so the reading was
available and it was wrong. `/usr/bin/time -v` reporting **98 percent of one CPU**
at `OMP_NUM_THREADS=28` is what separated the two, and `nm` on the object file
settled it.

**The fix is three parts and the third is the one that matters.**
`add_compile_options` at directory scope, where `-Werror=switch` already lives
for the same reason; `nbin::kernelsUseOpenMP()` and `nbin::kernelThreadCount()`
defined in `Kernels.cpp` and reachable from a command line as `npu-sim
--kernel-info`; and `Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`, which
compares this test's `_OPENMP` against the kernels' own answer.

**A second fault the fix exposed.** With the region finally compiled, an uncapped
team made five of the seven models **slower than serial** at 28 threads, by as
much as seven times on `depthwise_separable`, whose depthwise convolutions have
eight and sixteen output channels and were being handed twenty eight threads
each. The kernel caps its team at `batch * outputChannels`, the number of
independent output tiles the instruction has. That carries no tuned constant and
cannot move a bit, because neither the cap nor the `if` clause changes which
iterations exist, what one computes, or the order of the reductions inside it.


## What P11 measured, and still holds

`docs/NUMBERS.md` is the ledger and is the file to read. Six things worth
repeating here. **The suite runtime figures in this section are P11's and are
superseded**; every number that is a property of the design rather than of the
host is unchanged, which is P12's whole claim.

- **The roofline cannot fail against this cost model**, and that is the phase's
  most useful negative result. `effective_macs` is defined as `cycles * peak`, so
  the compute branch is the kernel's own cycle count; a transfer costs bytes over
  bandwidth **plus** a descriptor, so it always exceeds the memory branch its own
  bytes produce. All 175 cells and all 550 layers are at or above their bound and
  none of that is evidence. It is a regression bound waiting for P13, and both
  halves of the tautology are asserted so the day either stops holding a test
  says so.
- **The divergence prediction was mostly wrong and is answered as written.**
  Direction wrong on five of seven models, all three magnitude bands wrong, both
  rank fidelity figures wrong, the coverage floor on `lenet` wrong. Right about
  the mechanism behind the widest positive gaps, about pooling, and about there
  being a fragmentation disagreement. 340 of 550 layers exceed 25 percent where
  it predicted none would.
- **The root cause of that band is D-0045 and it is a real finding about this
  cost model.** The array's weight preload is charged once per instruction here
  and per fold by SCALE-Sim, worth about a factor of three on a narrow deep
  convolution. Not fixed at P11, because retuning a model against an external
  tool invalidates every ablation already recorded.
- **The suite wide divergence headline is the small remainder of two large
  opposite terms**: double buffering at plus 442289 cycles and array
  fragmentation at minus 435825, summing to plus 109756. Quoting the headline
  without them would be quoting an accident.
- **Energy runs from 1.7 uJ per inference on `depthwise_separable` to 54.4 uJ on
  `lenet`, on an 8.5 mm2 design at 45 nm.** The fp32 MAC coefficient **fails**
  Section 16.4's order of magnitude sanity check at a factor of 10.71, for an
  identified reason that is not this project, and `docs/NUMBERS.md` records that
  at the published coefficient the scratchpad would be the largest consumer on
  every model. **No conclusion here rests on the array being dominant.**
- **Fusion moves exactly zero picojoules**, on all seven models, which is the
  same zero the P10 ablation table records for instructions and cycles arriving
  in the currency fusion is usually argued in. What it would be worth where the
  intermediate spilled is quantified beside it.

### What P10 measured, and still holds

- **175 cells, 1.76 minutes, 0.60 seconds per cell**, against a 90 minute budget.
- **Two of the eight ablatable passes have a nonzero row.**
  `-npu-fold-batchnorm` saves 8 instructions and 212 cycles on
  `conv_bn_relu_stack`; `-npu-fuse-bias` saves 1 instruction and 9.625 cycles on
  `dilated_stack`. Both agree with the P9 measurements taken one pass at a time,
  arriving at the same numbers from the opposite direction.
- **`-O1` is exactly `-O0` on all seven models**, which P9 asked P10's report to
  state out loud. The reason is in the ledger: no model has an elementwise
  operation with two constant operands, so the folder has nothing to fold and the
  canonicalization after it has nothing dead to clean up.
- **`-sccp`'s row is zero for a structural reason**, one function and no calls,
  and that is a different kind of zero from `-canonicalize`'s. The next section is
  the phase's most interesting result.

### The finding: `-canonicalize`'s zero row is a limit of the method

The registered prediction expected `-canonicalize` to be one of three nonzero
rows. It is zero on all seven models, and the instrumentation says why:

```
with canonicalize            without canonicalize
  npu-fuse-ops    34 -> 38     npu-fuse-ops    34 -> 38
  canonicalize    38 -> 24
  cse             24 -> 21     cse             38 -> 21
```

The canonicalization removes fourteen operations, so it is not idle. `-cse`
reaches the same twenty one without it, because MLIR's CSE erases trivially dead
operations as it walks. **A leave one out ablation cannot see a pass whose work
another pass would have done.** Recorded in `docs/PASSES.md` and
`docs/NUMBERS.md`, because a table of deltas with no prose beside it would report
this identically to `-sccp`'s zero and the two have nothing in common.

## What reproducibility means in this project

P9b's handoff asked P10's report to be the first document that says this. It has
two halves and they have different answers.

**The compiler and the simulator are bit stable, at a tolerance of zero.** A
baseline recorded under gcc on WSL2 reproduces bit for bit under clang in the CI
container: every cell field, all 21 golden tensors, every suite count, over four
CI runs across at least two runner hardware generations. `GOLDEN_TOLERANCE` is
zero on evidence. P10 adds to that from a different direction: this phase put an
instrumentation on the pass manager, added a schema, a harness and ninety tests,
and the re-recorded baseline moved not one cell field and not one golden byte.
The diff of `5401d39` is that claim's proof.

**The distance to the oracle is a property of the measuring host, and is bounded
rather than fixed.** `max_abs_error_vs_onnxruntime` has two ends and only one
belongs to this project; `onnxruntime` dispatches its CPU kernels on what the
host supports, and eighteen cells moved between 1e-8 and 1e-7 in both directions
across runner hardware. It is compared against Section 17.4's band, imported from
`npu_frontend.tolerances`, and never for equality. That is D-0039.

**The two halves are why the result schema separates counted metrics from timed
ones.** `instruction_count` is exact and reproduces anywhere; a wall clock is a
measurement of one host and carries an interval saying so. Section 16.1's
determinism test asserts a re-run is byte identical **apart from the timestamp
and the timing object**, which is the same distinction made executable.

## Activation proofs and rehearsal recipes

### What P13 activates, which is nothing, and the two triggers that did not fire

**This branch activates no CI job and no CI step.** It widens one existing step's
assertion, which is not the same thing: the `external cross validation` step
still prints that it is off and still asserts the tools are absent, and what
changed is that it now asserts three rather than one.

**That widening was rehearsed four ways with the prediction written first**, and
the fourth is the one that made it worth doing.

| Shape | Predicted | Result |
|---|---|---|
| this machine, all three modules present | exits 1 naming `scalesim`, the first the loop reaches | exactly that |
| the CI shape, all three absent | exits 0, "confirmed absent" | exactly that |
| only ZigZag present | exits 1 naming `zigzag` | exactly that |
| the **old** step body, against that same shape | it names only `scalesim`, so it cannot see this one | **it printed "confirmed absent" in an image that has ZigZag in it** |

The last row is the silence Section 19.0 forbids, arriving through the one tool
the assertion did not name. It is not hypothetical: `zigzag` is in
`EXTERNAL_TOOLS` now, so `missing_tools` reports it and `require_external_tools`
would fail on it, and a step that told the log the tools were absent while one of
them was importable would be wrong in the direction that matters.

> **Trigger, carried forward unchanged: the commit that adds
> `-npu-tile-to-scratchpad`.** P12 recorded the trigger for wiring
> `experiments/compile_time_benchmark.py --check` into `ci.yml` as "P13, because
> tiling makes functions longer, which moves the crossover with the genuinely
> quadratic offset assignment scan toward the measured range". **Tiling has not
> landed on this branch, so functions are not longer**, and switching the gate on
> now would gate on a curve nothing has moved. Rehearse it red first with
> `--sizes 500`, which is the branch that has no fit and exits 1 naming what to
> do about it; then switch it on in the `build-and-test` job, under
> `pull_request` and `push` to `phase/**` like every other step, positioned after
> `check-reachability full` and before `regression-baseline --check`, which is
> documented as running last because it needs everything the job has.

> **Trigger, also unfired: `experiments/kernel_threads.py` into `nightly.yml`.**
> P12 recorded it as "the first phase that changes the convolution kernel's loop
> nest, which is P13's tiling or P14's integer kernels". **This branch does not
> touch the kernel**, so the table it would measure is P12's. It fires with the
> same commit as the one above, because a tiled convolution has a smaller output
> tile per instruction and more instructions, and the kernel's team cap is
> `batch * outputChannels`.

### 0. Reproducing the CI image locally, which is now a standing recipe

*Added at P11 after D-0046.* This project has to stay green in **two**
environments, and the second cannot be reached by running the suite here. The
recipe below is what CI's image looks like from this machine, and run
33707070166 is the evidence that getting it slightly wrong is worth catching:
the first version modelled two of the three differences and predicted a suite
row two tests off.

Three things, and all three are needed:

1. **`import scalesim`, `import accelergy` and `import zigzag` must fail.** A
   `sitecustomize.py` on `PYTHONPATH` installing a meta path finder that raises
   `ModuleNotFoundError` for those three roots. **`zigzag` is the third from
   P13**, and a shim that blocked only the first two would model an image the
   CI one is not, in the direction that makes the suite look smaller than it
   is.
2. **The `accelergy` binary must not be on `PATH`**, and everything else must
   be. The `PATH` is the venv's `bin` **minus** those entry points, not a `PATH`
   without the venv: the image has `gcovr`, `pytest` and `python3`, and the
   first shim dropped them and died on a missing `gcovr` rather than on anything
   real. Symlink each entry except `accelergy*` and `scalesim*`, and write
   `python`, `python3` and `python3.14` as `exec` wrappers rather than symlinks,
   because a symlinked venv interpreter resolves its prefix from the link's own
   directory and then finds no site-packages.
3. **The pinned SCALE-Sim source clone must not exist.** Point
   `NPU_SCALESIM_SOURCE` at a path that is not there. `~/npu-external/` is a
   developer machine artefact, and the two tests that read the example CSVs out
   of it never import `scalesim`, so 1 and 2 leave them running. **Those are the
   two tests run 33707070166 found.**

`NPU_EXTERNAL_TOOLS` must also be unset, so the guards take their skip branch
rather than their fail branch.

**What it does not model, stated so nobody assumes otherwise**: a different libc,
CPU, compiler or container uid. Those move golden tensors and timings rather than
which tests run, and `--check` has its own bands for them.

**mypy needs a separate reproduction**, because it resolves imports statically
rather than at run time and no meta path finder reaches it.
`mypy --python-executable /usr/bin/python3` is what makes it see what CI sees.

The recipe predicts CI's suite row exactly: **996 passed, 31 skipped** at
`1e77083`, which is what run 33707070166 reported. **At P12's tip it predicts
1063 passed, 31 skipped**, and the shim was re-run at that tip to get it.

### What P12 activates, which is nothing, and the two triggers that go with it

**This branch activates no CI step and no CI job.** That is a decision rather
than an omission and it is recorded with the trigger for reversing it, per
Section 19.1.

**`experiments/compile_time_benchmark.py --check` is not wired in.** It is a wall
clock measurement and the runner pool is heterogeneous. A fitted exponent is a
slope taken within one run on one host, so it is not the forbidden comparison of
a wall clock across hosts and gating it would be defensible. It is still not
being switched on, because a four vCPU shared runner measuring a five millisecond
pass at the smallest size has a noise floor this machine does not, and Section
19.0's rule cuts both ways: silence and success must not look alike, and a red
nobody believes is a red nobody reads. The P12 gate asks for the exponent to be
**reported** and consistent, and it is, in the matrix above.

> **Trigger: P13.** Tiling makes functions longer, which moves the crossover with
> the genuinely quadratic offset assignment scan toward the measured range, and
> that is the phase where this curve starts being able to catch something. The P5
> prediction entry already says so in those words. Switch it on under
> `pull_request` and `push` to `phase/**` like every other step in `ci.yml`, and
> rehearse it red first with `--sizes 500`, which is the branch that has no fit
> and exits 1 naming what to do about it.

**`experiments/kernel_threads.py` is not wired in either, and for a different
reason.** Its gate is the byte comparison, and the byte comparison already runs
in CI as `Determinism.OneThreadAndMaxThreadsAgreeBitwise`, in process, on a
synthetic convolution, **at full strength for the first time** now that the
kernels compile with OpenMP. What the script adds over that is the seven real
models, which is a nightly's worth of value rather than a per push step's.

> **Trigger: the first phase that changes the convolution kernel's loop nest**,
> which is P13's tiling or P14's integer kernels. Add it to `nightly.yml` beside
> `full-matrix`, not to `ci.yml`.

### 0b. The two faults this branch injected, with predictions written first

Both new checks were driven to their failure branches. Neither is a CI
activation, so neither needs a trigger; both are here because a gate nobody has
seen fail is a gate nobody knows works.

**Fault A: D-0047's own cause, reintroduced.** *Predicted:* removing the two new
lines from `lib/Simulator/CMakeLists.txt` turns
`Determinism.TheKernelsAgreeWithThisTestAboutOpenMP` red naming the object
library, and leaves the two beneath it **green**, because green is what they were
in exactly this state for three phases. *Result:* exactly that.
`npu-sim --kernel-info` printed `kernel openmp: no` in the same tree, and the
surviving test printed `OpenMP is on and reports 28 threads available, and the
kernels report 1`, which is the entire defect in one line, printed by the test
that could not see it. Restored, tree clean.

**Fault B: the reduction moved into the parallel region**, which is the mistake
Section 10.3 forbids by name: the outer pragma removed and
`#pragma omp parallel for reduction(+ : accumulator)` put on the input channel
loop. *Predicted:* the bytes move, `kernel_threads.py` prints DIFFER and exits 1,
and `Determinism.OneThreadAndMaxThreadsAgreeBitwise` goes red in the same tree.
*Result:* **DIFFER on all seven models**, exit code **1** confirmed directly
rather than through a pipe, and the C++ test red. Both gates see it, which is
what a second gate is for. Restored, tree clean.

**The ceiling branch of `--check` is driven in pytest rather than on the command
line**, and the reason is worth stating: making the real allocator quadratic is
not a fault injection, it is a different program.
`test_the_ceiling_separates_the_two_hypotheses_13_1_names` hands the same
function a synthetic quadratic curve and asserts it fails, and a synthetic
n log n curve and asserts it passes. That test runs in every CI job that runs
pytest, so the **discrimination** is checked everywhere even though the
**measurement** is not.

### The activations P11 carried

**That branch activated one CI step and one CI job.** Both were rehearsed under
their own step scripts with the prediction written first.

### 1. `pytest slow cells` in `ci.yml`

**Prediction.** Two tests carry the `slow` marker, the two P10 adds, and the full
run is green.

**Result.** **Nine.** The prediction was wrong and finding out why is D-0040:
seven `test_every_model_imports_at_a_second_seed` cases have been marked `slow`
since the model suite landed, and CI has never run one of them, because the only
step that runs slow tests is this one and this one was off. The suite runs green
at 957 passed, 18 skipped.

**Fault, and it is one only this step can catch.** Every `@pytest.mark.slow`
removed. The step's count guard reports zero and exits 1 with the message saying
a zero count means the step is a second copy of the one above it. Restored, tree
clean.

**Which trigger this needs:** none of its own. It runs under `push` to `phase/**`
and under `pull_request` like every other step in the file.

### 2. The `full-matrix` job in `nightly.yml`

**Prediction.** 175 cells, roughly 0.6 seconds each, inside the budget, exit 0,
and `experiments/results/` untouched because the job writes to `RUNNER_TEMP`.

**Result.** Exactly that: 175 cells, 1.84 minutes, 0.63 seconds per cell, exit 0,
`git status` on `experiments/results` empty.

**Fault: the budget gate driven to its failure branch.** `--budget-minutes 0` is
exceeded by any run at all. The harness prints the measured runtime against the
budget and exits 1; the step turns that into an error naming the two readings a
red run can have, a suite over budget or an ablation that moved the numerics.

**Which trigger this needs:** the job is new and runs on a schedule, so proving
it wants `gh workflow run nightly.yml --ref phase/p10-measurement` rather than a
wait until 03:30 UTC.

### 3. The shallow checkout, which is D-0041 and was not rehearsed in advance

**This one was found by CI rather than by a rehearsal, and that is the honest
label.** The two activations above were rehearsed under their own step scripts
with predictions written first, and neither rehearsal could have caught this,
because both ran in this repository with its history present. The variable was
the checkout itself, and nothing local varies it.

Rehearsed now, as four real fetches from a bare mirror rather than as mocks,
because what is under test is what git does. `push` and the `pull_request` merge
ref, each at depth 1 and depth 0. At depth 0 both are green and the harness exits
0; at depth 1 both refuse by name. The merge ref matters separately because its
`HEAD~1` is the base branch, so the ancestor assertions could plausibly have
broken there and do not.

**Which trigger this needs:** none of its own, and that is the point. The fix is
in the checkout step of three jobs, so the next push exercises it, and a green
`pytest` arm is the proof.

### 4. The dubious ownership refusal, D-0042, in the real image

Also found by CI rather than by a rehearsal, for the same reason as D-0041: the
variable was who owns the workspace, and nothing local varies it.

Rehearsed now in the pinned image, workspace chowned to uid 1001 and the
container running as root, which is the runner's shape. Every helper, before and
after the `safe.directory` step:

| | without `safe.directory` | with it |
|---|---|---|
| `repository_is_shallow` | refuses, quoting git's fatal | `False` |
| `commit_exists(present)` | refuses | `True` |
| `commit_exists(absent)` | refuses | `False` |
| `is_ancestor` | refuses | `True` |
| `head_sha` | refuses | the sha |
| `landing_sha` | refuses | `f92de427d1f3` |

**The left column is the point.** Before the fix those same six calls returned
`False`, `False`, `False`, "genuinely absent", `""` and `None`, which is six
wrong answers and is what CI reported. The code half of the fix stands on its
own: even with the environment still broken, nothing is answered wrongly.

**Which trigger this needs:** none. The `safe.directory` step runs in every job
that has it, so the next push exercises it.

### 5. The two clocks' quantum, D-0043, under the coverage build

Found by CI's coverage job, which is the one build where the fault is frequent
enough to be seen: gcov makes each pass slower and noisier, so the sum of eleven
rounded figures crosses zero often.

Rehearsed against `build-coverage` locally, ten runs of one cell:

```
run 0: mlir  4.1000  instr 3.978771  shortfall -0.121229
run 2: mlir  5.5000  instr 5.297579  shortfall -0.202421
run 3: mlir  3.2000  instr 3.262634  shortfall +0.062634   <-- fails a strict >=
run 7: mlir  3.0000  instr 2.859018  shortfall -0.140982
```

**One run in ten reproduces it**, and run 3 is CI's failure in a larger margin:
CI's was 1.7 microseconds, this is 63. Against the derived bounds, over the same
ten runs: worst per pass deficit **0.039971 ms** against 0.05, worst total
shortfall **+0.062634 ms** against an allowance of 0.55.

**The bound got tighter rather than wider**, from 0.15 ms per pass to 0.05, which
is the point: 0.05 is not a tolerance chosen against data, it is the largest
error a figure printed to four decimals of seconds can carry, and the containment
argument makes it exact.

**Which trigger this needs:** none. It is a pytest test and runs in every job that
runs pytest, including the coverage job that found it.

### 6. The two traceability tests, both faults

**Fault A**, a hand typed number: `resnet_block`'s cycle count changed from 1626
to 1499 in the README table. Red, naming the number and the row it sits in.

**Fault B**, stale provenance: `npuResultsSha` pointed at a commit that is not
the results'. Red, saying the results were re-recorded and the macros were not,
with the staleness check red beside it. Both restored.

**Which trigger these need:** none. They are pytest tests and run in every job
that runs pytest.

## The external tool install, recorded

Started before anything else in the phase, because Section 23 says P11 blocks on
it entirely. **Total wall clock under two minutes**, and then it cost most of a
session anyway for reasons unrelated to download time.

| Step | Seconds |
|---|---|
| clone and install SCALE-Sim | 35 |
| clone and install Accelergy | 11 |
| the four plug ins, first pass | 44, two of them failing |
| `accelergy-table-based-plug-ins` with `--no-build-isolation` | 1 |
| CACTI submodule, `make -j4`, and the plug in install | 9 |

Every install used a constraints file built from `requirements-lock.txt`, so no
external tool could move a pin the 175 committed results were measured under.
**Nothing in the lock file moved.**

Three deviations, each recorded where it belongs rather than absorbed:

- `accelergy-table-based-plug-ins` imports `yaml` in its `setup.py`, which pip's
  isolated build environment does not have. Installed with
  `--no-build-isolation`. `docs/adr/0003-resolved-tool-matrix.md`.
- `accelergy-cacti-plug-in` copies a built CACTI binary its clone does not
  contain. Submodule initialised and `make` run first. Same record.
- **SCALE-Sim does not run under numpy 2**, so its install is patched.
  D-0044, and `scalesim_installed_tree_sha256` in every manifest.

And one deviation from the specification itself: Section 16.3 says to read the
example topologies from the **installed path** of the pinned version, and the
pinned version's wheel ships the package without its `topologies/` or `layouts/`
directories. They are read from the pinned source clone instead, and
`test_the_column_order_is_the_pinned_versions_own` reads the real file and
asserts the exporter's header still matches it, so the rule that matters is
enforced from the pinned source rather than dropped.

### ZigZag, installed at P13, and it took four seconds

Started before anything else in the phase, for the reason Section 23 gives and
P11 followed: Section 16.5 blocks the cross check on it entirely.

| Step | Seconds |
|---|---|
| `pip install zigzag-dse` into `~/npu-venv`, wheel and five new dependencies | 4 |

**`zigzag-dse` 3.8.5, which imports as `zigzag`.** Section 16.5 pins that package
name, records that it is not the same name as the project, and warns that
`pip install timeloop` fetches an unrelated periodic task scheduling library
rather than the accelerator modelling tool. This project installs no Timeloop, so
that trap is one it cannot fall into.

**Recorded by version rather than by git sha**, which is the exception ADR 0003
already carves out for anything installed from a package index. There is no clone
and no sha to name, and a version that resolves on the index is the stronger
record for a tool that has one.

**Nothing in `requirements-lock.txt` moved**, checked the same way P11 checked
it. Seven pins resolved as already satisfied. Five packages arrived, `cerberus`,
`dill`, `multiprocessing_on_dill`, `seaborn` and `typeguard`, none of which any
committed result was measured under and none of which anything in this repository
imports.

**One deviation from Section 16.5 itself, and it resolves the easy way.** That
section says ZigZag requires Python 3.11 or newer and that the requirement must
be reconciled against the recorded build environment. This environment is 3.14.4,
which is **above** the floor rather than below it, so there is nothing to
reconcile: the wheel is `py3-none-any` and installed with no build step. The
floor `zigzag-dse` sets is why `requires-python` has said 3.11 since P0, and P13
is the first phase in which the tool that set it is present.


## Defects

**One new at P13, and it is in this file's sibling.** Every defect before D-0047
was in code, in a test, or in a claim about one; D-0047 was in the build. This
one is in the **defect log itself**, which is the document the project relies on
to be right about what is wrong with it.

- **D-0048**, D-0045 named a mechanism the cost model does not have and quoted a
  cell it does not match. **Resolved 2026-09-04, and the cost model is
  unchanged**: what was wrong was an entry and the reading of the divergence
  decomposition that rested on it. The full account is in the section above and
  the reproductions are in `docs/DEFECT_LOG.md`. **D-0045 is marked withdrawn
  with its body left exactly as P11 wrote it**, because an entry rewritten to be
  right is an entry nobody can learn from.

  **It is the sixth appearance of P10's shape and the second in a claim rather
  than in code.** D-0040 through D-0043 were each a value that arrived through a
  channel which loses information, treated as though it had not. D-0047 was a
  build property no reader could observe from inside the process that cared, with
  a test that had been vacuous for three phases. This one is nearer D-0047 than
  the other four: **the claim was checkable from inside the artefact the whole
  time, in six lines of arithmetic, and nothing ever asked.** The frozen
  constants test pins the value and says nothing about the accounting, so the
  accounting was under no assertion at all.

  **The practice that found it is D-0047's, applied to D-0047's own successor
  phase**: measure the thing before changing it. The difference is that here the
  measurement was arithmetic rather than `nm` on an object file, which makes it
  cheaper and therefore less excusable to have skipped.

### The two from P11, both now closed

- **D-0044**, SCALE-Sim v3 does not run under numpy 2 and reports a missing input
  file by exiting zero. Unchanged, and `scripts/patch-scalesim.py --check` still
  reports every edit in place.
- **D-0045**, **withdrawn at P13**. See D-0048.

## The prediction, adjudicated

`experiments/predictions/p11-scalesim-divergence.md`, committed at `f92de42`
before `experiments/scalesim_export.py` existed at all. **Most of it is wrong**,
and the full clause by clause answer with the entry quoted verbatim is in
`docs/NUMBERS.md`. The file is not edited.

| Claim | Verdict |
|---|---|
| Direction: this project reads **above** SCALE-Sim on the covered layers | **wrong on five of seven models.** Above on `lenet` and `lenet_batched`, below on the rest, by as much as 87 percent. The entry's own falsification list names this |
| The gap is dominated by named modelling differences rather than by error | **met.** The decomposition is a partition of named terms with a residual of exactly zero |
| Under 10 percent per layer on dense compute bound layers, 10 to 25 on the 1 by 1s, 5 to 20 whole model | **wrong on all three.** 340 of 550 layers exceed 25 percent; the 1 by 1s reach +334; whole model reaches -87 |
| No layer above 25 percent at the default budget with both ports on | **wrong**, and D-0045 is the root cause the clause demands |
| Mechanism 1: SCALE-Sim reads low on 1 by 1 convolutions, widest gaps there, this project reading higher | **confirmed**, and it is the one mechanism the entry got fully right |
| Mechanism 2: pooling leaves the comparison as coverage rather than as divergence | **confirmed.** 53114 cycles, its own term, never inside a divergence |
| Mechanism 3: the two models disagree about array fragmentation | **confirmed**, and larger than expected at minus 435825 cycles |
| Coverage 0.5 to 0.85 on the convolutional models | **met.** 0.591 to 0.797 |
| Coverage below 0.3 on `lenet` | **wrong.** 0.946, for a reason the entry's falsifier misdiagnoses. `scalesim_covered_op_fraction` is 0.208 and is the quantity the clause was reaching for |
| Kendall tau above 0.8, pairwise above 0.85 | **wrong on tau**, 0.6337 over cells and 0.7460 over layers. Pairwise 0.8211 over cells and 0.8783 over layers, so below on one and above on the other |

**The entry called rank fidelity "a stronger finding than any absolute error
here" and it was right about that.** The cost model is not merely imprecise; it
orders one pair in five differently from the reference.

### The P10 prediction, still adjudicated

`experiments/predictions/p10-ablation-deltas.md`, committed at `f92de42` before
the harness had been run once. **Two of its four claims are wrong.** The file is
not edited; the adjudication is in `docs/ENGINEERING_LOG.md`.

| Claim | Verdict |
|---|---|
| 1. Exactly three ablation rows are not zero | **wrong.** Two are. `-canonicalize` is zero, and the mechanism above is why |
| 2. No ablation moves numerics beyond tolerance, and ablating the batch norm fold returns the movement to 0.0 | **met**, with a wording defect recorded: "no other ablation moves that field at all" has two readings and the strict one is false |
| 3. The rows are identical at both budgets | **wrong.** 16 of 56 differ, because `resnet_block` and `inception_block` spill at their tight budget, which ADR 0008's own table already recorded and the prediction misread |
| 4. Suite total between 100 and 130, and no ablation raises it by more than 30 | **met.** 117, and the largest single raise is 8 |

Claim 3 being wrong is the more useful of the two: it is Section 16.2's stated
reason for requiring ablation rows at every budget, arriving as evidence rather
than as a rule this project took on trust.

**P13 adds one clause to this table and it is not in the entry.** The P11
entry's fourth row, that no layer exceeds 25 percent at the default budget
with both ports on, was answered wrong at P11 and D-0045 was recorded as the
root cause the clause demands. **That root cause is withdrawn at P13**, so
the row is still answered wrong and the reason for it is again open. See
D-0048. The prediction file itself is not edited, which is the rule, and this
correction lives here and in `docs/NUMBERS.md` rather than in it.

## Open questions

Twelve, and three are new at P13. One from P12 is answered.

**What the two tools actually disagree about, now that it is not the weight
preload.** This is D-0048's remainder and it is the phase's largest open
question. With the stalls removed, `dilated_stack`'s `conv1` still reads 8.65
times higher on SCALE-Sim and `inception_block`'s 1 by 1 reads 4.34 times higher
here. The dilation approximation is already its own term measured by a second
SCALE-Sim run at the true tap extent, so it is not the answer to the first. **The
next phase to look at it should measure the charge rather than read it**, which
is the whole of what P13 adds to the question.

**Whether the near cancellation in the divergence decomposition should be
reported differently.** The two dominant terms both carry SCALE-Sim's stall count
with opposite signs, so part of their equal and opposite character is
bookkeeping. Nothing about the decomposition is wrong and the residual is still
exactly zero; what is open is whether `docs/NUMBERS.md` should say so beside the
headline, because a reader who takes the near cancellation as a physical
coincidence is taking more from it than it carries.

**The one flake is answered, and it is D-0049.** It was observed once, was
green in every run after it, and was recorded as unexplained with the note that
the failure text had been lost to a script that tailed three lines. It was then
reproduced deliberately: **one red in eight runs under twenty four busy loops,
none in three on the idle machine.** The message is
`--mlir-timing reports Canonicalizer at 4.5000 ms and this project's
instrumentation at 0.4496 ms, a gap of 4.0504 ms against a bound of 2.3000 ms`.

**It is not the bound P12 asked P13 to watch, and that distinction is the
useful part.** `cross_check_against_mlir_timing` has two bounds pointing in
opposite directions. The **deficit** bound catches the instrumentation reading
above MLIR, is derived from the print quantum, and is D-0043's, whose margin
narrowed across 0.1577, 0.1177 and 0.1856 ms against 0.2000. This is the
**upper** bound, `half_ulp` plus 50 percent of MLIR's figure, and nothing here
is a fourth data point on the other one.

**The bound is conditional and the condition is not being checked**, which
`pass_stats.py` already knows how to say: it refuses to check this bound under a
traced interpreter, on the stated grounds that a tracer stretches everything
inside MLIR's window so the gap stops being the instrumentation's own walk. A
busy machine does the same thing for the same reason. **The fix is a
precondition and not a wider bound**, and `TIMING_GAP_FRACTION` must not move to
a number chosen to make a run green. Left open deliberately, because a red at a
gate is not answered by widening it, and handed to Section 17.9's flake
governance at P15.

**Where it matters is CI rather than here.** The test carries `slow`, CI's
`pytest slow cells` step runs slow tests, and the runners are shared four vCPU
machines. A developer machine with nothing on it is the least likely place for
this to fire.

**The `--mlir-timing` gap has no new quiet measurement.** P11's quiet runs
measured 0.1577 and 0.1177 against D-0043's 0.2000 bound and P12's measured
0.1856. **P13 adds no fourth point**, because the fourth point comes from
`run_benchmarks.py --force` and this branch had no reason to run it: no
ablatable pass landed, so the suite is the same 175 cells P12 recorded. The next
session that runs the suite gets the point. **Do not respond to a red at that
bound by widening it.**

**The allocator growth benchmark is still not wired into CI**, and the trigger it
is waiting for has not fired. P12 recorded the trigger as "P13, because tiling
makes functions longer". Tiling has not landed, so functions are not longer, and
switching the gate on now would gate on a curve nothing has moved. **The trigger
carries forward unchanged to the commit that adds `-npu-tile-to-scratchpad`**,
with the same recipe: rehearse it red first with `--sizes 500`, then switch it on
under `pull_request` and `push` to `phase/**` like every other step in `ci.yml`,
in the `build-and-test` job, before the `regression-baseline --check` step that
runs last.

**The kernel's team cap and the thread scaling table are unmeasured against
tiling**, for the same reason. `experiments/kernel_threads.py` was not re-run,
because the convolution kernel's loop nest did not change on this branch and the
instruction shapes it would measure are P12's. Its own trigger, the first phase
that changes the kernel's loop nest, has also not fired.

**How many other compile options do not reach the sources they were written
for.** Unchanged from P12. Nothing suggests another is wrong, and that is exactly
what was true of D-0047.

**Whether the thread scaling table should be recorded per host.** Unchanged.

**SCALE-Sim runs from a patched install and there is no upstream fix.**
Unchanged.

**The fp32 MAC coefficient fails Section 16.4's sanity check at 10.71 times.**
Unchanged, and P14's integer kernels are where the energy story changes shape.

**The clock is 1 ns and this project has no other reason to have one.**
Unchanged.

**The external tools do not run in CI and the committed numbers are the only
record of them.** Unchanged, and **ZigZag joins that story rather than changing
it**: the CI step now asserts all three absent instead of one, which was
rehearsed four ways.

**The measured per cell cost has not reached Section 2.** Unchanged, and it is
still the only part of the P10 and P11 gates no branch can close by itself.

**`lenet_batched`'s tight budget cell at batch 1 does not exist, and would not
have been tight if it did.** Unchanged and still worth a look when P13 re-measures
the budgets, which it has not yet done: ADR 0008's fraction becomes a live knob
only when tiling gives the compiler a way to fit an instruction whose operands
exceed the budget, and that is still ahead.

**The declared Python floor and the checkable one still differ by a minor
version.** Unchanged in its consequences, and P13 adds the observation that the
floor exists because of `zigzag-dse`, which is now installed for the first time
and clears it at 3.14.4.

**The NDEBUG and sanitizer directories still cannot build anything that links
MLIR.** D-0031, ADR 0009. Unchanged.

**Section 17.3a's fifth metamorphic relation still cannot be written.**
Unchanged.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both, and repeated at every phase because a fact that stops being repeated
is a fact somebody eventually does not know.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, not once. It may be read, and only through a command that cannot
write. **Only the owner may retire it.** Nothing on this branch went near it.

## Next phase

**P13, continued. Not P14.** The gate is not met and four of the six deliverables
have not started. What follows is the plan as it stands after D-0048, which is
different from the plan P12 handed over, and the differences are the point.

**What is deleted from the brief P12 wrote.**

- **The D-0045 governance sequence does not run.** No `docs/BREAKING_CHANGES.md`
  declaration, no `CostModel.h` change, no divergence re-measure driven by a
  constant that moved, no baseline re-record for a numeric reason, no
  re-versioning of `p11-scalesim-divergence.md`'s band against new constants.
  All of it was conditional on a charge moving and no charge moves. Section
  16.5's rule against retuning a model to match an external tool is what P11
  obeyed when it left D-0045 open, and it is the same rule that says not to
  change the charge now on a diagnosis that does not survive being checked.
- **The divergence terms are not re-measured for D-0045's sake.** What is worth
  re-reading is the decomposition's prose, not its arithmetic.

**What is unchanged and is the critical path.**

1. **The lowering**, which is now the critical path and is the section above.
   `-npu-tile-to-scratchpad` is done and unreachable until a tiled function can
   lower, and the second half of that change is what decides whether tiling
   relieves any pressure at all. Do the per slice DMA, not only the patterns.
   Expect `test/Dialect/NPUISA/dma-boundaries.mlir` to move, and expect to have
   to argue in Section 8's own terms why the values entering the scratchpad are
   now the slices.
2. **`-npu-double-buffer`**, over the tokens, **before allocation** per Section
   5.1, because the allocator has to see the doubled working set or it places a
   program that cannot fit and spills the wrong buffers.
3. **`-npu-assign-layout`** with the inverse transpose fold, without which the
   pass only ever adds instructions.
4. All three ablatable, wired into `-O2` per Section 12, positive **and**
   negative lit tests each per the P9 pattern, and **the Section 2 arithmetic
   re-derived in the same commit** rather than after it: 11 ablatable passes, 154
   ablation cells, 217 total.
5. **The Section 13.3 experiment, all three arms per model**, which is the
   phase's reason to exist. Only `resnet_block` and `inception_block` spill at
   their tight budget today, so those two are where the comparison has a subject;
   ADR 0010's six models that cannot allocate at batch 4 are the population
   tiling can bring into existence, which would make the experiment wider rather
   than only deeper.
6. **The ZigZag cross check**, Section 16.5: export the chosen mapping in ZigZag
   and Timeloop mapping form and compare **cost under the same mapping**, not two
   totals from two mappers. Bound the exploration to the layers actually tiled,
   record wall clock and peak memory, and **if it cannot be made to fit inside
   the 12 GB ceiling, stop that step and record why. Never suggest editing the
   WSL configuration.** If it finds a materially better tiling, say so, quantify
   it, and record it as a follow up in `docs/DEFECT_LOG.md` with a reproduction;
   **do not retune the search to match**.

**Three things this session learned that the next one should not rediscover.**

- **`run_benchmarks.py` needs no arithmetic change** to go from 175 to 217. It
  reads the ablatable set out of the driver at run time and computes both cell
  counts. What needs editing is its prose and the six places listed above where a
  count is written down by hand.
- **`external_tools.py`'s table is the single edit point** for a new external
  tool, and every consumer follows from it. That is D-0046's fix working.
- **The tiled implementations decline rather than fail.** A pass that gets a
  failure back from `getTiledImplementation` has been told the tile is not
  expressible, and Section 13.2's answer to that is the allocator's spilling, not
  a diagnostic.

## Next command

The branch is not ready to push: the gate is not met and the phase is
incomplete. **Do not open a pull request for it yet.**

The next command is the one that shows the blocker, because it is the shortest
statement of what has to change:

```
npu-opt test/Transforms/tile-to-scratchpad.mlir \
  --npu-tile-to-scratchpad=budget=2048 --npu-lower-to-npuisa
```

It reports `failed to legalize unresolved materialization` on the destination
argument. Making that command produce a program is the first change; making the
program it produces have a smaller scratchpad peak than the untiled one is the
second, and only the second is worth measuring.

**Four things to check before anything else.**

**The suite is still 175 cells and the baseline is recorded at `f7b32ba`.** A
`regression-baseline.sh --check` at the tip reports no drift, and the first
commit that puts an ablatable pass into a level moves the cell count, the
ablation table, `macros.tex` and six hardcoded numbers together. Move them in one
commit.

**`run_benchmarks.py --force` has not been run on this branch**, so there is no
fourth `--mlir-timing` data point and the committed 175 cells are P12's.
Serialise that run when it happens and say so, and read a red at D-0043's 0.2000
deficit bound as the fourth point rather than as noise. **A red at the other
bound is D-0049 and is a different thing**, so read the message before deciding
which one fired.

**`compile_time_benchmark.py --check` is still unwired and its trigger has still
not fired**, because functions do not get longer until a level runs the tiling
pass. Rehearse it red with `--sizes 500` in the same commit that wires it.

**D-0049 is open and is the reason to run measurements on a quiet machine even
when the measurement is a test.** It is reproducible at roughly one run in eight
under load and zero on an idle machine.
