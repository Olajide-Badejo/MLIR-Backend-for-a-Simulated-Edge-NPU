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

**Last updated:** 2026-09-06.

## Current phase

**P13, tiling, double buffering and layout.** Branch `phase/p13-tiling`, cut
from `main` at `2f59429`, which is the P12 merge. **The gate is met on all seven
clauses**, and the two that closed last are the ones this phase existed for:
Section 13.3's three arms, run on all seven models, and the Section 16.5 ZigZag
comparison under the mapping the pass chose. Both are recorded in
`docs/NUMBERS.md` beside the predictions they answer, and both answer some of
their own clauses in the negative.

**All three passes are in `-O2`, and tiling reaches the suite.**
`-npu-assign-layout`, `-npu-tile-to-scratchpad` and `-npu-double-buffer` went in
together at `8f79972`, in Section 12's own positions, with the ablatable set
from 8 to 11 and the suite from 175 cells to 217. **The suite has been
re-recorded three times on this branch, serially, on a quiet machine, and each
re-record is a commit of its own.** At the wiring commit not one counted field
of the 175 pre-existing cells moved, so there was nothing to declare for it. At
`80ac24d`, the tree where the compiler half of D-0052 lands, **31 cells moved
and every one of them is a tight budget cell**. At the tree where
`-npu-double-buffer` starts firing, **138 cells moved `npuisa_op_counts` and 57
moved the fragmentation ratio, and not one moved an instruction count, a cycle
or a golden byte**. The second and third were declared in
`docs/BREAKING_CHANGES.md` before the commits that caused them, and the first is
the other way an entry is satisfied: measured to be zero.

**The wiring and what followed it found seven defects, and two of them changed
the phase.** D-0051 to D-0057 are in `docs/DEFECT_LOG.md` with their
reproductions. **D-0052 was the first: a tiled result assembled in DRAM could
not be read back.** The tiles are written one store each, and the binary's
`operand-defined` and `operand-extent` checks satisfied a read out of a
**single** written span, so an assembled value that another operation read was
refused by the encoder. That is D-0050's third part, arriving through the wiring
rather than through reading, and it is why the first tiled program this suite
produced did not encode. **D-0056 was the second**, and it turned out to be in
the allocator rather than in the tiling pass.

**While that question was open the pass declined rather than emitting a program
its own encoder refuses**, which is Section 13.2's own answer to a tile that is
not expressible, and the allocator's spilling was the fallback. The one shape
that was expressible throughout is an assembly nothing reads: a tiled operation
whose result is the function's own, whose tiles are stored straight into the out
parameter. `test/Encoding/tiled-result-returned.mlir` carries that case from the
tensor level through the encoder and the disassembler, one load and one store
per tile and no load of the assembly back, and it is the permission that
`test/Encoding/tiled-assembly-in-scratchpad.mlir` is the refusal for.

**Both halves of D-0052 are in now, so tiling fires inside the suite**, on
`resnet_block` and `inception_block` at their tight budgets and on five more
models wherever `-npu-fuse-ops` is ablated. The committed prediction's main
clause, that nothing tiles, was right at the tree it was adjudicated against;
what supersedes it is the measurement below rather than an edit to the file,
which is the rule.

**The owner decided D-0052 the same day, and the format half of the fix is
in.** The decision is region scoped coverage on the DRAM side of checks 8 and 9,
with no `kVersion` bump: inside a declared spill slot the validator tracks exact
byte coverage, strided writes run by run, and accepts a read when every byte it
addresses lies inside that one slot and has been written. The scratchpad keeps
the no merge rule. `docs/BREAKING_CHANGES.md` carries the declaration, written
before the commit, and `20fc6c1` is the commit.

**The compiler half is in, and the cell that would not place does, because the
allocator was refusing a legal spill.** D-0056 is the diagnosis and it is not
the one the escalation guessed. The residual **does** have a view user: it is
the tiled convolution's input, and under the per slice convention a slice of a
scratchpad value is a view and no transfer, so tiling pinned it twice over,
whole resident as the subview base and unspillable because it had view users.
The spill rule refused any buffer a view was taken of, and what a reload cannot
serve is only a view that is **written through**, because `spill` re-bases a
read only view onto the reload with the same call that moves a reader. With the
rule narrowed the cell places, with one spill, at a peak of 6144 bytes against
the 6432 the untiled program needs.

**No discriminator was written and none was needed.** The two rules the previous
checkpoint tried and recorded as failures stay recorded; both were trying to
separate cases that did not need separating. Section 13.2's own trigger, tile
when the working set exceeds the budget, is what the pass does, and the
allocator decides what the program costs.

**What tiling costs and buys, over the whole suite.** Thirty one cells move and
**not one default budget cell moves**. `inception_block` at its tight budget
goes from 3799.0 cycles with three spills and 21936 DRAM bytes to **3395.0 with
none and 12720**; `resnet_block` goes from 17 instructions and 2018.0 cycles to
**21 and 2660.0**. **Tiling helps one model and costs the other**, and the
mechanism is which operand it relieves: a slice of a **DRAM** value becomes a
transfer and a slice of a **scratchpad** value becomes a view, so tiling
relieves an argument or a DRAM assembly and does not relieve an on chip
producer, which stays whole resident as the base the views are taken of. That
sentence is in `docs/PASSES.md` beside the pass and in `docs/NUMBERS.md` beside
the table.

**The commit order carries meaning and the first commit is the phase.** P13 was
briefed to fix D-0045 under the full declare then re-record governance, on the
stated understanding that fixing it changes the cost model. `b2b63df` is the
measurement that says there is nothing there to fix: the charge already does what
the entry says it does not, and the entry's own reproduction quotes a cell the
committed results contradict. Everything the governance sequence would have
governed is therefore not done, deliberately, and the rest of the phase was
re planned around that.

**Two negative results, and both are about the machine rather than about a
pass.** Double buffering is correct, exact, and moves no cycle on a tiled
program, because Section 5.5's model is a two port dataflow schedule and tiling
makes a program DMA bound, 1524 against 596, so there is nothing to hide a
transfer under. Layout assignment answers NCHW at every extent, because the
stride penalty of 0.5 cycles per element is eight times the 0.0625 a permutation
costs, so performing a transpose always beats moving the same data strided. Both
are measured, both are asserted by a test, and both are inputs to Section 13.3
rather than obstacles to it.

**Putting them in a level added a third negative to the first of those, and it
has since been fixed.** On the suite `-npu-double-buffer` did not fire **at
all**: `prefetched` was 0 and `not-hoisted` was every transfer, on all seven
models at both budgets, because every argument load is in the entry block beside
the other argument loads and a constant's load could not take its `npuisa.const`
with it. D-0054 carries it. **The constant joins the prologue now and the pass
fires on one to four transfers per model**, declining the hoists whose doubled
residency would not place, so the pass has one zero rather than two: the cycles
still do not move, for the two port reason above, and that is the only zero left.

**D-0054's own diagnosis was wrong and the measurement says where the fault
was.** The entry blamed the pass's hoist walk for the programs its probe
produced, on the strength of a verifier error. The walk is not at fault: the IR
the pass emits verifies, and the refused program appears two passes later.
**The allocator believed an asynchronous transfer is finished at its issue**, so
it put a spill store between the two halves and it ended the buffer's live range
there, and a reload was placed at an in flight destination's offset. Both are
fixed in `023edcd` with a pair of programs that reproduce them, and **neither
moves a number**, because the pass fires on nothing at this tip.

**Making it fire was held for one checkpoint and is done, and the budgets did
not move.** With the allocator correct, admitting `npuisa::ConstOp` to the
prologue produces programs the verifier accepts and the pass fires on one to four
transfers per model. It also stopped **five of the seven models placing at their
ADR 0008 tight budgets**, because a prefetched weight is resident across the
computation it hides under: `lenet` wanted 234880 against 194624 and
`resnet_block` 8736 against 6464. A prefetch that cannot be placed is not a
prefetch and those budgets are frozen, so **the pass asks before it commits**: it
runs the allocator's own `assignOffsets` over the intervals the hoist would
produce, at the same budget, strategy and alignment, and declines what does not
place. The sweep line peak was tried as that question first and is 56 bytes short
of it on `dilated_stack`, which is in D-0054 with the diagnostic.

**The third finding is about the compiler.** At `-O2`, fusion hides 30 of the 44
convolutions and matrix multiplications in the suite inside `npu.fused_op`
regions, where the tiling pass does not look, and two of the seven models have
none left visible. That decline is now counted and remarked on rather than
silent, and Section 13.3's resolution is written down before the experiment runs
rather than after: the arms stay at `-O2`, and arm two is reported in two
configurations, with fusion and without, which is the treatment Section 13.3
already gives arm one.

## Gate status

**Met, on every clause.** Clause by clause, with the evidence for each. The
tree level evidence is the verification table below, run at `101cb17` on a
machine that was quiet for every measurement in it.

| Clause | Status |
|---|---|
| Goldens byte identical for the tiling work, exactly | **met, and it is evidence now.** All 21 golden tensors are byte identical and `git status` on `test/baseline/golden` is empty at the tree that has all three passes in `-O2`. It became evidence at the wiring commit, which is what the previous handoff said would make it one |
| Any movement from layout or double buffering inside 1e-6, declared in `docs/BREAKING_CHANGES.md` before the causing commit | **met, and it has been answered three times.** At the wiring commit **not one counted field of the 175 pre-existing cells moved**, over instructions, cycles, compute and DMA cycles, scratchpad peak and bytes, spill count, spill DMA count, DRAM bytes, the oracle distance, the overlap fraction and the fragmentation ratio; an entry declaring a movement measured to be zero would be a false declaration. The 31 cells that moved next are **tiling's**. The 138 that moved after that are **double buffering's**, and the clause is met in its strongest form there: the movement is **exactly zero** on every numeric field, and what moved is the name a transfer is written under and the packing of an arena. Both were declared before the commits that caused them, and the second declaration is adjudicated clause by clause with one of its eleven wrong. `-npu-assign-layout` still moves nothing of its own, in all four counted columns, over all 14 pairs |
| No `scf` operation reaches the lowering, asserted by a lit test | **met, as a statement about the lowering.** `test/Pipeline/p13-passes-at-o2.mlir` runs `-O2` at a budget where tiling fires and asserts no `scf` operation anywhere in the level's output, and the `NOSCF` prefix in `test/Transforms/tile-to-scratchpad.mlir` is kept beside it |
| A tiling disabled ablation row reproduces the previous spilling numbers to the cycle | **met, to the cycle, and it is a measurement now rather than an identity.** With `-npu-tile-to-scratchpad` ablated, `resnet_block` at its tight budget reads 17 instructions, 2018.0 cycles and 1 spill and `inception_block` reads 22, 3799.0 and 3, which are the figures ADR 0008's budgets were measured against. Their **baselines** moved, because the pass fires: `resnet_block` to 21 and 2660.0 and `inception_block` to 3395.0 with no spills. `test_the_tiling_disabled_row_reproduces_the_spilling_numbers_to_the_cycle` pins the ablated cells exactly and asserts the baselines differ, because a row where the two agreed would be measuring nothing |
| The tight budget question answered per model with all three arms of Section 13.3 | **met.** All three arms ran on all seven models, at every model's frozen budget, over the swept ranges where tiling fires, at the floors `experiments/tight_budget_floor.py` measured, and at a control budget where nothing is over budget and all six configurations agree. `experiments/three_arms.py` is the script, `experiments/results-three-arms/arms.json` the record, and `docs/NUMBERS.md` carries the tables, the crossover per model and the clause by clause adjudication of `p13-three-arms`, of which two clauses are met, two are wrong and two are half right. **Tiling helps one model and costs another at the shared budget, and buys budget on four only once fusion stops hiding the operands.** A program an arm cannot place is reported as not placing, never as a slower one |
| The layout delta reported whichever way it went, with the DMA stride term shown to carry it | **met at the level.** The ablation row is zero on every model at both budgets, in all four counted columns, and the DMA stride term is exactly what makes it zero: 0.5 cycles per element strided against a permutation's 0.0625 makes a physical transpose eight times cheaper at every extent this machine can hold. `CostModel.AStridedMoveCostsMoreThanThePermutationThatAvoidsIt` asserts the direction and the factor in the file that owns both constants |
| The ZigZag comparison shown next to its prediction, compared under the same mapping | **met.** `experiments/zigzag_same_mapping.py` exports the mapping `npu.tiling_choice` records, in ZigZag and Timeloop form, for all 42 tiled layers of the suite's tight cells and the swept range, and compares cost under that one mapping. **It is the same mapping by construction and by check**: the ordering is asserted complete before the call and read back off ZigZag's own evaluation after it, on every layer. Geometric mean 1.160, range 0.598 to 3.053, above on 28 layers and below on 14, in 59.1 seconds and 747 MiB against an 8192 MiB ceiling. Three of the prediction's clauses are wrong and the direction clause is one of them. Two terms carry the divergence, the descriptor per transfer and the array fold, and where the fold is identical the two cost models agree to five percent. ZigZag's own engine prefers a different nest on one layer of 42, which is D-0060 with a reproduction and no retune |

**Two clauses of the brief around the gate are answered rather than pending, and
they are answered in opposite directions.** The D-0045 governance sequence does
not run, because no charge moves. The Section 2 carve out **is** re-derived, to
217 cells, 11 ablatable passes and 154 ablation cells, in "The Section 2 carve
out, for the owner" below, because the three passes Section 12 names did land in
`-O2` and the arithmetic became derivable in the commit that put them there.

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

- **The cost model's charges are untouched, and the sentence this file used to
  carry here was wider than that and false.** It said `git diff main..HEAD`
  touches no file under `lib/Simulator/`, and it does touch two. What is
  checkable and true: `python/npu_frontend/cost_model.py` and
  `include/NPU/Simulator/` are **untouched over the whole branch**, so no
  constant and no declared charge moved; `lib/Simulator/CostModel.cpp` moved to
  `lib/CostModel/` as a rename with a zero byte diff; and
  `lib/Simulator/Kernels.cpp` gained the version 2 scatter path at `670dd0b`,
  which **is** declared, in `docs/BREAKING_CHANGES.md` under the `kVersion` bump,
  and which no program before version 2 could reach. **The wiring commit and the
  two beside it touch none of those files at all**, which is the narrow claim
  this phase's last three commits are entitled to make.
- The correction is recorded rather than quietly applied because a claim of the
  form "this diff touches nothing under that directory" is checkable in one
  command, and this one was carried for four commits without anybody running it.
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
- **`-npu-tile-to-scratchpad` is implemented and is in `-O2`.** Its own commit
  argued at length for landing it in no level at all, and the wiring commit is
  where that changed and why. It fires only when an
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
- **`-npu-double-buffer` is implemented and is in `-O2`.** Over the tokens,
  before allocation, per Section 5.1. A `npuisa.dma_load` is hoisted
  above the computation before it and becomes a `npuisa.dma_load_async` with an
  `npuisa.await` left where it was. Safety is `npuisa::overlaps` with `Unknown`
  as a refusal, which is Section 8's rule 4 as code, asked of the analysis rather
  than of an identity comparison even though this pass runs before allocation
  where an identity check would happen to be right. The hoist stops at another
  transfer, because both are charged to the same port and lifting a load above a
  load moves work along a saturated timeline and hides nothing. The allocation
  moves with the transfer, because a tile's destination buffer is defined
  immediately before the load that fills it. Three lit cases plus statistics.
- **`-npu-assign-layout` is implemented and is in `-O2`.** The choice,
  scored on Section 5.5 and counted in `kept-nchw`; the inverse transpose fold
  Section 12 names; and the sink through a relu that lets the fold reach a pair
  the graph did not write adjacent. There is deliberately no code that rewrites
  an operation into NHWC, because the comparison refuses that trade at every
  shape this machine can hold and a materialisation path would be a branch no
  input could reach. Five lit cases, three of them negatives, plus statistics.
- **The suite is 217 cells and the ablatable set is 11**, from the wiring
  commit. The three passes are in `-O2` in Section 12's own order, each carrying
  its `ablatable` property, and `run_benchmarks.py` needed no arithmetic change
  to follow: it reads the set out of the driver at run time.

## What tiling was waiting on, all of which has landed

**All three of the changes below are done**, at `633c116` for the lowering and
`670dd0b` for the format. The section is kept rather than deleted because the
measurement in it is the evidence for why the lowering had to change the way it
did, and a later reader asking why the slice enters the scratchpad rather than a
view of a resident whole should find the number rather than the argument.

**`-npu-lower-to-npuisa` had no pattern for `tensor.extract_slice` or
`tensor.insert_slice`**, which is what a tiled program is stitched from, so a
tiled function did not lower at all. `npu-opt --npu-tile-to-scratchpad
--npu-lower-to-npuisa` reported `failed to legalize unresolved materialization`
on the destination argument. That was the first change and the smaller one.

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

## The lowering patterns, fully scoped, with the last unknown resolved

**Everything this needs already exists.** The design below was checked against
the code rather than sketched, and the one mechanism it turned on is confirmed:
**a `memref.alloc` of a `#npu.dram` type carrying the `npuisa.spill_slot`
attribute is a DRAM buffer the encoder places and maps.** The allocator has
created them since P5 for spilling, `layOutDram` finds them by predicate rather
than by analysis, and `dramAddresses` gets the result. That is the assembly
buffer primitive, and it needs no encoder change.

Reusing the mark rather than inventing one is deliberate and the name is
accurate: a spill slot is a DRAM buffer that exists to hold a value the
scratchpad could not hold, and a tile assembly buffer is exactly that. A second
mark would mean teaching the encoder a second predicate for a concept it already
has.

**Four patterns, and the third is the decision made executable.**

| Source | Becomes |
|---|---|
| `tensor.extract_slice` of a **DRAM** value | `memref.subview` of the DRAM buffer, a scratchpad `memref.alloc` at the tile's extents, and one `npuisa.dma_load` between them |
| `tensor.extract_slice` of a **scratchpad** value | `memref.subview`, and no transfer. The bytes are already on chip and a scratchpad to scratchpad DMA is not representable in this ISA |
| `tensor.insert_slice` into a DRAM buffer | `npuisa.dma_store` from the tile's own buffer into a `memref.subview` of the destination, with the insert's result mapping to that destination |
| `tensor.insert_slice` into a **scratchpad** buffer | **refused, by name.** That is the assembly checks 8 and 9 reject and `test/Encoding/tiled-assembly-in-scratchpad.mlir` records; the pass says so rather than emitting a program the encoder will refuse later |

**Where the assembly buffer comes from.** A `tensor.empty` whose uses include
being an `insert_slice` destination is an assembly buffer, not a scratchpad
destination, so `EmptyOpLowering` gives it a DRAM allocation with the spill slot
mark instead of a scratchpad one. **One special case avoids a transfer that is
not representable**: when that value reaches a `func.return`, it is mapped
directly to the out parameter the function gained for that result, because
otherwise the return's own `dma_store` would be DRAM to DRAM. That is the shape
`peak_perslice.mlir` already has, and it encodes, runs and matches its reference.

**What the boundary test has to say afterwards.**
`test/Dialect/NPUISA/dma-boundaries.mlir` pins Section 8's count immediately
after lowering, and the count changes: an untiled function still gets one load
per argument that is read and one store per returned value, and a **tiled** one
gets one load per slice that enters the scratchpad and one store per tile that
leaves it. The invariant is the same sentence applied to a different set of
values, which is the argument `docs/ARCHITECTURE.md` carries, and the test moves
in the same commit as the change it tests.

**The three things that make this smaller than it looks.** The encoder already
resolves a DRAM view chain, so a subview of an argument has an address. The
format already carries strides on both sides, so a tile scatters correctly. And
the tiling pass already emits exactly the `extract_slice` and `insert_slice`
shapes these patterns consume, at constant offsets, fully unrolled.

## The prediction, answered before the wiring rather than after it

`experiments/predictions/p13-tiling-cell-movement.md`, committed at `33c17bc`
strictly before the pass was wired into any level and before a cell was measured
with it wired. **Its main clause is right and the measurement is stronger than
the bracket asked for.**

### Nothing tiles, at either budget, on any model

The pass run alone over every model's tensor level IR, reading its own
statistics, as tiled / already fitting / declined:

| Model | Tight budget | at the default 1048576 | at its tight budget |
|---|---|---|---|
| `lenet` | 194624 | 0 / 1 / 0 | 0 / 1 / 0 |
| `depthwise_separable` | 8192 | 0 / 0 / 0 | 0 / 0 / 0 |
| `resnet_block` | 6464 | 0 / 1 / 0 | 0 / 1 / 0 |
| `inception_block` | 6144 | 0 / 3 / 0 | 0 / 3 / 0 |
| `conv_bn_relu_stack` | 6464 | 0 / 1 / 0 | 0 / 1 / 0 |
| `dilated_stack` | 8064 | 0 / 0 / 0 | 0 / 0 / 0 |
| `lenet_batched` | 200832 | 0 / 1 / 0 | 0 / 1 / 0 |

**So wiring the pass into `-O2` moved no cell at either budget**, all three new
ablation rows were zero everywhere at that commit, and there was nothing to
declare in `docs/BREAKING_CHANGES.md` for it. The reason is the one the
prediction gave: ADR 0008's tight budget is the smallest at which a **program**
allocates, which needs every simultaneously live buffer to fit, and that is a
stronger requirement than any one operation's working set. **What changed the
answer afterwards was not a budget**: it was the two defects the wiring found,
and the third table below is the same measurement taken again once both were
fixed.

### The same table re-measured at the wiring commit, which is a different measurement

**The table above is the pass run alone at its own default options. The table
below is `-O2` with the options the pipeline chooses**, which is what the suite
actually compiles, and it had to be taken again because the wiring changes the
premise twice: the pipeline hands the tiling search the allocator's budget and
tells it that `-npu-double-buffer` is in the pipeline, which doubles the
prefetched operand's contribution per Section 13.2. **This is the wiring
commit's tree, one commit before the compiler half of D-0052**; the table after
it is the same measurement at the tip and is the one to read for what the
compiler does today.

Tiled / already fitting / declined, through `-O2`, at the wiring commit:

| Model | Tight budget | at the default 1048576 | at its tight budget | tight, `-npu-double-buffer` ablated |
|---|---|---|---|---|
| `lenet` | 194624 | 0 / 1 / 0 | 0 / 1 / 1 | 0 / 1 / 1 |
| `depthwise_separable` | 8192 | 0 / 0 / 0 | 0 / 0 / 2 | 0 / 0 / 1 |
| `resnet_block` | 6464 | 0 / 1 / 0 | 0 / 0 / 2 | 0 / 1 / 1 |
| `inception_block` | 6144 | 0 / 3 / 0 | 0 / 1 / 2 | 0 / 3 / 0 |
| `conv_bn_relu_stack` | 6464 | 0 / 1 / 0 | 0 / 1 / 2 | 0 / 1 / 1 |
| `dilated_stack` | 8064 | 0 / 0 / 0 | 0 / 0 / 2 | 0 / 0 / 1 |
| `lenet_batched` | 200832 | 0 / 1 / 0 | 0 / 1 / 1 | 0 / 1 / 1 |

**Which cells differ from the pass alone table, and why.** The **default budget**
column is identical, and that matters most: a default budget cell that moved
would be a wiring defect rather than a declaration to write, and none moved. The
**tight budget** column differs in the `declined` count on every model, and the
last column is why: with double buffering ablated the search sizes the working
set without the prefetch and finds fewer operations over budget. **That is the
coupling Section 13.2 asks for, measured**, and it is why `docs/PASSES.md` says
beside the `-npu-double-buffer` row that ablating it also relaxes the tiling
search.

**The declines are two different declines and the difference is the finding.**
Some are fused regions over budget, which the pass has counted and explained
since `515560c`. The rest are D-0052: an operation over budget whose assembled
result would be read by another operation, which the binary format cannot
express. **Before D-0052's rule the tight budget column read
`resnet_block` 1 / 0 / 1 and `inception_block` 2 / 1 / 0**, tiling really fired,
and all three of those programs were refused by `npu-translate`. So the pass
alone table's zeros and the wired table's zeros agree on the number and disagree
completely on the reason, and only the second reason is about this compiler.

**The allocator's own figures at the wiring commit**, which is the other half of
a wiring check, reproduce ADR 0008's recorded table exactly: `resnet_block` tight
is 6432 peak with 1 spill and 3 DMA, `inception_block` tight is 6144 with 3 and
8, and the other five spill nothing at either budget.

### The same table again at the tip, which is what the compiler does today

Both halves of D-0052 are in and D-0056's allocator fix is in, so the decline
rule keeps only what the format still cannot express and the search's answers
reach the suite. Measured with `npu-opt` over each model's tensor level IR
through `builtin.module(npu-O2{budget=N})`, reading the pass's own statistics,
as tiled / already fitting / declined:

| Model | Tight budget | at the default 1048576 | at its tight budget | tight, `-npu-double-buffer` ablated | tight, `-npu-fuse-ops` ablated |
|---|---|---|---|---|---|
| `lenet` | 194624 | 0 / 1 / 0 | 0 / 1 / 1 | 0 / 1 / 1 | **1** / 4 / 0 |
| `depthwise_separable` | 8192 | 0 / 0 / 0 | 0 / 0 / 2 | 0 / 0 / 1 | **1** / 1 / 0 |
| `resnet_block` | 6464 | 0 / 1 / 0 | **1** / 0 / 1 | 0 / 1 / 1 | **2** / 0 / 0 |
| `inception_block` | 6144 | 0 / 3 / 0 | **2** / 1 / 0 | 0 / 3 / 0 | **2** / 1 / 0 |
| `conv_bn_relu_stack` | 6464 | 0 / 1 / 0 | 0 / 1 / 2 | 0 / 1 / 1 | **1** / 2 / 0 |
| `dilated_stack` | 8064 | 0 / 0 / 0 | 0 / 0 / 2 | 0 / 0 / 1 | **2** / 0 / 0 |
| `lenet_batched` | 200832 | 0 / 1 / 0 | 0 / 1 / 1 | 0 / 1 / 1 | **1** / 4 / 0 |

**Three things this table says and the previous one could not.**

- **The default budget column has not moved through any of it**, on any model,
  across three trees. That is the line the declaration drew and it is the one
  that would have said a wiring defect.
- **Two models tile at their published tight budgets** and every one of the seven
  tiles once `-npu-fuse-ops` stops hiding its convolutions, which is where the 31
  moved cells come from and which makes the fusion finding below a measurement
  rather than a caveat.
- **Every remaining decline is a fused region over budget**, counted and
  remarked on since `515560c`. The D-0052 declines are gone, because the format
  can express the shape now.

**The allocator's figures at the tip, on the two models that tile at their own
budgets**: `resnet_block` places with 1 spill over 16 buffers where the untiled
program placed 14 with 1, and `inception_block` places 16 buffers with **no**
spill where the untiled program placed 17 with 3. That is the trade in one line:
tiling costs `resnet_block` buffers it did not need and buys `inception_block`
its spills back.

**`-npu-double-buffer` fires on nothing in any column.** `prefetched` is 0 and
`not-hoisted` runs from 5 to 12 per model, at both budgets, on all seven, which
is D-0054 and is a different zero from the tiling coupling above it. The
`-npu-double-buffer` ablated column reads `not-hoisted` 0 rather than 5 to 12,
which is the pass not being in the pipeline rather than the pass answering no,
and having those two read differently is what the statistic is for.

### A second finding the prediction did not make, and it is the more interesting one

**At `-O2` most convolutions are inside `npu.fused_op` regions and the tiling
pass does not see them.** The pass walks the function's own body and leaves a
fused region alone, deliberately, because tiling one member of a fused pair
would put the intermediate back in DRAM and undo what the fusion was for. The
consequence was not predicted:

| Model | compute ops at `-O0` | fused regions at `-O2` | ops the pass sees at `-O2` |
|---|---|---|---|
| `lenet` | 5 | 4 | 1 |
| `depthwise_separable` | 2 | 2 | **0** |
| `resnet_block` | 2 | 1 | 1 |
| `inception_block` | 3 | 0 | 3 |
| `conv_bn_relu_stack` | 3 | 2 | 1 |
| `dilated_stack` | 2 | 2 | **0** |
| `lenet_batched` | 5 | 4 | 1 |

**Two models have no visible compute operation at `-O2` at all.** So even at a
budget where tiling would fire, `-O2` would tile strictly less than `-O0`, and on
those two it would tile nothing whatever the budget.

**That is P10's `-canonicalize` finding in a new place**, and it is worth saying
in the same words: a leave one out ablation cannot see a pass whose work another
pass has made invisible. Here it is not that another pass did the work, it is
that another pass hid the operand, and the effect on the ablation table is the
same: a row of zeros that means something quite different from a pass with
nothing to do.

### Where tiling does start to fire, which is what Section 13.3 needs

Sweeping the budget down at `-O0`, tiled / fitting / declined:

| Model | tight | 6000 | 4000 | 3000 | 2048 |
|---|---|---|---|---|---|
| `resnet_block` | 0 / 2 / 0 | **2** / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| `conv_bn_relu_stack` | 0 / 3 / 0 | **1** / 2 / 0 | 1 / 2 / 0 | 2 / 1 / 0 | 2 / 1 / 0 |
| `inception_block` | 0 / 3 / 0 | 0 / 3 / 0 | **2** / 1 / 0 | 3 / 0 / 0 | 3 / 0 / 0 |

**The threshold sits just below the tight budgets**, between 6000 and 6464 for
two models and between 4000 and 6000 for the third. That is the range Section
13.3's experiment has to sweep to have a subject at all, and it is now a measured
range rather than a guess. ADR 0010 already pointed the same way by recording six
cells that cannot allocate at batch 4.

## The numbers the next session needs and should not re-derive

**The tight budget spilling numbers a tiling disabled ablation row has to
reproduce to the cycle.** They were read out of the committed `-O2` cells at the
wiring commit, before tiling reached the suite, and since `80ac24d` they are what
the `-ablate-npu-tile-to-scratchpad` cells read rather than what the baselines
read. The two `-O2` baselines this table gives for `resnet_block` and
`inception_block` at their tight budgets have moved, to 21 instructions and
2660.0 cycles and to 3395.0 cycles with no spills; the ablated cells have not,
and a test asserts it:

| Model | instructions, default to tight | cycles, default to tight | spills, buffers and DMA |
|---|---|---|---|
| `lenet` | 25 to 25 | 17766.25 to 17766.25 | 0, 0 |
| `depthwise_separable` | 12 to 12 | 1324 to 1324 | 0, 0 |
| `resnet_block` | **14 to 17** | **1626 to 2018.0** | **1, 3** |
| `inception_block` | **14 to 22** | **2398.5 to 3799.0** | **3, 8** |
| `conv_bn_relu_stack` | 15 to 15 | 1160.5 to 1160.5 | 0, 0 |
| `dilated_stack` | 12 to 12 | 1234.0625 to 1234.0625 | 0, 0 |
| `lenet_batched` | 25 to 25 | 20000 to 20000 | 0, 0 |

**Only two models spill at their tight budget** without tiling, which is ADR
0008's own finding and was the whole population the three arm experiment had to
say something about at a budget below the peak. **With tiling in the level that
is no longer the whole population**: `inception_block` loses all three of its
spills and the five models whose convolutions fusion hides tile as soon as
`-npu-fuse-ops` is ablated. ADR 0010 records that the remaining five do not
allocate at batch 4 under their recorded tight budget at all, and names tiling as
the remedy, so P13 is also the phase that can make those six cells exist.

**Where the counts were hardcoded, all moved together in the wiring commit**, and
there were **nine** sites rather than the six this file predicted. The six that
were listed: `test/Python/test_benchmarks.py` at the assertions for 8, 63, 112
and 175 and the `checked == 112` and `cells_total == 175` ones; `docs/NUMBERS.md`
cell count table; `report/generated/macros.tex` through `results_to_tex.py`;
`test/Pipeline/opt-levels.mlir` flat `-O2` argument list and `DESCRIBE` sequence;
`test/Pipeline/pass-stats.mlir` ordered name list; `docs/PASSES.md` pass table
and its "eight ablatable" paragraph.

**The three the list did not have, and every one of them is a tripwire that
worked.**

- `test/Python/test_compile_driver.py::test_the_level_table_comes_from_the_compiler`,
  the `-O2` pass list by name and in order.
- `test/Python/test_pass_instrumentation.py::test_the_ablatable_set_is_read_from_the_compiler_and_is_not_written_here`,
  whose own docstring says a pass added to `-O2` and marked ablatable makes it
  red as a prompt to add the row and the entry. It did exactly that.
- `test/Python/test_predictions.py::test_at_least_one_committed_result_names_a_prediction`,
  which counts the cells naming the ablation prediction: 126 to 168, of which
  the ablation rows are 112 to 154 and the fourteen baselines did not move.

**No arithmetic in `run_benchmarks.py` changed**, which is what the previous
handoff said and it held: it reads the ablatable set out of the driver at run
time and the 63 and 154 are computed. Its **prose** changed, which is the tenth
site and was named.

**Two more present tense claims were swept out of the tree** and are recorded
here so a later reader knows they were considered rather than missed:
`docs/adr/0010`, which stated 8, 112 and 175 as though the ADR fixed all three
when it fixes only the 63, and `docs/BREAKING_CHANGES.md`'s P10 entry, which
described a test as running "each of the eight ablatable passes" when the test
sweeps whatever the driver reports. Historical statements inside those documents
were left exactly as they were.

## The Section 2 carve out, for the owner

**Re-derived at P13, in the commit that made it derivable, and it is the owner's
to apply because nothing in this project may edit the specification.**

**What Section 2 says today.** 7 models times 3 levels times 2 budgets times 2
batches is 84 benchmark cells; 11 ablatable `-O2` passes times 7 models times 2
budgets is 154 ablation cells; 238 in total, at 15 seconds per cell as a
planning figure, giving 59.5 minutes against a stated budget of 90.

**What the repository measures.** 63 benchmark cells, 154 ablation cells, **217
in total**, at **1.14 seconds per cell**, 4.12 minutes, serially, on an Intel
Core i7-14700K under WSL2 with nothing else running.

**One of the two disagreements is now closed and the other is a real decision.**
The **ablation** half agrees exactly: 11 ablatable passes and 154 ablation cells,
because the three passes Section 12's table names went into `-O2` at P13 and the
ablatable set is read out of the driver at run time. The **benchmark** half does
not, 63 against 84, and that is ADR 0010: a tight budget is defined as the
smallest budget at which **that program** allocates, and a model at batch 4 is a
different program from the same model at batch 1, so budget and batch are not a
free cross product. Six of the seven models do not allocate at batch 4 under
their recorded tight budget at all.

**The drop in replacement paragraph, for Section 2.**

> **Benchmark cells:** 7 models times 3 levels times 3 budget and batch
> combinations equals **63**. The combinations are not a free product of 2
> budgets and 2 batches: a tight budget is the smallest at which that program
> allocates, measured from its own peak, and a model at another batch is another
> program, so a cell that names the tight budget runs at the model's declared
> batch. `docs/adr/0010` records that decision and the peaks behind it.
> **Ablation cells:** 11 ablatable `-O2` passes, Section 12 defines the property
> and Section 16.2 reads the set from the driver, times 7 models times 2 budgets,
> at `-O2` and batch 1, equals **154**. Total **217 cells**. Each cell costs one
> instrumented compile, one encode, one simulation, one onnxruntime reference
> run, plus the `n_trials = 10` whole pipeline repetitions Section 16.1 requires
> for the timing object. **The measured cost is 1.16 seconds per cell**, 217
> cells in 4.19 minutes serially on the reference machine, so **the 90 minute
> budget stands with a factor of twenty one in hand** and the 15 second per cell
> planning figure is replaced by the measurement, per the rule at the end of this
> section.

**Why the number is worth applying now rather than at the end of the phase.**
The 217 is not going to move again inside P13: `-npu-calibrate` is P14's and is
never in a default `-O` level, and the model suite is fixed. The 1.16 seconds
will move with the host and is a wall clock, which is why the paragraph gives it
as a measurement on a named machine rather than as a property, and it has read
1.14, 1.17 and 1.16 across three re-records of the same 217 cells.

## Verification output

Every command run at the tip of this branch, from `/home/elijah/npu-mlir-v2`, in
`~/npu-venv`.

**Everything was run, and the quiet machine was spent here.** The suite is 217
cells rather than 175, so every measurement over it is a measurement of a
different population, and the whole set has been re-run **three times**,
serially, with nothing else on the machine: at the wiring commit, where nothing
moved; at the tree where the compiler half of D-0052 lands, where 31 tight
budget cells moved; and at the tree where `-npu-double-buffer` starts firing,
where 138 cells renamed a transfer and 57 repacked an arena and nothing counted
moved at all.

**Three is the right number and here is why.** Each re-record is the one run of
the tree whose numbers it records, and nothing after any of them moved a
measured quantity: the code was final before each run, and what followed was
documentation, the baseline record and the CI evaluation.
`regression-baseline --check` at the tip reports no drift, which is the gate that
would say otherwise.

**The second re-record took three attempts and the two that failed are recorded
rather than retried into silence.** All three died or completed on the
`--mlir-timing` cross check at the **upper** bound, which is D-0049 rather than
D-0043: the first on a machine with nothing running but a one minute load average
still around 3 from the builds seconds earlier, and two more on a machine whose
one minute load average was 0.01 and 0.03, missing by 0.0104 and 0.0059
milliseconds. **No bound was touched**, and the arithmetic observation that fits
both idle machine misses is written down in D-0049 and deliberately not acted on.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build-ndebug -j6` | clean, no warnings |
| `ninja -C build check-npu` | **37 discovered, 37 passed**. 32 at the previous handoff, plus `test/Pipeline/p13-passes-at-o2.mlir`, `test/Encoding/tiled-result-returned.mlir`, the two the allocator fix added, and `test/Dialect/NPUISA/double-buffer-unknown-option.mlir` |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | **20 passed**. 12 at P12, plus the eight tiled implementation tests |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | **84 passed**, 1 skipped. 76 before, plus the five `SpillSlotCoverage` cases and the three the slot lookup change added: a two tile assembly read whole and accepted, a read of bytes no tile wrote refused, a read across two slots refused for leaving its region, the scratchpad's no merge rule still refusing, and a whole spilled buffer still passing |
| `build/bin/NPUSimulatorTests` | **58 passed**, 1 skipped. 55 at P12, plus the per fold assertion, the version 2 strided transfer and the layout crossover |
| `build-ndebug/bin/NPUSimulatorTests` | 58 passed, 1 skipped |
| `build-ndebug/bin/NPUEncodingTests` | 84 passed, 1 skipped |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **1093 passed, 18 skipped**. 1085 before this checkpoint, plus the seven cases that pin the baseline's failing test names and the one that pins the walk's reading of an asynchronous transfer. Four runs of the suite during this checkpoint reported 1092 and one failure instead, and every one of them is D-0049 on a machine still draining: the two tests involved are green in isolation, six times out of six between them |
| `mypy` | no issues found in 26 source files |
| `black --check .` | 66 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, **539 of 539** files. 489 at the P12 handoff, plus the 42 new result cells, the five new lit tests, `unittests/Encoding/SpillSlotCoverageTest.cpp` and the two halves of `ScratchpadLiveness` |
| `pre-commit run` | all twelve hooks passed, on every commit of this branch |
| `python scripts/build-model-ir.py` | 84 IR files written |
| `python scripts/check-reachability.py` | pass, all five layers, no exemptions in force |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date, regenerated over the 217 cells |
| `python scripts/patch-scalesim.py --check` | every edit in place, exit 0 |
| `bash scripts/regression-baseline.sh --check` | **no drift**, exit 0, 21 golden tensors byte identical, in both shapes. It took two reds to get there and **neither was a cell**. The first was `test_a_rerun_reproduces_the_external_fields_too` reporting 1092 against 1093, which is D-0049 on a machine another workload had arrived on, and the second was `check-npu` reporting 37 against the baseline's 36, which is the option test this checkpoint added and is answered by recording the baseline again rather than by an entry: a test that did not exist cannot have regressed. **Both reds named what moved**, which is the change this checkpoint made to the script |
| `bash scripts/coverage.sh 85 93 16 58` | C++ **86.0** PASS against 85, 5924 lines of 6891, branch 75.1; per tree **93.4165 / 19.3928 / 74.5156** PASS, exit 0. **The margin is 1.0 point, and it took a second reading to keep it.** The first run after the prefetch read 85.7, and the two reasons were separable: `peakOf` was exposed on the liveness header and called by nobody, and the two options that decide a decline had no test that mistyped them. The dead function is gone and `double-buffer-unknown-option.mlir` is the sibling of `alloc-unknown-option.mlir`. `scripts` moved from 16.1 to 19.4 because the baseline script's comparison half now has tests. **The thresholds do not move: they are gates.** It was 0.4 two commits earlier and went red at 84.9 while this checkpoint was working, and what closed it was not a threshold. The two named tiling baselines, `fixed` and `largest-fit`, were reachable only through a pass option no test set: about eighty five lines of search that ran on nobody's machine. Running all three strategies over the same IR asserts the regret the log had claimed and covers them at the same time. The validator's two caps have a unit test; its saturation branch still has none, because reaching it needs a file with more than sixty five thousand discontiguous runs in one access. **The threshold does not move: it is a gate.** `build-coverage/` has to be cleared before a run, because it holds gcov data for `lib/Simulator/CostModel.cpp`, which moved to its own library earlier in this phase, and gcovr errors on a source it cannot find rather than skipping it |
| the whole suite in the CI shape, four differences | **1080 passed, 31 skipped, 0 failed**, mypy clean under `--python-executable /usr/bin/python3`. **Predicted before each of the three runs and measured exactly all three times**: 1069 at the wiring commit, 1071 after the two ablation row tests, and 1080 here, which is the dev shape's 1093 less the same 13 the image has always skipped. The eight cases this checkpoint adds are environment independent, which is why the difference between the shapes is unchanged |
| `regression-baseline --check` in the CI shape | **no drift**, exit 0, with both environments named, the count difference printed as 1093 against 1080 and not compared, and three notes reported rather than swallowed |
| the same environment with `NPU_EXTERNAL_TOOLS=1` | `missing_tools` reports **all three**, `scalesim`, `accelergy` and `zigzag`, and `tools_reachable` is false, so the guards fail naming the variable rather than skipping. `test_external_tools.py` is 10 passed in that shape |
| `python experiments/roofline.py` | 217 cells, **728 layers, 220 memory bound, 508 compute bound**, every cell at or above its bound, tightest layer `node_conv2d` at 0.0006 headroom. It was 682 layers at the wiring commit and the 46 extra are the tiles; the prefetch adds none, because an asynchronous transfer is the same transfer |
| `python experiments/scalesim_export.py` | 217 cells, worst whole model divergence -87.14% on `dilated_stack-O0-tight` at coverage 0.711, tau b **0.6143** over cells and **0.7242** over layers |
| `python experiments/accelergy_energy.py` | 217 cells at 45nm, 49.2860 pJ per MAC against a published 4.60, a factor of 10.71, unchanged |
| `python experiments/kernel_threads.py` | **0.98 to 3.75 times** at 28 threads, **output bytes equal on every model at every thread count**, at this tip on a quiet machine. It has read 0.96 to 4.00, 0.90 to 3.98, 0.84 to 3.68 and 0.98 to 3.75 across four runs of the same seven programs, which is the host rather than the kernel: `instruction_count` moved on **none** of the 217 cells, so the stream this script hands the kernel is the same stream with one transfer earlier in it, and the loop nest and the team cap are untouched |
| `python experiments/compile_time_benchmark.py --check` | fitted exponent **1.1093** against a ceiling of 1.5683, exit 0. It has read 1.1081, 1.1072, 1.0982, 1.1136 and 1.1093 at five trees of this branch, which is the spread of a slope fitted inside one process rather than a trend |
| `python experiments/compile_time_benchmark.py --check --sizes 500` | **red**, "No fit: a growth exponent needs at least two sizes", exit 1, which is the rehearsal P12's recipe asked for |
| `python experiments/run_benchmarks.py --force` | **217 cells, 4.19 minutes, 1.16 s per cell**, inside the 90 minute budget, worst upper `--mlir-timing` gap **0.1377 ms** at `NPUAllocateScratchpad` in `conv_bn_relu_stack-O0-default-n4-fp32-normal`, and **no red at either bound on any of the 217 cells**. One attempt this time, on a machine whose one minute load average was 0.22 before it started |
| `git status --short` | empty |
| the machine the measurements were taken on | **quiet for every one of them, and it stopped being quiet afterwards.** The 217 cell re-record, the baseline record and the four experiment scripts each ran with nothing else on the machine. A second project's solver sweep arrived at 03:00 and again at 03:13, and what is above it is the count based half of the verification, which a busy machine can slow and cannot change |
| `git log -p main..HEAD` grepped for tooling and authorship traces | 0 matches, case insensitive with word boundaries |
| the same diff grepped for em and en dashes | 0 matches |
| `git diff main..HEAD` over `lib/Simulator`, `include/NPU/Simulator` and the Python mirror | **empty over two of the three, and the third is named rather than counted as empty.** `include/NPU/Simulator` and `python/npu_frontend/cost_model.py` are byte identical to `main` across the whole branch, which is the claim about the **constants**. `lib/Simulator` is not: `Kernels.cpp` gained the version 2 scatter path at `670dd0b` and `CostModel.cpp` lost 92 lines at `e8910f2` when it moved to its own library. **This row said empty at the A4 tip and that was wrong**, in the direction that makes a claim stronger than the measurement; `CHANGELOG.md` had the correct sentence beside it the whole time. Corrected at checkpoint B, which touched none of the three |

**The CI shape has a fourth difference from P13 and the recipe above carries
it.** `zigzag` is a third module the image does not have, so the shim's meta path
finder blocks three roots rather than two. The prediction for the suite row was
written before the run: 1069 passed and 31 skipped, which is P12's 1063 and 31
plus the six new mirror cases, since those need no external tool and run in both
shapes and the one renamed test moves neither count. **The run measured exactly
that.**

**The branch is 298 files, 77490 insertions and 11656 deletions over `main`**,
which is what `git diff --shortstat 2f59429..HEAD` reports and which is mostly
the 217 recorded result cells and the regenerated report macros rather than
compiler code.

**The C++ coverage margin is 1.0 point and the row above carries the reading.**
It is named here rather than left as an unexplained dip. What the tiled
implementations left uncovered is the decline paths: `constantsOf` returning
nothing on a dynamic offset, and `windowSlice` refusing a tile whose every window
lies in the padding. **Neither is reachable from a test**, because every
iteration domain in this dialect is static and the verifier's rule that a pad is
smaller than its kernel makes the second impossible for a well formed operation.
They are guards against a caller this dialect does not currently have, and the
alternative to an uncovered guard there is an unguarded assumption.

### Checkpoint B's own run, at `101cb17`

*The suite row was predicted before any of it was started, in both shapes, and
both were measured exactly. The machine was quiet for every measurement, and the
one measurement it was not quiet for did not happen: see the baseline row.*

| Command | Result |
|---|---|
| `ninja -C build -j6` | no work to do; checkpoint B changed no C++ |
| `build/bin/NPUInterfaceTests` | 23 passed |
| `build/bin/NPUTilingTests` | 20 passed |
| `build/bin/NPUAllocatorTests` | 29 passed |
| `build/bin/NPUEncodingTests` | 84 passed, 1 skipped |
| `build/bin/NPUSimulatorTests` | 58 passed, 1 skipped |
| `ninja -C build check-npu` | **37 of 37**, unchanged: this checkpoint adds Python tests and no lit tests |
| `python -m pytest test/Python -q -m 'slow or not slow'` | **1131 passed, 18 skipped**, in 219.92s. **Predicted 1131 and 18 before the run**, which is the 1114 the baseline check measured at `019847e` plus the 17 cases of `test_zigzag_same_mapping.py` |
| the whole suite in the CI shape | **1116 passed, 33 skipped**, in 174.14s, mypy clean under `--python-executable /usr/bin/python3`. **Predicted 1116 and 33 before the run.** The difference between the shapes moves from 13 to **15** for the first time since the shim was written, and the two extra are `test_the_workload_form_is_the_installed_parsers_own` and `test_zigzag_evaluates_the_ordering_it_is_handed`, which take `pytest.importorskip("zigzag")` and skip where the module is blocked |
| the CI shape's external tools step | exits **0**, prints "confirmed absent", and the loop names all three of `scalesim`, `accelergy` and `zigzag`. `test_external_tools.py` is 10 passed under `NPU_EXTERNAL_TOOLS=1` in that shape, so the guards fail naming the variable rather than skipping |
| `mypy` | no issues found in 26 source files. `experiments/` is outside its `files` list, which is why the new module cannot reach it |
| `black --check .` | 72 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant with version 3.3 |
| `bash scripts/coverage.sh 85 93 16 58` | **C++ 86.0 PASS** against 85, decoder branch coverage 91.6 percent over 202 branches, every Python tree at or above its threshold, exit 0. **No threshold moved.** The margin is the same 1.0 point A4 left it at, which is what a checkpoint that adds tests and no C++ should do |
| `python scripts/regression_baseline.py` | **recorded at `101cb17`, and then clean.** The first attempt **stopped rather than record**: the quiet gate waited 900 seconds and the machine never fell below the 0.30 one and five minute load averages A4 tightened it to. It ran 1080 seconds later at **0.04 and 0.28**. **21 golden tensors byte identical**, `git status` on `test/baseline/golden` empty, and the only movement in `baseline.json` is the pytest count and the 21 test names this checkpoint added. `--check` at the same tip on the same quiet machine: **no drift**, exit 0, 42 cells, largest movement against `-O0` **4.470e-08** |
| `python scripts/check-reachability.py` | pass |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `python experiments/results_to_tex.py --check` | `macros.tex` is up to date |
| `python scripts/patch-scalesim.py --check` | every edit in place |
| `python scripts/build-model-ir.py` | 84 IR files written, working tree **clean** afterwards. This is the check that D-0059's fix moved nothing: `build-model-ir.py` passes no budget, so the stage it writes is the stage it always wrote |
| `python experiments/zigzag_same_mapping.py --search` | 42 layers, 59.1 seconds, 747 MiB peak against an 8192 MiB ceiling, every layer's ordering verified against ZigZag's own evaluation |
| `git status --short` | empty |
| `git log -p main..HEAD` grepped for tooling and authorship traces | **0 matches**, case insensitive with word boundaries |
| the same diff grepped for em and en dashes | **0 matches** |
| the branch's commit messages and trailers | 0 matches for either, one author |
| `git diff --shortstat main..HEAD` | 403 files, 95294 insertions, 14715 deletions |
| the machine the measurements were taken on | **quiet for coverage, the two suite runs and the ZigZag comparison.** The baseline's first attempt is the exception and it is the row above: the gate refused a machine at 0.33 and stopped rather than record from it |

## What P12 measured, and still holds

`docs/NUMBERS.md` is the ledger. **Every figure below was P12's, over the 175
cell suite, and P13 has re-measured all of them over 217 cells**; the section is
kept because the P12 readings are what the P13 ones are compared against and
because the reasons beside them are properties of the design rather than of a
suite size. Five things worth repeating here.

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

### What P13 activates, which is one CI step, and the trigger that still has not fired

**This branch activates one CI step and no CI job.** The step is
`experiments/compile_time_benchmark.py --check` in `build-and-test`, wired at
`57d9c34` because its trigger fired at the third evaluation; the blockquote below
is that evaluation with its numbers. **It also widens one existing step's
assertion**, which is a third thing again and not an activation: the `external
cross validation` step still prints that it is off and still asserts the tools
are absent, and what changed is that it now asserts three rather than one.

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

> **Trigger fired at the third evaluation, and the step is on.** P12 recorded
> the trigger for wiring `experiments/compile_time_benchmark.py --check` into
> `ci.yml` as "P13, because tiling makes functions longer, which moves the
> crossover with the genuinely quadratic offset assignment scan toward the
> measured range". It was evaluated three times: at the wiring commit, where
> nothing tiled and no function moved; after the D-0052 validator fix, where the
> compiler half was held back and nothing moved again; and here, where tiling
> reaches the suite. **Seventeen cells have longer functions, the longest went
> from 25 instructions to 33, and the largest single growth is 12 to 30 on
> `dilated_stack` with fusion ablated.** The premise of the trigger is a longer
> function and there are seventeen, so it is wired.
>
> **Rehearsed red first**, which is P12's recipe and the reason to rehearse at
> all: `python experiments/compile_time_benchmark.py --check --sizes 500` prints
> "No fit: a growth exponent needs at least two sizes and a nonzero pass time at
> every one of them" and **exits 1**. `--check` at the four real sizes exits 0
> with a fitted exponent of 1.1072 against a ceiling of 1.5683.
>
> **Where it sits and why**: the `build-and-test` job, under `pull_request` and
> `push` to `phase/**` like every other step, after `check-reachability full`,
> which builds the model IR, and before `regression-baseline --check`, which is
> documented as running last because it needs everything the job has.
>
> **What it can and cannot catch, restated so a red is read correctly.** The
> curve is fitted over synthetic sizes inside one process on one host, so it is
> a slope and not a wall clock compared across machines. A red means the
> allocator's growth left the band between the two hypotheses Section 13.1
> names, and it does not mean the runner was slow.

> **Trigger evaluated a third time and still unfired:
> `experiments/kernel_threads.py` into `nightly.yml`.** P12 recorded it as "the
> first phase that changes the convolution kernel's loop nest, which is P13's
> tiling or P14's integer kernels". **The kernel's loop nest is untouched and so
> is its team cap**, which is `batch * outputChannels`: the tilings this pass
> chooses split output **rows**, so the number of independent output tiles an
> instruction has does not move. And `kernel_threads.py` measures each model at
> its default configuration, where **nothing tiles at all**, so the instructions
> it hands the kernel are the same ones. The table read 0.96 to 4.00, then 0.90
> to 3.98, then 0.84 to 3.68 over three runs at this tree, **with the output
> bytes equal on every model at every thread count in every one of them**, which
> is the host and not the kernel. **Not wired.**
>
> **What would fire it** is now sharper than "P13's tiling": a change to the
> kernel's loop nest, or a tiling that splits the output channel axis, which
> would move the team cap. P14's integer kernels are the next candidate. The
> recipe is unchanged: add it to `nightly.yml` beside `full-matrix`, not to
> `ci.yml`.

### 0a. The fault P13 injected, with the prediction written first

**The regression baseline's red branch, driven by a deliberately failing test.**
It is not a CI activation and needs no trigger; it is here because the branch's
red path was changed, and a message nobody has seen printed is a message nobody
knows is right.

**What changed.** `scripts/regression_baseline.py` records the identifier of
every test that failed in each suite and prints them beside any suite count that
moved, and the final line of a red run names what moved rather than always
naming a cycle count. D-0049 is why: a `--check` whose pytest suite reported
1084 passed where the tree has 1085 named neither the suite's failing test nor
its message, and the summary underneath it talked about cycle counts.

*Predicted:* `test/Python/test_a_deliberate_failure.py` with one failing test
makes `--check` print `suite pytest: failed 0 -> 1`, a line naming
`test.Python.test_a_deliberate_failure::test_the_check_names_the_test_that_failed`
as the test that failed, the same identifier again as a test added, and a final
line naming a test suite rather than a cycle count. `passed` does not move, no
cell line and no golden line, exit 1.

*Result:* every clause held **except the final line, which said "10 test suites
moved" where one had.** The summary was counting drift lines, and one red suite
writes a line per moved count, a line naming the failing tests and a line per
test added. It counts distinct subjects now, and
`test_the_summary_counts_subjects_and_not_lines` is the case that was wrong.
Re-run after the fix, on an idle machine, the injection gives eleven lines, the
failing test named among them, and

```
regression-baseline: FAIL. one test suite moved. A suite that moved is a test
that changed its answer, and the identifiers printed above are which ones. Fix
the test or the code; a baseline is never re-recorded around a red suite.
```

with exit 1. Restored, tree clean. **The prediction was wrong about the count
and the rehearsal is what found it**, which is the whole argument for driving a
message rather than reading it.

**And the first run of the new field caught a red nobody aimed it at.** Beside
the injected failure it named `test_the_run_fails_when_it_exceeds_its_budget`,
which is one of D-0049's own sightings and whose message was lost the first time
it went red.

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

**Ten new at P13, D-0048 to D-0057**, and the first of them is in this file's
sibling. Every defect before D-0047 was in code, in a test, or in a claim about
one; D-0047 was in the build. D-0048 is in the **defect log itself**, which is
the document the project relies on to be right about what is wrong with it.

**The other nine, one line each, with the full entries in
`docs/DEFECT_LOG.md`.** D-0049, the `--mlir-timing` upper bound assumes the
process had the processor and does not check it, **open** and handed to Section
17.9 at P15. D-0050, the binary format could not express a buffer written in
pieces, **fixed** by version 2. D-0051, `-cse` merged the tiling destination with
every other `tensor.empty` of its shape, **fixed**. D-0052, a tiled result
assembled in DRAM could not be read back, **fixed on both halves**, the format
half by region scoped coverage and the compiler half by the allocator change
D-0056 found. D-0053, an argument whose every use is a whole value slice was
never loaded, **fixed**. D-0054, `-npu-double-buffer` fires on nothing this
compiler emits, **half fixed**: the allocator half is in and the pass half is the
open one. D-0055, three handoffs quoted a `--mlir-timing` bound that is not in
the code, **fixed in the documents and no bound moved**. D-0056, tiling is
expressible and is not always an improvement, **resolved in the allocator**, which
was refusing a legal spill. D-0057, the roofline walk dropped an operand whose
type carries a strided offset, **fixed**.

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

## What is left in P13, in the order it should be done

**Six of the previous list's eight items are done and are the wiring commit**:
the wiring itself, the no `scf` assertion as a statement about the lowering, the
tiling disabled ablation row reproducing to the cycle, the layout delta as a
measured row, the two CI triggers re-evaluated, and the quiet serialized
re-record of all 217 cells. What follows is what remains, and the first entry is
new.

1. **D-0052, D-0054 and D-0056 are closed, and both P13 passes that can fire
   do.** The
   validator accepts a tiled assembly read back inside a declared spill slot and
   the allocator no longer refuses a legal spill, so all 217 cells compile and
   thirty one of them moved, all at the tight budgets. **Section 13.3's tiling
   arm has its subject**: the two models that tile at their tight budgets and
   the five whose fusion ablated rows tile, with one model helped and one cost.
   The two follow ups D-0056 records, a pass that consults the allocator and a
   pass that tiles a chain, are out of P13's budget and carry its evidence.
2. **Section 13.3's three arms**, at `-O2`, with arm one in two configurations
   for the two spill heuristics and arm two in two configurations for fusion on
   and off, over the measured budget range of 6000 to 6464 for two models and
   4000 to 6000 for the third. ADR 0008's tight budgets are re-measured at the
   wired tree rather than quoted from P12, and that re-measurement is this
   item's first step.
3. **The ZigZag comparison under the same mapping**, which is what
   `npu.tiling_choice` was recorded for. The mappings come from where the pass
   actually emits tiles, which is inside the suite now: the tight budget cells of
   `resnet_block` and `inception_block`, the fusion ablated rows of five more
   models, and whatever the swept budget range brings in.
4. **The predictions for both, committed before either experiment runs**, in the
   repository's prediction schema, which has a validator.
5. **D-0054 is closed on both halves and the pass fires.** The allocator half
   went in first and moved nothing. The pass half is the budget aware hoist:
   `npuisa.const` joins the prologue, and before committing a hoist the pass
   runs the allocator's own `assignOffsets` over the intervals the hoist would
   produce, at the same budget, strategy and alignment, and declines what will
   not place. **The frozen tight budgets did not move**, which is what the rule
   exists for, and the sweep line peak was tried as that rule first and is 56
   bytes short of it on `dilated_stack`. Declared before the commit and
   adjudicated clause by clause afterwards, with one of eleven clauses wrong.

**The one thing the wiring commit deliberately did not do** is move the suite's
recorded tight budgets. The tiling disabled row was only checkable at the
budgets the old numbers were measured at, so re-measuring ADR 0008's budgets is
an input to Section 13.3 rather than part of the wiring.

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

**It is not the bound P12 asked P13 to watch, and the distinction turned out to
be more than a distinction.** `cross_check_against_mlir_timing` has two bounds
pointing in opposite directions. The **deficit** bound catches the
instrumentation reading above MLIR, is derived from the print quantum, and is
D-0043's; it is `half_ulp_ms`, **0.0500 ms** at the four decimals of seconds
MLIR prints. The **upper** bound is `half_ulp` plus 50 percent of MLIR's own
figure, so it is a different number for every pass.

**The 0.1577, 0.1177 and 0.1856 this file recorded as a narrowing margin against
"D-0043's 0.2000 bound" are readings of the upper gap, and 0.2000 is not a bound
in this code at all.** That is D-0055, corrected here rather than carried a
fourth time. `run_benchmarks.py` prints `CrossCheck.worst_gap_ms`, which is the
upper direction, so three handoffs read one bound and named the other.

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

**The `--mlir-timing` gap has a new quiet measurement and it is read correctly
for the first time.** The 217 cell re-record measures a worst **upper** gap of
**0.2430 ms**, at `NPULowerToNPUISA` in `lenet-O2-tight-n1-fp32-normal`, green
against that pass's own allowance of `half_ulp` plus half of MLIR's figure. **No
cell of the 217 produced a red at the deficit bound**, which is D-0043's and is
0.0500 ms, and that is the statement the previous three handoffs were trying to
make about a number they were reading from the other bound. See D-0055.

**The first attempt at the re-record went red at the upper bound** with a gap of
0.6897 ms against 0.5000, on a machine whose one minute load average was still
around 3 from the builds seconds earlier. That is D-0049's fifth data point and
its clearest: the process table was empty and the machine was not idle. Ninety
seconds of settling and the run completed. **Do not respond to a red at either
bound by widening it.**

**The allocator growth benchmark is wired into CI now**, and it is this branch's
one activation. P12 recorded the trigger as "P13, because tiling makes functions
longer"; it was evaluated three times and fired at the third, where seventeen
cells have longer functions and the longest went from 25 instructions to 33.
`57d9c34` is the commit, the step sits in the `build-and-test` job after
`check-reachability full` and before the `regression-baseline --check` step that
runs last, and the red branch was rehearsed first with `--sizes 500`.

**The kernel's team cap is still unmeasured against tiling, and its own trigger
has not fired.** `experiments/kernel_threads.py` was re-run and its table moved
only as a host wall clock, with the output bytes equal on every model at every
thread count. The convolution kernel's loop nest is untouched on this branch and
so is its team cap of `batch * outputChannels`: the tilings this pass chooses
split output **rows**, so the number of independent output tiles an instruction
has does not move, and the script measures each model at its default
configuration, where nothing tiles. **Not wired**, and what would fire it is a
change to the loop nest or a tiling that splits the output channel axis.

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

**P13, continued. Not P14.** The gate is met: all seven clauses are met, and
the two that were open, Section 13.3's three arms and the ZigZag cross check,
were run at checkpoint B and are recorded with their predictions adjudicated
clause by clause. What follows is the plan as it stands after D-0048 and the nine defects
the wiring and the work after it found, which is different from the plan P12
handed over, and the differences are the point.

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

**The first four items of that plan are done, and this is what each of them
became.**

1. **The lowering.** `633c116`, and it is the per slice DMA rather than only the
   patterns. `test/Dialect/NPUISA/dma-boundaries.mlir` moved in the same commit
   and Section 8's count is argued in its own terms over the slices, which is
   what the item asked for.
2. **`-npu-double-buffer`**, over the tokens and **before allocation** per
   Section 5.1. Implemented, wired, and firing on nothing this compiler emits,
   which is D-0054 and is the one item of this plan that is still open.
3. **`-npu-assign-layout`** with the inverse transpose fold. Implemented and
   wired, and it answers NCHW at every extent in the suite, for the stride term
   reason rather than for a property of these models.
4. **All three ablatable and wired into `-O2`** per Section 12, with positive
   **and** negative lit tests each per the P9 pattern, and **the Section 2
   arithmetic re-derived in the same commit**: 11 ablatable passes, 154 ablation
   cells, 217 total.

**What is left, and is the critical path now.**

5. **The Section 13.3 experiment, all three arms per model**, which is the
   phase's reason to exist. The subject is no longer only the two models that
   spill: it is the 31 cells tiling moved, which are `resnet_block` and
   `inception_block` at their tight budgets and the tight budget rows of five
   more models wherever `-npu-fuse-ops` is ablated, plus whatever the swept
   budget range brings in. ADR 0010's six models that cannot allocate at batch 4
   are the population tiling can bring into existence, which would make the
   experiment wider rather than only deeper.
6. **The ZigZag cross check**, Section 16.5: export the chosen mapping in ZigZag
   and Timeloop mapping form and compare **cost under the same mapping**, not two
   totals from two mappers. Bound the exploration to the layers actually tiled,
   record wall clock and peak memory, and **if it cannot be made to fit inside
   this machine's memory ceiling, stop that step and record why. Never suggest
   editing the WSL configuration.** If it finds a materially better tiling, say
   so, quantify it, and record it as a follow up in `docs/DEFECT_LOG.md` with a
   reproduction; **do not retune the search to match**.

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

**The gate is met and the push is not this session's to make.** Pushing this
branch and opening a pull request for it are the owner's, and nothing in the
gate being met changes that. **Do not open a pull request for it here.**

The next command is the one that shows what the phase's two experiments
actually answered, because both of them contradict part of what was predicted
and the tables are where that is legible:

```
python experiments/three_arms.py --models resnet_block inception_block
python experiments/zigzag_same_mapping.py --models resnet_block --search
```

The first prints the six configurations per budget with the arm each belongs to,
and the crossover per model underneath. The second prints one line per tiled
layer with this project's cycles beside ZigZag's under the same mapping, and the
array fill on each side, which is the term that decides which way a row goes.
**The command that shows the mapping itself**, so the next session starts from
the artefact rather than from the prose:

```
ninja -C build check-npu    # test/Pipeline/p13-passes-at-o2.mlir
```

Its first case is a convolution over the budget whose result is the function's
own. It tiles, it lowers, it encodes and it runs.

**Four things to check before anything else.**

**The suite is 217 cells and the baseline is recorded at the tree where tiling
reaches it.** `regression-baseline.sh --check` at the tip reports no drift and
the 21 golden tensors are byte identical. Anything that moves a cell from here
needs its declaration first.

**The 217 cells have been measured twice, and both runs are commits.** The first
is the wiring commit, where nothing moved; the second is `80ac24d`, where 31
tight budget cells moved and every one of them was declared beforehand. If
something in Section 13.3 moves a number, the re-record repeats in its own commit
with the declaration before it.

**One CI trigger fired and one did not, and both were re-evaluated with numbers
rather than with an argument.** `compile_time_benchmark.py --check` waited on a
function getting longer; seventeen cells have one, so it is wired, with its red
branch rehearsed first at `--sizes 500`. `kernel_threads.py` waits on the
convolution kernel's loop nest changing and this branch does not touch it; the
table was re-run and moved only as a host wall clock, with the output bytes equal
on every model at every thread count.

**D-0049 is open and is the reason to run measurements on a quiet machine even
when the measurement is a test.** P13's own re-record went red at the upper bound
on a machine whose process table was empty and whose one minute load average was
3, and the entry has two further misses on a machine that was genuinely idle.
Wait for the load to fall before measuring anything, and capture the whole
failure rather than the last few lines of it.
