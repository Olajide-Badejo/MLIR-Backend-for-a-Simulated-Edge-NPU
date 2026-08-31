<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Passes

*Diataxis type: reference.*

Ground rule 12: a pass that changes behaviour updates this file in the same
commit as the code, not later. Every entry carries what the pass does, before
and after IR, what it refuses, whether it is ablatable, and its measured
ablation delta.

**On the ablation deltas.** Section 12 requires every entry here to cite a
**measured** delta rather than a qualitative claim. **Since P7 there is a
simulator to measure against**, and `Stats::cycles` is the number a delta would
be taken from; the harness that runs a pipeline twice and subtracts is still
Phase P10's, so no entry below quotes one yet. Each says so in its own row rather
than leaving the field
blank, because a blank field reads as "no effect" and an absent measurement is
not a measurement of zero. The two passes that exist today are both marked **not
ablatable** in Section 12's table anyway: removing either produces no program at
all, so an ablation of one measures nothing and fails the run for a reason that
has nothing to do with the pass.

The full pass table, including the eleven ablatable passes that arrive later, is
Section 12 of the build specification. This file describes the passes that
**exist**, and it grows as they land.

## The pass list, as it stands today

| Pass | Level | Ablatable | Phase | Status |
|---|---|---|---|---|
| `-npu-lower-to-npuisa` | all | no | P4 | implemented |
| `-npu-allocate-scratchpad` | all | no | P5 | implemented |

---

## The optimization levels

*Added at P8, in `lib/Pipeline/Pipeline.cpp`.*

A level is a named list of passes, and the list lives in C++ rather than in the
Python driver. Section 6 settles that and the reason survives the summary: the
`PassInstrumentation` of Section 16.2 has to sit on the `PassManager` that
actually runs the passes, so a pipeline assembled in Python out of one `npu-opt`
invocation per pass would be a different pipeline from the one under test.
`npu-compile` names a level, `lib/Pipeline` builds it.

| Level | Registered as | Passes | Status |
|---|---|---|---|
| `-O0` | `npu-O0` | `-npu-lower-to-npuisa`, `-npu-allocate-scratchpad` | implemented |
| `-O1` | not registered | canonicalize and constant fold, on top of `-O0` | P9 |
| `-O2` | not registered | the full set of Section 12 | P9 |

**`-O0` is "import and verify" and verification is not a row in the table.**
MLIR verifies every operation when it parses it and again after every pass, so
the level gets its verification from the pass manager. A row that ran a verifier
would be a second and weaker one beside the one that already runs.

**`-O1` and `-O2` are named and not registered.** Asking `npu-opt` for one is an
unknown argument. That is deliberate: a level registered with an empty pipeline
would run, produce `-O0`'s answer, and every ablation cell measured against it
would be measuring nothing. The description below still lists them, because a
driver that could not see `-O2` at all could not tell a level that is missing
from one that is empty, and `npu-compile -O2` says "arrives at P9" rather than
"unknown argument".

**Every entry carries an `ablatable` property and a missing one does not
compile.** Section 12 asks for that in those words. `PassEntry` has a single
constructor taking every field with no defaults, so a row written as
`{"npu-fuse-ops"}` is a build error. A default of `false` would quietly shrink
Section 16.2's ablation table by exactly the passes nobody thought about.

**Each entry also declares whether it eliminates dead code**, under the same
rule and for a second consumer. Section 17.3a's dead subgraph injection asserts
that a subgraph feeding nothing leaves the instruction count unchanged, which is
only true at a level whose pipeline holds a pass that removes it. The check
reads `eliminates_dead_code` out of the description rather than carrying a list
of pass names that would go stale the first time one was added. Both of `-O0`'s
passes declare `false`, so the set of levels that eliminate dead code is empty
at P8 and `-O0`'s form of the check is the opposite claim: the count grows by
exactly the instructions the injection brought.

**The description is readable at run time**, which is what Section 16.2 requires
of the ablatable set:

```
npu-opt --npu-describe-pipeline
```

prints every level as JSON with its passes, their `ablatable` flags, whether the
level is implemented, and the phase an unimplemented one arrives at. The flag is
handled before `MlirOptMain` because it asks a question about the compiler
rather than about a file and therefore takes no input.

**The level and its pass list are asserted to agree.**
`test/Pipeline/opt-levels.mlir` runs `--npu-O0` and the explicit
`--npu-lower-to-npuisa --npu-allocate-scratchpad` over the same input and diffs
the two outputs. Section 17.4 says a test that runs a hardcoded pass list
matching no optimization level enforces nothing; the converse obligation is this
one, that a level nobody compares against anything can drift from the passes it
claims to run.

The pipeline forwards the allocator's four options, so `npu-compile --budget`
reaches the allocator without the driver knowing which pass consumes it:

```
npu-opt model.mlir --npu-O0=budget=8192
npu-opt model.mlir '--npu-O0=budget=8192 strategy=interval'
```

---

## `-npu-lower-to-npuisa`

Lowers the `npu` tensor dialect to `npuisa` instructions on `memref`s in the
two memory spaces of Section 8. Implemented in
`lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp`, registered by
`mlir::npuisa::registerNPUISAPasses()`, and runnable from `npu-opt`.

**Ablatable: no.** Section 12's table marks it so. Removing it leaves a program
with no instructions in it.

**Ablation delta: not measured.** The harness lands at P10.

### What it does

The operator map, which is the pass's contract:

| `npu` | becomes |
|---|---|
| `constant` | `npuisa.const` in DRAM, plus the one `dma_load` that brings it on chip |
| `conv2d` | `npuisa.conv2d` |
| `matmul` | `npuisa.matmul` |
| `add` | `npuisa.add`, with a stride 0 view when the right hand operand is rank 1 |
| `mul` | `npuisa.mul`, the same way |
| `relu` | `npuisa.relu` |
| `max_pool2d` | `npuisa.pool_max` |
| `avg_pool2d` | `npuisa.pool_avg` |
| `reshape` | `npuisa.reshape`, with a destination the pass allocates |
| `transpose` | `npuisa.transpose` |
| `concat` | `npuisa.concat` |
| `batch_norm` | a multiply and an add over per channel constants computed at rewrite time |
| `fused_op` | no instruction: the region is flattened into its parent |
| `yield` | erased with the region it terminated |

Three rules govern the memory, and together they are Section 8's boundary
invariant:

1. A function argument becomes a `memref` in `#npu.dram`, and if the body reads
   it, exactly one `npuisa.dma_load` brings it into a scratchpad buffer. An
   argument nothing reads gets no load.
2. A function **result** becomes a trailing `#npu.dram` argument, and one
   `npuisa.dma_store` writes it. The lowered function returns nothing.
3. Everything else lives in `#npu.scratchpad`: a `tensor.empty` destination
   becomes a `memref.alloc` there and the instruction's `outs`.

**Every argument of the lowered function carries `npuisa.arg`**, a string
attribute holding `"in"` or `"out"`. *Added at P6.* The order is unchanged, the
model's arguments first and its outputs appended after them, so argument N of
the lowered function is still argument N of the model; what the attribute adds
is that the split is stated rather than counted. The encoder reads it to build
the input and output regions of the binary and refuses a function without it,
which turns a convention the compiler could not check into one it can.
`docs/ARCHITECTURE.md` carries the reasoning and the alternatives.

The attribute is written by this pass because this pass performs the split. A
function the idempotence guard finds already lowered is left alone and therefore
gains nothing, so hand written `npuisa` IR that wants to be encodable carries
its own.

### Before and after

Input, one convolution and one relu, in the shape the frontend emits:

```mlir
func.func @conv_relu(%x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %d0 = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d0 : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  %d1 = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.relu ins(%c : tensor<1x8x8x8xf32>) outs(%d1 : tensor<1x8x8x8xf32>)
       -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}
```

Output:

```mlir
func.func @conv_relu(%arg0: memref<1x3x8x8xf32, #npu.dram> {npuisa.arg = "in"},
                     %arg1: memref<1x8x8x8xf32, #npu.dram> {npuisa.arg = "out"}) {
  %alloc = memref.alloc() : memref<1x3x8x8xf32, #npu.scratchpad>
  npuisa.dma_load %arg0, %alloc
    : memref<1x3x8x8xf32, #npu.dram> to memref<1x3x8x8xf32, #npu.scratchpad>
  %cst = npuisa.const dense<1.000000e+00> : tensor<8x3x3x3xf32>
       -> memref<8x3x3x3xf32, #npu.dram>
  %alloc_0 = memref.alloc() : memref<8x3x3x3xf32, #npu.scratchpad>
  npuisa.dma_load %cst, %alloc_0
    : memref<8x3x3x3xf32, #npu.dram> to memref<8x3x3x3xf32, #npu.scratchpad>
  %alloc_1 = memref.alloc() : memref<1x8x8x8xf32, #npu.scratchpad>
  npuisa.conv2d ins(%alloc, %alloc_0 : memref<1x3x8x8xf32, #npu.scratchpad>,
                                       memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%alloc_1 : memref<1x8x8x8xf32, #npu.scratchpad>)
                {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 strides = array<i64: 1, 1>}
  %alloc_2 = memref.alloc() : memref<1x8x8x8xf32, #npu.scratchpad>
  npuisa.relu ins(%alloc_1 : memref<1x8x8x8xf32, #npu.scratchpad>)
              outs(%alloc_2 : memref<1x8x8x8xf32, #npu.scratchpad>)
  npuisa.dma_store %alloc_2, %arg1
    : memref<1x8x8x8xf32, #npu.scratchpad> to memref<1x8x8x8xf32, #npu.dram>
  return
}
```

Two loads in, one store out, and nothing between the convolution and the
activation. That is the invariant, and
`test/Dialect/NPUISA/dma-boundaries.mlir` asserts it.

### The rank 1 channel broadcast

`docs/adr/0005-channel-broadcast-on-add-and-mul.md` obliges this pass and P7
together: an `npu.add` or `npu.mul` whose right hand operand is rank 1 lowers to
a per channel broadcast, and the simulator kernel reads that operand with a
channel stride of one and a spatial stride of zero.

Here the contract is a **type** rather than a flag. The rank 1 buffer is viewed
at the destination's extents with strides that are 1 on the channel axis and 0
on every other:

```mlir
%cst = npuisa.const dense<2.000000e+00> : tensor<8xf32> -> memref<8xf32, #npu.dram>
%alloc_0 = memref.alloc() : memref<8xf32, #npu.scratchpad>
npuisa.dma_load %cst, %alloc_0 : memref<8xf32, #npu.dram> to memref<8xf32, #npu.scratchpad>
%view = memref.reinterpret_cast %alloc_0 to
          offset: [0], sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
      : memref<8xf32, #npu.scratchpad>
     to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
npuisa.mul ins(%alloc, %view : memref<1x8x4x4xf32, #npu.scratchpad>,
                               memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>)
           outs(%alloc_1 : memref<1x8x4x4xf32, #npu.scratchpad>)
```

The view has the destination's shape, which is what `npuisa.mul` requires of its
operands and why no dialect change was needed, and it reads the same C values
over and over, which is what a per channel scale means. It adds no transfer: the
`C` floats were loaded once and the view is over them. Buffers are NCHW below
the tensor level whatever the layout was, so the channel axis is always 1 and
this view is identical under both layouts.

**Why a view and not a new instruction.** A stride 0 read is an addressing mode,
not a machine operation, and Section 9.1 already carries operand strides in the
`Instruction` record for the layout decision. One field, two uses, and no new
opcode: the instruction set of Section 5.4 is closed and this stays inside it.
The obligation it places on P7 is exact: a compute kernel indexes its operands
through their strides, and the broadcast then costs it no special case.

### The batch norm decomposition

A batch norm the folding pass did not fold is **legal rather than a hard error**,
per Section 5.2. It decomposes into a multiply and an add, with the per channel
constants computed at rewrite time:

```
invStd = 1 / sqrt(variance + epsilon)
scale  = gamma * invStd
shift  = beta - mean * scale
y      = x * scale + shift
```

That evaluation order is the order the code evaluates in, and it is written down
because it is observable: floating point multiplication is not associative, so a
reader comparing this against onnxruntime needs to know which of several
algebraically equal forms produced the number.

Both halves come out as the rank 1 broadcast above, so there is one code path
for per channel arithmetic rather than two. The four parameter constants are
consumed at rewrite time and erased, so an unfolded batch norm costs two
transfers rather than four; see D-0016 for what happened before they were.

The decomposition happens on tensors, before the conversion, because it is a
tensor level identity. Written against memrefs it would have to allocate its own
intermediate and thread its own destination, which is work the conversion
already does once.

### `npu.fused_op`

The region is **flattened** into its parent: block arguments become operands,
the yielded value becomes the result, and the operations inside become ordinary
instructions. The operation is `IsolatedFromAbove`, so the block arguments are
the only route out of the region and the substitution is complete by
construction.

This is what makes fusion mean what Section 5.2 says it means. Afterwards the
chain's intermediate is an ordinary scratchpad buffer that no `dma_store` ever
writes, so a fused convolution and its activation have no DMA between them. It
is the same reason the property holds for an unfused chain, which is the point:
fusion's benefit under this memory model is that the intermediate stays on chip,
and flattening is how the lowering makes that true rather than a special case it
has to remember.

`-npu-fuse-ops` is the pass that creates these regions and it has not landed
yet. The handling is here anyway rather than deferred with it, because the P4
gate asks for no DMA between a convolution and its fused activation, and a
diagnostic would have met the letter of "do something named" while leaving the
gate unmeetable.

### The layout encoding becomes the strided layout map

Section 5.5: `#npu.layout` is a tensor encoding and does not survive
bufferization by itself, so the lowering materialises it as the memref's strided
layout map.

A tensor's extents are written in the order its layout names, so an NHWC tensor
is written `N, H, W, C`. A buffer below this level is always NCHW, which is what
every `npuisa` verifier reads and what the simulator's kernels index. So the
extents are permuted back into NCHW and the permutation moves into the strides,
over the same underlying bytes:

```mlir
// tensor<1x8x8x3xf32, #npu.layout<nhwc>>  becomes
memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
```

Over `1 x 8 x 8 x 3` elements laid out NHWC the channel stride is 1, the width
stride is 3, the height stride is 24 and the batch stride is 192. An NCHW tensor
gets no layout map at all, because the identity written out at length is still
the identity and printing it would make the NHWC case prove nothing.

Keeping the extents in the order the layout wrote them was the alternative, and
it is wrong for a reason worth recording: the strides come out contiguous again
and the layout leaves no trace below the tensor level, which is exactly the
outcome Section 5.5 says would make `-npu-assign-layout` a pass whose delta is
structurally zero.

### What it refuses, by name

Every refusal is emitted from a validation stage that runs before any operation
has been rewritten. A diagnostic emitted from inside a conversion pattern
competes with the framework's own "failed to legalize operation" message, and
the reader is then left with two messages and no way to tell which is the
answer. `test/Dialect/NPUISA/lowering-diagnostics.mlir` covers all of these.

| Refused | Because |
|---|---|
| an `scf` operation | Section 5.2: this instruction set has no branches, so a tiled loop is fully unrolled before lowering |
| a function with more than one block | the same rule, seen from the other side |
| a function declaration | this compiler has no calls and no linking |
| a tensor with a dynamic extent | nothing below this level can represent one |
| a tensor whose element type is not f32 or i8 | those are the two memories this machine has |
| a batch norm parameter that is not an `npu.constant` | the decomposition computes its multiplier and addend at rewrite time |
| a batch norm whose variance plus epsilon is not positive | the decomposition takes a square root of it |
| an `npu.constant` carrying a layout encoding | the lowering does not permute constant data; a layout assignment pass materialises the permuted constant itself |
| an `npu.transpose` that changes the layout as well as the extents | the lowering represents a permutation of extents only; a layout assignment pass folds its own inverse transposes |

The last two are unreachable from the frontend, which emits no layout encodings
at all. They become reachable when `-npu-assign-layout` lands, and they are
diagnosed now so that pass arrives at a stated rule rather than at a verifier
failure from inside a pass.

### Where it does not fire

Section 12's negative test rule: a pass with only positive tests is not
adequately tested, because a pass that fired unconditionally would pass them
all. Three cases in `test/Dialect/NPUISA/lowering.mlir`:

- **A function that is already lowered is left exactly as it is.** No second
  signature conversion, no second load, no second store. This also makes the
  pass idempotent, which is what lets a pipeline run it without first
  establishing whether something else already did.
- **A right hand operand that already has the destination's shape is not
  broadcast.** No `memref.reinterpret_cast` appears.
- **An argument nothing reads is not loaded.**

### Tests

| File | What it pins |
|---|---|
| `test/Dialect/NPUISA/lowering.mlir` | one case per pattern, plus the three negative cases |
| `test/Dialect/NPUISA/dma-boundaries.mlir` | Section 8's scoped invariant, immediately after this pass |
| `test/Dialect/NPUISA/lowering-diagnostics.mlir` | every refusal above, by the substring it emits |

An end to end test is the other half of the 17.1 row for a lowering pattern,
"plus an e2e test if it is reachable from ONNX". There is no end to end pipeline
until P8, so none of these patterns has one yet. Every pattern in the table above
except `fused_op` and `yield` is reachable from ONNX and therefore owes one at
P8; those two are structural, no ONNX node imports to them, and they are reached
by `-npu-fuse-ops` instead.

---

## `-npu-allocate-scratchpad`

Assigns every scratchpad buffer a byte offset in one flat arena, and spills to
DRAM when the offsets do not fit. Section 13.1 in full. Implemented in
`lib/Dialect/NPUISA/Transforms/AllocateScratchpad.cpp`, with the arithmetic in
`ScratchpadAllocation.cpp` beside it, registered by
`mlir::npuisa::registerNPUISAPasses()`, and runnable from `npu-opt`.

**Ablatable: no.** Section 12's table marks it so. Removing it leaves a program
whose buffers have no addresses, which is not a program.

**Ablation delta: not measured.** The harness lands at P10.

### What it does

Four steps, and the order of the middle two is the part that is easy to get
backwards.

1. **Liveness**, over the `memref.alloc` operations in `#npu.scratchpad` of a
   single block function. Each range runs from the allocation to the last
   operation that reads or writes that memref. **A use through a view is a use
   of the buffer**: the walk follows `memref.view`, `memref.subview`,
   `memref.reinterpret_cast` and `memref.cast`, so the rank 1 broadcast of
   ADR 0005 keeps its scale buffer alive for as long as the multiply reads it.
   Sizes come from the memref type and its element type, through the same
   `computeBufferRange` the overlap rule of Section 8 measures with, so the
   allocator and the aliasing analysis cannot disagree about how large a buffer
   is.
2. **The sweep line.** One event list of `(index, +size)` at each definition and
   `(lastUse + 1, -size)` at each death, sorted once and walked once. O(n log n)
   for the sort and O(n) for the walk. The first index with a strictly greater
   sum wins the peak, and at equal indices deaths are ordered before
   definitions.
3. **Offset assignment**, in one of two strategies, described below.
4. **Spilling**, if and only if step 3 failed.

**The peak is a lower bound, not a placement test, and the spill trigger is
"offset assignment failed".** Peak simultaneous live bytes is the smallest arena
any placement could possibly need. Fragmentation means a program whose peak sits
under the budget can still fail to place, so a trigger of "peak exceeded budget"
would spill when it need not and fail to spill when it must. The two questions
have different answers on a real program, and
`test/Dialect/NPUISA/spill-heuristic.mlir` carries the one that separates them:
a function whose peak is exactly the budget, which the packer places and the
interval scheme does not.

Every allocation is then replaced by a `memref.view` at a constant byte offset
over one flat `memref<Nxi8, #npu.scratchpad>`, which is Section 8's rule that
the offset is an SSA operand rather than a discardable attribute. `N` is the
high water mark, not the budget: the arena is what was used.

### The two offset assignment strategies

Selected with `strategy=`, and both are present because Section 13.1 asks for
the production algorithm and a named baseline to measure it against.

| Value | What it does |
|---|---|
| `pack` (default) | The greedy by size offset calculation algorithm [R28], the one TFLite Micro's arena planner ships. Buffers are placed largest first |
| `interval` | The named baseline: the same placement rule in definition order, which is the interval scheme this project started from |

Both share one placement rule, and the sharing is the point: the difference
between them is the order and nothing else. For each buffer, gather the already
placed buffers whose live ranges overlap this one, walk their occupied byte
ranges in increasing offset order, and take the first gap the buffer fits in,
rounding the candidate offset up to the alignment after every block it skips.

Ties break deterministically, per ground rule 16:

- `pack`: larger bytes first, then longer span, then earlier definition index.
- `interval`: earlier definition index, then larger bytes, then longer span.

Offsets are aligned to 64 bytes by default, because the array of Section 5.3 is
16 by 16 and consumes a row of 16 `f32` lanes at a time, which is 64 bytes. The
alignment is a pass option so a test can pin it. **Sizes are not padded**, only
offsets aligned, so the high water mark is the last byte genuinely occupied.

### The fragmentation ratio

`npuisa.fragmentation_ratio` is the assigned high water mark divided by the
sweep line peak, which Section 13.1 calls the headline allocator metric and
which TelaMalloc [R26], MiniMalloc [R27] and the TFLite Micro planner [R28] all
report. It is 1.0 when the placement achieves the lower bound.

The two integers it is computed from are written on the function beside it, so a
reader can check the ratio rather than trust it.
`experiments/allocator_fragmentation.py` reports it per model under both
strategies.

### The two spill heuristics

Selected with `spill-heuristic=`. The candidate set is the buffers live across
the pressure peak index.

| Value | The rule |
|---|---|
| `longest-range` (default) | The longest live range crossing the peak |
| `cost` | A Belady style rule: `cost = bytes * (1 + reloads)` where `reloads` is the number of uses strictly after the peak index, and the **smallest** cost is spilled |

Ties break deterministically. `cost` breaks them by larger bytes first, then
longer span, then earlier definition index, which is Section 13.1 word for word.
`longest-range` uses the same three keys with span promoted to the front, so
that an ablation between the two heuristics is not partly a measurement of two
different tie breakers.

**The default is provisional and is marked as such.** Section 13.1 requires the
default to be chosen with data from the ablation across the whole suite, and
that harness lands at P10 with the experiment at P13. `longest-range` is the
simpler rule and the baseline, so changing the default later will be a move
*towards* the smarter rule with evidence behind it rather than away from one.

### Spilling, and what it emits

Section 13.1's semantics exactly: a `npuisa.dma_store` after the definition and
a `npuisa.dma_load` before each later use, with the reload replacing that use.
This is the **second of the three permitted DMA producers** of Section 8, and
the count is written on the function as `npuisa.spill_dma_count` so the sum over
the three is checkable.

The reload gets its own `memref.alloc`, which participates in liveness like any
other buffer. That is why the pass recollects everything from the IR after every
spill instead of patching a side table: a second spill round that had not seen
the reloads would mis-size the peak, which Section 13.1 names as the failure.

A buffer is **spillable** only if all of the following hold, and each has a
reason rather than a convention:

| Rule | Why |
|---|---|
| exactly one writer | the semantics are a store after *the* definition, and a buffer written twice has two |
| no view of it | a view is a second SSA name for the same bytes, and rewriting only the direct uses would leave the view reading a buffer whose contents had moved |
| not a reload, and not already spilled | both would let the loop spill its own output, which is how a spill loop fails to terminate |
| at least one read after the write | spilling a buffer nothing reads later adds a transfer and shortens no live range |
| an identity layout | `dma_store` requires its operands to agree, and a permuted buffer has no DRAM counterpart without deciding what order to write it in. That decision belongs to the relayouting transfer Section 12 marks as a future extension |

**The spill slot is a `memref.alloc` in `#npu.dram`**, marked with
`npuisa.spill_slot`. This is the one place in the compiler that allocates DRAM,
and it amends the P4 sentence in `docs/ARCHITECTURE.md` that nothing below the
tensor level does; the amendment is recorded there as a marked P5 extension.

### The function attributes it sets

| Attribute | Meaning |
|---|---|
| `npuisa.scratchpad_budget` | the budget the allocation was made against |
| `npuisa.scratchpad_bytes` | the arena actually used, which is the assigned high water mark |
| `npuisa.scratchpad_peak_bytes` | the sweep line peak, the lower bound the above is measured against |
| `npuisa.fragmentation_ratio` | the first divided by the second |
| `npuisa.spill_count` | buffers spilled |
| `npuisa.spill_dma_count` | DMA operations this pass inserted |

The budget comes from the `budget` pass option if it was given, then from the
function's `npuisa.scratchpad_budget` attribute, then from the default of
1048576 bytes. The option wins over the attribute because the option is a
command line override and the attribute is data the driver wrote. The attribute
is written back either way, so after this pass a function always says which
budget it was allocated against.

**One mebibyte is inherited, not invented.** It is the budget the previous build
of this project called the default and reported every generous budget cell at,
and Section 15 requires each model's tight budget to be a fraction of the peak
observed at the default. Moving it silently would move every tight budget cell
in the project's history at once.

### Before and after

Input, a three buffer chain of 1 x 8 x 4 x 4 `f32`, which is 512 bytes apiece:

```mlir
func.func @chain(%in: memref<1x8x4x4xf32, #npu.dram>,
                 %out: memref<1x8x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
  %c = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%c : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %c, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
```

Output. Three buffers, two offsets: `%c` takes `%a`'s bytes back, because `%a`
died at the first relu.

```mlir
func.func @chain(%arg0: memref<1x8x4x4xf32, #npu.dram>,
                 %arg1: memref<1x8x4x4xf32, #npu.dram>)
    attributes {npuisa.fragmentation_ratio = 1.000000e+00 : f64,
                npuisa.scratchpad_budget = 1048576 : i64,
                npuisa.scratchpad_bytes = 1024 : i64,
                npuisa.scratchpad_peak_bytes = 1024 : i64,
                npuisa.spill_count = 0 : i64,
                npuisa.spill_dma_count = 0 : i64} {
  %alloc = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<1024xi8, #npu.scratchpad>
  %c0 = arith.constant 0 : index
  %view = memref.view %alloc[%c0][]
        : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %arg0, %view
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %c512 = arith.constant 512 : index
  %view_0 = memref.view %alloc[%c512][]
          : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%view : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%view_0 : memref<1x8x4x4xf32, #npu.scratchpad>)
  %c0_1 = arith.constant 0 : index
  %view_2 = memref.view %alloc[%c0_1][]
          : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%view_0 : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%view_2 : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %view_2, %arg1
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
```

A buffer whose type carries a strided layout map, which is what an NHWC tensor
lowers to, gets the view at its extents followed by a `memref.reinterpret_cast`
that restores the layout, because `memref.view` requires an identity layout on
its result. The bytes are the same bytes: a permutation layout spans exactly the
contiguous extent its shape does.

### What it refuses, by name

| Refused | Because |
|---|---|
| a function with more than one block | Section 8. Liveness here is an ordering of one straight line stream, and an index in a second block is not comparable with one in the first |
| a budget too small even after everything spillable has been spilled | Section 13.1. The message carries the budget, the size of the buffer that could not be placed, the offset it wanted, and the sweep line peak as a lower bound, so a reader can tell whether a larger budget could ever help |
| an unknown `strategy` or `spill-heuristic` value | Section 13.1: a typo must not silently select a heuristic nobody asked for. The message names the offending string and lists the accepted values |
| an alignment that is not a positive power of two | the rounding is a mask |
| an allocation whose byte size cannot be computed | Section 13.1 takes sizes from the type, so a buffer it cannot measure is refused rather than guessed at |
| a malformed `npuisa.scratchpad_budget` attribute | a silent replacement by the default would produce a valid program allocated against a budget nobody asked for, and the number would travel into a result cell as though it had been measured |

**Every bad option is reported, not just the first**, so somebody who mistyped
two of them does not fix one, rerun, and discover the second.

### Where it does not fire

Section 12's negative test rule.

- **A function that has already been allocated is left exactly as it is**, which
  is the idempotence guard. It is not decoration: the arena is itself a
  scratchpad allocation, so a second run without the guard would allocate an
  arena for the arena and grow the program every time the pipeline ran. The
  guard is the `npuisa.scratchpad_bytes` attribute.
- **A function with nothing to allocate gets the attributes and no arena.** Zero
  is written down rather than left absent, because an absent attribute and an
  attribute of zero are different claims and only one of them says the allocator
  ran.
- **At a budget nothing needs spilling for, nothing is spilled.** A spiller with
  only positive tests would pass them all while spilling unconditionally, and
  the cost of that would be invisible in a lit file and very visible in the DRAM
  traffic numbers three phases later.

### Tests

| File | What it pins |
|---|---|
| `test/Dialect/NPUISA/scratchpad-alloc.mlir` | the offsets, by value, under both strategies: fits, reuse after death, no reuse while live, alignment rounding, fragmentation, the strided layout, the broadcast view, and the two negative cases |
| `test/Dialect/NPUISA/spill-heuristic.mlir` | the spill trigger, the emitted store and reload, the two heuristics choosing differently on one program, and the negative case at a generous budget |
| `test/Dialect/NPUISA/alloc-budget-too-small.mlir` | the budget diagnostic, under `-verify-diagnostics` |
| `test/Dialect/NPUISA/alloc-multiblock.mlir` | the multi block, unmeasurable size and malformed budget refusals |
| `test/Dialect/NPUISA/alloc-unknown-option.mlir` | all three option refusals at once |
| `unittests/Dialect/NPUISA/AllocatorTest.cpp` | Section 17.2's property test, the placement invariant over the same randomized sets, and every tie break |
| `experiments/compile_time_benchmark.py` | the growth curve at 500, 1000, 2000 and 5000 operations |
| `experiments/allocator_fragmentation.py` | the fragmentation ratio per model under both strategies |
