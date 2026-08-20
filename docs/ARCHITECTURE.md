<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Architecture

*Diataxis type: explanation.*

This file explains why the compiler is shaped the way it is. It is not a
reference and not a tutorial: it does not enumerate every operation, and it does
not tell you how to build anything. What it does is record the decisions that
later phases are not free to relitigate, together with the reasoning that makes
them binding rather than habitual. Where a decision has a cost, the cost is
written down next to it, because a design document that lists only advantages is
a document nobody can use to make the next decision.

Entries arrive with the phase that makes the decision. This file grows; it does
not get rewritten.

## The memory model

*Added at P2, with the `npuisa` dialect. Binding on every later phase.*

### Two memory spaces, on the builtin `memref` type

A scratchpad value is `memref<SHAPExTYPE, #npu.scratchpad>` and a DRAM value is
`memref<SHAPExTYPE, #npu.dram>`. Both attributes are defined once, by the `npu`
dialect at P1, and the `npuisa` dialect reuses them rather than defining a second
pair. One machine has one scratchpad, and two spellings of it would be two things
somebody has to keep equal by hand.

The consequential half of this is the choice of `memref` with a memory space
attribute over a bespoke buffer type carrying an address attribute painted on
each instruction. It is the single most consequential design decision in the
project, and the argument for it is about what it makes reachable rather than
about elegance. `memref` puts this project on MLIR's bufferization and memory
space infrastructure rather than beside it, which is the prerequisite for tiling,
for quantization's non `f32` element types, and for any real hardware target
later. Building it on day one costs a phase. Retrofitting it costs six.

The cost, stated plainly: every operand type in the dialect is now a predicate
over a memref rather than a named type, the predicates are generated per space
and per element type set, and a bug in one of them is a bug in several. P2 paid
that cost once, in a way worth recording because it is exactly the failure mode
this shape invites. The space predicates were written with `llvm::isa`, and a
memref written without a memory space, `memref<4x4xf32>`, has a **null** memory
space attribute. `isa` on a null attribute is undefined behaviour, and in a build
without assertions it read through a null pointer: `npu-opt` died with a
segmentation fault and no diagnostic at all. `isa_and_present` is the correct
idiom and answers false for null. The defect is D-0008 and the regression tests
are the default space cases in `test/Dialect/NPUISA/invalid.mlir`.

### Destination passing, and why the interfaces are not decoration

Compute instructions are destination passing: `ins` operands are read, one `outs`
destination memref is written, and there are no results. `npuisa.const` is the
single exception and it is the reason for the rule rather than a violation of it,
because a constant is where a buffer comes from and not a transformation of one.

`DestinationStyleOpInterface` gives the operand partition. The destination is
always the last operand and there is always exactly one, which is the simplest
consecutive range that cannot be got wrong, and it is the same convention the
`npu` dialect uses one level up.

`MemoryEffectOpInterface` is the load bearing half at this level, and the reason
is mechanical rather than stylistic. These operations have no results. An
operation with no results that reports itself free of memory effects is an
operation `isOpTriviallyDead` will delete, and MLIR's own canonicalizer is
entitled to do exactly that. `Pure` would be a lie here in a way it is not on the
tensor side: these operations write memory, and a pass that does not know it will
move them, delete them, or reorder them past a read, and the result is a silent
wrong answer rather than a crash.

The effects are attached per operand through the ODS `Arg` mechanism rather than
as a blanket `MemoryEffects<[MemRead, MemWrite]>` on the operation. A blanket
declaration says the operation touches memory somewhere; the overlap rule below
needs to know *which* memref is written, and an effect with no value attached is
an effect nothing can be proved disjoint from.

Both claims are asserted directly, in `unittests/Dialect/NPUISA/InterfaceTest.cpp`
rather than only through the behaviour they cause: `ins` and `outs` partition the
operands exactly once on every compute operation including the optional bias and
variadic forms, and no compute operation reports itself free of effects. The
second is checked both as an empty effect list and through `isMemoryEffectFree`,
because an operation that does not implement the interface at all is treated as
effect free by every pass in MLIR and would fail one check while passing the
other.

### The DMA boundary invariant is scoped, and unscoped it is false

Function arguments and `npuisa.const` results are DRAM memrefs. `npuisa.dma_load`
takes a DRAM memref and writes a scratchpad memref; `npuisa.dma_store` is the
mirror.

**Immediately after lowering and before allocation**, and only then, DMA appears
exactly at the boundaries between the two memories: one load per DRAM value
entering the scratchpad, one store per returned value. That is the invariant, and
the scope is not a hedge. Stated without it the invariant is simply false, because
later passes legitimately add DMA.

`test/Dialect/NPUISA/dma-boundaries.mlir` is the file that asserts it, at exactly
the point in the pipeline where it holds, which is the only point where asserting
it would be true. **That file belongs to P4**, with the lowering pass, because
there is no lowering before then for it to run after. A version of it written at
P2 could only hand write IR already in the right shape, which asserts that its
author can produce a correct example rather than that the pipeline does. The
invariant is written down here so that P4 implements against a stated rule
instead of inventing one. *P4 landed it, on `npu` dialect input throughout, so
what it checks is what the lowering produced.*

`test/Dialect/NPUISA/ops-memref.mlir` does carry the boundary *shape* today, as a
round trip case: function arguments in DRAM, one load in, computation on
scratchpad buffers, one store out. That pins what the IR looks like. It does not
and cannot pin that the lowering produces exactly that, which is the different
and stronger claim P4 owns.

There are **exactly three** permitted producers of DMA, named so that an
unexplained fourth is recognisable as a defect rather than debated as a style
question:

1. **The lowering itself**, per the sentence above.
2. **Spilling.** A `dma_store` after the definition and a `dma_load` before each
   later use of a spilled buffer.
3. **Tiling.** One load and one store per tile of an operation the pass split.

Any copy beyond those three is extra DRAM traffic. That matters more than it
sounds: a DRAM access costs orders of magnitude more energy than a scratchpad
access, so an unexplained copy moves published numbers, which makes it a bug. The
allocator and the tiling pass each count the DMA they insert into the result cell,
so the sum is checkable rather than asserted.

A fourth form, **relayout-and-move**, is a named future extension and is **not
implemented**. Nothing in this project may cite it as an available mechanism
today. If it ever lands it becomes the fourth permitted producer and this list is
amended in the same commit.

### Extension, P4: the function boundary, the broadcast view, and the layout map

*This subsection extends the memory model with three things the lowering had to
settle and this file did not previously say. It is marked as an extension
because everything above it is P2's and binding, and a later reader should be
able to tell which parts were designed before a lowering existed and which parts
the lowering decided. Nothing above is contradicted.*

**A lowered function takes its outputs as trailing arguments and returns
nothing.** Arguments become `#npu.dram` memrefs in place, results become
`#npu.dram` memrefs appended after them, the result list is empty and
`func.return` carries no operands. The alternative, returning a DRAM buffer, is
not available: the buffer would be one the function allocated, and **nothing
below this level allocates DRAM.** `memref.alloc` in the scratchpad is what the
allocator assigns offsets into and there is no second allocator for the other
space. The out parameters are **appended** rather than prepended, so argument N
of a lowered function is still argument N of the model. This is the shape
`test/Dialect/NPUISA/ops-memref.mlir` has called a lowered function since P2,
and it is where the encoder reads its input and output regions from.

**The rank 1 channel broadcast is a stride 0 view, not an instruction.**
`docs/adr/0005-channel-broadcast-on-add-and-mul.md` obliges the lowering to turn
an `npu.add` or `npu.mul` with a rank 1 right hand operand into a form the ISA
can execute, read with a channel stride of one and a spatial stride of zero. The
lowering emits a `memref.reinterpret_cast` of the rank 1 scratchpad buffer at
the destination's extents, with strides `[0, 1, 0, 0]`. The view has the
destination's shape, which is what `npuisa.add` and `npuisa.mul` already require
of their operands, so no dialect change was needed; and it adds no transfer, so
the carve out costs C floats of DRAM traffic rather than N by C by H by W of it,
which is the reason ADR 0005 gives for leaving the constant unexpanded at all.
**No opcode was added and none may be.** A stride 0 read is an addressing mode,
Section 9.1 already carries operand strides in the `Instruction` record, and the
instruction set of Section 5.4 is closed. The obligation this places on P7 is
exact: a compute kernel indexes its operands through their strides, and the
broadcast then needs no special case in the kernel at all.

**The layout encoding becomes the memref's strided layout map, over NCHW
extents.** Section 5.5 requires the tensor level layout to survive into the
format's stride fields. The buffer keeps NCHW extents, which is the order every
`npuisa` verifier already reads and the order the simulator's kernels index, and
the permutation moves into the strides:
`tensor<1x8x8x3xf32, #npu.layout<nhwc>>` becomes
`memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>`. Keeping the
extents in the order the layout wrote them was the alternative and it is wrong
for a stateable reason: the strides come out contiguous again, the layout leaves
no trace below the tensor level, and `-npu-assign-layout` becomes a pass whose
delta is structurally zero. An NCHW tensor gets no layout map at all, because
the identity written out at length is still the identity.

Two consequences fall on `-npu-assign-layout`, named here rather than discovered
there. A constant carrying a layout encoding, and a transpose that changes the
layout as well as the extents, are both **diagnosed** by the lowering: it does
not permute constant data, and it represents a permutation of extents only. That
pass materialises its own permuted constants and folds its own inverse
transposes, which Section 12 already says it does.

### Extension, P5: the DRAM spill slot

*This subsection extends the memory model with something the allocator had to
settle. It is marked as an extension for the same reason P4's is: a later reader
should be able to tell which parts were designed before an allocator existed and
which parts the allocator decided. **It amends a sentence in P4's extension
above**, which is called out rather than left for somebody to notice, and the
amendment is in the same commit as the code that needs it, per the rule this
file sets for its own list of DMA producers.*

**The allocator allocates DRAM, and it is the only thing that does.** P4's
extension says that nothing below the tensor level allocates DRAM, and it said
so to explain why a lowered function takes its outputs as trailing arguments.
That reason still holds and the out parameter convention is unchanged. What has
changed is that a spilled buffer needs somewhere to go: Section 13.1's spill
semantics are a `dma_store` after the definition and a `dma_load` before each
later use, and the store needs a destination that is not an input, not an output
and not a constant. So `-npu-allocate-scratchpad` emits `memref.alloc` in
`#npu.dram`, marked `npuisa.spill_slot`.

The alternatives were considered and each is worse in a stateable way. A
trailing function argument per slot would change the shape the encoder reads its
input and output regions out of, which P4 fixed as "the first N are inputs and
the last M are outputs"; adding a third category by position is how that
convention becomes ambiguous. An `npuisa.const` would be a constant with data
nobody wrote. Reusing the DRAM buffer a value was loaded from only works for
values that came from DRAM, which is a special case masquerading as a rule.

What this obliges P6 to do is exact, and it is one sentence: **the encoder gives
each `npuisa.spill_slot` allocation an address in the DRAM map, the way it
already does for constants and for the input and output regions.** The
allocation is marked in the IR rather than inferred from context, so finding
them is a predicate and not an analysis.

**A view's byte range is computed from its strides, not from its extents, and a
stride 0 view therefore spans the buffer it is over.** P4's handoff left this as
an open question with the right shape: `computeBufferRange` had never been shown
a `memref.reinterpret_cast`, the broadcast view of ADR 0005 is one, and somebody
had to decide whether a stride 0 view has a byte range at all. It does. The
range is `1 + sum((extent - 1) * stride)` elements from its first byte, which
for `sizes [1, 8, 4, 4]` over `strides [0, 1, 0, 0]` is 8 elements and not 128.

That is the true answer, and the reason to insist on it is not tidiness. Taking
the span from the extents instead would be *safe* in the overlap rule's
direction, since a range that is too large only ever reports more overlaps than
there are. It would be wrong in the usable direction: every per channel scale in
the pipeline would appear to collide with whatever the allocator packed within
480 bytes of it, and the double buffering pass of Section 5.1 would be refused
transfers it is entitled to, for races that do not exist. A conservative
analysis that refuses everything is not conservative, it is broken.

The rule is a closed hull rather than an exact set: a view whose stride exceeds
its inner extents addresses some bytes in its range and not others, and the
whole interval is reported. That is the only approximation in the analysis and
it is in the safe direction.

Fixing the size to come from the strides also fixed a latent unsoundness in the
`memref.subview` case, D-0018, which had been measuring a subview by the product
of its extents. That is the number of elements a subview holds and not the
number of bytes it reaches across, and two subviews of one buffer that genuinely
shared elements were reported `Disjoint`. Nothing emits a subview yet, so it had
never fired.

### Offsets are SSA operands, not attributes

`AllocateScratchpad` assigns each allocation a byte offset, and materialises it
as a `memref.view` over one flat `memref<Nxi8, #npu.scratchpad>` rather than as a
discardable attribute. The offset is therefore an SSA operand that every verifier
and every consumer can see, and a pass cannot silently drop it the way it can
drop a discardable attribute it does not recognise. Spill reloads become new
views, so aliasing is visible in the IR rather than conventional.

This choice is what forces the shape of the overlap rule below, and the two have
to be understood together.

### The token, and the overlap rule

Asynchronous DMA is `%token = npuisa.dma_load_async %source, %dest` followed
later by `npuisa.await %token`. The token is a scheduling handle, not data: it
carries no value and nothing computes with it. What it carries is an obligation,
namely that the transfer it names has been started and not yet waited on, and
that the destination buffer is being written by hardware for the whole of that
window.

It is a real type rather than a positional convention because a value with a use
is something a verifier can follow. Every rule below is a property of that use.
Matched by position instead, each rule would be a search over the block, and a
pass that reordered two transfers would silently repair the search while breaking
the program.

The four rules:

1. The token has exactly one use and that use is an `npuisa.await`.
2. The `await` is in the same block as its producer and comes after it.
3. Source and destination shapes and element types agree.
4. No operation between the async operation and its `await` may access memory
   overlapping the destination.

**Rule 4 is decided on `MemoryEffectOpInterface` effects plus an explicit overlap
test on view offsets and extents, and never by scanning for the same SSA value.**
This is the design point the whole memory model turns on. Once the allocator has
materialised offsets as views over one flat buffer, two different SSA values are
routinely two halves of the same memory. A partially overlapping view is a real
race that an identity check does not see, and it is not an exotic case: it is
what the allocator produces every time it packs two buffers adjacently and a pass
later views across the boundary.

So the decision procedure walks each value back to its root allocation,
accumulating byte offsets through `memref.view` and contiguous static
`memref.subview`, and compares half open byte ranges. Identity needs no special
case and does not get one: a range always overlaps itself, so two identical
values fall out as overlapping through the same arithmetic as everything else.
Distinct roots are distinct memories and are disjoint whatever their offsets.

The answer is three valued, not two, and the third value is the point.
`Unknown` is what an unanalysable offset produces, and it is treated as a
refusal: **if the offsets are not static, the async form is refused with a
diagnostic rather than assumed disjoint.** "I cannot prove these overlap" and
"these are disjoint" are different answers and only one of them is safe. The same
applies to an intervening operation that does not implement
`MemoryEffectOpInterface` at all, or that declares an effect on a value it does
not name: an operation that cannot say what it touches has not said it touches
nothing.

There is one deliberate exception to that conservatism and it is worth stating
because it looks like a hole. An intervening `npuisa.await` belonging to a
*different* transfer is skipped. The await declares no memory effect of its own,
by design: what it does is order an effect the asynchronous operation already
declared, and giving it a duplicate write effect on a buffer it does not name as
an operand would mean inventing the operand for the purpose. Skipping it is sound
because an await touches no memory, and the bytes it implies are already
accounted for by the intervening asynchronous operation, which the same scan
visits and checks. Without the skip, two transfers in flight at once were
unrepresentable, which is the one thing asynchronous DMA exists for: the double
buffering of Section 5.1 issues the next tile's load before awaiting the current
one, so both awaits necessarily sit inside the other transfer's window. That was
defect D-0009, found by a canonicalization test rather than by review.

An asynchronous operation whose `await` is the very next operation canonicalizes
back to the synchronous form, so a scheduling pass that looked for something to
overlap the transfer with and found nothing leaves no residue. The fold is
strictly about adjacency. An await two operations later with a harmless operation
in between is a transfer that really does overlap some work, and folding it would
undo a scheduling decision somebody made on purpose.

### Why `TilingInterface` lives on `npu` and not here

`TilingInterface` is implemented on the `npu` tensor operations at P1. It is
**not** on the `npuisa` memref operations and it never will be, and this is the
question most likely to be reopened by someone who notices that tiling is about
memory, so the reasoning is recorded rather than left to be reconstructed.

The tiling pass runs **before** lowering. The interface therefore has to exist
where the pass can see it, which is the tensor level. Beyond scheduling, the two
methods that make tile-and-fuse work are `getResultTilePosition` and
`generateResultTileValue`, and both are meaningful only on operations that have
results. The `npuisa` compute operations deliberately have none: a machine
instruction writes a buffer, it does not produce a value.

Putting `TilingInterface` on the memref side would buy iteration domain
introspection and nothing else, while leaving tile-and-fuse unreachable. That is
the whole trade, and it is not close.

The price is real and is paid rather than hidden: the `npu` compute operations
take a destination operand and implement `DestinationStyleOpInterface` too, at
the tensor level, where a reader coming from linalg might not expect it. That was
paid in P1 rather than retrofitted.

The interface is also **implemented at P1 and consumed at P13**, deliberately.
Implementing an interface and using it in the same session makes it impossible to
tell an interface bug from a policy bug. P2 implements
`DestinationStyleOpInterface` and `MemoryEffectOpInterface` on the `npuisa` side
and nothing tiling related, because there is nothing on the memref side for
`TilingInterface` to be useful on.

### Registration, and the shape of a forgotten one

Interfaces are registered as external models from a separate translation unit,
through a `registerNPUTilingInterfaceExternalModels` entry point, and declared
with `declarePromisedInterface`. A forgotten registration is then a named error
saying the interface was promised and never provided, rather than a mysterious
failure at the first query. It also keeps the dialect library free of a
dependency on the SCF and tensor tiling stack, which matters because that stack
is large and the dialect library is linked into everything.

### What this section binds

Later phases inherit these as decisions, not as suggestions:

- Two memory spaces on `memref`, reusing the `npu` attributes. No third space and
  no second spelling without amending this file in the same commit.
- Destination passing with exactly one trailing `outs` on every compute
  instruction, no results, and both interfaces implemented.
- Byte offsets as `memref.view` SSA operands over a flat buffer. Not attributes.
- Exactly three producers of DMA. A fourth is a defect until this list is amended.
- The overlap rule decided on effects plus byte ranges. A future pass that needs
  to ask whether two buffers alias calls `mlir::npuisa::overlaps` and honours all
  three of its answers, including `Unknown`. An identity comparison anywhere in
  that role is a defect.
- `TilingInterface` on `npu`, never on `npuisa`.

From the P4 extension above, on the same terms:

- A lowered function takes its outputs as trailing DRAM arguments and returns
  nothing. Nothing below the tensor level allocates DRAM **except the allocator's
  spill slots**, per the P5 extension, which is the one amendment this list has
  taken.
- The rank 1 channel broadcast is a stride 0 view over the loaded rank 1 buffer.
  No opcode for it, now or later, unless Section 5.4's instruction set is
  reopened and this file is amended in the same commit.
- A layout encoding becomes a strided layout map over NCHW extents. Buffers are
  NCHW below the tensor level, always, whatever the tensor said.

From the P5 extension, on the same terms:

- A spilled value lives in a `memref.alloc` in `#npu.dram` marked
  `npuisa.spill_slot`, and the encoder gives each of them a DRAM address. No
  other pass allocates DRAM without amending this list in the same commit.
- A byte range comes from a memref's strides. An analysis that measured a view
  by the product of its extents is a defect, not a conservative approximation.
- Every scratchpad offset is a multiple of 64 bytes by default, which is a row
  of the 16 by 16 array at `f32`. The alignment is an option; the arena's own
  allocation carries it too, because an aligned offset into an unaligned arena
  is not aligned.
