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
it would be true. **That file does not exist yet and belongs to P4**, with the
lowering pass, because there is no lowering before then for it to run after. A
version of it written at P2 could only hand write IR already in the right shape,
which asserts that its author can produce a correct example rather than that the
pipeline does. The invariant is written down here so that P4 implements against a
stated rule instead of inventing one.

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
