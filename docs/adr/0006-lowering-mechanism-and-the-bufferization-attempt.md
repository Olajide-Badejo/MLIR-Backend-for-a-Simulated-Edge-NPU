<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 6. The lowering produces memrefs directly, and One-Shot Bufferize was measured before that was decided

- **Status:** Accepted
- **Date:** 2026-08-20
- **Diataxis type:** explanation

## Context

The P4 roadmap entry does not say which mechanism to use. It says something
stricter:

> Attempt `one-shot-bufferize` first and record the outcome either way; if two
> space DMA insertion turns out not to be expressible through a single default
> memory space bufferization, write the evidence into an architecture decision
> record, keep a converter that produces memrefs directly, and say plainly which
> parts of the infrastructure were adopted. What is not acceptable is quietly
> skipping the attempt.

So this record exists because the attempt was required, not because the outcome
was in doubt. The evidence below is what the attempt produced, on this machine,
on 2026-08-20, against the pinned `llvmorg-22.1.8` build.

The input for every run was one convolution followed by one relu, in the shape
the frontend emits, with a `tensor.empty` destination on each compute operation:

```mlir
func.func @conv_relu(%x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %d0 = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d0 : tensor<1x8x8x8xf32>) { ... }
       -> tensor<1x8x8x8xf32>
  %d1 = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.relu ins(%c : tensor<1x8x8x8xf32>) outs(%d1 : tensor<1x8x8x8xf32>)
       -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}
```

### What the attempt showed, run by run

**Run 1, the default.** `npu-opt in.mlir --one-shot-bufferize`

```
error: op was not bufferized
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
```

Exit 1. No `npu` operation implements `BufferizableOpInterface`, so the pass
refuses the module at the first one it meets. This is the expected starting
point rather than the result: it says the interface is not implemented, which is
a thing that could be fixed.

**Run 2, with function boundaries.**
`--one-shot-bufferize=bufferize-function-boundaries=1`

Identical failure at the same operation. The boundary option changes what
happens to the signature, not what happens to an unknown operation.

**Run 3, allowing unknown operations.**
`--one-shot-bufferize="bufferize-function-boundaries=1 allow-unknown-ops=1"`

Exit 0, and the output is the informative one:

```mlir
func.func @conv_relu(%arg0: memref<1x3x8x8xf32, strided<[?, ?, ?, ?], offset: ?>>)
    -> memref<1x8x8x8xf32, strided<[?, ?, ?, ?], offset: ?>> {
  %0 = bufferization.to_tensor %arg0 : memref<...> to tensor<1x3x8x8xf32>
  %cst = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x8x8xf32>
  %1 = bufferization.to_tensor %alloc : memref<...> to tensor<1x8x8x8xf32>
  %2 = npu.conv2d ins(%0, %cst : ...) outs(%1 : ...) { ... }
  ...
}
```

Three things are wrong with that output, and none of them is a configuration
mistake:

1. **Every buffer is in the default memory space.** There is no `#npu.dram` on
   the argument and no `#npu.scratchpad` on the temporary. The two spaces are
   the whole memory model of Section 8.
2. **There is no DMA anywhere**, and there is nowhere for one to have come from.
3. The `npu` operations are still on tensors, bridged by
   `bufferization.to_tensor`, so nothing was actually lowered.

**Run 4, demanding an inferred space.** Adding `must-infer-memory-space=1`:

```
error: could not infer memory space
  %d0 = tensor.empty() : tensor<1x8x8x8xf32>
```

**This is the crux, and it is worth being precise about why.** The hook the
project would attach its spaces through is

```cpp
using DefaultMemorySpaceFn = std::function<std::optional<Attribute>(TensorType)>;
```

It takes a tensor **type** and returns one memory space. Two tensors of
identical type in different roles, one a function argument that lives in DRAM
and one a scratchpad temporary, are the same argument to that function and must
get the same answer. The infrastructure's model is one space per value; this
machine's model is that a value entering the compute units exists in two spaces
with a transfer between them. Those are different questions, and no setting of
the first answers the second.

**Run 5, the encoding route.** `use-encoding-for-memory-space=1` together with
`must-infer-memory-space=1`:

```
error: only one of 'must-infer-memory-space' and 'use-encoding-for-memory-space'
are allowed in one-shot-bufferize
```

The encoding route is closed twice over: the two options are mutually exclusive,
and this dialect's tensor encoding slot is already spent on `#npu.layout`, which
Section 5.5 requires it for.

### What implementing the interface would have changed

Nothing that matters, and it is worth saying why rather than leaving it as an
untried option. `BufferizableOpInterface::getBufferType` returns **one** buffer
type per tensor value, and `getBuffer` returns **one** buffer. The lowering this
machine needs produces, for a single function argument, a DRAM buffer and a
scratchpad buffer and an `npuisa.dma_load` between them. There is no method on
the interface whose return type can carry that, so implementing it on all twelve
operations would have reproduced run 3 with the `npu` operations gone: correct
memrefs in one space, and the memory model still absent.

## Decision

**The lowering is a dialect conversion that produces memrefs directly**, in
`lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp`, built on `TypeConverter`,
`ConversionTarget` and `applyPartialConversion`, which is what the roadmap entry
names first.

**Which parts of the infrastructure were adopted, plainly.**

Adopted:

- the builtin `memref` type and its memory space attribute, which Section 5.2
  calls the single most consequential design decision in the document. That is
  the part One-Shot Bufferize exists to reach, and this project reaches it by a
  different road rather than not reaching it;
- `StridedLayoutAttr` as the home of the layout decision below tensor level,
  which is what makes the memref carry the layout rather than an attribute
  painted beside it;
- the dialect conversion framework in full: the type converter, the conversion
  target, the signature conversion, and the partial conversion driver;
- destination passing, which the frontend already produces as `tensor.empty`
  destinations, so the buffer a compute instruction writes is the buffer the
  tensor level already nominated. This is the part that would have been hardest
  without the infrastructure and it costs nothing here, because
  `DestinationStyleOpInterface` on both dialects means the operand partition is
  already stated.

Not adopted:

- One-Shot Bufferize's analysis, its `BufferizableOpInterface`, its aliasing and
  conflict machinery, and its allocation and copy insertion.

**No `npu` operation implements `BufferizableOpInterface`**, and none should.
An interface implemented but never used is an interface nobody notices has
rotted.

## Consequences

**The `-npu-lower-to-npuisa` pass owns buffer creation.** A `tensor.empty`
destination becomes a `memref.alloc` in the scratchpad; a function argument
becomes a DRAM memref plus one `npuisa.dma_load`; an `npu.constant` becomes an
`npuisa.const` in DRAM plus one load; a returned value becomes one
`npuisa.dma_store` into a trailing DRAM out parameter. That is Section 8's
boundary invariant, produced rather than asserted, and
`test/Dialect/NPUISA/dma-boundaries.mlir` asserts it immediately after this pass
and before allocation, which is the only point at which it is true.

**No deallocation is emitted.** `-npu-allocate-scratchpad` assigns byte offsets
into one flat buffer and derives live intervals from the instruction stream, so
a `memref.dealloc` here would be a weaker second statement of a lifetime that
pass computes exactly.

**The cost of not adopting the analysis is a capability this project does not
need.** One-Shot Bufferize's value is deciding when a tensor can be written in
place and when a copy is required, in the presence of aliasing it cannot see
through. Here every destination is explicit in the IR before the pass runs,
because the frontend is required to emit one `tensor.empty` per compute
operation and a pytest over all seven models asserts it. The analysis would have
been deciding a question the frontend has already answered.

**If a future phase needs the analysis, this record is what it argues against.**
The two conditions that would change the answer are named so a later reader can
check them: a `DefaultMemorySpaceFn` that takes the value rather than the type,
or a bufferization that can express one tensor becoming two buffers and a copy.
Neither exists in 22.1.8, and the 23.x note in Section 0.3 changes the signature
to `TensorLikeType` without changing which of the two it takes.
