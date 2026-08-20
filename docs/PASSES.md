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
**measured** delta rather than a qualitative claim. The ablation harness is
built at P10 and there is no simulator to measure against until P7, so no entry
below quotes one yet. Each says so in its own row rather than leaving the field
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
| `-npu-allocate-scratchpad` | all | no | P5 | not yet written |

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
func.func @conv_relu(%arg0: memref<1x3x8x8xf32, #npu.dram>,
                     %arg1: memref<1x8x8x8xf32, #npu.dram>) {
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

Not yet written. It lands at P5 with Section 13.1 in full, and this file gains
its entry in the same commit.
