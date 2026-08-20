<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# The npu instruction set

*Diataxis type: reference.*

This is the manual for the machine the compiler targets: its instruction set,
its memory model, its binary format, and the rules a `.nbin` file has to satisfy
before anything will run it. It is written the way a small processor manual is
written, because that is what it is.

**Parts of this file are generated and are marked as such.** The opcode table
and the validation check table come out of
`include/NPU/Encoding/NPUISADescription.td` by way of `ninja -C build
npu-isa-doc`, and `scripts/check-isa-staleness.sh` regenerates them in CI and
diffs, so an edit inside a marked section is reverted by the next build and
reported as staleness by the next run. Everything outside the markers is hand
written and is the place to explain anything the tables cannot.

## The machine

A simulated edge NPU with two memories and no control flow.

**Two memories.** A scratchpad, small and fast, and DRAM, large and slow. The
compute units address the scratchpad and nothing else. DRAM is reachable only
through the two DMA instructions, which move bytes between the two memories and
do not change a layout on the way.

**No branches.** A program is one straight line of instructions ending in
`HALT`. There is no jump, no call and no predicate, because the workload is a
feedforward tensor graph and nothing in it needs one. That is why the encoder
refuses a function with more than one block: the instruction set has nowhere to
put the second one.

**Destination passing.** Every instruction that computes writes a buffer whose
address the instruction carries. Nothing produces a value; there are no
registers and there is no stack.

## The instruction set

<!-- BEGIN GENERATED: opcode table -->

<!--
THIS SECTION IS GENERATED from include/NPU/Encoding/NPUISADescription.td by

    ninja -C build npu-isa-doc

and scripts/check-isa-staleness.sh regenerates it and diffs, so an edit here
is reverted by the next build and reported as staleness by the next run.
-->

| Opcode | Value | Operands | Result | Element types | Fields | Semantics |
|---|---|---|---|---|---|---|
| `NOP` | 0 | none | none | n/a | none | Does nothing and advances the program counter. |
| `HALT` | 1 | none | none | n/a | none | Stops the machine. The encoder emits one as the last instruction of every program. |
| `DMA_LOAD` | 2 | 1 in dram | scratchpad | f32, i8, i32 | none | Copies a buffer from DRAM into the scratchpad. |
| `DMA_STORE` | 3 | 1 in scratchpad | dram | f32, i8, i32 | none | Copies a buffer from the scratchpad back to DRAM. |
| `MATMUL` | 4 | 2 or 3 in scratchpad | scratchpad | f32 | `requantize`, `activation` | A rank 2 by rank 2 matrix multiplication with an optional bias. |
| `CONV2D` | 5 | 2 or 3 in scratchpad | scratchpad | f32 | `strides`, `pads`, `dilations`, `group`, `requantize`, `activation` | A two dimensional grouped convolution with an optional bias. |
| `ADD` | 6 | 2 in scratchpad | scratchpad | f32 | `requantize`, `activation` | Elementwise addition. |
| `MUL` | 7 | 2 in scratchpad | scratchpad | f32 | `requantize`, `activation` | Elementwise multiplication. |
| `RELU` | 8 | 1 in scratchpad | scratchpad | f32 | none | The rectified linear unit, elementwise. |
| `POOL_MAX` | 9 | 1 in scratchpad | scratchpad | f32 | `kernel`, `strides`, `pads`, `dilations` | Two dimensional max pooling. |
| `POOL_AVG` | 10 | 1 in scratchpad | scratchpad | f32 | `kernel`, `strides`, `pads`, `dilations` | Two dimensional average pooling. |
| `RESHAPE` | 11 | 1 in scratchpad | scratchpad | f32, i8, i32 | none | Copies a buffer under a different set of extents. |
| `TRANSPOSE` | 12 | 1 in scratchpad | scratchpad | f32, i8, i32 | `axes` | Permutes a buffer's dimensions into the result. |
| `CONCAT` | 13 | 1 or more in scratchpad | scratchpad | f32, i8, i32 | `axes` | Concatenates buffers along one axis into the result. |
| `QUANT` | 14 | 1 in scratchpad | scratchpad | i8 (operands f32) | `scale`, `zeroPoint` | Quantizes f32 to i8 with a scale and a zero point. |
| `DEQUANT` | 15 | 1 in scratchpad | scratchpad | f32 (operands i8) | `scale`, `zeroPoint` | Dequantizes i8 to f32 with a scale and a zero point. |

### Shape relations and disassembly

| Opcode | Shape relation | Disassembly | Semantics land at |
|---|---|---|---|
| `NOP` | No operands and no result. | `NOP` | P6 |
| `HALT` | No operands and no result. | `HALT` | P6 |
| `DMA_LOAD` | The operand and the result have the same shape and the same element type. The DMA moves bytes and does not change a layout on the way. | `DMA_LOAD %r <- %0` | P7 |
| `DMA_STORE` | The mirror of DMA_LOAD: same shape, same element type. | `DMA_STORE %r <- %0` | P7 |
| `MATMUL` | Operand 0 is (M, K), operand 1 is (K, N), the result is (M, N). The optional operand 2 is the bias and has length N. | `MATMUL %r <- %0, %1 {bias %2} {activation}` | P7 |
| `CONV2D` | Operand 0 is (N, C, H, W), operand 1 is (F, C/group, KH, KW), the result is (N, F, OH, OW). The optional operand 2 is the bias and has length F. | `CONV2D %r <- %0, %1 {bias %2} {strides} {pads} {dilations} {group} {activation}` | P7 |
| `ADD` | Both operands have the result's shape. No broadcasting: the frontend materialises every broadcast it accepts, and a rank 1 channel operand arrives as a stride 0 operand rather than as a shape. | `ADD %r <- %0, %1 {activation}` | P7 |
| `MUL` | The same rule as ADD. | `MUL %r <- %0, %1 {activation}` | P7 |
| `RELU` | The operand has the result's shape. The operand and the result may be the same buffer: an in place relu is what the allocator produces when it reuses a dead interval. | `RELU %r <- %0` | P7 |
| `POOL_MAX` | Operand and result are both (N, C, H, W) with the same N and C. The output extents are the result's, taken as declared rather than recomputed, because the encoder has already resolved ceil_mode. | `POOL_MAX %r <- %0 {kernel} {strides} {pads} {dilations}` | P7 |
| `POOL_AVG` | The same rule as POOL_MAX. The mean divides by the number of elements that actually contributed, which is ONNX's count_include_pad = 0 and the only behaviour implemented. | `POOL_AVG %r <- %0 {kernel} {strides} {pads} {dilations}` | P7 |
| `RESHAPE` | Element counts and element types agree; the extents need not. | `RESHAPE %r <- %0` | P7 |
| `TRANSPOSE` | `axes` is a permutation of exactly the result rank, and result extent i equals operand extent axes[i]. | `TRANSPOSE %r <- %0 {axes}` | P7 |
| `CONCAT` | `axes` holds exactly one entry, the axis, non negative and less than the rank. Every operand has the result's rank, the extents agree on every other axis, and along the axis they sum to the result extent. | `CONCAT %r <- %n {axes}` | P7 |
| `QUANT` | The operand and the result have the same shape. The operand is f32 and the result is i8. | `QUANT %r <- %0 {scale} {zeroPoint}` | P14 |
| `DEQUANT` | The mirror of QUANT: same shape, i8 operand, f32 result. | `DEQUANT %r <- %0 {scale} {zeroPoint}` | P14 |

### Which `npu` operation reaches which opcode

| `npu` operation | Reaches |
|---|---|
| `npu.add` | `ADD` |
| `npu.avg_pool2d` | `POOL_AVG` |
| `npu.batch_norm` | `ADD`, `MUL` |
| `npu.concat` | `CONCAT` |
| `npu.conv2d` | `CONV2D` |
| `npu.matmul` | `MATMUL` |
| `npu.max_pool2d` | `POOL_MAX` |
| `npu.mul` | `MUL` |
| `npu.relu` | `RELU` |
| `npu.reshape` | `RESHAPE` |
| `npu.transpose` | `TRANSPOSE` |
| `npu.constant` | no instruction: encoded as a constant region in the DRAM map with its data, not as an instruction; the load that brings it on chip is a DMA_LOAD |
| `npu.fused_op` | no instruction: flattened into its parent block by -npu-lower-to-npuisa before any instruction is emitted, so the operations the region held are encoded and the region is not |
| `npu.yield` | no instruction: the terminator of an npu.fused_op region, which disappears with the region when the fused operation is flattened |

### Element types

| Name | Value | Bytes per element |
|---|---|---|
| `f32` | 0 | 4 |
| `i8` | 1 | 1 |
| `i32` | 2 | 4 |

### Memory spaces

| Name | Value |
|---|---|
| `scratchpad` | 0 |
| `dram` | 1 |

### Activations

| Name | Value |
|---|---|
| `none` | 0 |
| `relu` | 1 |

<!-- END GENERATED: opcode table -->

## Opcode numbering, and how to add one

**Opcode numeric values are assigned once and are never renumbered.** New
opcodes are appended with the next free value. A `.nbin` written by an older
build has to keep meaning what it meant, and a renumbering would silently
reinterpret every file in the fuzz corpus rather than rejecting them.

Adding an opcode is one edit to
`include/NPU/Encoding/NPUISADescription.td` followed by a build that fails until
every layer has a case for it. That claim is the measurable one from Section 9.4
of the build specification and it is worth being precise about what enforces it:

- The `Opcode` enum, `kMaxOpcode`, the arity and field presence rules the
  validator reads, the disassembler's format strings, the simulator's dispatch
  skeleton, this manual's tables, and the opcode list the reachability checker
  reads are all **generated**. None of them can be forgotten because none of
  them is written by hand.
- The generator **refuses** a description whose opcode has no format string, no
  semantics line, no shape rule, no accepted element type, or fewer declared
  operand memory spaces than it takes operands. A half filled record is a
  generation failure, not a silently empty table row.
- The layers that need a human decision are hand written switches over the
  generated enum **with no `default` label**, compiled with `-Werror=switch`.
  Adding an opcode stops them compiling until somebody has said what the answer
  is, even when the answer is that there is nothing to do.
- `scripts/check-isa-staleness.sh` fails when the committed generated artifacts
  do not match the description.

**Adding an opcode does not bump `Program::kVersion`.** An opcode value is data
inside a layout that did not move. What bumps the version is a change to the
layout itself.

## Version policy

`Program::kVersion` starts at **1**.

Any change to the layout of the binary bumps it, and a file at a version this
build does not know is rejected with the `version` check rather than
reinterpreted. There is no tag mechanism in this format and therefore no way to
skip a field a reader does not recognise, so reinterpreting is not merely
unwise, it is not implementable.

**One future change is already accounted for and will not bump the version.**
The quantization work of Section 14 needs `ElemType::I8`, `ElemType::I32`, a
`scale`, a `zeroPoint`, a `requantMultiplier` and a `requantShift`, and every
one of those is in the format from version 1. The claim is narrow on purpose:
any *other* layout change still bumps the version. A version bump at that point
would invalidate the binary stability test and every seed in the fuzz corpus in
the same commit that introduced quantization, which is a migration worth
spending six unused fields to avoid.

## The validation checks

Every failure to validate a `.nbin` returns a structured error naming a stable
check name, and where the failure concerns one instruction, its index.

**The names below are taken from the source rather than copied into it.** They
are the `ISACheck` records of `NPUISADescription.td`, the same records the C++
`Check` enum is generated from, so this table cannot describe a check that does
not exist and cannot omit one that does.

<!-- BEGIN GENERATED: validation checks -->

<!--
THIS SECTION IS GENERATED from the ISACheck records in
include/NPU/Encoding/NPUISADescription.td. Section 9.2 asks for the manual's
check names to be taken from the source rather than copied, and this is
that: the C++ enum and this table come out of the same records.
-->

| Check | What it asserts |
|---|---|
| `version` | The version word is one this decoder accepts. An unknown version is rejected by name rather than reinterpreted. |
| `structure` | The file is framed correctly: the magic matches, every section is fully present, and nothing follows the last one. |
| `arity` | The operand count is within the range the opcode declares. |
| `count-cap` | Every u32 count field is at most kMaxCount. The cap is a number and it is checked before anything is allocated. |
| `result-shape` | Every result extent is positive and the product stays within the shape limit, tested without ever performing the multiplication that would overflow. |
| `result-address` | The result address is non negative, and zero when the opcode writes no result. |
| `result-in-range` | The result's bytes lie inside the memory it names. |
| `operand-in-range` | Each operand's byte span lies inside the memory it names. |
| `operand-defined` | Each operand's bytes were written by an earlier instruction, or belong to a declared input, constant or spill region. |
| `operand-extent` | The consumer's element need fits the element count actually written to the buffer it reads. Membership alone is not enough: a DMA_STORE reading 100 elements from a 4 element buffer would pass a membership test and then trap. |
| `dram-address` | A DRAM address is non negative. |
| `dram-in-range` | A DRAM access lies inside the declared DRAM size. |
| `region-offset` | A memory region's DRAM offset is non negative and does not overflow when its size is added. |
| `region-shape` | A memory region's extents are positive and its element count stays within the shape limit. |
| `region-in-range` | A memory region lies inside the declared DRAM size. |
| `constant-data` | A constant's data length equals its region's byte size, computed from the element type rather than assumed to be four bytes per element. |
| `attribute-size` | Each attribute vector has the length the opcode declares, and an attribute the opcode gives no meaning to is empty. |
| `attribute-value` | Strides and dilations are positive, pads are non negative, and the group count is positive. |
| `activation` | The activation field holds a defined value, and holds `none` on an opcode that fuses no activation. |
| `element-type` | Every element type byte is one of the defined values. |
| `element-type-supported` | The opcode accepts the element type it was given. |
| `quant-scale` | The quantization scale is finite and strictly positive, and is zero on an opcode that does not quantize. |
| `quant-zero-point` | The zero point lies inside the range of the integer type it belongs to. |
| `quant-types` | QUANT reads f32 and writes i8; DEQUANT reads i8 and writes f32. |
| `quant-shape` | A quantization instruction's operand and result have the same shape. |
| `quant-requantize` | The requantization shift is within [0, 31] and the multiplier is a positive int32, which is the range the fixed point decomposition produces. A value outside it is a corrupt file, not an exotic configuration. |
| `axes-permutation` | TRANSPOSE's axes vector is a permutation of exactly the result rank. |
| `concat-axis` | CONCAT's axes vector holds one entry, non negative and less than the rank. |
| `concat-extents` | CONCAT's operands agree on every axis but the concatenation axis and sum to the result extent along it. |
| `debug-pc` | A debug entry's program counter names an instruction that exists. |
| `debug-order` | Debug entries are strictly increasing in program counter, so a lookup is a binary search and a duplicate is a corrupt file. |
| `debug-name` | A debug name holds no NUL and no byte above 0x7f. ONNX node names are ASCII, and a name that is not is a file somebody built by hand. |
| `debug-size` | A debug name's length is within kMaxCount and within the bytes the file actually has. |

<!-- END GENERATED: validation checks -->

## Regenerating this file

```
ninja -C build npu-isa-doc
bash scripts/check-isa-staleness.sh
```

The first regenerates the marked sections of this file and
`docs/ISA_OPCODES.json`. The second is the CI gate: it regenerates and diffs,
and fails if the committed files moved.
