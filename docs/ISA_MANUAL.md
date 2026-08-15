# NPU ISA manual

This documents the `npuisa` instruction set, its two level memory model, and the
`.nbin` binary encoding, the way a small processor's reference manual would.

## Memory model

Two address spaces:

- **DRAM**: holds the model inputs, the constant weights and biases, and the
  outputs. Addressed by byte offset from 0.
- **Scratchpad**: a small fixed size fast memory (default 1 MB). Compute
  instructions read and write it only. Addressed by byte offset from 0. The
  scratchpad allocator assigns each buffer an offset, recorded as the `address`
  field of the producing instruction.

All data is fp32 (4 bytes per element). The instruction stream is straight line:
inference graphs are static DAGs, so there are no branches.

## Opcodes

| Value | Mnemonic | Effect |
|---|---|---|
| 0 | NOP | Nothing. |
| 1 | HALT | End of stream. |
| 2 | DMA_LOAD | Copy a DRAM tensor into a scratchpad buffer. |
| 3 | DMA_STORE | Copy a scratchpad buffer into a DRAM tensor. |
| 4 | CONV2D | Convolution, optional bias operand, fused activation. |
| 5 | MATMUL | Matrix multiply, optional bias operand, fused activation. |
| 6 | RELU | Elementwise max(x, 0). |
| 7 | ADD | Elementwise add. |
| 8 | MUL | Elementwise multiply. |
| 9 | POOL_MAX | 2D max pooling. |
| 10 | POOL_AVG | 2D average pooling. |
| 11 | RESHAPE | Reinterpret a buffer's shape (no data movement). |

The numeric values are part of the binary format and are not renumbered without a
format version bump. The fused activation field is 0 for none and 1 for relu.

## Binary format (.nbin)

A fixed header followed by fixed order sections, each repeated section prefixed by
a `u32` count. There are no tags. Every field sits at a position determined by the
fields before it, so a reader cannot skip one it does not recognise and there is
no forward compatibility within a version; the `version` field carries
compatibility instead, and the policy below says how. The layout is byte oriented
rather than bit packed, to stay simple to get right and to disassemble. Integers
are `i64` unless noted; counts and the version are `u32`; the magic is four bytes.

```
magic            "NPUB"
version          u32
scratchpad_bytes i64
dram_bytes       i64
num_inputs       u32,   then that many MemRegion
num_outputs      u32,   then that many MemRegion
num_constants    u32,   then that many (MemRegion, u32 count, count * f32 data)
num_instructions u32,   then that many Instruction

MemRegion    = dram_offset i64, shape (u32 rank, rank * i64)
Instruction  = opcode u16, result_addr i64, result_shape (vec),
               operand_addrs (vec), dram_addr i64, activation i32, group i64,
               strides (vec), pads (vec), dilations (vec), kernel_shape (vec)
vec          = u32 count, count * i64
```

### Byte order

Host byte order, not a fixed endianness. The writer copies the object
representation of each value straight into the stream and the reader copies it
straight back out, so the bytes on disk are whatever the machine's layout is. A
`.nbin` is therefore **not portable across byte orders**: a file written on a big
endian machine will not read correctly on a little endian one, and the magic will
not catch it because the magic is four separate bytes.

Every machine this project targets is little endian, so in practice files are
little endian, but that is a property of the machine and not a guarantee of the
format. Nothing in the encoder or decoder swaps bytes. Making the format portable
means fixing an endianness and swapping on both sides, which is a format change
and would bump the version.

### Reading a file: decode versus decodeUnvalidated

Two entry points, and the difference matters.

- `Program::decode()` means decode **and** validate. It returns a program only if
  every rule in the next section passes, and otherwise reports which one failed.
  This is what `npu-sim` and the library entry points use, and it is the only way
  to obtain a program the simulator is willing to trust.
- `Program::decodeUnvalidated()` parses the byte stream and stops. It exists for
  `npu-objdump`, because dumping a file you already suspect is broken is what a
  disassembler is for. Its output is prefixed with a warning. It still refuses a
  stream that is structurally unreadable, such as a bad magic or a truncation, but
  it applies none of the semantic rules, so what it returns may be nonsense.

## Validation

`Program::validate()` enforces the rules below, and reports the first failure as a
`ValidationError` carrying a check name, the instruction index (or a program level
marker), and a detail string naming the offending construct. The check name is
stable and is what a test asserts against; the detail string is for a human. This
list is generated from the `fail(...)` call sites in `lib/Encoding/Program.cpp` and
every name here has at least one test in
`unittests/Encoding/ValidationTest.cpp`.

Program level:

| Check | Rejects |
|---|---|
| `structure` | the byte stream is truncated, or does not begin with the `NPUB` magic. Reported by `decode`, not by `validate`. |
| `version` | the file declares a version this build does not understand. |
| `scratchpad-size` | a negative scratchpad, or one above the 64 MiB format limit. |
| `dram-size` | a negative DRAM size, or one above the 64 MiB format limit. |
| `region-shape` | an input, output, or constant region with an empty shape or a non positive extent. |
| `region-offset` | a region at a negative DRAM offset, or one not 4 byte aligned. |
| `region-in-range` | a region whose extent runs past the declared DRAM size. |
| `constant-data` | a mismatch between the number of constant regions and data blocks, or a constant whose data length does not match its shape. |

Per instruction:

| Check | Rejects |
|---|---|
| `opcode` | an opcode outside the defined range. |
| `arity` | an operand count the opcode does not accept. |
| `result-shape` | an empty result shape, a non positive extent, or a product above the 2^40 element cap. |
| `result-address` | a negative result address, or one not 4 byte aligned. |
| `result-in-range` | a result whose extent runs past the declared scratchpad. |
| `operand-in-range` | an operand address outside `[0, scratchpadBytes)`. |
| `operand-defined` | an operand address no earlier instruction wrote, so its shape is unknown. |
| `operand-extent` | an operand read larger than what was written at that address. |
| `dram-address` | a negative DRAM address on an instruction that touches DRAM. |
| `dram-in-range` | a DRAM access running past the declared DRAM size. |
| `attribute-size` | a strides, pads, dilations, or kernel_shape vector of the wrong length. |
| `attribute-value` | a negative entry in one of those vectors, or a zero stride, dilation, or pool kernel. |
| `group` | a convolution group below 1. |
| `activation` | a fused activation other than 0 (none) or 1 (relu). |

`operand-extent` is exact for `DMA_STORE`, `RELU`, `ADD`, `MUL`, and `RESHAPE`,
and a lower bound for the two pools. For `CONV2D` and `MATMUL` it requires only a
non zero recorded count, because their operand extents follow from tensor shapes
and the walk tracks element counts; a full check there needs shape tracking and is
not implemented.

## Version policy

The current version is **1**, as `Program::kVersion` in
`include/NPU/Encoding/Program.h`.

- The `version` field is read and checked on every decode. A file declaring any
  other version is rejected with check `version`, rather than being parsed
  hopefully. There is no forward compatibility inside a version, because the
  format has no tags to skip.
- Opcode numeric values are part of the format and are **never renumbered**. An
  opcode that is retired keeps its number reserved.
- New opcodes are **appended** with the next free number. Appending an opcode does
  not bump the version, because an old reader rejects the unknown opcode with
  check `opcode` rather than misreading it.
- Any change to the **layout** bumps the version: adding, removing, reordering, or
  resizing a field, and changing the byte order. There is no way for a reader to
  detect such a change from the bytes, so the version is the only signal.

## Tools

- `npu-translate model.isa.mlir -o model.nbin` encodes an allocated npuisa
  function.
- `npu-objdump model.nbin` prints the DRAM layout and the disassembled
  instruction stream. It decodes without validating, so it can dump a file that
  `npu-sim` would refuse; such a dump is prefixed with a warning.
- `npu-sim model.nbin [--input in.bin]... [--output out.bin] [--stats s.json]`
  runs a program and prints its simulated statistics as JSON.

  `--input` is given once per declared input region, in declaration order, and
  each file is a flat row major fp32 buffer. A count that does not match the
  program is refused, naming both numbers: passing one `--input` to a two input
  model used to run with the second input left as zeros and say nothing.
  Each file's float count is also checked against its region's shape.

  `--output` names one file for a single output model. A model with several
  outputs gets one file per output, numbered beside the given path, so
  `--output out.bin` writes `out.0.bin` and `out.1.bin`.
