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

A fixed header followed by tagged records, little endian. Simple to get right and
to disassemble rather than bit packed. Integers are `i64` unless noted; counts and
the version are `u32`; the magic is four bytes.

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
