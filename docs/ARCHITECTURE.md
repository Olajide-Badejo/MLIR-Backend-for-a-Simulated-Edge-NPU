# Architecture

This is a from scratch MLIR compiler backend that takes a trained ONNX model to an
executable instruction stream for a simulated edge NPU, plus a simulator that runs
the stream and reports correctness and simulated performance.

## Pipeline

```
.onnx
  |  Python importer (npu_frontend, MLIR Python bindings)
  v
npu dialect IR   (tensor level, DRAM values)
  |  canonicalize, constant fold, fold batchnorm, fuse ops, DCE
  v
optimized npu IR
  |  dialect conversion (npu-lower-to-npuisa)
  v
npuisa dialect IR   (instruction level, scratchpad buffers, explicit DMA)
  |  scratchpad allocation (npu-allocate-scratchpad)
  v
allocated npuisa IR   (each buffer has a byte address)
  |  InstructionEncoder (npu-translate)
  v
.nbin binary   (disassemblable with npu-objdump)
  |  Simulator (npu-sim)
  v
outputs + JSON stats (simulated cycles, DRAM bytes)
```

## Two dialects

- **`npu`** is the tensor level. Values are builtin ranked fp32 tensors living
  conceptually in DRAM. Operators mirror the supported ONNX subset. Fused
  operators carry an optional bias operand and an activation attribute rather than
  a combinatorial set of named fused ops. Pure traits let the generic folding and
  dead code elimination apply.
- **`npuisa`** is the instruction level, after memory placement. A value of
  `!npuisa.buffer` type is scratchpad resident; DRAM tensors move to and from it
  through `dma_load` and `dma_store`. The instruction stream is straight line,
  since inference graphs are static DAGs.

The lowering between them is a representation changing conversion built on the
dialect conversion framework: a `TypeConverter` maps every tensor to a buffer, and
DMA is materialized only at the DRAM boundaries.

## Components

| Area | Location | What it does |
|---|---|---|
| Frontend | `python/npu_frontend/` | ONNX import, op mapping, seeded model generation, the npu-compile driver |
| npu dialect | `include,lib/Dialect/NPU/IR` | ops, verifiers, folders, canonicalizers |
| npu passes | `lib/Dialect/NPU/Transforms` | fold batchnorm, fuse ops |
| npuisa dialect | `include,lib/Dialect/NPUISA/IR` | instructions, the buffer type |
| npuisa passes | `lib/Dialect/NPUISA/Transforms` | lower to npuisa, allocate scratchpad |
| Encoding | `lib/Encoding` | the .nbin format, encoder, disassembler |
| Simulator | `lib/Simulator` | fp32 kernels, the analytical cost model |
| Tools | `tools/` | npu-opt, npu-translate, npu-objdump, npu-sim |

## Correctness and performance

Numerical correctness is validated end to end against `onnxruntime` within stated
tolerances on seeded models. Performance is reported by an analytical cost model
and is always labeled as a simulated estimate, never a measurement. See
[ISA_MANUAL.md](ISA_MANUAL.md) for the memory model and encoding, and
[PASSES.md](PASSES.md) for the optimization passes.
