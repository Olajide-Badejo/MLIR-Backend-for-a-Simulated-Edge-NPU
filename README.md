# npu-mlir

From scratch MLIR compiler backend for a simulated edge NPU: custom `npu` and `npuisa`
dialects, BatchNorm folding and operator fusion, an ONNX frontend, a binary ISA with a
disassembler, and a cost modeled C++ simulator, validated against onnxruntime.

> Status: under construction. This repository is being built phase by phase per the
> project specification. See [docs/ENGINEERING_LOG.md](docs/ENGINEERING_LOG.md) for the
> running record and the roadmap below for what is done.

## What this is

The pipeline takes a trained ONNX model and lowers it to an executable instruction stream
for a simulated, simplified edge NPU, then runs that stream in a C++ simulator that reports
numerical correctness against onnxruntime and simulated performance from an analytical cost
model.

```
.onnx -> ONNX importer -> npu dialect IR -> canonicalize / fold / fuse / DCE
      -> lower to npuisa (explicit scratchpad addresses) -> scratchpad allocation with DMA
      -> binary .nbin encoding -> C++ simulator -> outputs + JSON stats
```

All performance numbers the project reports are **simulated estimates** from an analytical
cost model, not cycle accurate measurements, and are labeled as such everywhere.

## Quick start

Build the pinned LLVM/MLIR once, then build this project against it. Full instructions,
including the memory budget this machine builds under, are in [docs/BUILD.md](docs/BUILD.md).

## Roadmap

- [x] P0 toolchain, one time LLVM build, scaffold, dash lint
- [x] P1 npu dialect (ODS, verifiers, round trip lit)
- [x] P2 canonicalize, constant fold, batchnorm fold
- [x] P3 operator fusion
- [x] P4 DCE via traits and SymbolDCE
- [x] P5 ONNX frontend and model generator
- [x] P6 lowering to npuisa and scratchpad allocator
- [x] P7 encoder, objdump, simulator, end to end vs onnxruntime
- [x] P8 npu-compile driver and opt levels
- [x] P9 benchmark harness and ablations
- [x] P10 docs set and auto generated dialect reference
- [x] P11 main report and debug report
- [ ] P12 final QA and v1.0.0

## Documentation

- [ARCHITECTURE](docs/ARCHITECTURE.md) the pipeline, the two dialects, the components
- [BUILD](docs/BUILD.md) exact build steps and the memory budget this machine uses
- [PASSES](docs/PASSES.md) each optimization pass with before and after IR
- [ONNX_FRONTEND](docs/ONNX_FRONTEND.md) the importer and the supported op subset
- [ISA_MANUAL](docs/ISA_MANUAL.md) the instruction set, memory model, and .nbin format
- [DIALECT_REFERENCE](docs/DIALECT_REFERENCE.md) generated from ODS
- [DESIGN_DECISIONS](docs/DESIGN_DECISIONS.md) the choices and why
- [ENGINEERING_LOG](docs/ENGINEERING_LOG.md) the running record of problems and fixes
- [CONTRIBUTING](docs/CONTRIBUTING.md) building, testing, and style
- [CHANGELOG](CHANGELOG.md)

## License

Apache License v2.0 with LLVM Exceptions. See [LICENSE](LICENSE).
