<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# MLIR Backend for a Simulated Edge NPU

This repository is mid rebuild. I am rewriting the compiler from scratch
against the v3 build specification, and this README is a stub that says where
the work stands rather than a description of something finished. The v1 release
is tagged `v1.0.0` and stays reachable in git history; the rebuild releases as
`v2.0.0`.

**Current phase: P8, the walking skeleton and the safety net.** The compiler
runs end to end. An ONNX model at opset 23 goes in and a simulated answer comes
out, and it is checked against `onnxruntime` and against an independent numpy
reference interpreter, over seven models, two batch sizes and five input
classes.

What exists: the out of tree MLIR project and its `npu-opt` (P0), the `npu`
dialect on tensors (P1), the `npuisa` dialect and its two space memory model on
memrefs (P2), the ONNX frontend and the seeded model suite (P3), the lowering
between the two dialects (P4), the scratchpad allocator with DMA spills (P5),
the `.nbin` binary format generated from one ISA description (P6), the simulator
and its two port cost model (P7), and the `npu-compile` driver with its `-O0`
pipeline, the end to end matrix, and the regression baseline (P8).

What does not: `-O1` and `-O2`, which the driver names and refuses. Every
optimization pass, tiling, layout assignment, double buffering, quantization,
and every external cross check. Those are P9 and later.

```
scripts/npu-compile model.onnx -O 0 --emit nbin -o model.nbin
build/bin/npu-sim model.nbin --input x.bin --output y.bin
```

`docs/PHASE_STATE.md` is the authoritative answer to where the work stands. This
section is a summary of it and the file is what gets updated every session.

**There are no performance numbers here, and there will not be until there are
numbers from a recorded experiment.** The v3 specification forbids a hand typed
number in this file. The machinery that makes a number quotable landed at P8, in
`scripts/regression-baseline.sh`, and the benchmark harness that turns it into a
table is P10's.

**What CI enforces, and what those numbers are worth.** 85 percent line coverage
on `lib/Dialect`, `lib/Encoding` and `lib/Simulator`, and 90 percent on
`python/npu_frontend`. Both are **floors**, set at P8 from a measurement, and
both are project management artifacts rather than correctness arguments: line
coverage measures execution and not assertion, and a suite that runs every line
of the decoder and asserts nothing would score 100 percent. The correctness
argument is the end to end matrix, the reference interpreter, the metamorphic
relations and the regression baseline, none of which is a percentage.

## Layout

The tree follows Section 6 of the build specification. `include/NPU` and `lib`
hold the dialects and passes, `tools` holds the drivers, `test` holds the lit
suite and the Python tests, `python/npu_frontend` holds the ONNX importer and
the model generator, `experiments` holds the benchmark harness and its recorded
results, and `report` holds the LaTeX that reads those results. Directories
arrive at the phase that fills them, so some of that list is not present yet.

## Documentation

`docs/` carries the set Section 20 specifies, each labelled with its Diataxis
type. The ones worth knowing about today: `ARCHITECTURE.md` for the memory
model, `DIALECT_REFERENCE.md` generated from the `npu` dialect's ODS source,
`ONNX_FRONTEND.md` for the frontend's contract, `DESIGN_DECISIONS.md` as a
generated index over the numbered records in `docs/adr/`, and
`ENGINEERING_LOG.md` and `DEFECT_LOG.md`, which are first class deliverables
rather than chores.

## Building

```
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld
ninja -C build -j6
ninja -C build check-npu
```

The build links against an existing LLVM at `llvmorg-22.1.8` and never builds
LLVM itself.

The Python tests need the MLIR Python bindings, which
`test/Python/conftest.py` puts on the path itself, and a built `npu-opt`, which
the frontend uses to verify every module it emits:

```
python -m pytest test/Python -q
```

## Licence

MIT, in `LICENSE`. Files derived from the LLVM standalone example keep their
own Apache-2.0 with LLVM-exception licence and say so in their headers. Every
file carries SPDX tags and `reuse lint` runs in CI, so the provenance is
checkable rather than asserted.
