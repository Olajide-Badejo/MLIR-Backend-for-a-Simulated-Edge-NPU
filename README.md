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

**Current phase: P3, the ONNX frontend and the model suite.** What exists now:
the out of tree MLIR project and its `npu-opt`, the `npu` dialect on tensors
(P1), the `npuisa` dialect and its two space memory model on memrefs (P2), and
a Python frontend that imports an ONNX model at opset 23 to `npu` dialect IR
together with a suite of seven seeded models to feed it (P3). There is no
lowering between the two dialects yet, no simulator, and no end to end pipeline.
Those are P4, P7 and P8.

`docs/PHASE_STATE.md` is the authoritative answer to where the work stands. This
paragraph is a summary of it and the file is what gets updated every session.

I am not putting any numbers here until there are numbers that come from a
recorded experiment. The v3 specification forbids a hand typed number in this
file, and the machinery that makes a number quotable does not land until P8.

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
