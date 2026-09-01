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

**Current phase: P10, measurement.** The compiler runs end to end at three
optimization levels. An ONNX model at opset 23 goes in and a simulated answer
comes out, and it is checked against `onnxruntime` and against an independent
numpy reference interpreter, over seven models, three levels, two batch sizes and
five input classes. Every pass reports what it did to the program and what it
cost, from inside the pipeline that ran it.

What exists: the out of tree MLIR project and its `npu-opt` (P0), the `npu`
dialect on tensors (P1), the `npuisa` dialect and its two space memory model on
memrefs (P2), the ONNX frontend and the seeded model suite (P3), the lowering
between the two dialects (P4), the scratchpad allocator with DMA spills (P5),
the `.nbin` binary format generated from one ISA description (P6), the simulator
and its two port cost model (P7), the `npu-compile` driver with the end to end
matrix and the regression baseline (P8), the optimization passes and the `-O1`
and `-O2` levels (P9), and the result schema, the per pass instrumentation, the
benchmark suite and the prediction mechanism (P10).

What does not: tiling, layout assignment, double buffering and quantization, and
every external cross check. The roofline bound, SCALE-Sim and Accelergy are P11;
tiling and layout are P13; quantization is P14.

```
scripts/npu-compile model.onnx -O 2 --emit nbin -o model.nbin
build/bin/npu-sim model.nbin --input x.bin --output y.bin
```

`docs/PHASE_STATE.md` is the authoritative answer to where the work stands. This
section is a summary of it and the file is what gets updated every session.

## Results

Every number in this table is a field of a committed file under
`experiments/results/`, and a test parses the table and refuses a number that is
not. The specification forbids a hand typed number in this file; that is a rule
with a mechanism behind it rather than a promise.

| Model | Batch | `-O0` | `-O2` | Cycles at `-O2` | DRAM bytes at `-O2` | Distance to `onnxruntime` |
|---|---|---|---|---|---|---|
| `lenet` | 1 | 25 | 25 | 17766.25 | 250000 | 2.980e-08 |
| `depthwise_separable` | 1 | 12 | 12 | 1324 | 3008 | 1.490e-08 |
| `resnet_block` | 1 | 14 | 14 | 1626 | 8800 | 2.235e-07 |
| `inception_block` | 1 | 14 | 14 | 2398.5 | 8624 | 5.960e-07 |
| `conv_bn_relu_stack` | 1 | 23 | 15 | 1160.5 | 4144 | 7.451e-08 |
| `dilated_stack` | 1 | 13 | 12 | 1234.0625 | 5364 | 5.960e-07 |
| `lenet_batched` | 4 | 25 | 25 | 20000 | 259528 | 2.980e-08 |

The first two number columns are instruction counts, taken from the simulator's
own statistics and never counted from anything else. The last is the largest
absolute difference from `onnxruntime` on a seeded standard normal input, against
a band of 5e-5.

**`-O2` beats `-O0` on two of the seven models and ties on the other five, and
`-O1` is exactly `-O0` on all seven.** That is a measurement of this suite rather
than a claim about optimization levels, and it is stated here rather than left
for a reader to notice. `docs/NUMBERS.md` carries the full ablation table, one
row per optimization pass, with the mechanism behind every zero row: `-sccp` has
nothing to propagate across because every module holds one function, and
`-canonicalize` reads as zero because `-cse` reaches the same program without it,
which is a limit of leave one out ablation rather than a fact about
canonicalization.

**What the measurement cost.** 175 cells in 1.76 minutes, 0.60 seconds per cell,
run serially against a 90 minute budget the harness enforces by failing.

**What is not measured yet.** Energy, area, the roofline bound and the SCALE-Sim
cross check arrive at P11, tiling at P13, quantization at P14. Every result file
carries those fields as `null` with a reason naming the phase, rather than as a
plausible number or an absent key.

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
