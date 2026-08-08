# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning once a release is tagged.

## [Unreleased]

### Added

- Phase U0: `scripts/regression-baseline.sh`, the safety net that gates every
  later upgrade phase. It builds the project, runs lit, both GoogleTest binaries,
  pytest, and dash-lint, then compiles and simulates LeNet at every combination
  of optimization level and scratchpad budget. Everything it measures is written
  to `test/baseline/baseline.json`, and the simulated output tensors are frozen
  as `.npy` golden files in `test/baseline/golden/`. The `--check` form
  re-measures and exits nonzero on drift, naming what moved and by how much.
  Also available as `ninja baseline-check`.
- `docs/BREAKING_CHANGES.md`, where any deliberate regression against that
  baseline has to be written down before the commit that causes it.

- Phase U1: `experiments/results/*.json` are tracked in git. They were excluded
  by a `.gitignore` line, so a fresh clone had no results and `results_to_tex.py`
  had nothing to read, even though every number in the README and both reports
  comes from them.
- Phase U1: `run_benchmarks.py --allow-stale`, for deliberately reusing results
  whose manifest no longer matches the tree.
- Phase U1: benchmark manifests now record the model generator seed and a
  `GENERATOR_VERSION`. The exported `.onnx` files stay untracked because they are
  regenerated from the seed, which is only reproducible if the seed travels with
  the result.

### Changed

- Phase U1 (**behaviour change, deliberate**): a batch size other than 1 is now
  rejected instead of silently computing wrong numbers. The simulator's
  convolution kernel hardcodes batch index 0 and its pooling kernel never sees
  the batch dimension, so N greater than 1 returned correct data for the first
  image and scratchpad residue for the rest, with no diagnostic anywhere in the
  pipeline. The refusal is enforced in the ONNX importer and again in the
  `conv2d`, `max_pool2d`, `avg_pool2d`, and `batch_norm` verifiers, and it names
  the offending tensor and its shape. This is a limitation with a tracked plan,
  not a permanent design choice: phase U6 adds real batch loops and removes the
  guard. No supported model changes behaviour, since everything in the suite is
  batch 1.
- Phase U1 (**behaviour change, deliberate**): `run_benchmarks.py` no longer
  reuses a recorded result just because it parses as JSON. A result is
  regenerated unless its cost model constants match and either its manifest sha
  is HEAD or nothing that can move a number has changed since that sha.
- Phase U1 (**tightened assertion**): the end to end tolerance goes from
  `rtol=1e-3, atol=1e-3` to `rtol=1e-5, atol=1e-6`. Observed error is 2.98e-8, so
  the old assertion was five orders of magnitude looser than reality and a four
  order of magnitude accuracy regression would have passed while invalidating the
  README. No current result moves; the assertion simply now means something.
- Phase U1: `op_mapping.py`'s module docstring said all 13 ONNX ops were
  supported and that the unimplemented ones raised a clear "not yet implemented".
  Seven are implemented and the other six were never mentioned in `CONVERTERS` at
  all. Corrected to say which is which.

### Fixed

- Phase U1: `scripts/coverage.sh` ran `ninja check-npu || true`, so it reported a
  coverage percentage whether or not the lit suite passed, and that percentage is
  what the README badge showed. The `|| true` is gone. The script also configured
  `build-coverage` while a stale `build-cov` tree sat in the repository; both are
  now `build-cov`.

### Notes

- The baseline records the instruction count from the simulator's own
  `stats.instructions`. For LeNet that is 28 / 25 / 21 at `-O0` / `-O1` / `-O2`,
  not the 91 / 82 / 70 in the README's headline table, which came from a regex
  over the IR dump that also counts `npuisa.const` and matches inside type
  strings. The README is not corrected here; that is Phase U4, which changes the
  published numbers deliberately and documents the correction.

## [1.0.0] - 2026-07-15

First release: a complete MLIR compiler backend and simulator for a simulated edge NPU,
taking a trained ONNX model to an executable instruction stream validated end to end against
onnxruntime.

### Added

- Phase 0 scaffold: out of tree CMake project, `npu-opt` driver skeleton, lit test harness
  with a smoke test, repository layout per the specification.
- `scripts/dash-lint.sh` enforcing the no em dash and no en dash rule, wired into
  pre-commit and CI.
- Documentation seed: README, BUILD, and an engineering log started at Phase 0.
- Pinned LLVM/MLIR toolchain at tag `llvmorg-22.1.8`, built once with a memory safe
  configuration for this machine's WSL2 budget.
- Phase 1: the `npu` tensor dialect. Ops constant, relu, add, mul, conv2d, matmul,
  max_pool2d, avg_pool2d, reshape, transpose, concat, and batch_norm, with Pure traits,
  a fused activation enum on conv2d and matmul, an optional bias operand for the fused
  ops, and ODS verifiers for shape and type constraints. Round trip and verifier failure
  lit tests, plus an autogenerated `docs/DIALECT_REFERENCE.md` (target `npu-dialect-doc`).
- Phase 2: constant folding for add, mul, and relu; canonicalization patterns (relu
  idempotence, reshape identity, reshape of reshape); and a dedicated `npu-fold-batchnorm`
  pass that folds batch normalization into a preceding convolution's weights and bias with
  real compile time tensor arithmetic. Canonicalize and fold lit tests.
- Phase 3: the `npu-fuse-ops` pass that folds a trailing `relu` into a producing `conv2d`
  or `matmul` (which already carries its bias operand) by setting the fused activation, so
  the intermediate stays in scratchpad instead of round tripping through DRAM.
- Phase 4: dead code elimination validated. Because every op carries the Pure trait, unused
  ops are removed by the upstream canonicalizer, and unused private functions by symbol-dce,
  so no custom DCE pass is needed. lit tests cover both.
- Phase 5: the ONNX frontend (`python/npu_frontend/`). A seeded LeNet style model generator
  exports to ONNX; the importer runs onnx.checker and shape inference, then builds npu dialect
  IR with the MLIR Python bindings, mapping Conv, Gemm, Relu, MaxPool, AveragePool, Reshape,
  and Flatten and failing loudly on anything else. pytest covers model structure, an imported
  LeNet verified through npu-opt, and the loud failure path. black and ruff configured.
- Phase 6: the npuisa instruction dialect (scratchpad buffer type, DMA, compute, and control
  instructions), the `npu-lower-to-npuisa` dialect conversion lowering (DRAM tensor to
  scratchpad buffer with DMA at the boundaries), and the `npu-allocate-scratchpad` linear scan
  allocator that assigns byte offsets and spills the longest lived buffer to DRAM when the
  working set exceeds the budget. lit covers round trip, lowering, and the fits and spill cases.
- Phase 7: the `.nbin` binary encoding with the InstructionEncoder, the npu-objdump
  disassembler, and a C++ simulator with fp32 kernels (conv2d, matmul, relu, add, mul, pooling,
  reshape, DMA) and an analytical cost model reporting simulated cycles and DRAM bytes. GoogleTest
  covers the format round trip, per instruction semantics, and cost arithmetic; a pytest end to
  end test compiles and simulates a LeNet and matches onnxruntime within tolerance. The tools are
  npu-translate, npu-objdump, and npu-sim. All performance numbers are simulated estimates.
- Phase 8: the `npu-compile` driver. One entry point from ONNX to .nbin with optimization levels
  (-O0 import and verify, -O1 canonicalize and fold, -O2 fuse and DCE) and staged output
  (--emit import, npu, npuisa, or nbin), plus --verbose stage timings. pytest covers the emit
  stages, that the opt levels change the IR, and that a driver produced nbin matches onnxruntime.
- Phase 9: the benchmark harness (`experiments/run_benchmarks.py`) sweeping models, optimization
  levels, and scratchpad budgets, writing one JSON per cell with op counts, simulated cycles, DRAM
  bytes, numerical error, and a manifest (git sha, LLVM tag, tool versions, cost constants). It is
  resumable and writes atomically. The harness caught a spill correctness bug: the encoder did not
  assign DRAM offsets to spill temporaries, so spills clobbered the input; fixed by giving each
  spill store its own DRAM region, with an end to end spilling test added.

- Phase 10: the documentation set. ARCHITECTURE, PASSES (each pass with before and after IR),
  DESIGN_DECISIONS, and CONTRIBUTING join the existing BUILD, ONNX_FRONTEND, ISA_MANUAL, the
  auto generated DIALECT_REFERENCE, the engineering log, and this changelog, indexed from the
  README.
- Phase 11: the two reports. The main report (report/) covers the architecture, dialects, passes,
  frontend, ISA, and an evaluation whose results table and inline numbers are generated from the
  benchmark JSON by results_to_tex.py, so nothing is hand copied. The debug report (report_debug/)
  is a first person postmortem grouped by theme. Both build with tectonic to zero errors.

### Fixed

- Encoder: spill `dma_store` temporaries now get their own DRAM regions instead of defaulting to
  offset 0, so scratchpad spilling preserves numerics (verified against onnxruntime).
- npu-compile: the library entry point no longer writes text stages to stdout as a side effect.
