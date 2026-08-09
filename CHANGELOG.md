# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning once a release is tagged.

## [Unreleased]

### Added

- Phase U3: `Program::validate()`, a structured check of every invariant the
  simulator relies on, and `Program::decode()` now means decode plus validate.
  Before this, `decode()` checked the magic, capped a few vector lengths, and
  stopped; everything downstream trusted the result, so the simulator did raw
  pointer arithmetic on unchecked addresses, `Conv2D` indexed `operandAddrs[1]`
  whether or not it existed, and an out of range opcode fell through a `switch`
  as undefined behaviour. A `ValidationError` names the failing rule, the
  instruction index, and the offending construct.
- Phase U3: `Program::decodeUnvalidated()`, for `npu-objdump` alone. A
  disassembler has to be able to dump a suspect file, so it deliberately keeps
  the old permissive path and prefixes the dump with a warning.
- Phase U3: an always on bounds checked scratchpad accessor in the simulator,
  input size checking and multi output writing in `npu-sim`, a multi function
  diagnostic in `npu-translate`, and a multi block diagnostic in
  `AllocateScratchpad`.
- Phase U3: `unittests/Encoding/ValidationTest.cpp` (one test per validation
  rule, each asserting which rule caught it, not merely that something did),
  `PropertyTest.cpp` (a 1000 iteration encode and decode round trip), and
  `FuzzTest.cpp` (a 322 case corpus of malformed inputs, asserting that every
  one is refused or survived rather than crashing).
- Phase U2: CI that builds the compiler and runs every suite. `ci.yml` gains
  `build-and-test` (configure, build, lit, both GoogleTest binaries, pytest, the
  reachability check with its model layer, and the regression baseline),
  `sanitizers` (ASan and UBSan), and `coverage` (which fails below 85 percent
  line coverage). The `lint` job gains black, mypy, and the reachability check.
  Before this, CI ran dash-lint and ruff and nothing else, while the README
  carried a badge that reads as "the test suite passes on every push".
- Phase U2: `.github/workflows/report.yml` builds both PDFs and uploads them as
  artifacts, with a forced rebuild so a checkout's uniform timestamps cannot
  make it succeed having built nothing.
- Phase U2: `.github/workflows/llvm-image.yml` builds the pinned LLVM and MLIR
  base image and publishes it to GHCR. Manual only, and only when the tag
  changes.
- Phase U2: `scripts/check-reachability.py` enforces that every op in the `npu`
  dialect is importable, lowerable, encodable, simulatable, and exercised by a
  benchmark model, or carries a dated exemption in `docs/DESIGN_DECISIONS.md`.
  Its first run found six unreachable ops, three more than the audit had: `add`
  and `mul` have no ONNX converter at all, and `avg_pool2d` is wired end to end
  but exercised by no model.
- Phase U2: mypy over `python/npu_frontend`, configured in `pyproject.toml` and
  wired into pre-commit and CI. The package is now fully annotated, and the
  contract between the importer and its per op converters is expressed as an
  `ImportContext` Protocol instead of a comment.

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

- Phase U3: `Validation.RejectsRegionPastTheEndOfDram` named the
  `region-in-range` rule but set a DRAM offset of 8190, which is not 4 byte
  aligned, so `region-offset` claimed the program first and the rule the test
  exists for was never reached. The offset is now 8160, which is aligned and
  still runs the 10 element output past the end of an 8192 byte DRAM. Nothing in
  the decoder changed; the rule was correct and untested.
- Phase U3: `shapeElements()` tested its running product against the 2^40 element
  cap only after multiplying, so the multiply was itself the signed overflow the
  cap exists to prevent (UBSan reported it at `Program.cpp:197`). A shape such as
  `{2^40, 2^24}` wrapped to a small product, passed the cap, and was then
  compared against a region bound as if it were small, so the guard accepted
  exactly the input it was written to refuse. Each extent is now tested against
  the headroom that is left before it is multiplied in. `kLimit` and the
  function's signature are unchanged, so all three callers keep their contract.
- Phase U2: `docker/Dockerfile.llvm` kept the entire LLVM build tree *and* ran
  `ninja install` to place a second copy alongside it, which is tens of
  gigabytes and not realistically pushable to a registry. Rewritten as two
  stages: the builder compiles, and the shipped stage copies only the headers,
  static libraries, CMake package files, tools, and Python bindings that an out
  of tree project consumes, at the same absolute paths, since the build tree's
  CMake files hardcode them. It also self checks at build time so a broken image
  fails there rather than in somebody's CI run.
- Phase U2: `scripts/coverage.sh` deletes stale `.gcda` counters before running.
  The repository shipped a `build-cov` tree from an earlier run, and gcov refuses
  to merge a counter file whose checksum no longer matches its rebuilt object,
  so coverage for those translation units was whatever survived the collision.
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
