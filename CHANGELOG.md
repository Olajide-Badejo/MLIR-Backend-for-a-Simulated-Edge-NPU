# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning once a release is tagged.

## [Unreleased]

### Interphase P9b: the open questions P9 handed on, decided

Four items the P9 handoff carried as open questions, taken deliberately between
P9 and P10 rather than folded into either. No new pass, no new phase gate.

- **`-npu-fuse-bias` fires on a model of the suite now.** `dilated_stack`'s
  `conv1` was biasless and followed directly by a `Relu`, and it now carries a
  separate channel shaped bias `Add` between them, which is the shape Section
  11's broadcast carve out exists for and which no exported graph produces. So
  the pass's Section 16.2 ablation row at P10 is a measurement rather than a row
  of zeros: `dilated_stack` is thirteen instructions at `-O0` and twelve at
  `-O2`, with the answer bit identical and the cycle count back to exactly what
  it was before the node existed. `GENERATOR_VERSION` moves from `1.0.0` to
  `1.1.0`, every `dilated_stack` cell of the baseline moves, and
  `docs/BREAKING_CHANGES.md` declared all of it before the commit that caused it.
- **The CI image carries `libomp-18-dev`, so two of its three jobs stop making
  a weaker claim than the third.** `build-and-test` and `sanitizers` configure
  with clang, which needs `libomp` and `omp.h`; the coverage job passes no
  compiler, gets gcc, and finds `libgomp` because `g++` brings it. So one image
  printed `OpenMP: not found` in two jobs and `found 4.5` in the third, and
  Section 10.3's determinism assertion, that one thread and the maximum produce
  bitwise equal buffers, was comparing a single threaded run against itself
  wherever OpenMP was absent. The image's own smoke test now compiles, links and
  runs an OpenMP program rather than testing for a header, which is the failure
  class D-0025 was. **This takes effect only once the image is republished**,
  which is a dispatch of `llvm-image.yml` and an hour of runner time.
- **CI has an NDEBUG build for the first time**, which is the release half of
  Section 9.3's "every build mode" clause. The new `ndebug` job configures
  `-DNPU_FORCE_NDEBUG=ON`, asserts in its own configure log that the option
  took, and builds and runs `NPUSimulatorTests` and `NPUEncodingTests`, which
  are the two binaries that link no MLIR and are therefore the two D-0031 allows
  that directory to build. **CI previously claimed this coverage and did not
  have it**, in a comment that said the sanitizers job's `RelWithDebInfo`
  implied `-DNDEBUG`; against an assertions LLVM it does not, which is D-0028
  and now D-0036. A second LLVM tree built without assertions is deliberately
  declined, in `docs/adr/0009`.
- **`scripts/regression-baseline.sh --check` is a CI step**, at the end of
  `build-and-test`, with the golden tolerance left at **zero**. P8 and P9 both
  left it out because a byte identical golden comparison bounds two runs of the
  same build and CI is a different compiler and a different libc; neither phase
  could answer whether that matters without a run. The step is the run. If the
  container reproduces the developer machine bit for bit, that is a property
  worth having proven. **It does.** On the step's first run, all 42 cells, all
  21 golden tensors and the 4.470e-08 largest movement reproduced bit for bit
  under clang in the container against a baseline recorded under gcc on WSL2,
  with zero drift on every numeric field. The tolerance stays at zero, now on
  evidence rather than on principle. Had it not, the drift report is the
  measurement, and
  the report was rewritten to carry it: a golden difference now names how many
  elements moved, at which index, from what to what, and how many units in the
  last place that is.
- **The dash linter runs on the interpreter CI actually has.** Two
  unparenthesised `except A, B:` clauses were PEP 758 syntax, legal on Python
  3.14 and a `SyntaxError` on the 3.12 the CI image ships, so the linter failed
  to parse the first time anything ran it inside the container. The grammar
  target of black and ruff is `py311` now, which is what `requires-python`
  declares, so the floor is enforced by a tool rather than promised in a field
  nothing reads; mypy is at `3.12`, the lowest it can go while numpy's own
  shipped stubs need 3.12. And the baseline runner prints the child's output
  when a dash lint invocation fails, because that suite has no machine readable
  file and a count of zero says nothing a reader can act on. D-0038, found by
  the first CI run of the step above.
- **`scripts/coverage.sh` clears the previous run's execution counters**, so the
  number it reports describes the suite that just ran rather than the union of
  every suite the build directory has ever run. gcov accumulates, and nothing
  had ever deleted a `.gcda`; the visible symptom is a collection that aborts
  once a hot line's counter passes 2^32, and the quiet one is that a line
  covered by a deleted test stays covered. D-0037. The measured percentages do
  not move: C++ 86.5 and Python 90.50 either way.
- **`-sccp`'s ablation row stays zero, and the distinction is written down.**
  Constant propagation needs a call graph to cross and an imported model is one
  function, so no model change alters that row. Two zero rows, one a gap in the
  suite and closed, one a true property of the programs this compiler compiles
  and kept. `docs/PASSES.md` says which is which and
  `test_sccp_has_nothing_to_do_on_a_single_function` holds the second.

### Phase P9: optimization passes and optimization levels

Every user visible movement of this phase is named here, and the one that moves a
recorded number is declared in `docs/BREAKING_CHANGES.md` as well.

- **`-O1` and `-O2` exist and `npu-compile` builds them.** All three levels are
  registered as `npu-opt` pass pipelines and all three are implemented; until
  now the higher two were named and refused by name.

  ```
  scripts/npu-compile model.onnx -O 2 --emit nbin -o model.nbin
  npu-opt model.mlir --npu-O2
  npu-opt model.mlir --npu-O2=stop-after=npu
  ```

  `-O1` is `-npu-constant-fold` and `-canonicalize` on top of `-O0`. `-O2` adds
  `-npu-fuse-bias`, `-npu-fold-batchnorm`, `-npu-fuse-ops`, a second
  `-canonicalize`, `-cse`, `-sccp` and `-symbol-dce`. Layout assignment, tiling,
  double buffering and calibration are the four rows of Section 12's table this
  phase excludes by name; they arrive at P13 and P14 and no level claims them.
- **Four new passes in the `npu` dialect**, each with a positive lit test and at
  least one case proving it does not fire where it should not:
  `-npu-constant-fold`, `-npu-fuse-bias`, `-npu-fold-batchnorm` and
  `-npu-fuse-ops`. `docs/PASSES.md` carries before and after IR, the guards, and
  what each was measured to do on the model suite.
- **`npu.fused_op` and `npu.yield` have a producer**, so both reachability
  exemptions are deleted and `scripts/check-reachability.py` passes with an
  **empty** exemption block for the first time.
- **The answers move at `-O2` on one model, by 4.47e-08.**
  `conv_bn_relu_stack` is the only model in the suite carrying an unfolded batch
  norm; folding it changes the accumulation order, and it also takes that model
  from 23 instructions to 15 and from 1372.50 simulated cycles to 1160.50. Every
  other cell of the baseline is unchanged, `-O0` included and `-O1` included.
  `docs/BREAKING_CHANGES.md` declared it before the commit that caused it.
- **`--emit npu` now runs the level's tensor level passes** rather than
  returning the importer's text unchanged. At `-O0` there are none, so the stage
  parses, verifies and reprints, and it comes out byte identical to
  `--emit import`. At `-O1` and `-O2` it is the optimized IR, which is also what
  the reference interpreter is given, so `refexec` is an oracle for the passes
  and not only for the backend.
- **`npu_frontend.refgraph` executes `npu.fused_op`**, by binding the region's
  block arguments to its operands and walking the body.
- **The end to end matrix gained its level axis.** Seven models, three levels,
  two batch sizes, five input classes, two oracles: four hundred and twenty
  cells, up from a hundred and forty. The tolerances did not move.
- **The regression baseline records every level**, at `schema_version` 2. Forty
  two cells where there were fourteen, twenty one golden tensors where there
  were seven, a `levels` field, and a `max_abs_movement_vs_o0` on every cell.
  `per_level` has left `absent_fields`; `energy` is still there and still names
  P11.
- **`scripts/build-model-ir.py` sweeps every level the compiler builds**, so an
  operation that only a `-O2` pipeline creates has a model IR file to appear in.
  The filenames gained an `-O<level>` component.
- **`npu-opt --npu-describe-pipeline` reports a `stage` per pass**, `npu` or
  `npuisa`, which is how the driver asks for the tensor level half of a level
  without naming its passes.

#### Fixed at P9

- **`-sccp` did nothing at all on this dialect** and now does something: the
  `npu` dialect implements a constant materializer, without which MLIR's
  constant propagation reached the right answer and had nowhere to write it.
  D-0033.
- **Two operations that share a destination get two buffers.** `-cse` merges
  identical `tensor.empty` operations, which is correct on tensors, and the
  lowering used to turn one of them into one `memref.alloc`, which made two
  instructions write one buffer. D-0034.
- **A constant's transfer, and a destination's allocation, are emitted where the
  data is used** rather than where `-canonicalize` and `-npu-fuse-ops` hoisted
  them. Without this `-O1` was 37 percent slower than `-O0` on LeNet, because
  every transfer was serialised ahead of every computation, and `-O2` could not
  place LeNet's tight budget cell at all. D-0035.

### Phase P8: the walking skeleton and the safety net

- **`npu-compile` exists.** One entry point from an ONNX model at the pinned
  opset to a `.nbin`, at `-O0`, with staged output:

  ```
  scripts/npu-compile model.onnx -O 0 --emit nbin -o model.nbin
  scripts/npu-compile model.onnx --emit npuisa
  scripts/npu-compile model.onnx --emit npu --verbose
  scripts/npu-compile --describe-pipeline
  ```

  `--emit` stops after `import`, `npu`, `npuisa` or `nbin`. At `-O0` the first
  two are the same text, because `-O0` runs no `npu` level pass; a test asserts
  that, so the day `-O1` lands the assertion moves with it. `--budget` sets the
  scratchpad budget, `--strip-debug` writes an empty debug section, and
  `--verbose` prints the stage timings on stderr, where they do not land in the
  output somebody is piping. It is a Python driver at
  `python/npu_frontend/compile.py` with a thin launcher at `scripts/npu-compile`;
  there is no `tools/npu-compile` C++ tool, because the import step is Python by
  design.
- **Running the result stays `npu-sim`'s job.** The driver compiles, the way a
  compiler does. `npu_frontend.run_program` wraps the simulator for the test
  harnesses and returns the outputs as arrays and the statistics as a
  dictionary.
- **The end to end matrix runs.** Seven models, two batch sizes, five seeded
  input classes, at `-O0`, each cell checked against `onnxruntime` and against
  the reference interpreter, with the absolute and the relative bound asserted
  separately. `test/Python/test_end_to_end.py`.
- **`generate_model` takes a `batch`.** Section 17.4 sweeps the batch size over
  every model where Section 15 pins one per model, and the batch is now a
  parameter of an export rather than a second registry. The weights and the node
  counts do not move with it, and a batch equal to the registry's writes the
  registry's own file.
- **`npu_frontend.refgraph` executes an `npu` module with `refexec`**, which
  makes the reference interpreter an oracle for a whole model rather than for
  one operation.
- **The coverage thresholds are real, replacing the zeros of Section 19.0.**
  Measured on 2026-08-31: C++ lines 86.1 percent over `lib/Dialect`,
  `lib/Encoding` and `lib/Simulator`, C++ branches 77.3 percent, Python lines
  90.6 percent over `python/npu_frontend`. CI enforces 85 for C++, which is
  Section 17.7's floor, and 90 for Python, which is that section's rule of the
  measured value rounded down to a whole percent. `scripts/coverage.sh` takes
  the Python threshold as a second argument, measures with `pytest-cov` over the
  whole matrix rather than the fast subset, and reports branch coverage for the
  allocator and the decoder separately, where Section 17.7 says the error paths
  matter most: the allocator 97.8 percent of lines and 90.2 of branches, the
  decoder 94.6 and 91.4.
- **A Python threshold that cannot be measured is a failure, not a skip.** If
  the suite's dependencies are not importable and a nonzero Python threshold was
  asked for, `coverage.sh` fails rather than passing on a number nobody
  computed. At a threshold of 0 it prints an off line and continues.
- **`scripts/coverage.sh` tells the Python suite which build directory it
  built.** It exports `NPU_BUILD_DIR`, derives `MLIR_PYTHON_PACKAGES_DIR` from
  that directory's CMake cache when the caller has not set one, and refuses
  before running pytest if the binaries it just built are missing. Without this
  the suite looked for them in `build/`, which the coverage job does not have.
  D-0032, found by CI.
- **The test suite has one rule for finding a built binary**,
  `test/Python/tools.py`, replacing three that disagreed. A missing binary is a
  skip when nobody named a build directory and a **failure** when somebody did,
  because naming one asserts that the build is there.
  `test/Python/test_tool_discovery.py` makes a second copy of the rule a red
  test.
- **`scripts/check-reachability.py` runs in full and passes.** All five layers
  of law 2 are decided for every operation of the `npu` dialect, and the CI
  step is on. Every **imported computation** operation meets all five with no
  exemption of any kind; the two structural operations, `npu.fused_op` and
  `npu.yield`, carry dated exemptions from the model layer naming P9, because
  `-npu-fuse-ops` is the only thing that creates one and it lands at P9.
- **The simulation layer became mechanical**, which is what P7 left on this
  phase's desk. It used to be a substring search over
  `lib/Simulator/Simulator.cpp`'s operation table, which is a comment; it is now
  decided from `docs/ISA_OPCODES.json`, generated from the ISA description and
  kept honest by the staleness gate, the way P6 made the encoding layer
  decidable. The description gained a `needs_kernel` field per opcode for it.
- **`scripts/build-model-ir.py`** writes every model's `npu` and `npuisa` level
  IR into `experiments/models/`, at both batch sizes, which is what the model
  layer of Section 17.5 reads. It is a build artifact and nothing commits it.
- **`scripts/regression-baseline.sh` records the safety net of Section 17.6, and
  `--check` diffs against it.**

  ```
  bash scripts/regression-baseline.sh            build, measure, record
  bash scripts/regression-baseline.sh --check    build, measure, diff, fail on drift
  ```

  It records into `test/baseline/baseline.json` the pass, fail and skip counts
  and the test names of every suite plus the dash lint, the git sha, the tool
  versions, and one cell per model per level per budget carrying the instruction
  count, the cycles, the two DRAM byte counts, the MAC count and the largest
  absolute error against onnxruntime. The `-O0` output tensors go to
  `test/baseline/golden/` as `.npy`. Every number comes from a machine readable
  source: lit's `--output`, GoogleTest's `--gtest_output=json:`, pytest's
  `--junitxml` and the simulator's `--json-stats`. Nothing parses a log.
- **The baseline schema is versioned and does not claim what it cannot
  compute.** `schema_version` is 1, `--check` refuses a version it does not
  recognise rather than reading a later field as a regression from zero, and
  `absent_fields` names `energy` as arriving at P11 and the per level fields as
  arriving at P9.
- **Every model carries a tight scratchpad budget**, measured on 2026-08-31 and
  frozen as a required field of its `ModelSpec`, with `TIGHT_BUDGETS` derived
  from the registry. `docs/adr/0008-per-model-tight-scratchpad-budgets.md`
  records the measurement and two deliberate deviations from Section 15: the
  rounding quantum is the allocator's own 64 byte alignment rather than 4096,
  and the fixed fraction that section asks for is inoperative until tiling
  lands at P13, because on five of the seven models the smallest allocatable
  budget is the peak itself.
- **`npu_frontend.metamorphic`** holds four of Section 17.3a's five metamorphic
  relations and its dead subgraph injection. The fifth, pad then slice back,
  cannot be written: `Pad` is refused by this importer by name and `Slice` has
  no converter, and adding either so a test could use it would be growing the
  operator set to satisfy a test. It is recorded in `NOT_IMPLEMENTED` with that
  reason and a test asserts the reason is still true.
- **`npu-opt --npu-describe-pipeline` also reports `eliminates_dead_code` per
  pass**, which is how the dead subgraph check knows which levels its
  instruction count claim applies to. Both of `-O0`'s passes declare `false`.
- **`npu_frontend.input_classes`** holds the five classes of Section 17.4 with
  their seeds derived from the cell's own identity, so a failing cell's input
  is reconstructible from the cell's name with no run log.
- **`npu-opt` has optimization levels.** `--npu-O0` runs the `-O0` pipeline of
  Section 12, which is import and verify followed by `-npu-lower-to-npuisa` and
  `-npu-allocate-scratchpad`. The level's pass list lives in `lib/Pipeline/` in
  C++ rather than in the Python driver, because the `PassInstrumentation` of
  Section 16.2 has to sit on the `PassManager` that actually runs the passes.
  The allocator's options are forwarded, so `--npu-O0=budget=8192` reaches it.
- **`--npu-O1` and `--npu-O2` are not registered and asking for one is an
  unknown argument.** They arrive at P9. A level registered with an empty
  pipeline would run and produce `-O0`'s answer, which is worse than a refusal.
- **`npu-opt --npu-describe-pipeline`** prints every level as JSON with its
  passes and their `ablatable` properties. That is how a caller reads the
  ablatable set at run time instead of keeping a second copy of it.
- **`npu-sim --json-stats <path>`** writes the run's statistics as JSON, which
  is what a caller reads. Nothing scrapes them out of the human readable form,
  because the one number nothing may guess at is `stats.instructions` and it is
  exactly the number a text parser would guess at. The keys are the text labels
  with their spaces turned into underscores, plus `reached_halt` and
  `single_port`, and the file is written only when the run succeeded. The flag
  is not spelled `--stats-json`: LLVM's Support library registers that name for
  its own statistics counters and every tool linking Support inherits it.

### Phase P6: the binary format and the generated ISA

- **The instruction set is described once, in
  `include/NPU/Encoding/NPUISADescription.td`.** The `Opcode` enum, `kMaxOpcode`,
  the arity, field presence, memory space and element type rules the validator
  reads, the validation check names, the disassembler's format strings, the
  simulator's dispatch skeleton, the opcode table in `docs/ISA_MANUAL.md` and the
  opcode list `scripts/check-reachability.py` reads are all generated from it by
  `npu-isa-tblgen`. Before this they were four hand maintained places that
  nothing but discipline kept consistent.
- **`ninja -C build npu-isa-doc` regenerates `docs/ISA_MANUAL.md` and
  `docs/ISA_OPCODES.json`**, and `scripts/check-isa-staleness.sh` is the CI gate
  that fails when the committed artifacts drift from the description. It is the
  same shape as the `DIALECT_REFERENCE.md` gate that has run since P1, so there
  is one staleness mechanism in this project rather than two.
- **`scripts/check-reachability.py` decides the encoding layer from
  `docs/ISA_OPCODES.json`** rather than by searching a source file for the
  operation's mnemonic. This is a user visible change to what the check reports:
  the encoding layer becomes decidable from this phase, and the `--skip-models`
  run now says `layers checked: import, lowering, encoding`.
- **`docs/ISA_MANUAL.md` exists**, written like a small processor manual.
- **The `.nbin` binary format exists**, at `Program::kVersion = 1`. A fixed
  header, then input regions, output regions, constants with their data, the
  allocator's spill slots, the instructions, and an optional debug section, each
  length prefixed by a `u32` count and none of them tagged. The byte order is
  **host order**, so a `.nbin` is not portable across byte orders, and the
  manual says so plainly rather than claiming little endian.
- **`ElemType` is `{ F32 = 0, I8 = 1, I32 = 2 }` and every instruction carries a
  `scale`, a `zeroPoint`, a `requantMultiplier` and a `requantShift` from
  version 1.** Those fields, and nothing broader, are what let P14 land without
  a version bump.
- **`Program::kMaxCount` is `1 << 28`**, one constant applied to every `u32`
  count field, reported by the `count-cap` check. The decoder additionally
  refuses any count whose payload cannot fit in the bytes that remain, before
  anything is reserved.
- **`Program::validate()` implements all thirty three checks of Section 9.2**
  and returns a structured error naming the check and the instruction index.
  `Program::decodeUnvalidated()` exists for `npu-objdump`.
- **`-npu-lower-to-npuisa` writes `npuisa.arg` on every argument of the function
  it produces**, holding `"in"` or `"out"`. This is a user visible change to the
  IR the pass emits. The argument order is unchanged; what is new is that the
  input and output split is stated rather than counted, and the encoder refuses
  a function without it. `docs/ARCHITECTURE.md` records the decision and the
  alternatives, and `docs/PASSES.md` records the behaviour.
- **The encoder assigns the DRAM map**: inputs, then outputs, then constants,
  then the allocator's `npuisa.spill_slot` allocations, each aligned to 64
  bytes. That discharges the obligation P5 left on this phase.
- **`npu-translate` exists.** Allocated `npuisa` IR in, a `.nbin` out, with
  `--strip-debug` for an empty debug section. It fails and writes no output file
  on an operation it cannot encode, and the output file is not created before
  the encode result is known. A module holding more than one function is
  diagnosed rather than truncated.
- **`npu-objdump` exists.** It disassembles a `.nbin`, and it decodes without
  validating so that a suspect file can still be dumped, with a warning block
  naming the check that rejected it.
- **The debug section is populated from MLIR locations**, so a pipeline
  assembled with a pipe needs `npu-opt --mlir-print-debuginfo` for the ONNX node
  names to survive as far as the encoder. Without it the binary carries an empty
  debug section, which is legal.
- **`NPUEncodingTests` exists**, the fourth GoogleTest binary. It carries the
  round trips and the frozen constants, every named check of Section 9.2
  triggered once, the property test of Section 17.2 at 1000 iterations and a
  fixed seed, and the seed and regression corpus of Section 17.3 at 733 cases.
- **A `RESHAPE` whose result holds a different number of elements from its
  operand is now rejected**, by the `result-shape` check. The manual stated the
  rule and nothing enforced it.
- **`fuzz/nbin_decode_fuzzer` exists**, behind `-DNPU_ENABLE_FUZZERS=ON`, which
  needs a clang toolchain. It decodes, validates, and asserts two properties
  beyond crash freedom: a file it accepts re-encodes to exactly the bytes it
  came from, and a file it frames survives the disassembler. `fuzz/corpus/`
  carries eight seeds, every one produced by the tools rather than typed, and
  `fuzz/README.md` carries the recipe that regenerates them.
- **Four range diagnostics changed wording**, from "runs from A to B" to "runs
  from A for N bytes". This is a user visible change to error messages. The old
  form computed `A + N` for a range the check had just refused, which is signed
  overflow when the address is near the limit; the operand extent comparison had
  the same defect and was not only a message. Found by
  UndefinedBehaviorSanitizer over the corpus, recorded as D-0021.
- **`npu-objdump` prints an instruction whose mandatory operand is missing**,
  as `<missing operand N>`, where it used to print a blank line and drop the
  opcode with it. That is the tool failing at its only job on exactly the file
  it exists for. Recorded as D-0023.
- **The disassembler no longer overflows on an unvalidated shape.** Deciding
  whether to print an operand's strides walked the contiguous layout its shape
  implies, multiplying extents without a guard, on the one path in the project
  that runs before validation. Found by `nbin_decode_fuzzer`, minimized, and
  committed as a corpus seed. Recorded as D-0022.
- **Four CI activations**, per the table in Section 19.0: the
  `NPUEncodingTests` step, the ISA staleness step, the `sanitizers` job, which
  becomes a real clang build with `-fsanitize=address,undefined` plus a
  budgeted minute of the coverage guided target, and `nightly.yml`'s fuzz job,
  which runs for thirty minutes from the committed corpus and the exported
  seeds.
- **`scripts/check-isa-staleness.sh` compares against `HEAD` rather than the
  index.** A hand edit to the committed manual passed the gate, because
  regeneration overwrote it before the diff ran. Found while rehearsing the
  activation proof.

### Phase P5: scratchpad allocation

- **`-npu-allocate-scratchpad` exists.** Every `memref.alloc` in
  `#npu.scratchpad` is replaced by a `memref.view` at a constant byte offset
  over one flat `memref<Nxi8, #npu.scratchpad>`, which is Section 8's rule that
  an offset is an SSA operand and not a discardable attribute. This is a user
  visible change in the shape of the IR the pipeline produces: after this pass
  there are no typed scratchpad allocations left, and two buffers that are never
  live together share bytes.
- **Liveness follows views.** A use through a `memref.reinterpret_cast`,
  `memref.view`, `memref.subview` or `memref.cast` is a use of the buffer
  underneath, so the rank 1 scale buffer of ADR 0005 stays live for as long as
  the multiply reads it.
- **Both offset assignment strategies are selectable**, through
  `strategy=pack`, the greedy by size algorithm TFLite Micro's arena planner
  ships, and `strategy=interval`, the named baseline that places in definition
  order. They produce different placements on the same program: the
  fragmentation case in `test/Dialect/NPUISA/scratchpad-alloc.mlir` comes out at
  768 bytes under one and 1024 under the other, for a peak of 768.
- **Both spill heuristics are selectable**, through
  `spill-heuristic=longest-range` and `spill-heuristic=cost`. The default is
  `longest-range` and it is **provisional**: Section 13.1 requires the default to
  be chosen with ablation data, which lands at P13.
- **Spilling emits a `dma_store` after the definition and a `dma_load` before
  each later use**, which is the second of the three permitted DMA producers of
  Section 8. The count is recorded on the function so the sum over the three is
  checkable.
- **The allocator allocates DRAM, for spill slots only.** A spilled value lives
  in a `memref.alloc` in `#npu.dram` marked `npuisa.spill_slot`. This amends
  P4's statement that nothing below the tensor level allocates DRAM;
  `docs/ARCHITECTURE.md` records the amendment and the obligation it places on
  the encoder at P6.
- **Six new function attributes**, which the encoder and the simulator read:
  `npuisa.scratchpad_budget`, `npuisa.scratchpad_bytes`,
  `npuisa.scratchpad_peak_bytes`, `npuisa.fragmentation_ratio`,
  `npuisa.spill_count` and `npuisa.spill_dma_count`. The default budget is
  1048576 bytes when a function carries none.
- **New diagnostics.** A multi block function, a budget too small even after
  everything spillable has been spilled, an unknown `strategy` or
  `spill-heuristic` value, an alignment that is not a positive power of two, an
  allocation whose size cannot be computed, and a malformed
  `npuisa.scratchpad_budget` attribute are each refused by name. Every bad
  option is reported rather than only the first.
- **A memref's byte range is now measured from its strides.** This changes what
  `mlir::npuisa::overlaps` answers in two cases: a stride 0 broadcast view spans
  the C floats it addresses rather than the shape it is cast to, and a
  `memref.subview` spans the bytes it reaches across rather than the elements it
  holds. The second was unsound in the unsafe direction and is defect D-0018;
  the first is the question P4's handoff left open. No pass emits a subview yet,
  so nothing observable moves today.
- **An asymmetric `pads` array is accepted by the `npuisa` windowed verifier**,
  where it was rejected before. `npuisa.conv2d`, `npuisa.pool_max` and
  `npuisa.pool_avg` reordered their pads before computing the output extent, so
  every window with `padTop != padBottom` or `padLeft != padRight` was refused
  with a wrong implied extent. This is a user visible change: `dilated_stack`
  compiles for the first time, and it is the one model in the suite with
  asymmetric padding. Defect D-0019. No symmetric case moves, because under a
  symmetric pad the two orders agree, which is why it survived from P2.
- **`NPUAllocatorTests` exists** and its CI step is switched on, per the
  activation table. It carries Section 17.2's property test at 1000 randomized
  interval sets and a fixed seed.
- **`experiments/` exists**, with the compile time benchmark of Section 13.1 at
  four sizes and the per model fragmentation ratio report.

### Phase P4: lowering to `npuisa`

- **`-npu-lower-to-npuisa` exists.** It converts the `npu` tensor dialect to
  `npuisa` instructions on `memref`s in the two memory spaces of Section 8, and
  it is registered in and runnable from `npu-opt`. A function argument becomes a
  `#npu.dram` buffer with one `npuisa.dma_load` into the scratchpad if the body
  reads it, an `npu.constant` becomes an `npuisa.const` in DRAM with one load, a
  `tensor.empty` destination becomes a `memref.alloc` in the scratchpad, and a
  returned value becomes one `npuisa.dma_store`. Immediately after the pass, DMA
  appears only at those boundaries.
- **A lowered function returns nothing.** Its results become trailing
  `#npu.dram` arguments, appended after the inputs, and `func.return` carries no
  operands. This is a user visible change in the shape of the IR the pipeline
  produces: the encoder's input and output regions are read out of the argument
  list.
- **An unfolded batch norm compiles rather than failing.** It decomposes into a
  multiply and an add over per channel constants computed at rewrite time, with
  the evaluation order documented in `docs/PASSES.md` because it is observable.
  A parameter that is not an `npu.constant`, and a variance plus epsilon that is
  not positive, are refused with a diagnostic naming the operation.
- **A rank 1 right hand operand on `npu.add` or `npu.mul` lowers to a stride 0
  view**, which is the obligation `docs/adr/0005-channel-broadcast-on-add-and-mul.md`
  placed on this phase. No opcode was added: the view is a
  `memref.reinterpret_cast` at the destination's extents with strides
  `[0, 1, 0, 0]`, and it adds no transfer.
- **An `npu.fused_op` region is flattened** into its parent, so a fused
  convolution and its activation have no DMA between them.
- **A layout encoding becomes the memref's strided layout map.** An NHWC tensor
  lowers to a buffer with NCHW extents and the strides that permutation implies;
  an NCHW tensor gets no layout map.
- **New diagnostics.** An `scf` operation, a function with more than one block, a
  function declaration, a dynamic extent, an unsupported element type, a
  non constant batch norm parameter, a layout encoded constant and a layout
  changing transpose are each refused by name rather than as a generic
  legalization failure.
- **A dynamic extent is now refused by the `npu` dialect's type constraints**
  rather than reaching a verifier that aborted on it. `NPU_FloatTensor`,
  `NPU_QuantTensor` and `NPU_AnyTensor` are `StaticShapeTensorOf` instead of
  `RankedTensorOf`, which is what `NPUTypes.td` has claimed since P1. This
  narrows what the dialect accepts: `tensor<?x4xf32>` no longer parses on any
  `npu` operation, where before it parsed and then aborted the tool inside
  `npu.reshape`. Defect D-0015. `docs/DIALECT_REFERENCE.md` changes 46 operand
  rows from "ranked tensor of" to "statically shaped tensor of".
- **`docs/PASSES.md` exists**, per ground rule 12, with before and after IR for
  every pass that has landed. No ablation delta is quoted yet and each entry says
  so rather than leaving the field blank; the harness lands at P10.
- **`test/Dialect/NPUISA/dma-boundaries.mlir` exists**, deferred from P2 to the
  phase that has a lowering to assert against.

### Phase P3: the ONNX frontend and the model suite

- **`onnxscript` is a dependency now.** torch's dynamo exporter imports it at
  export time and nothing else in the environment pulled it in, so the model
  generator could not have run without it. `pyproject.toml` pins
  `onnxscript==0.7.1` and `requirements-lock.txt` carries it and the
  `onnx-ir==1.0.0` it resolved to.
- **The ONNX frontend exists.** `python/npu_frontend` imports an ONNX model at
  opset 23 to `npu` dialect IR on tensors, over sixteen converters: `Add`,
  `AveragePool`, `BatchNormalization`, `Clip`, `Concat`, `Conv`, `Flatten`,
  `Gemm`, `GlobalAveragePool`, `Identity`, `MatMul`, `MaxPool`, `Mul`, `Relu`,
  `Reshape` and `Transpose`. Anything else is refused by name, and
  `QuantizeLinear`, `DequantizeLinear` and `Pad` are refused with a reason
  rather than as merely unknown.
- **Nothing leaves the frontend unverified.** `import_model` returns the text
  `./build/bin/npu-opt` printed, so every module goes through the real parser,
  the real verifiers and the real printer, and any `npu` operation carrying a
  discardable attribute is rejected. `npu-opt` is a runtime dependency of the
  package rather than a test one; when it cannot be found the importer raises
  and names the three places it looked.
- **Every compute operation gets a `tensor.empty` destination**, materialised
  immediately before it, and every operation carries a `NameLoc` with its ONNX
  node name which the returned text prints.
- **`docs/ONNX_FRONTEND.md`** is the frontend's contract: the converter table,
  the broadcasting policy and its channel carve out, the `Clip` policy, the
  `AveragePool` and `Reshape` rules, and a how to section.
- **The seeded model suite exists**, seven structurally distinct models from one
  fixed seed: `lenet`, `depthwise_separable`, `resnet_block`,
  `inception_block`, `conv_bn_relu_stack`, `dilated_stack` and `lenet_batched`.
  Five are exported from PyTorch through the dynamo exporter at opset 23; the
  conv plus batch norm stack and the dilated stack are built with the ONNX
  construction API, for the two reasons Section 15 gives. `GENERATOR_VERSION` is
  `1.0.0`. No `.onnx` file is committed; they are regenerated from the seed.
- **`mypy` and `pytest` are on in CI.** `mypy` runs in the lint job over
  `python/npu_frontend` and `scripts`, with `numpy` installed beside it because
  it ships the type information the frontend is checked against. `pytest` runs
  in `build-and-test`, after the build because it needs a built `npu-opt`, with
  the Python dependencies installed from `requirements-lock.txt` and an
  `actions/cache` on the pip cache keyed on that file's hash. An exit code of 5
  is turned into a failure with a message: an empty collection is never read as
  a pass.
- **The depthwise block's global pooling exports as `AveragePool` with a full
  extent kernel, not as `GlobalAveragePool`.** The dynamo exporter lowers every
  spelling of adaptive average pooling in torch 2.13, including
  `x.mean(dim=(2, 3))`, to a `ReduceMean` node, which is not in this project's
  operator set. The two compute the same thing and import to the same
  `npu.avg_pool2d`. `GlobalAveragePool` gets its suite model in the conv plus
  batch norm stack instead.
- **`npu.add` and `npu.mul` accept a rank 1 channel broadcast on the rhs.** IR
  that `npu-opt` previously rejected now verifies: an addend or a scale of rank
  1, whose length is the result's channel extent read through its layout,
  against a rank 4 result. The lhs is still always the result shape and only the
  rhs may be rank 1, so a channel broadcast has exactly one spelling. Nothing
  that verified before is rejected now and no numerics move. This is Section
  11's carve out made representable: without it a per channel `Mul` has nowhere
  to go and `-npu-fuse-bias` has nothing to fuse. Reasoning in
  `docs/adr/0005-channel-broadcast-on-add-and-mul.md`, and D-0012.

### Phase P2: the `npuisa` dialect and the memory model

- **The `npuisa` dialect exists.** `npu-opt` now parses, verifies and prints the
  instruction level operations on memrefs in two memory spaces: `const`,
  `dma_load`, `dma_store`, `dma_load_async`, `dma_store_async`, `await`,
  `matmul`, `conv2d`, `add`, `mul`, `relu`, `pool_max`, `pool_avg`, `reshape`,
  `transpose` and `concat`. `NOP` and `HALT` are properties of the encoding
  rather than of the instruction stream and are not operations here; `QUANT` and
  `DEQUANT` arrive with their integer kernels at P14.
- **Every compute instruction is destination passing on memrefs** and has no
  results, with `ins` read and one trailing `outs` written, implementing
  `DestinationStyleOpInterface` and `MemoryEffectOpInterface`. The effects are
  declared per operand, so a consumer can ask which buffer is written rather than
  only whether memory is touched.
- **`!npuisa.token` and the asynchronous transfers.** A token is a scheduling
  handle with exactly one use, which must be an `npuisa.await` in the same block
  and after its producer. No operation between an asynchronous transfer and its
  await may access memory overlapping the destination, decided on declared memory
  effects plus byte range arithmetic over `memref.view` and `memref.subview`
  offsets, never on SSA identity. A non static offset is refused with a
  diagnostic rather than assumed disjoint.
- **An asynchronous transfer whose await is the very next operation canonicalizes
  to the synchronous form.** This is a new observable rewrite: IR that entered
  `-canonicalize` as `dma_load_async` plus `await` leaves it as `dma_load`.
- **A memref with no memory space is now rejected with a diagnostic** where it
  previously crashed `npu-opt` with no output at all (D-0008).
- **Two asynchronous transfers in flight at once now verify** where they were
  previously rejected (D-0009). This is the shape double buffering produces, so
  the change is what makes the asynchronous form usable at all.
- **`NPUInterfaceTests` is built and is on in CI**, covering the two interfaces
  and the overlap arithmetic. The unit test binaries now build against either an
  LLVM build tree's bundled gtest or a system GoogleTest package, so they build
  in CI as well as locally.
- **`scripts/coverage.sh`** measures C++ line and branch coverage in a separate
  build directory, writes a gcovr JSON summary, and exits nonzero below a
  threshold passed as its first argument, defaulting to 0.
- **The CI coverage job is on**, at a threshold of 0, and uploads the JSON and
  HTML summaries as an artifact. Real thresholds arrive at P8.
- **The LLVM image gains googletest, the MLIR Python bindings, and gcovr.** The
  bindings install to `/opt/llvm/python_packages/mlir_core`, which is on
  `PYTHONPATH` in the image. This requires republishing the image.
- **`docs/ARCHITECTURE.md`** records the memory model design and is binding on
  later phases.

### Phase P1: the `npu` dialect

- **The `npu` dialect exists.** `npu-opt` now parses, verifies and prints
  fourteen operations on tensors: `constant`, `conv2d`, `matmul`, `add`, `mul`,
  `relu`, `max_pool2d`, `avg_pool2d`, `reshape`, `transpose`, `concat`,
  `batch_norm`, `fused_op` and `yield`. `quantize` and `dequantize` are
  deliberately absent until the phase that brings their converters, kernels and
  models with them.
- **Two memory space attributes and one layout attribute.** `#npu.scratchpad`
  and `#npu.dram` are usable as `memref` memory spaces and print exactly as
  written. `#npu.layout<nchw>` and `#npu.layout<nhwc>` are the encoding of a
  rank 4 tensor, and an absent encoding means NCHW.
- **The compute operations are destination passing from the first commit.**
  `conv2d`, `matmul`, `add`, `mul`, `relu`, both pools, `transpose`, `concat`
  and `batch_norm` each take a destination tensor as their last operand and
  implement `DestinationStyleOpInterface`. Two rules are verified on every one
  of them: the destination type equals the result type exactly including any
  layout encoding, and `ins` and `outs` partition the operands exactly once.
- **Arithmetic shape verification**, resolved against the opset 19 pooling
  specification including the `ceil_mode = 1` rule that drops a window starting
  in the right padded region. One helper computes the output extent for all
  four windowed operations and is shared between `inferReturnTypes` and the
  verifiers, so the two paths cannot disagree.
- **`InferTypeOpInterface`** on `constant`, `conv2d`, `matmul` and both pools.
- **`TilingInterface` is implemented** on the ten compute operations, as
  external models registered by `registerNPUTilingInterfaceExternalModels` and
  promised by the dialect with `declarePromisedInterface`. No pass consumes it
  yet, deliberately: an interface bug and a policy bug are told apart by not
  writing them in the same session.
- **`docs/DIALECT_REFERENCE.md` is generated and committed**, by the new
  `npu-dialect-doc` build target. CI regenerates it and diffs, so the reference
  cannot drift from the dialect it documents.
- **`scripts/check-reachability.py`** enforces law 2 over the operation list,
  reading each operation's classification out of its ODS description rather
  than out of a table the script keeps. `--skip-models` runs the subset that
  needs no built model, and both modes report which layers they actually
  checked rather than reporting a bare pass.
- **Two CI steps switch on** per the activation table: the
  `check-reachability.py --skip-models` lint step and the
  `DIALECT_REFERENCE.md` staleness step.

### Headline change: the published instruction counts were wrong and are corrected

**The LeNet instruction counts in the README, the report, and the plots change
from 91 / 82 / 70 to 28 / 25 / 21 at `-O0` / `-O1` / `-O2`.** The tight budget
cells change from 91 / 94 / 86 to 28 / 31 / 29. Nothing about the compiler
changed; the old numbers were never the instruction count.

`run_benchmarks.py` computed `instruction_count` as
`sum(count_ops(ir, "npuisa").values())`, a regex over the final IR dump. That
regex matches inside type strings, so every `!npuisa.buffer<...>` was counted as
an occurrence of an op named `npuisa.buffer`, and it counts `npuisa.const`,
which the encoder emits as DRAM data and never as an instruction. The simulator
has reported the true count in `stats.instructions` all along and the harness
ignored it.

The repository committed both answers side by side: `experiments/results/*.json`
said 91 / 82 / 70 while `test/baseline/baseline.json` recorded 28 / 25 / 21 for
the same six cells, and the README's own disassembly excerpt said "21
instructions" three screens below a table claiming 70. All six cells now agree
with the recorded baseline exactly.

`count_ops` and `npuisa_op_counts` are kept: the per op histogram is real data
and is useful as a histogram. Only the scalar was wrong. A missing
`instructions` field now raises rather than falling back to the regex.

### Phase U3 summary: the user visible surface

Everything below is detailed in its own entry; this is what changed for someone
running the tools rather than reading the source.

- `Program::decode()` now means decode **and** validate. A `.nbin` that decodes
  but violates an invariant is refused, naming the rule and the offending
  construct, instead of being handed to the simulator. `npu-objdump` keeps the
  old permissive path through `decodeUnvalidated()` and warns.
- `npu-sim` takes one `--input` per declared input region, refuses a count that
  does not match the program, refuses an input file whose float count does not
  match its region, and writes every output rather than only the first.
- `npu-translate` refuses a multi function module instead of encoding the first
  function and dropping the rest, and refuses an op it cannot encode instead of
  printing an error, writing the `.nbin`, and exiting 0.
- `AllocateScratchpad` refuses a multi block function instead of silently
  allocating for the first block.
- The simulator sizes its scratchpad from the declared budget only, so a program
  writing outside it is refused rather than accommodated, and it refuses an out
  of bounds access gracefully in every build mode rather than aborting under
  asserts.
- `docs/ISA_MANUAL.md` documents the real format, the byte order, every
  validation rule by check name, and the version policy.

### Added

- Phase U4: `test/Python/test_end_to_end.py` is now a parametrized matrix of
  every model times three optimization levels times both scratchpad budgets
  times five input classes (`normal`, `zeros`, `large_pos`, `large_neg`,
  `relu_knee`), thirty cells, replacing two tests that ran one hardcoded pass
  list matching no `-O` level and validated against a single standard normal
  draw. Compilation goes through `compile_model(opt_level=...)`, so the tests
  exercise the pipelines that ship.

  Both bounds are asserted separately rather than through
  `np.testing.assert_allclose`, whose combined `atol + rtol * |b|` criterion lets
  an absolute allowance absorb a relative failure. The absolute bound is scale
  aware, expressed as float32 ulps at the magnitude of the output with `ATOL` as
  a floor: `large_pos` and `large_neg` produce outputs two orders of magnitude
  larger than `normal`, where a fixed `1e-6` is half an ulp and unsatisfiable by
  any correct implementation. Nothing that passed before is loosened, since the
  floor still dominates at the `normal` scale. Worst observed across the matrix:
  absolute 1.526e-5 (8 ulps at that scale), relative 4.848e-6 against a 1e-5
  bound.

  Cells outside the fast subset carry a `slow` marker and `pyproject.toml` sets
  `addopts = "-m 'not slow'"`, so the default run is one cell per level in about
  a second while CI runs the whole matrix.
- Phase U4: every result records `max_rel_error_vs_onnxruntime` alongside the
  absolute error. The absolute error alone cannot tell a small error on a small
  output from a small error on a large one.
- Phase U4: leave one out ablations. For every distinct pass in `-O2`, the
  harness compiles the model with that pass removed, encodes, simulates, and
  checks against onnxruntime exactly as a normal cell does, recording the result
  as `lenet_O2_ablate_<pass>_<budget>.json` with a `baseline_cell` naming the
  full `-O2` result it is a delta against, absolute values, and
  `delta_instruction_count`, `delta_simulated_cycles`, `delta_dram_bytes_total`,
  and `delta_compile_ms`. Run at both budgets, and generated into
  `report/generated/ablation_table.tex` and `docs/images/ablations.png`.

  Two results worth naming. At the 1 MB budget only `-npu-fuse-ops` buys
  anything (+4 instructions and +298 simulated cycles when removed); removing
  `-canonicalize` or `-symbol-dce` produces a byte identical program, because
  `-npu-fuse-ops` drives its patterns with `applyPatternsGreedily`, whose fixed
  point loop already folds and erases dead ops. Canonicalization is still
  responsible for the whole DRAM halving at `-O1`, where it is the only pass.

  And at the 140 KB budget `-npu-fuse-ops` is **counterproductive**: removing it
  saves 96 simulated cycles and 6.1 KB of DRAM traffic, because fusing an
  activation into its producer extends that value's live range and under a
  budget that already spills, a longer live range costs a spill and a reload. A
  table reporting only the generous budget would have missed it.

  An ablation whose numerics move beyond the end to end tolerance raises rather
  than being recorded, since that would mean the pass is load bearing for
  correctness rather than performance.
- Phase U4: every benchmark result carries a `passes` array, one entry per pass
  in that cell's pipeline, in order, recording the pass name, its zero based
  position, the op histogram before and after it, both totals, and its wall
  clock in milliseconds. The v2 specification called per pass measurement "the
  evaluation's backbone" and the harness recorded one total `compile_ms` and one
  post lowering histogram, so the report could say what `-O2` buys over `-O0`
  but not what canonicalization buys, or fusion, or symbol DCE.

  Op counts come from `npu-opt`'s own `print-op-stats` pass in JSON form, not
  from a regex over the printed IR. Wall clock comes from `--mlir-timing
  --mlir-output-format=json`, recorded in a `pass_timing_source` field so a
  reader can tell it was measured. A pass in the pipeline with no timing raises
  rather than recording zero, and a `print-op-stats` format shift raises rather
  than returning an empty histogram, which would record every pass as having
  changed nothing.

  The pipeline is read from `_passes_for_level` at run time, so a pass added to
  a level is instrumented without editing the harness, and `-O0` correctly
  records only the two lowering passes.
- Phase U4: the result manifest records `cpu_model` and a `wall_clock_note`.
  Per pass wall clock and `compile_ms` are real measurements rather than
  simulated estimates, so they mean nothing without the machine attached and
  must not be compared across machines.
- Phase U3: `npu-sim` accepts `--input` once per declared input region, in
  declaration order, and checks each file's float count against that region's
  shape. It used to keep a single input path, so a two input program ran with
  its second input left as zeros and said nothing, which is a confident wrong
  answer. A count that does not match the program is now refused with both
  numbers in the message (**behaviour change, deliberate**: a program with
  declared inputs run with no `--input` used to simulate them as zeros). The
  usage string and the Tools section of `docs/ISA_MANUAL.md` describe it, and
  the manual now documents `npu-sim` and the numbered multi output files at all.
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

- The six benchmark results are regenerated on the clean post U3 tree, so
  `experiments/results/*.json` traces to a commit that exists and matches HEAD.
  No measured number moved: `instruction_count`, `simulated_cycles`, the DRAM
  counters, and `max_abs_error_vs_onnxruntime` are identical across all six
  cells, and only `manifest.git_sha`, `manifest.timestamp`, and the wall clock
  `compile_ms` differ. That is the expected outcome, since U3 was validation and
  diagnostics and added no optimization.
- `test_no_result_traces_to_a_missing_commit` now asserts that every recorded
  `git_sha` resolves with `git cat-file -e`. The staleness guard compares the
  recorded sha against HEAD but never asked whether it was real, which is how
  results generated at `8095dbec`, a commit that has never existed in this
  repository, passed every check for a month.
- Phase U3 (**behaviour change, deliberate**): the simulator sizes its
  scratchpad strictly from the declared `scratchpadBytes`. It used to grow the
  scratchpad to cover every `resultAddr` it found in the instruction stream,
  which swallowed the U3 hardening whole: an out of range result address was
  quietly accommodated rather than refused, so the new result bounds check could
  only ever fire on a negative or misaligned address. The growing arithmetic
  also ran before any validation, computing `resultAddr + elements * 4` on
  hostile input at the exact entry point the check exists to defend. A program
  that writes outside its declared budget is now refused. Nothing in the
  compiler regressed: `-npu-allocate-scratchpad` records its high water mark in
  `npuisa.scratchpad_bytes` and the encoder emits that as `scratchpadBytes`, so
  every compiled program declares exactly what it uses. Only hand built
  `Program` values that declared no scratchpad at all relied on the expansion.
- Phase U3: every hand built `Program` in `unittests/Simulator/SimulatorTest.cpp`
  now sets `scratchpadBytes` explicitly, at the smallest size that covers its
  writes (32, 48, 80, 64, 20, and 32 bytes), with the arithmetic in a comment.
  They set `dramBytes` but never `scratchpadBytes`, so they ran only because the
  simulator grew the scratchpad to fit whatever the instructions referenced. A
  tight explicit size is what makes a future off by one visible.
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

- Phase U3: the `.nbin` format was documented as "a fixed header followed by
  tagged records, little endian" in `docs/ISA_MANUAL.md`, in the
  `include/NPU/Encoding/Program.h` header comment, and in a
  `docs/DESIGN_DECISIONS.md` heading. Both halves were false. There are no tags:
  every field sits at a position determined by the fields before it, so nothing
  can skip an unrecognised field, which is exactly why the version field exists.
  And the helpers copy the object representation in and out of the stream, so the
  encoding is host byte order, not a fixed little endian; a `.nbin` is not
  portable across byte orders. All three now say what is true, and the code
  comment claiming little endian helpers says so too. No byte swapping was added:
  that is a format change. `docs/ASSESSMENT.md` 13.4 item 7 had found two of the
  three places.
- Phase U3: `Fuzz.DecodeUnvalidatedNeverCrashesEither` accumulated the fields it
  touches into a signed `int64_t`, and the corpus holds a file declaring
  `INT64_MAX` bytes of scratchpad, which `decodeUnvalidated` returns by design.
  Adding to that was undefined behaviour in the test itself, reported by UBSan
  the first time the whole corpus was run under it. The accumulator is unsigned
  now. Constant regions are touched by shape length rather than `byteSize()`,
  which multiplies extents out and is only ever called by the encoder on shapes
  from the MLIR type system, never on decoded input.
- Phase U3: `encodeFunction`'s `.Default` case emitted `cannot encode unexpected
  op`, skipped the op, and returned the program anyway, so `npu-translate`
  printed an error, wrote the `.nbin`, and exited 0. Verified before the fix on
  an unlowered `npu.relu`: exit code 0 and a 150 byte output file. The file was
  the program with that work silently deleted from it, which is worse than no
  file because it looks like a successful compile. It now returns failure, after
  the loop rather than inside it so one run names every op it cannot encode.
  `npu-translate` already checked the result before opening the output stream,
  so it now exits nonzero and leaves no file.
- Phase U3: the decoder's `getCount()` capped element counts at 2^28 and stopped
  there, which bounds the number but not the work. At the cap a single shape
  vector is 2 GiB, and `getVec()` sized that vector from the count before
  reading a byte behind it, so a thirty byte file naming a count near the cap
  was a decompression bomb. Measured on the new probes: 2.0 GiB peak RSS and 43
  seconds to refuse 36 tiny files. `getCount()` now also takes the smallest
  space one element can occupy and refuses a count the remaining bytes cannot
  back, which is the truncation it always was. The same 36 files now cost 6 MB
  and no measurable time. A well formed file always carries those bytes, so
  nothing the format permits is rejected; the 1000 iteration round trip property
  test covers that.
- Phase U3: `Program::validate()` checked operand reads for membership only. Its
  written before read walk records an element count per result address, then
  asked whether the address had been written and never whether enough had been.
  A `DMA_STORE` reading 100 elements from a 4 element buffer passed validation
  and then trapped in the simulator, which contradicted the contract comment on
  `Program.h` saying validate checks every invariant the simulator relies on. An
  over read that stays inside the scratchpad was worse: it validated, ran, and
  silently folded stale memory into the result. A new `operand-extent` rule
  requires the recorded count to cover what the consumer reads, naming the
  instruction, the operand, elements needed, and elements written. `CONV2D` and
  `MATMUL` get the weaker rule of requiring a non zero recorded count, because
  their extents follow from tensor shapes and the walk tracks counts.
- Phase U3: the simulator's trap path called
  `assert(false && "simulator memory access out of bounds")`, so an assert
  enabled build aborted the process on exactly the input the bounds check was
  added to handle, while a release build returned a diagnostic. Graceful refusal
  is now the behaviour in every build mode: the first refusal is recorded in
  `SimResult.error`, `nullptr` is returned, and the caller skips the access. The
  comment above the accessor also claimed a failure "aborts in a debug build and
  clamps to a scratch cell in a release build"; neither half was ever true of
  the code below it, and it now describes what the code does.
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
