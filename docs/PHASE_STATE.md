<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Phase state

*Diataxis type: reference.*

Ground rule 17: this file is updated at the end of **every** session, including
a session that achieved nothing, and it carries four things: the current phase,
the status of its gate, the open questions, and the exact next command. This
build spans dozens of sessions, and reconstructing where it stood from `git log`
costs more than writing these lines did.

**Last updated:** 2026-08-31.

## Current phase

**P7, the simulator and the reference interpreter.** Branch
`phase/p7-simulator`, cut from `main` at `7b9d18c`, which is the P6 merge plus a
README edit. Ten commits, not pushed. The tenth is the one that carries this
table, so it is the branch tip and is named by subject rather than by a sha it
cannot know.

| Commit | Subject |
|---|---|
| `1939b58` | `style(readme): trim the trailing space the web editor left on the title` |
| `e454d6d` | `feat(simulator): one home for the cost model, and the Python mirror a test holds to it` |
| `018f8c2` | `feat(simulator): the machine's two memories, bounds checked in every build mode` |
| `078b78d` | `feat(simulator): the fp32 kernels, the two timelines, and npu-sim` |
| `b1df707` | `test(simulator): NPUSimulatorTests, a semantics test per opcode and the trap pair` |
| `b931fac` | `test(simulator): npu-sim over a program the whole pipeline produced` |
| `319c52d` | `feat(python): refexec, the independent numpy reference interpreter` |
| `015e437` | `ci: switch on NPUSimulatorTests, in the default build and under the sanitizers` |
| `a1b0920` | `fix(test): the differential oracle's random inputs were all negative` |
| tip | `docs: record the P7 defects and hand off the phase` |

**The commit order is the roadmap's, and P6's handoff is why it is written
down.** The cost model comes first because it is the one home Section 5.5 asks
for and because everything after it charges against it. The memories come
second, because a kernel cannot be written before there is something safe to
address. The kernels and the executor come third. Each test follows the thing it
tests, and the reference interpreter comes last of the code because it is the
oracle rather than the subject.

**Every commit was checked out and built on its own**, not merely the tip. A
worktree at `/tmp/p7check`, since removed, walked the ten in order and ran
`ninja -C build -j6` from a fresh configure at each. All built clean. A commit
sequence nobody has bisected is a sequence that only reads as one.

**This phase was resumed from an uncommitted tree**, and the log's honesty
register says to write that down rather than present the result as if it arrived
in order. The session that wrote the simulator did not survive to commit it.
What this one found was `7b9d18c` with zero commits and about five thousand
lines of untracked work beside it, passing its own tests. Passing tests written
by the same session prove internal consistency and not conformance, so the work
above is that tree audited against the specification section by section, plus
the four defects the audit found. Which lines came from which session is not
recoverable and is not worth reconstructing. What matters is that nothing here
is committed on the strength of having been found in the working tree.

## Gate status

The P7 gate is the roadmap entry's, plus Section 9.3's contract with the format,
plus Section 19.0's activation. Every item is **met locally**. Item by item,
with the proof.

### The roadmap's P7 gate

| Gate item | Proof |
|---|---|
| A semantics test per opcode against hand computed values | 16 of 16 opcodes. `NOP` and `HALT` in `Control.NopAdvancesAndHaltStops`; `DMA_LOAD` and `DMA_STORE` in `Dma.DmaRoundTrip`; the other twelve each by name in `SimulatorTest.cpp`, `QUANT` and `DEQUANT` by the refusal they give. Not one expected value was produced by running the kernel: each is written out above the test that asserts it, from `docs/ISA_MANUAL.md` and the ODS description, in arithmetic short enough for a reader to redo |
| Every case in Section 10.1's **P7 list** | All of them, and the index at the top of `SimulatorTest.cpp` maps the list to the tests one line each. `POOL_MAX`; the DMA pair and `Dma.SpillRoundTrip`; `GroupedConvolution` at `group == 2`; `DepthwiseConvolution` at `group == C`; `DilatedConvolution` at dilation 2; `AsymmetricPadding` at pads `[1, 0, 0, 1]`; batch 2 and batch 4 on convolution, on **both** pooling kernels and on **both** elementwise opcodes; `TransposeIdentity` and `TransposeNchwToNhwc`; `CONCAT` on the channel axis, on the last axis, with three operands and at batch 4; `AllPaddingWindow`, which must not divide by zero; and `Dma.StridedLoad`, which exercises the stride term rather than merely defining it |
| And no integer case | None. The two quantization opcodes are covered by the refusal they give, which names Phase P14, and `Quantization.QuantRefusesByNameUntilPhaseP14` and its `DEQUANT` mirror assert it. Nothing here asserts what an integer kernel computes |
| An out of bounds result address traps gracefully in an assertions build and an NDEBUG build, all four runs shown | The four runs are in their own table below. The NDEBUG build is `build-ndebug`, configured `-DNPU_FORCE_NDEBUG=ON`, and that option exists because `-DCMAKE_BUILD_TYPE=Release` alone does not produce one against this LLVM. That is D-0028, and the first attempt at this gate item passed in an assertions build |
| The two port overlap test | `Timelines.IndependentStreamsOverlapCompletely`. Twenty transfers and five computes with nothing to say to each other: the total equals the DMA timeline exactly, so the whole compute timeline is hidden underneath it |
| The dependency serialization test | `Timelines.DependencySerializesAndOverlapReadsZero`. A chain in which every instruction waits for the one before it, ending on the compute port so the `HALT` cannot hide underneath a transfer that is still running. The total equals the sum of the two timelines |
| `overlap_fraction` reads 0 fully serialized and approaches 1 fully overlapped, asserted | Both endpoints, twice. `CostModel.OverlapFractionReachesItsEndpoints` asserts the formula at 0, 1 and 0.5, and asserts it is **not** the naive form, which tops out near 0.5 and which somebody would otherwise simplify it back to. The two timeline tests assert the same endpoints over real programs |
| The single port mode reproduces the sum | `Timelines.SinglePortReproducesTheSum`. Under the flag the total is the sum of the two ports and the fraction is 0; the same program on two ports is strictly faster; and both runs place the same charges on the same ports, which is what makes the flag a reproducibility switch rather than a second model |
| The reference interpreter and the simulator agree on every operation on randomized inputs | 24 cases covering every executable operation of the dialect, `test_every_case_agrees`. `test_the_export_covers_every_operation_refexec_can_run` asserts the case set covers all ten opcode backed operations, because the failure mode of a table driven suite is that it stops growing while the cases it has keep passing |
| A deliberately perturbed kernel makes that comparison fail, shown | `POOL_MAX` initialised to zero rather than negative infinity: `max_pool2d` 5 of 54 elements mismatched, `max_pool2d_padded` 1 of 50, `test_every_case_agrees` red. Recipe and restore below. A standing cheap version, `test_the_comparison_is_not_vacuous`, perturbs the reference by one part in a thousand on every run, so a tolerance quietly widened until everything passed goes red in CI rather than in a recipe |

### Section 9.3, the simulator's contract with the format

| Gate item | Proof |
|---|---|
| The scratchpad is sized strictly from `scratchpadBytes`, never grown | `Simulator::Simulator` passes `program.scratchpadBytes` to `Machine` and nothing inspects the instruction stream to size anything. `Trap.AnOutOfRangeResultAddressTrapsGracefully` asserts `scratchpadBytes() == 32` after the fault, so the machine that refused the write is still exactly the size the file declared |
| Every hand built test program sets an explicit tight value with the arithmetic in a comment | Every one, and it is mechanical rather than a habit: `Builder::finish` takes the declared number and fails the test when the buffers do not add up to it, so a comment that drifted from the code is a red test |
| A declared size this host cannot allocate is refused rather than clamped | `Trap.AnOversizedMemoryIsRefusedRatherThanClamped`, at 2^50 bytes. A clamp would be a different machine from the one the file describes and every bounds check afterwards would be checking against the wrong number |
| Bounds checked accessors in **every** build mode, no `assert(false)` on the trap path | `grep -n 'assert(' include/NPU/Simulator/*.h lib/Simulator/*.cpp` finds one `static_assert` on the dispatch table's size and one occurrence inside a comment. The checks are real branches, in `Machine::inRange` and `Machine::elementAddress`, compiled identically in both build directories |
| Proven in both | The four runs below |
| An out of range access records the first trap, returns null, the caller skips the access | `Trap.TheFirstTrapIsTheOneReported` asserts the message names element zero's address and **not** element one's, over a relu that goes out of range on all four of its elements |
| `npu-sim` compares the input file size against the declared region | `test/Simulator/npu-sim.mlir`, SIZE prefix: `input region 0 is 128 bytes and the file supplied is 64` |
| One `--input` per declared region, refused with a message naming both numbers | Same file, COUNT prefix: `declares 1 input regions and 0 --input arguments were given` |
| It writes **all** outputs, not just the first | Same file: two output regions, both asked for, both measured with `wc -c`, both 128 bytes. The claim is about the second one, so the second one is the one measured |
| `validate()` is called again before execution, by contract | `Simulator::run` calls `program.validate()` and returns the named check on failure. `Trap.TheValidatorRefusesAnOutOfRangeResultAddress` asserts the message names `result-in-range` **and** that `stats.instructions` is 0, so the program did not reach a kernel |

### Section 19.0, the activation table

| Gate item | Proof |
|---|---|
| `NPUSimulatorTests` CI step on | Guard kept, else branch turned into a failure, which is the shape `NPUInterfaceTests` took at P2 and `NPUEncodingTests` at P6 |
| The same binary under the sanitizers | Added to the sanitizers job's existing GoogleTest step rather than given a step of its own. A new step is a new activation to prove; a second binary inside a step that is already on and already proven red is a strengthening of a net that exists |
| Every later phase guard still prints its off line | pytest slow cells P10, check-reachability full P8, nightly full matrix P10, mutation P15, flake P15 |

### The five things P6 left on this phase's desk

| Obligation | Proof |
|---|---|
| The generated dispatch skeleton expanded, so a new opcode with no kernel is a build error | Shown, not asserted. A sixteenth opcode appended to `NPUISADescription.td` and nothing else fails the build in **four** places, listed below with their messages. Restored, and the tree is clean afterwards |
| `validate()` called again before execution | Above, with the instruction count that proves the program never started |
| The scratchpad sized strictly, tight values with the arithmetic in a comment | Above, and `Builder::finish` makes it mechanical rather than conventional |
| A kernel indexes its operands through their strides, so ADR 0005's broadcast needs no special case | `Elementwise.ChannelBroadcastArrivesAsAStrideZeroOperand` reads a three element buffer as a rank 4 view with strides `[0, 1, 0, 0]`. There is no broadcast branch in `Kernels.cpp` to find; grepping for one returns the comment saying there is none and must never be one. `Shape.TransposeNchwToNhwc` is the other half, NCHW extents with permuted strides |
| The namespace is `nbin`, not `npu` | It never came up, and that is the finding. Nothing under `include/NPU/Simulator`, `lib/Simulator`, `tools/npu-sim` or `unittests/Simulator` includes an MLIR header at all, so no translation unit in this phase holds both names. `docs/ARCHITECTURE.md`'s P7 extension records the prediction and what actually happened, because a prediction quietly dropped is a prediction nobody learns from |

### Verification output

Every command below was run on this branch at `a1b0920`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 19 discovered, 19 passed, 0 failed. Eighteen at P6, plus this phase's one |
| `build/bin/NPUSimulatorTests` | 55 tests from 14 suites, 54 passed, 1 skipped. The skip is the differential export, which needs `NPU_DIFFERENTIAL_OUT` |
| `build-ndebug/bin/NPUSimulatorTests` | 55 tests, 54 passed, 1 skipped. Configured `-DCMAKE_BUILD_TYPE=Release -DNPU_FORCE_NDEBUG=ON` |
| `build-fuzz/bin/NPUSimulatorTests` under ASan and UBSan | 55 tests, 54 passed, 1 skipped, exit 0. clang, `-fsanitize=address,undefined`, `UBSAN_OPTIONS=halt_on_error=1` |
| `build/bin/NPUEncodingTests` | 77 tests, 76 passed, 1 skipped, untouched by this phase |
| `build-fuzz/bin/NPUEncodingTests` under ASan and UBSan | 75 passed, 2 skipped, exit 0, unchanged from P6 |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed, untouched |
| `build/bin/NPUAllocatorTests` | 29 tests, 29 passed, untouched |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed, untouched |
| `python -m pytest test/Python -q` | 180 passed, 7 deselected, exit 0. 142 at P6, plus this phase's 38 |
| `mypy` | no issues found in 13 source files |
| `black --check .` | 26 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 211 of 211 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, `layers checked: import, lowering, encoding, simulation` |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `git status --short` | empty |

**Five gtest binaries exist now, not four.** `NPUSimulatorTests` is this
phase's, and it is the last one Section 19.0's table has to switch on.

**`check-reachability` checks four layers now, not three.** The simulation layer
became decidable the moment `lib/Simulator/Simulator.cpp` existed, and it passes.
It is answered by a substring search over that file's operation table, which is
weaker than the encoding layer's; the file says so and making it mechanical is on
P8's desk.

### The four runs Section 9.3 asks for

| Build | Test | Result |
|---|---|---|
| `build`, assertions | `Trap.AnOutOfRangeResultAddressTrapsGracefully` | 1 test, passed |
| `build`, assertions | `Trap.AnOutOfRangeOperandAddressTrapsGracefully` | 1 test, passed |
| `build-ndebug`, NDEBUG | `Trap.AnOutOfRangeResultAddressTrapsGracefully` | 1 test, passed |
| `build-ndebug`, NDEBUG | `Trap.AnOutOfRangeOperandAddressTrapsGracefully` | 1 test, passed |

The whole `Trap` suite is nine tests and passes in both directories.

**The NDEBUG build is proven to be one rather than assumed to be**, and that is
the part worth keeping. A file whose first lines are `#ifndef NDEBUG` and
`#error`, compiled with the exact flags `compile_commands.json` records for
`lib/Simulator/Memory.cpp`, exits 0 in `build-ndebug` and exits 1 in `build`.
The same probe refuses `_GLIBCXX_ASSERTIONS` and `_DEBUG` as well, because it was
the libstdc++ hardening rather than `NDEBUG` itself that turned D-0026 into an
abort, and a build that cleared one and kept the other would be a third thing
neither name describes.

## Activation proofs

**One step activates at P7**, `NPUSimulatorTests`, and it was broken twice
deliberately, shown red, and restored.

Both rehearsals ran under **the exact CI invocation**, inside
`set -euo pipefail`, with no `--gtest_filter` naming the broken test. P3, P5 and
P6 each recorded a proof that proved nothing, so the distinction is not
theoretical: filtering proves that a broken test fails, which was never in doubt,
where what needs proving is that a failure inside the binary reaches the step's
exit code.

**Both were rehearsed locally first and then run against CI**, on the scratch
branch `phase/p7-activation-rehearsal`, since deleted. The branch's own first
run, with the step active and everything green, is
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33362572992>.
The product fault went red at the `NPUSimulatorTests` step, at the sanitizers
job's GoogleTest step, and at coverage besides:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363102927>.
The test side fault went red at exactly the same three steps and nowhere else,
which is the isolation claim in CI's terms: every step that executes the binary,
and nothing that does not:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363280962>.
The restore, a tree byte identical to the phase branch tip, returned green:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363608121>.

**CI shows less of the product fault's net than the local rehearsal did**, and
the reason is worth a line: the job stops at its first red step, and the
`NPUSimulatorTests` step precedes pytest, so the differential's two extra
catches are in the local record only. The engineering log carries the detail,
along with a process finding about the rehearsal branch's name.

### 1. Product side, which measures the net

**The fault.** `POOL_MAX` starts its accumulator at zero rather than at negative
infinity, which is the classic form of this bug:

```bash
sed -i 's/IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;/0.0f;/' \
    lib/Simulator/Kernels.cpp
```

**The prediction, written before the run.** `NPUSimulatorTests` red at
`Pooling.AllPaddingWindow` and nowhere else, because every other hand computed
pooling case uses positive inputs and `max(0, positive)` is the positive.
`check-npu` still 19 of 19, because the lit test asserts the tool's contract and
not its numbers. pytest red in the differential, whose inputs are drawn from
`[-1, 1)` and therefore contain windows whose maximum is negative.

**The result.** Exactly that. Step exit 1, one test red,
`Pooling.AllPaddingWindow`; `check-npu` 19 of 19; pytest 1 failed at
`test_every_case_agrees`, with `max_pool2d` 5 of 54 elements mismatched and
`max_pool2d_padded` 1 of 50. Restoring returns the step to exit 0 with 54
passing, `check-npu` to 19 of 19, and pytest to 180 passed.

**What the run measured is the shape of the net rather than its depth.** One
hand written test caught this, and the differential caught two cases the hand
written tests structurally cannot reach, because the hand computed pooling
inputs are the small positive integers a person picks when writing an expected
value by hand and the oracle's are not. That is the division of labour Section
17.3a describes, seen working. `check-npu` saw nothing, which is correct rather
than a gap: `npu-sim.mlir` exists to assert the tool's contract with the format,
and asserting numerics there would duplicate the unit suite in a worse language.

**This rehearsal is also how D-0029 was found**, and it is recorded here rather
than tidied away. The first run reported **54 of 54** elements mismatched on
`max_pool2d`. That figure is wrong for the fault: with inputs spread over
`[-1, 1)` and windows of four, about one window in sixteen has an all negative
maximum, so three or four was the expectation. The generator was shifting its
state right by 33 bits rather than 32, so every value it had ever produced was
negative; the `relu` differential case had been comparing zeros against zeros and
the two pooling cases had never seen a positive maximum. A rehearsal is supposed
to confirm a prediction. The useful part of one is the number that does not
match.

### 2. Test side, which isolates the step

```bash
sed -i 's/EXPECT_EQ(kArrayDim, 16);/EXPECT_EQ(kArrayDim, 17);/' \
    unittests/Simulator/CostModelTest.cpp
```

**The prediction.** `FrozenConstants.TheCostModelsNumbers` red and nothing else;
step exit 1; `check-npu` still 19 of 19, because lit never compiles the unit
tests; pytest green, because the header the Python mirror parses did not move and
the mirror compares against that header rather than against this expectation.

**The result.** Exactly that. Step exit 1, one test red,
`FrozenConstants.TheCostModelsNumbers`; `check-npu` 19 of 19; pytest 180 passed.
Restoring returns the step to exit 0.

### Adding an opcode, which Section 9.4 asks for separately

Append a sixteenth opcode to `include/NPU/Encoding/NPUISADescription.td` and
build. It fails in four places and every one of them names the opcode:

```
Kernels.cpp:917: error: 'kernelSCRATCHOP' was not declared in this scope
Kernels.cpp:930: error: static assertion failed: the kernel table and the ISA
                 description disagree about how many opcodes there are
Kernels.cpp:948: error: enumeration value 'SCRATCHOP' not handled in switch
Validation.cpp:526: error: enumeration value 'SCRATCHOP' not handled in switch
```

Two mechanisms, both out of the one description. The dispatch table expands the
generated `NPUISADispatch.def`, so a row marked as computation with no kernel
written is a missing identifier; and the hand written switch that assigns each
opcode a port carries no `default`, so `-Werror=switch` catches it as well. P6
generated the skeleton and could not demonstrate the claim, because there was no
simulator to demonstrate it in. This is that demonstration.

## Open questions

Six. None blocks the gate.

**The NDEBUG build is proven locally and not in CI.** Section 9.3 asks for the
accessors to hold in every build mode and the gate asks for four runs; all four
are above, but `build-ndebug` is a developer's second directory rather than a
job. The sanitizers job is `RelWithDebInfo` against a container LLVM whose
assertion setting this session did not read, so it cannot be claimed as the
NDEBUG half without checking it, and the `ci.yml` comment does not claim it. **A
third CI build is a decision for the orchestrator**, not something to add
unilaterally at the end of a phase: it is a new activation to prove and it costs
a job.

**Write after write and write after read are not ordered on the timeline.** An
instruction starts at the later of its port becoming free and its **operands**
becoming ready, which is Section 5.5's wording implemented literally. Two
instructions writing overlapping spans on different ports are therefore not
ordered against each other. No program this compiler emits has that shape, since
the allocator gives a live value one span, and inventing an ordering rule the
specification does not state would be inventing a machine. Flagged because a
reader comparing the executor against a real machine will notice it.

**`readyAt` is linear in the writes so far, so the executor is quadratic in the
instruction count.** Invisible at this phase's sizes, where the largest program
is a few dozen instructions, and not invisible at P8's, where a model is
thousands. An interval structure is the fix and P8 is the phase that will feel
it. Recorded now so that a slow end to end run is diagnosed rather than
investigated.

**`npu-sim` requires one `--output` per declared region, which is stricter than
Section 9.3 asks.** The specification requires the tool to write all outputs and
says nothing about refusing a caller who wanted fewer files. The refusal is here
because "wrote one of your two outputs" is the failure the requirement exists to
prevent, and a silent partial write is how it would happen. Recorded as a
deliberate deviation rather than left for somebody to find in the lit test.

**A trapping instruction is still charged its cycles.** `transfer` computes its
DMA charge before it checks the operand's rank, and the compute kernels charge
before they walk. The run stops at the end of that instruction and the statistics
go out with `SimResult::error` set, so nothing publishes them; a caller that read
`stats` without checking `ok()` would read a number including work the machine
refused to do. `SimResult::ok()` is the guard and every caller in the tree uses
it.

**The simulation layer of `check-reachability.py` is a substring search over
`lib/Simulator/Simulator.cpp`.** The encoding layer became mechanical at P6 and
this one did not. The file carries the table the checker reads, as a table rather
than a scattering of mentions, so a human gets the same answer the checker does;
but a mnemonic appearing inside an unrelated word would satisfy it. Making it
mechanical is on P8's desk and `docs/ARCHITECTURE.md` says so.

## Next command

Open the merge pull request for `phase/p7-simulator`.

The branch is pushed and its first run is green, with `NPUSimulatorTests`
running for the first time in the build and test job and again inside the
sanitizers job:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33362572992>.
Both activation proofs have run against CI, red and restored, with the URLs in
the section above.

Three things are worth watching specifically.

Three predictions from the pre push draft of this section, and what the first
run made of them.

**The sanitizers job linked its third target without incident.** The simulator
library carries no MLIR dependency, so the extra target cost seconds.

**OpenMP split the jobs, which the prediction did not foresee.** The build and
test job's configure printed `OpenMP: not found`, and the determinism test said
in its own output that both of its runs were single threaded, which is the
weaker assertion and not a failure. The coverage job's configure found OpenMP
4.5, so the thread count determinism assertion did run at full strength in CI,
one job over from where it was expected. Why the two configures in one image
disagree is not established and is left with P8.

**`check-npu` reported 19, `pytest` 180, and `check-reachability
--skip-models` printed its four layers**, each confirmed in the run log.

## Next phase

**P8, the walking skeleton and the safety net.** `npu-compile` with `-O0` and
staged `--emit`, then the first end to end run: ONNX to simulated output,
validated against onnxruntime across the full input class matrix at `-O0`. Then
the net: `scripts/regression-baseline.sh` with `--check`, the `-O0` golden
tensors, `scripts/check-reachability.py` in full, `scripts/coverage.sh`, and the
proof of failure gate of Section 19.1. Also the metamorphic relations and the
dead subgraph injection of Section 17.3a, and the per model tight budget
constants of Section 15.

**It is the first legitimate stop and ship point.**

Six things P7 leaves on P8's desk.

1. **The reference interpreter is the oracle and it is ready.** `refexec.execute`
   dispatches on the `npu` mnemonic with every operation listed rather than swept
   into a lookup with a default, so an operation added to the dialect and not to
   that file raises by name rather than going unchecked. The end to end
   comparison at `-O0` is P8's, and the harness it scales up from is the twenty
   four case one in `test_refexec_differential.py`.
2. **`stats.instructions` is the only instruction count in this project** and it
   exists now. Section 10.2 is explicit that a regex over an IR dump is not one,
   and the benchmark harness must raise if the field is missing rather than fall
   back to counting lines.
3. **The executor is quadratic in the instruction count.** Fine at a few dozen,
   not fine at a few thousand. See the open question above.
4. **`batch_norm` has no differential case and should get one at P8.** It has no
   opcode, it decomposes at lowering into a multiply and an add, and comparing it
   against the simulator needs the end to end pipeline. `test_refexec.py` pins
   the decomposition rule in the meantime and
   `test_the_export_covers_every_operation_refexec_can_run` names it as the one
   executable operation absent by design, so the gap is asserted rather than
   assumed.
5. **The simulation layer of the reachability check is the weakest of the four.**
   Making it mechanical, the way P6 made the encoding layer, is P8's.
6. **A third CI build for NDEBUG is an open decision**, not an oversight. See the
   open questions.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both, and repeated at every phase because a fact that stops being
repeated is a fact somebody eventually does not know.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, no agent, not once. It may be read, and only through a command
that cannot write. **Only the owner may retire it.**

The reason it exists on top of git history is that the two protect against
different failures. History protects against a bad commit. A second directory
protects against everything else, because if this rebuild goes wrong at any
point, deleting `~/npu-mlir-v2` returns the machine exactly to its pre build
state with no reasoning about reflogs required. That guarantee holds only while
the frozen copy is untouched.
