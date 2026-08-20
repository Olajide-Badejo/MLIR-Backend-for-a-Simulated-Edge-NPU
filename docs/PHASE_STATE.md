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

**Last updated:** 2026-08-20.

## Current phase

**P6, the binary format and the generated ISA.** Branch
`phase/p6-binary-format`, cut from `main` at `f6baff2`, which is the P5 merge.
Eight commits, not pushed. The eighth is the one that carries this table, so it
is the branch tip and is named by subject rather than by a sha it cannot know.

| Commit | Subject |
|---|---|
| `3d03377` | `feat(encoding): describe the instruction set once and generate the layers that drift` |
| `42562b7` | `feat(encoding): the .nbin binary format, its encoder, and its validator` |
| `8155b87` | `feat(tools): npu-translate writes a .nbin and npu-objdump reads one back` |
| `a70d757` | `test(encoding): NPUEncodingTests, with the property test and the seed corpus` |
| `2302e3b` | `test(encoding): the coverage guided decode fuzzer, and the overflow it found` |
| `210a263` | `fix(fuzz): commit the seed corpus, which .gitignore was swallowing` |
| `1e2f767` | `ci: switch on the four P6 gates, and fix what rehearsing them found` |
| tip | `docs: record the P6 defects and hand off the phase` |

The branch point again, since every handoff since P3 has had something to say
about it. Local `main` was stale when this session began, exactly as the last
three handoffs predicted. The first command of the session was
`git fetch origin && git branch -f main origin/main`, which moved local `main`
to `f6baff2`, and the branch was cut from that.

**The commit order carries the roadmap's own instruction.** `3d03377` is the ISA
description and its generator; `42562b7` is the encoder, written against the
generated header. The roadmap says generating the opcode enum after hand writing
the encoder means writing the encoder twice, and doing it in that order is what
that sentence buys.

## Gate status

The P6 gate is the roadmap entry's, plus Section 9.4's generation claim and
Section 19.0's four activations. Every item is **met locally**. Item by item,
with the proof.

| Gate item | Proof |
|---|---|
| 300 or more malformed inputs | 734, printed by the test itself. Truncation at every count field boundary and on a sweep; the three cap boundary values for every count field the format has; out of range opcodes, element types, activations and memory spaces; shapes overflowing the product cap from both sides; malformed permutations and concatenation axes; the whole quantization field set; every debug section rule; bit flips across the header; and a hundred single byte changes at random offsets from a fixed seed |
| None crash | `MalformedInput.TheUnvalidatedPathAndTheDisassemblerSurviveTheCorpus` walks all 734 through `decodeUnvalidated` and then the disassembler, which is the path with no validation in front of it, and the sanitizer build runs the same binary |
| All rejected with a specific named check | `MalformedInput.EveryRejectionNamesACheck` asserts the message begins with the stable name. The corpus reaches **33 of 33** check names, printed |
| None allocating more than a few megabytes, measured through an allocator hook | 25237 allocations across the corpus, worst single case **78064 bytes** against a budget of four megabytes, measured through a replaced global `operator new` counting total bytes requested. The test asserts the hook was called at all, because a replacement that lost the link would report zero for every case and pass |
| ASan and UBSan clean over the corpus | `build-fuzz`, clang 21.1.8, `-fsanitize=address,undefined`, `UBSAN_OPTIONS=halt_on_error=1`: 75 passing, exit 0. It was **not** clean on the first run, which is D-0021 |
| The libFuzzer target builds and runs clean for its budgeted time from the seed corpus | 1964713 runs in 61 seconds, no crashes, no artifacts. An earlier run from a grown corpus found D-0022 |
| The property test passes 1000 iterations and re-encoding is byte identical, with `requantMultiplier` and `requantShift` round tripping | 1000 programs at seed `0x6e62696e5031365f`, 8741 instructions, largest file 5606 bytes. Structural equality **and** byte identical re-encoding are both asserted, and the generated programs are valid rather than merely well formed, so the round trip goes through `Program::decode`, which validates |
| Every opcode including the quantization ones, every element type, sometimes a debug section | 16 of 16 opcodes, 3 of 3 element types, 403 of 1000 programs with a debug section, all printed. The test asserts the debug section is neither always nor never |
| Adding a deliberate opcode to the description and nothing else fails the build in every layer that has no case for it | `-Werror=switch` over the generated `Opcode` enum with no `default`, in `Validation.cpp`'s semantic dispatch and `PropertyTest.cpp`'s builder. Recipe below |
| The ISA staleness gate perturbed, shown red, restored | Twice, two different faults, both exit 1, both restored to exit 0. The first attempt was **green**, which is recorded below rather than tidied away |
| `docs/ISA_MANUAL.md` documents the layout field for field, the byte order policy, `kMaxCount`, every validation rule by check name taken from the source, and the version policy | All five. The opcode table and the check table are generated from `NPUISADescription.td` and spliced between markers; everything else is hand written prose |
| `npu-translate` fails and writes no output file on an operation it cannot encode, and the output file is not created before the encode result is known | `test/Encoding/diagnostics.mlir`, seven refusals, each asserting **twice**: that the message names the refusal, and that `not test -e %t.nbin` afterwards |
| Multiple functions in a module diagnosed, not truncated | `Inputs/two-functions.mlir`, with a note attached per function |
| `--strip-debug` | `objdump.mlir`'s STRIPPED prefix: zero debug entries, and the node names absent |
| `decodeUnvalidated()` with a warning prefix | `objdump.mlir`'s WARN prefix, over a file corrupted by `dd` at a named byte offset |
| The generated ISA description produces the opcode enum, `kMaxOpcode`, the arity and field presence rules, the disassembler format strings, the simulator dispatch skeleton, and the reachability checker's opcode list | Six artifacts from one `.td`. `check-reachability.py` now reports `layers checked: import, lowering, encoding` |
| `NPUEncodingTests` CI step on | Guard kept, else branch turned into a failure, which is the shape `NPUInterfaceTests` took at P2 |
| ISA staleness CI step on | Same shape, and it refuses to run against an untracked artifact |
| The `sanitizers` job real | A clang build with `-fsanitize=address,undefined` in its own directory, the whole of `MalformedInputTest`, plus Section 3.3's budgeted minute of the coverage guided target |
| `nightly.yml` fuzz job on | Thirty minutes, seeded from the committed corpus and all 734 exported cases |
| Every later phase guard still prints its off line | `NPUSimulatorTests` P7, pytest slow cells P10, check-reachability full P8, nightly full matrix P10, mutation P15, flake P15 |
| The spill slots have DRAM addresses, per P5's handoff | `Program::spillSlots`, placed after the constants in the DRAM map, found by the `npuisa.spill_slot` predicate rather than by an analysis |
| The in and out argument convention decided and documented | `npuisa.arg` on every argument, written by the lowering, required by the encoder. The `docs/ARCHITECTURE.md` P6 extension carries the decision and the three alternatives |
| `docs/PASSES.md` updated in the same commit as the pass change | `42562b7` carries both |

### Verification output

Every command below was run on this branch at `1e2f767`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 18 discovered, 18 passed, 0 failed. Fifteen at P5, plus this phase's three |
| `build/bin/NPUEncodingTests` | 77 tests from 12 suites, 76 passed, 1 skipped. The skip is the corpus export, which needs `NPU_CORPUS_OUT` |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed, untouched by this phase |
| `build/bin/NPUAllocatorTests` | 29 tests, 29 passed, untouched |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed, untouched |
| `build-fuzz/bin/NPUEncodingTests` under ASan and UBSan | 75 passed, 2 skipped, exit 0 |
| `build-fuzz/bin/nbin_decode_fuzzer -max_total_time=60` | 1964713 runs in 61 seconds, no crashes |
| `python -m pytest test/Python -q` | 142 passed, 7 deselected, exit 0, unchanged from P5 |
| `mypy` | no issues found in 11 source files |
| `black --check .` | 21 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 186 of 186 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, `layers checked: import, lowering, encoding` |
| `bash scripts/check-isa-staleness.sh build` | up to date |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `git status --short` | empty |

**Four gtest binaries exist now, not three.** `NPUEncodingTests` is this
phase's. `NPUSimulatorTests` is P7's, per the activation table.

## Activation proofs

**Four steps activate at P6**, and each was broken once deliberately, shown red,
and restored. The recipes are below so the next person can repeat them rather
than take this table's word for it.

Every rehearsal was run under **the exact CI invocation**, inside
`set -euo pipefail`, with no `--gtest_filter` naming the broken test. P3 and P5
each recorded a proof that proved nothing the first time, and P6 recorded two
more, so the distinction is not theoretical: filtering proves that a broken test
fails, which was never in doubt, where what needs proving is that a failure
inside the binary reaches the step's exit code.

### 1. `NPUEncodingTests`

**Product side fault, which measures the net.** Swap the two adjacent `i32`
writes in `Program::encode`, so that `requantShift` is written before
`requantMultiplier`. Files then decode with the pair exchanged, and every
program carrying the identity pair comes back with a multiplier of zero.

Result: 7 tests red across `RoundTrip`, `EncodingProperty` and `MalformedInput`,
step exit 1. `check-npu` goes to 17 of 18 at `Encoding/objdump.mlir`, **but only
since D-0024 was fixed**; before that it reported 18 of 18 against an encoder
producing files that failed their own validator.

**Test side fault, which isolates the step:**

```bash
sed -i 's/EXPECT_EQ(Program::kMaxCount, 1u << 28);/EXPECT_EQ(Program::kMaxCount, 1u << 27);/' \
    unittests/Encoding/EncodingTest.cpp
```

Result: `FrozenConstants.TheFormatsNumbers` red, step exit 1, `check-npu` still
18 of 18. lit never compiles the unit tests, so nothing else can see it.

### 2. The ISA staleness gate

**Fault A, the committed artifact:**

```bash
sed -i 's/| `MATMUL` | 4 |/| `MATMUL` | 44 |/' docs/ISA_MANUAL.md
git commit -m "temp: perturb" -- docs/ISA_MANUAL.md
bash scripts/check-isa-staleness.sh build
git reset --soft HEAD~1 && git restore --staged --worktree docs/ISA_MANUAL.md
```

The commit is not optional and that is the finding. The gate's subject is the
**committed** artifact, and an uncommitted hand edit is overwritten by the
regeneration the gate runs before it diffs.

**Fault B, the description:**

```bash
sed -i 's/"Elementwise addition."/"Elementwise addition, saturating."/' \
    include/NPU/Encoding/NPUISADescription.td
bash scripts/check-isa-staleness.sh build
```

Both exit 1 with the diff printed. Restoring returns exit 0.

### 3. The `sanitizers` job

**The isolating fault**, which the gcc build cannot see: put D-0021's message
back in `lib/Encoding/Validation.cpp`, so that the scratchpad range diagnostic
reads `"... from " + to_string(addr) + " to " + to_string(addr + bytes)`.

Result: `Validation.cpp:732:65: runtime error: signed integer overflow:
9223372036854775807 + 64`, step exit 1, and the default gcc build still green,
because the fault changes only a number inside a diagnostic string no test
asserts.

A product side fault for the same job is reverting D-0022's guard in
`Disassembler.cpp`, which also lights the gcc build's `Disassembly` tests. Both
were run.

**The first attempt proved nothing.** Reverting the *other* half of D-0021, the
operand extent comparison, left the job green: the corpus reaches the result
range message and not that comparison. Recorded rather than adjusted, per
Section 19.1.

### 4. The `nightly.yml` fuzz job

```bash
sed -i 's/  if (!state.reader.atEnd())/  if (false \&\& !state.reader.atEnd())/' \
    lib/Encoding/Program.cpp
```

Such a file decodes, validates, and re-encodes shorter than it came in, which is
the round trip property the target asserts and which no unit test covers for an
arbitrary mutated input.

Result: `a file that decoded and validated did not re-encode to itself` within
seconds, exit 77, crash artifact written. Restoring returns 1964713 runs in 61
seconds with no artifacts.

### Adding an opcode, which the gate asks for separately

Append a sixteenth opcode to `include/NPU/Encoding/NPUISADescription.td` and
build. The build fails in `Validation.cpp` and `PropertyTest.cpp`, at the two
switches over `Opcode` that carry no `default`, until each has a case. If the
record is incomplete the generator fails first, before any C++ is compiled. If
the committed manual is not regenerated, the staleness gate fails after that.

## Open questions

Four. None blocks the gate.

**The structure aware fuzzer was cut, deliberately, citing Section 2.**
`fuzz/nbin_structured_fuzzer.cc` is the third item on Section 2's cut list,
paired with mutation testing, and Section 2 says cutting it keeps the libFuzzer
target and the seed corpus, which are the parts that find bugs nobody imagined.
Both are here and both found defects this phase. The specific reason it went:
libprotobuf-mutator is in neither the CI image nor this machine's toolchain, so
it would have meant vendoring a dependency **and** maintaining a protobuf mirror
of `Program` by hand, which is a second hand maintained copy of exactly the
thing this phase's first commit exists to eliminate. **If it is ever
reinstated**, Section 2's reduced definition of done says the README records
what was struck; nothing else is owed.

**`Instruction` carries an operand shape, which Section 9.1's field list does
not name.** The list names operand addresses and operand strides. A stride
vector with no extents beside it addresses nothing in particular, and no check
in Section 9.2 could be written without one: `operand-extent` asks whether the
consumer's need fits the buffer it reads, and for `MATMUL` the K extent appears
in no result shape at all. It is documented field for field in
`docs/ISA_MANUAL.md`. Flagged here rather than buried, because it is the one
place the format is wider than the specification's prose.

**`HALT` at the end of a program is not a validation rule.** The encoder emits
one and every program this compiler produces ends with one, but Section 9.2's
check list carries no name for its absence, and inventing one would be inventing
a rule the specification does not have. A file without a trailing `HALT` decodes
and validates. **P7 decides what the simulator does when it runs out of
instructions**; the manual says it stops.

**The `count-cap` check lives in the decoder and not in `validate()`.** Section
9.2's third rule is about bounding a length before allocating from it, and the
only place a length arrives from untrusted input is the decoder, which checks the
cap and then checks the payload against the bytes that remain before reserving
anything. A copy in `validate()` could only fire on a `Program` built in memory
holding more than 2^28 of something, which is a branch no test can take, and
ground rule 6 forbids code kept just in case.

## Next command

```bash
git push -u origin phase/p6-binary-format
```

Then watch CI. **Four steps run for the first time**: `NPUEncodingTests`, the
ISA staleness gate, the whole `sanitizers` job, and, on its next schedule,
`nightly.yml`'s fuzz job. Their activation proofs are above and every one was
rehearsed locally under the same invocation, so a red run means the image
differs from this machine rather than that the tests are wrong.

Three things are worth watching specifically.

**The `sanitizers` job has never run in the image.** It needs `clang`, `lld` and
the system GoogleTest, and it configures a second build directory. The step that
greps for `Fuzzers: ON` is there so that an option which silently failed to take
is a red step rather than a job quietly rebuilding what the main build already
built.

**`reuse lint` sees `fuzz/` for the first time**, which is a new top level
directory and therefore a new place for a missing SPDX tag to hide. The corpus is
covered by a `REUSE.toml` entry, because a `.nbin` cannot carry a header.

**`check-npu` reports 18 rather than 15**, and `check-reachability --skip-models`
now checks three layers rather than two.

Then open the merge pull request.

## Next phase

**P7, the simulator and the reference interpreter.** Every fp32 kernel of
Section 10.1, the two port cost model with the single port reproducibility flag,
`Stats`, the bounds checked accessors, the strict scratchpad sizing, and
`python/npu_frontend/refexec.py`.

Five things P6 leaves on P7's desk.

1. **The dispatch skeleton is generated and waiting.**
   `build/include/NPU/Encoding/NPUISADispatch.def` is an X macro over every
   opcode with a `NEEDS_KERNEL` flag. Expanding it is what makes a new opcode
   with no kernel a build error rather than a runtime surprise, which is the
   half of Section 9.4's claim this phase could not demonstrate because there is
   no simulator yet.
2. **`validate()` is called again before execution, by contract.** Section 9.3
   says so, and the reason is that the two calls guard different moments. It is
   cheap: the whole 734 case corpus validates in single digit milliseconds.
3. **The scratchpad is sized strictly from `scratchpadBytes`.** Never grown to
   cover the writes it finds. Every hand built test program in this phase sets
   an explicit tight value with the arithmetic in a comment, and P7's should
   too.
4. **A kernel indexes its operands through their strides.** The rank 1 channel
   broadcast of ADR 0005 arrives as a stride 0 operand and needs no special case
   at all; an NHWC buffer arrives as NCHW extents with permuted strides. That is
   the obligation `docs/ARCHITECTURE.md` placed on P7 at P4, and the format now
   carries what it needs.
5. **The namespace is `nbin`, not `npu`.** The dialect owns `mlir::npu`, and P7
   is the first phase to include both headers in one translation unit.

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
