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

**P5, scratchpad allocation.** Branch `phase/p5-allocation`, cut from `main` at
`44d865c`, which is the P4 merge. Seven commits, not pushed. The seventh is the
one that carries this table, so it is the branch tip and is named by subject
rather than by a sha it cannot know.

| Commit | Subject |
|---|---|
| `24e93ee` | `feat(npuisa): assign scratchpad offsets with a sweep line and spill when they do not fit` |
| `078701d` | `fix(npuisa): measure a memref's byte range from its strides, not from its extents` |
| `7ed3263` | `test(npuisa): property test the sweep line against a brute force recomputation` |
| `4069bc2` | `docs: predict the allocator's compile time curve and its fragmentation ratios` |
| `253c842` | `fix(npuisa): stop reordering the pads before the windowed extent arithmetic` |
| `3dc7c66` | `perf(experiments): measure the allocator's compile time curve and its fragmentation ratios` |
| tip | `docs: record the P5 defects and hand off the phase` |

The branch point again, since P3 and P4 both had something to say about it.
Local `main` was stale when this session began, exactly as the last two handoffs
predicted it would be. The first command of the session was
`git fetch origin && git branch -f main origin/main`, which moved local `main`
to `44d865c`, and the branch was cut from that.

**The commit order carries a ground rule.** `4069bc2` is the prediction and
`3dc7c66` is the first number, in that order and in separate commits, which is
what ground rule 15 asks for. One of the two predictions in `4069bc2` was wrong
and it stays in that file unedited, with the answer in the engineering log.

## Gate status

The P5 gate is the roadmap entry's, plus what Sections 8 and 13.1 attach to this
pass. Every item is **met locally**. Item by item, with the proof.

| Gate item | Proof |
|---|---|
| The five allocator lit cases of Section 17.1: fits, spill, fragmentation, no reuse while live, budget too small | `scratchpad-alloc.mlir` carries fits, fragmentation and no reuse while live; `spill-heuristic.mlir` carries spill; `alloc-budget-too-small.mlir` carries the negative. Every offset is checked **by value** with the arithmetic written out beside it, because a test that checked only that an offset appeared would pass against an allocator that gave every buffer offset zero |
| Fragmentation with out of order deaths, and reuse pinned by offset | The `@fragmentation` function: `%a` [0, 3] at 256 bytes, `%b` [1, 5] at 256, `%c` [4, 6] at 512, so `%a` and `%c` never share an index. `pack` places `%c` at 0 and reuses; `interval` leaves a 256 byte hole and lands at 1024 for a peak of 768 |
| No reuse while live, which Section 13.1 calls the silently wrong case | `@no_reuse_while_live` checks both halves: two buffers that overlap at one index get different offsets, and a third that is disjoint from the first takes its offset back. An allocator that never reused would pass the first half alone. `PlacementProperty.NoTwoBuffersLiveAtOnceShareAByte` asserts it over 1000 randomized sets under both strategies as well |
| The sweep line property test against brute force in `NPUAllocatorTests` | `SweepLineProperty.TheSweepLineAgreesWithBruteForce`: 1000 randomized interval sets at seed 24301, 904 of them with more than one interval and a non zero peak. **The peak index is asserted as well as the value**, because the index is what the spill heuristic reads |
| The compile time benchmark committed at four sizes | `experiments/compile_time_benchmark.py` at 500, 1000, 2000 and 5000 operations. Measured: 2.3, 4.6, 9.6 and 26.1 milliseconds, a mean exponent of 1.05 |
| The three diagnostics each with a `-verify-diagnostics` test | multi block in `alloc-multiblock.mlir`, budget too small in `alloc-budget-too-small.mlir` in two shapes, unknown heuristic in `alloc-unknown-option.mlir` alongside unknown strategy and a bad alignment, all three at once because the pass reports every bad option rather than the first |
| Both allocation strategies present and selectable | `strategy=pack` and `strategy=interval`, exercised on the same file by two run lines with two check prefixes in `scratchpad-alloc.mlir`, and against each other on seven models in `experiments/allocator_fragmentation.py` |
| `fragmentation_ratio` computed and reported per model | The pass writes `npuisa.fragmentation_ratio` plus the two integers it is computed from. Per model, under both strategies, all seven models: `pack` between 1.0000 and 1.0178, `interval` between 1.0002 and 1.5030, `pack` never worse. The table is in `docs/ENGINEERING_LOG.md` |
| Both spill heuristics, with the tie breaks Section 13.1 specifies | `@two_victims` in `spill-heuristic.mlir` is one program on which `longest-range` and `cost` choose different buffers, identified by the type of the spill slot rather than by an offset. Every tie break key has its own unit test in `AllocatorTest.cpp`, one key at a time |
| The sweep line from the start, with its ordering rule | `ADeathAtTheSameIndexAsADefinitionComesFirst` and `ALastUseAtTheSameIndexIsStillLive` differ by one in one field and by a factor of two in the answer. `TheFirstIndexWithTheGreatestSumWins` pins the other half |
| The spill trigger is offset assignment failing, never the peak exceeding the budget | `@the_trigger_is_placement_failure_not_the_peak`: the peak is 1536, the budget is 1536, so "peak exceeded budget" is false, and the interval placement still fails and spills. The same function under `pack` needs no spill at all |
| Spill semantics per Section 13.1 | A `dma_store` after the definition and a `dma_load` before each later use, with the reload replacing that use, checked with `CHECK-NEXT` so the store really is adjacent to the definition |
| The allocator counts the DMA it inserts, per Section 8 | `npuisa.spill_dma_count` on the function, plus an `inserted-dma` pass statistic |
| `npuisa.scratchpad_bytes` and `npuisa.scratchpad_budget` set, per Section 8 | Both, plus the peak, the ratio and the two spill counts. Checked by value in every lit case |
| Offsets as `memref.view` over one flat buffer, per Section 8 | A dedicated run line asserts that **no scratchpad buffer allocation survives anywhere in the module**, which a `CHECK-NOT` inside a labelled block could not do because it only covers the gap between two positive matches |
| Multiple blocks diagnosed, not ignored | `@two_blocks` in `alloc-multiblock.mlir` |
| The `computeBufferRange` question P4 left open | Answered, with the reasoning in `docs/ARCHITECTURE.md` as a marked P5 extension and five tests in `InterfaceTest.cpp`. A stride 0 view has a byte range and it is the range of the bytes it addresses |
| `NPUAllocatorTests` built and its CI step switched on | `.github/workflows/ci.yml`, guard kept and the else branch turned into a failure, which is the shape `NPUInterfaceTests` took at P2 |
| `docs/PASSES.md` updated in the same commit as the pass, per ground rule 12 | `24e93ee` carries both |

### Verification output

Every command below was run on this branch at `3dc7c66`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 15 discovered, 15 passed, 0 failed. Ten at P4, plus this phase's five |
| `build/bin/NPUAllocatorTests` | 29 tests from 6 suites, 29 passed. Two of them are property tests at 1000 cases each |
| `build/bin/NPUInterfaceTests` | 23 tests, 23 passed. Eighteen at P4, plus this phase's five |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed, untouched by this phase |
| `python -m pytest test/Python -q` | 142 passed, 7 deselected, exit 0, unchanged from P4 |
| `mypy` | no issues found in 11 source files |
| `black --check .` | 21 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 138 of 138 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, import and lowering layers checked |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `git status --short` | empty |

**Three gtest binaries exist now, not two.** `NPUAllocatorTests` is this phase's
and is the third. `NPUEncodingTests` is P6's and `NPUSimulatorTests` is P7's, per
the activation table.

## Activation proofs

**One step activates at P5: `NPUAllocatorTests`.** Section 19.0 requires it be
broken once deliberately, shown red, and restored, and this is that proof,
performed on this branch and reproducible from the recipe below.

**The recipe, one line:**

```bash
sed -i 's/return left.isDefinition < right.isDefinition;/return left.isDefinition > right.isDefinition;/' \
    lib/Dialect/NPUISA/Transforms/ScratchpadAllocation.cpp
```

That inverts the sweep line's ordering at equal indices, so a death is no longer
processed before a definition, which is the off by one Section 13.1 names as
changing which buffer gets spilled.

**Rehearsed under the exact invocation the CI step uses**, which is
`./build/bin/NPUAllocatorTests` with no arguments inside `set -euo pipefail`, and
not under a `--gtest_filter` naming the test that was broken. That distinction is
the whole point of the rehearsal and the engineering log records why: filtering
proves that the broken test fails, which was never in doubt, where what needs
proving is that a failure inside the binary reaches the step's exit code and
turns the job red. Those are different claims. P3's handoff already recorded an
activation proof that proved nothing the first time it was performed, and this is
the same hazard wearing different clothes.

Result, rebuilt and run:

- `SweepLine.ADeathAtTheSameIndexAsADefinitionComesFirst` failed, which is the
  hand written case aimed straight at the rule.
- `SweepLineProperty.TheSweepLineAgreesWithBruteForce` failed at **case 0 of
  1000**, with the message naming the case number and the seed, so the property
  test caught it on its first randomized input rather than needing a rare shape.
- `PlacementProperty.NoTwoBuffersLiveAtOnceShareAByte` failed too, because the
  placement reads the peak.
- **The step exited 1.**

Restored with `git checkout --`, rebuilt, and the binary exits 0 with 29 of 29
passing. `git status --short` is empty.

## Open questions

Five. None blocks the gate.

**The default spill heuristic is provisional and is marked so in three places.**
Section 13.1 says the default is chosen with data, not intuition, and the
ablation that produces that data needs the harness at P10 and the experiment at
P13. `longest-range` is the default today because it is the simpler rule and the
baseline, so a change at P13 will be a move towards the smarter rule with
evidence behind it rather than away from one. **Whoever runs that ablation should
change the default in the same commit as the data**, and not before.

**The placement is quadratic in the buffer count and that is a measured
decision, not an oversight.** Offset assignment scans every already placed buffer
for each new one. The compile time benchmark says the linear term still dominates
at 5000 operations, which is an order of magnitude longer than any model in the
suite, so the interval tree that would fix it is not going in. The benchmark is
committed; if P13's tiling makes functions much longer, rerun it before assuming
this still holds.

**The fragmentation ratio includes alignment padding.** The peak sums raw buffer
sizes and the high water mark sums offsets rounded up to 64 bytes, so every
published ratio has a floor slightly above 1.0 that has nothing to do with the
packing algorithm. Measured on `lenet`, the effect is 32 bytes, or 0.02 percent.
Separating the two would need a second peak definition and a second number to
keep straight, and Section 13.1 defines the ratio the way the three papers it
cites do. **P10 should carry this sentence into the result schema's field
documentation** rather than leaving a reader to wonder why a chain reads 1.0002.

**The encoder inherits the spill slots.** A spilled value lives in a
`memref.alloc` in `#npu.dram` marked `npuisa.spill_slot`, which is the one place
in this compiler that allocates DRAM and which amends a sentence P4 wrote.
`docs/ARCHITECTURE.md` states the obligation in one line: **the encoder gives
each such allocation an address in the DRAM map, the way it already does for
constants and for the input and output regions.** They are marked in the IR
rather than inferred, so finding them is a predicate and not an analysis.

**A spilled buffer that was loaded from DRAM is stored back to DRAM
unnecessarily.** Section 13.1's spill semantics are a `dma_store` after the
definition and a `dma_load` before each later use, and this pass implements
exactly that. When the buffer's definition is itself a `dma_load` from a DRAM
value that is still live, the value is already in DRAM and the store is
redundant: the reload could read the original source. That would remove one
transfer per spilled input. It is **not implemented**, deliberately, because it
changes the DMA count and Section 13.1 specifies the semantics without it, and
because the ablation of Section 16.2 compares against those semantics. If it is
ever done it belongs with the tiling work at P13, in its own commit, with the
DRAM byte counts before and after.

## Next command

```bash
git push -u origin phase/p5-allocation
```

Then watch CI. **One step runs for the first time**: `NPUAllocatorTests`, which
was guarded off since P0 and prints its own error now rather than an off line if
the binary is missing. Its activation proof is above and was rehearsed locally
under the same invocation, so a red run there means the image cannot build the
binary, not that the tests are wrong; the configure log's `GoogleTest:` line is
where to look, because CI resolves gtest from the system package where this
machine resolves it from the LLVM build tree.

Two other things are worth watching. `check-npu` reports 15 rather than 10.
`reuse lint` sees `experiments/` for the first time, which is a new top level
directory and therefore a new place for a missing SPDX header to hide.

Then open the merge pull request.

## Next phase

**P6, the binary format and the generated ISA.** Section 9 in full, plus the
generated ISA description of Section 9.4 and the fuzzing of Section 17.3. It is
the five to seven session phase, and the roadmap says why: generating the opcode
enum after hand writing the encoder means writing the encoder twice.

Four things P5 leaves on P6's desk, all expanded in the open questions above:

1. **The spill slots need DRAM addresses.** Every `memref.alloc` in `#npu.dram`
   marked `npuisa.spill_slot` is a buffer the encoder must place, alongside the
   constants and the input and output regions.
2. **The six function attributes are the encoder's input.**
   `npuisa.scratchpad_bytes` is what Section 9 calls `scratchpadBytes`, and
   `npuisa.scratchpad_budget` is what the simulator sizes against.
   `npuisa.fragmentation_ratio`, `npuisa.scratchpad_peak_bytes`,
   `npuisa.spill_count` and `npuisa.spill_dma_count` are for the result cell at
   P10 and are not part of the binary format unless P6 decides otherwise and
   says so.
3. **Allocated IR is what the encoder will see**, which means `memref.view` over
   one flat arena and `memref.reinterpret_cast` above it, not typed
   `memref.alloc` operations. An address is `computeBufferRange(value)->offset`,
   which is the function the overlap rule already uses, so there is no second
   place for the arithmetic to live.
4. **P4's out parameter convention is still only positional**, and P4's handoff
   already flagged that if P6 wants it explicit, an argument attribute is the
   place. That question is now sharper, because the spill slots give the DRAM map
   a third category of buffer.

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
