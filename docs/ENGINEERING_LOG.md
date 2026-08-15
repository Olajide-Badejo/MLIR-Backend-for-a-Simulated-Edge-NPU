# Engineering log

Dated entries recorded as problems happen: symptom, root cause, options considered,
chosen fix and why, commit, verification. This log is the raw material for the debug
report (report_debug) assembled at Phase 11.

## 2026-08-09 Republishing the numbers, and why they had gone stale

**Symptom.** Both committed PDFs were byte identical to their 2026-07-15 builds.
The committed results had been regenerated since. `report/generated/macros.tex`
in the working tree read `GitSha 38af13633388`, one regeneration behind the
results, which themselves read `66e5af5a`. Three artifacts, three different
answers to "what commit produced this".

**Root cause, and it is not the one the work order expected.** The plan assumed
the rebuild had simply been forgotten after commit `cc99973` reverted the
rebuilt PDFs out of the U2 merge. Running `make -C report` said:

```
make: Nothing to be done for 'all'.
```

The rule was

```make
main.pdf: main.tex references.bib $(wildcard sections/*.tex)
```

`main.tex` does `\input{generated/macros}`, and those macros are generated from
`experiments/results/`. Neither the generated tex nor the results were
dependencies. So regenerating the numbers could never rebuild the report: make
correctly observed that `main.tex` had not changed and did nothing. The rebuild
was not forgotten, it was unbuildable by the normal command. Anyone who ran the
documented command and saw "nothing to be done" would reasonably conclude the
PDF was current.

That also explains how the state persisted for a month across people who were
presumably running `make`.

**Chosen fix.** Three things, each independently necessary:

1. Regenerate the six results on the clean post U3 tree. No measured number
   moved: instruction counts, cycles, DRAM counters and max absolute error are
   identical across all six cells, and only `git_sha`, `timestamp`, and the wall
   clock `compile_ms` differ. That is the expected result for a validation and
   diagnostics phase, and checking it was the point of doing the diff.
2. Track `report/generated/`. It was gitignored, so the file linking committed
   results to committed PDFs was the one file not under version control. Now
   `test_macros_match_the_committed_results` compares the two on every run.
3. Add the results to the PDF's dependencies in `report/Makefile`, so the
   failure cannot recur silently. `make clean && make` was needed this once to
   get past the stale timestamps.

**A caveat worth recording.** The gate for this part checks
`strings report/main.pdf | grep -c 8095dbec` and expects 0. It was 0 before any
of this work as well. `\GitSha` is defined in `macros.tex` but no `.tex` file
cites it, so the sha never reaches either PDF and that check cannot distinguish
a rebuilt PDF from a stale one. What actually demonstrates the rebuild is that
both files changed size, 111648 to 112023 bytes and 31532 to 31890, and that the
figures they print are the ones in the committed results. A report that stated
its own provenance would make the check meaningful; that is a change to the
document and is left for a part that owns the report prose.

**Verification.** `test_committed_results_are_current` green for the first time
since the tree was audited, `test_no_result_traces_to_a_missing_commit` and
`test_macros_match_the_committed_results` green, 27 of 27 pytest, and the
evaluation section prints 91, 70, 23421, and 12710, matching the committed
results exactly.

## 2026-08-09 Phase U3 landed: what the phase was actually about

**Not a bug.** The closing entry for the phase, written at the merge.

U3 was scoped as "harden the binary interface", and the shape of the work turned
out to be consistent enough to be worth naming. Every defect fixed in it was the
same defect: **a check that could not fail.**

- `shapeElements()` tested its product against the cap after multiplying, so the
  guard performed the overflow it existed to catch.
- The simulator grew its scratchpad to cover every result address before
  checking whether result addresses were in range, so the bounds check was
  answering a question it had already made true.
- The trap path called `assert(false)`, so in the build most people compile, the
  refusal path aborted instead of refusing.
- `validate()` recorded an element count per address and then only asked whether
  the address existed, so an over read validated.
- `getCount()` capped counts at 2^28 and called that a bound on work, while the
  vector sized from it was 2 GiB.
- `encodeFunction` emitted an error and returned success.
- `npu-sim` compared its input against the program's declared inputs only for
  input 0.
- A test asserted that hostile input cannot cause undefined behaviour, in a way
  that was itself undefined behaviour.

Eight defects, one pattern. The lesson is not "write more checks", it is that a
check needs a test that proves it can *fail*, and the corpus, the negative tests,
and the two build modes are what turn that from an intention into a fact. The
validation suite's habit of asserting **which** rule rejected a program, rather
than only that something did, is what caught two of these; a suite asserting only
rejection would have been green throughout.

**What landed.** `Program::validate()` with 22 named rules, `decode` meaning
decode plus validate, `decodeUnvalidated` for the disassembler, a bounds checked
simulator that refuses gracefully in every build mode and sizes strictly from the
declared budget, diagnostics in `npu-sim`, `npu-translate`, and
`AllocateScratchpad`, one `--input` per declared region, 36 validation tests, a
1000 iteration property test, a 358 case fuzz corpus, and a manual that documents
the format, the byte order, every check name, and the version policy.

**Gate.** 51 of 51 encoding, 9 of 9 simulator, 14 of 14 lit, 25 of 26 pytest,
ASan and UBSan clean over both GoogleTest binaries, dash-lint clean. The one red
pytest is `test_committed_results_are_current`, the provenance guard that is
stale by design for the whole phase and is Part 5's to clear; it is also the only
item the regression baseline reports as drift. No benchmark number moved, which
is what a validation and diagnostics phase should do.

## 2026-08-09 Phase U3: the fuzz test was the thing with undefined behaviour

**Symptom.** The gate for this part is the first to run the whole fuzz corpus
under UBSan; earlier parts ran `Validation.*` only. It aborted:

```
FuzzTest.cpp:383:54: runtime error: signed integer overflow:
9223372036854775807 + 8192 cannot be represented in type 'long int'
```

**Root cause.** In the test, not the decoder.
`Fuzz.DecodeUnvalidatedNeverCrashesEither` touches every field a disassembler
would, to prove that reading them cannot crash, and accumulated them into a
`volatile int64_t`. The corpus contains a file declaring `INT64_MAX` bytes of
scratchpad, and `decodeUnvalidated` returns it, correctly, because not
validating is the entire point of that entry point. So the first addition
overflowed. The test asserting that hostile input cannot cause undefined
behaviour was itself the undefined behaviour.

Confirmed pre-existing rather than introduced here: at Part 2's tip both the
`INT64_MAX` corpus case and the signed accumulator are already present,
unchanged. The near cap cases added in this part are all rejected, so they
never reach the accumulator. Nothing in this part caused it; the part's gate is
simply the first thing that looked.

**Chosen fix.** Accumulate in `uint64_t`, where wrapping is defined. The sum
was always meaningless, since the point is to touch the fields rather than to
compute anything.

One related decision. The loop over constant regions called `r.byteSize()`,
which multiplies the extents out and would overflow on a hostile shape for the
same reason. Rather than make `byteSize()` saturate, the loop now touches the
shape length. `byteSize()` has exactly four callers, all in
`InstructionEncoder.cpp`, all on shapes that came from the MLIR type system,
and `disassemble()` does not call it, so no shipping path reaches it with
decoded input. Hardening it would have been guarding against a caller that does
not exist, and the test would have been inventing the risk it then caught.

**Verification.** The full corpus under ASan and UBSan, 46 of 46 with
`EncodeFunction.*` excluded for the documented MLIR slab poisoning reason, and
no diagnostic of any kind. Peak RSS 99.9 MB under ASan, 0.15 s.

## 2026-08-09 Phase U3: the second input was always zero

**Symptom.** `npu-sim` parsed `--input` into a single `std::string`, so a second
`--input` overwrote the first and a program declaring two inputs ran with one of
them left as whatever the DRAM was initialised to, which is zeros. Nothing was
printed. The simulation completed, wrote an output, and reported statistics.

**Root cause.** Not really a bug in the parsing, which does exactly what a
single string can do. The bug is that the tool never compared what it was given
against what the program declared. `program->inputs` has the answer in it and
was consulted only for `inputs[0]`, to size check that one file. Everything
about the second input region was ignored, including its existence.

This is the same defect as the multi output one fixed earlier in this phase and
the multi function one in `npu-translate`: a tool written against the single
case, then handed a program that is not the single case, silently doing a
fraction of the work. Worth noting that none of the three were found by a test,
because every test in the repository uses LeNet, which has one input, one
output, and one function.

**Options considered.**

1. Accept `--input` repeatedly and require the count to match. Chosen. Spec 5.3
   allows refusing instead of implementing, but there is nothing hard here: the
   simulator already takes a vector of inputs, and `run()` already loops over
   `program->inputs`. The single string was the only thing in the way.
2. Accept one `--input` holding all inputs concatenated. Rejected: it needs the
   caller to know the exact byte layout, and a wrong split would be silent,
   which is the failure mode being removed.
3. Refuse a multi input program outright. Rejected as worse than the two lines
   of work needed to support it.

**Chosen fix.** `--input` collects into a vector, the count is compared against
`program->inputs.size()` before anything is read, and a mismatch is refused with
both numbers in the message. Each file's float count is then checked against its
own region's shape rather than only input 0. The usage string and the Tools
section of `docs/ISA_MANUAL.md` say so; the manual did not document `npu-sim` at
all before, so the numbered multi output files are now written down too.

A deliberate behaviour change rides along: a program with declared inputs run
with no `--input` used to simulate them as zeros and now is refused. Nothing in
the repository relied on it, since every caller passes one `--input` for a one
input model, and the benchmark and end to end paths are unchanged. No
`docs/BREAKING_CHANGES.md` entry, and the pytest suite is what confirms it.

**Verification.** A new pytest builds a genuinely two input program, an
`npu.add` of two arguments, through the real pipeline. With two `--input` flags
it runs and the output is `[11 22 33 44]`, which is the sum; had the second
input still been zeros it would have been `[1 2 3 4]`, so the assertion
distinguishes the fix from the bug rather than merely observing a clean exit.
With one flag it exits nonzero and the message names 2 and 1; with three, 2 and
3. Demonstrated at the terminal as well as in the test.

## 2026-08-09 Phase U3: an error message is not a failure

**Symptom.** Run `npu-translate` on a function holding an op the encoder has no
case for, in this instance an `npu.relu` that was never lowered to npuisa:

```
$ npu-translate test/Encoding/unencodable.mlir -o /tmp/u.nbin
loc(...): error: cannot encode unexpected op
$ echo $?
0
$ ls -la /tmp/u.nbin
-rw-r--r-- 1 elijah elijah 150 Aug  9 18:31 /tmp/u.nbin
```

It printed an error, exited successfully, and wrote a file.

**Root cause.** The `.Default` case of the encoder's `TypeSwitch` set the local
`emit` flag false and emitted a diagnostic, and `emit` is the same flag used by
the two ops that legitimately produce no instruction, `npuisa.const` and the
terminator. So "I have nothing to emit for this" and "I do not understand this"
were the same signal, and the function fell through to `return program` either
way.

What makes this the worst of the four defects in this part is the shape of the
artifact. A crash is fine, a nonzero exit is fine, no output is fine. This
produced a well formed `.nbin` that decodes, validates, and runs, and is simply
missing the relu. Every downstream check passes on it. A build script that
looks at the exit code sees success. The only evidence is a line of stderr that
scrolled past.

**Options considered.**

1. Return failure from inside the `.Default` lambda. Not possible directly, and
   working around it would stop at the first bad op.
2. Track it in the existing `emit` flag. Rejected: that flag means "no
   instruction for this op", which is a legitimate state for two ops. Conflating
   them is what caused this.
3. A separate `unencodable` flag checked after the loop. Chosen. One run then
   names every op it cannot encode instead of only the first, which matters when
   a lowering pass was skipped entirely and a dozen ops are unlowered.

**Chosen fix.** A `bool unencodable` alongside the loop, set only in `.Default`,
and `if (unencodable) return failure();` after the loop and before the halt is
appended. The caller in `npu-translate` already tested `mlir::failed(program)`
and already opened its output stream after that test, so no change was needed
there; that was checked rather than assumed.

**Incidental.** The lit test for this needs `not npu-translate`, and `not` was
not among the substituted tools in `test/lit.cfg.py`, so it failed with
`not: command not found`. Added it. Worth recording because it means no negative
tool test could have been written for this suite before now, which is a plausible
part of why a tool that exits 0 on failure went unnoticed.

**Verification.** `EncodeFunction.RefusesAnUnencodableOp` shown failing before
and passing after, asserting both the failure and that the diagnostic names the
problem. `test/Encoding/unencodable.mlir` runs the real tool and checks all
three properties: nonzero exit, the diagnostic, and no output file. After the
fix the same command exits nonzero and writes nothing. lit 14 of 14.

## 2026-08-09 Phase U3: a cap on the count is not a cap on the work

**Symptom.** The work order for this part asked only for corpus cases probing
the decoder's count cap from both sides, on the grounds that the corpus never
probed just under it. Adding the probes turned the request into a defect
report. The 36 new cases are a few dozen bytes each, and every one of them was
already correctly refused, but refusing them cost:

```
Maximum resident set size (kbytes): 2102888
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:43.37
```

Two gigabytes and forty three seconds to say no to 36 files that together
weigh under four kilobytes.

**Root cause.** `Reader::getCount()` rejected counts above `1u << 28` and
returned anything at or below it as trustworthy. That bounds the number, not
the work behind it. `getVec()` then does

```cpp
uint32_t n = getCount();
std::vector<int64_t> v(n);
```

which sizes and zero fills the vector before reading a single element, so a
count of 2^28 is a 2 GiB allocation regardless of how many bytes the file
actually holds. The cap was doing the opposite of its stated job for exactly
the values just below it: 2^28 and 2^28 minus one are the two counts it lets
through, and they are the two most expensive counts expressible. The cases the
old corpus did probe, `0xFFFFFFFF` and `1u << 29`, were all above the cap and
so were refused for free, which is why the hole never showed up.

Worth naming the general shape: a length prefixed format where the length is
validated against a constant rather than against the bytes remaining is a
decompression bomb by construction. The constant can only ever be a guess about
what is reasonable; the remaining byte count is the truth.

**Options considered.**

1. Lower the 2^28 cap. Rejected: it is a guess either way, and any cap high
   enough to be useful for real shapes is still high enough to be expensive.
2. Reserve incrementally and read as you go, so the allocation follows the
   bytes actually present. Rejected as a bigger change than needed, and it
   leaves the count itself unvalidated, so the loop bound is still attacker
   controlled.
3. Validate the count against the bytes that remain. Chosen. A count of `n`
   elements needs at least `n * minBytesPerElement` bytes behind it, and if
   they are not there the file is truncated, which the reader already knows how
   to say.

**Chosen fix.** `getCount` takes the least space one element can occupy and
refuses a count the remaining bytes cannot back:

```cpp
if (n > (1u << 28) || n > remaining() / minBytesPerElement) {
  ok = false;
  return 0;
}
```

with the per section minimums written down next to the magic: 12 bytes for a
region, 16 for a constant, 54 for an instruction, `sizeof(int64_t)` for a shape
element, `sizeof(float)` for constant data. They are lower bounds, so they stay
correct if a field is added later. The old cap stays as a second line, since a
file large enough to back a huge count is still not a file worth decoding.

This cannot reject a well formed file, because a well formed file by definition
carries the bytes it names. The 1000 iteration round trip property test is what
holds that down, and it stayed green.

**Verification.** The same 36 cases, same binary, after the change:

```
Maximum resident set size (kbytes): 6208
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:00.00
```

2.0 GiB to 6 MB. The whole encoding suite now peaks at 25.6 MB and runs in
0.41 s. Corpus grew from 322 to 358, which the `CorpusIsLargeEnough` test now
pins exactly rather than only flooring at 200, so a future change to the
generator has to justify a number. Encoding 47 of 47, simulator 9 of 9, lit 13
of 13.

## 2026-08-09 Phase U3: membership is not an extent

**Symptom.** Found by adversarial testing rather than by a failing test, and
recorded as ASSESSMENT 13.2 item 4. A `DMA_STORE` reading 100 elements from a
4 element buffer near the top of the scratchpad passes `Program::validate()`
and then traps in the simulator. The header on `Program.h` says validate checks
every invariant the simulator relies on, so either the header or the code was
wrong, and it was the code.

**Root cause.** The written before read walk keeps
`std::map<int64_t, int64_t> writtenElements`, address to element count. It
stores the count, and then only ever asks `find(addr) == end()`. The count sat
there unused. So the walk answered "did anything write here?" while the
question the simulator needs answered is "did enough get written here?".

The trapping case is the mild one, because it is at least loud. Move the same
over read down into the middle of the scratchpad and it stays in bounds: the
program validates, the simulator runs to completion, and 96 elements of
whatever is adjacent get folded into the result. No trap, no diagnostic, a
plausible looking wrong answer. That is precisely the failure mode the no
silent failures rule exists to prevent, and it survived a phase whose whole
purpose was to prevent it.

**Options considered.**

1. Track full shapes instead of counts, and check every operand exactly.
   Rejected for now: for `CONV2D` that means reproducing convolution shape
   inference inside the validator, which is a second implementation of
   something the compiler already does and a second thing to keep in sync.
2. Check only the ops whose operand extent is determined by the result extent,
   and require a weaker property of the rest. Chosen.
3. Leave it and rely on the simulator's bounds check. Rejected: the bounds
   check cannot see the interior over read at all, since it is in bounds.

**Chosen fix.** A new `operand-extent` check. Before the operand loop the walk
computes what this consumer reads:

- `DMA_STORE`, `RELU`, `ADD`, `MUL`, `RESHAPE` read `resultElements` from each
  operand, which is exact.
- `POOL_MAX` and `POOL_AVG` require at least `resultElements`, a lower bound,
  since a pool reads a window at least as large as its output for everything
  the backend emits.
- `CONV2D` and `MATMUL` require a non zero recorded count and nothing more.
  This is the weak rule, stated as such in the code comment. Their extents
  follow from the recorded shapes and this walk tracks counts, so a real check
  needs shape tracking. Recorded here so the gap is visible rather than
  implied.

The failure names the instruction index, the operand index, elements needed,
and elements written, so the message is enough to find the bug without a
debugger.

**Verification.** Three new tests, the first two shown red before and green
after. `RejectsAnOperandReadLargerThanWhatWasWritten` is the ASSESSMENT case,
reading 100 from 4 at address 32000 so the read would run past a 32768 byte
scratchpad. `RejectsAnInteriorOverRead` is the same over read at 20480, which
stays in bounds and would otherwise never be caught by anything. Both were
accepted before. `AcceptsAnExactExtentRead` reads exactly what was written and
was green before and after, which is what stops the new rule from being a
blanket refusal of `DMA_STORE`. Encoding 46 of 46, simulator 9 of 9, lit 13 of
13.

## 2026-08-09 Phase U3: a convenience that swallowed the hardening

**Symptom.** `Validation.SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory`
failed. It plants a `resultAddr` far past the end of the scratchpad, runs the
program, and expects `SimResult.error` to name instruction 2. The simulator ran
it clean and returned no error at all, even though the bounds checked accessor
that the same phase had just added was sitting right there in the path.

**Root cause.** A convenience that predates the hardening, at the top of `run()`:

```cpp
int64_t spBytes = program.scratchpadBytes;
for (const Instruction &in : program.instructions)
  if (in.resultAddr >= 0)
    spBytes = std::max(spBytes, in.resultAddr + numElements(in.resultShape) * 4);
```

The scratchpad was then sized from `spBytes`. So the answer to "is this address
inside the scratchpad?" was computed by first making the scratchpad big enough
to contain the address. The check could still catch a negative address or a
misaligned one, but for the case it was written for, an oversized result
address, it was structurally unable to fire. The test was not wrong; the check
was unreachable.

Two things make this worse than a dead check. The loop runs before any
validation, so `in.resultAddr + numElements(in.resultShape) * 4` is arithmetic
on unvalidated attacker controlled values at the exact entry point U3 exists to
defend, and it can overflow int64 or ask for an absurd allocation. And the
comment under it told the reader the scratchpad "can exceed
program.scratchpadBytes", which documented the hole without anyone reading it
as one.

**Options considered.**

1. Gate the expansion behind `validate()`, so it only grows for programs already
   known good. Rejected: it reorders the arithmetic rather than removing it, and
   it keeps a rule where the declared budget means one thing for validated
   callers and another for library callers.
2. Keep the expansion behind an opt in flag for convenience callers. Rejected:
   spec 11.4 item 2 allows it, but a flag that makes the memory safety property
   optional is a flag someone will set.
3. Size strictly from the declared `scratchpadBytes`. Chosen, and this is what
   spec 11.4 item 2 lists first. It removes the arithmetic on unvalidated input
   instead of moving it, and it gives `scratchpadBytes` one meaning everywhere:
   the memory the program asked for.

**Chosen fix.** Delete the loop and size from the declared field:

```cpp
std::vector<float> sp(std::max<int64_t>(1, program.scratchpadBytes / 4), 0.0f);
```

**Did anything real regress?** No. `-npu-allocate-scratchpad` computes
`highWater = max(offset + size)` over every buffer it places and writes it to
`npuisa.scratchpad_bytes` (`AllocateScratchpad.cpp:226` and `:232`), and the
encoder copies that straight into `program.scratchpadBytes`
(`InstructionEncoder.cpp:108`). So every compiled program declares exactly the
extent it uses, with no slack and no shortfall, and the expansion was always a
no op for it. Confirmed by running the suites: lit 13 of 13, and the end to end
pytest, which is the real LeNet path through `npu-sim` including the 140 KB
spilling budget, unchanged. Both lit tests that reach the encoder run
`-npu-allocate-scratchpad` first, so neither depended on the expansion either.
Nothing goes in `docs/BREAKING_CHANGES.md`.

What did break is six hand built unit programs in `SimulatorTest.cpp`, which set
`dramBytes` but never `scratchpadBytes` and so had been running on a scratchpad
conjured entirely from their instruction stream. That is the finding, not
collateral damage: those tests asserted numerics against a memory the program
never declared. They now declare the smallest size that covers their writes (32,
48, 80, 64, 20, and 32 bytes) with the arithmetic written down, and that landed
as its own commit first so no commit on this branch carries a red simulator
suite.

**Verification.** The target test shown failing before and passing after, in
`build-asserts` and `build-ndebug` both. Encoding 43 of 43, simulator 9 of 9
including two new tests: `RefusesAWriteJustPastTheScratchpadEnd` puts a result
4 bytes past a 32 byte declaration and asserts the refusal names instruction 1
and reports the declared 32 byte region rather than a grown one, with a control
at 36 bytes that runs clean; `ScratchpadIsSizedFromTheDeclaredFieldOnly` works
in the last four cells of a 4096 byte declaration, which only succeeds if the
declaration and not the instruction extent is what allocated the memory. lit 13
of 13, pytest 24 passed with `test_committed_results_are_current` red, which is
the dirty tree provenance guard expected red for all of U3 (ASSESSMENT 13.3) and
is cleared by Part 5.

## 2026-08-09 Phase U3: an assert that fired on the case the check was for

**Symptom.** Not a test failure, which is why it survived review. The bounds
checked accessor added earlier in U3 ends its refusal path with

```cpp
assert(false && "simulator memory access out of bounds");
return nullptr;
```

so the moment the check actually fires, an assert enabled build aborts the
process. The `return nullptr` below it, the `SimResult.error` string filled in
just above it, and every null test at the call sites are all unreachable in the
build most people compile.

**Root cause.** Two contracts written into one function. `SimResult.error`
exists so that a caller can be handed a diagnostic, which is the U3 promise:
no silent failure, and no crash either. The assert says the opposite, that
reaching this point is a programming error worth aborting for. Both cannot be
true. It was reachable by design: the header says the simulator is reachable as
a library and from hand built `Program` values, which is exactly where an
unvalidated program comes from.

The comment above the accessor had drifted too. It claimed a failure "aborts in
a debug build and clamps to a scratch cell in a release build". The first half
described the assert. The second half described nothing at all: there is no
clamp anywhere in the function, and there never was in this revision. A comment
that describes a design that was considered and not built is worse than no
comment, because it is what a reader will believe.

**Options considered.**

1. Keep the assert and treat a trap as a bug in the caller. Rejected: it
   contradicts the reason `SimResult.error` exists, and it makes the library
   entry point abort the host process on hostile input.
2. Replace it with `llvm_unreachable` or an `abort` with a better message.
   Rejected for the same reason, and spec 11.4 item 5 asks for graceful refusal
   in every build mode.
3. Keep the assert but only under a debug flag the tests can turn off. Rejected:
   it makes the tested behaviour differ from the shipped behaviour, which is the
   defect, not the fix.
4. Delete the assert so the already correct refusal path is the only path.
   Chosen.

**Chosen fix.** Delete the assert and the now unused `<cassert>` include, and
rewrite the comment to state what the code does: every access is checked in
every build mode, the first refusal records its message and returns `nullptr`,
each caller tests the pointer and skips the access, and execution runs to the
end so the caller gets a result carrying the diagnostic. Only the first refusal
is kept, because it is the one that explains the run and the rest are its
consequences.

Audited all eleven `spAt` and `dramAt` call sites while here, since a missing
null test becomes a null dereference the moment the assert stops aborting
first. All eleven already test the pointer: the constant and input preload and
the output readback use `if (float *p = ...)`, and every opcode arm binds its
pointers and then tests them together before touching memory. Conv2D and MatMul
correctly distinguish a bias that is absent (two operands, `nullptr` is legal)
from a bias that was refused. No call site needed fixing.

**Verification.** `Validation.SimulatorRefusesAnOutOfBoundsAccessInsteadOfCorruptingMemory`
passes in both `build-asserts` (`-DCMAKE_BUILD_TYPE=Debug`, asserts on) and
`build-ndebug` (`-DCMAKE_BUILD_TYPE=Release`, `NDEBUG`), which is the pair that
would have diverged before. `grep -rn "assert(false)" lib/ tools/` is empty.

## 2026-08-09 Phase U3: a test that never reached the rule it named

**Symptom.** `Validation.RejectsRegionPastTheEndOfDram` failed, but not by the
program being accepted. `validate()` rejected it. What failed was the assertion
about which rule did the rejecting:

```
Expected equality of these values:
  error->check
    Which is: "region-offset"
  check
    Which is: "region-in-range"
rejected, but by the wrong rule: program: region-offset: output 0 has DRAM
offset 8190, which is not 4 byte aligned
```

**Root cause.** Test authoring, not product code. The test set
`p.outputs[0].dramOffset = 8190` to push a 40 byte output past the end of an
8192 byte DRAM. 8190 is not a multiple of 4, and `checkRegion()` tests alignment
before it tests range, so the alignment rule claimed the program and the range
rule never ran.

The design point is worth recording, because this is the first time it paid for
itself. `expectRejected()` asserts the rule name rather than merely that
something was rejected. A test that only checked "rejected" would have been
green here while exercising a completely different rule, and `region-in-range`
would have shipped with no coverage at all while appearing to have some. That
is the exact failure mode the file's header comment warns about, and it caught
its own author.

**Options considered.**

1. Reorder `checkRegion()` so range is tested before alignment. Rejected: it
   edits product code to suit a test, and alignment first is the correct order
   anyway, since every access indexes as `addr / 4` and a misaligned offset
   makes the range arithmetic meaningless.
2. Relax the assertion to "rejected by something". Rejected: that is precisely
   the weakness this file was written to avoid, and it would leave
   `region-in-range` untested while looking tested.
3. Choose an aligned offset that still overruns. Chosen.

**Chosen fix.** 8160. It is 4 byte aligned (8160 = 4 * 2040), and the output
region holds 10 fp32 elements, so it spans [8160, 8200), which ends 8 bytes past
an 8192 byte DRAM. Alignment passes, range fires, and the test asserts the rule
it was written for. The trailing comment now carries that arithmetic so the
constant is not a magic number, and says what the old one got wrong.

**Verification.** Shown failing before ("rejected, but by the wrong rule ...
region-offset ... not 4 byte aligned") and passing after. No product code was
touched, and the encoding suite moved by exactly this one test.

## 2026-08-09 Phase U3: the overflow guard was itself the overflow

**Symptom.** `Validation.RejectsShapeThatWouldOverflow` failed, in the most
pointed way available: the program it plants is the exact case the guard exists
for, and `validate()` accepted it. Built with `-fsanitize=undefined` the same
test also reported signed integer overflow at `Program.cpp:197`.

**Root cause.** `shapeElements()` accumulated the product and then checked it:

```cpp
n *= d;
if (n > kLimit)
  return std::nullopt;
```

A check can only run on a value that has already been computed, and computing
that value is the undefined behaviour. For a shape like `{2^40, 2^24}` the
multiply overflows int64 and wraps, so what the comparison sees is a small
number and the function returns a plausible element count for a shape it exists
to refuse. Downstream that count is multiplied by 4 and compared against a
region bound, where a wrapped product compares as comfortably in range. The
failure mode is therefore not a crash at decode but a program that validates
and then walks off the end of a buffer at run time, which is precisely what
phase U3 exists to make impossible.

Worth recording that the test was written before the implementation and was red
for a real reason, not a wrong expectation. A guard exercised only with inputs
an order of magnitude below its own limit looks correct forever.

**Options considered.**

1. Accumulate in a wider type (`__int128`) or in unsigned arithmetic and keep
   checking after the multiply. Rejected: it buys correctness with a compiler
   extension, or with wrapping semantics that are defined but still leave the
   check written the wrong way round for the next person to copy.
2. Test `n > kLimit / d` before the multiply. Chosen. It is the standard
   division based overflow test, it needs no wider type, and the `d <= 0`
   rejection immediately above it guarantees the divisor is at least 1.
3. Lower `kLimit` far enough that no product can overflow. Rejected: it changes
   an encoding contract to work around an arithmetic bug, and any cap large
   enough to be useful is still large enough to overflow when squared.

**Chosen fix.** Test the headroom before consuming it, so the guard never
performs the operation it is guarding:

```cpp
if (d <= 0 || d > kLimit)
  return std::nullopt;
if (n > kLimit / d)
  return std::nullopt;
n *= d;
```

`kLimit` stays at `int64_t{1} << 40` and the signature is unchanged, so the
three callers (the region check, the constant data check, and the instruction
result check) keep the contract they were written against. The comment above
the function now says the cap is inclusive, because that is the one part of the
behaviour a reader cannot recover without working through the integer division.

**Verification.** `Validation.RejectsShapeThatWouldOverflow` shown failing
before the change ("expected result-shape to reject this") and passing after.
Added `Validation.RejectsShapeAtTheOverflowBoundary`, which pins both sides of
the cap: `{2^20, 2^20}` is exactly 2^40 and has to fall through the shape rule
to `result-in-range`, while `{2^20, 2^21}` is one bit past and has to be caught
by `result-shape`. Asserting which rule fires is what keeps the boundary from
drifting silently, since either shape is rejected either way. A rebuilt
`build-san` running `Validation.*` under ASan and UBSan reports no diagnostic of
any kind, in particular nothing at `Program.cpp`.

## 2026-08-08 Phase U0: a regression net before any upgrade work starts

**Symptom.** Not a bug. The upgrade specification (`UPGRADE_SPEC_V3.md`) asks for a
long sequence of changes to a repository that currently works, under a prime
directive of not breaking it. Before Phase U0 there was no way to answer "did that
change move anything?" other than eyeballing test output. The four suites report
pass or fail but say nothing about instruction counts, cycles, DRAM traffic, or
numerics, which are exactly the things an optimizer change moves silently.

**Root cause.** The project has assertions but no recorded expectations. The end to
end pytest asserts agreement with onnxruntime at `rtol=1e-3`, five orders of
magnitude looser than the 3e-8 actually observed, so a numerics regression of four
orders of magnitude would pass. Nothing at all watched cycles or DRAM bytes.

**Options considered.**

1. Rely on the existing suites and read the diffs by hand each phase. Rejected:
   the interesting quantities are not asserted anywhere, so there is nothing to
   diff.
2. Add tighter assertions to the existing tests and stop there. Rejected as
   insufficient on its own: it catches numerics but not instruction counts,
   cycles, or DRAM traffic, and it would not notice a test quietly disappearing.
3. Record a machine readable snapshot of everything the repository does today and
   diff against it at every phase gate. Chosen.

**Chosen fix.** `scripts/regression-baseline.sh` (a thin wrapper that finds the
venv and the prebuilt lit) over `scripts/regression_baseline.py` (the engine).
It records per suite pass, fail, and skip counts plus the full test name list, the
cost model constants parsed straight out of `CostModel.h`, the tool versions, and
for each of the six LeNet cells the instruction count, simulated cycles, DRAM
bytes read and written, and the max absolute error against onnxruntime. The
simulated output tensor of every cell is frozen as a `.npy` golden file.

Three details worth recording:

- All four suites emit JUnit or XUnit XML natively (`llvm-lit --xunit-xml-output`,
  `--gtest_output=xml:`, `pytest --junitxml=`), so one parser covers all of them
  and the recorded shape is uniform. That was better than scraping four different
  human readable summaries.
- The instruction count comes from the simulator's `stats.instructions`, not from
  the regex in `run_benchmarks.py`. The two disagree badly: the regex reports
  91 / 82 / 70 for `-O0` / `-O1` / `-O2` and the simulator reports 28 / 25 / 21,
  because the regex counts `npuisa.const` (data, not an instruction) and matches
  inside type strings such as `!npuisa.buffer`. Recording the wrong number as the
  baseline would have baked the error in.
- The git sha and the tool versions are recorded but are notes rather than drift.
  The sha moves with every commit, and a machine update is not a behaviour change.
  `--strict-tools` promotes tool changes to failures when that is wanted.

**Verification.** Recorded at `7de39c6`: 37 passed, 0 failed across the five
suites. Ran `--check` immediately on unchanged code and it reported zero drift.
Then perturbed one cost model constant (`macsPerCycle` 256 to 255, chosen because
it feeds every conv and matmul cycle estimate while changing no numerics) and
`--check` failed with 10 items, catching it on three independent axes: the
constant itself, the `CostModelArithmetic::MatchesFormulas` GoogleTest flipping to
failed, and all six cells reporting `simulated_cycles` up by exactly 5. Reverted,
and `--check` went clean again. The net catches things.

**Incidental finding.** The first `ninja baseline-check` run reported a cmake
version change that the direct run did not. The cause is that a pip installed
cmake 4.4.0 in `~/.local/bin` shadows the apt cmake 4.2.3 on `PATH`, but only in
login shells. So the project can be configured by either of two cmake versions
depending on how the build was invoked, which is a real reproducibility hazard on
this machine. The baseline now records each tool as "resolved path: version" so
this surfaces as a legible note rather than a mysterious version change.

## 2026-08-08 Phase U2: ASan fires inside MLIR, and it is not our bug

**Symptom.** The new `sanitizers` CI job, run locally before pushing, aborts:

```
ERROR: AddressSanitizer: use-after-poison on address 0x70459d3e15b8
WRITE of size 32 at 0x70459d3e15b8 thread T0
    #1 llvm::SmallVectorImpl<std::pair<mlir::TypeID, void*>>::operator=(...)
    #2 mlir::Dialect::addType(mlir::TypeID, mlir::AbstractType&&)
    #5 mlir::BuiltinDialect::initialize()
    #10 EncodeFunction_LowersSmallProgram_Test::TestBody() EncodingTest.cpp:100
```

Every frame between the test and the fault is MLIR's own. `NPUSimulatorTests`
runs clean; only the one encoding test that constructs an `MLIRContext` fails.

**Root cause.** An instrumentation mismatch, not a memory bug. `llvm/Support/
Compiler.h` defines `LLVM_ADDRESS_SANITIZER_BUILD` purely from
`__SANITIZE_ADDRESS__`, which gcc defines whenever `-fsanitize=address` is on.
`BumpPtrAllocator` in `llvm/Support/Allocator.h` is header only and, under that
macro, poisons each new slab (line 351) and unpoisons per allocation. So the
copy of the allocator instantiated in our instrumented translation units
believes LLVM is an ASan build and poisons slab memory, while the prebuilt
`libMLIRIR.a`, compiled with no instrumentation and no matching unpoison, writes
straight into it. The shadow map shows exactly that: a clean allocation followed
by a poisoned tail being written.

**Options considered.**

1. Build a second LLVM with `-DLLVM_USE_SANITIZER=Address`. Correct, and the
   only thing that makes MLIR itself sanitizable. Rejected on cost: a second
   multi hour image, hours of CI per tag change, to find bugs in LLVM rather
   than in this project.
2. `ASAN_OPTIONS=detect_container_overflow=0`, the usual advice for linking
   instrumented code against uninstrumented LLVM. Rejected: that suppresses
   container annotation reports, and this is a `use-after-poison` from
   `BumpPtrAllocator`'s explicit poison calls, which the option does not cover.
3. Disable the annotations by defining the macro away. Rejected: `Compiler.h`
   defines it unconditionally inside the `__SANITIZE_ADDRESS__` branch, so it
   cannot be overridden without patching LLVM.
4. Scope the job to the code the job exists for. Chosen.

**Chosen fix.** The sanitizer job runs `NPUEncodingTests
--gtest_filter=-EncodeFunction.*` and the whole of `NPUSimulatorTests`. Nothing
is actually given up. ASan is here for the memory unsafe surface, which is
`Program::decode`, the `Program::validate` and fuzz corpus work landing in phase
U3, and the simulator's raw `spAt`/`dramAt` pointer arithmetic. None of it
touches MLIR. `EncodeFunction.*` still runs uninstrumented in `build-and-test`.
The exclusion is commented in `ci.yml` and in `docs/BUILD.md` with the reason,
because a bare `--gtest_filter` that looks like someone hiding a failure is
worse than the failure.

**Verification.** `NPUEncodingTests --gtest_filter=-EncodeFunction.*` passes 3
tests and `NPUSimulatorTests` passes 7, both clean under
`-fsanitize=address,undefined` with `halt_on_error=1` and `detect_leaks=1`.

## 2026-08-08 Phase U2: what the first reachability run found

**Symptom.** Not a failure so much as a measurement. The first run of
`scripts/check-reachability.py` reported six unreachable ops out of twelve, where
`docs/ASSESSMENT.md` section 2.2 had identified three.

**The three the audit already knew about**, `transpose`, `concat`, and
`batch_norm`, are missing every layer: no importer converter, no lowering
pattern, no encoder case, no simulator kernel.

**The three it did not.** `add` and `mul` are fully lowered, encoded, and
simulated, and have no ONNX converter at all, so nothing can produce one from a
real model; they exist only for hand written IR and for the lit tests. And
`avg_pool2d` is present at every layer and is exercised by nothing, because
LeNet uses max pooling throughout. That last one is the interesting category:
not broken, never run on a real graph, and invisible to a test suite that only
asks whether the tests pass.

**Chosen fix.** None yet, deliberately. Phase U2's job is the check, not the
implementation, and the gaps belong to U7 and U8. All six carry dated exemptions
in `docs/DESIGN_DECISIONS.md` and the check fails once a date passes.

One design note. The checker refuses to run at all if an op in `NPUOps.td` has no
entry in its `NPUISA_EQUIVALENT` table, rather than inferring a mapping. Adding
an op to the dialect should force an answer to "what does this become in the
ISA". An inferring checker would have quietly passed `transpose` on the day it
was added, which is the whole failure being guarded against.

## 2026-08-08 Phase U1: the published numbers came from a commit that never existed

**Symptom.** Phase U1 was meant to be routine cleanup: track the benchmark results,
and make the harness stop reusing stale ones. I wrote the staleness check first and
a test asserting the committed results match the tree. It failed, which was
expected, since `docs/ASSESSMENT.md` section 4.2 had already found the results were
three commits behind. What was not expected was the reason it gave:

```
generated at unknown commit 8095dbec, HEAD is d93e73de
```

**Root cause.** `git cat-file -e 8095dbec^{commit}` fails. That sha is not in this
repository. The assessment had read the manifest and assumed an old commit; in fact
the results were produced from a working tree that was never committed in the form
that produced them, probably an amended or discarded Phase 9 state. So the six
numbers the README headline table, the evaluation section, and both PDFs are built
from were reproducible from no point in the history. This is worse than stale. A
stale result can at least be reproduced by checking out its commit.

**Options considered.**

1. Compare the manifest sha to HEAD exactly, as `UPGRADE_SPEC_V3.md` section 8.1
   item 4 words it. Rejected after trying it. A result can only be committed by a
   commit that comes after the run that produced it, so under this rule every
   result is stale the moment it lands, every invocation regenerates the whole
   suite, and the committed files never match their own commit. The rule is
   self defeating as literally written.
2. Compare only the cost model constants. Rejected: it would not have caught this,
   since the cost model was unchanged throughout.
3. Compare the sha, and when it differs, ask whether anything that can actually
   move a number changed between then and the working tree. Chosen.

**Chosen fix.** `staleness()` replaces `valid()` and returns the reason rather than
a bool, so the harness can print why it is regenerating. A result is reusable when
its cost model constants match and either its manifest sha is HEAD, or `git diff`
between that sha and the tree touches nothing under `RESULT_INPUTS`: the dialect,
the passes, the encoder, the simulator, the tools, the frontend, and the harness
itself. Uncommitted edits count, because a result produced from a dirty tree is not
reproducible from any commit either. A sha the repository does not recognise is
stale by definition, which is what caught this. Deviation from the spec's literal
wording recorded in `docs/DESIGN_DECISIONS.md`.

**Verification.** Nine tests in `test/Python/test_benchmarks.py`, including one that
walks recent history for a documentation only commit and asserts the filter sees
through it, and `test_committed_results_are_current`, which is the standing guard.
Shown failing before the regeneration and passing after.

## 2026-08-08 Phase U1: refusing a batch size rather than lying about it

**Symptom.** `docs/ASSESSMENT.md` section 2.1: a 2 batch conv plus relu returns
1.19e-07 error on image 0 and 1.85 on image 1. No diagnostic anywhere.

**Root cause.** `Simulator.cpp` conv2d has `int64_t n = 0; // batch is 1 for the
supported models`, and `pool()` iterates `C = inS[1]` without ever seeing the batch
dimension. Neither is a coding mistake so much as a faithful implementation of a
specification whose stated scope was one LeNet at batch 1. Nothing else in the
pipeline had a reason to disagree, so the importer, the verifiers, the lowering,
and the allocator all accepted N greater than 1 and the allocator even sized the
buffers correctly for it.

**Options considered.**

1. Fix the kernels now. Rejected for this phase, not on merit: it is phase U6, it
   needs batched cost model changes and new GoogleTests, and U1's job is to stop
   the compiler being wrong, not to widen it. Shipping the guard first means the
   silent failure closes today rather than in two phases.
2. Reject any rank 4 tensor whose leading dimension is not 1. Rejected: a
   convolution weight is OIHW, so LeNet's 6x1x5x5 first kernel would be read as a
   batch of six and every real model would be refused. There is a test pinning
   this, because it is the obvious wrong way to write the guard.
3. Guard activations only, at the ops that are actually wrong. Chosen.

**Chosen fix.** `check_unbatched_activation` in the importer, called from the conv
and both pool converters, and `verifyUnbatchedActivation` in `NPUOps.cpp`, called
from the `conv2d`, `max_pool2d`, `avg_pool2d`, and `batch_norm` verifiers. Two
layers, as the spec asks, and both name the tensor and its shape and say the
limitation is tracked rather than permanent.

Deliberately not guarded: `matmul` and the elementwise ops. `matmul` iterates `M`
correctly, so a rank 2 batch is genuinely fine there, and relu, add, and mul are
elementwise over the whole buffer. Refusing those would be refusing something that
works, which the no silent failure rule does not ask for.

**Verification.** Four pytest cases and four lit cases, all shown failing before the
guard and passing after. Two of them are controls: batch 1 still imports, and a
6x1x5x5 weight is not mistaken for a batch.

## 2026-07-15 Phase 0: environment reconciliation

**Symptom.** The build spec (Section 3) assumes Ubuntu 24.04, a WSL budget of 20 GB
RAM / 24 processors, `LLVM_PARALLEL_COMPILE_JOBS=20`, and Python 3.10+. The actual
target machine reports different facts on every one of those axes, and one of them is
safety critical.

**Findings (verified 2026-07-15).**

- Disk is a non issue here. The WSL2 rootfs lives on `/dev/sdd` with about 932 GB free,
  and Windows C: has about 135 GB free. None of the spec's disk mitigations are needed.
- WSL2 Ubuntu 26.04 (not 24.04), kernel 6.18, user `elijah`, passwordless sudo.
- Toolchain present: gcc/g++ 15.2.0, cmake 4.2.3, ninja 1.13.2, git 2.53.0,
  python3 3.14.4, venv working. Missing and installed via apt: lld 21.1.8, ccache 4.12.3.
- The host `~/.wslconfig` deliberately caps WSL2 at `memory=12GB`, `processors=8`,
  `swap=8GB`, `autoMemoryReclaim=gradual`.

**Root cause of the deviation that matters.** The `.wslconfig` cap is not an oversight.
Its own header records that its absence crashed the machine on 2026-07-14: with no budget,
vmmemWSL grew to about 12.78 GB and never released it, leaving 2.2 GB free of 31.7 GB, and
the host paged itself to death (aggravated by GPU benchmark allocations spilling through
WDDM). The file explicitly instructs keeping build parallelism at or below 6 jobs.

**Options considered.**

1. Follow the spec literally: raise `.wslconfig` to 20 GB / 24 proc, build with 20 compile
   jobs. Rejected: directly reintroduces the condition that crashed the machine one day
   earlier, and 20 parallel gcc jobs compiling TableGen heavy MLIR translation units would
   exceed a 12 GB budget regardless.
2. Keep `.wslconfig` as is and adapt the build to the real budget. Chosen.

**Chosen fix.** Do not touch `.wslconfig`. Configure the one time LLVM/MLIR build with
`LLVM_PARALLEL_COMPILE_JOBS=6` and `LLVM_PARALLEL_LINK_JOBS=1`, link with lld, and enable
ccache. Accept that the build takes longer than the spec's 1 to 3 hour estimate at this
parallelism. Pin the LLVM tag to `llvmorg-22.1.8` (current tip of `release/22.x`) rather
than the moving branch, for reproducibility.

**Open risk carried forward.** Python is 3.14.4. torch, onnxruntime, and nanobind wheel
availability for 3.14 is unverified and is a risk for Phase 5 and later. To be checked
before the ONNX frontend work, with a 3.12 venv as the fallback plan.

**Verification.** `df`, `free`, `nproc`, `lsb_release`, tool `--version` probes captured
above. LLVM pinned tag confirmed by `git ls-remote`; shallow clone HEAD resolves to
`llvmorg-22.1.8`.

## 2026-07-15 Phase 0: repository relocated to WSL native storage (spaces break lit)

**Symptom.** After the LLVM build finished, `npu-opt` built and `npu-opt --help` worked,
but the very first lit test failed with `FileCheck: Too many positional arguments
specified!` and a shell error splitting a path at a space. The build tree sat under the
Windows project folder whose ancestors contain spaces (`Corrected Projects`,
`MLIR Backend for a Simulated Edge NPU`).

**Root cause.** LLVM's lit expands the `%s` substitution to the test file's real path and
does not quote it. Under the external `sh`, `npu-opt /mnt/c/.../Corrected Projects/...mlir`
splits into multiple arguments, so both `npu-opt` and `FileCheck` see too many positional
arguments. LLVM lit and FileCheck do not support spaces in paths, and lit resolves symlinks
to the real path, so a space-free symlink over the spaced directory did not help either
(verified: `%s` still expanded to the spaced `/mnt/c` realpath).

**Options considered.**

1. Keep the repo in the Windows folder and patch lit substitutions to quote `%s`. Rejected:
   fights the framework, fragile, and would have to be re-done for every future config.
2. Put the repo at a space-free Windows path such as `C:/Users/jidro/npu-mlir`. Space-free
   and my editing tools stay native, but WSL builds and lit still run over the slow 9p
   `/mnt/c` mount for the whole 12 phase project.
3. Host the canonical repo in WSL native storage at `/home/elijah/npu-mlir`. Space-free,
   native build and test speed, and the standard recommended layout for WSL2 development.
   Chosen.

**Chosen fix.** Relocated the git working tree (with history) to `/home/elijah/npu-mlir`.
Builds go to `/home/elijah/npu-mlir/build`. From Windows the repo is reachable at
`\\wsl.localhost\Ubuntu\home\elijah\npu-mlir`, which the Windows side editing tools handle
correctly (verified by reading and enumerating files over that path). A pointer file in the
original Windows folder records the new location. The original folder keeps only the spec
and that pointer, so there is a single source of truth.

**Verification.** After relocation, `ninja check-npu` reports `Passed: 1 (100.00%)` and
`npu-opt --help` lists the registered dialects. All paths in the generated
`lit.site.cfg.py` and in the `%s` expansion are now space-free.

## 2026-07-15 Phase 1: two ODS surprises building the first ops

Two failures worth recording while standing up the npu dialect, both of the "the
TableGen half and the C++ half of an interface are separate includes" family.

1. **`Variable not defined: 'SameOperandsAndResultType'`** from mlir-tblgen. This trait is
   not in `mlir/IR/OpBase.td` (it appears there only in FIXME comments). It is defined in
   `mlir/Interfaces/InferTypeOpInterface.td`, which must be included from the ops `.td`.

2. **`'InferTypeOpInterface' is not a member of 'mlir'`** at C++ compile time, after fixing
   1. Reason: `SameOperandsAndResultType` is a trait list that also mixes in
   `InferTypeOpInterface::Trait`, so the generated op class references
   `::mlir::InferTypeOpInterface`. The TableGen include is not enough; the C++ side needs
   `#include "mlir/Interfaces/InferTypeOpInterface.h"` in the ops header and the
   `MLIRInferTypeOpInterface` library on the dialect's `LINK_LIBS`.

**Lesson for later ops.** Any convenience trait may drag in an interface with a matching
C++ header and link library. When a trait is added in ODS, add its interface header and
link library at the same time rather than after the next build failure.

**Verification.** `ninja check-npu` reports `Passed: 2 (100.00%)`; the core ops round trip
through two `npu-opt` passes without change.

## 2026-07-15 Phase 2: CommonFolders poison template argument

**Symptom.** Using `constFoldBinaryOp<FloatAttr>` and `constFoldUnaryOp<FloatAttr>` from
`mlir/Dialect/CommonFolders.h` to fold the pointwise ops failed to compile with a static
assertion: "PoisonAttr is undefined, either add a dependency on UB dialect or pass void as
template argument to opt-out from poison semantics."

**Root cause.** In this LLVM release these helpers gained a `PoisonAttr` template parameter
that defaults to `ub::PoisonAttr`, whose definition is only available if the UB dialect is
linked. The parameter is the third one, after `AttrElementT` and `ElementValueT`. My first
fix passed `void` as the second argument, which wrongly set `ElementValueT` to void.

**Chosen fix.** Opt out of poison propagation by passing all three explicitly:
`constFoldBinaryOp<FloatAttr, APFloat, void>(...)` and the unary equivalent. This avoids
taking a dependency on the UB dialect for folds that cannot produce poison.

**Verification.** `-canonicalize` folds `add`, `mul`, and `relu` over constants to a single
`npu.constant`, and the canonicalization patterns (relu idempotence, reshape identity and
reshape of reshape) fire. The BatchNorm folding pass turns `bn(conv(x, W))` into one conv
with hand verified weights `[2, 6]` and bias `[-0.5, -1.5]`. `check-npu` reports 5 of 5.

## 2026-07-15 Phase 9: benchmark harness caught a spill correctness bug

**Symptom.** The first benchmark run at a tight scratchpad budget (140 KB, which forces the
allocator to spill) showed the maximum absolute error against onnxruntime jumping from
2.98e-08 (exact up to fp rounding) to about 0.08 at optimization levels O1 and O2. The
default 1 MB budget, which does not spill, stayed exact.

**Root cause.** The instruction encoder assigned DRAM offsets only to inputs, constants, and
the values returned from the function. A spill inserts a `dma_store` of an intermediate
buffer followed by a `dma_load` reload, and that spill store's result is a DRAM temporary
that is neither an input, a constant, nor a return value. With no offset assigned, the
`DenseMap` lookup returned the default 0, so every spill wrote to DRAM offset 0, clobbering
the model input, and the reload read the input back instead of the spilled buffer.

The scratchpad allocation lit test had checked only that spilling inserted the right number
of DMA instructions, never the numerics, so it did not catch this.

**Chosen fix.** Give every spill temporary its own DRAM region in the encoder. Scan the
block for `dma_store` ops whose result is not returned, assign each a fresh DRAM offset in
the unified `dramOffset` map, and lay them out after the constants and before the outputs.
The reload `dma_load` reads from the same map, so it now finds the spill region.

**Verification.** After the fix the objdump of a spilling program shows the spill store going
to a distinct DRAM offset (`dram[0x800]`, separate from the inputs at 0x0 and 0x400 and the
outputs), and every benchmark cell, spilling or not, is back to 2.98e-08 against onnxruntime.
Added an end to end pytest at a 140 KB budget to lock the fix in.
