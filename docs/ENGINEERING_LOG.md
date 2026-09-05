# Engineering log

Dated entries recorded as problems happen: symptom, root cause, options considered,
chosen fix and why, commit, verification. This log is the raw material for the debug
report (report_debug) assembled at Phase 11.

## 2026-08-09 Phase U4: a fixed absolute tolerance cannot survive a scale change

**Symptom.** The new end to end matrix went red on twelve of its thirty cells,
every `large_pos` and `large_neg` cell at all three levels and both budgets:

```
AssertionError: absolute error 1.526e-05 exceeds 1.000e-06
```

**Root cause, and it is not the compiler.** `ATOL = 1e-6` was set in phase U1
against the observed 2.98e-8, measured on a standard normal input. That input
produces LeNet outputs of order 0.15, where a float32 ulp is 1.49e-8, so 1e-6 is
about 67 ulps of headroom and the bound is generous.

The `large_pos` and `large_neg` classes drive a constant $\pm$1e3 through the
network. The outputs come out of order 25, where a float32 ulp is 1.91e-6. The
measured absolute error there is 1.526e-5, which is **8 ulps**: the same
arithmetic quality as the 2 ulps seen on `normal`, from the same cause,
reordered but mathematically identical fp32 accumulation. Meanwhile 1e-6 is half
an ulp at that scale, so the bound is not merely tight, it is unsatisfiable by
any correct implementation, including onnxruntime compared against itself in a
different summation order.

The relative bound passed comfortably in all twelve cells. Relative error across
the whole matrix peaks at 4.85e-6 against a 1e-5 budget. So every signal that
scales correctly said the results were fine, and the one fixed constant said
they were not.

**The rule this ran into.** The work order says never loosen a tolerance to make
a cell pass, and separately says to keep `ATOL = 1e-6`. Those instructions
conflict once input classes with different output magnitudes are introduced,
which is what the same work order asks for. Loosening the constant to 2e-5 would
have been exactly the forbidden move: it would weaken the `normal` cells, where
1e-6 is doing real work, in order to accommodate a different scale.

**Chosen fix.** Express the absolute bound as what it always meant, a number of
ulps at the scale of the output being checked, with `ATOL` as a floor:

```python
ULP_BUDGET = 16

def absolute_bound(reference):
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        return ATOL
    return max(ATOL, ULP_BUDGET * float(np.spacing(np.float32(scale))))
```

At the `normal` scale, 16 ulps is 2.4e-7 and the 1e-6 floor still dominates, so
those cells are checked exactly as strictly as before and nothing that passed is
loosened. At the large scale the budget is 3.05e-5 against a measured 1.53e-5,
so a genuine doubling of the error still fails. Near zero outputs keep an
absolute guarantee from the floor.

This is a deviation from the work order and is recorded as one rather than
applied quietly.

**A smaller decision.** The `zeros` class was expected to make the relative
bound vacuous by driving the reference to exactly zero. It does not: LeNet has
biases, so a zero input produces a non zero output and the relative check is
meaningful, and in fact `zeros` is where the worst relative error in the whole
matrix occurs, 4.85e-6. The skip for an exactly zero reference is kept because
it is the correct guard for a model without biases, but it does not fire here,
and saying so is better than implying the class is untested.

**Verification.** Thirty cells, all green, full matrix in 3.4 seconds. Worst
absolute error across the matrix 1.526e-5 at `-O0`, 1 MB, `large_pos`; worst
relative error 4.848e-6 at `-O0`, 1 MB, `zeros`. The default run is 3 cells plus
the two meta tests in under a second; CI runs all thirty.

## 2026-08-09 Phase U4: the ablation says two passes do nothing, and one is harmful

**Not a bug.** The leave one out ablation the v2 specification called "the
evaluation's backbone", finally built. Part 8 measured each pass on the IR it is
handed. This measures what the finished program loses when a pass is removed
from an otherwise complete `-O2`, which is a different question wherever passes
interact.

**What it found, at the 1 MB budget:**

```
canonicalize    instrs  +0   cycles    +0   dram  +0.0 KB
npu-fuse-ops    instrs  +4   cycles  +298   dram  +0.0 KB
symbol-dce      instrs  +0   cycles    +0   dram  +0.0 KB
```

Two of the three `-O2` passes make no difference at all. My first assumption was
that the ablation was not ablating, so I checked it directly rather than
reporting it: compile the model with `-canonicalize -npu-fuse-ops -canonicalize
-symbol-dce` and with `-npu-fuse-ops -symbol-dce`, and diff the IR.

```
npu-dialect op count after full -O2 opt   : 18
npu-dialect op count after -O2 minus canon: 18
optimized IR identical? True
lowered npuisa IR identical? True
```

Byte identical. The zero is real.

**Root cause of the zero.** `FuseOps.cpp:72` runs `applyPatternsGreedily`. The
greedy driver folds constants and erases dead operations as part of its fixed
point loop, so by the time it has finished fusing there is nothing left for
canonicalization to do. Canonicalization is not useless: at `-O1` it is the only
pass and is responsible for the entire drop from 339 KB to 176 KB of DRAM. It is
redundant *in the presence of fusion*.

This is exactly the interaction the ablation exists to expose and that Part 8's
per pass measurement cannot see. Measured in isolation, canonicalization at
position 0 removes three operations, 28 down to 25. Measured by removal, it
removes nothing, because something else would have.

**The finding that matters more, at 140 KB:**

```
npu-fuse-ops    instrs  +2   cycles   -96   dram  -6.1 KB
```

The signs flip. Removing fusion at the tight budget makes the program *faster*
and reduces DRAM traffic. Fusing an activation into its producer extends that
value's live range across the fused op, and under a budget that already forces
spilling, a longer live range buys a spill and its reload, which cost more than
the instruction the fusion saved. `-O2` fusion is a win at 1 MB and a loss at
140 KB.

ASSESSMENT 5.1 predicted the passes would behave oppositely at the tight budget,
which is why the work order insisted on both budgets. It was right, and an
ablation table reporting only the generous budget would have concluded that
fusion is the one pass worth having.

**Design decision recorded.** `-canonicalize` appears twice at `-O2`. Ablating
it removes every occurrence, because "is this pass worth having" is the question
an ablation answers, and removing only the second occurrence answers "is running
it twice worth it", which is narrower. Written down in
`docs/DESIGN_DECISIONS.md` because both readings produce a row labelled
`canonicalize` and they can disagree.

**A trap avoided.** Ablation records live in `experiments/results/` alongside the
full cells and carry the same `model`, `opt_level`, and `scratchpad_budget`
keys. `plot_results.load()` and `results_to_tex.load_rows()` both key by those
fields, so an ablation would have silently replaced the real `-O2` row in the
figure and the results table, and which one would have depended on glob order.
Both now filter on the presence of `ablated_pass`. The same collision would have
hit five of the existing tests, which now ask for full rows explicitly.

**Verification.** Four new tests. `test_every_o2_pass_has_an_ablation_row` takes
the expected set from `_passes_for_level(2)` at assert time, so adding a pass
without an ablation fails. `test_ablation_deltas_are_consistent` recomputes every
delta from the two committed files, so a reader never has to trust the
subtraction. `test_ablation_numerics_are_unchanged` holds every ablation to the
end to end tolerance. `test_ablation_result_paths_follow_the_convention` checks
they are covered by `staleness()` like any other result rather than being
quietly exempt from the guard that keeps the published numbers honest.

## 2026-08-09 Phase U4: measuring each pass, without inventing a measurement

**Not a bug.** The v2 specification called per pass ablation deltas "the
evaluation's backbone". `run_benchmarks.py` iterated
`product(models, levels, budgets)` and stored one total `compile_ms` and one
post lowering op histogram, so the report could say what `-O2` buys over `-O0`
and nothing about any individual pass.

**The trap this part had to avoid.** Part 7 had just finished removing a scalar
that came from the wrong source, and the obvious way to build this one would
have repeated the mistake: regex the IR dump before and after each pass and
count. That is the same wrong measurement, applied six times per cell instead of
once. `npu-opt` calls `registerAllPasses()`, so the compiler will answer both
questions itself and neither needs new C++.

**Op counts.** `--print-op-stats` prints a text table by default:

```
Operations encountered:
-----------------------
  builtin.module     , 1
     func.func       , 1
```

The work order allowed parsing that behind a pinning test, but `--help-list`
shows the pass takes a `json` option, reachable through the textual pipeline
form:

```
--pass-pipeline=builtin.module(print-op-stats{json=true})
```

which prints a plain JSON object. Used that, so there is no whitespace sensitive
parser to pin in the first place. The parser that remains raises on the old text
form, on truncated JSON, and on an empty object, because an empty histogram
would record every pass as having changed nothing, and that reads as data rather
than as a failure.

**Wall clock.** `--mlir-timing --mlir-output-format=json` emits a tree, and the
names in it are C++ class display names rather than command line flags:
`Canonicalizer` for `-canonicalize`, `NPUFuseOps` for `-npu-fuse-ops`. Nested
pipelines appear as `'func.func' Pipeline` groups. Walking the tree and taking
the leaves in document order gives the passes in pipeline order once `Parser`,
`Output`, `Rest`, and `Total` are dropped.

The mapping from flag to display name is written down explicitly and checked
position by position. That matters for `-O2`, where `-canonicalize` appears
twice and a name based lookup would be ambiguous; matching by position with the
name as a check catches a pipeline change that nobody taught the harness about,
and a pass with no entry in the map raises rather than being recorded unnamed.

**What the measurement says.** For the `-O2` default cell:

```
 0 canonicalize                   28 -> 25   0.40 ms
 1 npu-fuse-ops                   25 -> 21   0.20 ms
 2 canonicalize                   21 -> 21   0.10 ms
 3 symbol-dce                     21 -> 21   0.10 ms
 4 npu-lower-to-npuisa            21 -> 33   0.20 ms
 5 npu-allocate-scratchpad        33 -> 33   0.10 ms
```

Two things worth noting before Part 9 reads too much into it. The second
`canonicalize` and `symbol-dce` change no op count at all on this model, which
is a fact about LeNet rather than about the passes. And lowering *raises* the op
count from 21 to 33, which is correct: one `npu` op becomes several `npuisa`
ops plus the DMA that feeds it. An op count is not a cost.

**A limit to record.** `--mlir-timing` rounds to four decimal places of a
second, so 0.1 ms is the smallest non zero value it can report. Every pass on
LeNet lands between 0.10 and 0.40 ms, comfortably above the floor, but on a
smaller model a pass could round to zero and the "every pass has a positive wall
clock" test would fail. That is the correct failure: it would mean the timing
source can no longer resolve the thing being measured, which is worth stopping
for rather than recording a zero.

**Runtime.** The whole harness is 2.8 seconds for six cells, against a five
minute budget. Each cell now runs `npu-opt` once for timings plus twice per pass
for the before and after histograms, so roughly fifteen extra invocations per
cell, and it does not matter at this size.

**Verification.** Five new tests. `test_every_pass_in_the_pipeline_has_a_record`
reads the expected pipeline from `_passes_for_level` at assert time rather than
hardcoding it, so adding a pass without instrumenting it fails.
`test_o0_has_no_optimization_passes` is the negative case that would catch a
hardcoded enumeration. `test_every_pass_has_a_wall_clock` rejects a zero.
`test_op_stats_parser_pins_the_format` and
`test_pass_timing_parser_raises_on_a_missing_pass` pin both external formats,
including that a pipeline pass missing from the timing output raises.

## 2026-08-09 Phase U4: the headline number was never an instruction count

**Symptom.** The repository committed two different answers to the same
question and shipped both. `experiments/results/lenet_O2_default.json` said
`instruction_count: 70`. `test/baseline/baseline.json` said `instructions: 21`
for the same cell. The README printed 70 in its headline table and, three
screens further down, a real disassembly excerpt reading "1 inputs, 1 outputs,
10 constants, 21 instructions". Nobody noticed for a month, which is the part
worth thinking about.

**Root cause.** `run_benchmarks.py` computed the scalar as

```python
"instruction_count": int(sum(isa_ops.values()))
```

where `isa_ops = count_ops(isa, "npuisa")` and `count_ops` is

```python
dict(Counter(re.findall(rf"{dialect}\.[a-z_0-9]+", mlir_text)))
```

A regex over the printed IR. It inflates the count two ways, and both are
visible in one line of LeNet IR:

```
%0 = npuisa.dma_load %x : (tensor<4xf32>) -> !npuisa.buffer<tensor<4xf32>>
```

That is one instruction. The regex sees `npuisa.dma_load` and `npuisa.buffer`,
because a type is text too and `!npuisa.buffer<...>` matches the same pattern as
an op mnemonic. Separately it counts `npuisa.const`, which the encoder turns
into a DRAM region and never emits into the instruction stream at all. For
LeNet the two together turn 21 into 70.

The simulator has been reporting the real number in `stats.instructions` since
it was written, and `simulate()` already parsed and returned that JSON. The
harness had the right answer in a local variable and used the wrong one.

**Why it survived.** Two committed artifacts disagreed and no test compared
them. The baseline recorded the truth in U0 precisely so drift would be visible,
but it compared each run against the previous run, never against the results the
report publishes. The disagreement was between two files that nothing read at
the same time.

**Chosen fix.** Take the scalar from the simulator, and refuse rather than fall
back:

```python
if "instructions" not in stats:
    raise RuntimeError(...)
"instruction_count": int(stats["instructions"]),
```

The fall back matters. A silent `except KeyError: use the regex` would reinstate
the defect the first time the stats format shifted, and it would do it quietly,
which is exactly how the number got published the first time.

`count_ops` and the `npuisa_op_counts` histogram stay. The histogram is real
data and Part 8 builds on it; only its use as a scalar was wrong. Its docstring
now states both failure modes so nobody reaches for `sum(...)` again.

**Verification.** All six cells regenerated, and every one now equals the
recorded baseline for its cell: 28, 28, 25, 31, 21, 29 against baseline entries
of the same. Four new tests: `test_instruction_count_comes_from_the_simulator`
asserts the value is not the regex sum and is smaller than it,
`test_results_agree_with_the_recorded_baseline` ends the committed
contradiction, `test_count_ops_is_not_an_instruction_count` exercises both
inflation modes on a three line dump so the reason is executable rather than
only written down, and `test_readme_table_matches_the_results` pins the hand
written table to the generated numbers so it cannot drift again. The README
objdump excerpt was regenerated from a real run and confirms 21.

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

---

# v3 rebuild starts here (2026-08-19)

Everything above this line is v1 and v2 era work against the old source tree,
which the commit at the head of `phase/p0-foundations` removed from the working
tree and left in history. Everything below is the v3 rebuild, which releases as
`v2.0.0`. I keep both in one file on purpose: the debug report groups entries by
theme rather than by era, and a defect I hit in v1 and hit again in v3 is worth
more read next to itself than filed in a second document.

## 2026-08-19 Phase 0: three traps in the scaffold, none of them in the compiler

**Symptom.** None of the three showed up as a failing test. All three were
things that looked configured and were not, which is the failure mode a
skeleton is most exposed to, because at P0 there is no code whose behaviour
would contradict a bad setting.

**lit will not pass an empty suite.** The build specification says `ninja -C
build check-npu` passes on an empty suite at P0. It does not. llvm-lit prints
`did not discover any tests for provided path(s)` and exits 2, and
`--allow-empty-runs` does not change that: `llvm/utils/lit/tests/selecting.py`
in the pinned LLVM contains a test asserting the flag deliberately does not
suppress that error, which is as clear a statement of intent as upstream ever
gives. The flag only covers a suite that found tests and then filtered them all
away.

I could have made the target pass by making it not run lit. I did the opposite
and gave the suite one real test: `test/Smoke/npu-opt-roundtrip.mlir` parses and
prints a two operation module through `npu-opt` and checks the output with
FileCheck. That is a test of the thing P0 actually delivers, which is the
harness, and it means the gate is met by something that ran. If P1 opens with a
broken dialect, this test having passed at P0 says the breakage is in the
dialect and not in the build. Logged as D-0001.

**The pytest PYTHONPATH wiring was set in a key nothing reads.** Section 3.3
insists the MLIR bindings path be wired in three places and warns that missing
it fails every Python test at import for a reason unrelated to the code under
test. I wired it as `env = [...]` under `[tool.pytest.ini_options]`, which is
the obvious spelling and is wrong here: `env` belongs to the `pytest-env`
plugin, this environment does not have it, and pytest treats an unknown config
key as a warning and carries on. The wiring would have passed review and done
nothing.

`test/Python/conftest.py` does it instead, with no plugin needed, and resolves
the path from `MLIR_PYTHON_PACKAGES_DIR`, then from this repository's own CMake
cache, then from the default. Reading the cache is the part I would keep if I
kept only one: it means pytest and lit take the path from the same value the
configure step computed, rather than from two hardcoded strings that have to be
kept in step by hand. I proved it by running pytest with `PYTHONPATH` unset in
the environment, and by writing `test_bindings_wiring.py`, which imports
`mlir.ir` and builds a module so that a future break names the wiring instead of
blaming whichever test ran first. Logged as D-0002.

**A lock file that locks nothing.** `pip freeze` writes torch as
`2.13.0+cpu`, because the installed build is the CPU one. That local version
lives on the PyTorch CPU index and has never been on PyPI, so a clean install
from the frozen file fails outright with `No matching distribution found`. The
gate asks for proof that a clean venv installs from the lock, and the first
attempt at that proof is what found this; the file had looked fine.

The fix is `--extra-index-url https://download.pytorch.org/whl/cpu` written
into the lock file itself, so the index travels with the pins. I want to record
why I did not take the easier route of deleting the `+cpu` suffix: that makes
PyPI resolve, and what it resolves to is the CUDA build. It would have turned a
loud failure into a quiet one, changed what reference generation runs on, and
pulled about two gigabytes of CUDA wheels onto a machine whose GPU this project
deliberately does not use. Logged as D-0003.

**A fourth thing, smaller, worth one line.** My configure time check for the
MLIR Python bindings looked for `mlir/__init__.py` and warned that a perfectly
working install was missing. The bindings are a namespace package and ship no
`__init__.py`. The check looks for `mlir/ir.py` now.

**Verification.** `cmake` configures clean, `ninja -C build -j6` builds
`npu-opt`, `./build/bin/npu-opt --help` exits 0, `ninja -C build check-npu`
reports 1 of 1, `python -m pytest -q` reports 2 passed with `PYTHONPATH` unset,
`bash scripts/dash-lint.sh --self-test` meets 8 of 8 expectations,
`bash scripts/dash-lint.sh` is clean over the tree, and `reuse lint` reports
33 of 33 files carrying copyright and licence information.

## 2026-08-19 Phase 0: reconciling the upgrade line, and a push that would not push

**Symptom.** Three separate things, none of which was a compiler bug and all
three of which were in the way of the P0 gate. The reconcile itself went
quietly. Then `git push origin main` hung, indefinitely, with no output and no
prompt. Then, separately, the opset probe printed a number I had been told to
expect, which is its own small trap because a confirmed prediction is the
easiest kind of result to stop reading carefully.

**The reconcile, and the four commits that netted to nothing.** The clone
recipe of Section 0.5 leaves `upgrade/u4-measurement-integrity` checked out,
because that is what the source repository had checked out, and it leaves `main`
as a local branch materialised while `origin/*` still pointed at the local
source. Merging u4 into `main` in the clone was the substantive half. The other
half was four commits I had forgotten existed: `Update README.md`, four times,
made through the GitHub web editor months apart. Two of them added text and two
of them took the same text back out, so the net diff of all four against the
merge base is empty.

That is worth a paragraph rather than a footnote, because an empty net diff is
exactly the shape of change that a reconcile can silently drop and nobody
notices for a year. I merged them rather than resetting past them, at `52ed1da`,
which keeps the four commits reachable and makes the reconcile a merge of the
real remote history instead of an assertion that my local `main` was the truth.
The tree after the merge is byte identical to the tree before it, and that
identity is the check I ran rather than reading the four diffs and deciding they
looked harmless.

**Root cause of the hang: git-credential-manager on the Windows side.** WSL2
inherits the Windows git credential helper, and this environment resolves it to
`git-credential-manager`. That helper wants to raise a GUI prompt. From a
non-interactive WSL shell there is nothing to raise it into, so it waits, and
`git push` waits with it, forever, with no output at all. The failure gives no
indication that credentials are what it is stuck on, which is the reason this
cost more than it should have: a hang with no message reads as a network problem
long before it reads as an authentication problem.

**Options I considered.** Storing a token in `~/.git-credentials` was the
obvious one and I did not take it: that writes a long lived credential to disk
in plaintext for the sake of one push. Reconfiguring the global credential
helper was the second, and it edits state outside this repository to fix a
problem inside it. I wanted something scoped to the command.

**The fix.** Pipe a token from `gh` into an inline credential helper that lives
only for the duration of the push:

```
git -c credential.helper='!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f' push origin main
```

with `GH_TOKEN` set from `gh auth token` in the same command. Nothing is
written to disk and the global configuration is untouched.

Two details cost me a second and third attempt, and both are about the token
string rather than about git. First, capturing `gh auth token` through
PowerShell gives back a **UTF-8 byte order mark** on the front of the string.
Git sends it verbatim, GitHub rejects the credential, and the error is a plain
authentication failure that says nothing about an invisible three byte prefix.
Second, the same capture leaves a **trailing carriage return**, because
PowerShell line endings are CRLF and the shell on the other side does not strip
it. Same symptom, same uninformative message. Stripping both, the push went
through first time.

I am recording the two stripping steps as separate facts because they are
separate bugs with an identical symptom, and fixing one and not the other looks
exactly like the fix not working.

**The opset probe, run rather than assumed.** Section 3.3 gives the probe and
Section 0.3 predicts it resolves to 23. I ran it in `~/npu-venv` on this machine
on 2026-08-19 rather than copying the number, which is what the specification
asks for and is also the only thing that makes the pin a fact about this
machine. Output: torch 2.13.0+cpu reports an exporter maximum of 23, onnx 1.22.0
reports a checker ceiling of 27, onnxruntime is 1.27.0. Walking downward from
27, opset 27 was rejected by the checker or the runtime with `Fail`, and 26 was
accepted by both. So the pin is `min(23, 26) = 23`, bound by the exporter.

The predicted number was right and the interesting part was not the number. It
was the gap between onnx's declared ceiling of 27 and the 26 that actually round
trips: the installed onnx announces support for an opset that its own checker or
the installed runtime then refuses. Had I copied 23 out of the specification I
would have got the same pin and would not have learned that the tools disagree
with themselves one opset above where I am working. Recorded as
`docs/adr/0002-onnx-opset-pin.md`.

**Verification.** `git remote get-url origin` prints the GitHub URL and not a
local path; `main` is at `52ed1da` and matches the remote; all six upgrade
branches are published; `git -C ~/npu-mlir rev-parse HEAD` still reads
`99408bc14b4f6331ce03ebf1dc0aecce1529afa8` with only the untracked
`upgrade_parts/` dirty, which is the frozen fallback exactly as it was. The
opset probe output is quoted verbatim in record 0002.

## 2026-08-19 Phase 0: the owner raised the WSL2 memory cap to 15 GB

Not a defect, a deliberate environment change, recorded because three project
documents state the old value and because the ceiling is the one environmental
constraint everything else plans around.

The guest memory cap in the host `~/.wslconfig` went from 12 GB to 15 GB at my
explicit instruction, mid Phase P0. The 12 GB figure dated from 2026-07-14,
when running WSL2 uncapped crashed the machine outright; the cap is the crash
mitigation, and the specification wrote it down as non negotiable on that
history. Raising it to 15 GB trades about 3 GB of Windows headroom for guest
capacity on a 31.7 GB host. The swap (8 GB), `autoMemoryReclaim=gradual`,
`pageReporting`, and the 28 processors are all unchanged, and the guest
verified the new ceiling after `wsl --shutdown` with `free -h` reporting 14Gi
total.

What deliberately did not change: the parallelism numbers. `-j6` for memory
hungry compiles and `LLVM_PARALLEL_LINK_JOBS=1` for any forced LLVM rebuild
were measured under the 12 GB cap, and nothing has re-measured them at 15 GB,
so they stay in force as the known safe settings rather than being scaled up
by arithmetic. If the host starts paging under combined load, the revert is
one line in `~/.wslconfig` and a `wsl --shutdown`.

Updated in the same commit: `docs/adr/0001`, `docs/adr/0003`, `docs/BUILD.md`,
and one comment in `.github/workflows/ci.yml`. The v1 era entries above that
cite 12 GB are history and stay as written.

## 2026-08-19 Phase 0: the skeleton went red twice before it went green, and both reds were real

The gate wants CI green with the image pulling, and the sequence that got there
is worth its URLs because it is the proof of failure discipline arriving early
and uninvited. The image published on the first attempt (run 32205653261,
roughly two hours on the hosted runner). The first run of build-and-test with
that image then died on its opening step: inside a job container the runner
falls back to `sh` for run steps, dash rejects `set -o pipefail`, and every
step in the job had been written against bash and had never once executed under
the shell it actually got. That is D-0004, red at
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32213043383>,
fixed by a job level `defaults.run.shell: bash`.

The next run got through configure and the build and then failed check-npu:
`build/bin/llvm-lit: not found`. The local build links the LLVM build tree,
which carries `llvm-lit`; the image carries an install tree, which does not,
and the pip installed lit has to be named with `LLVM_EXTERNAL_LIT`. That is
D-0005, red at
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32213209291>.

Green, with check-npu 1 of 1 inside the container and every guarded off step
printing that it is off:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32213397267>.

Neither defect was in the compiler, both were in the difference between the
environment the code was written in and the environment it ran in, and both
were caught by the skeleton on its first day. Cheap net, first catch.

## 2026-08-19 Phase 1: the dialect, and three things that were harder than the design implies

The `npu` dialect landed in three commits: the operations with their verifiers,
the tiling interface as external models, and the tests with the generated
reference and the reachability script. Fourteen operations, three attributes,
one shared piece of shape arithmetic, and no types of its own. What follows is
what did not go the way the design's prose suggested it would.

**The shared windowed arithmetic was the easy part, and the drop rule was not.**
The design says the opset 19 pooling formula gains a rule: with `ceil_mode = 1`,
a sliding window whose first element would start inside the right padded region
is dropped. It says the rule is routinely missed. What it does not say, and what
took a while to see, is that the rule is nearly untestable by accident. Most
parameter sets where the drop fires are parameter sets where the ceiling and the
floor already agree, so the drop takes the ceiling's answer back down to the
floor's and the whole rule is invisible. A test written without noticing that
would pass against an implementation that had never implemented the drop at all.

The case in `ops.mlir` is therefore chosen rather than found: input 6, kernel 2,
stride 3, pads 0 and 1. The ceiling gives 3, the floor gives 2, and the drop
takes the ceiling's 3 down to 2. Beside it sits input 7 with the same kernel and
stride and no padding, where the ceiling gives 3, the floor gives 2, and the drop
does **not** fire because the last window starts at 6 and the input runs to 7.
The pair is what pins the rule: an implementation without the drop passes the
second and fails the first, and an implementation that always drops fails the
second. The arithmetic for both is written into the test file as a comment,
because a test that asserts a shape without saying where the shape came from is
a test nobody can check.

**Two ODS features the design would have let me use turned out to be wrong for
this dialect, and both failed at link time rather than at their cause.**
`useDefaultTypePrinterParser` generates declarations for `parseType` and
`printType`; this dialect defines no types, so nothing defines them, and the
build gets an undefined symbol at the final link of `npu-opt` with no indication
that a one line TableGen flag caused it. And `SingleBlockImplicitTerminator`
builds a terminator for an empty region by calling `YieldOp::build` with no
operands, which does not exist and should not: `npu.yield` always carries the
value it yields, and a fused region with an implicitly created empty yield would
be a region yielding nothing while its operation has a result. Both are now
`SingleBlock` plus an explicit verifier rule, and both have their reason written
in the source beside them, because the next person to reach for the convenient
flag will reach for it for the same reason I did.

**The unit tests do not use `add_mlir_unittest`, and that is not a style
choice.** `add_mlir_unittest` wraps LLVM's `add_unittest`, which lives inside
the LLVM build and is not exported to an out of tree project, so it is simply
not a command here. What an out of tree project against an LLVM *build tree* does
have is the exported `llvm_gtest` target and the bundled gtest headers under
`third-party/`. Against an installed LLVM *prefix*, which is what the CI image
is, it has neither. So the unit test subdirectory is guarded on the target
existing and the configure log says which of the two it found. The consequence is
real and is not hidden: `NPUTilingTests` builds and passes locally and is not
built in CI at all. That is the honest state at P1 and the activation table
already keeps every GoogleTest binary guarded off until its own phase.

**The reachability check found its own first defect before it checked anything
else.** Switching the CI step on meant reading its guard, and the guard named
`include/npu/NPUOps.td`, a path that has never existed in this repository. The
step had been printing "OFF until P1" since P0 and would have gone on printing it
forever. That is D-0006, and the general lesson is worth more than the fix: a
guard whose condition can never be satisfied is a check that has been deleted
rather than deferred, and the two are indistinguishable from the run log, because
both are green and both say the step is waiting. The only way to tell them apart
is to prove the step red on the day it activates, which is what the proof of
failure discipline already asks for and what I did for all three of the
reachability check's rules.

## 2026-08-19 Phase 1: both new gates proven red on the day they activated

The activation table says a step that switches on gets broken once,
deliberately, because a step that has only ever been green is a step nobody
has tested. P1 switched on two steps, and the proof ran as three commits on a
scratch branch, phase/p1-activation-proof, deleted after use.

First fault: one word changed in the committed `docs/DIALECT_REFERENCE.md`.
Caught by build-and-test at the staleness step, which printed the diff and the
regeneration command. Red at
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32216846851>.

Second fault: the `Reachability: imported computation.` line removed from
`npu.relu`'s ODS description, with the reference restored. This one fault
turned both gates red at once, and that is the design agreeing with itself: the
classification lives in the description, the description feeds the generated
reference, so declassifying an operation cannot escape either check. lint
failed at check-reachability and build-and-test failed at staleness. Red at
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32216983456>.

Restore commit, everything green again:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32217118546>.

Both faults were caught by the job and step designed to catch them, which is
the finding worth recording: no fault had to be adjusted to fit its net.

In the same phase, the llvm-image workflow's push trigger is retired. It
existed to bootstrap the first publish, since workflow_dispatch only offers
workflows already on the default branch. After the P0 merge it fired a
redundant hour long rebuild of an image that already existed, whose only
effects would have been runner time and a moved tag digest, and whose
cancellation left a red X on the merge commit. Rebuilds are now a decision, a
button, not a merge side effect.

## 2026-08-19 Phase 2: two defects the tests found, and one the test harness hid

Picking up P2 from an interrupted session, the dialect core was committed at
`00cce3b` and the lit suites were on disk and untracked. Running them for the
first time turned up two defects in the committed code, and both are the kind
that survive review because the reasoning around them is right and only the
mechanism is wrong.

**A null memory space crashed the operand type predicate.** `memref<4x4xf32>`
with no memory space has a null memory space attribute, and the predicates were
written with `llvm::isa`, which dereferences its argument to reach its type id.
`npu-opt` died with a segmentation fault and no diagnostic. `isa_and_present`
answers false for null and is the right idiom. That is D-0008.

The symptom is worth separating from the cause, because the symptom is what
cost the time. Under lit this showed up as the whole of `invalid.mlir` failing,
not as one case failing, because a crash partway through a `-split-input-file`
run takes the process down and the remaining sections never execute. A file with
thirty independent cases reported as one failure whose message was a stack
trace. The instinct was to bisect the file; the faster route, which is what
worked, was to count the `// -----` separators the tool printed before it died
and read the section after that count. Worth remembering: with
`-split-input-file`, the number of separators echoed on stdout before the crash
is the index of the section that crashed.

**The overlap scan made two transfers in flight unrepresentable.** This is the
one I would not have found by reading. Rule 4 refuses an intervening operation
that does not implement `MemoryEffectOpInterface`, on the correct grounds that an
operation which cannot say what it touches has not said it touches nothing. But
`npuisa.await` declares no memory effect deliberately, per Section 8, so the
conservative branch caught the one operation guaranteed to be harmless. The
consequence: two asynchronous loads to provably disjoint destinations, with both
awaits after both producers, were rejected. That shape is not an edge case, it is
the double buffering of Section 5.1, which is the entire reason the asynchronous
form exists. The verifier was wrong about its own reason for existing. That is
D-0009.

What actually surfaced it was a test written for a different rule. I was pinning
the canonicalization's *negative* half, the cases where an async operation must
not fold back to the synchronous form, and one of those cases is two transfers in
flight. It never reached the fold, because it never parsed. The general point,
and the reason this is in the log rather than only in the defect file: the rule
had been tested only in the shape where a compute instruction sits between a
transfer and its await, and every test of it passed. A verifier can be
comprehensively tested against the configuration its author had in mind and still
refuse the configuration the feature was built for. Writing the negative cases for
an unrelated rule is what covered the gap, which argues for writing them even when
the positive cases look complete.

**A test harness trap that cost a debugging round.** I drew the section banners
in `canonicalize.mlir` with lines of dashes, matching the style of the other
suites. `-split-input-file` looks for a comment line of dashes as its separator,
so every banner silently became a split marker and the checks after each one
landed in the wrong section. The failure presented as a `CHECK-LABEL` not finding
a function that was plainly in the file. The banners in that file are drawn with
`=` now and the reason is written at the top of it.

Two smaller findings, recorded because each is a claim in a test that is not what
it appears to be. `CHECK-SAME` continues a match on the *same output line*, and
MLIR prints an operation on one line, so a `CHECK` ending in `%{{.*}}` consumes
to end of line and leaves the `CHECK-SAME` nothing to match. Worse, the in place
relu test used `%[[X:.*]]` to assert that the input and the destination are the
same value, and the greedy capture swallowed the rest of the line, so the check
that was supposed to prove aliasing was asserting nothing at all. Both are now
`[^ ]*`. And `npuisa.concat`'s empty variadic form cannot be spelled in the
custom assembly syntax at all: `ins()` and `ins( : )` are both parse errors,
because the format wants operands and types around a literal colon and with no
operands there is nothing on either side. The verifier rule that rejects an empty
concatenation is still right, since a pass building the operation programmatically
can produce one, but it is reachable only through the generic form, and that is
how the test spells it now.

**The CI gtest debt is paid, and the image republish covers P3 as well.** P1 left
this open: the unit test binaries link `llvm_gtest`, which an LLVM *build tree*
exports and an installed *prefix* does not, so they were built locally and not in
CI. P2 activates `NPUInterfaceTests`, so it had to be solved. The fix is a search
with a fallback in the top level CMakeLists, preferring the build tree's bundled
gtest when present and falling back to `find_package(GTest)` otherwise, with both
paths ending at the same two variables so no test CMakeLists knows which was
taken. Both branches are proven to build and run the same 18 tests locally, the
fallback by configuring against a deliberately nonexistent third-party directory.
`docker/Dockerfile.llvm` installs `libgtest-dev` into the final image for the
fallback to find; on this base it ships prebuilt static libraries and a CMake
package config, which older Debian derivatives did not, so the image's smoke test
checks for the config file rather than assuming it.

Since publishing the image is an hour of runner time, the same revision also
turns `MLIR_ENABLE_BINDINGS_PYTHON` on, which P3 needs and which the P0 notes had
scheduled as a P3 deliverable. One republish now instead of two. The bindings land
at `/opt/llvm/python_packages/mlir_core`, which is the directory to put on
`PYTHONPATH`; the `mlir` package itself is one level below that, because MLIR's
`MLIR_BINDINGS_PYTHON_INSTALL_PREFIX` defaults to a path ending *at* the package
directory rather than at the directory to import from. Getting that level wrong
produces an `ImportError` naming a package the image does contain, so both paths
are written into the Dockerfile and the smoke test imports `mlir.ir` and
constructs a `Context` rather than merely checking that a directory exists.

## 2026-08-19 Phase 2: an image republish with one defect, then both new gates proven red

Closing P2 took one image republish and one proof cycle, and both left evidence.

The republish came first, because the activation table switches NPUInterfaceTests
on at P2 and the P0 image could not build a gtest target at all. The first
attempt (run 32222527819) died at CMake configure inside MLIRDetectPythonEnv:
the builder stage had the interpreter but not python3-dev, and FindPython3's
Development component wants the headers. That is D-0010, invisible at P0 for
the boring reason that the bindings were off and nothing asked for headers. One
package name later the rebuild went through (run 32223127919) and the tag moved
to sha256:43f2f9d6, an image that carries googletest, the MLIR Python bindings
at /opt/llvm/python_packages/mlir_core, and gcovr. The dev image follows it by
digest, and the P0 digest stays resolvable for reproducing the earlier runs.

With the image in place the branch went green across all four jobs (run
32245240094), which is the first CI run in this repository to execute the
interface tests and the coverage job for real.

Then the proof, per the activation discipline: one wrong expectation in
InterfaceTest.cpp, a test failure rather than a build break, pushed on a scratch
branch. One fault, two nets, both red in run 32245632105: build-and-test at the
NPUInterfaceTests step, coverage before it reported a number. That second red is
the honest form of the coverage proof at P2, because at a threshold of zero the
threshold arm cannot fail and perturbing it would prove nothing; the
threshold-gate proof belongs to P8, where the thresholds become real. The revert
went green in run 32246428660 and the scratch branch is deleted.

One process lesson closes the phase. A session working from a stale local main
concluded P1 had never merged and rewrote the merge plan around that; the fetch
showed P1 merged all along. A merge conclusion is drawn against origin/main
after a fetch, never against a local ref nobody has pulled.

## 2026-08-19 Phase 3: the frontend, and four things the exporter decided for me

Four of this phase's decisions were not made by reading Section 11. They were
made by exporting a model, looking at what came back, and finding that the
document's premise did not hold on this toolchain. Recording all four together
because they have the same shape: a rule that reads as arbitrary in the
specification turns out to be describing a graph the exporter does not produce.

**The exporter writes `count_include_pad = 1` on every `AveragePool`.** Section
11 says to reject `count_include_pad = 1` with a diagnostic naming the node, and
the reason it gives is that this project's kernel divides by the contributing
count. Implemented literally, that rejects every average pool the dynamo
exporter can emit, pads or no pads, because it sets the attribute
unconditionally. The two settings disagree only when a window overlaps the
padded region, so the rule became: refuse `count_include_pad = 1` when any pad
is non zero, accept it when every pad is zero, and say which in the diagnostic.
That is not a softening. The acceptance condition is checked, it is the
condition under which the two behaviours are provably identical, and both halves
have a test.

The same shape again with `Reshape`: the exporter writes `allowzero = 1`
unconditionally, and `allowzero` only changes anything when the target shape
contains a zero. Same resolution, same reasoning, same pair of tests.

**`AdaptiveAvgPool2d` does not export as `GlobalAveragePool`.** Section 15 puts
global average pooling in the depthwise separable block. On torch 2.13 every
spelling of it, `nn.AdaptiveAvgPool2d((1, 1))`, `F.adaptive_avg_pool2d`, and
`x.mean(dim=(2, 3))`, lowers to a `ReduceMean` node, which is not in this
project's operator set. `nn.AvgPool2d` with a kernel equal to the spatial extent
does export as `AveragePool` and computes exactly the same thing, so the block
keeps its purpose and the `GlobalAveragePool` converter got its suite model in
the conv plus batch norm stack instead. The alternative was hand building a
third model, which would have cost the suite its only torch exported grouped
convolution.

**The `Mul` that Section 15 calls a rank 1 per channel scale exports as a rank 4
initializer.** Written in PyTorch as `scale.reshape(1, -1, 1, 1)`, which is the
only way to write it, constant folding turns the reshape into an initializer of
dims `[1, 8, 1, 1]`. A carve out that matched only a literally rank 1
initializer would have expanded every per channel constant in the suite and
fired on nothing. So the carve out is defined by what a constant broadcasts as
rather than by the rank it was stored with.

Which led straight into the phase's worst near miss, D-0014. Having widened the
match, I widened it one shape too far and accepted `[C]` as well. ONNX
broadcasting aligns from the trailing axis, so `[C]` against `N x C x H x W`
broadcasts over the width and not over the channels, and on any model where the
channel count and the width are equal that would have imported a per column
vector as a per channel one. Legal IR, passes every verifier, wrong numbers, and
nothing would have caught it before the end to end comparison at P8. The
regression test is written on a `1 x 4 x 3 x 4` activation on purpose, where the
two readings differ only if you know the rule.

**The specification's own layers disagreed, and the dialect moved.** Section 11
keeps a channel shaped constant unexpanded so that `-npu-fuse-bias` has a
channel shaped addend to guard on; Section 15 puts the same carve out on a per
channel `Mul`. P1's `NPUOps.td` had instead required both operands of `npu.add`
and `npu.mul` to have the result shape, with the carve out becoming a bias
operand on the consuming convolution. That reading fails both cases: folding the
addend into the convolution at import leaves the fusion pass nothing to fuse,
which is the outcome the carve out exists to prevent, and a per channel scale
has no bias operand anywhere to be folded into. So the verifier relaxed, to a
rank 1 rhs of the result's channel extent and nothing else, with only the rhs
allowed to be the broadcast side so there is one spelling for the pass to match.
D-0012 and `adr/0005`.

## 2026-08-19 Phase 3: a suite that reported every test passing and then segfaulted

**Symptom.** The first full run of the new pytest suite printed
`13 failed, 122 passed` and then exited 139. The thirteen failures were fixture
problems and were expected to be fixable. The 139 was not in the summary at all.

**Root cause.** `ModuleBuilder` entered MLIR's `InsertionPoint` context in
`begin_function` and left it in `end_function`, which is the happy path only.
About a third of this suite's tests exist to make a converter raise, and every
one of those unwound out of the builder without reaching `end_function`, leaving
the insertion point on MLIR's thread local stack pointing into a module that was
then freed. Nothing failed at the time. The crash came at interpreter shutdown,
after the last test had reported a clean expected failure.

**Why it is worth an entry rather than only a defect number.** The failure mode
is a suite whose summary line says everything passed and whose exit code says
the process died, and those are the two things a reader and a runner
respectively look at. A CI step that grepped the summary would have called it
green. The reason this project's would not is that the step checks the exit
code, which is also what makes the exit 5 rule for an empty collection worth
writing down: both are cases where the exit code is the only honest signal.

**Chosen fix.** Two `contextlib.ExitStack`s, an outer one for the context and
the location and an inner one for the function's insertion point, with
`__exit__` closing both unconditionally. `ExitStack.close` is idempotent, so
`end_function` closing the inner one on the happy path costs nothing. Verified
in both directions: patched back to the old teardown the suite exits 139 on 74
passing tests, restored it exits 0 on the same 74. D-0013.

**One smaller trap from the same session, for the next person who adds a package
directory.** `reuse lint` failed on the `__pycache__` files under the new,
entirely untracked `python/` directory, even though `.gitignore` covers
`__pycache__/` and `git check-ignore` agrees. Staging the directory made it
pass. reuse's gitignore handling does not reach inside a directory git has never
seen, so a new package root will fail reuse until its first `git add`. Not a
defect in this repository, but it costs ten minutes if you go looking for a
missing SPDX header that is not missing.

## 2026-08-19 Phase 3: how the frontend emits IR, and the one hazard it buys

The `npu` dialect is C++ only and there are no Python bindings for it, which
Section 5.1 does not mention because it draws the importer and the dialect
without saying how the first reaches the second. Three ways exist and the record
is `adr/0004`; the chosen one is unregistered operations through the MLIR Python
bindings with `./build/bin/npu-opt` as the verification gate, made mandatory
rather than optional, so the text `npu-opt` prints is what `import_model`
returns.

The reason to write it in the log as well as in the record is the hazard, which
took a probe to find and would not have been found by reading. MLIR promotes the
inherent attributes of a registered operation into its properties when it parses
a generic form, and an attribute whose name matches no inherent one is kept as a
**discardable** attribute rather than rejected. So `"npu.conv2d"(...) {strydes =
array<i64: 9, 9>}` parses, prints, verifies, and runs with `strides` at its ODS
default. Nothing anywhere says a word about it.

The probe that found it is worth repeating on any future dialect: print the
module generically through `npu-opt --mlir-print-op-generic` and look at where
each attribute landed. Properties print as `<{...}>` and discardables as a bare
`{...}` after them, so the distinction is visible in text and the check is a
parse rather than a judgement. The importer runs that scan on every module and
refuses any `npu` operation carrying a discardable dictionary. No operation it
emits has a legitimate one, so the rule is total and needs no exceptions list.

Two smaller findings from the same probe. Optional and variadic operands need no
`operandSegmentSizes` on any operation of this dialect, because each has at most
one variadic group and MLIR infers the split; passing one anyway leaves it
sitting in the discardable dictionary, which is how the hazard was noticed in
the first place. And `str(module)` prints without debug information, so the
`NameLoc` every operation carries was being dropped on the way to `npu-opt` and
the round trip was handing back file and line locations pointing at stdin. The
builder asks for `get_asm(enable_debug_info=True)` and the round trip asks for
`--mlir-print-debuginfo`, which is what makes Section 11's requirement that
every operation carries its ONNX node name something the next stage can actually
see.

## 2026-08-20 Phase 3: three activation proofs, and the one that proved nothing first

P3 switched on two steps, mypy and pytest, and the pytest step carries a
separate claim, that an empty collection exiting 5 is read as failure. Three
proofs, on a scratch branch since deleted, each fault built so exactly one net
could catch it.

mypy: the opset pin annotated Final[str] while assigned 23. pytest, black and
ruff have no opinion about an annotation contradiction; only mypy went red, in
the lint job, run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32307339884>.

pytest: one suite expectation turned wrong, the resnet block asserted to hold
two Mul nodes where the generator emits one. Only the pytest step went red, in
build-and-test, run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32307711201>.

The exit 5 arm took two attempts, and the first is the lesson. The handoff
recipe pointed testpaths at a directory with no tests, verified locally with a
bare pytest invocation. CI stayed green, run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32307981414>,
and the green was correct: the CI step passes test/Python on the command line,
and a command line path makes pytest ignore testpaths entirely, so the
perturbation never reached the step it was meant to break. A proof rehearsed
under a different invocation than the gate uses proves the rehearsal, not the
gate. The second attempt deselects every test through an unsatisfiable default
marker expression, which travels through any invocation, was verified locally
to exit 5 through the exact command CI runs, and went red where it should, run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32308435681>,
with the step naming exit 5 and what it means. The restore went green in run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32308808081>.

One mechanical footnote: reverting the second perturbation also reverted that
commit's bundled restoration of testpaths, so the restore took two commits. A
proof commit that fixes one thing and breaks another makes its own revert a
half measure; keep perturbation commits to exactly one change.

## 2026-08-20, phase P4: lowering the npu dialect to npuisa

### The one-shot-bufferize attempt, and what it actually settled

The roadmap entry for this phase does not let me choose a mechanism without
measuring first: attempt One-Shot Bufferize, record the outcome either way, and
if two space DMA insertion is not expressible through it, write the evidence
into a decision record. So I ran it, five ways, on one convolution and one relu
in the shape the frontend emits, and the runs are in
`docs/adr/0006-lowering-mechanism-and-the-bufferization-attempt.md` with their
output.

What I expected to find was a configuration problem. What I found was a
signature:

    using DefaultMemorySpaceFn = std::function<std::optional<Attribute>(TensorType)>;

That hook takes a tensor **type** and returns one memory space. A function
argument that lives in DRAM and a scratchpad temporary of identical type are the
same argument to that function, so they get the same answer, always. The
infrastructure's model is one space per value. This machine's model is that a
value entering the compute units exists in two spaces with a transfer between
them. No setting of the first expresses the second, and `must-infer-memory-space`
says so out loud: "could not infer memory space" on the first `tensor.empty`.

The thing I nearly got wrong was concluding it from run 1. Run 1 fails with "op
was not bufferized" because no `npu` operation implements
`BufferizableOpInterface`, and that is a fixable complaint, not an answer. It
would have been easy to stop there and write a record saying the interface was
not implemented, which is true and beside the point. The answer needed runs 3
and 4: run 3 to see what the infrastructure produces when it is allowed to
proceed, which is memrefs in the default space with no DMA anywhere, and run 4
to see the hook refuse. Implementing the interface on all twelve operations
would have reproduced run 3 with the `npu` operations gone, because
`getBufferType` returns one buffer type per tensor value and the lowering needs
two buffers and a copy for one value.

Recorded because the failure mode generalises: an infrastructure that refuses
your input at the first step is telling you about step one, and the question you
came with is usually answered three steps later.

### A crash found by a test that was asking about something else

Writing the refusal cases for the lowering, I wanted an operation whose result
type has no memory space, and reached for a reshape of a `tensor<?x4xf32>`. The
tool aborted, inside `ReshapeOp::verify`, on an assertion in `getNumElements()`,
with a stack trace pointing into LLVM and no diagnostic at all.

`NPUTypes.td` has said since P1 that "a dynamic dimension is refused at the type
level rather than by a verifier". The three constraints under that sentence were
`RankedTensorOf`, which requires a static rank and says nothing about extents. So
the sentence was a comment describing a rule nothing enforced, and it had been
sitting there for three phases with tests passing over it, because the frontend
refuses a dynamic extent at import and nothing else in the tree writes one.

The fix is `StaticShapeTensorOf`, which is what the sentence always claimed and
what the `npuisa` side has done since P2. That is D-0015, and it is a P1 change
made at P4, so a reviewer should look at it as one. I did consider recording it
and leaving it: it is not this phase's code and the phase was already large. I
fixed it because the alternative is a known crash left in the tree behind a
sentence that says it cannot happen, and because the fix is three lines plus a
regenerated reference. The three regression cases abort against the old spelling
and are diagnosed against the new one.

The generalisable half: a comment asserting an invariant is worth grepping for
the enforcement of. This one had the enforcement in the same file, one word away
from being right.

### Four transfers nobody asked for, found by reading output rather than by a test

The batch norm decomposition consumes gamma, beta, mean and variance at rewrite
time and computes a multiplier and an addend from their values. It left the four
`npu.constant` operations behind with no uses. Everything passed. The program was
correct.

Then I read the output, and there were seven `npuisa.dma_load` operations where
there should have been three. The conversion does not care whether a constant is
used: it is an illegal operation, the pattern fires, and out comes a DRAM buffer
and a transfer to bring it on chip. Nothing downstream removes them, because a
transfer has memory effects by design and is not dead code a canonicalizer may
delete. They would have reached the encoder and been executed.

This is worth more than its two line fix. Section 8 names exactly three
permitted producers of DMA so that a fourth is recognisable as a defect, and I
had produced a fourth **inside the first**, in the pass that is meant to be the
one place the invariant is established. No test caught it because no test asked
how much DMA there was, only what it looked like. That is what
`dma-boundaries.mlir` is for, and it is why the counting cases in it are written
with `CHECK-NOT` after the count rather than as a list of the transfers I
expected to see: a check that enumerates what should be there passes on a
program that also has something else.

D-0016, and the regression test is named after the claim rather than the
mechanism.

### A smaller trap, for whoever writes the next lit test

Two FileCheck failures in `lowering.mlir` cost more time than the pass did, and
both were the same mistake. `CHECK: npuisa.relu ins(%{{.*}} : memref<...>)`
followed by `CHECK-SAME: outs(...)` cannot match: the `.*` is greedy, each
CHECK-SAME is a separate match that must begin where the previous one ended, and
the first pattern has already consumed the rest of the line. Within one pattern
FileCheck backtracks and it works; across a CHECK-SAME chain it does not.
Bounding every SSA placeholder as `%{{[a-z_0-9]+}}` fixes it and reads better,
since the thing being matched really is a name and not arbitrary text.

The second was an ordering error rather than a regex one. I wrote
`CHECK: npuisa.const`, then `CHECK-COUNT-2: npuisa.dma_load`, on a function
whose first transfer is the argument load and therefore appears **before** the
constant. FileCheck matches in order, so only one load remained to be counted.
Checks encode an order as well as a presence, and on straight line output that
order is the program's.

## 2026-08-20, phase P5: the allocator, and what its two experiments measured

Two experiments land with the allocator and both were predicted before they were
run, in `experiments/predictions/p5-allocator-compile-time.md`, committed at
`4069bc2`, strictly before the commit carrying these numbers. Ground rule 15
makes the commit order the evidence. **One prediction was met and one was
wrong**, and the wrong one is more interesting, so it goes first.

**All numbers below are wall clock measurements on one machine**: WSL2 on
Windows 11, kernel 6.18.33.2, 28 logical processors visible, 14 GiB of guest
memory, an assertions enabled Debug build of LLVM 22.1.8. They are not a claim
about complexity and they are not comparable with a number from another machine.

### The compile time curve, and a prediction that was wrong

Section 13.1 asks for a synthetic function at 500, 1000, 2000 and 5000
operations so the growth curve is visible rather than a single point.
`experiments/compile_time_benchmark.py` is that benchmark. It reads the pass's
own wall time out of `--mlir-timing`, so parsing and printing are excluded;
both are linear with a large constant and at these sizes they dominate the
total, which would hide exactly what the benchmark exists to show.

| size | operations | buffers | pass s | total s | exponent |
|---|---|---|---|---|---|
| 500 | 500 | 249 | 0.0023 | 0.0136 | |
| 1000 | 1000 | 499 | 0.0046 | 0.0209 | 1.00 |
| 2000 | 2000 | 999 | 0.0096 | 0.0382 | 1.06 |
| 5000 | 5000 | 2499 | 0.0261 | 0.0870 | 1.09 |

Best of five runs per size. The exponent is `log(t2 / t1) / log(n2 / n1)`
between consecutive rows, and the mean over the curve is 1.05.

**I predicted 1.5 to 2.0, rising across the curve, and said that below 1.3 would
mean I had mismodelled the constant factors. It came out at 1.05.** So I had
mismodelled the constant factors, and the prediction stands unedited in the
prediction file with this entry as its answer.

The reasoning behind the prediction was not wrong about the code. Offset
assignment really does scan every already placed buffer for each new one, so
there really is a quadratic term, and it really does grow a hundredfold between
500 and 5000 where the linear term grows tenfold. What I got wrong was the
constants on either side of that. The quadratic term's inner loop is a struct
copy and two integer comparisons on a `SmallVector` that is already in cache.
The linear term's inner loop walks MLIR users, does `DenseMap` lookups keyed on
`Operation *`, and builds a `SmallVector` per buffer, in a build with
`_GLIBCXX_ASSERTIONS` on. Three million of the former is cheaper than five
thousand of the latter, and at 5000 operations the linear term is still winning.

That is worth writing down as a general lesson rather than as an excuse: **an
asymptotic argument about which term dominates is a statement about the limit,
and a benchmark at four sizes is a statement about four sizes.** The crossover
exists and is somewhere well beyond 5000 operations. Whether it matters is a
different question and the answer today is no: 5000 operations is an order of
magnitude larger than any model in the suite, and it allocates in 26
milliseconds.

The other three predictions in that section were met. Under 100 milliseconds at
5000: 26 ms. Total time dominated by parsing at every size: the ratio of total
to pass time falls from 5.9 to 3.3 across the curve, so parsing dominates
throughout while its share shrinks, which is what a linear parser against a
slightly superlinear pass should do. No spilling at any size: `spill_count` is
zero everywhere, since the chain's peak is two buffers of 256 bytes against a
mebibyte of budget.

**What I am not doing about it.** The obvious optimization is an interval tree
in place of the linear scan over placed buffers. It is not going in. Section
13.1 asks for the sweep line and says nothing about the placement's data
structure, the measurement says the term it would fix is not the one that costs
anything at any size this project compiles, and an unmeasured optimization is
how a phase turns into two. If P13's tiling pass makes functions an order of
magnitude longer, this benchmark is already committed and will say so.

### The fragmentation ratio per model, and a prediction that was mostly met

`experiments/allocator_fragmentation.py` compiles each of the seven suite models
through `-npu-lower-to-npuisa` and `-npu-allocate-scratchpad` under both
strategies at the default budget of 1048576 bytes, and reads the ratio off the
function. Ratio is `npuisa.scratchpad_bytes` over
`npuisa.scratchpad_peak_bytes`: the assigned high water mark over the sweep line
peak, which is the lower bound any placement could reach.

| model | pack bytes | pack ratio | interval bytes | interval ratio | peak |
|---|---|---|---|---|---|
| `conv_bn_relu_stack` | 6432 | 1.0000 | 7872 | 1.2239 | 6432 |
| `depthwise_separable` | 8192 | 1.0000 | 10816 | 1.3203 | 8192 |
| `dilated_stack` | 8036 | 1.0035 | 11812 | 1.4750 | 8008 |
| `inception_block` | 6848 | 1.0178 | 10112 | 1.5030 | 6728 |
| `lenet` | 194592 | 1.0002 | 194592 | 1.0002 | 194560 |
| `lenet_batched` | 200800 | 1.0000 | 200832 | 1.0002 | 200800 |
| `resnet_block` | 8480 | 1.0000 | 8512 | 1.0038 | 8480 |

Nothing spills at the default budget, so every number here is a placement result
rather than a spilling result, which is what makes them comparable.

Met: `pack` is never worse than `interval` on any model, which was the claim I
flagged as the interesting one because greedy by size is not optimal in general
and a program where the interval scheme wins is constructible. `pack` stays at
or below 1.05 everywhere, peaking at 1.0178 on `inception_block`. At least one
branching model shows `interval` at 1.10 or worse: three of them do, and
`inception_block` reaches 1.5030, which is a placement using half again as many
bytes as the program needs. Every model fits the default budget, and `lenet` is
190 KiB, which is the low hundreds of kilobytes I said.

Missed, in a small and instructive way: I predicted `lenet` and `lenet_batched`
at exactly 1.00 under both strategies because they are chains, and `lenet` comes
out at 1.0002 under both. The 32 byte difference is **alignment**, not
fragmentation. The peak is a sum of raw buffer sizes and the high water mark is
a sum of offsets rounded up to 64 bytes, so a buffer whose size is not a
multiple of 64 can push everything above it up by a few bytes the peak does not
count. `lenet` has six distinct buffer sizes that are not multiples of 64, at
24, 40, 336, 480, 600 and 4704 bytes, and only 32 bytes of rounding survives
into the high water mark, because the rest of it lands under buffers that were
going to be there anyway.

That is a real property of the metric and not a defect, but it is worth stating
plainly because the metric is the headline one: **the fragmentation ratio as
defined by Section 13.1 includes alignment padding.** Every published ratio has
a floor slightly above 1.0 that has nothing to do with the packing algorithm.
Separating the two would mean defining a second peak that rounds each buffer up
before summing, and I have not done that, because the ratio the three papers
report is the one Section 13.1 names and a second definition would be a second
number to keep straight. The size of the effect is on the record here: 0.02
percent on `lenet`.

### The pad order defect, and why the suite caught it three phases late

Running the suite through the allocator is what found D-0019: the `npuisa`
windowed verifier reordered its pads before computing the output extent, so
every asymmetric pad got a wrong implied extent and was refused. `dilated_stack`
would not lower at all.

The mechanism is in the defect log. What belongs here is the shape of the miss.
The formula lives in one shared helper for exactly the reason the helper's own
header gives: two copies would eventually disagree. There is one copy and both
levels call it, and the `npuisa` level still got a different answer, because the
disagreement moved from the formula into the **argument order at one call
site**. Sharing an implementation removes one class of divergence and leaves its
interface as the new place for the same failure to live.

The reason it survived from P2 is simpler and worse: every pad in every test was
symmetric, and under a symmetric pad the two orders produce the same numbers. A
test suite built from `[1, 1, 1, 1]` and `[0, 0, 0, 0]` cannot see this bug at
all. The model that carries asymmetric padding is `dilated_stack`, which Section
15 put in the suite precisely because it forces cases nothing else reaches, and
it did exactly that, three phases after it was written, in a verifier rather
than in the importer it was aimed at.

**The rule I am taking out of it: a parameter with an order needs a test whose
entries differ.** Symmetric fixtures are the default because they are easy to
read, and they are the reason an order bug can pass every test in a file.

### Why the activation proof gets rehearsed under the exact CI invocation

`NPUAllocatorTests` switches on at this phase, so Section 19.0 requires it be
broken once deliberately and shown red. The recipe is in
`docs/PHASE_STATE.md` and it is one line, which is the whole point.

The rule the recipe follows, and the reason for it: **the rehearsal has to run
the binary the way the CI step runs it, not the way a developer runs it.** The
step is `./build/bin/NPUAllocatorTests` with no arguments and `set -euo
pipefail` around it. A developer rehearsing with `--gtest_filter` on the one
test they broke proves that the test fails, which was never in doubt; what needs
proving is that a failure inside the binary reaches the step's exit code and
turns the job red. Those are different claims, and the second is the one the
activation table asks about. P3's handoff already recorded an activation proof
that proved nothing the first time it was performed, and this is the same hazard
wearing different clothes: a proof performed under conditions the real thing
does not share.

### The two things about MLIR that cost time this phase

**A pass option is not an integer.** An ODS option declared `int64_t` is an
`llvm::cl::opt<int64_t>`, and streaming one into a diagnostic picks the `char`
overload: an alignment of 48 printed as the character `0`. That is D-0017, and
the reason it is worth an entry rather than a shrug is that it is invisible at
the point of use, applies to every numeric option in the project, and produces a
message that actively misleads. Told the alignment is 0 when you passed 48, the
obvious conclusion is that the option was not read, and you go looking in the
pass manager.

**`memref.view` requires an identity layout on its result.** Section 8 says the
allocator materialises every offset as a `memref.view` over one flat
`memref<Nxi8>`, and that is exactly right until a buffer carries a strided
layout map, which is what an NHWC tensor lowers to. The verifier rejects it. The
answer is a view at the buffer's extents with a `memref.reinterpret_cast` on top
restoring the layout, which costs nothing at run time and is visible in the IR.
It is worth knowing before writing the pass rather than after, and it is the
kind of constraint that is documented in the operation's description and nowhere
a reader would look first.

## 2026-08-20 Phase 5: the activation proof that was caught by the wrong net first

The NPUAllocatorTests step switched on at P5, and the proof discipline asks for
it to be broken once and shown red. The first fault inverted the sweep line's
deaths before definitions tie break inside the pass itself, and it never
reached the step it was aimed at: check-npu caught it first, in run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32326641102>,
because the P5 lit suites assert byte offsets by value and a miscounted peak
moves them. The CI section says a fault caught by a different net than expected
is a finding to record rather than a reason to adjust the fault, and the
finding here is a good one: the lit tests and the unit tests cover the same
arithmetic from two sides, and the lit side sits earlier in the job.

That run still left the new step without a red of its own, so a second fault
was built that only the step could see: one wrong expected peak in
AllocatorTest.cpp, invisible to lit because lit never compiles the unit tests.
It went red exactly at NPUAllocatorTests, with coverage agreeing, in run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32326939623>.
The restore went green in run
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32327122856>.

The general shape, for the next activation: a product side fault proves the
whole net and usually lights the earliest gate; a test side fault is what
isolates the one step whose activation is being proved. Doing both, in that
order, proves the step and measures the depth of the net in front of it.

## 2026-08-20, phase P6: the binary format, and the four gates that switched on

Section 9 in full, plus the generated ISA description of Section 9.4 and the
fuzzing of Section 17.3. Seven commits on `phase/p6-binary-format`, cut from
`f6baff2`.

### Generating the ISA first, which the roadmap is right about

The roadmap says to build the description before the encoder, because
generating the opcode enum afterwards means writing the encoder twice. That is
correct and it is worth saying why it is more than an ordering preference. The
generated `OpcodeInfo` table carries the arity, the field presence mask, the
memory space per operand slot and the element type masks, and `validate()` reads
all four out of it. Written by hand first, those rules would have been written
as `if (opcode == MATMUL && operands.size() < 2)` in a dozen places, and the
second pass would not have been a port, it would have been a rewrite of the
validator's whole shape.

The generator refuses a half filled record rather than emitting a blank row: an
opcode with no format string, no semantics line, no shape rule, no accepted
element type, or fewer declared operand spaces than it takes operands is a
generation failure. That is cheap to write and it is what makes the claim
"adding an opcode touches one description file and then fails to build" true
rather than aspirational.

**The mechanism that makes it fail to build is `-Werror=switch` over the
generated enum with no `default` label.** Two layers use it: the semantic
dispatch in `Validation.cpp`, and the per opcode builder in `PropertyTest.cpp`.
Everything else is table driven and needs no case. That is the honest scope: the
generated tables cover the mechanical layers on their own, and the switches
catch the two places where a human has to decide something. The simulator's
dispatch skeleton is generated as an X macro for P7 to expand.

### The namespace collision, caught in the first hour

The binary format wanted to be `namespace npu`. The tensor level dialect already
owns `mlir::npu`, and a second `::npu` is ambiguous in every translation unit
that says `using namespace mlir;` and then names either one. Renaming to `nbin`
cost four characters and a rename pass over six files. Leaving it would have cost
a qualification argument at every phase that touches both levels, and P7 touches
both levels on its first day.

### What the corpus and the fuzzer each found

The seed corpus of Section 17.3 is 734 hand written cases and it found one
defect, D-0020: a `RESHAPE` whose result held a different number of elements
from its operand was accepted, because the manual stated the rule and nothing
enforced it. That is exactly the kind of thing a table of enumerated malformed
inputs is good at, and exactly the limit of it: the case was one somebody
thought of while reading the manual.

The coverage guided target found two the corpus did not. D-0022 is a signed
overflow in the disassembler's contiguity walk, on the one path in the project
that runs before validation, and it needed a stride vector that was *right* for
a shape that was absurd. A person writing a malformed operand writes a wrong
stride vector. D-0023 turned up while reading the minimized input's listing: an
instruction with a missing mandatory operand disassembled to a blank line,
opcode and all, which is `npu-objdump` failing at its only job on exactly the
file it exists for.

D-0021 came from UndefinedBehaviorSanitizer on the corpus, on the sanitizers
job's first real run: four range diagnostics computed the end of a range the
check above had just refused, and the operand extent check did the same in a
comparison rather than a message. The consequence on this machine is a wrapped
number in a message nobody reads. What it is, though, is undefined behaviour at
a site whose surrounding code is a bounds check, and the assumption a compiler
would draw from it is that the address is small.

**Three tools, three different classes of defect, and no overlap between them.**
That is the argument for having all three rather than the cheapest one.

### The allocator hook, twice wrong before it was right

Section 17.3 wants the allocation bound measured through a hook rather than
asserted in prose. The first version replaced `operator new` and
`operator delete` with a size header in front of every block and measured peak
live bytes. It works in the gcc build and segfaults under AddressSanitizer
before the first test finishes: ASan defines both operators too, both as strong
symbols, and which definition wins can be decided differently for the two of
them. Allocations came from this file with the header offset applied and
deallocations went to ASan's `operator delete`, which handed the offset pointer
to its own `free`.

The second version dropped the header and forwarded to `malloc`. ASan then
correctly reported an alloc-dealloc-mismatch, because it records which family a
block came from. `alloc_dealloc_mismatch=0` would have silenced it, and
suppressing an AddressSanitizer check inside the one job whose entire purpose is
AddressSanitizer is not a fix.

The third version is the one that ships: the hook is compiled out under the
sanitizers entirely, and the two claims are split across the two builds. The gcc
build measures the bytes; the clang build proves the memory safety; both run the
whole corpus. The metric changed with it, from peak live to total requested,
which is a stricter bound and immune to a deallocation the hook never saw. The
test asserts the hook was called at all, because a replacement that lost the
link would report zero for every case and pass.

### The four activation proofs, and the one that measured the net

Section 19.0 switches on `NPUEncodingTests`, the ISA staleness gate, the
`sanitizers` job and `nightly.yml`'s fuzz job at this phase, and 19.1 says to
break each once and show it red.

The P5 handoff left a shape for this: a product side fault proves the whole net
and usually lights the earliest gate; a test side fault isolates the one step
being proved. Both were done for `NPUEncodingTests`, and the product side one
produced a finding rather than a confirmation.

**The product side fault lit nothing but the step it was aimed at.** Swapping
two adjacent `i32` writes in `Program::encode` produces files that fail their own
validator, and `check-npu` reported 18 of 18. The chain is D-0024:
`npu-translate` validates the program in memory rather than the bytes it writes,
which is right and catches a different class of defect; `npu-objdump` printed
its warning block above a listing whose every line still matched, because
swapping two fields that both round trip moves no address and no shape; and
`objdump.mlir`'s `DUMP-NOT: WARNING` sat below the listing, where a `CHECK-NOT`
covers only the tail. The net in front of the encoder's output was one badly
placed directive deep. Moved above the first positive match, the same fault
takes `check-npu` to 17 of 18 at `Encoding/objdump.mlir`.

The sanitizers job needed its own isolating fault for the same reason, and
finding one took two attempts. Reverting half of D-0021, the operand extent
comparison, left the job green: the corpus reaches the result range *message*
and not that comparison. The revert that isolates it is the message, which
changes only a number inside a diagnostic string no test asserts, so the gcc
build stays green and only UBSan sees it. Recording that the first attempt
proved nothing is the point of the exercise; a rehearsal that is adjusted until
it passes has measured nothing.

Measured, all four, under the exact CI invocations:

| Gate | Fault | Red | Restored |
|---|---|---|---|
| `NPUEncodingTests` | `Program::encode` swaps two `i32` fields | 7 tests, exit 1 | 76 passing |
| `NPUEncodingTests`, isolating | one wrong constant in `EncodingTest.cpp` | `FrozenConstants.TheFormatsNumbers`, exit 1, `check-npu` still 18 of 18 | 76 passing |
| ISA staleness | the committed manual perturbed, in a commit | exit 1 with the diff | exit 0 |
| ISA staleness | the description moved, not regenerated | exit 1 with the diff | exit 0 |
| `sanitizers` | D-0022 reverted | UBSan runtime error, exit 1 | 75 passing |
| `sanitizers`, isolating | D-0021's message reverted | UBSan runtime error, exit 1, gcc build still green | 75 passing |
| nightly fuzz | the decoder stops refusing trailing bytes | property violated in seconds, exit 77, crash written | 1964713 runs in 61 seconds, no artifacts |

### The staleness gate had a hole, and the rehearsal is what found it

Perturbing the committed manual left the gate green on the first attempt. The
script regenerates and then diffs, and `git diff` compares the working tree
against the **index**: regeneration had already overwritten the edit, so the
diff was against something that no longer disagreed. It compares against `HEAD`
now, which is what "the committed artifacts are stale" means, and in CI the two
forms agree because a checkout leaves the index equal to `HEAD`.

The same rehearsal is why the script refuses to run against an untracked
artifact. `git diff` says nothing at all about a file git is not tracking, so an
untracked manual would have passed this gate forever while being regenerated on
every run.

### `.gitignore` swallowed the seed corpus

`*.nbin` is right for the compiler's output and exactly wrong for eight input
files whose exact bytes are the test. `git add fuzz` reported nothing, committed
nothing, and a commit went in describing a corpus it did not contain. It was
caught by running `git ls-files fuzz/` afterwards rather than by trusting that
`git add` on a directory adds the directory, which is now a habit worth keeping.

### What was cut, and why

`fuzz/nbin_structured_fuzzer.cc`, the libprotobuf-mutator target of Section
17.3, is not here. It is the third item on Section 2's cut list, paired with
mutation testing, and Section 2 says cutting it keeps the libFuzzer target and
the seed corpus, which are the parts that find bugs nobody imagined. Both of
those are here and both found defects this phase. The reason it went rather than
something else: libprotobuf-mutator is not in the CI image or in this machine's
toolchain, so it would have meant vendoring a dependency and a protobuf mirror
of `Program` that has to be kept equal to `Program` by hand, which is a second
hand maintained copy of the thing this phase spent its first commit eliminating.

## 2026-08-20 Phase 6: four activations proven, and the net kept being deeper than predicted

Four gates switched on at P6 and each was broken once, shown red, and restored,
on a scratch branch since deleted. The rehearsed recipes from the handoff ran
as written; what the runs added was a measurement of how deep the net in front
of each gate actually is.

The frozen constant expectation flip was aimed at NPUEncodingTests and lit
three jobs at once: build-and-test at the encoding step, the sanitizers job
running the same suite under ASan, and coverage. Red at
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32361933634>.

The hand edited opcode number in the committed manual was caught by exactly one
step, the ISA staleness gate, everything else green:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32362196768>.
In CI the perturbation is committed by construction, so the finding from the
local rehearsal, that the gate diffs the committed artifact and an uncommitted
edit is overwritten before the diff, does not arise there.

Putting D-0021's overflowing addition back inside the ResultInRange diagnostic
produced the cleanest isolation of the phase: the sanitizers job red on the
UBSan corpus walk, and the gcc build, lit, and coverage all green, because no
test asserts the arithmetic inside a diagnostic string:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32362551365>.

Disabling the decoder's trailing bytes refusal was aimed at the nightly fuzz
job and hit four nets. Nightly went red at the long coverage guided run, with
the three still guarded jobs printing their off lines:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32362989210>.
The branch CI went red at NPUEncodingTests, sanitizers, and coverage besides:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32362989932>.
The prediction was that only the fuzzers assert the round trip property on
arbitrary inputs, and it was wrong in the good direction: the malformed corpus
carries trailing bytes cases that assert the same refusal on fixed inputs, so
the unit suite sees the fault before any fuzzer mutates its way to it. The
depth of a net is not knowable from the recipe that aims at one strand of it.

Restores green: CI
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32363276829>
and nightly
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32363275922>.

The phase also cost one image republish: the sanitizers job's first run died at
CMake's compiler check because the image carried clang without Ubuntu's
separately packaged sanitizer runtimes, D-0025, fixed in the image per the
pinning rule rather than with a job time install.

## 2026-08-31 Phase 7: a phase resumed from an uncommitted tree, and four things its own tests could not see

**How the phase started.** The session that wrote the simulator did not survive
to commit it. What this one found on `phase/p7-simulator` was `7b9d18c`, zero
commits, and about five thousand lines of untracked work sitting beside it:
`include/NPU/Simulator`, `lib/Simulator`, `tools/npu-sim`, `unittests/Simulator`,
`refexec.py`, three pytest files and five one line edits to the CMake and lit
wiring. It built clean. Fifty one unit tests passed, `check-npu` reported 19 of
19, pytest reported 178 passed.

None of that is evidence, and saying why is the point of this entry. Tests
written by the session that wrote the code prove the two agree with each other.
They say nothing about whether either agrees with the specification, and a phase
accepted on the strength of its own green suite is a phase whose gate was never
run. So the first work was an audit against Sections 5.5, 9.3, 10.1, 10.2, 10.3,
17.3a and 19.0, clause by clause, against the tree rather than against a summary
of it. Three defects came out of that. The fourth came out of a rehearsal.

### The gate asked for an NDEBUG build and this project could not make one

Section 9.3 wants the bounds checked accessors to refuse gracefully in every
build mode, and the P7 gate wants the trap tests in an assertions build and an
NDEBUG build with all four runs shown. The obvious move is a second build
directory at `-DCMAKE_BUILD_TYPE=Release`. It was made, it built, and the four
tests passed.

They passed in an assertions build. `LLVMConfig.cmake` sets
`LLVM_ENABLE_ASSERTIONS` as a plain variable out of the LLVM tree this project is
configured against, which shadows the cache value the obvious override sets, and
`HandleLLVMOptions` then adds `_DEBUG`, `_GLIBCXX_ASSERTIONS` and an explicit
`-UNDEBUG` that lands **after** the `-DNDEBUG` a Release configuration supplies.
The last `-D` or `-U` on the line wins. Nothing about the result looks unusual:
the directory is named for what was wanted, the configuration says Release, the
tests are green.

It was caught by reading the compile line rather than the result, which is the
habit that caught P3's, P5's and both of P6's proofs that proved nothing. The fix
is `NPU_FORCE_NDEBUG`, off by default, and a compile time probe that now backs
the claim: a file whose first lines are `#ifndef NDEBUG` and `#error`, compiled
with the exact flags `compile_commands.json` records for
`lib/Simulator/Memory.cpp`. It exits 0 in `build-ndebug` and 1 in `build`.
D-0028.

Two attempts in between are in the defect log because each looked right. Putting
the switch after `include(HandleLLVMOptions)` sets a variable nothing reads
again. And rewriting `LLVM_DEFINITIONS` with `string(REPLACE)` does take the
macros out, and then hands `add_definitions` one argument full of spaces, which
CMake stops parsing at the first `=` and passes to the compiler as a definition
of `_GLIBCXX_USE_CXX11_ABI` whose value is the rest of the line. Twelve
redefinition warnings, out of a build that otherwise did exactly what was wanted.

### The simulator aborted, on the one path built to prove that it would not

`Simulator::runUnvalidated` exists for a single reason, stated in its own
comment: a program that has passed `validate()` cannot reach the trap path, so a
test that could only submit validated programs would be asserting that the last
line of defence exists rather than that it works. It is the entry point the
contract is proven through.

Handed an instruction with no operands, it reached `operands.front()` on an empty
vector and the process aborted inside libstdc++. An abort is precisely what
Section 9.3 forbids on the trap path, in the words "graceful refusal is the
contract in a release build and in an assertions build alike", and in a build
without `_GLIBCXX_ASSERTIONS` the same call is silent undefined behaviour rather
than a loud one. D-0026. The fix reads `minOperands` out of the table already
generated from `NPUISADescription.td`, so there is no second arity rule to keep
in agreement with the first.

Why it was there is worth stating without excusing it. Kernels index their
operands positionally and on the validated path they may: `validate()` refuses
short arity and names the `arity` check, and re-testing that per element would be
paying twice in the inner loop. The gap is exactly the width of the entry point
that skips the first check, and that entry point was added in the same file that
then failed to guard it.

### A docstring described a check that did not exist

`cost_model.py` said its charges were "checked against the simulator's own output
on a real program". They were not. The test parsed the constants out of
`CostModel.h` and compared them name by name, which is real and mechanical, and
then checked the Python formulas against literals written beside them, which is a
different claim. A mirror reproducing every constant and then charging with them
differently would have passed every assertion in the file, and the numbers the
later phases plot are the answers rather than the constants.

The docstring is the defect rather than a symptom of one, because the next person
reads the claim and stops looking. D-0027. The check exists now: `npu-sim` runs
an exported case and the Python mirror reconstructs the statistics it prints. The
case is `matmul_narrow_bias`, and narrow is the whole point. 5 by 19 by 3 folds
into two tiles and neither fills the array; against a tile that does, the
utilization and preload terms are both 1 and a mirror missing them would still
pass.

### The rehearsal found the fourth, by disagreeing with its own prediction

The product side activation fault was `POOL_MAX` starting its accumulator at zero
rather than negative infinity. The prediction, written down before the run: one
hand computed test red, `Pooling.AllPaddingWindow`, because every other pooling
case uses positive inputs and the maximum of a positive window is unchanged by a
zero floor; `check-npu` untouched; the differential red on the two max pooling
cases, whose inputs are random over `[-1, 1)`.

All of that happened. But the differential reported **54 of 54** elements
mismatched on `max_pool2d`, and with windows of four over `[-1, 1)` about one
window in sixteen should have an all negative maximum. Three or four was the
number. Fifty four was not a stronger result, it was a different one.

The generator shifted its state right by 33 bits and not 32. That leaves thirty
one significant bits, which divide into `[0, 1)` and subtract into `[-1, 0)`.
Every input the differential suite had ever exported was negative. The `relu`
case was comparing a buffer of zeros against a buffer of zeros. The two pooling
cases had never seen a positive maximum. And the determinism test, which carries
a copy of the same generator, was convolving negative inputs against negative
weights, so every product in the reduction it exists to hold still carried the
same sign, which is the easiest possible case for a summation order to survive.
D-0029.

**Nothing about the output looked wrong**, and that is the part to carry
forward. The values were random, deterministic, reproducible from a seed, and
inside the stated interval at one end. Twenty four cases agreed. A reader opening
one of the `.bin` files would have seen plausible floats. The only signal was a
number in a rehearsal that did not match a prediction, which is an argument for
writing the prediction down first rather than reading the result and agreeing
with it afterwards.

Three guards went in rather than one, because a comment is not a mechanism: an
assertion on the generator's range, an assertion on the bytes that actually
reached the files, and an assertion that the `relu` case's reference output is
not entirely zero. The second survives a rewrite of the C++ that keeps the
comment and loses the property, which is the failure the first cannot see. The
per operand form of the second is conditioned on size, and that is not
decoration: the three element bias of `matmul_narrow_bias` is all positive in
this export, which happens to a fair three element sample one time in four, and a
rule that called that a defect is a rule somebody eventually deletes.

### What the gate looked like once it was met

Fifty five unit tests where fifty one arrived. The four additions are all things
the specification asked for and the tree did not have: `POOL_AVG` at batch 2 and
both elementwise opcodes at both batch sizes, because Section 10.1's list says
"both pooling kernels" and "the elementwise operations" and a list asking for two
things is not satisfied by covering each of them once between them; a `DEQUANT`
refusal test, because the same section's first sentence asks for a test per
opcode and `DEQUANT` had none; and the two arity traps from D-0026. Beside them
`docs/adr/0007-dataflow.md`, which `CostModel.h` had been citing by path since
before it existed.

Both activation rehearsals matched their predictions exactly on the second run:
the product fault red at one unit test and at the differential with `check-npu`
untouched, and the test side fault red at one unit test and nowhere else.
`check-npu` seeing neither is correct rather than a hole. `npu-sim.mlir` exists
to assert the tool's contract with the format, and asserting numerics there would
duplicate the unit suite in a worse language.

One prediction from P6's handoff did not come true, and it is recorded rather
than dropped: P7 was expected to be the first phase holding `nbin` and
`mlir::npu` in one translation unit. It is not. Nothing in the simulator includes
an MLIR header at all, which is what keeps `npu-sim` a tool that reads a file and
runs it rather than one that links a compiler, and what keeps `NPUSimulatorTests`
a binary that links in seconds. The namespace split stays. The phase that will
actually test it is `npu-compile` at P8.

## 2026-08-31 Phase 7: the activation proof on CI, and the branch that CI could not see

The two rehearsed faults ran against CI on a scratch branch and both matched
their local predictions. The branch's own first run, the step's first time on,
everything green:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33362572992>.

The product fault, the pooling accumulator starting at zero, went red in three
places: the build and test job at the `NPUSimulatorTests` step, the sanitizers
job at its GoogleTest step, and coverage besides, which is the same three net
pattern P6 recorded:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363102927>.
CI showed less of this fault's net than the local rehearsal did, and the reason
is mechanical rather than interesting: a job stops at its first red step, and
`NPUSimulatorTests` precedes pytest, so the differential's two extra catches,
the ones the hand computed tests structurally cannot reach, are in the local
record only. A red step hides the depth of the net behind it.

The test side fault, the frozen constants expectation moved from 16 to 17, went
red at exactly the same three steps and nowhere else, which is what isolation
means once CI is the frame: every step that executes the binary, and nothing
that does not. lit and pytest saw nothing:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363280962>.

The restore, byte identical to the phase branch tip, returned green:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33363608121>.

One process finding, recorded so the next phase does not repeat the dead push.
The first attempt ran nothing at all: the rehearsal branch was named
`scratch/p7-activation`, and `ci.yml` triggers on `phase/**` and `main` only, so
the push produced no run and the absence of a run is what had to be noticed. A
rehearsal branch must be named under `phase/`; this one was renamed to
`phase/p7-activation-rehearsal` and deleted after the restore run. A workflow
that triggers on everything would remove the trap, but widening a trigger to
serve a rehearsal that happens once per phase is the tail wagging the dog, and
the rule is cheaper than the change.

## 2026-08-31 Phase 8: the walking skeleton walked on the first try, and everything interesting was somewhere else

**The end to end pipeline worked the first time it was run.** ONNX in, a
simulated answer out, matching onnxruntime to 2.98e-8 on LeNet, at the first
attempt, with no debugging at all. That is the least interesting sentence in
this entry and it is here because it is true: P4 through P7 each landed a stage
with its own tests, and the stages fitted. What follows is everything that was
not the pipeline.

### The models are twenty five instructions, not thousands

P7's handoff left the quadratic executor on this phase's desk: `readyAt` is
linear in the writes so far, so the executor is quadratic in the instruction
count, and "P8's models are thousands of instructions". They are not. The
largest is LeNet at **25**, and the smallest is the dilated stack at 11.

The reason is structural rather than lucky, and it is worth writing down because
it decides when the concern comes back. This machine's instructions operate on
whole tensors: one `npu.conv2d` is one `CONV2D`, and a whole 400 by 120 matrix
multiply is one `MATMUL`. A model is as long as its operator count plus its DMA,
and Section 15's models have between eight and fourteen operators. Nothing in
the pipeline multiplies that.

Measured anyway, on a synthetic chain of relus, because a decision made without
a number is a decision made twice:

| instructions | wall clock | per instruction |
|---|---|---|
| 28 | 3 ms | 108 us |
| 103 | 2 ms | 19 us |
| 503 | 4 ms | 8.7 us |
| 1003 | 8 ms | 7.8 us |
| 2003 | 16 ms | 8.0 us |
| 4003 | 37 ms | 9.1 us |

Flat from 500 to 2000 and up by 14 percent at 4000, which is where the quadratic
term starts to be visible over the per instruction kernel cost. At the sizes
this compiler actually emits it is not measurable at all. **No interval
structure, and the reason is a number rather than a shrug.** The phase that will
feel it is P13: tiling fully unrolls its loops before lowering, per Section 5.2,
so P13 is the first phase whose programs are long by construction.

### Section 15's tight budget rule does not survive contact with these models

Section 15 says to measure each model's peak from the allocated
`npuisa.scratchpad_bytes` at the default budget and set the tight budget as a
fixed fraction of it, rounded to a multiple of 4096. At a fraction of 0.75 all
seven models are **refused**, and the refusal is not fragmentation:

```
lenet, budget 143360: the scratchpad budget of 143360 bytes is too small: this
buffer of 192000 bytes could not be placed below offset 143360 in @main, and no
buffer live across the pressure peak can be spilled
```

The peak of these models is set by one instruction's own operand set. LeNet's
largest fully connected layer holds a 192000 byte weight matrix that has to be
resident while the `MATMUL` reading it runs, and spilling does not help, because
a spilled buffer is reloaded before each use and the reload is resident at the
same moment the original would have been. Section 13 already names the remedy
and it is tiling, at P13.

Swept in 64 byte steps, the smallest allocatable budget is the peak itself on
five of the seven models and 0.76 and 0.90 of it on the two whose pressure comes
from several concurrently live buffers. Those two spill and the other five
cannot be made to. The rounding quantum made it worse rather than better:
`inception_block` spills three buffers at 6144 bytes and spills nothing at 8192,
so applying the specification's own 4096 byte rounding would have taken the one
model with the most interesting tight budget cell and turned it into a second
copy of its default budget cell.

Both deviations are in `docs/adr/0008-per-model-tight-scratchpad-budgets.md`
with the measurement that forced them. The fraction is recorded as inoperative
rather than as 0.75, because a floor that overrides a fraction on all seven
models makes the fraction a number that does nothing.

### Section 17.4 predicted the wrong cell for its own vacuous bound

The specification says the `zeros` input class produces an exactly zero
reference, so the relative bound is vacuous there and only the absolute one
should be asserted. It does not. Every model in this suite has biases, so a zero
input produces a nonzero answer and the relative bound bites normally.

The class that does produce an all zero reference is `large_neg` on
`resnet_block`, where a closing relu takes everything to its dead side. There
the assertion is stronger than any tolerance: the simulated answer must be
**exactly** zero as well. The handling is general rather than a special case for
that cell, so a later model that acquires the same shape is covered.

### The tolerances, and the v1 lesson they were set with

Measured over the whole matrix: worst absolute 4.77e-06, worst relative
8.08e-07 against onnxruntime and 4.17e-07 against the reference interpreter.
The bounds are 5e-05 and 5e-06.

The absolute number is scale dependent and the U4 entry above is why that is
stated rather than assumed. The worst case is `dilated_stack` at `large_neg`,
whose answers are of order 14.5, where a float32 ulp is 9.5e-07. So 4.77e-06 is
about **five ulps** and the bound is about fifty two, which is the same
arithmetic quality the `normal` cells show at their own scale. The constant
classes are at magnitude 10 rather than the 1e3 v1 used, which is what keeps one
absolute bound usable across every class instead of unsatisfiable on two of
them.

### Four oracles, one relation that cannot be written

Section 17.3a lists five metamorphic relations. The fifth, pad then slice back,
needs an ONNX `Pad`, which this importer refuses by name for a documented
reason, and an ONNX `Slice`, which has no converter at all. Writing a converter
for two operators so that a test could use them would be growing the operator
set to satisfy a test, which is what law 2 exists to prevent. It is recorded in
`metamorphic.NOT_IMPLEMENTED` with that reason, and a test asserts the reason is
still true, so if either operator ever gains a converter the relation gets
written rather than staying forgotten.

The four that exist agree **exactly**, not to a tolerance. None of them changes
the order any output element's terms are summed in: an identity is erased at
import, a transpose and its inverse move elements without arithmetic, splitting
a convolution over output channels leaves each output element's own reduction
untouched, and permuting independent nodes changes nothing about what is
computed. A tolerance there would have been a place for a real disagreement to
hide.

`node_order_permutation` reaches exactly one model, and that is the relation
rather than a limitation: a straight line graph has one topological order.
`inception_block` is the model that exists to have parallel branches, so it is
the one the relation acts on, and the test names it, so a suite change that
flattened it goes red instead of leaving the relation with nowhere to apply.

### The dead subgraph, and a gate clause about a level that does not exist

The gate asks that a dead subgraph change neither the outputs nor the **-O2**
instruction count. `-O2` arrives at P9. Worse, at `-O0` the second half is not
merely unmeasurable, it is false by construction: Section 12 puts every pass
that removes anything at `-O1` and above, so a dead subgraph at `-O0` must
change the count.

The resolution is written down rather than chosen quietly. The outputs half is
asserted now, bit identical, on every model and every input class. The count
half is asserted at every level whose pipeline eliminates dead code, and which
levels those are is read out of the compiler rather than from a list of pass
names: `PassEntry` gained an `eliminatesDeadCode` property under the same rule
as `ablatable`, a missing one is a build error, and the level set is empty at P8
and fills itself at P9. Beside it sits the P8 form of the same check, which is
just as falsifiable in the other direction: the count grows by exactly the three
instructions the injection brought, and by no more. A count that grew by four
would mean the injection cost something it did not declare; one that grew by two
would mean a pass nobody registered removed something.

### Three defects, and the test suite found none of them

**D-0030** is a test that spawned a second interpreter and let it inherit
`PYTHONPATH`. `pythonpath` in `pyproject.toml` puts the package root on pytest's
own `sys.path` and exports nothing, so the child found `npu_frontend` only
because the developer wrapper used all through this phase exports it. Green on
every run anybody had done. It would have gone red in exactly one CI job, with a
message naming a missing module rather than a missing variable. It was found by
`scripts/coverage.sh`, which is the same command with one variable fewer.

**D-0031** is larger and is not this project's code. Running the end to end
pipeline against the `build-ndebug` binaries aborted with a double free inside
`npu-opt`. `echo 'module {}' | build-ndebug/bin/npu-opt -` aborts too, and that
input reaches no operation of this dialect, no pass and no tool code at all.
`NPU_FORCE_NDEBUG` turns `_GLIBCXX_ASSERTIONS` off in this project's translation
units, which is exactly what it exists to do, while the LLVM they link is an
assertions build whose archives have it on. Mixing libstdc++ hardening across a
link changes the definition of the standard containers between translation
units.

`build-fuzz` has the mirror. AddressSanitizer's container annotations are on in
this project's translation units and off in the LLVM archives, so an `npu-opt`
built there reports `use-after-poison` inside `mlir::BuiltinDialect::initialize`
before `main` has done anything of its own. The stack is thirteen frames of MLIR
and none of this project's.

Neither is new breakage. `build-ndebug` was created at P7 to prove Section 9.3's
claim about the simulator's bounds checked accessors, and the simulator links no
MLIR at all, so nothing had ever built an MLIR linking target in it. P8 is the
first phase whose end to end run wants `npu-opt`. The CI sanitizers job builds
three targets by name and none of them links MLIR, which is why CI has not seen
it either. The real fix is a second LLVM tree built without assertions, an hour
of build time and a decision with a cost, so it is documented loudly and left to
be taken deliberately.

**And a third, D-0032, which arrived after this entry was written**, on the
first CI run of the coverage job's new Python arm. It belongs here rather than
in a later entry because it is the same sentence as the other two, said a third
way: a result that depended on what else was lying around.

`scripts/coverage.sh` builds into `build-coverage/` and never told the Python
suite so. On a developer machine the suite found its binaries in `build/`,
sitting beside it. In the job there is no `build/`, so `npu-opt` was not found
and the run died at collection with exit 2. The fix is that the script exports
`NPU_BUILD_DIR` set to the directory it actually built.

**The interesting half was underneath.** Three places knew how to find a binary:
`npu_frontend.find_tool`, and a hand written `build_directory()` in each of two
test modules. Only the first said anything; the other two **skipped**. Measured
by letting collection survive and changing nothing else: `484 passed, 12
skipped`, five of those skips being tests that should have run, one of them
`test_every_case_agrees`, which is P7's gate item that the reference interpreter
and the simulator agree on every operation. A green run, with a coverage number
attached, describing a suite that had not run its differential oracle. Section
17.7 says coverage is only counted from a run where every test passed; every
test did pass, and five of them passed by not running.

The rule is one now, in `test/Python/tools.py`, and the policy with it: a
missing binary is a **skip** when nobody named a build directory, which is a
developer running the pure Python tests, and a **failure** when somebody did,
because naming one is asserting the build is there.
`test/Python/test_tool_discovery.py` scans for a second copy so the next one is
a red test rather than a thing somebody notices in a year.

**What it says about rehearsals.** This phase rehearsed the coverage arm before
switching it on and the rehearsal matched its prediction exactly. The prediction
was about the threshold arithmetic and it was right about that. It was never a
test of the arm's environment, and the environment is what broke. **A local
rehearsal proves a step's logic and cannot prove its surroundings**, because the
surroundings are the thing that differs, and the first run of a new CI step is
the first time anybody learns what they are. That is an argument for switching
steps on one at a time, which is what the activation table already asks for, and
it is the strongest thing this phase learned about its own method.

After the fix, under the job's conditions in a worktree with no `build/`:
`495 passed, 7 skipped`, C++ 86.1, Python 90.61, exit 0. The Python coverage
headroom widened from 0.27 points to 0.61, because the tests that had been
skipping now run.

### The OpenMP split, which P7 left open, is the compiler and not the image

P7 recorded that the build and test job's configure prints `OpenMP: not found`
while the coverage job in the same image finds OpenMP 4.5, and left the
mechanism unestablished. It is the compiler. `build-and-test` and `sanitizers`
configure with `-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++`; the
coverage job passes no compiler and gets gcc. `find_package(OpenMP)` under gcc
finds `libgomp`, which `g++` brings; under clang it needs `libomp` and `omp.h`,
which `libomp-dev` brings and which the image does not install.

Reproduced locally rather than reasoned about. This machine has `libgomp1` and
no clang `libomp`, `clang -fopenmp` fails at `'omp.h' file not found`, and
configuring this project with clang prints the CI line verbatim:

```
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- OpenMP: not found. The convolution kernel runs single threaded.
```

The consequence is smaller than it looks. Section 10.3's determinism assertion,
that one thread and the maximum produce bitwise equal buffers, runs at full
strength wherever OpenMP is found, which is the coverage job and every
developer machine. The fix is one package in `docker/Dockerfile.llvm` and an
image republish, which costs an hour and is the orchestrator's call.

### A rehearsal that disagreed with its prediction, again

The activation rehearsal for the reachability step edited
`docs/ISA_OPCODES.json` by hand, flipping `needs_kernel` to false on
`POOL_MAX`. The prediction was two catches: the reachability step red naming
`npu.max_pool2d missing: simulation`, and the ISA staleness step red because a
committed generated artifact no longer matched its description.

The first happened. The second did not: the staleness step ran green.

The mechanism is that `check-isa-staleness.sh` **regenerates** the artifacts
before it diffs them, so a hand edit in the working tree is overwritten by the
regeneration and the diff finds nothing. That is correct behaviour rather than a
hole, and the distinction is worth stating: an edit somebody **commits** is
still caught, because then the regeneration overwrites the working tree and the
diff against `HEAD` shows the difference. Only an uncommitted edit is silently
repaired, and silently repairing a generated file is what a generator is for.

It also means the two steps interact through the filesystem: running staleness
before reachability would have repaired the fault before reachability could see
it. The rehearsal ran them in the other order by luck rather than by design.

### The name `--stats-json` was already taken

`npu-sim` writes its statistics as JSON now, so that nothing parses the human
readable form for a number that Section 10.2 makes the only instruction count in
this project. The flag wanted to be `--stats-json` and cannot be: LLVM's Support
library registers that name for the `llvm::Statistic` counters, and every tool
linking Support inherits it. The collision aborts inside
`ParseCommandLineOptions` at the first run, which is loud rather than silent.
The flag is `--json-stats` and the reason is beside the name rather than in a
commit message nobody reads twice.

### What the suite runtime turned out to be

Section 17.4 says to mark the slow cells so a fast subset runs by default, and
to reach for `pytest -n auto` if the full matrix exceeds ten minutes. The full
`-O0` matrix is one hundred and forty cells and takes **twelve seconds**,
including the exports. There is nothing to carve out, so no cell is marked
`slow`: marking a cell that costs a tenth of a second as slow would be a label
rather than a measurement. The marker and the CI step that runs it stay in
place, and they start doing work at P9 and P10, when three levels and two
budgets multiply this matrix by six and the ablation cells arrive beside it.

The whole verification matrix, including two extra build directories, the
coverage measurement and the regression baseline, is under ten minutes. The
90 minute budget of Section 2 is a P10 figure and this phase says nothing about
it.

## 2026-08-31 Phase 8: the proof of failure gate run against CI, and what the first run of the coverage arm caught

The branch's first run went red before any fault was aimed at it, and that red
is the best result of the closing sequence. The coverage job's new Python arm
died at collection: `npu-opt was not found`. The job configures only
`build-coverage/`, and discovery had been finding `build/` beside it on every
developer run, which is D-0030's failure class landing a second time, D-0032.
The audit of the fix found the larger fault: two hand written copies of the
discovery in the differential tests were skipping instead of failing, so five
tests, `test_every_case_agrees` among them, had been silently absent from every
coverage number recorded so far. The fix put discovery in one place with one
policy, a missing binary is a skip when nobody named a build directory and a
failure when somebody did, and a test now hunts for a third copy. The red run:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33367169622>.
The fix, green in every job, Python coverage 90.61 on the CI host against 90.60
here:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33428771811>.

Then Section 19.1's four fault classes. The first pass at them was wrong and is
recorded as the finding it is: all four ran as pushes to one rehearsal branch,
went red at the right steps, and proved the wrong trigger. Section 19.1 says
each fault is its own pull request **so the `pull_request` trigger fires**,
because that trigger is the one that guards a merge, and a push triggered red
says nothing about it. The push runs stand as evidence the steps catch the
faults (33429399189, 33429985494, 33430305159, 33431674770, each red at the
same steps as below); the gate's record is the four pull requests, numbers 9
through 12, one fault each, opened as drafts and closed unmerged with their
branches deleted.

An em dash appended to `docs/BREAKING_CHANGES.md` went red at the lint job's
dash lint step and nowhere else:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33433615421>.
Committing that fault locally required `--no-verify`, because the pre-commit
hook refuses exactly this edit, and that is the point of the class rather than a
corner cut: the gate models the commit that never ran the hooks, and the
repository has already had one, the web edit that P7 cut as `1939b58`.

A lit expectation changed from `npuisa.dma_store` to `npuisa.dma_never` went
red at check-npu, and at coverage besides, which runs the same lit suite under
instrumentation:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33433628400>.

A frozen constant expectation moved from 256 to 257 went red at the
`NPUSimulatorTests` step, the sanitizers job's GoogleTest step, and coverage,
the same three step net P7 measured:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33433632114>.

A tight budget expectation moved by one byte went red at the pytest step and at
the coverage job's Python arm:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33433634366>.

The reachability activation took a different fault in CI than in the local
rehearsal, and the reason is a finding. The local product side fault perturbs
`docs/ISA_OPCODES.json`, which the staleness gate regenerates before diffing,
so an uncommitted edit is silently repaired; a committed one, which is what a
rehearsal branch pushes, is caught by the staleness step, which runs before the
reachability step and would have stopped the job at the wrong net. The model
side fault has no generated artifact in it: the `npu.yield` exemption row
deleted from `docs/EXEMPTIONS.md` went red at exactly the
`check-reachability full` step, with the lint job's `--skip-models` variant
green beside it, which is the isolation stated in CI's terms:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33432288385>.

The rehearsal branch's restore, byte identical to the phase branch tip,
returned green:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33432842831>.
The `pull_request` trigger's own green is shown by the phase's merge pull
request in the ordinary course of merging.

## 2026-09-01 Phase 9: the passes were the easy half, and every interesting thing was one level down

**Where the time went.** Writing four passes and wiring eight into two levels
took an afternoon and produced no surprises: each pass fired where its lit test
said it would and declined where its negative case said it would, first try.
Then the levels ran on real models and three defects fell out in ninety minutes,
none of which any pass's own test could have found, because all three are
properties of a **composition** rather than of a pass.

That is the entry's whole point, so it goes first. A pass is testable in
isolation and this phase tested every one of them that way. A pipeline is not
the sum of its passes, and the only way to learn what it does is to run it and
read the output.

**D-0034, found by reading `-O2`'s output while writing CHECK lines.** Not by a
test. `-cse` merges two identical `tensor.empty` operations, which is correct:
they are identical operations with no operands, `npu` operations are `Pure` and
take their destination by value, and a destination has no contents, so two
operations sharing one are two pure functions of the same meaningless input.
`-npu-lower-to-npuisa` turns one `tensor.empty` into one `memref.alloc`, which
had been correct for four phases because the importer emits one per compute
operation and nothing had ever merged them. Put together they made two
instructions write one buffer, and when the second read that buffer through a
three by three window the program was simply wrong.

Every test still passed. The shape that reaches it in the model suite is a relu
reading its own output, which is elementwise and gives the right answer, so the
suite would have shipped it. The convolution chain that shows it was written by
hand while looking at something else.

**D-0035, found by measuring `-O1` against `-O0` before recording the
baseline.** LeNet went from 17766.25 cycles to 24392.75, a 37 percent regression
from an optimization level, with the same twenty five instructions, the same
16441 cycles of transfer and the same 7955.75 of compute. Only the overlap
moved: 0.8334 to 0.0005.

The cause is one line of MLIR's canonicalizer doing exactly what it should.
Constant hoisting moves every `ConstantLike` operation to the top of the block,
which is right for an operation whose cost is zero. In this compiler an
`npu.constant` becomes an `npuisa.const` and a `dma_load` at the position it
sits in, so its position is when its bytes are fetched, and Section 10.1's two
port model charges exactly that. Hoisting put all eleven of LeNet's transfers
above all of its compute, in an order that happened to put the last layer's
weights first, so the first convolution waited for the whole transfer budget.

The second half arrived an hour later from the same measurement. `-npu-fuse-ops`
takes every value its region reads as an operand, destinations included, so a
fused chain's destinations sit above the region and the flattening leaves them
there. Liveness runs from the allocation, so LeNet's `-O2` sweep line peak went
from 194624 bytes to 195040 and **the tight budget cell stopped compiling**, at
a budget `docs/adr/0008` froze at P8. An optimization level that could not
compile a program `-O0` compiled.

**Both halves are the same mistake and the fix is one rule.** A pass moved an
operation whose position was free at the level it was reasoning about and
expensive one level down. Neither pass is wrong; what was missing is that the
layer which turns a position into a cost had no opinion about position. It has
one now: in the pre conversion stage, a constant sinks back above its first
reader and a destination down to the operation that writes it, both past
computation and nothing else. Both are no operations at `-O0`, where the
importer's placement is already this one, and the `-O0` half of the baseline is
unchanged bit for bit, which was checked by reading the P8 file out of git and
diffing it field by field rather than by argument.

**The alternative was to re-measure the tight budgets and it would have been
wrong.** It was available, it would have taken twenty minutes, and it would have
moved every tight budget cell in the project's history to accommodate a
placement artefact. Ground rule 7 would have been satisfied by declaring it
first, which is exactly why declaring a movement is not the same as being
allowed to cause one.

**D-0033, found by looking for a positive test.** Section 12's rule is that
every pass has a positive lit test and at least one negative one. `-sccp` is an
upstream pass and I expected the positive case to be the easy half; instead
`npu-opt --sccp` returned every module I gave it byte identical, including one
built specifically to have a constant to propagate through a private function's
only call site. The dialect implemented no `materializeConstant`. MLIR's
constant propagation is a lattice plus a materialiser, the lattice worked, and
the solver asked the dialect for an operation to hold the answer, got null, and
left the value alone. No diagnostic, exit code zero.

The hook has been missing since P1 and would have stayed missing. Nothing ran
`-sccp` until this phase, and a gate that asked only for the negative case would
have been met by a pass that never fires. That is the sharpest argument for the
positive half of Section 12's rule I have run into, and it is worth recording
because the negative half is the one that feels rigorous.

**The ordering of the levels was a measurement too.** Section 5.1 lists the
batch norm fold before the bias fusion. Written that way, `test/Pipeline/
opt-levels.mlir` went red on a function with a convolution, a separate bias add
and a batch norm: the fold matches on a convolution as the batch norm's
producer, and until the bias has moved the producer is the add, so the fold
declined, and then `-npu-fuse-ops` declined too because the activation's
producer was a batch norm rather than a convolution. One ordering choice turning
three passes off. Swapping them fixed it, and Section 12's own note asks for the
fold "before fusion, so the convolution still has no fused **activation**",
which `-npu-fuse-ops` still is.

**The metamorphic tolerance question came out with a different mechanism than
P8 predicted.** P8 said fusion changes accumulation order, so the four relations
would stop being exactly equal and P9 owed the number. Measured over 84 cells,
29 of which apply: at `-O0` and `-O1` every cell is still exactly zero, and at
`-O2` exactly one cell moves, by 2.98e-08.

Fusion in general is not the reason, and this took a few minutes to see. A
relation compiles the original and the variant **at the same level**, so a
reassociating pass runs on both sides and cancels. What does not cancel is a
rewrite that changes which passes can **match**: `convolution_split` replaces
one convolution with two over channel groups and a concatenation, the batch
norm's producer becomes an `npu.concat`, and `-npu-fold-batchnorm` folds the
original and declines on the variant. The 2.98e-08 is the fold's own movement
showing up on one side of a comparison between two programs that still compute
the same function.

So the tolerance is real and it is narrow, and which levels it applies at is
read out of the compiler rather than written down as a list of level numbers.
`-O0` and `-O1` keep byte equality, and a level that stopped folding would get
it back with no edit.

**The declare then re-record procedure has a wrinkle nobody had met yet.** The
baseline records the pass and fail count of every suite, and this phase's own
schema commit makes five assertions about the committed baseline fail until the
re-record lands. So the first `regression-baseline.sh` run recorded a red pytest
suite and warned about it in its own words: "a baseline recorded from a red tree
records what is broken as if it were correct". The fix is to record twice, the
second run reading the file the first wrote. P13 and P14 carry the same step and
will meet the same wrinkle; it is in `docs/PHASE_STATE.md` for them.

**Three assertions were left to move and two of them moved.** The third,
`test_import_and_npu_are_the_same_text_at_minus_o_zero`, was expected to go red
because the driver's `npu` stage now runs `npu-opt` at every level rather than
returning the importer's text. It did not: the importer already prints
locations, so its output round trips through `npu-opt` byte for byte. That is a
stronger property than the test was written to assert, and I kept the assertion
and rewrote the docstring rather than weakening it to something about operation
lists. A prediction that turns out unnecessary is a finding, and the reason it
was unnecessary is worth more than the prediction was.

**What this phase did not find is worth as much as what it did.** Two of the
eight ablatable passes will produce ablation rows of exactly zero at P10.
`-npu-fuse-bias` fires on no model of the suite, because every convolution in it
carries its bias inline as a third `Conv` input, which is what both
`torch.onnx.export` and this project's ONNX built models emit; the gate's clause
is that it fires on a real imported model and it does, on one built for it. And
`-sccp` cannot fire on a single function program at all, whatever the dialect
can materialise.

Neither is a defect and both would read as one in a results table, so both are
in `docs/PHASE_STATE.md`'s open questions with the fix and its cost:
`dilated_stack`'s `conv1` is already biasless and followed by a `Relu`, so one
`Add` node would give the bias fusion a target in the suite, at the price of a
`GENERATOR_VERSION` bump, every `dilated_stack` cell, and possibly a tight budget
`docs/adr/0008` froze. That is a decision for the phase that owns the
measurement, which is P10, and not one to take quietly inside the phase that
wrote the pass.

## 2026-09-01 Interphase P9b: four decisions, and three of the four rehearsals disagreed with me

**What this branch is.** Four items the P9 handoff carried as open questions,
taken between P9 and P10 rather than folded into either: a package in the CI
image, an NDEBUG CI build, the model suite gap that made `-npu-fuse-bias`
unfireable, and whether `regression-baseline --check` can be a CI step. None of
them is a feature and none of them has a gate. What they have in common is that
each had reached the point where leaving it open cost more than deciding it.

**The entry's finding, and it goes first because it is the only general one.**
Three of the six rehearsals on this branch came out differently from the
prediction written before them, and in all three cases the disagreement was worth
more than the confirmation would have been. One deleted a deliverable, one
narrowed a claim, and one turned into a defect. P8 recorded that a local
rehearsal proves a step's logic and cannot prove its surroundings. This branch
adds the other half: **a rehearsal whose prediction holds tells you what you
already believed, and the value of the practice is concentrated entirely in the
runs that do not.** Writing the prediction down first is what makes that value
collectable, because a prediction reconstructed afterwards is never wrong.

### The cache key I did not commit

The image change is one package, `libomp-18-dev`, so that the two CI jobs which
configure with clang stop making a weaker determinism claim than the one that
gets gcc. That part was straightforward and the rehearsal against the pinned base
digest confirmed it: before, `clang -fopenmp` fails at `'omp.h' file not found`
and cmake prints the CI line verbatim; after, `Found OpenMP: TRUE (found version
"5.1")`.

The part that was not straightforward was what the workflow needed beside it. I
predicted that `build-and-test`'s restored build cache would keep reporting
`OpenMP: not found` after the republish, because `find_package` writes
`OpenMP_CXX_FLAGS:STRING=NOTFOUND` into `CMakeCache.txt`, the cache key is keyed
on the LLVM tag, and the tag does not move on a republish. The planned
deliverable included an image revision component in that key.

Measured in a container rather than reasoned about: configure with no `libomp`,
install it, re-run cmake on the **same** build directory. The answer is
`OpenMP: found 5.1`. `FindOpenMP` re-runs its `try_compile` when the cached flag
variable is falsy, so a `NOTFOUND` does not stick the way a `find_library` result
does.

So the deliverable was deleted rather than committed. A cache key bumped for a
mechanism that does not exist would have invalidated every build cache in the
project to fix nothing, and it would have been invisible: the runs after it would
have been slower and green, and nobody would ever have gone looking for the
reason. **The failure mode of an unnecessary fix is that it works.**

### CI had been naming a build mode it did not have

The NDEBUG item was supposed to be a decision about a second LLVM tree. It became
that plus a defect, D-0036, and the defect is the more interesting half.

`ci.yml` said, in two places since P7, that the sanitizers job was the NDEBUG half
of Section 9.3's "every build mode" clause, "which configures RelWithDebInfo and
therefore compiles with `-DNDEBUG`". The therefore does not hold here. D-0028,
found and fixed **in that same phase**, is exactly the reason: against an
assertions LLVM, `HandleLLVMOptions` appends `-UNDEBUG` after whatever the build
type supplied, and the last `-D` or `-U` on the line wins. Measured:

```
-DNDEBUG -D_DEBUG -D_GLIBCXX_ASSERTIONS -UNDEBUG
```

So three CI jobs ran the trap tests, all three with assertions on, and the file
that decides what CI does asserted otherwise in prose. An accessor that had
quietly become assert-only would have been green in every one of them.

**Why it survived a phase is the part to remember.** The comment was written by
somebody who had just fixed D-0028, and the reasoning in it is correct for a
project whose LLVM has no assertions. Correct general knowledge applied to a
specific configuration is not a thing review catches, because it reads as
knowledge. What catches it is a mechanism, and the new `ndebug` job is one: it
greps its own configure log for the line beginning `NDEBUG:`, so the claim is now
made by a build that either has assertions compiled out or fails.

The second LLVM tree D-0031 named is declined, in ADR 0009, on the grounds that
Section 9.3's contract lives entirely in the two binaries that link no MLIR and
those are exactly the two this directory builds soundly. D-0031 stays open as the
limit it is.

### One test caught the fault, and I had predicted three

The product side activation fault for the new job compiles the range trap's
diagnostic out behind `#ifdef NDEBUG` in `readBytes`, leaving the check and the
null return in place so every caller still behaves. It is the shape a release
build acquires when somebody decides a diagnostic is a debug convenience, and no
assertions build can see it.

Prediction: the assertions build green, the NDEBUG build red at three graceful
trap tests. Result: the assertions build green, the NDEBUG build red at
**one**, `Trap.AnOutOfRangeOperandAddressTrapsGracefully`.

The arithmetic is the finding. The fault is in `readBytes` alone; the two result
address tests go through `writeBytes`. So exactly the one test whose fault this
was caught it, and a prediction of three would have been satisfied by a net
firing for reasons other than the fault. This is the same lesson D-0029 taught
from the other end, where a rehearsal reported 54 of 54 mismatches for a fault
that should have produced three or four and the discrepancy was a second defect.
**A rehearsal is not a pass or fail. The number is the measurement.**

### The suite change was the easy half and the tight budget was the question

`-npu-fuse-bias` fired on no model of Section 15's suite because every convolution
in it carries its bias inline, which is what exporters emit. One `Add` between
`dilated_stack`'s biasless `conv1` and its `Relu` closes that, and the governance
around it is four commits in ground rule 7's order: declare, move, re-record,
record the re-measurement.

Two things came out of measuring rather than assuming.

**The addend cannot be rank 1.** ONNX broadcasting aligns from the trailing axis,
so a `(5,)` initializer broadcasts against the width of 6 and `onnx.checker`
refuses the graph outright. It is written `(1, 5, 1, 1)`, which is what an
exported graph carries anyway and which the importer normalises to the rank 1
constant the pass guards on. The P9 handoff's sketch said "an `Add` of a rank 1
initializer", and the pass does see rank 1; the generator cannot write it.

**The tight budget did not move, and that was not obvious.** ADR 0008 froze
`dilated_stack` at 8064 and the P9 handoff named the budget as the thing this
change might break. Re-measured by that record's own 64 byte sweep at all three
levels: peak 8036, floor 8064, which are the P8 numbers. The mechanism is that a
sweep line peak is a maximum over time and not a sum over the program. The two
new buffers are 20 bytes of bias and a 360 byte destination, both live near the
end where the working set is a little over five kilobytes, and this model's peak
is set by `conv0`, whose input, filter and result are resident together.

The prettiest number on the branch is the `-O2` cycle count. `dilated_stack` gains
an instruction at `-O0` and `-O1` and the fusion takes it back at `-O2`, where the
cycle count and the compute cycle count return to **exactly** what they were
before the node existed: 1234.0625 and 710.8125. The twenty extra DRAM bytes stay
at every level, because the bias has to arrive whichever operation reads it.

### The verification matrix found the last defect by dying

`coverage.sh` aborted at collection with a `SuspiciousHits` stack trace naming
`lib/Simulator/Kernels.cpp:87`, exit 64, no percentage. gcov says why: that line's
counter had reached 5896524226, past gcovr's threshold of 2^32.

gcov accumulates. A `.gcda` left in place is added to by the next run rather than
replaced, and the script had never deleted one, so `build-coverage/` held the sum
of every run since P8 in one set of counters. `Kernels.cpp:87` is the stride loop
inside the odometer, the hottest line in the project, and it got there first.

**The crash is the harmless half.** A percentage collected from accumulated
counters is a percentage about the union of every suite the directory has ever
run, so a line executed by a test that was later deleted keeps the count that
executed it. This branch deletes a test, which is what makes the point concrete.
The measurable evidence is small and real: 42 `.gcda` files before the deletion
and 41 after a clean run, so one object had counters and no longer runs at all.

Measured either way the number did not move on this tree, C++ 86.5 and Python
90.50, so this is a defect in what the number **means** rather than in what it
currently **says**. That is the kind that survives until somebody looks.

**And it is D-0030 and D-0032 with the direction reversed.** Those were results
that depended on what else was lying around in CI, invisible on a developer
machine. This one depends on what is lying around on a developer machine and is
invisible in CI, because the coverage job checks out a fresh tree and has nothing
to inherit. The class is the same and the environments are swapped, which is
worth knowing about a class this project has now met four times: the question to
ask of any measurement is not "did it pass" but "what else could have produced
this result".

### What was deliberately left undecided

`regression-baseline --check` is a CI step now and `GOLDEN_TOLERANCE` is still
zero. P8 and P9 both left the step out for the same stated reason, that a byte
identical golden comparison bounds two runs of the same build and CI is a
different compiler against a different libc, and neither phase could answer
whether that matters because answering it needs a run.

Choosing a tolerance in advance would have thrown away the only measurement the
question is about. So the step goes on with the band unwidened and the first
container run is the experiment. What that required was making a red first run
readable from the log alone, because a red run nobody can interpret is a step
that gets disabled. The golden drift line said `largest movement 4.7e-08`, which
is the same sentence for one element moving in its last bit and for every element
moving, and those are opposite findings. It now names the count, the index, both
values and the movement in ulps, proven by pushing one golden element to the next
representable float and reading what came out.

## 2026-09-01 Interphase P9b, first CI run: the answer and the defect arrived together

Run 33458934438. `lint`, `sanitizers` and `coverage` green, the `ndebug` job
**green on its debut** at 1 minute 5 seconds, and the
`regression-baseline --check` step red, job 99704772026.

**The answer first, because the red nearly buried it.** The step reported 42
cells, 21 golden tensors and a largest movement against `-O0` of 4.470e-08, with
**zero drift on every numeric field**. A baseline recorded under gcc on WSL2
reproduces **bit for bit** under clang in the container, across two compilers,
two libcs and two hosts. That is the question P8 raised, P9 sharpened and neither
could answer, and it is answered in the affirmative. `GOLDEN_TOLERANCE` stays at
zero on evidence rather than on principle.

**Which is the first thing worth recording about the run.** The interesting
result was on stdout, above the failure, and the exit code was about something
else entirely. The handoff's watch plan listed three outcomes as alternatives;
two fired in the same run and the plan had not imagined that. **When reading a
red `--check`, read the whole report and not the verdict.** The cells and the
goldens are printed before the verdict precisely so a run that fails for one
reason still delivers the measurement it was asked for, and that design paid for
itself on its first outing.

### The defect, and the four mechanisms it was not

The recorded suite table says `dash-lint 2 passed 0 failed`; the container
reported `0 passed 2 failed`, while the `lint` job's own dash lint steps were
green in the same run. Four mechanisms were plausible, and ruling each out was
most of the work: the locale affecting the unicode scan, `PATH`, a GNU `grep -P`
dependence, and the working directory.

None of them. `LANG` is unset in the container and Python still reports `utf-8`
for both the filesystem and the preferred encoding, so the unicode scan was never
at risk, and there is no `grep` in the linter at all: it shells out to
`git ls-files` and scans in Python. The mechanism is one line.

```
File "/work/scripts/dash_lint.py", line 235
    except subprocess.CalledProcessError, FileNotFoundError:
SyntaxError: multiple exception types must be parenthesized
```

Two unparenthesised `except A, B:` clauses. PEP 758 made that legal in **Python
3.14** and it is a `SyntaxError` everywhere below it. The CI image is Ubuntu
24.04 and ships **3.12**; `dash-lint.sh` falls back to `python3` on `PATH`
because CI calls it before any venv exists. The module did not parse, so the
linter never ran, and both invocations failed for the same reason, which is why
the count was zero and two rather than one of each.

**Why it had been green everywhere.** Three of the four places that run this
linter use 3.14: the developer machine and pre-commit through the venv, the
`lint` job through `actions/setup-python`. The `lint` job is not in a container,
and nothing else in `build-and-test` invoked the linter. **Nothing had ever run
it inside the image**, and the `--check` step is the first thing in this
project's history that did.

### The part that deserved the real fix

`pyproject.toml` declared `requires-python = ">=3.11"` and then configured black,
ruff and mypy alike at `py314`. The promise sat in a field nothing reads while
every checker pointed at the developer's interpreter, so no tool could have
objected.

The comment beside `target-version` argued for py314 **deliberately**, against
the v1 tree's py312, on the grounds that an older grammar target applied to code
running on a 3.14 interpreter shows up as a formatting argument rather than as an
actionable error. That reasoning is sound and it looks in one direction only.
**The interpreter that matters is the lowest one that runs the code, not the
highest**, and this project's lowest is a container nobody was thinking about
when the comment was written.

Both grammar targets are `py311` now, which is what `requires-python` says, so
the floor is enforced by a tool rather than promised in a field. Measured before
the fix: at py311 ruff finds exactly those two errors over the whole tree and
nothing else, at py312 it finds them again, and
`black --check --target-version py311` leaves all forty two files unchanged. The
mechanism costs nothing, which is the argument for taking it rather than fixing
two lines and moving on.

mypy is `3.12` and not `3.11`, and the reason is worth keeping. At
`--python-version 3.11` it stops inside **numpy's own shipped stubs**,
`numpy/__init__.pyi:737: Type statement is only supported in Python 3.12 and
greater`, and checks nothing further. So the floor this project can type check
against is set by a dependency rather than by its own code, which leaves
`requires-python` and the checkable floor a minor version apart. That is recorded
as an open question rather than resolved by quietly moving a published contract.

### The half that made it a hunt

The CI log said `suite dash-lint: passed 2 -> 0` and nothing more. Every other
suite the baseline runs writes a machine readable file, so a failing test arrives
in the drift report by name; this one has none and contributes a count, and the
`SyntaxError` that explained everything went to a pipe nobody read.

`run_dash_lint` prints the child's output on failure now. That is the same
standard the golden drift lines were rewritten to meet earlier in this branch,
and the omission is instructive: the standard was applied to the failure mode
that was being thought about and not to the one that was not. **A rule about
diagnosability is worth applying to every output a step has, including the one
that has never failed.**

### What this says about the step, and about the branch

The step was added to answer a floating point question. It answered that on its
first run and then caught a defect with nothing to do with it, in a script five
phases old, because it is the first thing that ever ran that script in that
environment. **A step that runs the whole suite somewhere new is worth more than
the reason it was added for.**

That makes four defects on this branch and the set has a shape. D-0030 and
D-0032 were results that depended on what else was lying around in CI, invisible
locally. D-0037 is the mirror, invisible in CI. D-0038 is a third position: code
correct in every environment anybody had run it in and wrong in the one nobody
had. D-0036 is the fourth, a claim never measured anywhere at all. **In none of
the four was the code under test the thing that was wrong**, and in all four the
fix was to make an environment or a claim checkable by a tool rather than by a
habit.

## 2026-09-01 Interphase P9b, the activation proofs: two green runs were one sample

Runs 33461200759 and 33461203436, the two `pull_request` proofs for the job and
the step this branch switched on. **Both intended faults fired exactly as
predicted**, which is the boring half: the NDEBUG fault took the `ndebug` job red
at its GoogleTest step with `build-and-test`'s assertions half unbothered beside
it, and the golden ulp fault produced the rewritten drift line with its element
count, index, both values and the movement in ulps.

**And both runs reported eighteen differences nobody predicted**, the same
eighteen cells in both, in one field: `max_abs_error_vs_onnxruntime` on
`conv_bn_relu_stack`, `inception_block` and `resnet_block`, at every level and
both budgets, moving between 1e-8 and 1e-7 **in both directions**. No golden
tensor drifted. No cycle count moved.

### The diagnosis was free, and that is the design working

**The goldens are what made it certain rather than plausible.** They pin this
compiler's answer bit for bit at a tolerance of zero and they were green, so the
end of that distance which belongs to this project did not move and the end that
did is `onnxruntime`'s. `onnxruntime` dispatches its CPU kernels on what the host
supports, and GitHub's hosted runners are not homogeneous.

Two bands, two questions, one measurement each: that structure was argued into
`regression_baseline.py` at P9 and it is what turned an eighteen line red into a
one sentence diagnosis. **A comparison that keeps its subjects apart can tell you
which subject moved.** Had the goldens been compared at the same loose band as
the field, this would have been undiagnosable from the log.

### The knowledge was already in the repository

`test/Python/test_end_to_end.py` set its tolerances at ten and six times the
observed maxima at P8, and the comment beside them says, in these words: "this
suite runs on at least two hosts, the developer machine and the CI container, and
`onnxruntime` chooses its own vectorisation per host. A bound two times the
observed value on one machine is a bound that goes red on another for a reason
that is not a defect."

The regression baseline then recorded the same quantity and compared it at a
bound of **zero**. One file argued for a wide band on a number and another
asserted equality on it, two phases apart, and nothing connected them. That is
the finding and it is not really about `onnxruntime`: **a project can hold the
right answer in one file and the wrong assumption in another indefinitely,
because nothing makes two files disagree out loud.**

So the fix connects them rather than restating the conclusion. The constants
moved into `npu_frontend.tolerances` and both the matrix and the baseline import
one object, which is D-0032's rule applied to a number instead of to a function.
A tolerance is the worst thing in a project to have two of, because the copies
agree until somebody widens one and then the looser wins wherever it is read.

### Two green runs were one sample

**The first CI run reported zero drift on every numeric field** and the handoff
recorded the cross host question as answered on that basis. It was not wrong
about the numbers; it was wrong about how many samples it had. Both pushes
happened to land on runner hardware matching the recording host. The proof runs
were the third and fourth samples and their hardware differed.

The answer is two part now and the second part took four runs to see. **The
compiler and the simulator are bit stable across hosts**, which is a stronger
property than this project had evidence for and which `GOLDEN_TOLERANCE` at zero
now rests on. **The distance to the oracle is a property of the measuring host**,
and asserting equality on it made the step flaky per runner.

It generalises past this field. A green run on a fleet of heterogeneous machines
is a sample from a distribution, and "it was green" is a statement about the
sample. Nothing in the watch plan said how many runs an answer needs; the honest
number here was four, and it needed runs whose hardware happened to differ.

### What was deliberately not changed

`GOLDEN_TOLERANCE` stays at zero and every other cell field stays compared for
equality. Those proved themselves on the same runs, and widening anything else
because one field turned out to be badly posed would have thrown away the result
the runs actually delivered.

The field is bounded against Section 17.4's absolute end to end band, which is
the only thing it ever meant, and a movement inside the band is **printed** with
its magnitude and its direction rather than passed over. A check that has been
switched off has to say so in its own output: Section 19.0's rule about silence
and success, applied to a field rather than to a step. Rehearsed three ways,
including a synthetic per host shift built from the magnitudes CI reported and a
band tightened to 1e-9 so the real values fall outside it and the field goes red.

**What it costs, stated rather than glossed:** the field can no longer catch an
`onnxruntime` upgrade that moved the oracle. `tool_versions` records the version
and `requirements-lock.txt` is where an upgrade shows up in review. Neither is as
loud, and that is the price of the fix.

## 2026-09-01 The P9b closing runs: two proofs, one compound red, and a republished image

The branch's first run put the new pieces through CI against the old image. The
`ndebug` job was green on its debut in 1 minute 5 seconds, and the
`regression-baseline --check` step went red at D-0038, whose story the previous
entry carries:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33458934438>.
The fix's run, green in every job with the dash lint suite back at 2 passed:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33460570488>.

The two activation proofs ran as pull requests 15 and 16, drafts, closed
unmerged with their branches deleted. **The first attempt at both was a
compound red**: each intended fault fired exactly as predicted, and the
`--check` step failed beside it with 18 unpredicted `max_abs_error_vs_onnxruntime`
differences, both directions, three models, which is D-0039's finding: the
proof runs landed on different runner hardware than the recording runs, and the
oracle's own kernel dispatch moved. Those first runs, kept because a proof that
found something is worth more than a proof that matched:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33461200759>
and
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33461203436>.

After D-0039's fix, both rebased and rerun, each red at exactly its own step
and nowhere else. The NDEBUG product fault, red at the `ndebug` job's GoogleTest
step with the assertions build unbothered:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33464534276>.
The one ulp golden fault, red at the `--check` step with its single drift line
and no host noise behind it:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33464534594>.
The branch run beside them, green on whatever hardware it drew, which is the
point of D-0039's fix:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33464530309>.

The image republish ran from the branch by dispatch, ref
`phase/p9b-debt`, and published digest
`sha256:844aff90b5c422e133ec38527cf459a635b5f47cb580f3aebe445e4d94fc1e35`
under the reused tag:
<https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/33459558320>.
Every run above started before the publish landed and pulled the old image, so
their `OpenMP: not found` lines say nothing about the new one. The run that
carries this very commit is the first to pull the new digest, and the reading
that closes item 1 is its configure lines: `OpenMP: found` in build and test,
sanitizers and ndebug, `NPUSimulatorTests` reporting its thread count, and
coverage still on gcc's libgomp as before.

## 2026-09-01 Phase P10: instrumenting a pass manager an out of tree tool does not own

**Symptom, before anything went wrong.** Section 16.2 requires the per pass
operation counts to be computed by a `PassInstrumentation` in `runBeforePass` and
`runAfterPass`, on the pass manager `npu-compile` actually runs. `npu-compile`
runs `npu-opt`, and `npu-opt` was four lines around `MlirOptMain`, which builds
its own `PassManager` inside `performActions` and never hands it to the caller.
There is no `addInstrumentation` an out of tree tool can reach.

**Four routes were considered and three were rejected for reasons worth keeping.**

1. **Run one pass at a time from Python and subtract.** This is the arrangement
   Section 16.2 rejects by name: quadratic in the pass count, and it measures a
   pipeline that is not the one under test, which is the same fault Section 17.4
   names from the test side.
2. **Give `npu-opt` a second entry point that builds its own `PassManager` from
   `pipeline::build()`.** It would work and the pipeline would be the same table,
   but it would be a second path through which every future flag has to be
   threaded twice, and the two would drift.
3. **`--print-op-stats`.** Named in an earlier draft of the specification and
   corrected there. It prints one summary for one invocation. There is no before
   and after pair per pass in it, so a gate written against it is unmeetable.
4. **`MlirOptMainConfig::setPassPipelineSetupFn`**, which is the hook
   `performActions` calls with the real manager immediately before running it.
   This is what was used.

**The chosen fix.** `npu-opt` unrolls the four argument `MlirOptMain` into what it
does, which is `registerAndParseCLIOptions` followed by the five argument form,
and wraps the config's pipeline setup callback: the wrapper installs the
instrumentation on the manager it is given and then calls the callback the
command line had already installed. **The unrolled path runs only when
`--npu-pass-stats-json` is given.** The unrolled form does not reproduce
`--show-dialects` and `--list-passes`, which are answered inside the library by
functions an out of tree tool cannot call, and losing two flags to gain one would
be a bad trade. `test/Pipeline/pass-stats.mlir` diffs the tool's output with and
without the flag, so the claim that the default path did not move is a check
rather than a sentence.

### Two decisions inside the instrumentation, both measured

**Adaptors are filtered on the pass having no command line argument, not on a
type test.** MLIR instruments the `OpToOpPassAdaptor` that wraps a nested
pipeline exactly the way it instruments a real pass, so an unfiltered
instrumentation records every nested run twice. MLIR's own `PassTiming` writes a
type test against `OpToOpPassAdaptor`; that type is declared only in
`mlir/lib/Pass/PassDetail.h`, which an out of tree build does not get. An adaptor
is not registered and therefore has no command line argument, while every pass in
`lib/Pipeline/Pipeline.cpp` has one and is keyed on it in that table, so filtering
on the argument asks the same question through the field the pipeline description
already uses, and the two agree by construction.

**The operation walk is outside the timed span**, and this is the decision the
cross check is built on. Counting is a full traversal, which on the larger models
costs as much as a cheap pass. MLIR's timer opens before this instrumentation is
called and closes after it, so MLIR's figure per pass **contains** this project's
walk and this project's does not. That gives the comparison a direction: MLIR's
number is always the larger, and the difference is this file's own cost. A per
pass figure that came out larger than MLIR's, or smaller by more than the tree
display's resolution, would mean the two were not measuring the same run.

**Measured on 2026-09-01**, all seven models at all three levels, with both flags
on one invocation so that the two clocks describe the same execution:

```
worst gap, MLIR minus the instrumentation   0.069 ms, on a pass MLIR timed at 1.1 ms
worst case MLIR came out below              0.034 ms, inside the display resolution
```

The bound is a floor of 0.15 ms plus half of MLIR's figure, rather than one
absolute number. The floor is the display's own resolution, since the tree prints
seconds to four places. The fraction is the walk, which is work proportional to
the module and therefore to the time the pass took. An absolute bound would have
been a bound that held here and went red on a slower machine for a reason that is
not a defect, which is the mistake `tolerances.py` already records having been
made once about `onnxruntime`.

### The ablation could not refuse in C++, so it is checked by measurement

`PipelineOptions::ablatedPass` skips one ablatable entry when the pipeline is
built. It cannot refuse a request to remove `-npu-lower-to-npuisa`, because
MLIR's `PassPipelineRegistration` builder returns nothing and there is no path
from inside it to a readable command line error, and a fatal error raised from a
library is a worse answer than none.

What closes the hole is on the other side and is better than a refusal would have
been. The driver reads the ablatable set out of the compiler at run time and
refuses by name before it runs anything, and **every ablation run is instrumented
and its recorded pass list is compared against the level's list minus the named
pass**. An ablation that quietly did nothing is a raise naming the pass, not a
row of zeros that reads as a pass with no effect. The refusal is a claim; the
comparison is a measurement, and this phase is about preferring the second.

**Verification.** `ninja check-npu` 26 discovered, 26 passed, up from 25 with
`test/Pipeline/pass-stats.mlir`. `test/Python/test_pass_instrumentation.py` 26
passed, including the doctored file that drops one pass, the ten doctored files
that each drop one field, and the sweep that ablates each of the eight and reads
the removal back out of the instrumentation.

## 2026-09-01 Phase P10: the matrix has a hole in it, and the hole is the tight budget

**Symptom.** The first run of `experiments/run_benchmarks.py` died five cells in:

```
error: loc("clipped"): the scratchpad budget of 8064 bytes is too small: this
buffer of 16016 bytes could not be placed below offset 0 in @main, and no buffer
live across the pressure peak can be spilled. The sweep line peak is 32032
bytes, which is a lower bound on any placement
```

`dilated_stack`, `-O0`, tight budget, batch 4.

**Root cause, and it is a specification question rather than a defect.** Section
17.4's matrix sweeps the scratchpad budget and the batch size as two independent
axes. They are not independent. ADR 0008 defines a tight budget by measurement:
the smallest budget in a 64 byte sweep at which the model still allocates. That
is a property of **a program**, and `TIGHT_BUDGETS` is keyed by model, with every
entry measured at that model's own declared batch. A model at batch 4 is a
different program.

Measured across the suite before deciding anything:

| Model | Peak at declared batch | Peak at batch 4 | Allocates at batch 4 |
|---|---|---|---|
| `lenet` | 194592 | 200800 | no |
| `depthwise_separable` | 8192 | 32768 | no |
| `resnet_block` | 8480 | 26912 | no |
| `inception_block` | 6848 | 24576 | no |
| `conv_bn_relu_stack` | 6432 | 18720 | no |
| `dilated_stack` | 8036 | 32080 | no |
| `lenet_batched` | 200800 | 200800 | yes |

**`lenet` is the case that settles the argument.** Its peak barely moves, from
194592 to 200800, because a 400 by 120 weight matrix sets it and no batch size
touches that. It still fails, by six kilobytes. So the scaling factor is not the
batch, and any formula for a batch 4 tight budget would have been a constant
derived from a relationship this project had just measured to be false.

**Three options, and the reason each was taken or refused.**

1. *Scale the batch 1 budget by the batch.* Refused on the `lenet` row above.
2. *Extend ADR 0008's sweep to batch 4 and freeze seven more constants.* This
   keeps the full cross product and moves nothing already recorded, and it was
   the closest call. Refused because `docs/PHASE_STATE.md` hands re-measurement
   of the tight budgets to P13, and P13 is the phase that makes a budget below
   the peak reachable at all by tiling instead of spilling. Constants frozen here
   would be constants P13 invalidates, and every tight budget cell measured
   against them would have to be thrown away.
3. *A cell that names the tight budget runs at the model's declared batch.*
   Taken, and recorded as `docs/adr/0010`.

**ADR 0008's own procedure was re-run before the decision, and it reproduces all
seven of its constants to the byte** at each model's declared batch. That is
worth as much as the decision: it says the batch 4 failures are the rule working
rather than the rule having rotted, and it revalidates a P8 measurement against a
model suite P9b changed.

The suite is 175 cells rather than the 84 plus 112 a free cross product gives,
and `docs/adr/0010` carries the arithmetic.

## 2026-09-01 Phase P10: two of four predicted claims were wrong, and the instrumentation said why

`experiments/predictions/p10-ablation-deltas.md` was committed at `f92de42`,
before `run_benchmarks.py` had been run once. Two of its four claims are wrong.
The file is not edited; this entry is the adjudication.

### Claim 1 is wrong: two rows are not zero, not three

Predicted `-npu-fuse-bias`, `-npu-fold-batchnorm` and `-canonicalize`. Measured,
as instruction count deltas at both budgets:

| Pass | Nonzero on | Delta |
|---|---|---|
| `-npu-fuse-bias` | `dilated_stack` | 1 |
| `-npu-fold-batchnorm` | `conv_bn_relu_stack` | 8 |
| every other ablatable pass | nothing | 0 |

`-canonicalize` is **zero on all seven models at both budgets**, and the
prediction said it would be the surest of the three.

**The instrumentation gives the mechanism, which a delta alone could not.** The
before and after counts on `conv_bn_relu_stack` at `-O2`:

```
with canonicalize            without canonicalize
  npu-fuse-ops    34 -> 38     npu-fuse-ops    34 -> 38
  canonicalize    38 -> 24
  cse             24 -> 21     cse             38 -> 21
```

The second canonicalization is doing real work, fourteen operations of it. It is
simply not doing anything `-cse` cannot also do: MLIR's CSE erases trivially dead
operations as it walks, so with the canonicalization removed it arrives at the
same twenty one operations on its own. **The two passes are redundant on this
suite for this purpose, and a leave one out ablation cannot see a pass whose work
another pass would have done.** That is a known limit of leave one out designs
and it is now a measured one here rather than a caveat: the honest reading of
`-canonicalize`'s zero row is not "canonicalization does nothing" but "nothing in
this suite needs both".

This is the clearest payment the Section 16.2 instrumentation has made so far.
Without a before and after pair per pass, the row would have read as a pass with
no effect, and the entry in `docs/PASSES.md` would have said so.

### Claim 3 is wrong: sixteen of fifty six rows differ between the budgets

Predicted that every ablation row is identical at both budgets because nothing
spills at either. Measured: `resnet_block` and `inception_block` differ, on every
one of the eight passes.

```
resnet_block     default  instr=14  cycles=1626.00  spills=0  dram=8800
resnet_block     tight    instr=17  cycles=2018.00  spills=1  dram=14944
inception_block  default  instr=14  cycles=2398.50  spills=0  dram=8624
inception_block  tight    instr=22  cycles=3799.00  spills=3  dram=21936
```

**ADR 0008 contained the refutation and I misread my own table.** Those are
exactly the two models it identifies as able to go below their peak, with one and
three spills recorded there. The prediction asserted no spilling from the fact
that five of seven cannot go below their peak, and forgot the two that can.

The precise statement, which the prediction should have made: the ablation
**deltas** are identical at both budgets, all zero for the six passes with zero
rows and unchanged for the two with nonzero rows. The **absolutes** are not,
because the tight budget adds spill DMA. This is Section 16.2's own reason for
requiring ablation rows at every budget, arriving as evidence rather than as a
rule: a table that reported only the generous budget would have shown
`inception_block` at 14 instructions and never mentioned the 22.

### Claim 2 is met, with one wording defect worth recording

No ablation moved any cell outside the 5e-5 end to end band, so the run did not
fail on numerics. Ablating `-npu-fold-batchnorm` returns
`max_abs_movement_vs_o0` to exactly 0.0 on `conv_bn_relu_stack`, as predicted:
the fold is the one pass in this pipeline that moves numbers, and removing it
restores `-O0`'s answer bit for bit.

The wording defect: the prediction added "and no other ablation moves that field
at all". Fourteen ablation rows carry 4.470e-08 in it. They are not moving it;
they are leaving it at the unablated `-O2` value, because they leave the fold in
place. Read as "no other ablation changes the field relative to the unablated
cell" the claim holds exactly. Read literally against zero it is false. **A
prediction that can be read two ways has one reading too many**, and the fix is
to the next prediction rather than to this one.

### Claim 4 is met

Suite total at `-O2`, default budget, declared batch: **117 instructions**,
predicted between 100 and 130. No single ablation raises it by more than 8,
predicted no more than 30.

### The observation P9 asked P10's report to state out loud

`-O1` is exactly `-O0` on every model in the suite, measured here at 25, 12, 14,
14, 23, 13 and 25 instructions at both levels. `-O2` differs from `-O0` on
exactly two models, `conv_bn_relu_stack` at 23 to 15 and `dilated_stack` at 13 to
12, and those two savings are the two nonzero ablation rows seen from the other
side.

### Two numbers Section 16.1 predicted about itself

`top1_agreement_vs_onnxruntime` is **1.0** and
`cosine_similarity_vs_onnxruntime` is **0.99999999999996**. Section 16.1 forbids
ranking configurations on either and gives the reason in advance: this suite's
models are seeded and never trained, so argmax agreement saturates and cosine
similarity parks at four nines with no resolution. Both numbers are recorded, the
sentence forbidding their use is recorded in every file beside them, and SQNR,
which is 136.87 dB on the same cell, is the metric that moves.

### The suite's own cost

175 cells in **1.76 minutes**, **0.60 seconds per cell**, serially, against the 90
minute budget of Section 2. The worst gap between the instrumentation's clock and
`--mlir-timing` over all 175 cells and all 10 trials each was **0.1881 ms**, on
`NPUFuseOps`, inside the floor plus half bound.

**0.60 seconds per cell is the measured figure that replaces Section 2's 15
second planning number.** The spec file lives outside this repository and is not
edited from here; `docs/PHASE_STATE.md` carries the replacement text for the
owner.

## 2026-09-01 Phase P10 closing: the number Section 2 asks for, and where it is

**This entry exists because a gate clause cannot be met from inside this
repository, and recording that plainly is better than a phase that quietly
reports itself complete.**

P10's gate says "the measured per cell cost replaces the 15 second planning
figure in Section 2 in the same commit". Section 2 is in the build specification,
which lives outside this repository on the Windows side, and the standing rule is
that it is not edited from here. So the measurement is recorded and the
replacement text is written out, and applying it needs the owner.

**The number.**

```
175 cells, 105.699 seconds, serially
0.60 seconds per cell
against a budget of 90 minutes, which is a factor of 51 in hand
```

Measured on the 14700K under WSL2, at commit `d4210f3`, and reproduced to two
decimal places on a second run at a different commit. The figure lives in
`experiments/results-runtime.json` rather than only in this entry, because a
number that exists in a log and nowhere machine readable is a number the next
phase retypes.

**Two more corrections Section 2 needs, and they are about the cell count rather
than the cost.** Both are already recorded where the work happened, and are
gathered here because whoever edits Section 2 needs all three at once.

1. **Eleven ablatable passes becomes eight.** Section 12's table has eleven, but
   `-npu-assign-layout`, `-npu-tile-to-scratchpad` and `-npu-double-buffer`
   arrive at P13 and no `-O` level names them yet, because a level that named a
   pass nothing implements would give the ablation table a row it could not fill.
   154 ablation cells becomes 112.
2. **Budget and batch are not independent axes**, which is `docs/adr/0010`. 84
   benchmark cells becomes 63.

238 becomes 175.

**The replacement text is in `docs/PHASE_STATE.md`** under "The Section 2 carve
out, for the owner", written as a paragraph that can be dropped in rather than as
a list of edits to reconstruct.

### Why the suite is serial, since the budget is what pays for it

Every cell carries a `compile_ms` object with ten trials, a median and a
percentile interval, and each pass in it carries the same. Cells competing for
cores measure the contention rather than the compiler, so parallelising this
suite would corrupt the one group of fields it exists to produce. That is a
deliberate departure from `scripts/regression-baseline.sh`, which parallelises
freely at `NPU_BASELINE_JOBS` because it records no timing at all.

The decision was affordable because of the measurement rather than in spite of
it: at 0.60 seconds per cell the whole suite is under two minutes, and there is
no version of this trade that is close. Had the suite come out at forty minutes
serial and ten parallel, the right answer would have been to keep it serial and
say so, because the alternative is timing objects that describe a machine's load
rather than a compiler's cost.

### What the phase found, in one place

Three findings, and none of them is in the compiler.

**`-canonicalize`'s ablation row is zero and the pass is not idle.** It removes
fourteen operations on `conv_bn_relu_stack`, and `-cse` reaches the same program
without it because MLIR's CSE erases trivially dead operations as it walks. A
leave one out ablation cannot see a pass whose work another pass would have done.
This is the clearest thing the Section 16.2 instrumentation has bought: without a
before and after count per pass, this row is a zero indistinguishable from
`-sccp`'s structural zero, and `docs/PASSES.md` would have recorded them the same
way.

**The tight budget does not cross the batch axis**, `docs/adr/0010`. Six of the
seven models do not allocate at batch 4 under their recorded tight budget, and
`lenet` shows why no formula would have worked: its peak barely moves with the
batch, from 194592 to 200800, because a 400 by 120 weight matrix sets it, and it
still fails by six kilobytes.

**Seven tests were marked slow at P3 and CI has never run one**, D-0040, found by
a prediction that was wrong about how many tests carry the marker.

### The prediction, and what being wrong bought

Two of `p10-ablation-deltas`'s four claims are wrong. The full adjudication is in
the previous entry and in `docs/PHASE_STATE.md`; what belongs here is what the
mechanism was worth.

**A prediction that had been written after the measurement would have named two
nonzero rows and looked prescient.** The registered one named three, and the
third being wrong is what sent me to the per pass operation counts, which is
where the `-cse` redundancy is visible. The finding is a consequence of the
prediction being wrong and committed beforehand, which is the entire argument for
Section 17.8's protocol arriving as a mechanism rather than as an intention.

The same for claim 3. Predicting the budgets would agree, and finding sixteen
rows where they do not, produced the spill table now in `docs/PASSES.md` under
`-npu-allocate-scratchpad`. Section 16.2 already said to run ablations at every
budget because passes can behave oppositely at a tight one; this project now has
its own evidence for that rule rather than taking the specification's word.

**One wording defect in the prediction, recorded because the next prediction
should not repeat it.** Claim 2 ended "and no other ablation moves that field at
all". Fourteen rows carry 4.470e-08 in it. They are not moving it, they are
leaving it where the unablated cell has it, so the claim holds under one reading
and fails under the other. A prediction that can be read two ways has one reading
too many, and the fix belongs to the next prediction rather than to this one.

### Verification, and the thing the re-record proves

The closing matrix is in `docs/PHASE_STATE.md`. One line of it is worth calling
out. `regression-baseline` was re-recorded at `5401d39`, and the diff is four
kinds of line: `git_sha`, `check-npu` 25 to 26, `pytest` 871 to 954, and 84 test
names. **Not one cell field moved and all 21 golden tensors are byte identical.**

A phase that put a `PassInstrumentation` on the pass manager, unrolled
`npu-opt`'s entry point to reach it, added a pipeline option, a result schema, a
benchmark harness and ninety tests, and moved no recorded number, is a phase
whose changes are provably confined to the measuring apparatus. That is the
strongest single statement available about a measurement phase, and it is the
reason the re-record is its own commit with nothing else in it.

## 2026-09-02 D-0041: the first CI run of P10's tests, and a wrong answer that looked right

**Symptom.** The first push of `phase/p10-measurement`, run 33559636835, went red
with eight unique failures across the `pytest` and `pytest slow cells` arms. Every
one was in a file this phase added. Every one was green locally, at the same
commit, on the same suite.

```
test_every_manifest_git_sha_resolves       git_sha d4210f3... is not a commit in this repository
test_every_entry_landed_in_a_commit...     the prediction is not in any commit
test_every_result_that_names_a_prediction  prediction_sha f92de42... not a commit
test_a_prediction_landing_commit_is_an...  assert None is not None
test_the_ancestor_check_refuses_a_sha...   a parent is an ancestor of its child
test_the_macros_sha_resolves_to_a_real...  exit 128
test_a_rerun_is_byte_identical...          assert 2 == 0
test_the_run_fails_when_it_exceeds...      exit 2
```

**The fifth line is the one that gives it away.** "A parent is an ancestor of its
child" is not a claim that can be false. A test asserting it and failing is a test
whose environment is wrong, not whose logic is.

**Root cause.** `actions/checkout` defaults to `fetch-depth: 1`. The checkout
holds one commit and no history. P10 is the first phase whose tests ask questions
about history at all: law 3 of Section 0.2 asserts that every published number
traces to a commit that resolves, and law 4's mechanism is an ancestry relation
between the commit a prediction landed in and the commit a result was measured at.
Nine phases of green CI say nothing about this, because none of them asked.

The last two failures are the same cause one step removed:
`experiments/run_benchmarks.py` reads the prediction's landing commit before it
measures anything and exits 2 when it cannot, so both slow tests failed on the
harness refusing rather than on anything they were testing.

### Reproduced before anything was changed

```
$ git clone --depth 1 file:///home/elijah/npu-mlir-v2 /tmp/p10-shallow
$ git -C /tmp/p10-shallow rev-parse --is-shallow-repository
true
$ git -C /tmp/p10-shallow rev-list --count HEAD
1
```

Four of the eight fall straight out. The other two did not reproduce, which is
what sent me to the second shape: the `pull_request` trigger checks out a
synthetic merge commit whose parents are the base and the branch, and that is a
different graph from a `push`. Both shapes were built as real fetches from a bare
mirror rather than as mocks, because what is under test is what git does rather
than what this project believes git does.

| Shape | depth | shallow | `landing_sha` returns | historical sha resolves |
|---|---|---|---|---|
| `push` | 1 | yes | the graft commit | no |
| `push` | 0 | no | `f92de42`, correct | yes |
| `pull_request` merge ref | 1 | yes | the merge commit | no |
| `pull_request` merge ref | 0 | no | `f92de42`, correct | yes |

The merge ref at full depth answers everything correctly, including `HEAD~1`,
which on a merge commit is the base branch: a parent, and an ancestor, so the
refusal test holds there too. That was worth checking rather than assuming,
because the ancestor assertions are the ones a merge commit could plausibly have
broken.

### The second defect, which is the one worth the entry

The checkout depth is a setting. What the table above exposed is not.

**`landing_sha` did not fail in a shallow checkout. It returned an answer.**
`git log --diff-filter=A -- <path>` attributes every file to the graft commit,
because that commit has no parent to have differed from, so the function returned
the checkout's own tip. On the merge ref it returned the merge commit for an entry
that landed six commits earlier.

That sha would have gone into a result's `prediction_sha`. It resolves, it is an
ancestor of HEAD, and **the ancestor test would have passed on it**, while the
provenance link pointed at the wrong commit. A green run recording a false link is
worse than the red run that actually happened, and no `fetch-depth` prevents it,
because the function was willing to answer a question it could not answer.

The same fault in two milder forms sat beside it. `commit_exists` returned False,
which reads as "this commit does not exist" when the truth is "this commit is not
here". `is_ancestor` returned False, because `git merge-base --is-ancestor` exits
nonzero both for "no" and for "I have never heard of that commit", so
"unobservable" was reported as "the prediction does not predate its measurement",
which is a serious finding invented out of a checkout option.

**This is the fault this project forbids everywhere else**, an absent measurement
that cannot be told apart from a real one, appearing in the one mechanism whose
entire purpose is provenance. Section 16.1 spends a paragraph on it for result
fields and `values_of` refuses to average a null; the same discipline had not been
applied to a git query.

### The fix, both halves, and why depth 0 rather than something narrower

`fetch-depth: 0` on the three jobs that run the suite: `build-and-test` and
`coverage` in `ci.yml`, `full-matrix` in `nightly.yml`. The other four checkouts
stay at the default, because no step in them asks about history.

A narrower fetch was considered and rejected. It would have to name the commits to
deepen to, and those commits are the shas recorded in result files and prediction
entries, so the fetch would need updating every time a result is re-recorded and
would be wrong in exactly the situation it exists to serve. The repository is a
few hundred commits and the full fetch costs seconds.

`require_full_history` is the other half and is the part that does not depend on
a workflow file being right. It refuses once, readably, naming the checkout and
the fix, before any of the three functions answers. `is_ancestor` raises on an
unresolvable reference instead of returning False. `landing_sha` refuses instead
of returning the graft commit. A shallow checkout now produces one diagnosable
message instead of eight assertions about shas, and the message says
`fetch-depth: 0` and names the defect.

### What the fix was verified against

All four shapes, as real fetches. At depth 0, both `push` and `pull_request`:
**43 passed, 0 failed**, and `run_benchmarks.py` completes and exits 0. At depth
1, after the fix: the refusal by name rather than the eight failures.

`test_the_ancestor_check_refuses_to_guess_in_a_shallow_checkout` makes a real
`--depth 1` clone inside the test and asserts all three refusals, so this is
exercised on every run of the suite rather than only in this entry. That test is
the reason the entry can claim the guard works rather than that it was written.

### The pattern, now four deep

D-0030 and D-0032 depended on what was lying around in CI and were invisible
locally. D-0037 was the reverse. D-0040 was a test that only ever ran locally.
This is the fourth and the sharpest: code correct in every environment where the
history is present, run for the first time in one where it is not.

**In all four the code under test was fine and the environment differed.** That is
the argument for CI existing, and it is also the argument for this project's habit
of writing down what a red run actually measured rather than what it was expected
to measure. The eight failures looked like eight problems and were one, and the
one that mattered was not in the list at all.

## 2026-09-02 D-0042: the same mistake one day later, and a probe that could not have told

**Symptom.** The second CI run, 33571635111, on the commit that fixed D-0041. The
same eight tests red. But every fact had moved: `fetch-depth: 0` had taken and the
checkout log showed a full fetch of `+refs/heads/*` with no `--depth`, the commits
existed on the remote, and they were ancestors of the pushed tip. And the tests
were printing D-0041's **new** message:

```
not a commit in this repository. The repository is not shallow, so the
commit is genuinely absent rather than merely unfetched.
```

Every clause of that sentence was false. The repository was not shallow because
git could not tell us whether it was shallow; the commit was not absent; and the
one thing the message ruled out, "merely unfetched", was the only thing that had
actually been fixed.

**The datum that cracked it.** `test_the_macros_sha_resolves_to_a_real_commit`
failed with `assert 128 == 0`. Exit **128** is git's fatal, not its "no". A
genuinely missing object gives 1. Every other failure was a boolean, so this was
the only place in eight failures where the raw exit code survived to the log.

**Root cause, and it is two things again.**

The runner's shape is a workspace owned by the runner user with the job running as
root inside a container. git refuses a repository whose owner is not the caller
and exits 128 with `detected dubious ownership`. This project read any nonzero
exit as the answer "no".

Reproduced in the pinned image with the workspace chowned to uid 1001 and the
container as root, running the helpers exactly as the pytest step does:

```
repository_is_shallow       -> False        should have refused
commit_exists(present)      -> False        the commit is there
is_ancestor(pred, harness)  -> "genuinely absent"
head_sha                    -> ""           empty string
landing_sha(p10)            -> None         the assert None is not None
require_full_history        -> passed       it could not see the shallow flag either
```

Six wrong answers out of one unreadable repository, and they are exactly the eight
CI failures. `rev-parse --is-shallow-repository` is the worst of them: it prints
its fatal to **stdout**, so a caller comparing stdout against `"true"` gets False,
which is why D-0041's guard, one day old, passed and let everything downstream
run.

### Why the first run's log pointed away from git

Two things made this look like anything but a git refusal, and both have the same
explanation.

`regression-baseline --check` runs git in the same job and printed shas happily,
which reads as ruling out a git problem in that job. And the failures named
specific shas as absent, which reads as a data problem.

It is a step ordering. `git config --global --add safe.directory` was first set by
the `DIALECT_REFERENCE.md staleness` step at line 577. `pytest` is at 463 and
`pytest slow cells` at 520. **Every step above line 577 had a repository git would
not read, and every step below it was fine.** The suite was the only thing above
that line that asked git anything, and `--check` is at 684.

### The probe could not have made the distinction anyway

The first attempt at this fix read exit 1 as absence and 128 as a refusal, which
is correct and was still not enough. Measured on 2026-09-02:

| invocation | present | absent | unreadable |
|---|---|---|---|
| `cat-file -e <sha>^{commit}` | 0 | **128** | **128** |
| `cat-file -e <sha>` | 0 | 1 | 128 |
| `rev-parse --verify --quiet <sha>^{commit}` | 0 | **1** | **128** |

`cat-file -e` with a `^{commit}` peel returns 128 for a well formed but absent
object, the same code an unreadable repository gives. **The exit codes were being
read correctly and the tool was wrong.** The test that caught this asserted that a
well formed absent sha comes back as absent rather than as a refusal, and it went
red against `cat-file`; that assertion is what chose the probe.

`rev-parse --verify --quiet` separates the three cases and, as a bonus, answers 1
for a sha that resolves to a blob rather than a commit, which is the right answer
to "is this a commit" and one more thing `cat-file -e <sha>` would have said yes
to.

### The fix

`_git` is one runner for every git call in the module, taking the set of exit
codes that are genuinely answers and raising with git's own stderr otherwise. git
explains itself better than a paraphrase and its message carries the fix, which is
the same argument the golden drift lines and `run_dash_lint` were rewritten under
at P9b.

`commit_exists` uses `rev-parse --verify --quiet`.

`safe.directory` is set immediately after the checkout in all three container jobs
that run the suite. The `coverage` job had never set it at all, and would have
failed the same way the moment anyone read its log past the percentage.

`run_benchmarks.git_sha` raises rather than returning `"unknown"`. That string
would have gone into the manifest of every committed cell, and law 3 is that every
published number traces to a commit that resolves.

**Verified in the pinned image, workspace uid 1001, container as root**, which is
the runner's shape:

| | before | after |
|---|---|---|
| `repository_is_shallow` | `False` | refuses, quoting git |
| `commit_exists(present)` | `False` | refuses; `True` once trusted |
| `is_ancestor` | "genuinely absent" | refuses; `True` once trusted |
| `head_sha` | `""` | refuses; the sha |
| `landing_sha` | `None` | refuses; `f92de427d1f3` |

**The left column is the result worth having.** With the environment fault still
present, nothing is answered wrongly any more. The two halves of the fix are
independent, which is what makes the code half worth writing at all.

### The lesson, and this project already knew it

This is the second time in two days that a nonzero exit was folded into a boolean.
D-0041 was `is_ancestor` returning False for "cannot tell" in a shallow checkout;
D-0042 is the same sentence with a different cause, plus a probe that could not
have told either way.

The lesson is not about git. **A helper that returns a bool for a question with
three answers will eventually be asked the third one.** Section 16.1 spends a
paragraph forbidding exactly this for result fields, `values_of` refuses to
average a `null` because of it, and `docs/NUMBERS.md` says no number on the page
is an estimate. The discipline existed, was written down, was enforced by a test,
and had not been applied to a subprocess call.

There is a narrower lesson too, and it is the one worth carrying into P11, which
adds two external tools that are both shelled out to. **When a subprocess can
fail for a reason that is not the question, the return type has to have somewhere
to put that.** Neither Accelergy nor SCALE-Sim will be more reliable than git.

## 2026-09-02 D-0043: comparing a rounded sum against an exact one

**Symptom.** Third CI run, 33575891610. `build-and-test` green, including every
D-0041 and D-0042 test, so those two are proven in CI. The coverage job red on
one test:

```
test_pass_instrumentation.py:273: AssertionError: assert 5.3 >= 5.301691
```

**The two numbers are the diagnosis.** `5.3` carries one decimal. `5.301691`
carries six. The margin is 1.7 microseconds. Those are not two measurements that
disagree; they are one measurement and one rounding of another, compared as
though both were exact.

**Root cause.** `--mlir-timing` prints seconds to four decimal places. Every
figure this project parses out of it is therefore a multiple of 0.1 ms and stands
within 0.05 ms of a number it cannot see. Measured, the eleven values of an `-O2`
run: `0.1, 0.2, 0.1, 0.1, 0.4, 0.4, 0.2, 0.3, 0.2, 0.5, 0.2`. Not one has a digit
finer than the quantum. The instrumentation's own figures are `steady_clock`
differences at microsecond resolution.

The direction the check asserts is sound and is not what changed. MLIR's timer
opens before this instrumentation is called and closes after it, so
`true_mlir >= instrumented`. The error was comparing against the **printed**
figure as though it were the true one. Per pass that is worth half a unit in the
last place; over a sum of eleven it is worth eleven halves, which is 0.55 ms. The
assertion allowed nothing at all.

**Reproduced under the build CI failed on**, ten runs of one cell against
`build-coverage`:

```
run 0: mlir  4.1000  instr 3.978771  shortfall -0.121229
run 2: mlir  5.5000  instr 5.297579  shortfall -0.202421
run 3: mlir  3.2000  instr 3.262634  shortfall +0.062634   <-- fails a strict >=
run 7: mlir  3.0000  instr 2.859018  shortfall -0.140982
```

One in ten. gcov is not special: it makes each pass slower and noisier, which
moves the sum of eleven roundings across zero often enough to be seen.
`build-and-test` has been running the same flawed assertion since P10 landed and
passing it by luck, which is the more uncomfortable half of this. **A bound that
is wrong by construction can be green for days.**

### Two magic numbers beside it, which were the same fault milder

`TIMING_RESOLUTION_MS = 0.15` and `TIMING_FLOOR_MS = 0.15` were the per pass
bounds. Both were chosen loosely, at three times a quantum nobody had written
down, and their docstring called 0.15 "the display's own resolution", which is
wrong: the resolution is 0.1 ms and half of it is what a rounding can cost. They
were generous enough never to have fired, which is why nothing pointed at them.

### The fix, and why it is derived rather than set

The quantum is read off the text that was parsed, per report.
`parse_mlir_timing` returns a `TimingReport` carrying the rows, the decimals it
actually saw, and half a unit in the last place computed from them.

| bound | before | after |
|---|---|---|
| per pass, MLIR below ours | `0.15` | `half_ulp` |
| per pass, MLIR above ours | `0.15 + 0.5 * mlir` | `half_ulp + 0.5 * mlir` |
| totals | strict `>=` | `>= instrumented - n * half_ulp` |

Against those, over the ten coverage runs: worst per pass deficit **0.039971 ms**
against 0.05, worst total shortfall **+0.062634 ms** against 0.55.

**Deriving it is not ceremony.** A report printed to two decimals of seconds has
a quantum of 10 ms, and a constant of 0.05 would go on asserting a precision the
figures no longer had, which is this defect again rather than a smaller version
of it. A test feeds the parser a two decimal report and asserts the half unit
comes back as 5 ms.

**And the bound got tighter, not looser.** Per pass it went from 0.15 to 0.05.
The containment argument makes 0.05 exact rather than cautious:
`printed >= true - half_ulp >= instrumented - half_ulp`. Anything beyond it still
fails and still means the two clocks are not measuring the same run. The cross
check clause of P10's gate is only worth having if its bound is principled, and
widening to whatever made the observed failure pass would have thrown that away
to save one red run.

### The pattern, and this is the fourth

D-0040 was a test that only ever ran in one environment. D-0041 was a helper
answering a question it could not observe. D-0042 was the same helper answering a
different unobservable question through a probe that could not have told either
way. This one is a comparison that ignored the precision of one of its two
operands.

They are all the same shape: **a value arrived through a channel that loses
information, and the code treated it as though it had not.** An exit code that
collapses three answers into two. A figure printed to four decimals. Section 16.1
forbids exactly this for result fields, which is why `instruction_count` is an
integer and every wall clock is an object with an interval saying how uncertain
it is. The schema had the discipline; the code around it kept not having it.

The narrow lesson for P11, which shells out to two more tools and parses their
output: **whatever reads an external tool's numbers has to carry that tool's
precision alongside them**, or every comparison downstream silently assumes an
exactness that was never there. SCALE-Sim's cycle counts and Accelergy's energy
figures will arrive through exactly this kind of channel.

## 2026-09-02 Phase P11: the roofline first, and what it turned out to be worth

Section 16.6 says to build the roofline before the other three tools, because it
is the frame they are presented inside and building the frame after the pictures
is how the pictures end up framing themselves. So it went first, before either
external tool had finished installing, and it needed nothing from them.

### The per layer number had to be reconstructed, and then checked

`Stats` reports the program's totals. The roofline wants a bound per layer and
the SCALE-Sim exporter wants the shape of every convolution, and neither is in a
result file. So `python/npu_frontend/npuisa_walk.py` walks the allocated
`npuisa` module and charges each operation through the Python mirror of
`CostModel.h`, which is one walker for both consumers rather than two sets of
shapes to keep in agreement.

A reconstruction is worth nothing until it is checked against the thing it
reconstructs, so `check_against_result` compares the walk with the cell: raw
MACs, DRAM bytes read and written, and the instruction count exactly, because
those are counts on both sides and a band on a count hides the disagreement it
was added to tolerate. Then `effective_macs`, `utilization` and `delta` within a
band derived from double accumulation, because those three are what actually
exercise the convolution charge. `macs` would agree even if both occupancy terms
had been dropped.

`conv2d_charge` was missing from the Python mirror and had been since P7. The
C++ has had it all along. That is not a defect, because nothing had needed it and
the mirror never claimed to carry it, but it is the shape of gap that becomes one
the moment a phase needs the number: the mirror test compared constants and one
matmul, so a convolution charge could have drifted for four phases without a
single test noticing. It now reproduces the recorded `effective_macs`,
`utilization` and `delta` of four real cells exactly.

### The first roofline run was red, and the roofline was wrong

Every matmul in `lenet` came out below its bound, by a factor of three. The
reading that would have been convenient is that the cost model charges too little
for a matmul. The reading that is true is that the comparison had two different
quantities in it: `node_linear` charges 3404 cycles for the arithmetic over a
400 by 120 weight matrix, and the 192480 bytes of that matrix take 12030 cycles
to move. The bytes were not moved faster than the interconnect. They were moved
by a `dma_load` the comparison had left out.

Section 5.5 accumulates cycles on two independent ports and reports the later of
them at `HALT`, so the time one layer occupies is the later of the array's charge
for its arithmetic and the DMA port's charge for its bytes. Comparing the compute
charge alone against a bound with a memory branch in it is comparing a compute
time against a memory time. The fix is one `max`, and the wrong version is
recorded here because it produced 175 confident red rows that looked exactly
like a finding.

### What the check is actually worth, said plainly

Under the cost model of Section 5.5, `effective_macs` is **defined** as
`cycles * peak`. So the compute branch of the roofline, `effective_macs / peak`,
is identically the kernel's own cycle count, and the comparison is a tautology up
to the four cycle issue overhead. The memory branch is no better placed: a
transfer is charged bytes over bandwidth **plus** a descriptor, so it always
costs strictly more than the bound its own bytes produce.

**The roofline as specified cannot fail against this cost model.** Section 16.6
warns that the compute branch is partly the cost model grading its own homework;
the measurement is that both branches are, entirely. Writing that down is
preferable to reporting 175 green cells as though they were 175 pieces of
evidence.

It is not worth nothing. It is a **regression** bound, and the phase it is
waiting for is P13: double buffering and tiling are exactly the changes that
could produce a charge no longer built from the traffic it moves, and the day one
does, this check goes red for a real reason. Both halves of the tautology are
asserted in `test/Python/test_roofline.py` rather than only described, so the day
either stops holding a test says so instead of a docstring going quietly out of
date.

### Numbers

175 cells, 550 MAC bearing layers. 175 layers and 61 cells are bound by the
memory branch and the rest by the compute branch. The tightest layer in the suite
is `lenet`'s first convolution at 0.000635 headroom over its compute bound, which
is the issue overhead and nothing else, which is what the tautology predicts.

## 2026-09-02 Phase P11: two external tools, and both of them lie by exiting zero

The install was started before anything else in the phase, because Section 23
says P11 blocks on it entirely. It took under two minutes of wall clock and then
cost most of a session anyway, for reasons that had nothing to do with download
time.

### SCALE-Sim does not run

Not on this project's topology. On **its own shipped example**, at the pinned
sha, on every upstream branch:

```
self.total_cycles = int(max(ofmap_serviced_cycles))
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

numpy 2 removed `int()` on arrays of rank one or more, and three expressions in
the tool use it. The choice was between moving numpy, which would make all 175
committed results describe an environment that no longer exists, and a three
expression patch to the install. Ubuntu 26.04 ships CPython 3.14 only and no
numpy 1.x wheel exists for it, so the second interpreter that would have avoided
the choice was not available. D-0044 carries it. Every manifest now records
`scalesim_installed_tree_sha256` beside the upstream sha, so the record says the
tool was modified rather than showing a sha that does not describe the code that
ran.

### And both tools report failure by exiting zero

While measuring the above, the tool was run with a missing input file:

```
ERROR: scalesim.scale.py: Layout file not found
Exiting
```

Exit status **0**. `scale_sim.set_params` calls the builtin `exit()`, which is
status zero, so a hard input error and a successful run are reported the same
way. An uncaught exception exits 1, which makes the two failure modes report
differently from each other, which is worse than either.

Accelergy does the same thing in a different place. Its own shipped basic example
crashes on this install, prints `Accelergy has encountered an error and crashed`,
and exits **0**; other failures exit 255.

**This is D-0040 through D-0043's shape arriving from outside the project.** The
lesson those four left for P11 was that whatever reads an external tool's numbers
has to carry that tool's precision alongside them. The stronger version, which
this phase learned the hard way, is that it has to carry the tool's **failure
modes** too, and that an exit status is a channel that loses information the same
way a rounded figure is. So neither wrapper reads a status. SCALE-Sim's requires
`COMPUTE_REPORT.csv` to exist, to carry the header the parser was written
against, and to hold exactly one row per exported layer. Accelergy's requires
three output files and every component it asked about to appear in each. Both
raise with the tool's own stdout and stderr, because the tool explains itself
better than a paraphrase would.

### Accelergy 0.4 ships no primitive component library

`~/npu-venv/share/accelergy/primitive_component_libs` does not exist. A primitive
component therefore has no declared action list, the energy reference table comes
out as `tables: []`, and the energy calculator then asserts that it cannot find
an entry for the first component. That is why the architecture description is
three compound classes with their actions written out rather than three bare
primitives, and the docstring says so where somebody would otherwise simplify it
back.

### The comparison had to be made honest before it meant anything

The first SCALE-Sim export copied the activation's own height and width across.
SCALE-Sim's topology has no padding field and no batch field and this machine has
both, so the two tools were being charged for different amounts of arithmetic,
and every divergence figure would have carried that difference without naming it.
The extents are now derived from the output positions instead, so SCALE-Sim's own
output size formula produces exactly this layer's output count with the batch
folded into the row dimension, which is what the cost model of Section 5.5 does.

Two checks hold that in place and one of them can fail. `check_macs` asserts the
exported topology implies the MAC count this project charged.
`check_the_same_arithmetic` reconciles SCALE-Sim's **own** utilization figure
against that count, which closes the loop from the other side: the export
describes this program, and the tool agreed about what it was given. Without
those, a divergence figure could be two tools answering about two different
layers, and it would have a plausible size and no cause.

### The decomposition counted one effect twice

`dilated_stack` came out with a residual of minus 1838 cycles against named terms
of plus 1524 and minus 1410. A decomposition whose residual is the same order as
its largest term has counted something twice, and it had: the fragmentation term
was computed against the dilated topology, which does three times the multiplies,
so it absorbed the dilation approximation a second time. Every term except the
dilation one is now taken from the arithmetic matched run, and the residual is
zero on every cell.

**The residual being zero is not a result and the module says so.** The terms are
a partition of the difference, not a fit to it, so a nonzero residual would mean
cycles were lost between them. The check that can actually fail is
`check_the_same_arithmetic`.

### What the comparison found

D-0045, and it is the thing cross validation exists for: this project charges the
array's weight preload **once per instruction** and SCALE-Sim charges it **per
fold**. On `resnet_block`'s 3 by 3 convolution the two accounts differ by about a
factor of three. It is not fixed here. Retuning a cost model against an external
tool invalidates every ablation already recorded, and Section 16.5 states that
rule for ZigZag in the same words. P13 gets the reproduction.

### The prediction was mostly wrong

Direction wrong on five of seven models, all three magnitude bands wrong, both
rank fidelity figures wrong, the coverage floor on `lenet` wrong. What it got
right was the mechanism behind the widest positive gaps, the treatment of
pooling, and the existence of a fragmentation disagreement.

The entry was not edited. That is the whole of Section 17.8 and it is worth
saying plainly: a prediction written before the measurement, that turned out to
be mostly wrong, and that is answered as written, is more informative than a
prediction that was right, because the places it is wrong are where this project
learned something.

### The fp32 MAC coefficient fails the sanity check and is reported failing

49.286 pJ against a published 4.6 pJ for a multiply plus an add at 45 nm, a
factor of 10.71, where Section 16.4 asks for within an order of magnitude. The
cause is identifiable and is not this project: Aladdin's number is a synthesised
three stage pipelined unit at a 1 ns clock with its registers, and the published
figure is a combinational datapath. The scratchpad and DRAM coefficients both
pass.

The bound was not widened. The test pins the measured value so that it moving is
a failure, and separately asserts the ratio is still above ten so that
`docs/NUMBERS.md` going out of date is a failure too. And `docs/NUMBERS.md` says
what the overstatement means for every conclusion drawn from these numbers: at
the published coefficient the scratchpad would be the largest consumer on every
model in the suite, so nothing in this project may rest on the array being
dominant.

### Two smaller things worth recording

A DRAM byte count that is not a whole number of accesses raised, on the reasoning
that rounding must never invent or discard one. `dilated_stack` moves 5004 bytes,
because 1251 floats is 5004 bytes, and the refusal was reading a remainder as a
bug. A DRAM cannot fetch part of a word, so a partial access is paid in full and
the count rounds up.

And a coverage threshold failed the run on `lenet` at 0.9548, quoting the
prediction's own falsifier. The exporter is representing nothing it should not.
A covered layer's cycles are the later of its compute and the DMA that feeds it,
on both sides, and `lenet`'s matmuls are dominated by loading a 400 by 120 weight
matrix. The op fraction is 0.208 and is the figure the clause was reaching for.
The threshold went; both fractions and the explanation stayed.

### One flake, recorded because it will recur

The first re-record run died at cell 74 on the `--mlir-timing` cross check:
`NPUFuseBias` at 0.3000 ms against the instrumentation's 0.0843, a gap of 0.2157
against a bound of 0.2000. The machine was running SCALE-Sim in another process
at the time. Two subsequent runs on a quiet machine reported worst gaps of 0.1577
and 0.1177 ms and passed. The bound is D-0043's and is principled; it is also
sensitive to load, and the suite should not be run beside the external tools.

## 2026-09-03 D-0046: the phase was green only where the machine was special

The first CI run of this branch was red in three jobs. One cause, and it is one
this project has now met three times.

I installed two external tools on this machine and then, without deciding to,
wrote three things that assume they are there. A test of the **budget gate** and
a test of **rerun determinism** both drove the whole harness, so both invoked
Accelergy, so both died on `FileNotFoundError: 'accelergy'` in an image that has
never had it. A `# type: ignore[import-untyped]` on the `scalesim` import was
correct here and wrong there **twice**: the error that fires without the package
is `import-not-found`, which that code does not cover, and `warn_unused_ignores`
then reports the ignore itself. And the coverage job was the same two tests
again.

**None of it was visible locally, and that is the whole shape.** D-0032 was three
copies of a tool lookup where one failed and two skipped. D-0040 was seven tests
marked slow that CI had never run. This is both at once: tests that could only
pass where the author's machine was special, and no mechanism anywhere to say
which environments are supposed to have the tools.

### The fix is a policy, not three patches

`test/Python/tools.py` already owns this project's answer to "the tool is not
here": **skip when nobody promised it, fail when somebody did.** That file now
carries the same rule for the external tools, under `NPU_EXTERNAL_TOOLS`. A tool
counts as reachable only when the module imports **and** its binary is on
`PATH`, because Accelergy is driven as a subprocess and an importable package
with no binary is a tool this project cannot actually run. Reporting that as
present is how the failure moved from a readable skip to a `FileNotFoundError`
in the middle of a benchmark.

The two harness tests now pass `--skip-external`, which is Section 16.4's opt out
and exists for exactly this. What that gives up is the determinism of the P11
fields, so it is recovered rather than lost:
`test_a_rerun_reproduces_the_external_fields_too` runs the same comparison with
those fields included and is guarded by the policy above. And because two tests
now depend on a flag, the flag got its own contract test: the fields are present
and null, every reason names `--skip-external`, and `values_of` still refuses
them. A flag two tests rely on and nobody checks is the next entry in this file.

mypy gets a per module `ignore_missing_imports` override for `scalesim` and the
line level ignore goes. **No global strictness setting moved**, which was the
constraint worth keeping: the honest fix for an import that resolves in one
environment and not the other is to say so once about that module, not to relax
what the rest of the project is checked against. The first version of the
override also listed `scalesim.*`, `accelergy` and `accelergy.*` on the reasoning
that the neighbours would need it too, and `warn_unused_configs` reported all
three as unused sections. It is one module now, which is that setting doing its
job.

### Rehearsing it, which is the part that makes this finished

A defect found by an environment I cannot run is only fixed when I can run that
environment. A meta path finder that refuses `scalesim` and `accelergy`, plus a
`PATH` without the venv's `bin`, reproduces both observable facts of the CI
image. mypy needs its own reproduction because it resolves imports statically
rather than at runtime, and `--python-executable /usr/bin/python3` is what makes
it see what CI sees. Both reproduced the exact failures before anything was
changed, which is the only order in which a fix means anything.

After: **997 passed, 29 skipped, 0 failed** without the tools; **1008 passed, 18
skipped** with them; mypy clean in both; and with `NPU_EXTERNAL_TOOLS=1` set in
the tool free environment the guard **fails** naming the variable rather than
skipping, so the third branch of the policy is proven rather than asserted.

**And the coverage cluster was checked rather than assumed.** Python line
coverage without the tools was 91.5561 percent, the same figure as with them,
because `--cov=python/npu_frontend` measured the frontend package and the tool
driven code lives in `experiments/`. Nothing was hiding behind the two failures.
That measurement is also what made the next paragraph unavoidable.

**The standing lesson gains a line.** P10 left "whatever reads an external tool's
numbers has to carry that tool's precision". P11 added "and its failure modes".
This adds the one before both: **an environment that has a tool is not the
environment the project ships to**, and a suite that has only ever been green in
the richer one has not been tested.

## 2026-09-04 D-0047: the kernel was never parallel, and the test that proves it is bitwise stable was comparing two serial runs

**This phase was going to be a measurement and it turned out to be a repair.**
P12's brief is to prove the sweep line's growth, parallelise the convolution
kernel and confirm the suite runtime, with the whole thing numerically inert. The
convolution kernel had a `#pragma omp parallel for collapse(2)` in it since P7,
`find_package(OpenMP)` had been succeeding since P7, and the configure log had
been saying "the convolution kernel parallelises over the batch and output
channel dimensions" on every configure since P7. None of it was true.

### How it was found, which is the part worth copying

The first thing this phase did was measure, before changing anything. Seven
models, one thread against twenty eight, best of three:

```
model                     1t s     28t s   speedup
conv_bn_relu_stack      0.0035   0.0034      1.03
depthwise_separable     0.0016   0.0017      0.99
dilated_stack           0.0029   0.0029      1.02
inception_block         0.0035   0.0034      1.01
lenet                   0.0227   0.0231      0.98
lenet_batched           0.0847   0.0857      0.99
resnet_block            0.0045   0.0044      1.03
```

**A table of ones is exactly what small models look like**, and these models are
small: the largest is a batch of four LeNets and the whole simulation is eighty
five milliseconds. The tempting reading was there and it was wrong. What made the
difference was asking the machine a question with a yes or no answer instead:

```
$ /usr/bin/time -v ./build/bin/npu-sim lenet_batched.nbin ...
        Percent of CPU this job got: 98%
```

Ninety eight percent, with `OMP_NUM_THREADS=28`. A parallel region that finds no
work still costs more than one CPU somewhere. This was one CPU exactly, which is
not a small model, it is no parallelism.

```
$ nm -C build/lib/Simulator/CMakeFiles/obj.NPUSimulator.dir/Kernels.cpp.o \
    | grep -ci 'GOMP\|omp_'
0
$ ldd build/bin/npu-sim | grep -c gomp
0
$ grep -A4 'Kernels.cpp.o: CXX' build/build.ninja | grep -c fopenmp
0
$ grep -A4 'DeterminismTest.cpp.o: CXX' build/build.ninja | grep -c fopenmp
1
```

The last two lines are the defect. `-fopenmp` reached the **test** and never
reached the **kernel**.

### What was wrong

`add_mlir_library` compiles a library's sources in an object library named
`obj.<name>` and then assembles the library from the objects.
`lib/Simulator/CMakeLists.txt` had exactly one piece of OpenMP wiring:

```cmake
target_link_libraries(NPUSimulator PUBLIC OpenMP::OpenMP_CXX)
```

A usage requirement attached to `NPUSimulator` propagates to everything that
links `NPUSimulator`. It does not propagate to `obj.NPUSimulator`, which is a
different target and is where the four sources actually compile. So the imported
target's `-fopenmp` landed on `npu-sim`, on `NPUSimulatorTests`, on every
consumer, and on none of the kernels. `#ifdef _OPENMP` in `Kernels.cpp` was false
in every build in every environment for three phases.

**The link half worked and hid the compile half.** `NPUSimulatorTests` links
libgomp, so `omp_get_max_threads()` in `DeterminismTest.cpp` returns 28 and
`omp_set_num_threads` succeeds, and both do exactly what they are asked. They are
asked about a process that has an OpenMP runtime, which it does. Nobody asked
about the kernel.

`npu-sim` did **not** link libgomp, and that is the one place the fault was
visible from outside without a disassembler: the linker's `--as-needed` dropped a
library nothing referenced. It looked like a fact about npu-sim rather than a
fact about the kernels, and nothing was reading it.

### What it cost, which is worse than the missing speedup

`unittests/Simulator/DeterminismTest.cpp` is Section 10.3's requirement in a
test: the same input under one thread and under the maximum thread count must
produce bitwise equal output buffers. It ran two single threaded runs and
compared them. **It passed for three phases while asserting nothing.**

Its own comment header says, about the case where OpenMP is absent, that "a test
that silently becomes vacuous is worse than no test". The file was right and the
mechanism it named was not the one that got it. It guarded the case where CMake
cannot find OpenMP, which is loud, and it had no guard at all for the case where
CMake finds OpenMP and the flag does not arrive, which is silent.

`docs/PHASE_STATE.md` at P11 recorded that the determinism assertion "asserts at
full strength everywhere now, since the image has OpenMP". The CI image does have
OpenMP. The kernels did not.

### The fix, and why the third part is the one that matters

1. **`add_compile_options` at directory scope**, which is where
   `-Werror=switch` in the same file already lives for exactly this reason: a
   directory scope option applies to every target created after it, object
   library included. `separate_arguments` first, because `OpenMP_CXX_FLAGS` is
   one string and is more than one flag on some compilers. The
   `target_link_libraries` line stays; putting the runtime on a consumer's link
   line is a different job and this project needed both all along.
2. **`nbin::kernelsUseOpenMP()` and `nbin::kernelThreadCount()`**, defined in
   `Kernels.cpp`, declared in the public header, and exposed on a command line as
   `npu-sim --kernel-info`. They answer for the translation unit that has the
   parallel region in it. That is the question no caller could previously ask,
   because `_OPENMP` is a property of a translation unit and every unit that
   asked was answering correctly about itself.
3. **`Determinism.TheKernelsAgreeWithThisTestAboutOpenMP`**, which compares this
   test's `_OPENMP` against the kernels' answer and fails when they differ. The
   rehearsal is below and it is the reason this defect cannot recur silently.

### The rehearsal, with the prediction written first

**Predicted:** removing the two new lines from `lib/Simulator/CMakeLists.txt`
turns the new test red naming the object library, and leaves the two beneath it
**green**, because green is what they were in exactly this state for three
phases.

**Result:** exactly that.

```
[  FAILED  ] Determinism.TheKernelsAgreeWithThisTestAboutOpenMP
  Value of: nbin::kernelsUseOpenMP()
    Actual: false
  Expected: true
  this test was compiled with OpenMP and lib/Simulator/Kernels.cpp was not ...
[       OK ] Determinism.OneThreadAndMaxThreadsAgreeBitwise
[       OK ] Determinism.TheCycleCountDoesNotDependOnTheThreadCount
[          ] OpenMP is on and reports 28 threads available, and the kernels report 1.
```

The last line is the whole defect in one sentence, printed by the test that could
not see it. `npu-sim --kernel-info` says `kernel openmp: no` in the same tree.

### The second fault, which did not exist until the first was fixed

With the kernel finally parallel, five of the seven models ran **slower than
serial** at twenty eight threads:

```
model                  1t s    28t s   x28    (uncapped team)
conv_bn_relu_stack   0.0035   0.0094  0.36
depthwise_separable  0.0016   0.0113  0.14
dilated_stack        0.0029   0.0098  0.30
inception_block      0.0034   0.0175  0.19
lenet                0.0224   0.0186  1.20
lenet_batched        0.0847   0.0301  2.84
resnet_block         0.0045   0.0128  0.35
```

Seven times slower on `depthwise_separable`. The collapsed loop has
`batch * outputChannels` iterations; that model's two convolutions are depthwise
at batch 1, so they have **eight and sixteen** iterations and were being handed
twenty eight threads each. Every thread past the iteration count has nothing to
run and still arrives at the region's entry and its closing barrier, and on this
host that barrier across twenty eight logical processors of two different kinds
costs a few milliseconds. A few milliseconds against a simulation whose whole
arithmetic is a fraction of one.

**The cap is not a heuristic and carries no tuned constant.**
`num_threads(min(batch * outputChannels, omp_get_max_threads()))` with
`if (teamSize > 1)`. It is the number of independent output tiles the instruction
has, which is a fact about the instruction rather than about the machine, and a
threshold measured in milliseconds on a 14700K would have been the opposite kind
of thing. Neither clause can move a bit: neither changes which iterations exist,
what any one of them computes, or the order of the reductions inside it.

With the cap:

```
model                  1t s     2t s     4t s     8t s    28t s   x28   bytes
conv_bn_relu_stack   0.0035   0.0023   0.0017   0.0015   0.0015  2.36   equal
depthwise_separable  0.0016   0.0014   0.0013   0.0015   0.0019  0.86   equal
dilated_stack        0.0029   0.0021   0.0016   0.0014   0.0014  2.07   equal
inception_block      0.0034   0.0023   0.0021   0.0020   0.0019  1.77   equal
lenet                0.0233   0.0139   0.0096   0.0072   0.0074  3.17   equal
lenet_batched        0.0902   0.0511   0.0322   0.0251   0.0319  2.83   equal
resnet_block         0.0045   0.0028   0.0021   0.0018   0.0017  2.70   equal
```

**0.86 to 3.17 times, geometric mean 2.10**, and byte identical output at every
thread count on every model. `depthwise_separable` is still below one and it is
reported rather than tuned away: its whole simulation is 12800 multiply
accumulates inside a process that takes longer than that to start, and a work
threshold that fixed a three tenths of a millisecond row would be the tuned
constant the cap exists to avoid.

### Why this phase was the one that found it

Because it is the first phase whose job was to **measure** the parallelism rather
than to have it. P7 wrote the pragma and the test, both correct. P8 through P11
ran the test, which passed. Nothing between P7 and P12 had a reason to ask how
fast the simulator was, so nothing did, so the one observable that would have
disagreed was never observed.

The lesson goes on the pile with D-0040 through D-0043, and it is the same shape
a fourth time: **a value arrived through a channel that loses information, and
the code treated it as though it had not.** Here the value is "was this compiled
with OpenMP", the channel is CMake's distinction between a target and the object
library it is built from, and the two readers of that value never compared notes.
What is new is where it happened. The previous four were in scripts and tests.
This one was in the build, which is the layer everything else takes on trust, and
the only reason it surfaced at all is that somebody timed something.

## 2026-09-04 Phase P12: the allocator's growth, fitted, at the sizes the gate names

### The axis was wrong and the numbers were right

Section 13.1 asks for a compile time benchmark "at sizes 500, 1000, 2000, and
5000", and the P12 gate repeats it as "500, 1000, 2000, and 5000 **buffers**".
`experiments/compile_time_benchmark.py` has been committed since P5 and read its
`--sizes` as operation counts. The chain it generates allocates one buffer per
two operations, so the committed P5 curve was measured at **249, 499, 999 and
2499 buffers** under a table whose first column said 500, 1000, 2000 and 5000.

The P5 measurement was not wrong about anything it claimed; the exponent it
reported is the exponent of the curve it measured, and a growth exponent is a
slope, so relabelling the axis by a constant factor of two does not move it. What
was wrong was the label. `--size-unit operations` keeps the P5 table reproducible
from this script, because a correction that made an earlier measurement
irreproducible would be a second fault rather than a fix for the first.

### The curve, at the gate's sizes

```
  size     ops   buffers     pass s    total s     step   residual
   500    1002       500     0.0047     0.0209             +0.0377
  1000    2002      1000     0.0094     0.0370     1.00    -0.0343
  2000    4002      2000     0.0202     0.0696     1.10    -0.0344
  5000   10002      5000     0.0593     0.1750     1.18    +0.0311
```

Best of five per size, on the 14700K under WSL2, in an assertions enabled build.

- **Fitted growth exponent 1.1038**, r squared 0.9987, worst residual +0.0377 in
  log space.
- **References at these exact sizes**: n is 1.0000, n log n is **1.1365**, n
  squared is 2.0000.
- **The ceiling is 1.5683**, the midpoint between the n log n reference and the
  quadratic one.

**The fit is below the n log n reference**, which is the strongest form of "met"
this clause has: O is an upper bound, so growing more slowly than n log n is not
a failure, and the check is one sided for that reason.

### What a fitted exponent is worth here, and what it is not

Three things about that number, and the third is the one a reader should hold on
to.

**It is fitted over the whole curve rather than stepped between adjacent rows.**
A step is one number out of two measurements and inherits both of their noise.
The steps above rise from 1.00 to 1.18 and the fit is 1.10 with a residual
pattern that is convex, low in the middle and high at the ends, which is the
shape a genuinely superlinear term leaves. The residuals are printed per point
because an r squared of 0.9987 would otherwise let a reader believe the curve is
a power law, and it is not exactly one.

**"Consistent with O(n log n)" is a comparison against arithmetic done at these
sizes.** The effective exponent of `n log n` is not a constant: it is 1.1365 over
500 to 5000 and smaller a decade up, and a check written against a figure from
the wrong decade is a check written against nothing. The three references are
computed by the same least squares over the same four sizes, and the base of the
logarithm does not matter, because changing it multiplies every value by a
constant and moves the intercept rather than the slope. Both are asserted in
`test/Python/test_compile_time_benchmark.py`.

**A whole pass measurement cannot isolate the sweep line, and this one does not
claim to.** `NPUAllocateScratchpad` is liveness, then the sweep line, then offset
assignment, and P5 established that the linear liveness term dominates at every
size measured while the genuinely quadratic offset assignment scan does not. So
the exponent above is the pass' exponent. What it can do is exactly what Section
13.1 asks a curve at four sizes to do: separate the sweep line from the naive
nested formulation, which is O(instructions times buffers) recomputed inside the
spill loop and would sit near 2. It does that with 0.46 of margin, which is why
the ceiling is placed between the two hypotheses and not at either of them. The
sweep line's **correctness** is held elsewhere, by the property test in
`AllocatorTest.cpp` against a brute force recomputation; between the two the
claim is covered from both directions, and neither one of them alone is a proof
of a complexity class and this file does not call either one that.

### The suite, re-measured on a quiet machine

**175 cells, 3.43 minutes, 1.17 seconds per cell**, against a budget of 90
minutes. It was 1.27 seconds at P11 with the same external tools running inside
the same suite, so the factor in hand went from twenty four to twenty six. The
difference is the kernel and nothing else.

**Serialised, with nothing else running, and that is a requirement rather than
tidiness.** P11's handoff recorded that its first re-record died at cell 74 on
the `--mlir-timing` cross check with a gap of 0.2157 ms against a bound of 0.2000
while SCALE-Sim was running in another process, and that D-0043's bound is
principled **and** load sensitive. This phase makes the machine busier by design,
so every measurement run here was taken alone.

**The worst gap this run was 0.1856 ms against the 0.2000 bound.** That is inside
it and it is closer to it than P11's two quiet runs, which measured 0.1577 and
0.1177. One run is not a trend and this is not being reported as one; it is being
reported because a margin that narrowed from 0.042 ms to 0.014 ms while this
phase was putting twenty eight threads into the same machine is the kind of
coincidence that should be written down before somebody meets it as a red run.
The compile and the simulation are separate processes and run in sequence within
a cell, so there is no mechanism here that ought to couple them; that is an
argument and not a measurement, and P13 has the next data point.

### The inertness proof

**A diff of files, not a claim about them.** The 175 committed cells were copied
before the run and compared field by field after it.

```
cells compared:  175
leaf fields:     95614

moved, and permitted to:
  ci95_high_ms   1813    ci95_low_ms   1813
  iqr_ms         1813    median_ms     1813
  content_hash    175    git_sha        175    timestamp   175

MOVED AND FORBIDDEN: 0
```

Seven field names moved out of 95614 leaves, and every one is a wall clock or a
provenance record. The four `_ms` fields are the timing object Section 16.1
requires, on 1813 pass instances. `n_trials`, in the same object, did not move.
`content_hash` is a sha256 over the compiler sources and the cost model
constants, so it **had** to move, and it moving while everything underneath it
holds still is the claim rather than a counterexample to it.

Nothing else moved. Not a cycle count, a DRAM byte read or written, a scratchpad
element count, an instruction count, a MAC count, an effective MAC count, a
utilization, a delta, an overlap fraction, an energy figure, an area figure, a
roofline verdict or a SCALE-Sim cycle. **All 21 golden tensors are byte
identical**, and `git status` on `test/baseline/golden` reporting nothing is the
shortest way to see it.

**That claim would have been true at P11 for an uninteresting reason.** The
kernel was serial before and after, so a comparison across P11's re-record was
comparing two serial runs, in the same way the determinism test was. It is worth
something here because the convolution really did go parallel between the
recorded numbers and these ones. D-0047 is what makes the P12 inertness statement
a measurement instead of a tautology.

The three external tools were re-run against the new cells before the baseline
was recorded and none of them moved: the roofline still finds every cell at or
above its bound with the same tightest layer at 0.0006 headroom, SCALE-Sim still
reports -87.14 percent as its worst whole model divergence at coverage 0.711, and
Accelergy still reports 49.2860 pJ per MAC.

### Proof of failure, for the two new gates

Both were driven to their failure branches rather than argued about.

**The reduction moved into the parallel region**, which is the fault Section 10.3
forbids by name. The outer pragma was removed and
`#pragma omp parallel for reduction(+ : accumulator)` put on the input channel
loop, which is the shortest way to write the mistake somebody would actually
make. `experiments/kernel_threads.py` reported **DIFFER on all seven models** and
exited **1**; `Determinism.OneThreadAndMaxThreadsAgreeBitwise` went red in the
same tree. Both gates see it, which is what a second gate is for. Restored, tree
clean.

**`compile_time_benchmark.py --check`** exits 1 with one size, naming what to do
about it, and 1 when the fit reaches the ceiling. The ceiling branch is driven in
`test/Python/test_compile_time_benchmark.py` rather than on the command line,
because making the real allocator quadratic is not a fault injection, it is a
different program; the test hands the same function a synthetic quadratic curve
and asserts it fails, and a synthetic n log n curve and asserts it passes. That
test runs in every CI job that runs pytest, so the **discrimination** is checked
everywhere even though the **measurement** is not.

### No CI step was activated, and the trigger is written down

The growth benchmark is a wall clock measurement and the runner pool is
heterogeneous. A fitted exponent is a slope within one run on one host, so it is
not the forbidden comparison of a wall clock across hosts, and gating it in CI
would be defensible. It is still not being switched on here, for two reasons and
against one.

- The P12 gate asks for the exponent to be **reported** and consistent, and it is
  reported, with `--check` runnable and run, in the verification matrix.
- A four vCPU shared runner measuring a five millisecond pass at the smallest
  size has a noise floor this developer machine does not, and a gate that goes
  red for a reason nobody can act on is worse than no gate. Section 19.0's rule
  is that silence and success must not look alike; a flaky gate breaks the same
  rule from the other side, because a red nobody believes is a red nobody reads.
- Against: P13 adds tiling, and the P5 entry already predicted that "if P13's
  tiling pass makes functions an order of magnitude longer, this benchmark is
  already committed and will say so".

**The trigger is P13.** When tiling lands and the functions get longer, the
crossover with the quadratic offset assignment scan moves toward the measured
range, and that is the phase where a gate on this curve starts being able to
catch something. Switch it on then, under `pull_request` and `push` to `phase/**`
like every other step, and rehearse it red first by asking for `--sizes 500`,
which is the branch with no fit.

`experiments/kernel_threads.py` is not a CI step either, and the reason is
different: its gate is the byte comparison, and the byte comparison already runs
in CI as `Determinism.OneThreadAndMaxThreadsAgreeBitwise`, in process, on a
synthetic convolution, **at full strength for the first time** now that the
kernels compile with OpenMP. The script's contribution over that is the seven
real models, which is worth a nightly rather than a per push step. Trigger: the
first phase that changes the convolution kernel's loop nest, which is P13's
tiling or P14's integer kernels.

## 2026-09-04 Phase P13: the defect that was handed forward twice did not exist

### The brief, and why it changed in the first hour

P13 was briefed to fix D-0045 under the full declare then re-record governance:
an entry in `docs/BREAKING_CHANGES.md` in its own commit, the `CostModel.h`
change with its Python mirror, the divergence terms re-measured, a baseline
re-record in its own commit, and the pre-registered band of
`p11-scalesim-divergence.md` re-versioned against the new constants because
Section 16.3 requires it. Every one of those steps is conditional on a charge
moving, and none of them ran, because the charge is already what D-0045 says it
is not.

**The practice that found it is D-0047's, applied to D-0047's own successor
phase**: measure the thing before changing it. The difference here is that the
measurement took six lines of arithmetic rather than `nm` on an object file.

### The arithmetic

D-0045 says `gemmCharge` "computes `delta = m / (m + kWeightPreloadCycles)`
**once per instruction** and applies it to every tile, so the sixteen cycle
pipeline fill is amortised across the whole layer no matter how many times the
array is actually refilled". The premise is true. The conclusion does not follow
from it, and that is why the claim survived somebody reading the code: the line
it describes is exactly the line that is there.

`FrozenConstants.TheCostModelsNumbers` already asserts that the f32 peak is the
array's area, `kPeakMacsPerCycleF32 == kArrayDim * kArrayDim`. So for any tile,
whole or partial, `utilization * peak` is exactly `tileRows * tileColumns`, and
the tile's charge reduces:

```
tileMacs / (utilization * delta * peak)
    = (m * rows * columns) / (rows * columns * delta)
    = m / delta
    = m + kWeightPreloadCycles
```

With `T` folds the instruction is charged `T * (m + kWeightPreloadCycles)`. That
is the fill counted `T` times, once per refill, which is what a weight
stationary array does and what SCALE-Sim charges. **Applying the same fraction
to every tile is not the same operation as counting the fill once**, and the
entry moved from the first to the second in one sentence.

Checked numerically as well, over every combination of `m` in
{1, 2, 7, 16, 64, 196, 1024}, `k` in {1, 8, 16, 17, 72, 144, 256} and `n` in
{1, 6, 8, 16, 17, 120, 256}. **343 shapes, and the charge equals the explicit per
fold accounting on all 343.** It differs from the once per instruction
accounting on every shape with more than one fold, by exactly
`(folds - 1) * kWeightPreloadCycles`. On D-0045's own `72 by 8` weight matrix at
`m = 64`, five folds, that is 400 cycles against 336.

### The reproduction does not reproduce either, and the reason is the budget

D-0045 names `resnet_block-O2-default-n1-fp32-normal`, layer `node_conv2d`, and
quotes SCALE-Sim at **1465 cycles, overall utilization 0.098**. The committed
result for that cell and that layer says something else, and has since P11:

```
                                          scalesim   utilization   stalls
resnet_block-O2-default-n1  node_conv2d        549        0.2623        0
resnet_block-O2-tight-n1    node_conv2d       1465        0.0983      916
```

**1465 is the same layer at the tight budget, and 1465 minus 916 is 549.** The
entry crossed a tight budget SCALE-Sim reading with a default budget analytical
one. Both pairs are internally consistent, which is what made the mistake
invisible: `36864 / (1465 * 256)` is 0.0983 and `36864 / (549 * 256)` is 0.2623,
so each pair reconciles against the same MAC count and `check_the_same_arithmetic`
passes on both.

This project's own 478 is a **DMA bound** figure rather than a compute one. The
layer's `analytical_compute_cycles` is 404, of which 400 is the array and 4 is
the issue overhead; 478 is what the layer costs on the port that binds it, which
at the default budget is the transfer. So the entry compared 478 cycles of mostly
DMA against 1465 cycles of which 916 is SCALE-Sim waiting on memory, and
attributed the whole difference to the weight preload.

The 916 is not one layer's anomaly. Across the 550 layer rows of the committed
suite, **66 carry SCALE-Sim stall cycles and every one of the 66 is a tight
budget cell**; the 308 default budget rows carry none at all. Their median
divergence is -72.42 percent against +11.59 percent for the 484 that do not
stall, and the suite's total stall is 107206 cycles.

### What this says about the decomposition, which is the part worth carrying

The stall term enters `decompose()` **twice, with opposite signs**.
`array_fragmentation` is `analytical_compute - (matched_total - stalls)`, so the
stalls enter it positively; `double_buffering` is
`max(0, dma - compute) - stalls`, so they enter it negatively. They cancel in the
total, exactly, by construction.

That is worth stating beside the near cancellation the P11 report leads with.
`docs/NUMBERS.md` records double buffering at +442289 and array fragmentation at
-435825 and calls the near equality "the single most useful thing this comparison
produced". It is still a real finding, and the sign structure is part of it: the
cancellation is partly a property of how the two terms are written and not only
of the physics. Neither term is wrong. Both subtract the same stall count so that
memory time is charged once rather than twice, which is the double count the
first version of `decompose` made and its comment records.

**The stalls are not the size of either term.** 107206 against 442289 and 435825
is under a quarter, so removing them would not collapse either column. What they
are is the whole of the tight budget cells' extra divergence, and they are the
reason a layer's gap looks like a factor of three at one budget and 1.36 at the
other while the analytical charge does not move at all.

### What is still open, and it is the question D-0045 was reaching for

The two tools do disagree about the compute time of the same MAC count, with the
stalls already removed on SCALE-Sim's side. The widest rows:

```
model               layer          macs    this project   SCALE-Sim   ratio
dilated_stack       conv1          5670           140.0        1211    8.65
dilated_stack       conv0         36036           481.0        3672    7.63
inception_block     node_conv2d_2 25600          1044.0        6407    6.14
conv_bn_relu_stack  conv1         36864           404.0        1465    3.63
inception_block     node_conv2d    2048           473.0         109    0.23
```

The last row is the other direction and is the 1 by 1 convolution the P11
prediction got right about the mechanism: this project charges 4.34 times what
SCALE-Sim does there. The dilation approximation already has its own term and its
own second SCALE-Sim run at the true tap extent, so it is accounted for
separately and is not the answer to the `dilated_stack` rows either.

**Whatever the mechanism is, it is not the weight preload**, because both tools
charge that once per fold. The next phase to look at it should start by measuring
the charge rather than by reading it, which is the whole of what this session
adds to the question.

### The two assertions, and why the frozen constants test could not have caught it

`FrozenConstants.TheCostModelsNumbers` pins `kWeightPreloadCycles` at 16.0 and
says nothing about where the 16 is charged. **The accounting was never under any
assertion at all**, which is why a claim that contradicted it could stand for two
phases in the file that exists to be the audit trail.

`CostModel.TheWeightPreloadIsChargedOncePerFold` and
`test_the_weight_preload_is_charged_once_per_fold` assert the per fold accounting
**and assert it apart from the once per instruction accounting**. The second half
is the one that matters: the two accountings agree whenever there is exactly one
fold, and every shape small enough to check by hand has exactly one fold. A test
that only asserted the right answer would pass against the wrong model on every
example a reader would think to write down.

### The rehearsal, prediction written first

*Predicted:* pull `delta` out of the per tile divisor and add
`kWeightPreloadCycles` once at the end of `gemmCharge`, which is the model
D-0045 describes, and the new test goes red on the four multi fold cases while
`FrozenConstants.TheCostModelsNumbers` stays **green**, because no constant
moved.

*Result:* exactly that. The test named each shape and printed the difference:
16 cycles on `64 by 32 by 16`, 16 on `196 by 27 by 6`, 64 on D-0045's
`64 by 72 by 8`, and 2032 on the `16 by 256 by 120` tail of a fully connected
layer, each of them `(folds - 1) * 16`. The two single fold cases stayed green,
which is the reason the discriminating assertion is there.

`test_the_mirror_reproduces_the_machines_own_numbers` went red in the same tree,
because the machine moved and the Python mirror did not. **The mirror's own copy
of the per fold assertion stayed green**, since it tests the mirror rather than
the machine, and that asymmetry is worth recording: the mirror against machine
test is what couples the two, and neither per fold assertion alone would have
noticed a change made on only one side. Restored, tree clean.

### The halo arithmetic P1 declined to write

Section 7.2 has `TilingInterface` implemented at P1 and consumed at P13 so that
an interface bug and a policy bug cannot be mistaken for each other. P1 wrote the
introspection half for every operation and the generation half for the
elementwise ones, and returned failure on the windowed ones with a comment saying
the halo arithmetic belonged with the phase that would exercise it, because a
wrong tile is worse than no tile.

That arithmetic is now here, for the convolution, both pools and the matmul,
**over the parallel dimensions only**. It landed before the pass that consumes
it, which is the order this phase's commits are in.

**Section 13.2's restriction lives in the interface rather than in the pass.**
The interface is what knows whether a tile of the domain is expressible; a pass
that had to know it would be a second copy of the same judgement. A tile that
splits the input channel, the kernel window or a matmul's inner dimension is
declined, because under fp32 addition is not associative and re associating the
accumulation moves every golden file. Declining is a result rather than a
failure and the fallback is the allocator's spilling, which is Section 13.2's own
wording.

**The exactness property is asserted rather than described**, and it is what lets
the P13 gate ask for goldens byte identical rather than for a tolerance. For
every output position of every tile, the window touches the same input positions
it touched untiled, and the positions lying outside the input are the same ones.
The second half is the one an average pool depends on: it divides by the number
of elements that actually contributed rather than by the window area, so a tile
that turned a real element into a padded one would move a divisor rather than
only a sum. `Conv2DEveryTileReadsTheSamePositionsAsTheWhole` checks both over
five window shapes, every tile size that divides the output and every offset,
and it reads the slice offset off the operation the model built rather than
deriving it from the pads the model reported, so the two halves of the model have
to agree with each other rather than each with itself.

**One assertion moved with the level and one lifetime bug was found by the
harness.** `WindowedTileGenerationIsDeclinedRatherThanGuessed` asked for every
domain extent to be one, which splits the reduction as well as the parallel
dimensions, so it would have gone on passing against the new model for a reason
it did not state. It is rewritten rather than deleted, and both halves are now
asserted. Separately, the sweep parses five modules in one test and the first
version kept a tile alive across a reparse, which destroys values that still have
uses; MLIR asserted on it immediately, which is the kind of harness failure worth
having.

### ZigZag, installed and wired but not yet used

Section 16.5 blocks the cross check on an install, so it was started before
anything else in the phase, as Section 23 has the project do for P11's tools.
`zigzag-dse` 3.8.5, which imports as `zigzag`, four seconds of wall clock against
the two minutes P11's six installs took. Nothing in `requirements-lock.txt`
moved: seven pins resolved as already satisfied and five packages arrived that no
committed result was measured under and nothing in this repository imports.

**Section 16.5 asks for the 3.11 floor to be reconciled with the recorded build
environment, and it resolves the other way from the one that section
anticipates.** This environment is 3.14.4, which is above the floor rather than
below it, so there is nothing to reconcile: the wheel is `py3-none-any` and
installed with no build step of its own. The floor `zigzag-dse` sets is the
reason `requires-python` has said 3.11 since P0, and this is the first phase in
which the tool that set it is present. The other trap Section 16.5 names, that
`pip install timeloop` fetches an unrelated periodic task scheduling library, is
not one this project can fall into: it installs no Timeloop at all.

It goes in `EXTERNAL_TOOLS` rather than beside it, so every consumer of that
table follows with no further edits. That is D-0046's fix used rather than worked
around. `test_the_real_table_is_the_two_tools_this_project_installs` went red on
the addition, which is what it is for.

**The CI assertion was widened and rehearsed four ways with the prediction
written first.** This machine, all three present: exits 1 naming `scalesim`. The
CI shape, all three absent: exits 0. Only ZigZag present: exits 1 naming
`zigzag`. And the fourth, which is the one that makes the change worth making:
the **old** step body against that third shape printed "confirmed absent" in an
image that has ZigZag in it. That is the silence Section 19.0 forbids, arriving
through the one tool the assertion did not name.

### One flake, observed once and not reproduced

`test_the_opt_out_records_a_null_and_a_reason` failed once during the first full
battery run of this session and has not failed since: green alone, green with its
own file, and green in two subsequent full suite runs at 1082 passed and 18
skipped. **It is recorded rather than dismissed**, because the test asserts
`run_benchmarks.main(...) == 0` and the harness returns 1 on a finding, and one
of the findings it can produce is D-0043's `--mlir-timing` cross check, whose
bound is principled and **load sensitive**. A full suite run is the heaviest load
this repository puts on the machine.

That is a hypothesis and not a measurement, because the failure text was not
captured; the battery script tailed three lines and the reason was above them.
**The lesson is the script's, and it is fixed by capturing the failure rather
than by rerunning until it is green.** P12's handoff said to read a red at that
bound as the next data point rather than as noise, and this session cannot say
whether it was one. Recorded here so P14 can, and the flake governance of Section
17.9 at P15 is where a test that does this twice belongs.

## 2026-09-04 Phase P13: the tiling pass, and the flake that was a conditional bound

### What the pass is, and the one line that matters about it

`-npu-tile-to-scratchpad` is implemented and is in **no `-O` level**. It fires
only when an operation's working set exceeds the budget, enumerates the mapping
space exhaustively with capacity pruning, scores candidates on Section 5.5's two
port makespan through the simulator's own `gemmCharge` and `dmaCycles`, records
the chosen mapping on every tile, declines rather than splitting an fp32
reduction, and leaves no `scf` behind.

It is in no level because `-npu-lower-to-npuisa` cannot lower a tiled function.
Wiring it in would take every model in the suite from compiling to not compiling
at every budget tight enough to trigger it, so the ablatable set stays at eight
and Section 2's arithmetic stays where it is.

### The cost model moved house

`CostModel.cpp` is now `lib/CostModel`, built as `NPUCostModel`, which links
nothing. Section 5.5 says the one home "is also what the tiling pass scores
against, so the project has exactly one cost function rather than a modelled one
and a heuristic one that can disagree"; before this the only way for a pass to
reach `gemmCharge` was to link `NPUSimulator`, which would have put the machine,
its kernels, its memory and its OpenMP runtime on the link line of `npu-opt`.
**A compiler has no business linking a simulator**, which is the same argument
`lib/Simulator/CMakeLists.txt` already makes in the other direction about a
simulator linking MLIR. Its own directory rather than a second target beside the
simulator, because LLVM enforces one target per directory and discourages the
opt out for a reason this project agrees with.

### The tiles cannot be emitted as loops, and the reason is in the dialect

Section 13.2 describes the tiled loops as emitted with `scf` and then fully
unrolled inside the pass. The first version did exactly that, with
`scf::tileUsingSCF` and `loopUnrollFull`, and it failed at the first tile.

`tileUsingSCF` calls `getTiledImplementation` with the loop's **induction
variable** as the offset, because the tile has to be built once inside a body
that runs many times. **A convolution tile at a dynamic offset is not
representable in this dialect.** The first tile of an axis carries the leading
pad, the last carries the trailing one, the tiles between carry neither, and
`pads` is a static `DenseI64ArrayAttr`, so one loop body cannot stand for tiles
that disagree about it. The interface declines a non constant offset, which is
the refusal written at P13 with the halo arithmetic, and this is the other end of
it.

So the grid is walked at compile time and every tile is materialised at a
constant offset, which is what full unrolling produces anyway. **The observable
contract that description exists for is met exactly**: the tiles are fully
unrolled, no `scf` operation survives, the pass asserts that about its own
function, and the lit test asserts it from outside. What is not met is the
intermediate step, and it is not met because it cannot be.

That assertion earned its place immediately: the first failed run left an
`scf.for` behind, and the pass said so by name rather than letting the lowering
report something unrelated three passes later.

### Two bugs the tests caught, and the second is the interesting one

The pass printed a budget of 64 as `@`. Streaming a tablegen pass option into a
`Diagnostic` picks the character overload, and 64 is `@` in ASCII. It is D-0043's
shape in miniature, a value arriving through a channel that loses information,
and the diagnostic that told a reader the wrong number would have been the only
place anybody saw it. The fix is a plain `int64_t` copy with a comment saying
why it exists.

The second was in the search. The byte model computed a tile's input extent as
`(tile - 1) * stride + effectiveKernel` and did not clamp it to the input the
operation actually has. Under same padding a tile covering a whole axis was
therefore priced as reading **more rows than the axis contains**, because the
window runs off both edges and the padding is not fetched.

**That is not conservative in a safe direction.** It made the search believe
that not splitting an axis costs more memory than the whole operand does, which
pushed it to split axes it did not need to split and made `halo=cache` decline
budgets it could have met. With the clamp, the same convolution goes from eight
tiles at a makespan of 2988 cycles to four at 1704. The fix shares one helper
between the thing that decides whether a candidate fits and the thing that
decides whether it is good, so the two cannot disagree about what a tile is.

**It was found by a lit test asserting an exact choice rather than a property.**
A test that had checked only "it tiled" would have passed against both.

### The regret of the named baselines, which is now a number

On the eight channel convolution at a 2048 byte budget: exhaustive scores 1704,
`fixed` ties at 1704, `largest-fit` scores 1790. That is 5 percent, and it is
small on this shape rather than small in general; the point of keeping the two
baselines is that the exhaustive result has something to be compared against
rather than only asserted about.

### Section 13.3's third arm, as far as it goes honestly

The halo boolean is implemented and its limits are written into the pass
description rather than left to the report. `halo=recompute` allows the two
output spatial axes to be split, so adjacent tiles re-read the overlapping input
rows and the halo is paid for in DMA per tile. `halo=cache` refuses to split
them, so no halo is created and none is re-read.

**What `halo=cache` is not is a scratchpad resident halo carried from one tile
to the next.** That needs a transfer whose source is the previous tile's buffer
and a cost model term for a partially reused operand, and this machine has
neither. The boolean is the choice between **paying** the halo and **not
creating** one, which is the cheap version Section 13.3 asks for, and any
comparison reported from it has to say so.

The lit test makes the tradeoff a measurement: `halo=cache` tiles four ways at
2048 bytes and declines at 768, where `halo=recompute` splits the rows two at a
time and fits in 720. Not creating a halo costs the ability to shrink the input
at all, and the budget where that stops being affordable is a number.

### D-0049: the flake was a bound whose precondition is not checked

The single unexplained red from earlier in this phase was reproduced
deliberately: twenty four busy loops against twenty eight hardware threads, eight
runs, **one red**; three runs on the idle machine, none, on top of four clean
full suite runs earlier.

```
--mlir-timing reports Canonicalizer at 4.5000 ms and this project's
instrumentation at 0.4496 ms, a gap of 4.0504 ms against a bound of 2.3000 ms
```

**It is not the bound P12 asked P13 to watch**, and the distinction is the useful
part of the finding. `cross_check_against_mlir_timing` has two bounds pointing in
opposite directions. The **deficit** bound catches the instrumentation reading
above MLIR, is derived from the print quantum, and is D-0043's, whose margin
narrowed across 0.1577, 0.1177 and 0.1856 ms against 0.2000. This is the
**upper** bound, `half_ulp` plus fifty percent of MLIR's figure. Nothing here is
a fourth data point on the other one, and recording it as one would have put a
wrong number into the only place that number is tracked.

**The code already contains the argument for what is wrong.** The upper bound's
premise is that the gap *is* the instrumentation's own operation walk, and
`pass_stats.py` already refuses to check it under a traced interpreter, on the
stated grounds that a tracer stretches everything else inside MLIR's window so
the gap stops being the walk. A busy machine does the same thing for the same
reason: MLIR's timer is wall clock and brackets the whole pass, so time the
process spends descheduled lands inside MLIR's window and not inside this
project's. 4.5000 ms for a canonicalization this project measured at 0.4496 is
not a canonicalization that took four milliseconds.

**It is left open deliberately.** The fix is a precondition, not a wider bound:
measure the process's own CPU time against the wall clock and skip the upper
bound when the process did not have the processor, exactly as it is already
skipped under a tracer, with the deficit bound left active because its premise
survives either way. `TIMING_GAP_FRACTION` must not move to a number chosen to
make a run green, which is the rule this project already applies to D-0043's
bound and which applies here for the same reason.

**Where it matters is CI rather than here.** The test carries `slow`, CI's
`pytest slow cells` step runs slow tests, and the runners are shared four vCPU
machines. A developer machine with nothing on it is the least likely place for
this to fire, which is D-0037 and D-0040's shape a third time: a check whose
behaviour depends on the machine, validated on the machine where it behaves.

**The cost of nearly losing it is worth recording.** The first observation went
into a battery script that tailed three lines of output, so the message was gone
before anybody read it and the session recorded the flake as unexplained. What
turned it into an entry was capturing the whole failure, and the price of not
doing that the first time was running everything again.

### What tiling is waiting on, stated so it is not rediscovered

`-npu-lower-to-npuisa` has no pattern for `tensor.extract_slice` or
`tensor.insert_slice`, so a tiled function does not lower. That is the smaller of
the two changes.

The larger one decides whether tiling is worth anything. The conversion loads
each DRAM function argument into the scratchpad **once, whole**, and records it
so every consumer reads the same resident buffer. Under that arrangement a tiled
program's slices are views of buffers that are already resident: tiling would
split the compute instruction and leave the scratchpad footprint exactly where it
was, and the ablation row would show instructions moving and cycles moving and
pressure not moving at all. For tiling to relieve pressure the **slice** has to
be what enters the scratchpad, which is consistent with Section 8's count of one
load per DRAM value entering it, since under tiling the values are the slices.
It is a real change to the conversion and to `dma-boundaries.mlir`, and it is
where P13 continues.

## 2026-09-05 Phase P13: layout assignment answers NCHW, and the answer is a ratio between two constants

### What the pass is

`-npu-assign-layout` is implemented and is in **no `-O` level**, alongside
tiling and double buffering, for the reason the pass list gives: putting the
three in is one commit, because that is where the cell count moves and Section
2's arithmetic is re-derived, and doing that three times would leave two
intermediate states nothing will ever run again.

It has three parts. It **scores** the layout question for every rank 4
operation and counts the answer in `kept-nchw`. It **folds** an inverse
transpose pair into the value it permuted, which is the half Section 12 names
as the thing without which the pass only ever adds instructions. And it
**sinks** a transpose past a relu, which is what lets the fold reach a pair the
graph did not write adjacent.

### The answer is NCHW everywhere, and it is not close

Section 5.5 charges layout in exactly one place, the non unit innermost stride
penalty on a transfer, at `kDmaStridedElementCycles = 0.5` cycles per element.
Section 5.5 also fixes how a layout reaches that term: an NHWC tensor is
materialised below the tensor level as a buffer at **NCHW extents carrying
permuted strides**, so its innermost stride is the channel count and every
transfer of it is charged the penalty, while an NCHW tensor is contiguous and is
charged nothing.

The only alternative to paying that penalty is to perform the permutation, and a
permutation is one elementwise pass at `1 / kElementwiseLaneWidth = 0.0625`
cycles per element. So **performing the permutation is eight times cheaper than
moving the same data strided**, and it is a ratio rather than a threshold: a
transfer's other two terms, the bytes and the fixed descriptor cost, are charged
whichever layout the buffer is in and cancel out of the comparison instead of
tipping it at some size.

The whole decision is therefore a relation between two constants, and it is
asserted as one, in `CostModelTest.cpp`, in the file that owns both. That is
deliberate: the conclusion the report will publish is about the machine and not
about the pass, so recalibrating either constant has to fail a test rather than
quietly reverse a published answer.

### The observation that decided the pass's shape

Section 5.5 motivates the stride term with the sentence that without it "a
strided NCHW gather and a contiguous NHWC burst cost exactly the same". Read on
its own that sentence describes a machine whose canonical buffer order is NHWC,
where NCHW is the layout that gathers. This machine's canonical buffer order is
NCHW: every `npuisa` verifier reads NCHW extents and every kernel indexes them,
which is the paragraph immediately after, the one that says how the layout
reaches the term. **The term therefore charges the opposite layout from the one
the motivating sentence's example names.**

The mechanism paragraph is the normative one here, because it is the one that
says what the compiler does, and the code follows it. The two sentences are
worth reconciling in the specification at some point and that is an owner edit
rather than a code change; it is recorded here so the next reader who notices
the tension finds it already noticed rather than rediscovering it as a bug.

### What the pass therefore does not contain, which was the design decision

There is no code that rewrites an operation into NHWC. The scoring refuses that
trade at every shape this machine can hold, so a materialisation path would be a
branch no input could reach and no test could exercise, and untestable code is
worse than absent code.

The tempting alternative was to absorb a layout changing transpose into the
transfer that was going to move the bytes anyway. It is exactly the right
rewrite and it is exactly `relayout-and-move`, which Section 12 names as a
future extension, states is not implemented, says needs a descriptor form the
binary format does not have, and forbids any gate or report claim from depending
on. Recognising the tempting rewrite as the thing already scoped out is the
whole of that decision, and the alternative would have been a fourth DMA
producer against Section 8's invariant.

There is a second thing that would have been needed and it is worth writing down
because it is not obvious from the dialect: a layout changing transpose does not
lower. Its two sides become memrefs of the **same** extents differing only in
their strides, and the `npuisa` transpose verifier asks that result extent `i`
be input extent `permutation[i]`, which such a pair does not satisfy. So an NHWC
pipeline is a lowering change as well as a pass, on top of a cost model that
says it would never be used.

### What it does on the suite

Nothing, on all fourteen model configurations, at both `-O0` and `-O2`: zero
folds, zero sinks, and the printed IR is identical to the input. The layout
question is answered four to six times per model and answered NCHW every time.
`dilated_stack` is the only model that contains a transpose at all, the closing
NCHW to NHWC permutation Section 15 gives it, and it is left alone: it is the
last operation before the return, so there is no inverse below it to cancel
against and no consumer to sink anything through.

So the P13 gate's layout delta is **zero, and the DMA stride term is what makes
it zero**. That is the gate's "reported whichever way it went" answered with the
direction it went, and it is the second negative result of this phase after
double buffering, which is a fact about the machine rather than about either
pass: this cost model is a two port dataflow schedule with a large stride
penalty, and both of those properties remove a classic optimization's payoff.

### The negatives are where the tests are

Three of the five lit cases exist to hold a line rather than to show the pass
working. The load bearing one is a pair of permutations that compose to the
identity, returning the extents to exactly where they started, whose outer
result carries `#npu.layout<nhwc>` and whose inner input does not. A fold that
looked only at the permutations would delete it. It is not a round trip, it is a
relayout, and deleting it would change what the bytes mean while leaving every
type in the function looking right. The guard is that the surviving value's type
must equal the replaced result's exactly, encoding included, which is why the
fold compares types and not shapes.

The fold also hands the orphaned inner transpose back to its caller rather than
erasing it where it stands, because the caller is iterating a worklist that may
still hold it. A fold that left a dead full pass over the data behind would be
reporting a saving it had not made.
## 2026-09-05 Phase P13: the wiring commit, and the tiled program the encoder refuses

**The three passes went into `-O2` together and the suite was re-recorded once at
that tree.** The ablatable set is eleven, which is Section 12's own number with
nothing subtracted from it, the suite is 217 cells, and the ablation half of
Section 2's arithmetic agrees exactly at 154. **Not one counted field of the 175
pre-existing cells moved**, over instructions, cycles, compute and DMA cycles,
scratchpad peak and bytes, spill count, spill DMA count, DRAM bytes, the oracle
distance, the overlap fraction and the fragmentation ratio. All 21 golden tensors
are byte identical. There is nothing to declare in `docs/BREAKING_CHANGES.md` and
writing an entry anyway would be a false declaration.

**The wiring found five defects, which is the argument for wiring a pass into a
level rather than testing it beside one.** Each of the three passes had lit
tests, statistics and a measured delta before this session, and each was correct
about the program it was handed. What none of them could see is the pipeline:
what `-cse` does to the values the tiling pass reads, what the conversion driver
folds before a pattern sees it, what the encoder does with a buffer written in
pieces, and what the lowering's own transfer layout leaves for double buffering
to work on.

### D-0052, which changes the phase rather than the commit

**A tiled result assembled in DRAM cannot be read back.** The first tiled program
this suite produced, `resnet_block` at its tight budget, did not encode:

```
operand-extent: operand 0 reads 2048 bytes from 10944 and the buffer written
there ends at 11968 (instruction 14)
```

Instruction 14 is the load that brings the assembled convolution back on chip for
the multiply that reads it. The assembly is 2048 bytes and was written by two
stores of 1024, and `WrittenSpans` deliberately does not merge adjacent spans,
because merging them is what would let an over read off the end of one buffer and
into the next pass validation.

**The decision this contradicts is one of P13's own, and the contradiction is one
sentence deep.** `docs/PHASE_STATE.md` chose DRAM assembly over weakening checks
8 and 9, and gave as a reason that "nothing is ever written in pieces and read
whole". That is true of the tiles and false of the consumer. The decision assumed
the next layer loads the slices it needs; the next layer is not tiled, so it loads
the whole value.

**Three measurements settle the scope**, and they are the reason this is D-0052
rather than a patch. A **matching** tiled consumer does not help, because the
write side records `resultByteSize`, the element count, and the read side computes
`addressedByteSpan`, the stride reach, and for a strided tile those differ, 1024
against 1920 on the case above. A **scratchpad** assembly is refused by the same
rule, which `test/Encoding/tiled-assembly-in-scratchpad.mlir` already records. And
the **one** shape that does encode is an assembly nothing reads: a tiled operation
whose result is the function's own, whose tiles are stored straight into the out
parameter, which never gets read back.

**So the pass declines**, which is Section 13.2's own answer to a tile that is not
expressible, and the allocator's spilling is the fallback. The alternative was a
compiler that emits programs its own encoder refuses, reported three tools away
from the pass that caused it.

**What it costs is measured rather than estimated.** Before the rule,
`resnet_block` tiled one convolution at its tight budget and `inception_block`
tiled two, and every one of those three programs was refused. After it, nothing
in the suite tiles at `-O2` at either budget, which is what the committed
prediction said, for a reason the prediction did not give. **Section 13.3's
tiling arm has no subject inside the suite until the ISA question is decided, and
that is not a phase's decision.**

### The wired tree changes the tiles nothing measurement's premise, and the table had to be taken again

The pass alone at its own defaults is not what the suite compiles. The pipeline
hands the tiling search the allocator's budget and tells it that
`-npu-double-buffer` is in the pipeline, which doubles the prefetched operand's
contribution per Section 13.2, so an operation that fits without the prefetch is
over budget with it.

**The default budget column did not move and that is the check that mattered.** A
default budget cell that moved would have been a wiring defect rather than a
declaration to write. The tight budget column moved in its `declined` count on
every model, and ablating `-npu-double-buffer` moves it back, which is the
coupling made visible: **ablating double buffering also relaxes the tiling
search, so its ablation row measures the pass together with the sizing it
forces.** That is Section 13.2's coupling rather than this wiring's, and
`docs/PASSES.md` says so beside the row.

### Two zeros that look identical and are not, again

**`-npu-double-buffer` fires on nothing this compiler emits**, which is D-0054.
`docs/PASSES.md` already carried a measured reason for a zero from this pass: on a
hand written tiled convolution it fires, the instruction stream genuinely changes,
and no cycle moves, because tiling makes a program DMA bound. **That reason is
true and is not the suite's reason.** On the suite `prefetched` is 0 and
`not-hoisted` is every transfer, because every argument load sits in the entry
block beside the other argument loads, where the walk correctly stops at another
transfer, and a constant's load is the one transfer with a computation before it
and the one whose `npuisa.const` the prologue cannot carry.

**This is P10's `-canonicalize` finding for the third time**, and the general
statement is now worth making: in this project a zero ablation row has meant a
pass with nothing to do, a pass whose work another pass would do, a pass whose
answer is the input it was given, a pass whose subject the binary format cannot
express, and a pass that fires on nothing. Five mechanisms, one printed value.

### A bound that was quoted three times and is not in the code

**D-0055.** `cross_check_against_mlir_timing` has two bounds pointing in opposite
directions. The deficit bound is `half_ulp_ms`, which is 0.0500 ms at the four
decimals of seconds MLIR prints, and it is D-0043's. The upper bound is that plus
half of MLIR's own figure, so it is a different number for every pass. **There is
no 0.2000 anywhere.** Three handoffs recorded 0.1577, 0.1177 and 0.1856 as a
narrowing margin against "D-0043's 0.2000 bound"; all three are readings of the
**upper** gap, which is what `run_benchmarks.py` prints, and the bound they were
compared against does not exist.

P13's quiet run measures a worst upper gap of 0.2430 ms, green against that
pass's own allowance, and **no cell of the 217 came within a red of the deficit
bound**, which is the statement D-0043 wanted and that nobody was making. Nothing
was widened; a misread bound is corrected by reading it correctly.

### The first attempt at the re-record went red, and it is D-0049's fifth point

It died on the upper bound with a gap of 0.6897 ms against 0.5000, on a machine
with nothing running but a one minute load average still around 3 from the builds
seconds before. Ninety seconds of settling brought it to 0.74 and the run
completed clean. **The bound was not touched.** D-0049's own claim is that the
upper bound's premise, that the gap is this instrumentation's operation walk, does
not survive a machine that is not idle, and a machine still draining a build is
not idle however empty the process table looks.
## 2026-09-05 Phase P13: the format learns to read a buffer written in pieces, and the compiler does not use it yet

**The owner decided D-0052 and the decision is region scoped coverage on the
DRAM side of checks 8 and 9, with no version bump.** Inside a declared spill
slot the validator now tracks exact byte coverage, strided writes run by run,
and accepts a read when every byte it addresses lies inside that one slot and
has been written. The scratchpad keeps the no merge rule, because a buffer there
has no declared extent and a merged range would accept an over read into the
buffer next door. That asymmetry is the whole decision: **a spill slot has an
identity and the scratchpad arena does not**, and the relaxation goes exactly as
far as the identity does.

**No encoded byte moves, which is why no version moves.** Version 2 was declared
so that a buffer could be written in pieces, and that declaration's own table
promised checks 8 and 9 would go from "one written count per address" to
"written ranges per buffer". **Only the format half landed.** This is the other
half of a decision already taken rather than a new one, and saying so is what
makes the absence of a bump defensible rather than convenient.

**It also closes the count against reach asymmetry**, on the DRAM spill slot
side only. A strided store recorded its element count as one contiguous run
while the matching read computed its stride reach: 1024 against 1920 on the case
D-0052 measured. Both sides now use the bytes actually touched, and the
scratchpad side is untouched.

### What the corpus said, which is nothing, and why that is worth printing

All 778 seeds were run through the validator at the parent and at the fix.
**Zero verdict flips.** Not one seed exercises a spill slot read at all, so the
corpus could neither confirm nor contradict the change. That is a fact about the
corpus rather than about the change, and it is the reason the five unit tests in
`SpillSlotCoverageTest.cpp` exist and are written as four verdicts rather than as
four refusals: a corpus that cannot reach a rule cannot be the evidence for it.

### The compiler half is measured and held, and D-0056 is the reason

With the validator fixed, `-npu-tile-to-scratchpad` can tile an operation whose
result another operation reads. Compiling all 168 cells at that tree says what
one cell could not.

**It improves four of them**, and these are the first numbers in this project
that say tiling buys anything on a real model:

| Cell, at its tight budget | with tiling | tiling ablated |
|---|---|---|
| `conv_bn_relu_stack` ablate `npu-fuse-ops` | peak **4640** | peak 6432 |
| `inception_block` | **0 spills** | 3 spills |
| `lenet` ablate `npu-fuse-ops` | peak 194200 | peak 194560 |
| `lenet_batched` ablate `npu-fuse-ops` | peak 199840 | peak 200800 |

**And it takes one away.** `resnet_block` at its tight budget with fusion
ablated stops allocating: the residual keeps the block's input resident across
both convolutions, so tiling the second adds the tile buffers on top of an
activation it does not remove, and the assembly comes back on chip whole above
that. Sweep line peak 7456 against a budget of 6464.

**Two rules were written against that and both are recorded because both
failed.** Charging the assembly's re-entry to the search's budget made the tiles
smaller and the cell failed again with a 256 byte buffer unplaced instead of a
1024 byte one. Requiring the tiling to be an improvement, tile peak plus
re-entry against the untiled working set, **declined `inception_block`**, which
was one of the wins, and **still admitted `resnet_block`**, which was the loss.

**A rule that is both too strict and too weak is a wrong discriminator**, and
the reason it is wrong is worth more than the two attempts: the quantity that
decides is the **program's** sweep line peak, and the tiling pass sees one
operation. On `resnet_block` the deciding byte belongs to a different operation
and is live for a reason, the residual, that the tiled operation has no view of.
Section 13.2 sizes a tile's working set against the budget, which is the right
unit for choosing **between** tilings and the wrong one for choosing **whether**
to tile.

So the compiler half is not committed. The tree that ships is the one where all
168 cells compile, the four wins are left on the table and said so, and D-0056
carries the three ways forward: the pass consults the allocator, the consumer
chain tiles with its producer, or it stays as it is.

### D-0054's one line fix is not one line, and the verifier said so

Admitting `npuisa.const` to the double buffering prologue is the obvious change
and it was written, built and run. It produces programs this project's own
verifier rejects: a `dma_store` lands between the asynchronous load and its
await and overlaps the destination, which is the race the token exists to
prevent.

**So the pass's hoist safety walk is weaker than the verifier that checks its
output**, and that gap is the defect rather than the missing entry in a set.
Both ask `npuisa::overlaps`; the walk asks it of the operations it steps over
and the verifier asks it of the whole window. Where the two diverge is the thing
to measure, and it is larger than a change to an `isa<>` list. Not committed,
because a pass that fires and emits programs the verifier refuses is worse than
a pass that fires on nothing.

### What this run did not spend

**The quiet machine.** Nothing moved a measured quantity: no cell field, no
cycle, no DRAM byte, no golden tensor byte. The suite is the same 217 cells at
the same numbers, `regression-baseline --check` reports no drift in both shapes,
and the only baseline movement is five test names and five counts, which is the
composition change the declaration's own list of what does not move predicted.
