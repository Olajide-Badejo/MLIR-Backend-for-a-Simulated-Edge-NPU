<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Passes

*Diataxis type: reference.*

Ground rule 12: a pass that changes behaviour updates this file in the same
commit as the code, not later. Every entry carries what the pass does, before
and after IR, what it refuses, whether it is ablatable, and its measured
ablation delta.

**On the ablation deltas.** Section 12 requires every entry here to cite a
**measured** delta rather than a qualitative claim. **Since P7 there is a
simulator to measure against**, and `Stats::cycles` is the number a delta would
be taken from; the harness that runs a pipeline twice and subtracts is still
Phase P10's, so no entry below quotes one yet. Each says so in its own row rather
than leaving the field blank, because a blank field reads as "no effect" and an
absent measurement is not a measurement of zero.

**What P9 could measure without that harness, it did**, and the numbers are in
the entries: which passes fire on which models, whether forming a region changes
the instruction count, and how far each pass moves the answer. Those are
measurements of the same programs the ablation will run, taken one pass at a
time in `test/Python/test_transform_passes.py`.

The full pass table, including the three ablatable passes that arrive at P13 and
the calibration pass that arrives at P14, is Section 12 of the build
specification. This file describes the passes that **exist**, and it grows as
they land.

## The pass list, as it stands today

| Pass | Level | Ablatable | Phase | Status |
|---|---|---|---|---|
| `-npu-constant-fold` | O1, O2 | yes | P9 | implemented |
| `-canonicalize` | O1, O2 | yes | P9 | upstream, wired in |
| `-npu-fuse-bias` | O2 | yes | P9 | implemented |
| `-npu-fold-batchnorm` | O2 | yes | P9 | implemented |
| `-npu-fuse-ops` | O2 | yes | P9 | implemented |
| `-cse` | O2 | yes | P9 | upstream, wired in |
| `-sccp` | O2 | yes | P9 | upstream, wired in |
| `-symbol-dce` | O2 | yes | P9 | upstream, wired in |
| `-npu-lower-to-npuisa` | all | no | P4 | implemented |
| `-npu-allocate-scratchpad` | all | no | P5 | implemented |

**Eight ablatable, and Section 12's eleven is that plus three.**
`-npu-assign-layout`, `-npu-tile-to-scratchpad` and `-npu-double-buffer` are
excluded from P9 by name in the roadmap and arrive at P13; `-npu-calibrate` is
never in a default `-O` level. Nothing below claims a row for any of them,
because a level that named a pass nothing implements would give Section 16.2's
ablation table a row it could not fill.

---

## The optimization levels

*Added at P8, in `lib/Pipeline/Pipeline.cpp`.*

A level is a named list of passes, and the list lives in C++ rather than in the
Python driver. Section 6 settles that and the reason survives the summary: the
`PassInstrumentation` of Section 16.2 has to sit on the `PassManager` that
actually runs the passes, so a pipeline assembled in Python out of one `npu-opt`
invocation per pass would be a different pipeline from the one under test.
`npu-compile` names a level, `lib/Pipeline` builds it.

| Level | Registered as | Passes, in order |
|---|---|---|
| `-O0` | `npu-O0` | `-npu-lower-to-npuisa`, `-npu-allocate-scratchpad` |
| `-O1` | `npu-O1` | `-npu-constant-fold`, `-canonicalize`, then `-O0`'s two |
| `-O2` | `npu-O2` | `-npu-constant-fold`, `-canonicalize`, `-npu-fuse-bias`, `-npu-fold-batchnorm`, `-npu-fuse-ops`, `-canonicalize`, `-cse`, `-sccp`, `-symbol-dce`, then `-O0`'s two |

**All three are implemented since P9.** `-O1` and `-O2` were named and
unregistered from P8 until then, deliberately: a level registered with an empty
pipeline would have run, produced `-O0`'s answer, and made every ablation cell
measured against it measure nothing.

**`-O0` is "import and verify" and verification is not a row in the table.**
MLIR verifies every operation when it parses it and again after every pass, so
the level gets its verification from the pass manager. A row that ran a verifier
would be a second and weaker one beside the one that already runs. Since P9 the
driver's `npu` stage runs the level's tensor level half through `npu-opt` at
every level including `-O0`, so the verification is one somebody performs rather
than one the documentation claimed.

### Why the order is what it is, and where it departs from Section 5.1

Two departures, both measured rather than chosen.

**`-npu-constant-fold` runs before the first `-canonicalize`**, where Section
5.1 lists canonicalization first. The folder is what *creates* the dead operand
constants, so a canonicalization ahead of it has nothing to clean up. At `-O2`
there are two canonicalizations and the question would be academic; at `-O1`
there is one, and if it ran first every folded operand would survive as an
`npuisa.const` and a `dma_load` in the instruction stream.

**`-npu-fuse-bias` runs before `-npu-fold-batchnorm`**, where Section 5.1 lists
the batch norm fold first. `-npu-fold-batchnorm` matches on a convolution as the
batch norm's producer, and in `conv -> add(bias) -> batch_norm` the producer is
the add until the bias fusion has moved it. Folding first leaves **both** passes
declining, and then `-npu-fuse-ops` declines too because the activation's
producer is a batch norm rather than a convolution: one ordering choice turning
three passes off. `test/Pipeline/opt-levels.mlir` carries the function that shows
it, and that test went red the first time this table was written in the listing
order.

Section 12's own note on `-npu-fold-batchnorm` asks for it "before fusion, so
the convolution still has no fused **activation**", and the activation fusion is
`-npu-fuse-ops`, which still runs after it. The duplicate `-canonicalize` around
the fusion group is Section 12's, marked there as deliberate: the first removes
what the folder left dead, the second removes the four batch norm parameters the
fold consumed.

### Where each pass runs in the pass manager

Per pass rather than per level, and it is not cosmetic. `-sccp` reasons about
the call graph and `-symbol-dce` about symbols, so both are added at the module
level; everything else is nested under `func.func`. Nesting either of those
inside a function would give it a view in which every question it asks has the
wrong answer.

### `--emit npu` runs the tensor level half through the same pipeline

`npu-compile --emit npu` asks for `--npu-O2=stop-after=npu`, which builds the
level and stops before `-npu-lower-to-npuisa`. Which passes those are is
`isTensorLevel`, a switch over `PassKind` with no `default`, so a pass added to
the enumerator and not classified is a build error rather than a pass that
silently lands on the wrong side of the stage boundary. The description prints
the answer as each entry's `stage`.

The alternative was for the driver to name the tensor level passes itself, and
Section 17.4 rules it out from the other direction: a test that runs a hardcoded
pass list matching no optimization level enforces nothing, and a driver that
assembled one would make every such test vacuous at once.

**Every entry carries an `ablatable` property and a missing one does not
compile.** Section 12 asks for that in those words. `PassEntry` has a single
constructor taking every field with no defaults, so a row written as
`{"npu-fuse-ops"}` is a build error. A default of `false` would quietly shrink
Section 16.2's ablation table by exactly the passes nobody thought about.

**Each entry also declares whether it eliminates dead code**, under the same
rule and for a second consumer. Section 17.3a's dead subgraph injection asserts
that a subgraph feeding nothing leaves the instruction count unchanged, which is
only true at a level whose pipeline holds a pass that removes it. The check
reads `eliminates_dead_code` out of the description rather than carrying a list
of pass names that would go stale the first time one was added. Both of `-O0`'s
passes declare `false`, `-canonicalize` and `-symbol-dce` declare `true`, so the
set of levels that eliminate dead code was empty at P8 and became `-O1` and
`-O2` at P9 with no edit to the test that reads it. `-O0`'s form of the check is
the opposite claim and is kept as the control: the count grows by exactly the
instructions the injection brought.

**And each entry carries the `PassKind` the builder switches on.** *Added at
P9.* The switch has no `default`, so a pass added to a level's table and not to
the builder is a build error rather than a pipeline that silently skips it. The
argument string sits beside the kind for the description and the diagnostics,
and `test/Pipeline/opt-levels.mlir` runs each level against the explicit list of
those strings and diffs the two outputs, which is what catches a kind and a name
that disagree.

**The description is readable at run time**, which is what Section 16.2 requires
of the ablatable set:

```
npu-opt --npu-describe-pipeline
```

prints every level as JSON with its passes, their `ablatable` flags, whether the
level is implemented, and the phase an unimplemented one arrives at. The flag is
handled before `MlirOptMain` because it asks a question about the compiler
rather than about a file and therefore takes no input.

**The level and its pass list are asserted to agree, at all three levels.**
`test/Pipeline/opt-levels.mlir` runs each `--npu-O` pipeline and the explicit
list of its pass arguments over the same input and diffs the two outputs.
Section 17.4 says a test that runs a hardcoded pass list matching no
optimization level enforces nothing; the converse obligation is this one, that a
level nobody compares against anything can drift from the passes it claims to
run.

The pipeline forwards the allocator's four options, so `npu-compile --budget`
reaches the allocator without the driver knowing which pass consumes it, and
`stop-after` beside them:

```
npu-opt model.mlir --npu-O0=budget=8192
npu-opt model.mlir '--npu-O0=budget=8192 strategy=interval'
npu-opt model.mlir --npu-O2=stop-after=npu
```

---

## `-npu-constant-fold`

Evaluates `npu.add`, `npu.mul`, `npu.relu` and `npu.reshape` whose reads are all
`npu.constant` of the result's own shape. Implemented in
`lib/Dialect/NPU/Transforms/ConstantFold.cpp`, registered by
`mlir::npu::registerNPUPasses()`.

**Ablatable: yes.** **Ablation delta: not measured**, the harness lands at P10.
**Measured at P9:** it fires on no model in Section 15's suite, because an
exported graph's constants feed convolutions rather than each other. Its value
is to a graph a transform has already partly folded, and to the `-O1` and `-O2`
pipelines as the thing that makes the canonicalization after it have work to do.

**Why a pass and not four `fold` methods.** MLIR's folder runs inside
`-canonicalize` and would do the same arithmetic, but Section 12 asks for
`-npu-constant-fold` as a **named entry in the pipeline description** so that
Section 16.2's leave one out ablation has a row for it. A fold hook is not
something an ablation can remove; a pass is.

**It moves no bit.** Every operation it evaluates is elementwise with no
reduction in it, so `a + b` computed at compile time in `f32` and `a + b`
computed by the kernel in `f32` are the same value. That is half of why this
phase's numerics movement is attributable to `-npu-fold-batchnorm` alone.

### Before and after

```mlir
%a = npu.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
%b = npu.constant dense<[[10.0, 20.0], [30.0, 40.0]]> : tensor<2x2xf32>
%d = tensor.empty() : tensor<2x2xf32>
%r = npu.add ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>)
             outs(%d : tensor<2x2xf32>) -> tensor<2x2xf32>
```

becomes, after this pass and the canonicalization Section 12 puts beside it:

```mlir
%cst = npu.constant dense<[[11.0, 22.0], [33.0, 44.0]]> : tensor<2x2xf32>
```

The operand constants and the `tensor.empty` are left with no users and
`-canonicalize` removes them. This pass does not remove them itself, because a
constant with another reader is not dead and deciding that is the
canonicalizer's job.

### Where it does not fire

Section 12's negative test rule, in `test/Transforms/constant-fold.mlir`.

- **A rank 1 channel broadcast addend is not folded**, and this is the load
  bearing one. Folding `add(%x, %c)` where `%c` is the rank 1 constant of
  Section 11's carve out would mean materialising the `N x C x H x W` expansion
  the importer refuses to perform: it would inflate every DRAM byte count by the
  expansion factor and leave `-npu-fuse-bias` nothing to match.
- **An operand that is not a constant at all**, which is the case a pass firing
  unconditionally would get wrong.
- **A convolution over constant operands.** Foldable in principle and not folded
  here: a reduction evaluated at compile time would sum in a different order
  than the kernel does, which is a movement this pass does not make and
  `docs/BREAKING_CHANGES.md` does not declare.

---

## `-npu-fuse-bias`

Folds `add(conv2d(x, w), b)` into the convolution's bias operand. Implemented in
`lib/Dialect/NPU/Transforms/FuseBias.cpp`.

**Ablatable: yes.** **Ablation delta: not measured**, the harness lands at P10.
**Measured at P9:** it removes one instruction and one scratchpad buffer per
fused add, and it fires on **no model in Section 15's suite**, because every
convolution there carries its bias inline as a third `Conv` input. It fires on
an imported model built for it in `test/Python/test_transform_passes.py`, which
is the gate's clause, and the suite gap is recorded there as a finding.

**This is the pass Section 11's broadcast carve out exists for.** A rank 1
initializer of length C broadcasting against a rank 4 activation over the channel
axis is left unexpanded by the importer *so that this guard can match*. An
importer that expanded it would make this pass structurally unfireable, its
ablation row a row of zeros, and the phase look done while doing nothing.

**It is exact, and that is measured.** The simulator's convolution kernel
accumulates into an `f32` and adds the bias to that same `f32` before it writes,
which is the value the unfused program would have stored and then added to. So a
fused and an unfused answer agree bit for bit.

### Before and after

```mlir
%c = npu.conv2d ins(%x, %w : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>)
                outs(%d0 : tensor<1x2x4x4xf32>) {...} -> tensor<1x2x4x4xf32>
%b = npu.constant dense<[1.0, 2.0]> : tensor<2xf32>
%r = npu.add ins(%c, %b : tensor<1x2x4x4xf32>, tensor<2xf32>)
             outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
```

becomes

```mlir
%b = npu.constant dense<[1.0, 2.0]> : tensor<2xf32>
%r = npu.conv2d ins(%x, %w, %b : tensor<1x2x4x4xf32>, tensor<2x2x1x1xf32>,
                                 tensor<2xf32>)
                outs(%d0 : tensor<1x2x4x4xf32>) {...} -> tensor<1x2x4x4xf32>
```

The constant is moved above the convolution, because the importer emits it above
its own first reader and that reader has just moved earlier. `npu.constant` has
no operands, so moving one earlier can never break its own dominance.

### Where it does not fire

The three cases Section 12 names by name, in `test/Transforms/fuse-bias.mlir`.

| Refused | Because |
|---|---|
| the producer already carries a bias | there is nothing to move into, and a second bias is an operand the dialect does not have |
| the producer's result has another reader | that reader reads the value **without** the bias, and moving it would change what it sees |
| the addend has the result's own shape | that is a residual add, a different operation with the same spelling |
| the addend is not an `npu.constant` | a bias operand is data the encoder writes into the binary |
| the producer is not a convolution | there is no bias operand anywhere for the addend to become |

**The commuted form is explicitly not matched**, which is the second of the two
options Section 12 offers. `npu.add` refuses a rank 1 left hand operand in its
verifier, so `add(b, conv(x, w))` is not representable in this dialect; the
importer commutes at import and there is one spelling below it. A pattern for a
form the verifier rejects would be code no test could reach.

**A convolution and nothing else, for the same reason.** `npu.matmul` also
carries an optional bias, so a matmul case looks obvious. It is not
representable: `npu.add` admits a rank 1 right hand operand only when the result
is rank 4, and a matmul's result is rank 2. The one shape that does reach a
matmul is a same shaped addend, and that is a residual rather than a bias.

---

## `-npu-fold-batchnorm`

Folds an inference batch norm into the convolution that produced its input.
Implemented in `lib/Dialect/NPU/Transforms/FoldBatchNorm.cpp`.

**Ablatable: yes.** **Ablation delta: not measured**, the harness lands at P10.
**Measured at P9:** on `conv_bn_relu_stack`, the one model in the suite built to
carry unfolded batch norms, it removes both of them, which is eight instructions
and 212 simulated cycles at the default budget, and it moves the answer by
4.47e-08.

**This is the only pass at P9 that moves a number**, and
`docs/BREAKING_CHANGES.md` declared that before the commit that turned it on.

### The identity, and the order it is evaluated in

```
invStd = 1 / sqrt(variance + epsilon)
scale  = gamma * invStd
shift  = beta - mean * scale
w'[f]  = w[f] * scale[f]
b'[f]  = b[f] * scale[f] + shift[f]
```

The order is written out because it is observable: floating point multiplication
is not associative and these constants are computed in `f32`, so a reader
comparing against onnxruntime needs to know which of several algebraically equal
forms produced the number. It is deliberately the same order
`-npu-lower-to-npuisa` uses for the decomposition of a batch norm this pass did
not fold, so the two spellings of one identity agree with each other rather than
by luck.

**Why it moves bits.** Before the fold the machine convolves and then scales the
result; after it the machine convolves with pre scaled weights, so every product
in the reduction is scaled instead of the sum being scaled once at the end.
Equal in exact arithmetic, different in the last bits of `f32`.

**What it rewrites, and what it does not create.** The convolution is mutated in
place and the batch norm is replaced by the convolution's own result, rather than
a second convolution being built beside the first. The batch norm's destination
and its four parameter constants are left with no users, and `-canonicalize`
removes them: that is the canonicalization Section 12's table asks for after this
pass, doing the work it is there for.

### Where it does not fire

Five guards, each with a case in `test/Transforms/fold-batchnorm.mlir`. Every one
is a **non match and never a diagnostic**: Section 5.2 makes an unfolded batch
norm legal, and the lowering decomposes it into a multiply and an add.

| Refused | Because |
|---|---|
| the producer is not an `npu.conv2d` | there are no weights to scale |
| the convolution has more than one use | rewriting it would change what the other reader sees, and cloning it would double the weights in DRAM to save one scaling pass |
| a parameter is not an `npu.constant` | the multiplier is computed at rewrite time |
| the filter, or the existing bias, is not an `npu.constant` | the same |
| `variance + epsilon` is not strictly positive | the fold takes a reciprocal square root of it. `-npu-lower-to-npuisa` is the layer that refuses this with the numbers in the message, and this pass declines rather than competing with it |

---

## `-npu-fuse-ops`

Moves a `npu.relu` and the `npu.conv2d` or `npu.matmul` that produced its input
into one `npu.fused_op` region. Implemented in
`lib/Dialect/NPU/Transforms/FuseOps.cpp`.

**Ablatable: yes.** **Ablation delta: not measured**, the harness lands at P10.
**Measured at P9:** it forms regions on five of the seven models and changes the
instruction count on none of them, and it moves no bit.

**This is the pass that gave `npu.fused_op` and `npu.yield` a producer**, which
is what closed the two dated exemptions `docs/EXEMPTIONS.md` carried from P8.
Neither operation could appear in a model's IR until something created one.

**Why a region.** Section 7.2 settles it: separate fused operation names produce
a combinatorial explosion as the fusible set grows, and an enum cannot express
what fusion does, which is to keep an intermediate value in the scratchpad
instead of writing it to DRAM. A region expresses exactly that and grows without
a migration.

**It is numerically and structurally inert, and that is a property of the memory
model rather than a weakness.** `-npu-lower-to-npuisa` flattens the region into
its parent, so the instruction stream is the one the unfused chain produced, and
an unfused chain already keeps its intermediate in the scratchpad because the
only DMA producers are the boundary, the spiller and the double buffering pass.
What the region adds is that the fusion is **stated** in the IR, which is what
P13's tiling and double buffering read. Measuring that rather than asserting it
is what keeps the P10 ablation row honest.

### Before and after

```mlir
%d0 = tensor.empty() : tensor<1x2x4x4xf32>
%d1 = tensor.empty() : tensor<1x2x4x4xf32>
%r = npu.fused_op ins(%x, %w, %d0, %d1 : tensor<1x2x4x4xf32>,
                      tensor<2x2x1x1xf32>, tensor<1x2x4x4xf32>,
                      tensor<1x2x4x4xf32>) {
^bb0(%a: tensor<1x2x4x4xf32>, %f: tensor<2x2x1x1xf32>,
     %e0: tensor<1x2x4x4xf32>, %e1: tensor<1x2x4x4xf32>):
  %c = npu.conv2d ins(%a, %f : ...) outs(%e0 : ...) {...} -> tensor<1x2x4x4xf32>
  %t = npu.relu ins(%c : ...) outs(%e1 : ...) -> tensor<1x2x4x4xf32>
  npu.yield %t : tensor<1x2x4x4xf32>
} -> tensor<1x2x4x4xf32>
```

**Every value the region reads is an operand, destinations included.**
`npu.fused_op` is `IsolatedFromAbove`, so there is no other route in, and its
verifier admits only `npu` operations inside, so a `tensor.empty` cannot be
cloned into the body even if it were tempting.

### Where it does not fire

`test/Transforms/fuse-ops.mlir`, with the two guards Section 12 names.

- **The producer has more than one use.** A second reader of the intermediate
  would have to read a value living inside a region it is not in, and the only
  ways out of that are to duplicate the producer or to yield two results, which
  are a compute cost and a dialect change respectively.
- **An already fused producer is not fused again.** Running the pass twice
  produces one region and not a region inside a region.
- **The consumer's input is a function argument**, so there is no producer to
  pull in.
- **The producer is a pool.** Pooling reads a window rather than one element, so
  there is no elementwise activation to keep on chip with it, and a region
  around a pool alone would state a fusion that did not happen.

---

## The four upstream passes the levels run

`-canonicalize`, `-cse`, `-sccp` and `-symbol-dce` come from MLIR and Section
12's table puts all four in the `-O` levels. They are described here because
what they do to `npu` operations is what this project depends on, and Section
12's negative test rule applies to every pass in that table rather than only to
the ones written in this repository. `test/Transforms/level-passes.mlir` carries
a positive and a negative case for each.

| Pass | What it does here | Negative case |
|---|---|---|
| `-canonicalize` | removes every `npu` operation nothing reads, because they all carry `Pure`, and merges duplicate constants. This is the whole mechanism behind `eliminatesDeadCode = true` and behind Section 17.3a's dead subgraph leaving the instruction count unchanged | the operations the result depends on are all still there |
| `-cse` | merges identical operations over identical operands, including the `tensor.empty` destinations | a third operation differing in one operand is not merged |
| `-sccp` | propagates a constant into a private function all of whose callers pass the same one | a function called with two different constants keeps its argument |
| `-symbol-dce` | removes a private function nobody calls | a called private function, and every public one, is kept |

**`-canonicalize` hoists constants and that had a cost**, which is D-0035:
hoisting is right for an operation whose cost is zero and wrong for one that
becomes a DRAM transfer. The fix is in `-npu-lower-to-npuisa`, which sinks a
constant back to the operation that reads it, and it is described under that
pass below.

**`-cse` merges `tensor.empty` destinations and that had a cost too**, which is
D-0034: two operations sharing one destination are two pure functions of the
same meaningless input at the tensor level and two instructions writing one
buffer below it. The fix is in the lowering, for the same reason.

**`-sccp` did nothing at all until the dialect gained a constant materializer**,
which is D-0033. It computed the right lattice and had nowhere to put the
answer, so it reported no change on every input and its ablation row would have
been a row of zeros for a reason that was four missing lines rather than a
property of the pass.

---

## `-npu-lower-to-npuisa`

Lowers the `npu` tensor dialect to `npuisa` instructions on `memref`s in the
two memory spaces of Section 8. Implemented in
`lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp`, registered by
`mlir::npuisa::registerNPUISAPasses()`, and runnable from `npu-opt`.

**Ablatable: no.** Section 12's table marks it so. Removing it leaves a program
with no instructions in it.

**Ablation delta: not measured.** The harness lands at P10.

### What it does

The operator map, which is the pass's contract:

| `npu` | becomes |
|---|---|
| `constant` | `npuisa.const` in DRAM, plus the one `dma_load` that brings it on chip |
| `conv2d` | `npuisa.conv2d` |
| `matmul` | `npuisa.matmul` |
| `add` | `npuisa.add`, with a stride 0 view when the right hand operand is rank 1 |
| `mul` | `npuisa.mul`, the same way |
| `relu` | `npuisa.relu` |
| `max_pool2d` | `npuisa.pool_max` |
| `avg_pool2d` | `npuisa.pool_avg` |
| `reshape` | `npuisa.reshape`, with a destination the pass allocates |
| `transpose` | `npuisa.transpose` |
| `concat` | `npuisa.concat` |
| `batch_norm` | a multiply and an add over per channel constants computed at rewrite time |
| `fused_op` | no instruction: the region is flattened into its parent |
| `yield` | erased with the region it terminated |

Three rules govern the memory, and together they are Section 8's boundary
invariant:

1. A function argument becomes a `memref` in `#npu.dram`, and if the body reads
   it, exactly one `npuisa.dma_load` brings it into a scratchpad buffer. An
   argument nothing reads gets no load.
2. A function **result** becomes a trailing `#npu.dram` argument, and one
   `npuisa.dma_store` writes it. The lowered function returns nothing.
3. Everything else lives in `#npu.scratchpad`: a `tensor.empty` destination
   becomes a `memref.alloc` there and the instruction's `outs`.

**Every argument of the lowered function carries `npuisa.arg`**, a string
attribute holding `"in"` or `"out"`. *Added at P6.* The order is unchanged, the
model's arguments first and its outputs appended after them, so argument N of
the lowered function is still argument N of the model; what the attribute adds
is that the split is stated rather than counted. The encoder reads it to build
the input and output regions of the binary and refuses a function without it,
which turns a convention the compiler could not check into one it can.
`docs/ARCHITECTURE.md` carries the reasoning and the alternatives.

The attribute is written by this pass because this pass performs the split. A
function the idempotence guard finds already lowered is left alone and therefore
gains nothing, so hand written `npuisa` IR that wants to be encodable carries
its own.

### Before and after

Input, one convolution and one relu, in the shape the frontend emits:

```mlir
func.func @conv_relu(%x: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
  %d0 = tensor.empty() : tensor<1x8x8x8xf32>
  %c = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d0 : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  %d1 = tensor.empty() : tensor<1x8x8x8xf32>
  %r = npu.relu ins(%c : tensor<1x8x8x8xf32>) outs(%d1 : tensor<1x8x8x8xf32>)
       -> tensor<1x8x8x8xf32>
  return %r : tensor<1x8x8x8xf32>
}
```

Output:

```mlir
func.func @conv_relu(%arg0: memref<1x3x8x8xf32, #npu.dram> {npuisa.arg = "in"},
                     %arg1: memref<1x8x8x8xf32, #npu.dram> {npuisa.arg = "out"}) {
  %alloc = memref.alloc() : memref<1x3x8x8xf32, #npu.scratchpad>
  npuisa.dma_load %arg0, %alloc
    : memref<1x3x8x8xf32, #npu.dram> to memref<1x3x8x8xf32, #npu.scratchpad>
  %cst = npuisa.const dense<1.000000e+00> : tensor<8x3x3x3xf32>
       -> memref<8x3x3x3xf32, #npu.dram>
  %alloc_0 = memref.alloc() : memref<8x3x3x3xf32, #npu.scratchpad>
  npuisa.dma_load %cst, %alloc_0
    : memref<8x3x3x3xf32, #npu.dram> to memref<8x3x3x3xf32, #npu.scratchpad>
  %alloc_1 = memref.alloc() : memref<1x8x8x8xf32, #npu.scratchpad>
  npuisa.conv2d ins(%alloc, %alloc_0 : memref<1x3x8x8xf32, #npu.scratchpad>,
                                       memref<8x3x3x3xf32, #npu.scratchpad>)
                outs(%alloc_1 : memref<1x8x8x8xf32, #npu.scratchpad>)
                {dilations = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                 strides = array<i64: 1, 1>}
  %alloc_2 = memref.alloc() : memref<1x8x8x8xf32, #npu.scratchpad>
  npuisa.relu ins(%alloc_1 : memref<1x8x8x8xf32, #npu.scratchpad>)
              outs(%alloc_2 : memref<1x8x8x8xf32, #npu.scratchpad>)
  npuisa.dma_store %alloc_2, %arg1
    : memref<1x8x8x8xf32, #npu.scratchpad> to memref<1x8x8x8xf32, #npu.dram>
  return
}
```

Two loads in, one store out, and nothing between the convolution and the
activation. That is the invariant, and
`test/Dialect/NPUISA/dma-boundaries.mlir` asserts it.

### The rank 1 channel broadcast

`docs/adr/0005-channel-broadcast-on-add-and-mul.md` obliges this pass and P7
together: an `npu.add` or `npu.mul` whose right hand operand is rank 1 lowers to
a per channel broadcast, and the simulator kernel reads that operand with a
channel stride of one and a spatial stride of zero.

Here the contract is a **type** rather than a flag. The rank 1 buffer is viewed
at the destination's extents with strides that are 1 on the channel axis and 0
on every other:

```mlir
%cst = npuisa.const dense<2.000000e+00> : tensor<8xf32> -> memref<8xf32, #npu.dram>
%alloc_0 = memref.alloc() : memref<8xf32, #npu.scratchpad>
npuisa.dma_load %cst, %alloc_0 : memref<8xf32, #npu.dram> to memref<8xf32, #npu.scratchpad>
%view = memref.reinterpret_cast %alloc_0 to
          offset: [0], sizes: [1, 8, 4, 4], strides: [0, 1, 0, 0]
      : memref<8xf32, #npu.scratchpad>
     to memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>
npuisa.mul ins(%alloc, %view : memref<1x8x4x4xf32, #npu.scratchpad>,
                               memref<1x8x4x4xf32, strided<[0, 1, 0, 0]>, #npu.scratchpad>)
           outs(%alloc_1 : memref<1x8x4x4xf32, #npu.scratchpad>)
```

The view has the destination's shape, which is what `npuisa.mul` requires of its
operands and why no dialect change was needed, and it reads the same C values
over and over, which is what a per channel scale means. It adds no transfer: the
`C` floats were loaded once and the view is over them. Buffers are NCHW below
the tensor level whatever the layout was, so the channel axis is always 1 and
this view is identical under both layouts.

**Why a view and not a new instruction.** A stride 0 read is an addressing mode,
not a machine operation, and Section 9.1 already carries operand strides in the
`Instruction` record for the layout decision. One field, two uses, and no new
opcode: the instruction set of Section 5.4 is closed and this stays inside it.
The obligation it places on P7 is exact: a compute kernel indexes its operands
through their strides, and the broadcast then costs it no special case.

### The batch norm decomposition

A batch norm the folding pass did not fold is **legal rather than a hard error**,
per Section 5.2. It decomposes into a multiply and an add, with the per channel
constants computed at rewrite time:

```
invStd = 1 / sqrt(variance + epsilon)
scale  = gamma * invStd
shift  = beta - mean * scale
y      = x * scale + shift
```

That evaluation order is the order the code evaluates in, and it is written down
because it is observable: floating point multiplication is not associative, so a
reader comparing this against onnxruntime needs to know which of several
algebraically equal forms produced the number.

Both halves come out as the rank 1 broadcast above, so there is one code path
for per channel arithmetic rather than two. The four parameter constants are
consumed at rewrite time and erased, so an unfolded batch norm costs two
transfers rather than four; see D-0016 for what happened before they were.

The decomposition happens on tensors, before the conversion, because it is a
tensor level identity. Written against memrefs it would have to allocate its own
intermediate and thread its own destination, which is work the conversion
already does once.

### `npu.fused_op`

The region is **flattened** into its parent: block arguments become operands,
the yielded value becomes the result, and the operations inside become ordinary
instructions. The operation is `IsolatedFromAbove`, so the block arguments are
the only route out of the region and the substitution is complete by
construction.

This is what makes fusion mean what Section 5.2 says it means. Afterwards the
chain's intermediate is an ordinary scratchpad buffer that no `dma_store` ever
writes, so a fused convolution and its activation have no DMA between them. It
is the same reason the property holds for an unfused chain, which is the point:
fusion's benefit under this memory model is that the intermediate stays on chip,
and flattening is how the lowering makes that true rather than a special case it
has to remember.

`-npu-fuse-ops` is the pass that creates these regions and it landed at P9. The
handling here landed at P4, five phases ahead of its producer, because the P4
gate asks for no DMA between a convolution and its fused activation and a
diagnostic would have met the letter of "do something named" while leaving the
gate unmeetable. It was right the first time it was given something to flatten,
which is the argument for writing the consumer early rather than the other way
round.

### One destination, one buffer

*Added at P9, as the fix for D-0034.*

A `tensor.empty` is a value with no contents, so two operations that use the
same one as a destination are two pure functions of the same meaningless input,
and at the tensor level that is entirely correct. This pass is the layer at
which it stops being correct: it converts one `tensor.empty` into one
`memref.alloc`, so a shared destination becomes two instructions writing one
buffer, and when the second of them also *reads* that buffer through a window
the program is simply wrong.

Nothing produced this shape before `-O2` existed, because the importer emits one
`tensor.empty` per compute operation. `-cse` produces it in one step, because
two `tensor.empty` operations of the same type are identical operations with no
operands and merging them is exactly what a common subexpression eliminator is
for.

So every use after the first gets its own clone, before the conversion. The fix
is here rather than in `-cse` for the same reason the aliasing rule is here:
this is where a value becomes a buffer.

### A constant's transfer is emitted where its data is read

*Added at P9, as the fix for D-0035.*

This pass emits one `npuisa.const` and one `npuisa.dma_load` at the position of
each `npu.constant`, so where a constant sits in the block decides when its
bytes are fetched, and the two port cost model of Section 10.1 charges exactly
that: a transfer overlaps a computation only when the computation does not
depend on it. The importer emits every constant immediately above its first use,
so at `-O0` the loads interleave with the compute and LeNet's overlap fraction
is 0.83.

MLIR's canonicalizer hoists every `ConstantLike` operation to the top of the
block, which is right for an operation whose cost is zero and wrong for one that
turns into a DRAM transfer. On LeNet at `-O1` it moved all eleven loads above all
the compute, in an order that put the last layer's weights first, so the first
convolution waited for essentially the whole 16441 cycle transfer budget: the
overlap fell to 0.0005 and the cycle count rose by 37 percent. An optimization
level that made the program slower, from a pass that changed no instruction.

So each constant is moved back down, above the run of constants and destinations
that immediately precedes its first reader. It sinks past **computation and
nothing else**, which is why it is a no operation at `-O0`, where the importer's
placement is already this one, and why the `-O0` baseline did not move when it
landed.

There is nothing to schedule here and no scheduling pass is implied: this pass
chooses where to put a transfer it is about to create, and the answer is where
the data is needed.

### The layout encoding becomes the strided layout map

Section 5.5: `#npu.layout` is a tensor encoding and does not survive
bufferization by itself, so the lowering materialises it as the memref's strided
layout map.

A tensor's extents are written in the order its layout names, so an NHWC tensor
is written `N, H, W, C`. A buffer below this level is always NCHW, which is what
every `npuisa` verifier reads and what the simulator's kernels index. So the
extents are permuted back into NCHW and the permutation moves into the strides,
over the same underlying bytes:

```mlir
// tensor<1x8x8x3xf32, #npu.layout<nhwc>>  becomes
memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>
```

Over `1 x 8 x 8 x 3` elements laid out NHWC the channel stride is 1, the width
stride is 3, the height stride is 24 and the batch stride is 192. An NCHW tensor
gets no layout map at all, because the identity written out at length is still
the identity and printing it would make the NHWC case prove nothing.

Keeping the extents in the order the layout wrote them was the alternative, and
it is wrong for a reason worth recording: the strides come out contiguous again
and the layout leaves no trace below the tensor level, which is exactly the
outcome Section 5.5 says would make `-npu-assign-layout` a pass whose delta is
structurally zero.

### What it refuses, by name

Every refusal is emitted from a validation stage that runs before any operation
has been rewritten. A diagnostic emitted from inside a conversion pattern
competes with the framework's own "failed to legalize operation" message, and
the reader is then left with two messages and no way to tell which is the
answer. `test/Dialect/NPUISA/lowering-diagnostics.mlir` covers all of these.

| Refused | Because |
|---|---|
| an `scf` operation | Section 5.2: this instruction set has no branches, so a tiled loop is fully unrolled before lowering |
| a function with more than one block | the same rule, seen from the other side |
| a function declaration | this compiler has no calls and no linking |
| a tensor with a dynamic extent | nothing below this level can represent one |
| a tensor whose element type is not f32 or i8 | those are the two memories this machine has |
| a batch norm parameter that is not an `npu.constant` | the decomposition computes its multiplier and addend at rewrite time |
| a batch norm whose variance plus epsilon is not positive | the decomposition takes a square root of it |
| an `npu.constant` carrying a layout encoding | the lowering does not permute constant data; a layout assignment pass materialises the permuted constant itself |
| an `npu.transpose` that changes the layout as well as the extents | the lowering represents a permutation of extents only; a layout assignment pass folds its own inverse transposes |

The last two are unreachable from the frontend, which emits no layout encodings
at all. They become reachable when `-npu-assign-layout` lands, and they are
diagnosed now so that pass arrives at a stated rule rather than at a verifier
failure from inside a pass.

### Where it does not fire

Section 12's negative test rule: a pass with only positive tests is not
adequately tested, because a pass that fired unconditionally would pass them
all. Three cases in `test/Dialect/NPUISA/lowering.mlir`:

- **A function that is already lowered is left exactly as it is.** No second
  signature conversion, no second load, no second store. This also makes the
  pass idempotent, which is what lets a pipeline run it without first
  establishing whether something else already did.
- **A right hand operand that already has the destination's shape is not
  broadcast.** No `memref.reinterpret_cast` appears.
- **An argument nothing reads is not loaded.**

### Tests

| File | What it pins |
|---|---|
| `test/Dialect/NPUISA/lowering.mlir` | one case per pattern, plus the three negative cases |
| `test/Dialect/NPUISA/dma-boundaries.mlir` | Section 8's scoped invariant, immediately after this pass |
| `test/Dialect/NPUISA/lowering-diagnostics.mlir` | every refusal above, by the substring it emits |

An end to end test is the other half of the 17.1 row for a lowering pattern,
"plus an e2e test if it is reachable from ONNX". Every pattern in the table above
except `fused_op` and `yield` is reachable from ONNX and got one at P8, in
`test/Python/test_end_to_end.py`. Those two are structural, no ONNX node imports
to them, and `-npu-fuse-ops` reaches them instead: since P9 the same matrix runs
at `-O2`, where five of the seven models hold a region, so the flattening has an
end to end test like everything else.

---

## `-npu-allocate-scratchpad`

Assigns every scratchpad buffer a byte offset in one flat arena, and spills to
DRAM when the offsets do not fit. Section 13.1 in full. Implemented in
`lib/Dialect/NPUISA/Transforms/AllocateScratchpad.cpp`, with the arithmetic in
`ScratchpadAllocation.cpp` beside it, registered by
`mlir::npuisa::registerNPUISAPasses()`, and runnable from `npu-opt`.

**Ablatable: no.** Section 12's table marks it so. Removing it leaves a program
whose buffers have no addresses, which is not a program.

**Ablation delta: not measured.** The harness lands at P10.

### What it does

Four steps, and the order of the middle two is the part that is easy to get
backwards.

1. **Liveness**, over the `memref.alloc` operations in `#npu.scratchpad` of a
   single block function. Each range runs from the allocation to the last
   operation that reads or writes that memref. **A use through a view is a use
   of the buffer**: the walk follows `memref.view`, `memref.subview`,
   `memref.reinterpret_cast` and `memref.cast`, so the rank 1 broadcast of
   ADR 0005 keeps its scale buffer alive for as long as the multiply reads it.
   Sizes come from the memref type and its element type, through the same
   `computeBufferRange` the overlap rule of Section 8 measures with, so the
   allocator and the aliasing analysis cannot disagree about how large a buffer
   is.
2. **The sweep line.** One event list of `(index, +size)` at each definition and
   `(lastUse + 1, -size)` at each death, sorted once and walked once. O(n log n)
   for the sort and O(n) for the walk. The first index with a strictly greater
   sum wins the peak, and at equal indices deaths are ordered before
   definitions.
3. **Offset assignment**, in one of two strategies, described below.
4. **Spilling**, if and only if step 3 failed.

**The peak is a lower bound, not a placement test, and the spill trigger is
"offset assignment failed".** Peak simultaneous live bytes is the smallest arena
any placement could possibly need. Fragmentation means a program whose peak sits
under the budget can still fail to place, so a trigger of "peak exceeded budget"
would spill when it need not and fail to spill when it must. The two questions
have different answers on a real program, and
`test/Dialect/NPUISA/spill-heuristic.mlir` carries the one that separates them:
a function whose peak is exactly the budget, which the packer places and the
interval scheme does not.

Every allocation is then replaced by a `memref.view` at a constant byte offset
over one flat `memref<Nxi8, #npu.scratchpad>`, which is Section 8's rule that
the offset is an SSA operand rather than a discardable attribute. `N` is the
high water mark, not the budget: the arena is what was used.

### The two offset assignment strategies

Selected with `strategy=`, and both are present because Section 13.1 asks for
the production algorithm and a named baseline to measure it against.

| Value | What it does |
|---|---|
| `pack` (default) | The greedy by size offset calculation algorithm [R28], the one TFLite Micro's arena planner ships. Buffers are placed largest first |
| `interval` | The named baseline: the same placement rule in definition order, which is the interval scheme this project started from |

Both share one placement rule, and the sharing is the point: the difference
between them is the order and nothing else. For each buffer, gather the already
placed buffers whose live ranges overlap this one, walk their occupied byte
ranges in increasing offset order, and take the first gap the buffer fits in,
rounding the candidate offset up to the alignment after every block it skips.

Ties break deterministically, per ground rule 16:

- `pack`: larger bytes first, then longer span, then earlier definition index.
- `interval`: earlier definition index, then larger bytes, then longer span.

Offsets are aligned to 64 bytes by default, because the array of Section 5.3 is
16 by 16 and consumes a row of 16 `f32` lanes at a time, which is 64 bytes. The
alignment is a pass option so a test can pin it. **Sizes are not padded**, only
offsets aligned, so the high water mark is the last byte genuinely occupied.

### The fragmentation ratio

`npuisa.fragmentation_ratio` is the assigned high water mark divided by the
sweep line peak, which Section 13.1 calls the headline allocator metric and
which TelaMalloc [R26], MiniMalloc [R27] and the TFLite Micro planner [R28] all
report. It is 1.0 when the placement achieves the lower bound.

The two integers it is computed from are written on the function beside it, so a
reader can check the ratio rather than trust it.
`experiments/allocator_fragmentation.py` reports it per model under both
strategies.

### The two spill heuristics

Selected with `spill-heuristic=`. The candidate set is the buffers live across
the pressure peak index.

| Value | The rule |
|---|---|
| `longest-range` (default) | The longest live range crossing the peak |
| `cost` | A Belady style rule: `cost = bytes * (1 + reloads)` where `reloads` is the number of uses strictly after the peak index, and the **smallest** cost is spilled |

Ties break deterministically. `cost` breaks them by larger bytes first, then
longer span, then earlier definition index, which is Section 13.1 word for word.
`longest-range` uses the same three keys with span promoted to the front, so
that an ablation between the two heuristics is not partly a measurement of two
different tie breakers.

**The default is provisional and is marked as such.** Section 13.1 requires the
default to be chosen with data from the ablation across the whole suite, and
that harness lands at P10 with the experiment at P13. `longest-range` is the
simpler rule and the baseline, so changing the default later will be a move
*towards* the smarter rule with evidence behind it rather than away from one.

### Spilling, and what it emits

Section 13.1's semantics exactly: a `npuisa.dma_store` after the definition and
a `npuisa.dma_load` before each later use, with the reload replacing that use.
This is the **second of the three permitted DMA producers** of Section 8, and
the count is written on the function as `npuisa.spill_dma_count` so the sum over
the three is checkable.

The reload gets its own `memref.alloc`, which participates in liveness like any
other buffer. That is why the pass recollects everything from the IR after every
spill instead of patching a side table: a second spill round that had not seen
the reloads would mis-size the peak, which Section 13.1 names as the failure.

A buffer is **spillable** only if all of the following hold, and each has a
reason rather than a convention:

| Rule | Why |
|---|---|
| exactly one writer | the semantics are a store after *the* definition, and a buffer written twice has two |
| no view of it | a view is a second SSA name for the same bytes, and rewriting only the direct uses would leave the view reading a buffer whose contents had moved |
| not a reload, and not already spilled | both would let the loop spill its own output, which is how a spill loop fails to terminate |
| at least one read after the write | spilling a buffer nothing reads later adds a transfer and shortens no live range |
| an identity layout | `dma_store` requires its operands to agree, and a permuted buffer has no DRAM counterpart without deciding what order to write it in. That decision belongs to the relayouting transfer Section 12 marks as a future extension |

**The spill slot is a `memref.alloc` in `#npu.dram`**, marked with
`npuisa.spill_slot`. This is the one place in the compiler that allocates DRAM,
and it amends the P4 sentence in `docs/ARCHITECTURE.md` that nothing below the
tensor level does; the amendment is recorded there as a marked P5 extension.

### The function attributes it sets

| Attribute | Meaning |
|---|---|
| `npuisa.scratchpad_budget` | the budget the allocation was made against |
| `npuisa.scratchpad_bytes` | the arena actually used, which is the assigned high water mark |
| `npuisa.scratchpad_peak_bytes` | the sweep line peak, the lower bound the above is measured against |
| `npuisa.fragmentation_ratio` | the first divided by the second |
| `npuisa.spill_count` | buffers spilled |
| `npuisa.spill_dma_count` | DMA operations this pass inserted |

The budget comes from the `budget` pass option if it was given, then from the
function's `npuisa.scratchpad_budget` attribute, then from the default of
1048576 bytes. The option wins over the attribute because the option is a
command line override and the attribute is data the driver wrote. The attribute
is written back either way, so after this pass a function always says which
budget it was allocated against.

**One mebibyte is inherited, not invented.** It is the budget the previous build
of this project called the default and reported every generous budget cell at,
and Section 15 requires each model's tight budget to be a fraction of the peak
observed at the default. Moving it silently would move every tight budget cell
in the project's history at once.

### Before and after

Input, a three buffer chain of 1 x 8 x 4 x 4 `f32`, which is 512 bytes apiece:

```mlir
func.func @chain(%in: memref<1x8x4x4xf32, #npu.dram>,
                 %out: memref<1x8x4x4xf32, #npu.dram>) {
  %a = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %in, %a
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %b = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%a : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
  %c = memref.alloc() : memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%b : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%c : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %c, %out
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
```

Output. Three buffers, two offsets: `%c` takes `%a`'s bytes back, because `%a`
died at the first relu.

```mlir
func.func @chain(%arg0: memref<1x8x4x4xf32, #npu.dram>,
                 %arg1: memref<1x8x4x4xf32, #npu.dram>)
    attributes {npuisa.fragmentation_ratio = 1.000000e+00 : f64,
                npuisa.scratchpad_budget = 1048576 : i64,
                npuisa.scratchpad_bytes = 1024 : i64,
                npuisa.scratchpad_peak_bytes = 1024 : i64,
                npuisa.spill_count = 0 : i64,
                npuisa.spill_dma_count = 0 : i64} {
  %alloc = memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
         : memref<1024xi8, #npu.scratchpad>
  %c0 = arith.constant 0 : index
  %view = memref.view %alloc[%c0][]
        : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.dma_load %arg0, %view
    : memref<1x8x4x4xf32, #npu.dram> to memref<1x8x4x4xf32, #npu.scratchpad>
  %c512 = arith.constant 512 : index
  %view_0 = memref.view %alloc[%c512][]
          : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%view : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%view_0 : memref<1x8x4x4xf32, #npu.scratchpad>)
  %c0_1 = arith.constant 0 : index
  %view_2 = memref.view %alloc[%c0_1][]
          : memref<1024xi8, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.scratchpad>
  npuisa.relu ins(%view_0 : memref<1x8x4x4xf32, #npu.scratchpad>)
              outs(%view_2 : memref<1x8x4x4xf32, #npu.scratchpad>)
  npuisa.dma_store %view_2, %arg1
    : memref<1x8x4x4xf32, #npu.scratchpad> to memref<1x8x4x4xf32, #npu.dram>
  return
}
```

A buffer whose type carries a strided layout map, which is what an NHWC tensor
lowers to, gets the view at its extents followed by a `memref.reinterpret_cast`
that restores the layout, because `memref.view` requires an identity layout on
its result. The bytes are the same bytes: a permutation layout spans exactly the
contiguous extent its shape does.

### What it refuses, by name

| Refused | Because |
|---|---|
| a function with more than one block | Section 8. Liveness here is an ordering of one straight line stream, and an index in a second block is not comparable with one in the first |
| a budget too small even after everything spillable has been spilled | Section 13.1. The message carries the budget, the size of the buffer that could not be placed, the offset it wanted, and the sweep line peak as a lower bound, so a reader can tell whether a larger budget could ever help |
| an unknown `strategy` or `spill-heuristic` value | Section 13.1: a typo must not silently select a heuristic nobody asked for. The message names the offending string and lists the accepted values |
| an alignment that is not a positive power of two | the rounding is a mask |
| an allocation whose byte size cannot be computed | Section 13.1 takes sizes from the type, so a buffer it cannot measure is refused rather than guessed at |
| a malformed `npuisa.scratchpad_budget` attribute | a silent replacement by the default would produce a valid program allocated against a budget nobody asked for, and the number would travel into a result cell as though it had been measured |

**Every bad option is reported, not just the first**, so somebody who mistyped
two of them does not fix one, rerun, and discover the second.

### Where it does not fire

Section 12's negative test rule.

- **A function that has already been allocated is left exactly as it is**, which
  is the idempotence guard. It is not decoration: the arena is itself a
  scratchpad allocation, so a second run without the guard would allocate an
  arena for the arena and grow the program every time the pipeline ran. The
  guard is the `npuisa.scratchpad_bytes` attribute.
- **A function with nothing to allocate gets the attributes and no arena.** Zero
  is written down rather than left absent, because an absent attribute and an
  attribute of zero are different claims and only one of them says the allocator
  ran.
- **At a budget nothing needs spilling for, nothing is spilled.** A spiller with
  only positive tests would pass them all while spilling unconditionally, and
  the cost of that would be invisible in a lit file and very visible in the DRAM
  traffic numbers three phases later.

### Tests

| File | What it pins |
|---|---|
| `test/Dialect/NPUISA/scratchpad-alloc.mlir` | the offsets, by value, under both strategies: fits, reuse after death, no reuse while live, alignment rounding, fragmentation, the strided layout, the broadcast view, and the two negative cases |
| `test/Dialect/NPUISA/spill-heuristic.mlir` | the spill trigger, the emitted store and reload, the two heuristics choosing differently on one program, and the negative case at a generous budget |
| `test/Dialect/NPUISA/alloc-budget-too-small.mlir` | the budget diagnostic, under `-verify-diagnostics` |
| `test/Dialect/NPUISA/alloc-multiblock.mlir` | the multi block, unmeasurable size and malformed budget refusals |
| `test/Dialect/NPUISA/alloc-unknown-option.mlir` | all three option refusals at once |
| `unittests/Dialect/NPUISA/AllocatorTest.cpp` | Section 17.2's property test, the placement invariant over the same randomized sets, and every tie break |
| `experiments/compile_time_benchmark.py` | the growth curve at 500, 1000, 2000 and 5000 operations |
| `experiments/allocator_fragmentation.py` | the fragmentation ratio per model under both strategies |
