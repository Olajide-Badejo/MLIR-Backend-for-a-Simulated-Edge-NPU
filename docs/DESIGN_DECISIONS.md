# Design decisions

The choices that shaped the project, and why I made them.

## Builtin tensor type with fixed fp32 elements

The `npu` dialect uses the builtin ranked `tensor` type rather than a custom type.
That keeps interop with the MLIR infrastructure, so the generic folding, dead code
elimination, and conversion machinery apply without op specific glue. Elements are
fixed to fp32, matching the core scope of fp32 CNN inference.

## Bias operand plus activation attribute, not fused op names

Convolution and matmul take an optional bias operand and carry an activation
attribute rather than exploding into a set of named fused ops (conv, conv_bias,
conv_relu, conv_bias_relu, and so on). This mirrors the TFLite style of fused
operators and gives the fusion pass a single clean target: it just sets the
attribute.

## BatchNorm folding as its own named pass

After the weights are loaded, the batch norm scale, offset, mean, and variance are
constants, so the normalization folds into a preceding convolution's weights and
bias by real tensor arithmetic. Making this a dedicated, tested pass keeps the
arithmetic honest and gives the report a concrete subsection.

## No branch instructions in the ISA

Inference graphs are static DAGs, so the instruction stream is straight line. A
branchless ISA is a legitimate simplification, stated as such rather than hidden.

## Two level memory model with explicit DMA

The `npuisa` dialect makes the DRAM and scratchpad split explicit: compute
instructions touch the scratchpad only, and DMA moves data across the boundary.
This is what makes fusion experimentally interesting. A fused conv plus bias plus
relu keeps its intermediate in the scratchpad, and under a constrained budget the
allocator spills fewer buffers, a measurable difference in DRAM traffic that the
benchmark suite reports.

## Dialect conversion for the lowering, not ad hoc rewrites

The tensor to instruction lowering is a representation change (SSA tensors become
scratchpad buffers with addresses). That is exactly what the dialect conversion
framework with a `TypeConverter` and materializations is for, so the lowering uses
it and inserts DMA only at the DRAM boundaries.

## Fixed header plus length prefixed sections for the binary

The `.nbin` format is a fixed header followed by fixed order, length prefixed
sections rather than a bit packed encoding. It is simpler to get right and to
disassemble, and it is still a real, documented, decodable format.

The sections are **not** tagged, and this heading and paragraph said they were
until 2026-08-09. Nothing skips an unrecognised field, because nothing can: every
field's position depends on the fields before it. The consequence is deliberate
and is what the version field is for, and the version policy in
`docs/ISA_MANUAL.md` states it. Recorded here because the false phrase appeared in
three places, and `docs/ASSESSMENT.md` 13.4 item 7 had found only two of them.

## Analytical cost model, labeled as such

The simulator's performance numbers come from an analytical model (MACs over
systolic throughput, bytes over DMA bandwidth, elements over lane width, plus a
fixed issue overhead), not a cycle accurate model. Every number it produces is
labeled a simulated estimate, and the constants are documented assumptions.

## Reachability exemptions

Every op in the `npu` dialect is supposed to be reachable end to end: importable
from ONNX, lowerable to `npuisa`, encodable, simulatable, and exercised by a
model in the benchmark suite. `scripts/check-reachability.py` enforces that on
every push and fails on any op that is not, unless the op has a dated row here.

The rule exists because `transpose`, `concat`, and `batch_norm` sat in the
dialect for twelve phases with verifiers, round trip tests, and no conversion
pattern at all, so any graph containing one failed to legalize. Nothing noticed,
because LeNet contains none of them.

Running the checker for the first time found six unreachable ops rather than the
three the audit identified. The two extra findings are worth stating plainly.
`add` and `mul` are fully lowered, encoded, and simulated, and have no ONNX
converter, so nothing can ever produce one from a real model; they exist only
for hand written IR. And `avg_pool2d` is reachable at every layer but is not
exercised by any model, because LeNet uses max pooling throughout. That last one
is exactly the class of gap this check is for: nothing is broken, but a code path
the project claims to support has never run on a real graph.

These are targets tied to phases, not promises with dates attached to nothing. If
one slips, the row moves and the reason gets written down rather than the check
being disabled.

<!-- REACHABILITY-EXEMPT-BEGIN -->

| Op | Missing layers | Target | Phase and plan |
|---|---|---|---|
| `npu.transpose` | importer, lowering, encoder, simulator, model | 2026-12-31 | U7. New `TRANSPOSE` opcode appended (not renumbered), a simulator kernel, an ISA manual entry, and a lowering pattern. The Gemm importer already builds transposed weight constants at compile time, so nothing regresses. |
| `npu.concat` | importer, lowering, encoder, simulator, model | 2026-12-31 | U7. New `CONCAT` opcode, kernel, and lowering. This is the one that unlocks branching topologies, so the inception block in the U8 model suite depends on it. |
| `npu.batch_norm` | importer, lowering, encoder, simulator, model | 2026-12-31 | U7 and U8. Lowering decomposes an unfolded batch norm into mul plus add rather than adding an opcode, since the folding pass handles the common case. Reaching it also needs `Identity` in the importer, because `torch.onnx.export` only leaves a `BatchNormalization` node behind with `do_constant_folding=False`, which emits `Identity` nodes the importer currently rejects. |
| `npu.add` | importer, model | 2026-12-31 | U7. Add an `Add` converter to `op_mapping.py`. Exercised by the small ResNet block's residual connection in the U8 suite. |
| `npu.mul` | importer, model | 2026-12-31 | U7. Add a `Mul` converter. Also becomes reachable through the `batch_norm` decomposition above. |
| `npu.avg_pool2d` | model | 2026-12-31 | U8. Nothing to implement; it needs a model that uses average pooling. `GlobalAveragePool` in the depthwise separable block covers it. |

<!-- REACHABILITY-EXEMPT-END -->

## MIT license, not Apache-2.0 with LLVM exceptions

The v2 build specification pinned Apache-2.0 with LLVM exceptions, on the
reasoning that an out of tree MLIR project should match upstream LLVM's terms so
code could move between them. Commit `48a0be3` relicensed to MIT and the change
was never written down, which left a pinned assumption silently contradicted.

Recording it now, with the reason. Nothing in this repository is derived from
LLVM source; it links against LLVM and MLIR as libraries, which their license
permits under any downstream terms. The LLVM exception exists to let LLVM's own
runtime code be embedded in binaries without attribution obligations, and no
runtime library here is distributed that way. MIT is the simplest permissive
license that says what I mean for a portfolio project, and it is what the
README and the LICENSE file already claim. Changing the code to match the spec
would be worse than changing the record to match the code.

## A recorded benchmark result goes stale on compiler change, not on any commit

Every number in the README and both reports traces back to a JSON file in
`experiments/results/`. Those files carry a manifest naming the commit and cost
model constants that produced them, and the harness refuses to reuse a result
whose manifest no longer matches.

`UPGRADE_SPEC_V3.md` section 8.1 item 4 words the rule as "stale if its
`manifest.git_sha` differs from the current sha". Implemented literally that is
unusable. A result can only be committed by a commit that comes after the run
that produced it, so every result would be stale from the moment it landed and
every invocation would regenerate the entire suite, which also means the
committed files would never match their own commit.

What the rule protects is that a published number came from the code currently
in the tree. So the implemented check is: the manifest sha equals HEAD, or
nothing under `RESULT_INPUTS` (the dialect, passes, encoder, simulator, tools,
frontend, and the harness itself) has changed between that sha and the working
tree, uncommitted edits included. A README fix does not invalidate a
measurement; a one line change to the cost model does. `--allow-stale` reuses
results anyway when that is deliberately wanted.

One consequence worth stating: the results found in the tree at the start of
this work recorded `git_sha: 8095dbec`, and that is not a commit in this
repository at all. They were generated from a working tree that was never
committed in the form that produced them, so the published numbers were not
reproducible from any point in the history. That is the failure mode this rule
exists to prevent, and it is why the check treats an unrecognised sha as stale
rather than assuming it is merely old.

## Ablating a pass means removing every occurrence of it

`_passes_for_level(2)` is
`["-canonicalize", "-npu-fuse-ops", "-canonicalize", "-symbol-dce"]`.
`-canonicalize` runs twice, once before fusion and once after, so "ablate
canonicalize" has two possible meanings: remove one occurrence, or remove the
pass.

The ablation removes **every** occurrence. The question an ablation answers is
"is this pass worth having", and a table row that removed only the second
occurrence would answer "is running it a second time worth it", which is a
different and narrower question. Removing all occurrences is also the only
version of the question whose answer generalises to a pipeline where the pass
appears once.

Recorded because the choice is invisible in the output: both readings produce a
row labelled `canonicalize`, and they can disagree. On LeNet they do not, since
removing both occurrences already changes nothing, but that is a fact about this
model rather than a reason the choice does not matter. The evaluation prose
states that the pass runs twice in the full pipeline so a reader is not misled by
a single row.

## The generated report tex is tracked

`report/generated/` holds `macros.tex` and `results_table.tex`, which both PDFs
include and which are produced from `experiments/results/` by
`report/scripts/results_to_tex.py`. It was gitignored, on the reasonable looking
argument that generated files do not belong in version control.

That argument is wrong here, and 2026-08-09 is when it was reversed. The rule
this repository actually needs is that every number in a published artifact
traces to a committed result. While the generated tex was untracked, nothing
connected the two: the PDFs were committed, the results were committed, and the
file carrying numbers from one to the other was not, so the two could disagree
indefinitely and nothing would notice. They did disagree. The committed
`macros.tex` sitting in the working tree read `GitSha 38af13633388` while the
committed results said something else again, and both PDFs cited `8095dbec`.

Tracking the directory makes the link checkable, and
`test_macros_match_the_committed_results` checks it: the `\GitSha` macro and
every cited figure must equal what the committed results say. The cost is that
regenerating the tex shows up as a diff, which is the point. A generated file is
worth tracking exactly when it is the evidence linking two other tracked things.
