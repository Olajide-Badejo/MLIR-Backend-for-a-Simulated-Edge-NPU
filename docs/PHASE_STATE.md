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

**P4, lowering to `npuisa`.** Branch `phase/p4-lowering`, cut from `main` at
`2d12588`, which is the P3 merge. Four commits, not pushed. The fourth is the
one that carries this table, so it is the branch tip and is named by subject
rather than by a sha it cannot know.

| Commit | Subject |
|---|---|
| `f86bc64` | `fix(dialect): refuse a dynamic extent at the type level, as NPUTypes.td claimed` |
| `e08fc78` | `docs(adr): record the lowering mechanism and the bufferization attempt` |
| `5f08222` | `feat(npuisa): lower the npu dialect to npuisa instructions on memrefs` |
| tip | `docs: record the P4 defects and hand off the phase` |

On the branch point, since two phases running have had something to say about
it. Local `main` was stale when this session began, and P3's handoff had already
recorded that as a recurring hazard. The first command of the session was
`git fetch origin && git branch -f main origin/main`, which moved local `main`
to `2d12588`, and the branch was cut from that. The conclusion was drawn against
`origin/main` after a fetch, not against a local ref.

## Gate status

The P4 gate is the roadmap entry's, plus the deliverables Sections 8 and 12
attach to this pass. Every item is **met locally**. Item by item, with the
proof.

| Gate item | Proof |
|---|---|
| A lit lowering test per pattern, with `CHECK` and `CHECK-NOT` | `test/Dialect/NPUISA/lowering.mlir`, one case per row of the operator map in `docs/PASSES.md`: the function boundary, `constant` with `tensor.empty`, `conv2d` with and without a bias, `matmul`, `add` and `mul` in both the same shaped and the rank 1 forms, `relu`, both pools, `reshape`, `transpose`, `concat`, `batch_norm`, `fused_op`, and both layouts. Every case carries at least one `CHECK-NOT` |
| The negative case Section 12 requires | Three of them, and the strongest is `an_already_lowered_function_is_untouched`: a function already on memrefs comes out identical, asserted with `CHECK-NEXT` throughout, which also makes the pass idempotent. Plus `same_shaped_operands_are_not_broadcast`, which asserts no `memref.reinterpret_cast` appears, and `an_unread_argument_is_not_loaded` |
| `dma-boundaries.mlir` asserts the scoped invariant immediately after lowering | `test/Dialect/NPUISA/dma-boundaries.mlir`, six functions, all of them `npu` dialect input so that what is checked is what the lowering produced rather than what I could write by hand. The run line is `npu-opt %s --npu-lower-to-npuisa` and nothing after it, which is what scopes the claim to the one point at which it is true |
| DMA appears only at memory boundaries | Each case brackets its computation with `CHECK-NOT: npuisa.dma`, so the assertion is that there is nothing in between rather than that the transfers I listed are present. `a_chain_keeps_its_intermediates_on_chip` runs four instructions on one load and one store |
| None between a convolution and its fused activation | `nothing_between_a_convolution_and_its_activation`, on an `npu.fused_op` region: two loads, then `conv2d`, then `relu`, then one store, with `CHECK-NOT: npuisa.dma` between each pair |
| An NHWC tensor lowers to a memref carrying the strides that layout implies | `nhwc_becomes_a_strided_buffer` in `lowering.mlir`: `tensor<1x8x8x3xf32, #npu.layout<nhwc>>` becomes `memref<1x3x8x8xf32, strided<[192, 1, 24, 3]>, #npu.scratchpad>`, with `CHECK-NOT: memref<1x8x8x3xf32` so the extents are asserted to have moved. `nchw_carries_no_layout_map` is the other half: an NCHW tensor gets no strided map, without which the NHWC case would prove nothing |
| A value that cannot be assigned a memory space is refused with a naming diagnostic | `lowering-diagnostics.mlir`: `a_dynamic_extent`, `an_unsupported_element_type` and `a_dynamic_destination`. Each names the function or the operation, the argument or result number, quotes the type, and gives the reason as a clause. The decision and the explanation come out of the same branches of `convertTensorType`, so a diagnostic cannot contradict the refusal it explains |
| Batch norm decomposes into a multiply and an add, with the constants computed at rewrite time | `batch_norm_decomposes_into_a_multiply_and_an_add` checks the computed constants **by value**, not by shape: gamma 2 and 4, beta 1 and 0, mean 0 and 1, variance 3 and 3, epsilon 1 give a multiplier of 1 and 2 and an addend of 1 and -2. The arithmetic is the thing that could be wrong while the shapes stayed right |
| Non constant batch norm parameters produce a diagnostic naming the operation and the operand, never a generic legalization failure | `a_computed_batch_norm_parameter` matches `the gamma operand of this batch norm is not an npu.constant`, and `a_zero_denominator` covers a variance plus epsilon that is not positive |
| The ADR 0005 rank 1 broadcast lowering | `add_broadcasts_a_rank_one_addend` and `mul_broadcasts_a_rank_one_scale` check the whole `memref.reinterpret_cast` including `strides: [0, 1, 0, 0]` and the resulting `strided<[0, 1, 0, 0]>` operand type. `a_broadcast_view_adds_no_transfer` in `dma-boundaries.mlir` checks that it costs no DMA |
| `npu.fused_op` handling, decided and documented | Flattened, not diagnosed, and `docs/PASSES.md` says why: the gate asks for no DMA between a convolution and its fused activation, and a named diagnostic would have met the letter of "do something rather than nothing" while leaving the gate unmeetable. `fused_op_is_flattened` is the lit case |
| An `scf` operation or a multi block function is a named diagnostic | `an_scf_loop` and `two_blocks` in `lowering-diagnostics.mlir`. The validation walk is pre order so the `scf.for` is named rather than the `scf.yield` inside it |
| The `one-shot-bufferize` attempt made and its outcome recorded either way | `docs/adr/0006-lowering-mechanism-and-the-bufferization-attempt.md`: five runs with their commands and their output, and a section saying plainly which parts of the infrastructure were adopted and which were not |
| `-npu-lower-to-npuisa` registered and runnable from `npu-opt` | `mlir::npuisa::registerNPUISAPasses()` in `tools/npu-opt/npu-opt.cpp`. Every lit file above runs it through `npu-opt` |
| `docs/PASSES.md` created, per ground rule 12 | Created in the same commit as the pass, with before and after IR, the operator map, the refusal table, and the negative cases. The ablation delta field says it is not measured and why, rather than being left blank |

### Verification output

Every command below was run on this branch at `5f08222`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 10 discovered, 10 passed, 0 failed. Seven at P3, plus this phase's three |
| `build/bin/NPUInterfaceTests` | 18 tests, 18 passed |
| `build/bin/NPUTilingTests` | 12 tests, 12 passed |
| `python -m pytest test/Python -q` | 142 passed, 7 deselected, exit 0, unchanged from P3 |
| `mypy` | no issues found in 11 source files |
| `black --check .` | 19 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 126 of 126 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, and the **lowering layer is decidable for the first time**: 12 imported computation operations, every one found in `LowerNPUToNPUISA.cpp` |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `git status --short` | empty |

**Two gtest binaries exist, not three.** `NPUInterfaceTests` and
`NPUTilingTests` are what P1 and P2 built; `NPUAllocatorTests` is P5's and
`NPUEncodingTests` is P6's, per the activation table. Both existing binaries are
green and neither was touched by this phase, which is the claim that matters
here: the dialect change in `f86bc64` narrows a type constraint, and a gtest
building tensors with dynamic extents would have failed on it.

## Activation proofs, and why there are none

**No CI job or step activates at P4.** Section 19.0's table has no P4 row. Every
step this phase's work runs under, `check-npu`, `dash-lint.sh`, `reuse lint`,
`check-reachability.py --skip-models` and the `DIALECT_REFERENCE.md` staleness
gate, activated at P0 or P1 and has been on ever since. The next activations are
`NPUAllocatorTests` at P5, `NPUEncodingTests`, the ISA staleness gate and the
sanitizers job at P6, `NPUSimulatorTests` at P7, and the full reachability check
at P8.

This paragraph exists so that the absence is a decision rather than an omission.
A phase report with no activation proof section reads the same whether the
author checked and found nothing or did not check at all, and P3's handoff spent
real effort on two proofs, so a reader is entitled to wonder where this phase's
went.

One step did change behaviour without changing its guard, and it is worth
naming. `check-reachability.py --skip-models` has been on since P1 and reported
the lowering layer as "not yet decidable" ever since, because
`LAYER_HOMES["lowering"]` points at
`lib/Dialect/NPUISA/Transforms/LowerNPUToNPUISA.cpp` and that file did not
exist. It exists now, so the check started asking a question it had never asked,
and it failed the first time it asked it: six operations whose mnemonics did not
appear literally in the pass source, because C++ class names are `Conv2DOp` and
the check looks for `conv2d`. The fix is the operator map comment at the head of
that file, which is documentation the pass should have carried anyway. That is
the table becoming decidable exactly as designed, on the day its source
appeared, with no edit to the script.

## Open questions

Six. None blocks the gate.

**`f86bc64` is a P1 dialect change made at P4, and a reviewer should look at it
as one.** `NPU_FloatTensor`, `NPU_QuantTensor` and `NPU_AnyTensor` are
`StaticShapeTensorOf` rather than `RankedTensorOf`, which narrows what the
dialect accepts. The reasoning is D-0015: the comment above those definitions has
claimed since P1 that a dynamic extent is refused at the type level, it was not,
and `npu.reshape` then aborted the tool with an assertion and no diagnostic at
all on IR that had parsed. I considered recording it and leaving it, since it is
not this phase's code and the phase was already large. I fixed it because the
alternative is a known crash sitting behind a sentence saying it cannot happen.

**The out parameter convention is this phase's decision and P6 inherits it.** A
lowered function returns nothing and its results are trailing `#npu.dram`
arguments. `test/Dialect/NPUISA/ops-memref.mlir` has described a lowered function
that way since P2, so the shape is not new, but nothing before now committed the
compiler to producing it. The encoder reads its input and output regions out of
that argument list, and there is no marker distinguishing an input argument from
an output one beyond position: the first N are inputs and the last M are outputs,
where M is the original result count. **If P6 wants that explicit, an argument
attribute is the place, and it should be added there rather than inferred.**

**The two layout diagnostics are placeholders for work `-npu-assign-layout`
owns.** A constant carrying a layout encoding, and a transpose that changes the
layout as well as the extents, are both refused by name. Neither is reachable
today, because the frontend emits no layout encodings at all. Both become
reachable at P13 and both are that pass's work: it materialises its own permuted
constants and folds its own inverse transposes, which Section 12 already says it
does. If P13 finds it needs a relayouting copy after all, the shape of one is
available without a new opcode, since `npuisa.transpose` with an identity
permutation between two buffers of differing strides is exactly a relayout under
this representation. That is **not implemented**, no gate depends on it, and
nothing may cite it as available.

**The scratchpad attributes are not set by this pass.** Section 8 says the
function carries `npuisa.scratchpad_bytes` and `npuisa.scratchpad_budget`, which
the encoder and the simulator read. `scratchpad_bytes` is an output of the
allocator and cannot be known here; `scratchpad_budget` is an input to it. The
lowering copies the function's discardable attributes across, so a budget set on
the input function survives, and P5 owns both. Named because a reader comparing
the lowered output against Section 8 will notice they are absent.

**No `memref.dealloc` is emitted, deliberately.** The allocator derives live
intervals from the instruction stream, so a deallocation here would be a weaker
second statement of a lifetime it computes exactly. If P5's sweep line turns out
to want explicit lifetime markers, that is P5's decision and it should change
this pass rather than working around it.

**The overlap analysis has not been shown a `memref.reinterpret_cast`.**
`computeBufferRange` in `NPUISAMemoryOverlap.cpp` walks `memref.view` and
contiguous static `memref.subview` back to an allocation. The broadcast view this
phase introduces is a `reinterpret_cast`, which is a new shape for it to meet. It
does not matter yet, because that analysis runs only for the asynchronous DMA
rules and this pass emits no asynchronous transfer, which
`the_lowering_emits_no_asynchronous_transfer` asserts. It matters at P5, when the
allocator rewrites allocations as views underneath these casts, and at the double
buffering pass, whichever reaches it first. The conservative answer, `Unknown`,
is the safe one and is what an unrecognised producer already yields, so the
failure mode is a refusal rather than a race. **Whoever touches that analysis
next should add a `reinterpret_cast` case deliberately, and decide whether a
stride 0 view has a byte range at all**, which is a real question: it addresses C
floats and spans none.

## Next command

```bash
git push -u origin phase/p4-lowering
```

Then watch CI. Expect no job or step to run for the first time, per the
activation section above. Two things are worth looking at. The
`DIALECT_REFERENCE.md` staleness gate sees a regenerated reference for the first
time since P1, and it is the gate most likely to catch a mistake in `f86bc64`.
And `check-reachability.py --skip-models` reports the lowering layer as checked
rather than as undecidable for the first time. Then open the merge pull request.

There are **no activation proofs to perform this phase**, which is a change from
P3's handoff and is explained above rather than left as a gap.

## Next phase

**P5, scratchpad allocation.** Section 13.1 in full: both offset assignment
strategies, `strategy=pack` and the interval scheme as the named baseline, both
spill heuristics, the sweep line from the start, the five allocator lit cases,
the sweep line property test against brute force in `NPUAllocatorTests`, the
three diagnostics each with a `-verify-diagnostics` test, and
`fragmentation_ratio` computed and reported per model.

Four things P4 leaves on P5's desk beyond the roadmap entry, all expanded in the
open questions above: the two scratchpad attributes are P5's to set, this pass
emits no deallocations so P5 owns lifetime entirely, the allocator's views will
sit underneath this pass's `memref.reinterpret_cast` broadcast views, and the
overlap analysis has not been shown one.

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
