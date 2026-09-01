<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Reachability exemptions

*Diataxis type: reference.*

This file holds the `EXEMPT` block that `scripts/check-reachability.py` reads.
It is a machine parsed list, so the block below has a fixed shape and the prose
around it is not part of what the checker sees.

Law 2 of the build specification says that no operator representing imported
computation may exist in the `npu` dialect unless it is importable from ONNX,
lowerable to `npuisa`, encodable, simulatable, and exercised by at least one
model in the benchmark suite. An operator that cannot satisfy all five is
deleted. The one alternative to deleting it is an entry here, and an entry has
to be dated and has to name the phase that resolves it, because an exemption
with no end date is not an exemption but a quiet repeal of the law.

The checker fails with a per operation table naming the layer that is missing,
unless that operation is listed below. This list lives in its own file rather
than inside `DESIGN_DECISIONS.md`, which is a generated index: a frequently
edited, machine parsed list has no business inside a build artifact.

**Two categories are outside the importability clause by construction and are
not exemptions.** Region terminators such as `npu.yield`, and operations a pass
creates rather than the importer such as `npu.fused_op`, correspond to no ONNX
node, so importability is a requirement they can never meet. The checker
classifies them as structural by reading the classification out of the
operation's ODS description, and holds them to the remaining requirements:
lowerable, encodable or provably eliminated by lowering, simulatable through
whatever they lower to, and exercised by at least one model. Neither one needs
an entry in this file, and adding one here would be recording as a temporary
gap something that is a permanent and intended property.

**The block is empty, and has been since P9.** The P8 gate allowed either an
empty block or a dated entry per gap, and at P8 it was the second of those: two
entries, both the same fact seen twice. `npu.fused_op` and `npu.yield` are
created by `-npu-fuse-ops`, which Section 12 puts at `-O2`, and `-O2` did not
exist, so no model's IR could hold one. Nothing about either operation was
unfinished; what was missing was a producer.

`-npu-fuse-ops` landed at P9 and `-O2` runs it, `scripts/build-model-ir.py`
sweeps every level the compiler builds rather than `-O0` alone, and both
operations now appear in the IR of five of the seven models. The two entries
were deleted by the commit that wired the pass into a level, which is the commit
that made them false; a commit earlier would have deleted an exemption for a gap
that was still open and turned the check red.

An entry is a thing to be justified at a gate, not a routine way to land an
operator that is not finished.

## Entry form

Each entry is one block in the fenced list below, and carries:

- `op` the fully qualified operation name, for example `npu.some_op`.
- `layer` the requirement that is not met: `import`, `lowering`, `encoding`,
  `simulation`, or `model`.
- `date` the date the exemption was granted, as `YYYY-MM-DD`.
- `phase` the named phase that resolves it, for example `P9`.
- `reason` one line saying why the gap exists and what closes it.

An entry is deleted when the gap closes, and the commit that closes it says so.
Unlike the defect log, this file is not an audit trail: it is a live list of
what is currently exempt, and a stale entry here weakens a law rather than
recording history. The history lives in git.

## EXEMPT

```
# op                layer      date         phase   reason
```

None. Every operation of the `npu` dialect meets every layer of law 2 it is held
to: the twelve **imported computation** operations meet all five, and the two
**structural** ones meet the four that are not importability.

**What closed the last two, and why the timing was the whole of it.** The gap
was a phase boundary rather than unfinished work, and that decided what closing
it looked like. Nothing about `npu.fused_op` was missing at P8:
`-npu-lower-to-npuisa` flattened the region, `test/Dialect/NPUISA/lowering.mlir`
had a case for it, the ISA description recorded both operations as reaching the
encoder by elimination, and the simulator needed no kernel for either because
neither survives to the instruction stream. What was missing was a producer, and
`-npu-fuse-ops` is the only thing that creates one.

So the entries could not be deleted by the commit that landed the pass, only by
the commit that put it in a level and swept the model IR at that level. Those
are one commit at P9 and the entries went with it. Deleting them a commit
earlier would have been recording a gap as closed while it was open, and the
check would have said so.
