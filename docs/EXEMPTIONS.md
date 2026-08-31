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

The P8 gate expects this block to be empty **or** every gap to be a dated entry
naming its phase, and at P8 it is the second of those. Two entries are in force
and both are the same fact seen twice: `npu.fused_op` and `npu.yield` are
created by `-npu-fuse-ops`, which lands at P9, so no model's IR can contain one
at P8. They are structural and therefore outside the importability clause, and
they satisfy lowering, encoding and simulation; the model layer is the one they
cannot satisfy until the pass that creates them exists.

An entry is a thing to be justified at a gate, not a routine way to land an
operator that is not finished. These two are justified by a phase boundary
rather than by unfinished work, they name the phase that closes them, and the
commit that lands `-npu-fuse-ops` deletes them.

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
npu.fused_op        model      2026-08-31   P9      created by -npu-fuse-ops, which lands at P9, so no model's IR holds one yet
npu.yield           model      2026-08-31   P9      the terminator of an npu.fused_op region, absent for exactly as long as the region is
```

Two, and they are one fact. Every **imported computation** operation of the
`npu` dialect meets all five layers of law 2 at P8, with no exemption of any
kind. The two structural operations meet lowering, encoding and simulation, and
cannot meet the model layer until `-npu-fuse-ops` exists to create them.

**The gap is a phase boundary rather than unfinished work**, and the distinction
matters because it decides what closing it looks like. Nothing about
`npu.fused_op` is missing: `-npu-lower-to-npuisa` flattens the region,
`test/Dialect/NPUISA/lowering.mlir` has a case for it, the ISA description
records both operations as reaching the encoder by elimination, and the
simulator needs no kernel for either because neither survives to the instruction
stream. What is missing is a **producer**. `-npu-fuse-ops` is the only thing
that creates an `npu.fused_op`, Section 12 puts it at `-O2`, and `-O2` arrives
at P9.

So the commit that lands `-npu-fuse-ops` deletes both entries, and if it does
not, the check goes red at the next run rather than the pass going quietly
untested.
