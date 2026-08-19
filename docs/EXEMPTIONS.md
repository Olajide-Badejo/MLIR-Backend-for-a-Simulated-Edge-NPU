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

The P8 gate expects this block to be empty. An entry is a thing to be justified
at a gate, not a routine way to land an operator that is not finished.

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
# (no entries)
```

None. Every operation in the `npu` dialect meets every layer of law 2, or does
not exist yet.
