<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

<!--
GENERATED FILE. DO NOT EDIT BY HAND.

This index is generated from the architecture decision records in docs/adr/.
Every edit belongs in the record itself; an edit made here is lost the next
time the generator runs. Regenerate with:

    python scripts/gen-design-decisions.py

and check it for staleness with:

    python scripts/gen-design-decisions.py --check
-->

# Design decisions

*Diataxis type: explanation (index).*

This is the generated index over the architecture decision records in
[`docs/adr/`](adr/). Each record is written once, in the standard form of
Section 20 of the build specification: title, status, context, decision,
consequences. A decision that changed is not edited in place; the old record
is marked superseded and a new one is written, so the reasoning that was true
at the time stays readable.

The reachability `EXEMPT` block is **not** here. It lives in
[`EXEMPTIONS.md`](EXEMPTIONS.md), because an exemption list is live, machine
parsed and frequently edited, and a generated index is none of those.

| Number | Title | Status | Date |
|---|---|---|---|
| 0001 | [Pin LLVM at `llvmorg-22.1.8` and reuse the existing build](adr/0001-llvm-tag-and-build-reuse.md) | Accepted | 2026-08-18, reconfirmed working 2026-08-19 |
| 0002 | [Pin the ONNX opset at 23, bound by the exporter](adr/0002-onnx-opset-pin.md) | Accepted | 2026-08-19 |
| 0003 | [The resolved tool matrix, as installed on 2026-08-19](adr/0003-resolved-tool-matrix.md) | Accepted | 2026-08-19 |
| 0004 | [Emit `npu` IR from Python through unregistered operations, verified by `npu-opt`](adr/0004-frontend-ir-emission-mechanism.md) | Accepted | 2026-08-19 |
| 0005 | [`npu.add` and `npu.mul` take a rank 1 channel operand, and the importer normalises to it](adr/0005-channel-broadcast-on-add-and-mul.md) | Accepted | 2026-08-19 |
| 0006 | [The lowering produces memrefs directly, and One-Shot Bufferize was measured before that was decided](adr/0006-lowering-mechanism-and-the-bufferization-attempt.md) | Accepted | 2026-08-20 |
| 0007 | [The dataflow is weight stationary, and it is pinned](adr/0007-dataflow.md) | Accepted | 2026-08-31 |
| 0008 | [The per model tight scratchpad budgets, measured at P8](adr/0008-per-model-tight-scratchpad-budgets.md) | Accepted | 2026-08-31, re-measured unchanged 2026-09-01 |
| 0009 | [The NDEBUG contract is carried by the two non MLIR binaries, and the second LLVM tree is declined](adr/0009-ndebug-coverage-without-a-second-llvm-tree.md) | Accepted | 2026-09-01 |
| 0010 | [The tight budget is a property of a program, so it does not cross the batch axis](adr/0010-the-tight-budget-is-measured-per-program-not-per-model.md) | Accepted | 2026-09-01 |
