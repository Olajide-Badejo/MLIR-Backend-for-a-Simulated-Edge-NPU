<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 2. Pin the ONNX opset at 23, bound by the exporter

- **Status:** Accepted
- **Date:** 2026-08-19
- **Diataxis type:** explanation

## Context

The importer needs one opset number, fixed, because operator semantics move
between opsets and a compiler that does not know which revision of `AveragePool`
it is reading cannot be correct about padding. The v1 tree used opset 17, which
was already old when it was chosen and is much older now.

The number is not simply the newest opset the ONNX specification defines. It is
the intersection of three separate ceilings, and the specification's ceiling is
only one of them:

1. the highest opset `torch.onnx.export` can emit,
2. the highest opset `onnx.checker` accepts,
3. the highest opset onnxruntime can execute.

A model above any of those three is a model this project cannot produce, cannot
validate, or cannot get a reference output for. Since onnxruntime is the
correctness oracle for the whole project, the third ceiling is as binding as the
first.

The build specification predicted the exporter would bind it at 23 and then
told me to run the probe anyway rather than copy that number out of the
document. That is the right instruction. A pin has to be produced by
measurement on the machine that will use it, because the machine is what the
pin describes.

## Decision

**Pin the ONNX opset at 23.**

The probe from Section 3.3 of the build specification was run on this machine
on **2026-08-19**, inside `~/npu-venv`. It prints the three tool versions and
the three ceilings, then walks candidate opsets downward from the checker
ceiling, building a one node `Relu` model at each and requiring both
`onnx.checker.check_model` and an `onnxruntime.InferenceSession` to accept it.

What it reported:

| Ceiling | Tool | Version | Value |
|---|---|---|---|
| Exporter | torch | 2.13.0+cpu | `ONNX_MAX_OPSET` = **23** |
| Checker | onnx | 1.22.0 | `onnx_opset_version()` = **27** |
| Runtime | onnxruntime | 1.27.0 | see below |

The downward walk started at 27. Opset 27 was **rejected by the checker or the
runtime**, reported as `Fail`. Opset 26 was accepted by both, so the highest
opset the checker and the runtime both accept is **26**.

The pin is the minimum of the exporter ceiling and that number:

```
min(23, 26) = 23
```

**The exporter binds it.** onnx and onnxruntime are both ahead of torch here,
and the gap is three opsets, so the constraint is entirely on the production
side rather than the consumption side. That is worth recording because it says
where to look when the pin is next revisited: a torch upgrade moves this number,
an onnx or onnxruntime upgrade almost certainly does not.

The gap between the onnx package's declared ceiling of 27 and the 26 that
actually round trips is itself the argument for probing rather than reading. The
installed onnx 1.22.0 announces support for an opset that its own checker or the
installed runtime then refuses.

## Consequences

The importer of Phase P3 targets opset 23 and rejects a model that declares
anything else with a diagnostic naming the opset it found, rather than importing
it and hoping the semantics match. The model generator of Phase P3 exports at
opset 23.

Operator semantics are read at opset 23 and nowhere else. The pooling arithmetic
that the verifiers of Phase P1 implement, including the `ceil_mode` right padded
window rule, is the opset 23 behaviour.

The resolved number, the three ceilings, the three tool versions and the probe
date are recorded again in every result manifest from Phase P10 onward, so a
result file carries the opset it was produced under and a later reader does not
have to trust that the pin never moved.

Revisit this record when torch is upgraded, since torch is what binds the pin
today. Re-run the probe rather than reasoning about the new ceiling: the whole
point of this record is that the number came from a measurement, and a
successor number that came from a changelog would not be the same kind of fact.
