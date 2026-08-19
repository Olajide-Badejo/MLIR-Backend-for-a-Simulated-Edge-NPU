<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 4. Emit `npu` IR from Python through unregistered operations, verified by `npu-opt`

- **Status:** Accepted
- **Date:** 2026-08-19
- **Diataxis type:** explanation

## Context

Section 5.1 draws the pipeline as `.onnx` into a Python importer built on the
MLIR Python bindings, and out the other side as `npu` dialect IR on tensors.
Section 11 says what the importer must do operator by operator. Neither says
how the Python process is supposed to produce operations belonging to a dialect
that exists only as C++.

That gap is real rather than an oversight I can read past. The `npu` dialect of
P1 and the `npuisa` dialect of P2 are C++ only: there is no `MLIRPythonSources`
extension, no generated `npu` Python module, and no plan to build one, because
building one means adding a nanobind extension, a second build target that has
to find the same LLVM, and a second place every operation definition has to be
kept in step. The `mlir_core` package that the LLVM build installs carries the
upstream dialects and nothing of this project.

So the importer has three ways to produce an `npu.conv2d`, and they are not
equally honest.

1. **Registered Python bindings for the dialect.** The direct route, and the
   one that would give the Python process the real verifiers. It costs a
   nanobind extension module and its build wiring, and it puts a second
   definition of every operation in the tree. It is a phase of work for a
   convenience.
2. **Textual assembly.** Build the module as a string and hand it to `npu-opt`.
   It needs no bindings at all, which is its whole appeal, and it means writing
   an SSA namer, a type printer, and a float formatter by hand. The float
   formatter is the part that decides it: a LeNet has about sixty thousand
   parameters and every one of them has to round trip through decimal text
   without losing a bit.
3. **Unregistered operations through the bindings.** Set
   `allow_unregistered_dialects` on the context, create each `npu` operation
   with `ir.Operation.create`, and let MLIR handle SSA values, types,
   locations, and `DenseElementsAttr` construction straight from numpy. The
   module prints in generic form. Nothing verifies it, because an unregistered
   operation has no verifier to run.

Option 3 buys everything the bindings are good at and gives up exactly one
thing: verification in process. That is the thing this project cannot give up.
Law 1 is that there are no silent wrong answers, and an importer whose output
nothing checks is the purest possible violation of it.

## Decision

**Emit through option 3, and make `./build/bin/npu-opt` the verification gate,
mandatory rather than a test time extra.**

The importer builds the module with the MLIR Python bindings on a context with
`allow_unregistered_dialects = True`. `func`, `tensor`, and the builtin types
are used as registered dialects, which is what `tensor.empty` needs. Every
`npu` operation is created generically. The module then goes through `npu-opt`,
which has the real dialect registered and runs the real verifiers, and **the
text `npu-opt` prints is the importer's return value**. An importer run that
cannot reach `npu-opt`, or whose module `npu-opt` rejects, raises. There is no
mode in which unverified IR leaves this package.

Two consequences of that are worth stating as rules rather than leaving to be
discovered.

**The round trip is the contract, so the canonical form is what tests assert
against.** `npu-opt` parses the generic form and prints the custom assembly of
Section 5.3, so a test asserting `npu.conv2d ins(...) outs(...)` is asserting on
output that the real parser and the real printer both agreed on. A test written
against the generic form the bindings produced would be asserting on a string
this project wrote and this project read, which proves nothing.

**A misspelled attribute name is caught mechanically, not by review.** This is
the one failure mode option 3 introduces that the other two do not have. MLIR
promotes the inherent attributes of a registered operation out of the generic
attribute dictionary into its properties when it parses, and an attribute whose
name does not match an inherent one is left behind in the discardable
dictionary. It is still legal IR. `strydes` for `strides` on a `npu.conv2d`
would therefore parse, print, and verify, with the operation's real `strides`
taking whatever the ODS default is and the typo riding along as a discardable
attribute nobody reads.

The importer closes that by asking `npu-opt` for the generic form as well as
the custom one, and rejecting any `npu` operation that carries a discardable
attribute dictionary. The generic printer writes properties as `<{...}>` and
discardables as `{...}`, so the distinction is visible in the text and the
check is a parse rather than a judgement. No operation this importer emits has
a legitimate discardable attribute, so the rule is total and needs no
exceptions list.

## Consequences

`./build/bin/npu-opt` is a runtime dependency of `python/npu_frontend`, not
merely of its tests. The binary is located from `NPU_OPT` in the environment,
then from `build/bin/npu-opt` under the repository root, then from `PATH`. When
none of the three resolves, the importer raises and names all three places it
looked. It does not fall back to returning unverified text, and the pytest
suite fails rather than skipping, because a skipped verification and a passed
verification must not look the same.

This is why the CI `pytest` step is ordered after the build step rather than
beside it, and it is why the pytest job runs in the LLVM container: it needs a
built `npu-opt`, which a plain runner does not have.

Emitting IR this way means the importer's own source contains no `npu` verifier
logic, and that is deliberate. Shape arithmetic that the dialect already
verifies is not reimplemented in Python to produce a nicer error message; the
convolution whose output extent is impossible is diagnosed by
`NPUShapeUtils.cpp`, the one place Section 7.2's arithmetic lives. The
importer's diagnostics cover what the dialect cannot see, which is the ONNX
side: an unsupported operator, an attribute this project refuses, a broadcast
that is not one of the permitted shapes.

Revisit this record if the project ever grows a nanobind extension for its own
dialects, which would make option 1 cheap and would move verification back into
the importer's process. Nothing in this decision blocks that; the round trip
would become a redundant second check rather than the only one, and it would be
kept anyway, because it is what proves the printed form parses.
