<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 1. Pin LLVM at `llvmorg-22.1.8` and reuse the existing build

- **Status:** Accepted
- **Date:** 2026-08-18, reconfirmed working 2026-08-19
- **Diataxis type:** explanation

## Context

This project is an out of tree MLIR dialect, so it links against a build of
LLVM rather than vendoring one. The version policy in the build specification
says to pin the newest stable release of every tool, resolved by looking at
upstream on the day of resolution rather than by copying a number out of a
document. It also says never to pin a release candidate, and it says LLVM is
built exactly once on this machine and then linked against.

That last rule is not a preference. The guest has a hard memory ceiling, 15 GB
as of 2026-08-19 and 12 GB before that, set after running uncapped crashed the
host, and the only LLVM build this machine has
ever completed used `LLVM_PARALLEL_LINK_JOBS=1`. A rebuild is an hour or more of
wall clock time with a real chance of an out of memory kill partway through, so
a wrong pin here is expensive to undo in a way that a wrong pin of, say, a
Python package is not.

I resolved the tag on 2026-08-18 by checking upstream that day. `llvmorg-22.1.8`
was the newest stable, released 16 June 2026, eight patch releases into its
line. `llvmorg-23.1.0` was at release candidate 3 on 12 August with the final
announced for 25 August 2026, which means the 23.x line was close but was not
there yet.

Taking 23.x when it lands would not be a free upgrade. Four specific source
breaks sit between 22.x and 23.x, and every one of them lands on code this
project is about to write:

`BufferizationOptions::DefaultMemorySpaceFn` changes from taking `TensorType`
to `TensorLikeType`, and `UnknownTypeConverterFn` changes from returning
`BaseMemRefType` to `BufferLikeType`. Those two hooks are exactly how this
project attaches its memory space attributes during bufferization, so a lambda
written against the 22.x signatures does not compile.

The free helper `bufferization::getMemRefType()` is removed, along with
`BufferizationOptions::createAlloc`, `createMemCpy` and `createCast`. Upstream
tags the change as non functional, which it is for in tree code and is not for
out of tree code.

`TilingInterface` gained additional overloads of `getTiledImplementation`,
`generateResultTileValue`, `getTiledImplementationFromOperandTiles` and
`getIterationDomainTileFromOperandTiles`. This one breaks out of tree
implementers through plain C++ name hiding: implement only the old overload and
the inherited new one is hidden, and the class no longer compiles. The fix is a
`using` declaration per method. It is the item most likely to bite here, because
P1 puts `TilingInterface` on the `npu` compute operations and P13 builds the
tiling pass on it.

`usePropertiesForAttributes` is removed from ODS, so a `.td` file containing
`let usePropertiesForAttributes = 1;` becomes a hard TableGen error.

The real risk is none of those four individually. It is that there are no MLIR
release notes for either 22 or 23; the MLIR notes stop at LLVM 21. A 23.x
migration is discovered through compile errors rather than read about in
advance, and I would be discovering them on a machine that cannot afford a
second build.

## Decision

Pin LLVM at **`llvmorg-22.1.8`** and reuse the existing build at
`~/llvm-project/build` unchanged. No LLVM build happens in this project.

That build was verified on 2026-08-18 and reconfirmed on 2026-08-19 as Release
with assertions enabled, MLIR only, Native target only, linking with `lld`,
using `ccache`, and with the Python bindings ON. `mlir-opt`, `mlir-tblgen`,
`FileCheck` and `MLIRConfig.cmake` are all present and working, which is what
the P0 gate checks rather than taking the configuration on trust.

Section 3.2 of the build specification is therefore a contingency procedure and
not a P0 step. It runs only if the existing build is lost or is found broken,
and if it runs it runs inside WSL2 with `LLVM_PARALLEL_LINK_JOBS=1` and six
compile jobs. One exemption to "never build LLVM" exists and is deliberate:
`docker/Dockerfile.llvm` builds the same tag inside CI, on GitHub's hardware,
to produce the image every other job pulls. That is not a build on this machine.

Revisit this record when the existing build is lost, or when a phase needs an
MLIR feature that 22.1.8 does not have. Do not revisit it because 23.x became
newest; being newest is not by itself a reason to spend a rebuild.

## Consequences

The project is written against the 22.x APIs. That means the bufferization
hooks keep the `TensorType` and `BaseMemRefType` signatures, `getMemRefType()`
stays available, and the `TilingInterface` implementations of P1 need no `using`
declarations under this pin. **No `.td` file in this project ever writes
`let usePropertiesForAttributes = 1;`.** Under 22.x the default is already what
this project wants, so omitting the line costs nothing today and removes one
guaranteed breakage on any future move to 23.x. That rule is cheap enough to
follow unconditionally, so it is stated here rather than left to be remembered.

C++20 is available and is what the project builds as. LLVM 22 itself builds as
C++17, which is fine in one direction: this project's headers may use C++20
features, LLVM's may not require them, and nothing relies on a C++20 feature
crossing the ABI boundary.

A move to 23.x later is a real migration with a budget: four known source
breaks, no release notes, and a rebuild on a memory capped machine. It is not a version
bump. Whoever attempts it should read this record first and then read Section
3.2 before touching anything.

## The frozen v1 fallback

This record is one of the two places the build specification requires the
frozen fallback to be written down, so it is written here in full rather than
referenced.

The v1 repository lives at **`/home/elijah/npu-mlir`**, and its HEAD is
**`99408bc14b4f6331ce03ebf1dc0aecce1529afa8`**. It is frozen.

**Nothing in this project may ever write to that directory: no phase, no
script, no tool, no agent, not once.** It may be read, and only through a
command that cannot write. It is not deleted, not cleaned, not garbage
collected, and not tidied. **Only the owner may retire it.**

The reason is that git history and a second directory protect against different
failures. History protects against a bad commit. A second directory protects
against everything else, because if this rebuild goes wrong at any point,
deleting `~/npu-mlir-v2` returns the machine exactly to its pre build state with
no reasoning about reflogs required. That guarantee holds only for as long as
the frozen copy is untouched, which is why the rule is absolute rather than a
default with exceptions.

The LLVM build at `~/llvm-project/build` and the virtual environment at
`~/npu-venv` live outside both repositories and are shared by both. Freezing one
repository does not freeze the toolchain, and this record's reuse decision is
unaffected by the freeze.
