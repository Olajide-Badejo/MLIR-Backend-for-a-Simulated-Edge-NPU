<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# 9. The NDEBUG contract is carried by the two non MLIR binaries, and the second LLVM tree is declined

- **Status:** Accepted
- **Date:** 2026-09-01
- **Diataxis type:** explanation

## Context

Section 9.3 requires the simulator's bounds checked accessors to refuse
gracefully in **every build mode**, and the P7 gate asks for the trap tests run
in an assertions build and in a release build with all four invocations shown.
Producing the second of those turned out not to be a matter of
`-DCMAKE_BUILD_TYPE=Release`, which is D-0028: `LLVMConfig.cmake` sets
`LLVM_ENABLE_ASSERTIONS` from the LLVM tree this project is configured against,
`HandleLLVMOptions` reads it and appends an explicit `-UNDEBUG` **after** the
`-DNDEBUG` a release configuration supplies, and the last `-D` or `-U` on a
command line wins. `NPU_FORCE_NDEBUG` exists because of that and it sets the
switch before `HandleLLVMOptions` runs.

Then D-0031, at P8: a directory configured that way cannot build anything that
links MLIR. Turning `_GLIBCXX_ASSERTIONS` off in this project's translation units
while the LLVM archives they link keep it on changes the definition of the
standard containers across a link, which is an ODR violation and undefined
however carefully the flags are written. An `npu-opt` built there aborts inside
MLIR's own context construction on an empty module, before any of this project's
code runs. D-0031 named the real fix and left it: **a second LLVM tree, built
with `-DLLVM_ENABLE_ASSERTIONS=OFF`.**

Two things were left open together, and this record closes both. The P7 handoff
left "the NDEBUG third CI build" as an orchestrator decision, and D-0031 left the
second LLVM tree as a cost nobody had agreed to pay.

**A third fact arrived while this was being decided, and it changes the
arithmetic.** `.github/workflows/ci.yml` claimed the sanitizers job was already
the NDEBUG half of Section 9.3's clause, "which configures RelWithDebInfo and
therefore compiles with `-DNDEBUG`". Measured on 2026-09-01, that configuration
against the assertions LLVM ends its compile line with
`-D_DEBUG -D_GLIBCXX_ASSERTIONS -UNDEBUG`, so the sanitizers job runs the trap
tests with assertions **on**, exactly like `build-and-test`. That is D-0036. So
CI had no NDEBUG coverage at all, and had a comment saying it did.

## Decision

**The NDEBUG half of Section 9.3's contract is covered in CI by a build that
configures `-DNPU_FORCE_NDEBUG=ON` and builds `NPUSimulatorTests` and
`NPUEncodingTests` only. A second LLVM tree without assertions is declined.**

The `ndebug` job of `.github/workflows/ci.yml` is that build. It asserts in its
own configure log that the option took, by grepping for the line beginning
`NDEBUG:`, which is the same shape as the sanitizers job's `Fuzzers: ON` check
and exists for the same reason: an option that silently failed to take would
leave the job building what `build-and-test` already built, and passing. Without
that line the job is D-0028 with more steps.

**The argument for declining the tree is that the contract is entirely inside
those two binaries.** Section 9.3 is about the simulator's accessors. The
simulator links the format library and LLVM's `Support` and no MLIR at all, and
the encoder links less than that. Both are exactly the set `NPU_FORCE_NDEBUG`
can build soundly, which is what the option's own `message(WARNING)` says. A
non-assertions LLVM would buy the ability to run `npu-opt`, `npu-translate` and
the lit suite in a mode no clause asks about, and it would cost:

- an hour of runner time per build of that tree, and a second image to publish,
  version and keep in step with the first;
- a second `LLVM_IMAGE` reference in `ci.yml`, so that a phase which moved the
  LLVM tag would have to move two images rather than one;
- the same again on every developer machine that wanted to reproduce the job,
  where the existing `~/llvm-project/build` is a fixed cost the project has
  already paid once and does not propose to pay twice.

**What is deliberately not claimed.** This does not close D-0031. The MLIR
linking tools still cannot be built without assertions in this project, and if a
later phase writes a clause about *those* binaries in a non-assertions build,
this decision is the one to revisit and the second tree is the answer. What
changes is that the cost is being declined against a stated requirement rather
than deferred against an unstated one.

## Consequences

**CI gains a job rather than a step.** It is a third build configuration in a
third directory, which is what the sanitizers job is, and folding it into
`build-and-test` would put one job name over two builds so that a red run named
neither. It also runs in parallel with the other three, where a step would be
serial inside the job that already carries the longest budget in the file.

**The job builds two targets by name and that is a correctness requirement
rather than an optimization.** Sweeping the directory would build `npu-opt` and
reproduce D-0031 as heap corruption inside MLIR, in a job whose whole purpose is
to be trustworthy about undefined behaviour.

**The four runs Section 9.3 asks for are now two of them in CI and four of them
locally.** `docs/BUILD.md` carries the local configure line and
`docs/PHASE_STATE.md` records the local runs, as it has since P7. What P9b adds
is that the two release side runs happen on every push rather than on whichever
developer remembered the second directory.

**The claim in `ci.yml` that the sanitizers job covered this is deleted rather
than softened**, and D-0036 is the record of it having been there. A comment
asserting coverage the configuration does not provide is worse than no comment,
because it is the reason nobody looked.
