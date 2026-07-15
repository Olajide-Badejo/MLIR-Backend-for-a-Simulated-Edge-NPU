# Engineering log

Dated entries recorded as problems happen: symptom, root cause, options considered,
chosen fix and why, commit, verification. This log is the raw material for the debug
report (report_debug) assembled at Phase 11.

## 2026-07-15 Phase 0: environment reconciliation

**Symptom.** The build spec (Section 3) assumes Ubuntu 24.04, a WSL budget of 20 GB
RAM / 24 processors, `LLVM_PARALLEL_COMPILE_JOBS=20`, and Python 3.10+. The actual
target machine reports different facts on every one of those axes, and one of them is
safety critical.

**Findings (verified 2026-07-15).**

- Disk is a non issue here. The WSL2 rootfs lives on `/dev/sdd` with about 932 GB free,
  and Windows C: has about 135 GB free. None of the spec's disk mitigations are needed.
- WSL2 Ubuntu 26.04 (not 24.04), kernel 6.18, user `elijah`, passwordless sudo.
- Toolchain present: gcc/g++ 15.2.0, cmake 4.2.3, ninja 1.13.2, git 2.53.0,
  python3 3.14.4, venv working. Missing and installed via apt: lld 21.1.8, ccache 4.12.3.
- The host `~/.wslconfig` deliberately caps WSL2 at `memory=12GB`, `processors=8`,
  `swap=8GB`, `autoMemoryReclaim=gradual`.

**Root cause of the deviation that matters.** The `.wslconfig` cap is not an oversight.
Its own header records that its absence crashed the machine on 2026-07-14: with no budget,
vmmemWSL grew to about 12.78 GB and never released it, leaving 2.2 GB free of 31.7 GB, and
the host paged itself to death (aggravated by GPU benchmark allocations spilling through
WDDM). The file explicitly instructs keeping build parallelism at or below 6 jobs.

**Options considered.**

1. Follow the spec literally: raise `.wslconfig` to 20 GB / 24 proc, build with 20 compile
   jobs. Rejected: directly reintroduces the condition that crashed the machine one day
   earlier, and 20 parallel gcc jobs compiling TableGen heavy MLIR translation units would
   exceed a 12 GB budget regardless.
2. Keep `.wslconfig` as is and adapt the build to the real budget. Chosen.

**Chosen fix.** Do not touch `.wslconfig`. Configure the one time LLVM/MLIR build with
`LLVM_PARALLEL_COMPILE_JOBS=6` and `LLVM_PARALLEL_LINK_JOBS=1`, link with lld, and enable
ccache. Accept that the build takes longer than the spec's 1 to 3 hour estimate at this
parallelism. Pin the LLVM tag to `llvmorg-22.1.8` (current tip of `release/22.x`) rather
than the moving branch, for reproducibility.

**Open risk carried forward.** Python is 3.14.4. torch, onnxruntime, and nanobind wheel
availability for 3.14 is unverified and is a risk for Phase 5 and later. To be checked
before the ONNX frontend work, with a 3.12 venv as the fallback plan.

**Verification.** `df`, `free`, `nproc`, `lsb_release`, tool `--version` probes captured
above. LLVM pinned tag confirmed by `git ls-remote`; shallow clone HEAD resolves to
`llvmorg-22.1.8`.

## 2026-07-15 Phase 0: repository relocated to WSL native storage (spaces break lit)

**Symptom.** After the LLVM build finished, `npu-opt` built and `npu-opt --help` worked,
but the very first lit test failed with `FileCheck: Too many positional arguments
specified!` and a shell error splitting a path at a space. The build tree sat under the
Windows project folder whose ancestors contain spaces (`Corrected Projects`,
`MLIR Backend for a Simulated Edge NPU`).

**Root cause.** LLVM's lit expands the `%s` substitution to the test file's real path and
does not quote it. Under the external `sh`, `npu-opt /mnt/c/.../Corrected Projects/...mlir`
splits into multiple arguments, so both `npu-opt` and `FileCheck` see too many positional
arguments. LLVM lit and FileCheck do not support spaces in paths, and lit resolves symlinks
to the real path, so a space-free symlink over the spaced directory did not help either
(verified: `%s` still expanded to the spaced `/mnt/c` realpath).

**Options considered.**

1. Keep the repo in the Windows folder and patch lit substitutions to quote `%s`. Rejected:
   fights the framework, fragile, and would have to be re-done for every future config.
2. Put the repo at a space-free Windows path such as `C:/Users/jidro/npu-mlir`. Space-free
   and my editing tools stay native, but WSL builds and lit still run over the slow 9p
   `/mnt/c` mount for the whole 12 phase project.
3. Host the canonical repo in WSL native storage at `/home/elijah/npu-mlir`. Space-free,
   native build and test speed, and the standard recommended layout for WSL2 development.
   Chosen.

**Chosen fix.** Relocated the git working tree (with history) to `/home/elijah/npu-mlir`.
Builds go to `/home/elijah/npu-mlir/build`. From Windows the repo is reachable at
`\\wsl.localhost\Ubuntu\home\elijah\npu-mlir`, which the Windows side editing tools handle
correctly (verified by reading and enumerating files over that path). A pointer file in the
original Windows folder records the new location. The original folder keeps only the spec
and that pointer, so there is a single source of truth.

**Verification.** After relocation, `ninja check-npu` reports `Passed: 1 (100.00%)` and
`npu-opt --help` lists the registered dialects. All paths in the generated
`lit.site.cfg.py` and in the `%s` expansion are now space-free.

## 2026-07-15 Phase 1: two ODS surprises building the first ops

Two failures worth recording while standing up the npu dialect, both of the "the
TableGen half and the C++ half of an interface are separate includes" family.

1. **`Variable not defined: 'SameOperandsAndResultType'`** from mlir-tblgen. This trait is
   not in `mlir/IR/OpBase.td` (it appears there only in FIXME comments). It is defined in
   `mlir/Interfaces/InferTypeOpInterface.td`, which must be included from the ops `.td`.

2. **`'InferTypeOpInterface' is not a member of 'mlir'`** at C++ compile time, after fixing
   1. Reason: `SameOperandsAndResultType` is a trait list that also mixes in
   `InferTypeOpInterface::Trait`, so the generated op class references
   `::mlir::InferTypeOpInterface`. The TableGen include is not enough; the C++ side needs
   `#include "mlir/Interfaces/InferTypeOpInterface.h"` in the ops header and the
   `MLIRInferTypeOpInterface` library on the dialect's `LINK_LIBS`.

**Lesson for later ops.** Any convenience trait may drag in an interface with a matching
C++ header and link library. When a trait is added in ODS, add its interface header and
link library at the same time rather than after the next build failure.

**Verification.** `ninja check-npu` reports `Passed: 2 (100.00%)`; the core ops round trip
through two `npu-opt` passes without change.

## 2026-07-15 Phase 2: CommonFolders poison template argument

**Symptom.** Using `constFoldBinaryOp<FloatAttr>` and `constFoldUnaryOp<FloatAttr>` from
`mlir/Dialect/CommonFolders.h` to fold the pointwise ops failed to compile with a static
assertion: "PoisonAttr is undefined, either add a dependency on UB dialect or pass void as
template argument to opt-out from poison semantics."

**Root cause.** In this LLVM release these helpers gained a `PoisonAttr` template parameter
that defaults to `ub::PoisonAttr`, whose definition is only available if the UB dialect is
linked. The parameter is the third one, after `AttrElementT` and `ElementValueT`. My first
fix passed `void` as the second argument, which wrongly set `ElementValueT` to void.

**Chosen fix.** Opt out of poison propagation by passing all three explicitly:
`constFoldBinaryOp<FloatAttr, APFloat, void>(...)` and the unary equivalent. This avoids
taking a dependency on the UB dialect for folds that cannot produce poison.

**Verification.** `-canonicalize` folds `add`, `mul`, and `relu` over constants to a single
`npu.constant`, and the canonicalization patterns (relu idempotence, reshape identity and
reshape of reshape) fire. The BatchNorm folding pass turns `bn(conv(x, W))` into one conv
with hand verified weights `[2, 6]` and bias `[-0.5, -1.5]`. `check-npu` reports 5 of 5.
