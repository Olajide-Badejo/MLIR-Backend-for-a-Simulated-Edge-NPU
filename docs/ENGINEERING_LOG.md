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
