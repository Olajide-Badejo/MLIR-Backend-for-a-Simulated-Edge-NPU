<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Phase state

*Diataxis type: reference.*

Ground rule 17: this file is updated at the end of **every** session, including
a session that achieved nothing, and it carries four things: the current phase,
the status of its gate, the open questions, and the exact next command. This
build spans dozens of sessions, and reconstructing where it stood from `git log`
costs more than writing these lines did.

**Last updated:** 2026-08-19.

## Current phase

**P0, foundations.** Branch `phase/p0-foundations`, cut from `main`. Not merged.

## Gate status

The P0 gate is in the build specification's Section 23. Item by item.

### Done

- **The working clone exists** at `~/npu-mlir-v2`, made on 2026-08-19 by the
  Section 0.5 recipe including its ordering constraint. `git remote get-url
  origin` is
  `https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU.git`,
  which is the GitHub remote and not a local path.
- **`main` is reconciled and pushed**, at `52ed1da`, matching the remote. All
  six upgrade branches are published.
- **The v1 source tree is removed** in one commit, `2c756bd`, with the reason in
  the body.
- **The scaffold builds.** `npu-opt --help` runs, `ninja -C build check-npu`
  reports 1 of 1, `python -m pytest -q` reports 2 passed with `PYTHONPATH` unset
  in the environment.
- **Dash lint is clean** over the tree, and its self test meets 8 of 8
  expectations, proving the verbatim exemption in both directions.
- **`reuse lint` is clean**, 34 of 34 files carrying copyright and licence
  information.
- **pre-commit is wired**, 12 hooks, and passes over the whole tree.
- **The lock file is committed** and a clean venv install from it reproduces the
  environment, verified by installing into a fresh venv and importing torch,
  onnx, onnxruntime and numpy there.
- **The three architecture decision records are written and dated**, with the
  generated index over them:
  [0001](adr/0001-llvm-tag-and-build-reuse.md) the LLVM tag and build reuse,
  [0002](adr/0002-onnx-opset-pin.md) the opset pin at 23 from the probe run on
  this machine, [0003](adr/0003-resolved-tool-matrix.md) the resolved tool
  matrix.
- **The frozen fallback is recorded in writing**, in this file below and in
  record 0001, per the gate's explicit requirement that it appear in both.
- **The docs skeleton exists**: this file, `BREAKING_CHANGES.md`,
  `EXEMPTIONS.md`, `BUILD.md`, `DEFECT_LOG.md`, and the v3 era of
  `ENGINEERING_LOG.md` opened with its dated marker.
- **GHCR and Actions prerequisites are verified.** The repository is public,
  Actions are enabled, and `gh` is authenticated as `Olajide-Badejo`. The image
  publish runs through the CI `GITHUB_TOKEN`; the local token deliberately lacks
  `write:packages`.
- **The four CI workflows exist**, including `nightly.yml`, with every job
  guarded off per the activation table of Section 19.0, and each guarded off
  step prints in the run log that it is off and until which phase.
- **The GHCR LLVM image is published**:
  `ghcr.io/olajide-badejo/npu-mlir-llvm:llvmorg-22.1.8`, built by
  `llvm-image.yml` run 32205653261 on 2026-08-19, manifest digest
  `sha256:008fcc743cd9b7be1685b6cd24922081e87a0aaef3f9f3e5423368d949b545d6`.
  Both Dockerfile bases are pinned by `sha256` digest.
- **CI is green on the skeleton**, with the image pulling successfully inside
  `build-and-test` and `check-npu` passing 1 of 1 in the container. **The first
  green run:**
  <https://github.com/Olajide-Badejo/MLIR-Backend-for-a-Simulated-Edge-NPU/actions/runs/32213397267>.
  Getting there surfaced two real defects the skeleton caught, D-0004 (job
  container run steps fall back to sh) and D-0005 (an installed LLVM has no
  `llvm-lit`), both in `DEFECT_LOG.md` with the red run URLs in the engineering
  log.
- **The v1 line is separately preserved on GitHub** at the owner's request: the
  `v1` branch at `52ed1da`, a Release pinned to the `v1.0.0` tag, and an active
  ruleset blocking deletion and force pushes on `main` and `v1`.

### Remaining

- **The merge of `phase/p0-foundations` into `main`** through pull request 1
  with a merge commit, which is the non fast forward merge the gate asks for.
  It is the act that closes this gate and it is performed immediately after
  the commit that writes this line.

## Open questions

None. Nothing in P0 is blocked on a decision I have not made, and the three
resolutions the phase owed are written as records rather than left in prose.

The one thing worth flagging forward rather than as a question: record 0001
pins LLVM at `llvmorg-22.1.8` and `llvmorg-23.1.0` was scheduled for 25 August
2026. When it lands, it is not an upgrade to take casually. The record lists
four known source breaks, three of which fall on code P1 and P13 are about to
write, and there are no MLIR release notes for either 22 or 23.

## Next command

With the gate met and the merge landing, the next session opens P1 on a fresh
branch cut from the merged `main`:

```bash
cd ~/npu-mlir-v2 && git checkout main && git pull && \
  git checkout -b phase/p1-npu-dialect
```

## Next phase

**P1, the `npu` dialect.** Types, the memory space and layout attributes, every
operation in Section 5.3 except the quantization pair, verifiers per Section 7.2
including the opset 23 pooling arithmetic with its `ceil_mode` right padded
window rule, `InferTypeOpInterface` with the arithmetic shared between inference
and verification, `Pure` traits, and the `npu-dialect-doc` build target. The
destination operand and `TilingInterface` land at P1 and are consumed at P13, so
that an interface bug and a policy bug cannot be mistaken for each other.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, no agent, not once. It may be read, and only through a command
that cannot write. **Only the owner may retire it.**

The reason it exists on top of git history is that the two protect against
different failures. History protects against a bad commit. A second directory
protects against everything else, because if this rebuild goes wrong at any
point, deleting `~/npu-mlir-v2` returns the machine exactly to its pre build
state with no reasoning about reflogs required. That guarantee holds only while
the frozen copy is untouched.
