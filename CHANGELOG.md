# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
Semantic Versioning once a release is tagged.

## [Unreleased]

### Added

- Phase 0 scaffold: out of tree CMake project, `npu-opt` driver skeleton, lit test harness
  with a smoke test, repository layout per the specification.
- `scripts/dash-lint.sh` enforcing the no em dash and no en dash rule, wired into
  pre-commit and CI.
- Documentation seed: README, BUILD, and an engineering log started at Phase 0.
- Pinned LLVM/MLIR toolchain at tag `llvmorg-22.1.8`, built once with a memory safe
  configuration for this machine's WSL2 budget.
