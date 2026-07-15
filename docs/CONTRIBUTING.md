# Contributing

## Building and testing

Build LLVM/MLIR once at the pinned tag, then build this project against it. Full
steps, including the memory budget this machine builds under, are in
[BUILD.md](BUILD.md).

```bash
cd ~/npu-mlir
cmake -G Ninja -S . -B build \
  -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
  -DLLVM_USE_LINKER=lld
ninja -C build

ninja -C build check-npu            # lit and FileCheck
./build/bin/NPUEncodingTests        # GoogleTest: encoding
./build/bin/NPUSimulatorTests       # GoogleTest: simulator and cost model
python -m pytest test/Python        # importer, driver, end to end vs onnxruntime
```

## Style

- C++ in LLVM style, formatted with the inherited `.clang-format`. TableGen
  formatted consistently.
- Python is PEP 8 via `black` and `ruff`, configured in `pyproject.toml`.
- Commits follow Conventional Commits, one logical change each.
- Report and comment prose is first person and concrete, with no AI stock phrasing.

## The dash rule

No em dashes (U+2014) and no en dashes (U+2013) anywhere: code, comments, commits,
markdown, or LaTeX. `scripts/dash-lint.sh` enforces this and runs in pre-commit and
CI. Run it before pushing:

```bash
bash scripts/dash-lint.sh
```

## Adding an op

1. Define it in the dialect's `*Ops.td`, with a verifier if it has shape or type
   constraints, and `Pure` if it computes a pure function of its operands.
2. Add a round trip test and, if it can fail verification, a `-verify-diagnostics`
   case under `test/Dialect/`.
3. If it should lower, add a conversion pattern in `LowerNPUToNPUISA.cpp` and the
   matching `npuisa` instruction, encoder case, and simulator kernel.
4. Regenerate the dialect reference: `ninja -C build npu-dialect-doc`.

## Pull requests

Keep the test pyramid green (lit, GoogleTest, pytest), keep coverage at or above
the target on pass implementations, and update the CHANGELOG under Unreleased.
