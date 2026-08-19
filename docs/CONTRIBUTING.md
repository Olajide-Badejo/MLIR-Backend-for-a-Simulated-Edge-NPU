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

### The end to end matrix and the `slow` marker

`test/Python/test_end_to_end.py` is a cross product: every model, times three
optimization levels, times both scratchpad budgets, times five input classes
(`normal`, `zeros`, `large_pos`, `large_neg`, `relu_knee`). Every cell compiles
and simulates the model and compares against onnxruntime.

The default run deselects most of it. `pyproject.toml` sets
`addopts = "-m 'not slow'"`, and every cell outside the fast subset carries the
`slow` marker, so `python -m pytest test/Python` runs one cell per optimization
level: enough to catch an obvious break in an edit and rerun loop, in about a
second.

Run the whole matrix before you push, and CI runs it on every commit:

```bash
python -m pytest test/Python -q -m "slow or not slow"
```

Both bounds are asserted separately rather than through
`np.testing.assert_allclose`, whose combined criterion lets an absolute
allowance hide a relative failure. The absolute bound is scale aware, expressed
as a number of float32 ulps at the magnitude of the output with `ATOL` as a
floor, because the `large_pos` and `large_neg` classes produce outputs two
orders of magnitude larger than `normal` and a fixed absolute bound calibrated on
one is unsatisfiable on the other. Do not widen either bound to make a cell pass;
if a cell needs a wider bound that is a finding, and the measured value belongs
in `docs/ENGINEERING_LOG.md`.

## Style

- C++ in LLVM style, formatted with the inherited `.clang-format`. TableGen
  formatted consistently.
- Python is PEP 8 via `black` and `ruff`, configured in `pyproject.toml`.
- Commits follow Conventional Commits, one logical change each.
- Report and comment prose is first person and concrete, with no filler or stock phrasing.

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
