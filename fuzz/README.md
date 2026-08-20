<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Fuzzing the binary format

*Diataxis type: how to.*

`nbin_decode_fuzzer` is the coverage guided target of Section 17.3. It decodes a
byte string, validates it, and asserts two properties beyond crash freedom: a
file it accepts must re-encode to exactly the bytes it came from, and a file it
frames must survive the disassembler.

**The seed corpus in `corpus/` is not the same thing as
`unittests/Encoding/MalformedInputTest.cpp`, and neither replaces the other.**
That file is a negative test table: seven hundred and thirty three cases whose
power is capped at what its author imagined, run on every push, fast. This is a
generator that does not know what the author imagined. The corpus is the bridge
between them, and the test can export its whole contents as extra seeds.

## Building it

`-fsanitize=fuzzer` is a clang feature, so this needs a separate build
directory with a clang toolchain. The address and undefined sanitizers go on the
whole build rather than on the target alone, because a fuzz target instrumented
against a library that is not tells you only about the target.

```
cmake -G Ninja -S . -B build-fuzz \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DNPU_ENABLE_FUZZERS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
ninja -C build-fuzz nbin_decode_fuzzer
```

The default `build/` directory is gcc and prints `Fuzzers: OFF` at configure
time, which is deliberate: a step that is off says so.

## Running it

```
./build-fuzz/bin/nbin_decode_fuzzer -max_total_time=60 build-fuzz/fuzz-corpus
```

The corpus is copied into the build directory by the `nbin-fuzz-corpus` target
rather than used in place, so a run cannot write new inputs into the source
tree. A corpus that grows during a run is a corpus somebody commits by accident.

A longer run, which is what `.github/workflows/nightly.yml` does:

```
./build-fuzz/bin/nbin_decode_fuzzer -max_total_time=600 build-fuzz/fuzz-corpus
```

## Seeding it from the whole hand written corpus

`MalformedInputTest` writes its cases out when `NPU_CORPUS_OUT` names a
directory. Seven hundred more starting points cost nothing and save the mutator
the work of rediscovering a truncated section.

```
mkdir -p /tmp/nbin-seeds
NPU_CORPUS_OUT=/tmp/nbin-seeds ./build/bin/NPUEncodingTests \
  --gtest_filter='MalformedInput.TheCorpusCanBeWrittenOutForTheFuzzer'
./build-fuzz/bin/nbin_decode_fuzzer -max_total_time=600 \
  build-fuzz/fuzz-corpus /tmp/nbin-seeds
```

## When it finds something

Minimize it, then commit the minimized input as a regression case:

```
./build-fuzz/bin/nbin_decode_fuzzer -minimize_crash=1 -runs=100000 crash-<hash>
cp minimized-from-<hash> fuzz/corpus/regression_<what-it-was>.nbin
```

Add the same bytes to `MalformedInputTest.cpp` as a named case, so that the
thing the generator found becomes something the fast suite checks on every push.

## Regenerating the seed corpus

Every seed is produced by the tools rather than typed, so the recipe is the
definition of the corpus:

| Seed | How |
|---|---|
| `all_ops.nbin` | `npu-translate test/Encoding/all_ops.mlir` |
| `chain.nbin` | the `objdump.mlir` pipeline with `--mlir-print-debuginfo` |
| `chain_stripped.nbin` | the same, with `--strip-debug` |
| `header_only.nbin` | `head -c 24 chain.nbin` |
| `bad_version.nbin` | `chain.nbin` with the version word set to 99 |
| `zero_extent.nbin` | `chain.nbin` with eight zero bytes at offset 44 |
| `bad_magic.nbin` | `chain.nbin` with the magic word zeroed |
| `empty.nbin` | a zero byte file |

Three of them frame and validate, three frame and fail a semantic check, and two
are refused at the framing. Starting a run from all eight means the mutator does
not spend its first minutes rediscovering the magic word.
