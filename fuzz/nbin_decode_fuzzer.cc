//===- nbin_decode_fuzzer.cc - the coverage guided target -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 17.3's real fuzz target: `LLVMFuzzerTestOneInput` calling decode and
// then validate, built with `-fsanitize=fuzzer,address,undefined` and seeded
// from `fuzz/corpus/`.
//
// **Why this exists when there is already a corpus of seven hundred hand
// written malformed files.** That corpus is a negative test table. Its power is
// capped at what its author imagined on the day it was written and it never
// grows again; worse, because its contents are enumerated in the build
// specification, the cases were written to match that prose and therefore
// exercise exactly the branches already thought about. This target does not
// know what the author imagined. Coverage guided fuzzing of upstream MLIR found
// sixty three previously unknown bugs in a codebase already covered by
// thousands of hand written lit tests, which is the evidence that hand written
// tests saturate and generators do not.
//
// **It asserts more than crash freedom.** Crash freedom plus a sanitizer is the
// regime in which a silently wrong answer hides, so the two oracles below are
// properties rather than the absence of a signal:
//
//   1. A file the decoder **accepts** must re-encode to exactly the bytes it
//      came from. A decoder that believed half a file, or that normalised a
//      field on the way in, would be caught here and by nothing else: the file
//      would decode, validate, and quietly not be the file it claimed to be.
//   2. A file the decoder **frames** must survive the disassembler. That is the
//      path `npu-objdump` takes on a suspect file, with no validation in front
//      of it, which makes it the one where an out of bounds read would actually
//      happen.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/Program.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

/// Reports a broken property the way libFuzzer expects: print, then abort, so
/// the input is minimized and written out as a crash file.
[[noreturn]] void fail(const char *what, const uint8_t *data, size_t size) {
  std::fprintf(stderr, "nbin_decode_fuzzer: %s (input is %zu bytes)\n", what,
               size);
  for (size_t index = 0; index < size && index < 64; ++index)
    std::fprintf(stderr, "%02x", data[index]);
  std::fprintf(stderr, "\n");
  std::abort();
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  llvm::ArrayRef<uint8_t> bytes(data, size);

  // The validating path, which is what everything except npu-objdump uses.
  nbin::Program program;
  std::optional<nbin::ProgramError> error =
      nbin::Program::decode(bytes, program);

  if (!error) {
    // Property 1. This file was accepted in full, so writing it back has to
    // reproduce it exactly.
    std::vector<uint8_t> reEncoded = program.encode();
    if (reEncoded.size() != size ||
        (size != 0 && std::memcmp(reEncoded.data(), data, size) != 0))
      fail("a file that decoded and validated did not re-encode to itself",
           data, size);
  } else {
    // Every failure carries a name from the ISA description. A bare failure
    // would be the "bare null" Section 9.2 forbids, and a name that came back
    // as the unknown placeholder would mean the enum and the table disagreed.
    std::string message = error->toString();
    const char *name = nbin::checkName(error->check);
    if (message.rfind(std::string(name) + ": ", 0) != 0)
      fail("a rejection did not begin with its check name", data, size);
  }

  // Property 2. The unvalidated path, and the disassembler over whatever it
  // produced. A file that frames is a file npu-objdump will walk.
  nbin::Program suspect;
  if (!nbin::Program::decodeUnvalidated(bytes, suspect)) {
    std::string listing = nbin::disassemble(suspect, suspect.validate());
    // Consumed so that no compiler decides the whole call was dead.
    if (listing.empty())
      fail("a framed file disassembled to nothing at all", data, size);
  }

  return 0;
}
