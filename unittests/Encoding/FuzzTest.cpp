//===- FuzzTest.cpp - Malformed .nbin corpus ------------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// UPGRADE_SPEC_V3.md section 9.5 asks for at least 200 deliberately malformed
// .nbin byte strings, every one of which must be rejected cleanly with no
// crash, hang, or out of bounds access. This builds the corpus programmatically
// rather than checking in binary blobs, so it stays readable and grows with the
// format.
//
// The property being tested is not "these are all invalid". Some mutations
// land on a byte nothing reads and produce a program that is still perfectly
// valid, and that is a correct outcome. The property is that decode either
// returns a usable program or refuses with a reason, and never does anything
// else. Under the sanitizers job this is also the test that would catch a
// missing bounds check in the decoder itself.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Simulator.h"

#include "gtest/gtest.h"

#include <random>

using namespace npu;

namespace {

Program validProgram() {
  Program p;
  p.scratchpadBytes = 32768;
  p.dramBytes = 8192;
  p.inputs.push_back({0, {1, 1, 28, 28}});
  p.constants.push_back({4096, {6, 1, 5, 5}});
  p.constantData.push_back(std::vector<float>(150, 0.5f));
  p.outputs.push_back({8000, {1, 10}});

  Instruction loadInput;
  loadInput.op = Opcode::DmaLoad;
  loadInput.resultAddr = 0;
  loadInput.resultShape = {1, 1, 28, 28};
  loadInput.dramAddr = 0;
  p.instructions.push_back(loadInput);

  Instruction loadWeight;
  loadWeight.op = Opcode::DmaLoad;
  loadWeight.resultAddr = 3136;
  loadWeight.resultShape = {6, 1, 5, 5};
  loadWeight.dramAddr = 4096;
  p.instructions.push_back(loadWeight);

  Instruction conv;
  conv.op = Opcode::Conv2D;
  conv.resultAddr = 3736;
  conv.resultShape = {1, 6, 24, 24};
  conv.operandAddrs = {0, 3136};
  conv.strides = {1, 1};
  conv.pads = {0, 0, 0, 0};
  conv.dilations = {1, 1};
  p.instructions.push_back(conv);

  p.instructions.push_back(Instruction{Opcode::Halt});
  return p;
}

struct Case {
  std::string name;
  std::vector<uint8_t> bytes;
};

// Write a little endian value over an existing field.
template <typename T> void poke(std::vector<uint8_t> &bytes, size_t at, T v) {
  if (at + sizeof(T) > bytes.size())
    return;
  std::memcpy(bytes.data() + at, &v, sizeof(T));
}

// A hand built byte stream. Everything else in this file mutates a valid file;
// these cases need the opposite, a file that is tiny and claims to be huge.
struct Blob {
  std::vector<uint8_t> bytes;
  template <typename T> Blob &put(T v) {
    auto *p = reinterpret_cast<const uint8_t *>(&v);
    bytes.insert(bytes.end(), p, p + sizeof(T));
    return *this;
  }
  Blob &header() {
    bytes.insert(bytes.end(), {'N', 'P', 'U', 'B'});
    put<uint32_t>(Program::kVersion);
    put<int64_t>(32768);
    put<int64_t>(8192);
    return *this;
  }
};

// Every count field in the format, probed just under, exactly at, and just over
// the decoder's 2^28 cap. The cap alone is not a bound on work: 2^28 int64
// elements is a 2 GiB vector, and getVec sizes that vector from the count
// before reading a single element behind it. So a file of a few dozen bytes
// naming a count near the cap is a decompression bomb, and the just under and
// exactly at values are the ones the cap lets through. These are separate from
// buildCorpus so the dedicated test can iterate exactly them.
std::vector<Case> buildNearCapCases() {
  std::vector<Case> cases;
  for (uint32_t n : {(1u << 28) - 1, 1u << 28, (1u << 28) + 1}) {
    const std::string tag = "-" + std::to_string(n);

    // The four container counts, in the order decodeUnvalidated reads them.
    cases.push_back({"cap-input-count" + tag,
                     Blob().header().put<uint32_t>(n).bytes});
    cases.push_back({"cap-output-count" + tag,
                     Blob().header().put<uint32_t>(0).put<uint32_t>(n).bytes});
    cases.push_back({"cap-constant-count" + tag,
                     Blob().header().put<uint32_t>(0).put<uint32_t>(0)
                         .put<uint32_t>(n).bytes});
    cases.push_back({"cap-instruction-count" + tag,
                     Blob().header().put<uint32_t>(0).put<uint32_t>(0)
                         .put<uint32_t>(0).put<uint32_t>(n).bytes});

    // The shape vector inside a region, which is a getVec of int64.
    cases.push_back({"cap-region-shape-count" + tag,
                     Blob().header().put<uint32_t>(1).put<int64_t>(0)
                         .put<uint32_t>(n).bytes});

    // Constant data, which is a vector<float> sized the same way.
    cases.push_back({"cap-constant-data-count" + tag,
                     Blob().header().put<uint32_t>(0).put<uint32_t>(0)
                         .put<uint32_t>(1)                            // 1 const
                         .put<int64_t>(0).put<uint32_t>(1).put<int64_t>(4)
                         .put<uint32_t>(n).bytes});

    // Each of the six vectors inside an instruction. readInstr takes them as
    // resultShape, operandAddrs, then dramAddr / activation / group, then
    // strides, pads, dilations, kernelShape.
    for (int target = 0; target < 6; ++target) {
      Blob b;
      b.header()
          .put<uint32_t>(0)
          .put<uint32_t>(0)
          .put<uint32_t>(0)
          .put<uint32_t>(1)  // one instruction
          .put<uint16_t>(0)  // NOP
          .put<int64_t>(0);  // resultAddr
      auto scalars = [&] {
        b.put<int64_t>(0).put<int32_t>(0).put<int64_t>(1);
      };
      for (int k = 0; k < target; ++k) {
        if (k == 2)
          scalars();
        b.put<uint32_t>(0); // an empty vector, to reach the one being probed
      }
      if (target == 2)
        scalars();
      b.put<uint32_t>(n);
      cases.push_back(
          {"cap-instruction-vec-" + std::to_string(target) + tag, b.bytes});
    }
  }
  return cases;
}

std::vector<Case> buildCorpus() {
  const std::vector<uint8_t> good = validProgram().encode();
  std::vector<Case> corpus;

  // 1. Empty and tiny inputs, the classic off by one territory.
  corpus.push_back({"empty", {}});
  for (size_t n = 1; n <= 16 && n < good.size(); ++n)
    corpus.push_back({"prefix-" + std::to_string(n),
                      std::vector<uint8_t>(good.begin(), good.begin() + n)});

  // 2. Truncation at every 16 byte boundary, which walks every section header
  //    and most field boundaries in a file this size.
  for (size_t n = 16; n < good.size(); n += 16)
    corpus.push_back({"truncated-at-" + std::to_string(n),
                      std::vector<uint8_t>(good.begin(), good.begin() + n)});

  // 3. Bad magic, in each of the four bytes.
  for (size_t i = 0; i < 4; ++i) {
    auto bytes = good;
    bytes[i] ^= 0xFF;
    corpus.push_back({"magic-byte-" + std::to_string(i), bytes});
  }

  // 4. Version field, at offset 4, just past the magic.
  for (uint32_t v : {0u, 2u, 99u, 0xFFFFFFFFu}) {
    auto bytes = good;
    poke<uint32_t>(bytes, 4, v);
    corpus.push_back({"version-" + std::to_string(v), bytes});
  }

  // 5. Header sizes: scratchpad at offset 8, DRAM at 16.
  for (int64_t v : {int64_t{-1}, int64_t{-4096},
                    std::numeric_limits<int64_t>::min(),
                    std::numeric_limits<int64_t>::max()}) {
    auto sp = good;
    poke<int64_t>(sp, 8, v);
    corpus.push_back({"scratchpad-size", sp});
    auto dram = good;
    poke<int64_t>(dram, 16, v);
    corpus.push_back({"dram-size", dram});
  }

  // 6. Absurd counts. A corrupt length is the classic way to turn a parser into
  //    an allocator, so these must be refused rather than reserved.
  for (uint32_t count : {0xFFFFFFFFu, 0x7FFFFFFFu, 1u << 29}) {
    auto bytes = good;
    poke<uint32_t>(bytes, 24, count); // input region count
    corpus.push_back({"input-count-" + std::to_string(count), bytes});
  }

  // 7. Every byte of the header region flipped one at a time. This is where the
  //    opcode, address, and shape fields of the first instructions live.
  for (size_t i = 0; i < good.size() && i < 96; ++i) {
    auto bytes = good;
    bytes[i] ^= 0xFF;
    corpus.push_back({"flip-" + std::to_string(i), bytes});
  }

  // 8. A valid file with one byte changed at each of 100 pseudo random offsets,
  //    as the spec asks for. Fixed seed so a failure reproduces.
  std::mt19937 rng(0xF0FF1234);
  std::uniform_int_distribution<size_t> offset(0, good.size() - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  for (int i = 0; i < 100; ++i) {
    auto bytes = good;
    size_t at = offset(rng);
    bytes[at] = static_cast<uint8_t>(byte(rng));
    corpus.push_back({"random-" + std::to_string(i) + "-at-" +
                          std::to_string(at),
                      bytes});
  }

  // 9. Semantically broken programs, encoded properly. These exercise validate
  //    rather than the byte reader.
  auto encodeWith = [&](const char *name, void (*mutate)(Program &)) {
    Program p = validProgram();
    mutate(p);
    corpus.push_back({name, p.encode()});
  };
  encodeWith("opcode-out-of-range", [](Program &p) {
    p.instructions[2].op = static_cast<Opcode>(60000);
  });
  encodeWith("conv-one-operand",
             [](Program &p) { p.instructions[2].operandAddrs = {0}; });
  encodeWith("conv-no-operands",
             [](Program &p) { p.instructions[2].operandAddrs.clear(); });
  encodeWith("negative-result-address",
             [](Program &p) { p.instructions[2].resultAddr = -8; });
  encodeWith("result-past-scratchpad",
             [](Program &p) { p.instructions[2].resultAddr = 1 << 28; });
  encodeWith("operand-past-scratchpad",
             [](Program &p) { p.instructions[2].operandAddrs[0] = 1 << 28; });
  encodeWith("operand-never-written",
             [](Program &p) { p.instructions[2].operandAddrs[1] = 24000; });
  encodeWith("dma-past-dram",
             [](Program &p) { p.instructions[0].dramAddr = 8100; });
  encodeWith("negative-dma-address",
             [](Program &p) { p.instructions[0].dramAddr = -32; });
  encodeWith("empty-shape",
             [](Program &p) { p.instructions[2].resultShape.clear(); });
  encodeWith("zero-extent",
             [](Program &p) { p.instructions[2].resultShape = {1, 0, 24, 24}; });
  encodeWith("negative-extent", [](Program &p) {
    p.instructions[2].resultShape = {1, -6, 24, 24};
  });
  encodeWith("oversized-shape", [](Program &p) {
    p.instructions[2].resultShape = {1LL << 40, 1LL << 40, 1LL << 40};
  });
  encodeWith("zero-stride",
             [](Program &p) { p.instructions[2].strides = {0, 0}; });
  encodeWith("zero-group", [](Program &p) { p.instructions[2].group = 0; });
  encodeWith("negative-group", [](Program &p) { p.instructions[2].group = -4; });
  encodeWith("short-pads", [](Program &p) { p.instructions[2].pads = {0}; });
  encodeWith("constant-data-mismatch",
             [](Program &p) { p.constantData[0].resize(3); });
  encodeWith("region-past-dram",
             [](Program &p) { p.outputs[0].dramOffset = 8190; });

  // 10. Every count field probed from both sides of the decoder's cap.
  for (Case &c : buildNearCapCases())
    corpus.push_back(std::move(c));

  return corpus;
}

} // namespace

TEST(Fuzz, CorpusIsLargeEnough) {
  // The spec asks for at least 200 cases. Assert it, so trimming the generator
  // is a visible decision rather than a quiet one.
  const size_t size = buildCorpus().size();
  EXPECT_GE(size, 200u);
  // Pinned as well as floored, so a change in the generator has to be justified
  // by a number rather than silently absorbed. It was 322 before the 36 near
  // cap probes were added.
  EXPECT_EQ(size, 358u);
}

TEST(Fuzz, NearTheCountCapNothingOverAllocates) {
  // Each of these files is a few dozen bytes and names a count near 2^28. The
  // assertion that matters is not visible from here, which is that decoding
  // them costs a few kilobytes rather than a few gigabytes; that is measured by
  // running this test under a peak RSS check. What is asserted here is that the
  // file really is tiny and that every one of them is refused with a reason, so
  // the cost claim is about refusal and not about a successful giant decode.
  std::vector<Case> cases = buildNearCapCases();
  EXPECT_EQ(cases.size(), 36u);
  for (const Case &c : cases) {
    EXPECT_LT(c.bytes.size(), 128u) << c.name;

    ValidationError error;
    EXPECT_FALSE(Program::decode(c.bytes, &error).has_value()) << c.name;
    EXPECT_FALSE(error.check.empty()) << c.name;
    EXPECT_FALSE(error.detail.empty()) << c.name;

    // npu-objdump takes the permissive path, so it has to refuse these too.
    EXPECT_FALSE(Program::decodeUnvalidated(c.bytes).has_value()) << c.name;
  }
}

TEST(Fuzz, EveryMalformedInputIsHandledCleanly) {
  int rejected = 0;
  int accepted = 0;
  for (const Case &c : buildCorpus()) {
    ValidationError error;
    std::optional<Program> program = Program::decode(c.bytes, &error);

    if (!program) {
      // A refusal has to say something useful. "malformed .nbin" was the old
      // behaviour and it is what this phase exists to replace.
      EXPECT_FALSE(error.check.empty()) << c.name;
      EXPECT_FALSE(error.detail.empty()) << c.name;
      ++rejected;
      continue;
    }

    // Surviving decode means it validated, so it must also be runnable without
    // tripping the simulator's bounds checks. A mutation that lands on an
    // unread byte legitimately gets here.
    EXPECT_FALSE(program->validate().has_value()) << c.name;
    ++accepted;
  }

  // Both outcomes should occur. If nothing were ever accepted the corpus would
  // be testing a decoder that rejects everything, which passes trivially.
  EXPECT_GT(rejected, 150) << "the corpus is not exercising rejection";
  EXPECT_GT(accepted, 0) << "no mutation landed on an unread byte, which is "
                            "suspicious for a corpus this size";
}

TEST(Fuzz, NothingThatDecodesCanCorruptMemoryWhenRun) {
  // The decoder is only half of it. Anything that gets past validate is handed
  // to the simulator, which does raw pointer arithmetic. Run every survivor.
  // Under the sanitizers job this is where an out of bounds access shows up.
  int ran = 0;
  for (const Case &c : buildCorpus()) {
    std::optional<Program> program = Program::decode(c.bytes);
    if (!program)
      continue;
    Simulator sim(*program);
    SimResult result = sim.run({std::vector<float>(784, 0.25f)});
    // A validated program must not trip the simulator's own bounds checks.
    EXPECT_TRUE(result.error.empty()) << c.name << ": " << result.error;
    ++ran;
  }
  EXPECT_GT(ran, 0);
}

TEST(Fuzz, DecodeUnvalidatedNeverCrashesEither) {
  // npu-objdump takes this path deliberately, so it gets the same corpus.
  for (const Case &c : buildCorpus()) {
    std::optional<Program> program = Program::decodeUnvalidated(c.bytes);
    if (!program)
      continue;
    // Touch every field the disassembler would, without asserting anything
    // about the values. This is a crash and sanitizer test, not a value test.
    volatile int64_t sink = program->scratchpadBytes + program->dramBytes;
    for (const Instruction &in : program->instructions)
      sink += in.resultAddr + in.dramAddr + in.group +
              static_cast<int64_t>(in.resultShape.size()) +
              static_cast<int64_t>(in.operandAddrs.size());
    for (const MemRegion &r : program->constants)
      sink += r.byteSize();
    (void)sink;
  }
}
