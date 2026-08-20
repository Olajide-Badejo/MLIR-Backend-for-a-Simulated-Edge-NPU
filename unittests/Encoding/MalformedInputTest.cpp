//===- MalformedInputTest.cpp - the seed and regression corpus *- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 17.3: a corpus of deliberately malformed `.nbin` byte strings, every
// one of which must be rejected cleanly, none of which may crash, hang or read
// out of bounds, and none of which may allocate more than a few megabytes.
//
// **This is a seed and regression corpus, and it is not a fuzzer.** The
// distinction is not pedantry. Its power is capped at what its author imagined
// on the day it was written and it never grows again; worse, because its
// contents are enumerated in the build specification, the cases were written to
// match that prose and therefore exercise exactly the branches already thought
// about. Sanitizers detect bugs, they do not generate inputs. The coverage
// guided target that does generate inputs lives in `fuzz/nbin_decode_fuzzer.cc`
// and is seeded from what this file writes out.
//
// The allocation bound is measured rather than asserted in prose. Global
// `operator new` and `operator delete` are replaced below with counting
// versions, enabled only around the decode call, so the number in the assertion
// is a number somebody measured.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/Program.h"

#include "TestPrograms.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace nbin;
using namespace npu_test;

//===----------------------------------------------------------------------===//
// The allocator hook.
//
// A header in front of every block carries its size, so that `operator delete`
// knows how much to subtract without depending on the compiler emitting the
// sized form. The aligned and array forms are deliberately not replaced: their
// default implementations forward to these two, so every allocation is counted
// and every free is matched.
//===----------------------------------------------------------------------===//

namespace {

constexpr std::size_t kAllocationHeader = alignof(std::max_align_t);

std::size_t gLiveBytes = 0;
std::size_t gPeakBytes = 0;
std::size_t gAllocations = 0;
bool gCounting = false;

} // namespace

void *operator new(std::size_t size) {
  void *raw = std::malloc(size + kAllocationHeader);
  if (!raw) {
    // This library is built with -fno-exceptions, so there is no bad_alloc to
    // throw. Aborting is the honest failure: a test binary that carried on
    // after a failed allocation would be measuring nothing.
    std::fputs("MalformedInputTest: out of memory\n", stderr);
    std::abort();
  }
  std::memcpy(raw, &size, sizeof(size));
  if (gCounting) {
    gLiveBytes += size;
    ++gAllocations;
    if (gLiveBytes > gPeakBytes)
      gPeakBytes = gLiveBytes;
  }
  return static_cast<char *>(raw) + kAllocationHeader;
}

void operator delete(void *pointer) noexcept {
  if (!pointer)
    return;
  char *raw = static_cast<char *>(pointer) - kAllocationHeader;
  std::size_t size = 0;
  std::memcpy(&size, raw, sizeof(size));
  if (gCounting)
    gLiveBytes = gLiveBytes >= size ? gLiveBytes - size : 0;
  std::free(raw);
}

// The sized form. Its default implementation is required to forward to the
// unsized one, so replacing only that one would already be correct; it is
// written out because the compiler warns otherwise, and a warning suppressed by
// a pragma is a warning somebody has to look up.
void operator delete(void *pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

namespace {

/// A few megabytes, which is what Section 17.3 permits. The base program below
/// encodes to a few hundred bytes, so a malformed mutation of it that reached
/// this figure would be a length prefix that got believed.
constexpr std::size_t kAllocationBudget = 4u * 1024u * 1024u;

/// One case of the corpus.
///
/// `mustBeRejected` separates the two kinds of case in here, and the
/// distinction is real rather than a hedge.
///
/// Most cases are **deliberately malformed**: they break a rule the format
/// states, and nothing may accept them. The bit flips and the random single
/// byte changes are different. A file whose declared scratchpad size gains a
/// bit is a larger but entirely self consistent program, and calling that
/// malformed would be calling a valid file malformed. What those cases exist to
/// prove is robustness, so what they assert is robustness: no crash, no read out
/// of bounds under a sanitizer, a bounded allocation, and, when the decoder does
/// accept one, that the file it accepted round trips byte for byte rather than
/// being half believed.
struct Case {
  std::string name;
  std::vector<uint8_t> bytes;
  bool mustBeRejected = true;
};

/// Everything the corpus holds, built once.
class Corpus {
public:
  Corpus() { build(); }

  const std::vector<Case> &cases() const { return items; }

private:
  void add(std::string name, std::vector<uint8_t> bytes,
           bool mustBeRejected = true) {
    items.push_back({std::move(name), std::move(bytes), mustBeRejected});
  }

  /// A copy of the base file with `mutate` applied.
  template <typename Mutate>
  void addMutated(const std::string &name, Mutate mutate,
                  bool mustBeRejected = true) {
    std::vector<uint8_t> bytes = base;
    mutate(bytes);
    add(name, std::move(bytes), mustBeRejected);
  }

  /// A copy of the base program with `mutate` applied, then encoded. The
  /// encoder writes whatever the record holds, so this reaches fields whose
  /// byte offsets would otherwise have to be worked out by hand.
  template <typename Mutate>
  void addFromProgram(const std::string &name, Mutate mutate) {
    Program program = chainProgram();
    mutate(program);
    add(name, program.encode());
  }

  void build();

  Program baseProgram = chainProgram();
  std::vector<uint8_t> base = baseProgram.encode();
  std::vector<Case> items;
};

void Corpus::build() {
  // ---- Truncation. -------------------------------------------------------
  //
  // At every count field boundary, which is where a length prefix is about to
  // be believed, and then on a sweep across the whole file so that a boundary
  // the walker does not know about is covered anyway.
  for (const CountField &field : countFields(baseProgram)) {
    add("truncated before " + field.name,
        std::vector<uint8_t>(base.begin(), base.begin() + field.offset));
    add("truncated after " + field.name,
        std::vector<uint8_t>(base.begin(), base.begin() + field.offset + 4));
  }
  for (size_t length = 0; length < base.size(); length += 8)
    add("truncated at " + std::to_string(length),
        std::vector<uint8_t>(base.begin(), base.begin() + length));

  // ---- The count cap, at the three boundary values, for every count field. -
  //
  // Section 9.2 rule 3 asks for exactly this. Two of the three are rejected by
  // `structure` rather than `count-cap`, because a count of (1 << 28) - 1 is
  // within the cap and still cannot fit in a file this size; that is the point
  // of having both tests and doing them in that order.
  for (const CountField &field : countFields(baseProgram)) {
    for (auto [value, label] :
         {std::pair<uint32_t, const char *>{(1u << 28) - 1, "below the cap"},
          std::pair<uint32_t, const char *>{1u << 28, "at the cap"},
          std::pair<uint32_t, const char *>{(1u << 28) + 1, "above the cap"}}) {
      addMutated(field.name + " " + label, [&](std::vector<uint8_t> &bytes) {
        writeU32(bytes, field.offset, value);
      });
    }
  }

  // ---- Versions the decoder does not accept. ------------------------------
  for (uint32_t version : {0u, 2u, 99u, 0xFFFFFFFFu})
    addMutated("version " + std::to_string(version),
               [&](std::vector<uint8_t> &bytes) {
                 writeU32(bytes, 4, version);
               });

  // ---- A wrong magic, and a file that is nothing at all. ------------------
  addMutated("wrong magic", [](std::vector<uint8_t> &bytes) {
    writeU32(bytes, 0, 0x4E49424Fu);
  });
  add("empty", {});
  add("one byte", {0x4E});
  add("magic only", std::vector<uint8_t>(base.begin(), base.begin() + 4));
  addMutated("trailing byte", [](std::vector<uint8_t> &bytes) {
    bytes.push_back(0);
  });
  addMutated("many trailing bytes", [](std::vector<uint8_t> &bytes) {
    bytes.insert(bytes.end(), 64, 0xFF);
  });

  // ---- Out of range opcodes. ----------------------------------------------
  for (uint32_t opcode : {kMaxOpcode + 1, 100u, 0x80000000u, 0xFFFFFFFFu})
    addFromProgram("opcode " + std::to_string(opcode),
                   [&](Program &program) {
                     program.instructions[1].opcode =
                         static_cast<Opcode>(opcode);
                   });

  // ---- Addresses. ---------------------------------------------------------
  addFromProgram("negative result address", [](Program &program) {
    program.instructions[1].resultAddress = -1;
  });
  addFromProgram("result address at the scratchpad size", [](Program &program) {
    program.instructions[1].resultAddress =
        static_cast<int64_t>(program.scratchpadBytes);
  });
  addFromProgram("result address beyond the scratchpad", [](Program &program) {
    program.instructions[1].resultAddress = 1 << 30;
  });
  addFromProgram("result address at the signed limit", [](Program &program) {
    program.instructions[1].resultAddress = INT64_MAX;
  });
  addFromProgram("negative operand address", [](Program &program) {
    program.instructions[1].operands.front().address = -8;
  });
  addFromProgram("operand address beyond the scratchpad", [](Program &program) {
    program.instructions[1].operands.front().address = 1 << 30;
  });
  addFromProgram("negative dram operand address", [](Program &program) {
    program.instructions[0].operands.front().address = -64;
  });
  addFromProgram("dram result beyond the dram size", [](Program &program) {
    program.instructions[2].resultAddress = 1 << 30;
  });
  addFromProgram("region offset at the signed limit", [](Program &program) {
    program.inputs.front().offset = static_cast<uint64_t>(INT64_MAX);
  });
  addFromProgram("region offset at the unsigned limit", [](Program &program) {
    program.inputs.front().offset = ~uint64_t{0};
  });
  addFromProgram("region beyond the dram size", [](Program &program) {
    program.inputs.front().offset = program.dramBytes;
  });

  // ---- Operand count mismatches, one per opcode that has an arity. --------
  addFromProgram("relu with no operand", [](Program &program) {
    program.instructions[1].operands.clear();
  });
  addFromProgram("relu with two operands", [](Program &program) {
    program.instructions[1].operands.push_back(
        program.instructions[1].operands.front());
  });
  addFromProgram("halt with an operand", [](Program &program) {
    program.instructions[3].operands.push_back(
        operand(MemSpace::Scratchpad, ElemType::F32, 0, {4, 4}));
  });
  addFromProgram("dma_load with two operands", [](Program &program) {
    program.instructions[0].operands.push_back(
        program.instructions[0].operands.front());
  });
  // Every other opcode put where a relu was, with the relu's one operand and
  // no fields. Each fails something different: an arity, an attribute size, a
  // permutation, an element type. Which one is not the point; that none of them
  // is quietly accepted is.
  for (Opcode opcode : {Opcode::MATMUL, Opcode::CONV2D, Opcode::ADD,
                        Opcode::MUL, Opcode::POOL_MAX, Opcode::POOL_AVG,
                        Opcode::TRANSPOSE, Opcode::CONCAT, Opcode::QUANT,
                        Opcode::DEQUANT, Opcode::NOP, Opcode::HALT,
                        Opcode::DMA_LOAD, Opcode::DMA_STORE}) {
    addFromProgram(std::string(opcodeName(opcode)) + " in a relu's place",
                   [&](Program &program) {
                     program.instructions[1].opcode = opcode;
                   });
  }
  addFromProgram("reshape that loses elements", [](Program &program) {
    program.instructions[1].opcode = Opcode::RESHAPE;
    program.instructions[1].resultShape = {4};
  });
  addFromProgram("reshape that gains elements", [](Program &program) {
    program.instructions[1].opcode = Opcode::RESHAPE;
    program.instructions[1].resultShape = {4, 4, 2};
  });

  // ---- Shapes. ------------------------------------------------------------
  addFromProgram("zero extent in a result shape", [](Program &program) {
    program.instructions[1].resultShape = {0, 4};
  });
  addFromProgram("negative extent in a result shape", [](Program &program) {
    program.instructions[1].resultShape = {-4, 4};
  });
  addFromProgram("empty result shape", [](Program &program) {
    program.instructions[1].resultShape = {};
  });
  addFromProgram("shape product overflows, large extent first",
                 [](Program &program) {
                   program.instructions[1].resultShape = {int64_t{1} << 40,
                                                          int64_t{1} << 24};
                 });
  addFromProgram("shape product overflows, large extent last",
                 [](Program &program) {
                   program.instructions[1].resultShape = {int64_t{1} << 24,
                                                          int64_t{1} << 40};
                 });
  addFromProgram("shape product wraps a signed multiply",
                 [](Program &program) {
                   program.instructions[1].resultShape = {int64_t{1} << 32,
                                                          int64_t{1} << 32};
                 });
  addFromProgram("oversized result shape vector", [](Program &program) {
    program.instructions[1].resultShape.assign(4096, 2);
  });
  addFromProgram("oversized region shape vector", [](Program &program) {
    program.inputs.front().shape.assign(4096, 2);
  });
  addFromProgram("region shape overflows", [](Program &program) {
    program.inputs.front().shape = {int64_t{1} << 40, int64_t{1} << 24};
  });
  addFromProgram("operand shape and strides disagree", [](Program &program) {
    program.instructions[1].operands.front().strides = {4};
  });
  addFromProgram("negative operand stride", [](Program &program) {
    program.instructions[1].operands.front().strides = {-4, 1};
  });
  addFromProgram("operand stride overflows its extent", [](Program &program) {
    program.instructions[1].operands.front().strides = {int64_t{1} << 40, 1};
  });
  addFromProgram("operand reads a buffer nothing wrote", [](Program &program) {
    // The scratchpad is 128 bytes and only the first 64 have been written when
    // the relu runs, so this address is inside the memory and outside the
    // program.
    program.instructions[1].operands.front().address = 64;
  });
  addFromProgram("operand reads an output region before it is written",
                 [](Program &program) {
                   program.instructions[0].operands.front().address = 64;
                 });
  addFromProgram("operand reads more than was written", [](Program &program) {
    program.scratchpadBytes = 1 << 20;
    program.instructions[1].operands.front().shape = {100, 100};
    program.instructions[1].operands.front().strides = {100, 1};
    program.instructions[1].resultShape = {100, 100};
  });

  // ---- Attributes. --------------------------------------------------------
  addFromProgram("pads on an opcode with none", [](Program &program) {
    program.instructions[1].pads = {1, 1, 1, 1};
  });
  addFromProgram("kernel on an opcode with none", [](Program &program) {
    program.instructions[1].kernel = {2, 2};
  });
  addFromProgram("group on an opcode with none", [](Program &program) {
    program.instructions[1].group = 4;
  });
  addFromProgram("negative group", [](Program &program) {
    program.instructions[1].group = -1;
  });
  addFromProgram("axes on an opcode with none", [](Program &program) {
    program.instructions[1].axes = {0, 1};
  });
  addFromProgram("malformed transpose permutation", [](Program &program) {
    program.instructions[1].opcode = Opcode::TRANSPOSE;
    program.instructions[1].axes = {0, 0};
  });
  addFromProgram("transpose permutation out of range", [](Program &program) {
    program.instructions[1].opcode = Opcode::TRANSPOSE;
    program.instructions[1].axes = {0, 99};
  });
  addFromProgram("transpose permutation too short", [](Program &program) {
    program.instructions[1].opcode = Opcode::TRANSPOSE;
    program.instructions[1].axes = {0};
  });
  addFromProgram("concat with two axes", [](Program &program) {
    program.instructions[1].opcode = Opcode::CONCAT;
    program.instructions[1].axes = {0, 1};
  });
  addFromProgram("concat axis out of range", [](Program &program) {
    program.instructions[1].opcode = Opcode::CONCAT;
    program.instructions[1].axes = {7};
  });
  addFromProgram("concat axis negative", [](Program &program) {
    program.instructions[1].opcode = Opcode::CONCAT;
    program.instructions[1].axes = {-1};
  });
  addFromProgram("concat extents do not sum", [](Program &program) {
    // The scratchpad is widened so that the oversized result is in range: the
    // point of this case is the extents, and a result that also ran off the end
    // of the scratchpad would be rejected by `result-in-range` before the
    // concatenation rule was ever consulted.
    program.scratchpadBytes = 1024;
    program.instructions[1].opcode = Opcode::CONCAT;
    program.instructions[1].axes = {0};
    program.instructions[1].resultShape = {9, 4};
  });
  addFromProgram("concat operands disagree off the axis", [](Program &program) {
    program.scratchpadBytes = 1024;
    program.instructions[1].opcode = Opcode::CONCAT;
    program.instructions[1].axes = {0};
    program.instructions[1].operands.front().shape = {4, 2};
    program.instructions[1].operands.front().strides = {2, 1};
  });
  addFromProgram("unknown activation", [](Program &program) {
    program.instructions[1].activation = static_cast<Activation>(3);
  });
  addFromProgram("activation on an opcode that fuses none",
                 [](Program &program) {
                   program.instructions[1].activation = Activation::Relu;
                 });

  // ---- Element types. -----------------------------------------------------
  for (uint32_t type : {3u, 4u, 255u, 0xFFFFFFFFu}) {
    addFromProgram("undefined result element type " + std::to_string(type),
                   [&](Program &program) {
                     program.instructions[1].resultElementType =
                         static_cast<ElemType>(type);
                   });
    addFromProgram("undefined region element type " + std::to_string(type),
                   [&](Program &program) {
                     program.inputs.front().elementType =
                         static_cast<ElemType>(type);
                   });
  }
  addFromProgram("undefined memory space", [](Program &program) {
    program.instructions[1].operands.front().space =
        static_cast<MemSpace>(9);
  });
  addFromProgram("compute on an integer element type", [](Program &program) {
    program.instructions[1].resultElementType = ElemType::I32;
    program.instructions[1].operands.front().elementType = ElemType::I32;
  });
  addFromProgram("a dma that converts its element type", [](Program &program) {
    program.instructions[0].operands.front().elementType = ElemType::I8;
  });

  // ---- The i8 region whose byte size no longer matches its shape. ---------
  addFromProgram("i8 constant sized as f32", [](Program &program) {
    program.constants.front().region.elementType = ElemType::I8;
  });
  addFromProgram("constant data one byte short", [](Program &program) {
    program.constants.front().data.pop_back();
  });
  addFromProgram("constant data one byte long", [](Program &program) {
    program.constants.front().data.push_back(0);
  });
  addFromProgram("constant with no data", [](Program &program) {
    program.constants.front().data.clear();
  });

  // ---- Quantization. ------------------------------------------------------
  addFromProgram("scale on an opcode that does not quantize",
                 [](Program &program) {
                   program.instructions[1].scale = 1.0f;
                 });
  addFromProgram("zero point on an opcode that does not quantize",
                 [](Program &program) {
                   program.instructions[1].zeroPoint = 3;
                 });
  for (float scale : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    addFromProgram("quant with a bad scale", [&](Program &program) {
      program.instructions[1].opcode = Opcode::QUANT;
      program.instructions[1].resultElementType = ElemType::I8;
      program.instructions[1].scale = scale;
    });
  }
  for (int32_t zeroPoint : {-129, 128, 100000, -100000}) {
    addFromProgram("quant with zero point " + std::to_string(zeroPoint),
                   [&](Program &program) {
                     program.instructions[1].opcode = Opcode::QUANT;
                     program.instructions[1].resultElementType = ElemType::I8;
                     program.instructions[1].scale = 0.5f;
                     program.instructions[1].zeroPoint = zeroPoint;
                   });
  }
  addFromProgram("quant reading the wrong element type", [](Program &program) {
    program.instructions[1].opcode = Opcode::QUANT;
    program.instructions[1].resultElementType = ElemType::I8;
    program.instructions[1].scale = 0.5f;
    program.instructions[1].operands.front().elementType = ElemType::I8;
  });
  addFromProgram("quant with a different operand shape", [](Program &program) {
    program.instructions[1].opcode = Opcode::QUANT;
    program.instructions[1].resultElementType = ElemType::I8;
    program.instructions[1].scale = 0.5f;
    program.instructions[1].resultShape = {16};
  });
  for (int32_t multiplier : {0, -1, -2147483647 - 1}) {
    addFromProgram("requantization multiplier " + std::to_string(multiplier),
                   [&](Program &program) {
                     program.instructions[1].requantMultiplier = multiplier;
                   });
  }
  for (int32_t shift : {-1, 32, 1000, -2147483647 - 1}) {
    addFromProgram("requantization shift " + std::to_string(shift),
                   [&](Program &program) {
                     program.instructions[1].requantShift = shift;
                   });
  }
  addFromProgram("requantization on an opcode that does not requantize",
                 [](Program &program) {
                   program.instructions[1].requantMultiplier = 7;
                 });

  // ---- The debug section. -------------------------------------------------
  addFromProgram("debug program counter past the end", [](Program &program) {
    program.debug.front().pc = 99;
  });
  addFromProgram("debug program counter at the end", [](Program &program) {
    program.debug.front().pc =
        static_cast<uint32_t>(program.instructions.size());
  });
  addFromProgram("debug program counter at the unsigned limit",
                 [](Program &program) {
                   program.debug.front().pc = 0xFFFFFFFFu;
                 });
  addFromProgram("unsorted debug entries", [](Program &program) {
    program.debug.front().pc = 2;
    program.debug.push_back(DebugEntry{0, "earlier"});
  });
  addFromProgram("duplicate debug entries", [](Program &program) {
    program.debug.push_back(DebugEntry{1, "again"});
  });
  addFromProgram("debug name with an embedded NUL", [](Program &program) {
    program.debug.front().name = std::string("re\0lu", 5);
  });
  addFromProgram("debug name with a high byte", [](Program &program) {
    program.debug.front().name = std::string("rel\xC3\xBC");
  });
  addFromProgram("debug name of only NULs", [](Program &program) {
    program.debug.front().name = std::string(8, '\0');
  });
  addFromProgram("debug name past the length cap", [](Program &program) {
    program.debug.front().name =
        std::string(Program::kMaxDebugNameBytes + 1, 'a');
  });

  // A debug count larger than the file, and a name length that overflows the
  // bytes that remain. These are byte level because no Program can express
  // them: the encoder writes what it holds and what it holds is consistent.
  {
    std::vector<CountField> fields = countFields(baseProgram);
    for (const CountField &field : fields) {
      if (field.name == "debug entries")
        addMutated("debug count larger than the file",
                   [&](std::vector<uint8_t> &bytes) {
                     writeU32(bytes, field.offset, 1000000u);
                   });
      if (field.name.find("name length") != std::string::npos) {
        addMutated("debug name length past the end of the file",
                   [&](std::vector<uint8_t> &bytes) {
                     writeU32(bytes, field.offset, 1000000u);
                   });
        addMutated("debug name length just past the end",
                   [&](std::vector<uint8_t> &bytes) {
                     writeU32(bytes, field.offset, 5u);
                   });
      }
    }
  }

  // ---- Bit flips. ---------------------------------------------------------
  //
  // Every bit of the first sixty four bytes, which is the header and the start
  // of the input region: the part of the file whose meaning everything after it
  // depends on.
  for (size_t byte = 0; byte < 64 && byte < base.size(); ++byte) {
    for (int bit = 0; bit < 8; bit += 3) {
      addMutated("bit " + std::to_string(bit) + " of byte " +
                     std::to_string(byte) + " flipped",
                 [&](std::vector<uint8_t> &bytes) {
                   bytes[byte] ^= static_cast<uint8_t>(1u << bit);
                 },
                 /*mustBeRejected=*/false);
    }
  }

  // ---- A valid file with one byte changed at each of a hundred random
  // offsets. The seed is a constant, so a failure reproduces.
  std::mt19937_64 rng(0x6E62696E4D414C46ull);
  std::uniform_int_distribution<size_t> offset(0, base.size() - 1);
  std::uniform_int_distribution<int> value(0, 255);
  for (int index = 0; index < 100; ++index) {
    size_t at = offset(rng);
    auto replacement = static_cast<uint8_t>(value(rng));
    addMutated("random byte " + std::to_string(index) + " at offset " +
                   std::to_string(at),
               [&](std::vector<uint8_t> &bytes) { bytes[at] = replacement; },
               /*mustBeRejected=*/false);
  }
}

const Corpus &corpus() {
  static const Corpus instance;
  return instance;
}

//===----------------------------------------------------------------------===//
// The tests.
//===----------------------------------------------------------------------===//

// The corpus is at least the size Section 17.3 asks for. A corpus that shrank
// because a loop stopped running would otherwise pass every other test here.
TEST(MalformedInput, TheCorpusIsAtLeastThreeHundredCases) {
  EXPECT_GE(corpus().cases().size(), 300u);
  std::cout << "[          ] malformed corpus: " << corpus().cases().size()
            << " cases\n";
}

// Not one of them is a valid file. A case that accidentally decoded would be a
// case that tested the accepting path while claiming to test the rejecting one,
// and the corpus would shrink without anybody noticing.
//
// The base file is checked here too, from the other direction: if the program
// these are all mutations of stopped being valid, every case below would be
// rejected for the wrong reason.
TEST(MalformedInput, TheBaseFileIsValidAndNoDeliberateCaseIs) {
  Program decoded;
  ASSERT_FALSE(Program::decode(chainProgram().encode(), decoded).has_value());

  std::vector<std::string> accepted;
  int perturbationsAccepted = 0;
  for (const Case &item : corpus().cases()) {
    Program program;
    if (Program::decode(item.bytes, program).has_value())
      continue;
    if (item.mustBeRejected) {
      accepted.push_back(item.name);
      continue;
    }
    ++perturbationsAccepted;
    // A perturbation the decoder accepted is a file it believed in full, so it
    // has to be a file it can write back unchanged. Anything else would mean it
    // had believed half of one.
    EXPECT_EQ(item.bytes, program.encode()) << item.name;
  }
  EXPECT_TRUE(accepted.empty())
      << accepted.size() << " malformed cases were accepted, the first being "
      << (accepted.empty() ? std::string() : accepted.front());
  std::cout << "[          ] " << perturbationsAccepted
            << " byte level perturbations produced a valid file and round "
               "tripped\n";
}

// Every case is rejected with a check name from the description, and the
// message begins with it. A bare failure with no name would be exactly the
// "bare null" Section 9.2 forbids.
TEST(MalformedInput, EveryRejectionNamesACheck) {
  std::set<std::string> seen;
  for (const Case &item : corpus().cases()) {
    Program program;
    std::optional<ProgramError> error = Program::decode(item.bytes, program);
    if (item.mustBeRejected) {
      ASSERT_TRUE(error.has_value()) << item.name;
    }
    if (!error)
      continue;
    std::string name = checkName(error->check);
    EXPECT_NE(name, "<unknown check>") << item.name;
    EXPECT_EQ(error->toString().rfind(name + ": ", 0), 0u)
        << item.name << ": " << error->toString();
    seen.insert(name);
  }
  // The names the corpus never reaches are printed rather than asserted away.
  // A byte level corpus cannot be expected to reach every check, and the rule
  // that every check is triggered somewhere is ValidationTest.cpp's, where it
  // is a hard assertion. What this line is for is noticing when a check stops
  // being reachable from a file at all, which is a different and quieter kind
  // of drift.
  std::string unreached;
  for (uint32_t raw = 0; raw < kNumChecks; ++raw) {
    std::string name = checkName(static_cast<Check>(raw));
    if (seen.count(name))
      continue;
    if (!unreached.empty())
      unreached += ", ";
    unreached += name;
  }
  std::cout << "[          ] the corpus reached " << seen.size() << " of "
            << kNumChecks << " check names"
            << (unreached.empty() ? std::string()
                                  : ", never reaching " + unreached)
            << "\n";
}

// The measured half of Section 17.3. The budget is a few megabytes and the
// number is counted, not asserted in prose.
TEST(MalformedInput, NoCaseAllocatesMoreThanAFewMegabytes) {
  std::size_t worstPeak = 0;
  std::string worstCase;

  for (const Case &item : corpus().cases()) {
    // Everything the measurement needs is constructed before counting starts,
    // so the number is the decode's own footprint rather than the harness's.
    Program program;
    gLiveBytes = 0;
    gPeakBytes = 0;
    gAllocations = 0;
    gCounting = true;
    std::optional<ProgramError> error = Program::decode(item.bytes, program);
    gCounting = false;
    (void)error;

    if (gPeakBytes > worstPeak) {
      worstPeak = gPeakBytes;
      worstCase = item.name;
    }
    EXPECT_LT(gPeakBytes, kAllocationBudget)
        << item.name << " allocated " << gPeakBytes << " bytes";
  }

  std::cout << "[          ] worst case allocation: " << worstPeak
            << " bytes, from \"" << worstCase << "\"\n";
}

// The unvalidated path is the one `npu-objdump` uses on a suspect file, so it
// meets the same corpus. It accepts more of it, because it skips the semantic
// half by design; what it must never do is crash or read out of bounds, which
// is what the sanitizer build of this binary checks.
TEST(MalformedInput, TheUnvalidatedPathAndTheDisassemblerSurviveTheCorpus) {
  int framed = 0;
  size_t printed = 0;
  for (const Case &item : corpus().cases()) {
    Program program;
    if (Program::decodeUnvalidated(item.bytes, program).has_value())
      continue;
    ++framed;
    // A file that frames is a file npu-objdump will walk, so walking it here is
    // part of the same claim. This is the path with no validation in front of
    // it, which makes it the one where an out of bounds read would actually
    // happen, and the sanitizer build of this binary is what watches for one.
    printed += disassemble(program, program.validate()).size();
  }
  std::cout << "[          ] " << framed << " of " << corpus().cases().size()
            << " cases frame correctly, disassembling to " << printed
            << " characters\n";
}

// The corpus is written out so that `fuzz/nbin_decode_fuzzer` can be seeded
// from it. The path is an environment variable rather than a fixed directory
// because the only caller is a script that knows where it wants them.
TEST(MalformedInput, TheCorpusCanBeWrittenOutForTheFuzzer) {
  const char *directory = std::getenv("NPU_CORPUS_OUT");
  if (!directory) {
    std::cout << "[          ] NPU_CORPUS_OUT is not set, so the corpus was "
                 "not written out. Set it to a directory to export the seeds.\n";
    GTEST_SKIP();
  }

  size_t written = 0;
  for (size_t index = 0; index < corpus().cases().size(); ++index) {
    std::string path = std::string(directory) + "/case_" +
                       std::to_string(index) + ".nbin";
    FILE *file = std::fopen(path.c_str(), "wb");
    ASSERT_NE(file, nullptr) << path;
    const std::vector<uint8_t> &bytes = corpus().cases()[index].bytes;
    if (!bytes.empty())
      std::fwrite(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    ++written;
  }
  // The valid file too. A coverage guided run that only ever saw rejected
  // inputs would spend its whole budget in the first hundred bytes.
  std::string path = std::string(directory) + "/valid.nbin";
  FILE *file = std::fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr) << path;
  std::vector<uint8_t> valid = chainProgram().encode();
  std::fwrite(valid.data(), 1, valid.size(), file);
  std::fclose(file);

  std::cout << "[          ] wrote " << written << " seeds to " << directory
            << "\n";
}

} // namespace
