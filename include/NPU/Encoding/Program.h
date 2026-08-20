//===- Program.h - the .nbin binary format --------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The binary format of Section 9: a fixed header followed by fixed order
// sections, each length prefixed by a `u32` count.
//
// **There are no tags.** Nothing in this format may be skipped, no unknown
// field may be tolerated, and no documentation anywhere may claim otherwise. A
// decoder that met a field it did not understand would have no way to know how
// long it was, so an unknown version is rejected by name rather than
// reinterpreted.
//
// **The byte order is host order.** A `.nbin` is not portable across byte
// orders and this file says so plainly rather than claiming little endian.
// Every integer is written by copying its object representation, so a file
// written on a big endian machine and read on a little endian one is garbage
// that the validator will reject for the wrong reason. The project targets one
// machine class and pays for that with one sentence here, in
// `docs/ISA_MANUAL.md`, and nowhere else.
//
// The opcode enum, `kMaxOpcode`, and the per opcode arity, field presence,
// memory space and element type rules are **generated** from
// `include/NPU/Encoding/NPUISADescription.td`. So are the validation check
// names. That is Section 9.4: describe the instruction set once, and let the
// layers that would otherwise drift be produced rather than maintained.
//
// **The namespace is `nbin`, not `npu`, and that is deliberate.** The tensor
// level dialect already owns `mlir::npu`, so a second `::npu` would be
// ambiguous in every translation unit that said `using namespace mlir;` and
// then named either one: `npu::ScratchpadAttr` and `npu::Program` would both
// be reachable through two different paths. Naming this one after the artifact
// it describes costs four characters and removes a class of error that would
// otherwise arrive again at every phase that touches both levels.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_PROGRAM_H
#define NPU_ENCODING_PROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nbin {

//===----------------------------------------------------------------------===//
// The generated instruction set.
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/NPUISAOpcodes.inc"
#include "NPU/Encoding/NPUISAChecks.inc"

// The order matters: the opcode table's accessor takes an `Opcode`, so the
// enum has to exist first. It is included after the vocabulary below in the
// file layout sense only; the values it stores for memory spaces and element
// types are plain integers, so it has no dependency on those enums at all.
#include "NPU/Encoding/NPUISAOpcodeInfo.inc"

//===----------------------------------------------------------------------===//
// The hand written vocabulary that the generated tables refer to by value.
//===----------------------------------------------------------------------===//

/// The element types the format carries, from the **first** version.
///
/// Section 9.1 pins this: `F32 = 0, I8 = 1, I32 = 2` are present from version
/// one, together with `requantMultiplier` and `requantShift`, and those
/// specific fields and nothing broader are what let Phase P14 land without
/// bumping `kVersion`. A bump would invalidate every seed in the fuzz corpus
/// and the binary stability test in the same commit that introduced
/// quantization, which is exactly the migration this enum exists to avoid.
enum class ElemType : uint32_t {
  F32 = 0,
  I8 = 1,
  I32 = 2,
};

/// The two memories. The compute units address the scratchpad and nothing
/// else; DRAM is reachable only through `DMA_LOAD` and `DMA_STORE`.
enum class MemSpace : uint32_t {
  Scratchpad = 0,
  Dram = 1,
};

/// A fused activation on the result of a compute instruction.
///
/// The field exists from version one even though no pass fuses an activation
/// yet, for the same reason the requantization pair does: adding it later
/// would move the layout and bump the version.
enum class Activation : uint32_t {
  None = 0,
  Relu = 1,
};

/// The bytes one element of a type occupies. `MemRegion::byteSize` multiplies
/// by this rather than assuming four bytes per element, which is the whole
/// reason the function exists.
constexpr int64_t elementByteSize(ElemType type) {
  switch (type) {
  case ElemType::F32:
    return 4;
  case ElemType::I8:
    return 1;
  case ElemType::I32:
    return 4;
  }
  return 0;
}

/// Whether a raw word from a file names an element type this build knows.
constexpr bool isKnownElemType(uint32_t raw) { return raw <= 2; }

/// Whether a raw word from a file names a memory space this build knows.
constexpr bool isKnownMemSpace(uint32_t raw) { return raw <= 1; }

/// Whether a raw word from a file names an activation this build knows.
constexpr bool isKnownActivation(uint32_t raw) { return raw <= 1; }

/// The spelling of an element type, for the disassembler and for messages.
constexpr const char *elemTypeName(ElemType type) {
  switch (type) {
  case ElemType::F32:
    return "f32";
  case ElemType::I8:
    return "i8";
  case ElemType::I32:
    return "i32";
  }
  return "<unknown element type>";
}

/// The spelling of a memory space.
constexpr const char *memSpaceName(MemSpace space) {
  switch (space) {
  case MemSpace::Scratchpad:
    return "scratchpad";
  case MemSpace::Dram:
    return "dram";
  }
  return "<unknown memory space>";
}

/// The spelling of an activation.
constexpr const char *activationName(Activation activation) {
  switch (activation) {
  case Activation::None:
    return "none";
  case Activation::Relu:
    return "relu";
  }
  return "<unknown activation>";
}

//===----------------------------------------------------------------------===//
// Errors.
//===----------------------------------------------------------------------===//

/// Why a file was rejected.
///
/// Section 9.2: every failure returns a structured error naming the
/// instruction index and a stable check name, not a bare null. The check name
/// is stable because it comes from the ISA description, so a test asserts the
/// check rather than the wording and a reworded message is not a broken test.
struct ProgramError {
  /// Which rule was broken. `checkName(check)` is the stable spelling.
  Check check = Check::Structure;
  /// What was wrong, in a sentence. It does **not** repeat the check name;
  /// `toString` puts the two together so that every message has the same
  /// shape.
  std::string detail;
  /// The instruction this concerns, or -1 when the failure is about the file
  /// as a whole, a memory region, or a debug entry.
  int64_t instructionIndex = -1;
  /// The section relative index of the region or debug entry this concerns,
  /// or -1. Together with `where` it names the thing without a second field
  /// per section.
  int64_t elementIndex = -1;
  /// Which list `elementIndex` indexes, empty when it is -1.
  std::string where;

  /// `check-name: detail` with the location appended when there is one.
  std::string toString() const;
};

//===----------------------------------------------------------------------===//
// The record types.
//===----------------------------------------------------------------------===//

/// A buffer in DRAM: an offset, a shape and an element type.
struct MemRegion {
  uint64_t offset = 0;
  ElemType elementType = ElemType::F32;
  std::vector<int64_t> shape;

  /// The product of the extents, or -1 when it would exceed the shape limit.
  int64_t elementCount() const;
  /// The bytes this region occupies, or -1 when the element count is -1.
  ///
  /// **Multiplies by the element type's size** rather than assuming four bytes
  /// per element, which is what makes an `I8` region whose byte size no longer
  /// matches its shape a detectable corruption.
  int64_t byteSize() const;
};

/// A constant buffer, with the data the model carried.
struct Constant {
  MemRegion region;
  std::vector<uint8_t> data;
};

/// One operand of an instruction.
///
/// The strides are where Section 5.5's layout decision lands below the tensor
/// level: an NHWC tensor becomes NCHW extents with permuted strides, and the
/// rank 1 channel broadcast of ADR 0005 becomes a stride 0 operand. A kernel
/// indexes its operands through their strides and therefore needs no special
/// case for either.
///
/// The shape is here as well as the strides, and it has to be. A stride vector
/// with no extents beside it addresses nothing in particular, and no check in
/// Section 9.2 could be written without it: `operand-extent` asks whether the
/// consumer's need fits the buffer, and for `MATMUL` the K extent appears in
/// no result shape. `docs/ISA_MANUAL.md` documents the operand record field
/// for field for this reason.
struct Operand {
  MemSpace space = MemSpace::Scratchpad;
  ElemType elementType = ElemType::F32;
  int64_t address = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  /// The half open byte span this operand addresses, from `address`.
  ///
  /// `1 + sum((extent - 1) * stride)` elements, which is the rule
  /// `docs/ARCHITECTURE.md` fixed at P5: a view's byte range comes from its
  /// strides and not from its extents, so a stride 0 broadcast of eight
  /// channels spans eight elements and not one hundred and twenty eight.
  /// Returns -1 when the arithmetic would overflow the shape limit.
  int64_t addressedByteSpan() const;
};

/// One instruction.
///
/// Every field is physically present on every instruction, because the format
/// has no tags and therefore no optional fields. What varies is which fields
/// an opcode gives meaning to, and an opcode that gives a field no meaning
/// requires it to hold its neutral value: an empty vector, a zero, or
/// `Activation::None`. That is what makes `attribute-size` and
/// `attribute-value` checkable rather than advisory, and it is why a file that
/// sets an activation on `RESHAPE` is rejected rather than quietly ignored.
struct Instruction {
  Opcode opcode = Opcode::NOP;
  Activation activation = Activation::None;

  MemSpace resultSpace = MemSpace::Scratchpad;
  ElemType resultElementType = ElemType::F32;
  int64_t resultAddress = 0;
  std::vector<int64_t> resultShape;

  std::vector<Operand> operands;

  /// Four entries in ONNX order: padTop, padLeft, padBottom, padRight.
  std::vector<int64_t> pads;
  /// Two entries, the window stride, height then width.
  std::vector<int64_t> strides;
  /// Two entries, the window dilation, height then width.
  std::vector<int64_t> dilations;
  /// Two entries, the window extent, height then width.
  std::vector<int64_t> kernel;
  /// The channel group count. Zero on an opcode that has no groups.
  int64_t group = 0;
  /// The permutation for `TRANSPOSE`, the single axis for `CONCAT`.
  std::vector<int64_t> axes;

  /// The quantization scale. Zero on an opcode that does not quantize.
  float scale = 0.0f;
  /// The quantization zero point. Zero on an opcode that does not quantize.
  int32_t zeroPoint = 0;

  /// The fixed point rescaling pair of Section 14, present from version one.
  ///
  /// The identity is a multiplier of 1 and a shift of 0, and every opcode that
  /// does not requantize carries exactly that. Section 14 requires the pair to
  /// live in the binary rather than in an out of band calibration JSON, and a
  /// format that could not hold it would force a version bump at P14.
  int32_t requantMultiplier = 1;
  int32_t requantShift = 0;

  /// The elements the result holds, or -1 when the shape exceeds the limit.
  int64_t resultElementCount() const;
  /// The bytes the result occupies, or -1 when the element count is -1.
  int64_t resultByteSize() const;
};

/// One entry of the optional debug section: a program counter and the ONNX
/// node name the instruction came from.
struct DebugEntry {
  uint32_t pc = 0;
  std::string name;
};

//===----------------------------------------------------------------------===//
// The program.
//===----------------------------------------------------------------------===//

/// A whole `.nbin`.
class Program {
public:
  /// The magic word, as a `u32` in host order. Spelled `NBIN` on a little
  /// endian machine, and the host byte order rule means it is spelled `NIBN`
  /// on a big endian one; the constant is the number, not the letters.
  static constexpr uint32_t kMagic = 0x4E49424Eu;

  /// The format version. **Starts at 1.**
  ///
  /// Any change to the layout bumps it, and an unknown version is rejected
  /// with the `version` check rather than reinterpreted. Adding an opcode does
  /// not bump it, because an opcode value is data inside a layout that did not
  /// move. Section 9.1 names the one future change that is already accounted
  /// for: P14's quantization, whose fields are present from here.
  static constexpr uint32_t kVersion = 1;

  /// The cap on every `u32` count field the format has.
  ///
  /// Section 9.2 rule 3: the cap is a number, not an adjective. One constant,
  /// defined here, applied to every count, reported by the `count-cap` check,
  /// stated in `docs/ISA_MANUAL.md`, and probed at `(1 << 28) - 1`, `1 << 28`
  /// and `(1 << 28) + 1` in the corpus of Section 17.3. A length prefixed
  /// string in the debug section is exactly the shape of bug that turns a
  /// malformed file into a heap overflow.
  ///
  /// A count **equal** to this is permitted and a count above it is not, which
  /// is what makes it a maximum rather than a bound. The decoder additionally
  /// refuses any count whose payload cannot fit in the bytes that remain, so a
  /// count of `kMaxCount` in a hundred byte file is rejected before a single
  /// element is reserved.
  static constexpr uint32_t kMaxCount = 1u << 28;

  /// The shape product limit of Section 9.2 rule 1.
  ///
  /// The guard tests `extent > kShapeLimit / product` **before** multiplying,
  /// never after. A guard that multiplies first is itself the overflow, and a
  /// shape like `{2^40, 2^24}` then wraps to a small product and is accepted.
  static constexpr int64_t kShapeLimit = int64_t{1} << 40;

  /// The largest debug name this decoder accepts, in bytes. It is far below
  /// `kMaxCount` on purpose: an ONNX node name is a few dozen characters, and
  /// a cap that only stops a heap overflow is a cap that lets a file waste a
  /// quarter of a gigabyte legally.
  static constexpr uint32_t kMaxDebugNameBytes = 4096;

  /// The scratchpad the program declares it needs.
  ///
  /// Section 9.3: the simulator sizes its scratchpad **strictly from this**
  /// and never grows it to cover the writes it finds, because doing so would
  /// absorb out of range result addresses and neutralize the bounds checking.
  uint64_t scratchpadBytes = 0;
  /// The DRAM the program declares it needs: inputs, outputs, constants and
  /// spill slots all live inside it.
  uint64_t dramBytes = 0;

  std::vector<MemRegion> inputs;
  std::vector<MemRegion> outputs;
  std::vector<Constant> constants;
  /// The DRAM buffers the allocator spilled into, per the P5 extension of
  /// `docs/ARCHITECTURE.md`: the encoder gives each `npuisa.spill_slot`
  /// allocation an address in the DRAM map, the way it already does for
  /// constants and for the input and output regions.
  std::vector<MemRegion> spillSlots;
  std::vector<Instruction> instructions;
  /// The optional debug section. Empty encodes as a zero count, and a stripped
  /// binary is legal: `npu-translate --strip-debug` produces one.
  std::vector<DebugEntry> debug;

  /// The bytes of this program.
  std::vector<uint8_t> encode() const;

  /// Reads a file and validates it. Returns the check that rejected it, or
  /// nothing on success.
  ///
  /// This is the entry point everything except `npu-objdump` uses.
  /// `npu-sim` calls `validate()` again before execution, deliberately: the
  /// two calls guard different moments and a program that arrived through a
  /// path that skipped one still meets the other.
  static std::optional<ProgramError> decode(llvm::ArrayRef<uint8_t> bytes,
                                            Program &out);

  /// Reads a file **without** validating it.
  ///
  /// It exists so `npu-objdump` can dump a suspect file, and its output is
  /// prefixed with a warning. It still refuses a file it cannot frame: a
  /// truncated section has no well defined contents, and reading past the end
  /// of a buffer to show somebody what is there is the bug this whole section
  /// is about. What it skips is the semantic half, so an instruction with an
  /// impossible address is decoded and printed rather than rejected.
  static std::optional<ProgramError>
  decodeUnvalidated(llvm::ArrayRef<uint8_t> bytes, Program &out);

  /// Every rule of Section 9.2, in a fixed order, stopping at the first
  /// failure. Returns nothing when the program is well formed.
  std::optional<ProgramError> validate() const;

  /// The ONNX node name recorded for an instruction, or an empty string.
  llvm::StringRef debugNameFor(uint32_t pc) const;
};

//===----------------------------------------------------------------------===//
// Shared arithmetic.
//===----------------------------------------------------------------------===//

/// The product of `extents`, or -1 when it would exceed `Program::kShapeLimit`.
///
/// **The overflow safe form.** Each extent is tested against the remaining
/// headroom before it is multiplied in, never after, because a guard that
/// multiplies first is itself the overflow. A negative or zero extent returns
/// -1 too: a zero element buffer is not a shape this machine has an address
/// for, and letting one through would make every byte range calculation below
/// it degenerate.
int64_t checkedElementCount(llvm::ArrayRef<int64_t> extents);

} // namespace nbin

#endif // NPU_ENCODING_PROGRAM_H
