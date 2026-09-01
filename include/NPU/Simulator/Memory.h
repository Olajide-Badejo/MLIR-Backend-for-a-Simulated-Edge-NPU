//===- Memory.h - the machine's two memories, checked ---------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 9.3's contract with the format, in code.
//
// **The scratchpad is sized strictly from the declared `scratchpadBytes`.** It
// is never grown to cover the writes it finds, because doing so absorbs out of
// range result addresses and neutralizes the bounds checking, and because the
// sizing loop would then be arithmetic on unvalidated input at the exact entry
// point the validation exists to defend.
//
// **Every access goes through a bounds checked accessor, in every build mode.**
// There is no `assert(false)` on the trap path and there is no accessor that
// skips the check when `NDEBUG` is defined. An out of range access records the
// first trap message, returns a null pointer, and the caller skips the access.
// Graceful refusal is the contract in a release build and in an assertions
// build alike, and `unittests/Simulator/TrapTest.cpp` proves it in both.
//
// The accessors are `inline` and the trap path is not, which is the shape that
// keeps the check cheap enough that nobody is ever tempted to argue for
// removing it: the fast path is a comparison and a pointer addition, and the
// string building lives out of line where it is taken once per run.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_SIMULATOR_MEMORY_H
#define NPU_SIMULATOR_MEMORY_H

#include "NPU/Encoding/Program.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace nbin {

/// The two memories, and the only way to reach either.
class Machine {
public:
  /// The largest memory this simulator will allocate, per space.
  ///
  /// A `.nbin` is free to declare a scratchpad of nearly 2^64 bytes and
  /// `Program::validate()` has no rule against it: the format's job is to
  /// describe a machine, not to know how much memory this host has. So the
  /// refusal lives here, where the allocation is, and it is a refusal rather
  /// than a clamp. A clamp would size the scratchpad differently from what the
  /// file declared, which is precisely the thing Section 9.3 forbids.
  static constexpr uint64_t kMaxMemoryBytes = uint64_t{1} << 30;

  /// Sizes both memories from what the program declared, exactly.
  ///
  /// A declared size above `kMaxMemoryBytes` leaves the machine trapped before
  /// a single instruction runs, which the caller sees as a `SimResult` carrying
  /// the message and no statistics.
  Machine(uint64_t scratchpadBytes, uint64_t dramBytes);

  /// The first trap message, or nothing. First, not last: a program that ran
  /// off the end once usually does it a thousand times, and the thousandth
  /// message says nothing the first did not.
  const std::optional<std::string> &trap() const { return firstTrap; }
  bool trapped() const { return firstTrap.has_value(); }

  uint64_t scratchpadBytes() const { return scratchpad.size(); }
  uint64_t dramBytes() const { return dram.size(); }

  /// A checked pointer to `bytes` bytes at `byteAddress`, or null.
  ///
  /// `what` names the access in the trap message. It is a `const char *`
  /// literal at every call site, so the message costs nothing until it is
  /// needed.
  const uint8_t *readBytes(MemSpace space, int64_t byteAddress, int64_t bytes,
                           const char *what) {
    const std::vector<uint8_t> &memory = storage(space);
    if (!inRange(memory.size(), byteAddress, bytes))
#ifdef NDEBUG
      return nullptr;
#else
      return outOfRange(space, byteAddress, bytes, what);
#endif
    return memory.data() + byteAddress;
  }

  /// The mirror of `readBytes`, for a write.
  uint8_t *writeBytes(MemSpace space, int64_t byteAddress, int64_t bytes,
                      const char *what) {
    std::vector<uint8_t> &memory = storage(space);
    if (!inRange(memory.size(), byteAddress, bytes))
      return outOfRange(space, byteAddress, bytes, what);
    return memory.data() + byteAddress;
  }

  /// A checked f32 element, `elementOffset` elements past `byteAddress`.
  ///
  /// The offset is separate from the address because that is how a kernel
  /// addresses a strided operand: the address is the buffer's, and the offset
  /// is the dot product of the index with the strides, which the operand
  /// carries. Passing the two separately means the multiplication by the
  /// element size happens once, here, rather than at every call site.
  ///
  /// Three things are checked and each one is a different failure. The scaling
  /// of the offset by the element size can overflow, and signed overflow is
  /// undefined behaviour rather than a large number. The address has to be four
  /// byte aligned, because this machine's load unit has no unaligned path and
  /// because forming a misaligned `float *` is undefined behaviour on the host
  /// too. And the resulting span has to lie inside the memory.
  ///
  /// Three predictable branches per element is the price of the contract, and
  /// it is a price worth naming rather than optimising away: an accessor that
  /// skipped the check in the build that ships is an accessor that only checks
  /// where nothing was going to go wrong.
  const float *readF32(MemSpace space, int64_t byteAddress,
                       int64_t elementOffset, const char *what) {
    int64_t address = 0;
    if (!elementAddress(byteAddress, elementOffset, address))
      return reinterpret_cast<const float *>(
          badElement(space, byteAddress, elementOffset, what));
    return reinterpret_cast<const float *>(readBytes(space, address, 4, what));
  }

  /// The mirror of `readF32`, for a write.
  float *writeF32(MemSpace space, int64_t byteAddress, int64_t elementOffset,
                  const char *what) {
    int64_t address = 0;
    if (!elementAddress(byteAddress, elementOffset, address))
      return reinterpret_cast<float *>(
          badElement(space, byteAddress, elementOffset, what));
    return reinterpret_cast<float *>(writeBytes(space, address, 4, what));
  }

  /// Copies `bytes` bytes between two spaces, checking both ends.
  ///
  /// Returns false and records a trap when either end is out of range, without
  /// having copied anything. A partial copy would leave the machine in a state
  /// no program could have produced, which is a worse thing to hand a caller
  /// than a refusal.
  bool copy(MemSpace toSpace, int64_t toAddress, MemSpace fromSpace,
            int64_t fromAddress, int64_t bytes);

  /// Records a trap that is not a memory range failure.
  ///
  /// The unimplemented kernels of Phase P14 use it, and so does the executor
  /// when a program asks for something the machine cannot do. It is the same
  /// channel as a range trap on purpose: a caller has one field to read.
  void recordTrap(std::string message);

private:
  const std::vector<uint8_t> &storage(MemSpace space) const {
    return space == MemSpace::Scratchpad ? scratchpad : dram;
  }
  std::vector<uint8_t> &storage(MemSpace space) {
    return space == MemSpace::Scratchpad ? scratchpad : dram;
  }

  /// Whether `[address, address + bytes)` lies inside a memory of `size`.
  ///
  /// The addition is done as a subtraction on purpose. `address + bytes` can
  /// overflow a signed integer for an address near the limit, and signed
  /// overflow is undefined behaviour rather than a large number. This is the
  /// same trap D-0021 found in the validator's range messages, and it is
  /// written the safe way here from the start.
  static bool inRange(size_t size, int64_t address, int64_t bytes) {
    if (address < 0 || bytes < 0)
      return false;
    if (static_cast<uint64_t>(address) > size)
      return false;
    return static_cast<uint64_t>(bytes) <=
           static_cast<uint64_t>(size) - static_cast<uint64_t>(address);
  }

  /// Scales an element offset by the f32 element size and adds it to a byte
  /// address, refusing rather than overflowing, and refusing a result that is
  /// not four byte aligned.
  static bool elementAddress(int64_t byteAddress, int64_t elementOffset,
                             int64_t &address) {
    constexpr int64_t kElementSize = 4;
    constexpr int64_t kLimit = INT64_MAX / kElementSize;
    if (elementOffset > kLimit || elementOffset < -kLimit)
      return false;
    const int64_t scaled = elementOffset * kElementSize;
    if (scaled > 0 && byteAddress > INT64_MAX - scaled)
      return false;
    if (scaled < 0 && byteAddress < INT64_MIN - scaled)
      return false;
    address = byteAddress + scaled;
    return (address & (kElementSize - 1)) == 0;
  }

  /// Records the trap and returns null. Out of line because it is cold.
  uint8_t *outOfRange(MemSpace space, int64_t byteAddress, int64_t bytes,
                      const char *what);

  /// The mirror of `outOfRange` for an element address that overflowed or was
  /// misaligned. It is a separate message because "the address does not exist"
  /// and "the address is not one this machine can load from" send a reader to
  /// two different places.
  uint8_t *badElement(MemSpace space, int64_t byteAddress,
                      int64_t elementOffset, const char *what);

  std::vector<uint8_t> scratchpad;
  std::vector<uint8_t> dram;

  /// Guards `firstTrap`. The convolution kernel of Section 10.3 runs its batch
  /// and output channel loops in parallel, so two threads can reach a trap at
  /// once. Under that kernel "first" means first in time rather than first in
  /// iteration order, which is stated here rather than left for somebody to
  /// discover: a program that traps is already outside the contract, and the
  /// tests that assert a particular trap message use the serial kernels.
  std::mutex trapMutex;
  std::optional<std::string> firstTrap;
};

} // namespace nbin

#endif // NPU_SIMULATOR_MEMORY_H
