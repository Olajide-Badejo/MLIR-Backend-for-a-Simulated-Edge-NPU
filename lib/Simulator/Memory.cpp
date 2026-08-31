//===- Memory.cpp - the cold half of the checked accessors ----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The trap path and the sizing, which are the two parts of `Machine` that are
// not one comparison. Both are here rather than in the header so that the hot
// accessors stay small enough to inline and the messages stay long enough to be
// worth reading.
//
//===----------------------------------------------------------------------===//

#include "NPU/Simulator/Memory.h"

#include <cstring>

using namespace nbin;

namespace {

/// The message every range trap has, in one place, so that two traps a hundred
/// lines apart cannot describe the same failure two different ways.
std::string rangeMessage(MemSpace space, int64_t address, int64_t bytes,
                         const char *what, uint64_t size) {
  std::string message = "out of range ";
  message += what;
  message += ": ";
  message += memSpaceName(space);
  message += " address ";
  message += std::to_string(address);
  message += " for ";
  message += std::to_string(bytes);
  message += " bytes, and the ";
  message += memSpaceName(space);
  message += " is ";
  message += std::to_string(size);
  message += " bytes";
  return message;
}

} // namespace

Machine::Machine(uint64_t scratchpadBytes, uint64_t dramBytes) {
  // Strictly what the program declared, and nothing rounded up to cover
  // anything. Section 9.3 is explicit: growing the scratchpad to cover the
  // writes found in the instruction stream absorbs out of range result
  // addresses and neutralizes the bounds checking this class exists to be.
  if (scratchpadBytes > kMaxMemoryBytes) {
    recordTrap("the program declares a scratchpad of " +
               std::to_string(scratchpadBytes) +
               " bytes and this simulator refuses to allocate more than " +
               std::to_string(kMaxMemoryBytes) +
               ". It is a refusal rather than a clamp: a clamped scratchpad "
               "would be a different machine from the one the file describes");
    return;
  }
  if (dramBytes > kMaxMemoryBytes) {
    recordTrap("the program declares " + std::to_string(dramBytes) +
               " bytes of DRAM and this simulator refuses to allocate more "
               "than " +
               std::to_string(kMaxMemoryBytes));
    return;
  }

  scratchpad.assign(static_cast<size_t>(scratchpadBytes), 0);
  dram.assign(static_cast<size_t>(dramBytes), 0);
}

uint8_t *Machine::outOfRange(MemSpace space, int64_t byteAddress, int64_t bytes,
                             const char *what) {
  recordTrap(rangeMessage(space, byteAddress, bytes, what,
                          storage(space).size()));
  return nullptr;
}

uint8_t *Machine::badElement(MemSpace space, int64_t byteAddress,
                             int64_t elementOffset, const char *what) {
  std::string message = "unaddressable ";
  message += what;
  message += ": ";
  message += memSpaceName(space);
  message += " address ";
  message += std::to_string(byteAddress);
  message += " plus element offset ";
  message += std::to_string(elementOffset);
  message += " either overflows or is not four byte aligned";
  recordTrap(std::move(message));
  return nullptr;
}

void Machine::recordTrap(std::string message) {
  std::lock_guard<std::mutex> held(trapMutex);
  if (!firstTrap)
    firstTrap = std::move(message);
}

bool Machine::copy(MemSpace toSpace, int64_t toAddress, MemSpace fromSpace,
                   int64_t fromAddress, int64_t bytes) {
  // Both ends are resolved before either is touched, so a transfer with one bad
  // end copies nothing at all rather than half of something. A partially
  // completed DMA would leave the machine in a state no program could have
  // produced, which is a worse thing to hand a caller than a refusal.
  const uint8_t *source = readBytes(fromSpace, fromAddress, bytes, "DMA source");
  if (!source)
    return false;
  uint8_t *destination =
      writeBytes(toSpace, toAddress, bytes, "DMA destination");
  if (!destination)
    return false;
  if (bytes > 0)
    std::memmove(destination, source, static_cast<size_t>(bytes));
  return true;
}
