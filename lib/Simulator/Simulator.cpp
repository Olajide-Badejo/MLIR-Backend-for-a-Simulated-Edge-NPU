//===- Simulator.cpp - the executor and the two timelines -----*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The loop that runs a `.nbin`, and the cycle accounting of Section 5.5.
//
// **Two independent timelines.** An instruction starts at the later of its port
// becoming free and its last operand becoming ready, and the reported total is
// the later of the two timelines at `HALT`. The prologue and the epilogue are
// modelled rather than assumed away: the first fill and the last drain cannot
// overlap, and they do not, because the dependency edges say so.
//
//===----------------------------------------------------------------------===//
//
// WHICH `npu` OPERATION IS SIMULATED BY WHICH KERNEL
//
// `scripts/check-reachability.py` answers the simulation layer of law 2 by
// looking for each operation's mnemonic in this file, so the table below is
// what that check reads. It is a table rather than a scattering of mentions
// because a reader deserves the same answer the checker gets, and because a
// mnemonic that appeared only inside an unrelated word would satisfy a
// substring search while telling a human nothing.
//
//   npu.conv2d       CONV2D      lib/Simulator/Kernels.cpp
//   npu.matmul       MATMUL      lib/Simulator/Kernels.cpp
//   npu.add          ADD         lib/Simulator/Kernels.cpp
//   npu.mul          MUL         lib/Simulator/Kernels.cpp
//   npu.relu         RELU        lib/Simulator/Kernels.cpp
//   npu.max_pool2d   POOL_MAX    lib/Simulator/Kernels.cpp
//   npu.avg_pool2d   POOL_AVG    lib/Simulator/Kernels.cpp
//   npu.reshape      RESHAPE     lib/Simulator/Kernels.cpp
//   npu.transpose    TRANSPOSE   lib/Simulator/Kernels.cpp
//   npu.concat       CONCAT      lib/Simulator/Kernels.cpp
//
// Four operations reach the machine without a kernel of their own, and each one
// is accounted for rather than absent:
//
//   npu.batch_norm   decomposes at lowering into a MUL and an ADD, per its own
//                    ODS description, so it is simulated by the two kernels the
//                    decomposition produces. There is no batch normalization
//                    unit on this machine and inventing an opcode for one would
//                    be inventing hardware.
//   npu.constant     is encoded as a constant region in the DRAM map with its
//                    data rather than as an instruction. The simulator loads
//                    those regions into DRAM before the first instruction runs,
//                    which is what `Simulator::Simulator` does below, and the
//                    load that brings one on chip is a DMA_LOAD.
//   npu.fused_op     is flattened into its parent block by
//                    `-npu-lower-to-npuisa` before any instruction is emitted,
//                    so the operations its region held are the ones that run.
//   npu.yield        is that region's terminator and disappears with it.
//
// The same four are recorded as eliminated sources in
// `include/NPU/Encoding/NPUISADescription.td`, which is where the encoding
// layer reads them from. **The simulation layer is still answered by a
// substring search over this file and that is weaker than the encoding
// layer's**, which Phase P6 made mechanical. Making this one mechanical too is
// on Phase P8's desk, and `docs/PHASE_STATE.md` says so rather than leaving the
// difference for somebody to notice.
//
//===----------------------------------------------------------------------===//

#include "NPU/Simulator/Simulator.h"

#include "Kernels.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

using namespace nbin;
using namespace nbin::detail;

namespace {

/// A buffer an instruction wrote, and when it became readable.
///
/// The dependency model is byte overlap rather than name equality, because at
/// this level there are no names: the allocator has already turned every value
/// into a span, and two instructions that touch overlapping spans are ordered
/// whether or not they came from the same tensor.
struct WrittenSpan {
  MemSpace space;
  int64_t begin;
  int64_t end;
  double time;
};

/// The half open byte span an operand addresses, or nothing when the operand's
/// arithmetic does not close.
std::optional<std::pair<int64_t, int64_t>> spanOf(const Operand &operand) {
  const int64_t span = operand.addressedByteSpan();
  if (span < 0 || operand.address < 0)
    return std::nullopt;
  return std::make_pair(operand.address, operand.address + span);
}

} // namespace

//===----------------------------------------------------------------------===//
// Construction.
//===----------------------------------------------------------------------===//

Simulator::Simulator(const Program &program) : program(program) {
  memory = std::make_unique<Machine>(program.scratchpadBytes, program.dramBytes);
  if (memory->trapped()) {
    constructionError = *memory->trap();
    return;
  }

  // The constants go into DRAM before anything runs, at the offsets the DRAM
  // map assigned them. `npu.constant` is not an instruction on this machine:
  // the encoder writes a constant region with its data, and the load that
  // brings it on chip is a DMA_LOAD like any other.
  for (const auto &[index, constant] : llvm::enumerate(program.constants)) {
    const int64_t bytes = constant.region.byteSize();
    if (bytes < 0 || static_cast<size_t>(bytes) != constant.data.size()) {
      constructionError =
          "constant region " + std::to_string(index) + " declares " +
          (bytes < 0 ? std::string("an impossible size")
                     : std::to_string(bytes) + " bytes") +
          " and carries " + std::to_string(constant.data.size());
      return;
    }
    if (bytes == 0)
      continue;
    uint8_t *destination =
        memory->writeBytes(MemSpace::Dram,
                           static_cast<int64_t>(constant.region.offset), bytes,
                           "constant region");
    if (!destination) {
      constructionError = *memory->trap();
      return;
    }
    std::memcpy(destination, constant.data.data(),
                static_cast<size_t>(bytes));
  }
}

Simulator::~Simulator() = default;

Machine &Simulator::machine() { return *memory; }
const Machine &Simulator::machine() const { return *memory; }

bool Simulator::loadInput(size_t index, llvm::ArrayRef<uint8_t> bytes,
                          std::string &error) {
  if (index >= program.inputs.size()) {
    error = "this program declares " + std::to_string(program.inputs.size()) +
            " input regions and there is no region " + std::to_string(index);
    return false;
  }
  const MemRegion &region = program.inputs[index];
  const int64_t expected = region.byteSize();
  if (expected < 0) {
    error = "input region " + std::to_string(index) +
            " has a shape with no well defined size";
    return false;
  }
  // Section 9.3: the input's size is compared against the declared region and a
  // mismatch is refused. Naming both numbers is the difference between a
  // message somebody can act on and one they have to investigate.
  if (static_cast<size_t>(expected) != bytes.size()) {
    error = "input region " + std::to_string(index) + " is " +
            std::to_string(expected) + " bytes and the file supplied is " +
            std::to_string(bytes.size());
    return false;
  }
  if (expected == 0)
    return true;
  uint8_t *destination =
      memory->writeBytes(MemSpace::Dram, static_cast<int64_t>(region.offset),
                         expected, "input region");
  if (!destination) {
    error = memory->trap().value_or("the input region does not fit in DRAM");
    return false;
  }
  std::memcpy(destination, bytes.data(), bytes.size());
  return true;
}

llvm::ArrayRef<uint8_t> Simulator::outputBytes(size_t index) {
  if (index >= program.outputs.size())
    return {};
  const MemRegion &region = program.outputs[index];
  const int64_t bytes = region.byteSize();
  if (bytes < 0)
    return {};
  const uint8_t *start =
      memory->readBytes(MemSpace::Dram, static_cast<int64_t>(region.offset),
                        bytes, "output region");
  if (!start)
    return {};
  return llvm::ArrayRef<uint8_t>(start, static_cast<size_t>(bytes));
}

//===----------------------------------------------------------------------===//
// Running.
//===----------------------------------------------------------------------===//

SimResult Simulator::run(const SimOptions &options) {
  SimResult result;
  // Section 9.3 asks for this deliberately, even though `Program::decode`
  // already validated. The two calls guard different moments: a program that
  // arrived through a path that skipped one still meets the other, and a
  // `Program` built in memory by a pass has been through neither. It is cheap;
  // the whole 734 case malformed input corpus of Phase P6 validates in single
  // digit milliseconds.
  if (std::optional<ProgramError> failure = program.validate()) {
    result.error = failure->toString();
    return result;
  }
  return execute(options);
}

SimResult Simulator::runUnvalidated(const SimOptions &options) {
  return execute(options);
}

SimResult Simulator::execute(const SimOptions &options) {
  SimResult result;
  if (constructionError) {
    result.error = constructionError;
    return result;
  }
  if (memory->trapped()) {
    result.error = memory->trap();
    return result;
  }

  // The two ports. In single port mode both names refer to the same clock,
  // which is the whole of what the flag does: Section 5.5 keeps it so that a
  // number published under the simpler model stays reproducible.
  double dmaFree = 0.0;
  double computeFree = 0.0;
  double dmaTotal = 0.0;
  double computeTotal = 0.0;

  std::vector<WrittenSpan> writes;
  double weightedUtilization = 0.0;
  double weightedDelta = 0.0;

  auto readyAt = [&writes](MemSpace space, int64_t begin, int64_t end) {
    double at = 0.0;
    for (const WrittenSpan &written : writes)
      if (written.space == space && begin < written.end &&
          written.begin < end)
        at = std::max(at, written.time);
    return at;
  };

  for (const Instruction &instruction : program.instructions) {
    ++result.stats.instructions;

    const Port port = portFor(instruction.opcode);
    const bool onDma = port == Port::Dma && !options.singlePort;

    // The later of the port becoming free and the last operand becoming ready.
    double start = options.singlePort ? std::max(dmaFree, computeFree)
                                      : (onDma ? dmaFree : computeFree);
    for (const Operand &operand : instruction.operands) {
      if (std::optional<std::pair<int64_t, int64_t>> span = spanOf(operand))
        start = std::max(start, readyAt(operand.space, span->first,
                                        span->second));
    }

    KernelCost cost;
    if (Kernel kernel = kernelFor(instruction.opcode)) {
      // The arity is checked here, once, from the generated table, before any
      // kernel indexes an operand. `Program::validate()` already refuses an
      // instruction with too few operands and names the `arity` check, so on
      // the validated path this branch is never taken. `runUnvalidated` is the
      // path it exists for, and D-0026 is what it cost to learn that it was
      // needed: a `RELU` carrying no operands reached `operands.front()` and
      // the machine aborted inside the standard library, which is precisely
      // the assertion Section 9.3 says the trap path must not have.
      //
      // The number comes from `opcodeInfo`, which is generated from the same
      // description `validate()` reads, so there is no second arity table to
      // keep in agreement with the first.
      const OpcodeInfo &info = opcodeInfo(instruction.opcode);
      if (static_cast<int64_t>(instruction.operands.size()) <
          static_cast<int64_t>(info.minOperands)) {
        memory->recordTrap(
            std::string(info.name) + " takes at least " +
            std::to_string(info.minOperands) + " operands and this one has " +
            std::to_string(instruction.operands.size()) +
            ", so there is nothing for the kernel to read");
      } else {
        cost = kernel(*memory, instruction);
      }
    }

    // ---- Traffic, counted from the shapes the instruction names. -----------
    for (const Operand &operand : instruction.operands) {
      const int64_t elements = checkedElementCount(operand.shape);
      if (elements < 0)
        continue;
      if (operand.space == MemSpace::Dram) {
        const int64_t size = elementByteSize(operand.elementType);
        if (size > 0)
          result.stats.dramBytesRead +=
              static_cast<uint64_t>(elements) * static_cast<uint64_t>(size);
      } else {
        result.stats.scratchpadElementsRead += static_cast<uint64_t>(elements);
      }
    }
    const int64_t resultElements = instruction.resultElementCount();
    if (resultElements > 0) {
      if (instruction.resultSpace == MemSpace::Dram) {
        const int64_t bytes = instruction.resultByteSize();
        if (bytes > 0)
          result.stats.dramBytesWritten += static_cast<uint64_t>(bytes);
      } else {
        result.stats.scratchpadElementsWritten +=
            static_cast<uint64_t>(resultElements);
      }
    }

    // ---- The charge. -------------------------------------------------------
    const double duration = cost.cycles + kIssueOverheadCycles;
    const double end = start + duration;
    if (onDma) {
      dmaFree = end;
      dmaTotal += duration;
    } else if (options.singlePort) {
      dmaFree = computeFree = end;
      (port == Port::Dma ? dmaTotal : computeTotal) += duration;
    } else {
      computeFree = end;
      computeTotal += duration;
    }

    result.stats.macs += static_cast<uint64_t>(cost.macs);
    result.stats.int8Macs += static_cast<uint64_t>(cost.int8Macs);
    result.stats.effectiveMacs += cost.effectiveMacs;
    weightedUtilization += cost.utilization * static_cast<double>(cost.macs);
    weightedDelta += cost.delta * static_cast<double>(cost.macs);

    if (resultElements > 0) {
      const int64_t bytes = instruction.resultByteSize();
      if (bytes > 0 && instruction.resultAddress >= 0)
        writes.push_back(WrittenSpan{instruction.resultSpace,
                                     instruction.resultAddress,
                                     instruction.resultAddress + bytes, end});
    }

    // A trap stops the run at the end of the instruction that caused it. The
    // access itself was already skipped, which is the contract; carrying on
    // afterwards would compute a result out of whatever was in the buffer the
    // skipped write did not fill, and reporting that as an answer would be
    // worse than reporting nothing.
    if (memory->trapped()) {
      result.error = memory->trap();
      break;
    }

    if (instruction.opcode == Opcode::HALT) {
      result.reachedHalt = true;
      break;
    }
  }

  result.stats.dmaCycles = dmaTotal;
  result.stats.computeCycles = computeTotal;
  // The total is the later of the two timelines. In single port mode the two
  // names refer to one clock, so the same expression produces the sum, which is
  // what makes the flag a reproducibility switch rather than a second model.
  result.stats.cycles = std::max(dmaFree, computeFree);
  result.stats.overlapFraction =
      overlapFraction(dmaTotal, computeTotal, result.stats.cycles);
  if (result.stats.macs > 0) {
    const double macs = static_cast<double>(result.stats.macs);
    result.stats.utilization = weightedUtilization / macs;
    result.stats.delta = weightedDelta / macs;
  }
  return result;
}
