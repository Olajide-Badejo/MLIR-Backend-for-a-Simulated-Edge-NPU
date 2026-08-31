//===- Kernels.h - the per opcode kernels ---------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The interface between the executor and the arithmetic.
//
// This header is private to `lib/Simulator`. Nothing outside the simulator has
// a reason to call a kernel directly: the semantics tests drive whole programs,
// because a kernel tested through a `Program` is a kernel tested through the
// validation, the dispatch and the memory model it will actually run behind.
//
// **A kernel both computes and charges.** Section 10.2 requires MAC counts to
// be counted explicitly where the cost is charged rather than inferred from a
// cycle figure, and the shape a kernel walked is known in exactly one place,
// which is the kernel. Returning the charge is what keeps that literally true.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_LIB_SIMULATOR_KERNELS_H
#define NPU_LIB_SIMULATOR_KERNELS_H

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Memory.h"

#include <cstdint>

namespace nbin {
namespace detail {

/// The two ports of Section 5.5, which are the two independent timelines.
enum class Port {
  Compute,
  Dma,
};

/// What executing one instruction cost, before the issue overhead.
struct KernelCost {
  double cycles = 0.0;
  int64_t macs = 0;
  int64_t int8Macs = 0;
  double effectiveMacs = 0.0;
  double utilization = 1.0;
  double delta = 1.0;
};

/// A kernel: it reads its operands, writes its result, and says what it cost.
///
/// A kernel never throws and never asserts. Everything it touches goes through
/// the checked accessors of `Machine`, so a bad address records a trap and the
/// access is skipped rather than performed.
using Kernel = KernelCost (*)(Machine &, const Instruction &);

/// The kernel for an opcode, or null when the opcode is control rather than
/// computation.
///
/// The table is expanded from the generated `NPUISADispatch.def`, so an opcode
/// appended to `NPUISADescription.td` with `needsKernel` set and no kernel
/// written is a **compile error** in this table rather than a runtime surprise.
/// That is the half of Section 9.4's claim Phase P6 could not demonstrate,
/// because there was no simulator to demonstrate it in.
Kernel kernelFor(Opcode opcode);

/// Which timeline an opcode issues on.
Port portFor(Opcode opcode);

} // namespace detail
} // namespace nbin

#endif // NPU_LIB_SIMULATOR_KERNELS_H
