//===- CostModel.h - Analytical cost model ----------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//
//
// An analytical, not cycle accurate, cost model. Every number it produces is a
// simulated estimate, never a measurement. The constants are documented
// assumptions and are reported alongside results.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_SIMULATOR_COSTMODEL_H
#define NPU_SIMULATOR_COSTMODEL_H

#include <cstdint>

namespace npu {

struct CostModel {
  // Systolic array: a 16x16 grid of multiply accumulate units.
  int64_t macsPerCycle = 256;
  // Scratchpad DMA bandwidth in bytes per cycle.
  int64_t dramBytesPerCycle = 16;
  // Elementwise SIMD lane width.
  int64_t lanes = 16;
  // Fixed issue overhead charged once per instruction.
  int64_t issueOverhead = 1;

  int64_t dmaCycles(int64_t bytes) const {
    return (bytes + dramBytesPerCycle - 1) / dramBytesPerCycle + issueOverhead;
  }
  int64_t macCycles(int64_t macs) const {
    return (macs + macsPerCycle - 1) / macsPerCycle + issueOverhead;
  }
  int64_t elementwiseCycles(int64_t elements) const {
    return (elements + lanes - 1) / lanes + issueOverhead;
  }
};

} // namespace npu

#endif // NPU_SIMULATOR_COSTMODEL_H
