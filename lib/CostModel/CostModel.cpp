//===- CostModel.cpp - the array's occupancy arithmetic -------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The two charges that are more than one expression: the folded GEMM and the
// convolution that maps onto it. Everything else in the cost model is a
// constant or a one line formula and lives in the header, where a reader
// looking for the model finds the whole of it.
//
//===----------------------------------------------------------------------===//

#include "NPU/Simulator/CostModel.h"

using namespace nbin;

ComputeCharge nbin::gemmCharge(int64_t m, int64_t k, int64_t n, int64_t peak) {
  ComputeCharge charge;
  if (m <= 0 || k <= 0 || n <= 0 || peak <= 0)
    return charge;

  // The temporal factor. One weight tile is pushed into the array before the
  // first activation row can be streamed against it, and the push is as deep as
  // the array. It does not vary across the tiles of one instruction, because
  // every tile of one instruction sees the same number of streamed rows.
  const double delta =
      static_cast<double>(m) / (static_cast<double>(m) + kWeightPreloadCycles);

  const int64_t dim = kArrayDim;
  double cycles = 0.0;
  int64_t macs = 0;
  double weightedUtilization = 0.0;

  for (int64_t kBase = 0; kBase < k; kBase += dim) {
    const int64_t rows = std::min(dim, k - kBase);
    for (int64_t nBase = 0; nBase < n; nBase += dim) {
      const int64_t columns = std::min(dim, n - nBase);

      // The spatial factor: the fraction of the array this tile occupies. At
      // most one, and it falls as the tile gets narrower.
      const double utilization = (static_cast<double>(rows) / dim) *
                                 (static_cast<double>(columns) / dim);

      const int64_t tileMacs = m * rows * columns;
      macs += tileMacs;
      weightedUtilization += utilization * static_cast<double>(tileMacs);
      cycles += static_cast<double>(tileMacs) /
                (utilization * delta * static_cast<double>(peak));
    }
  }

  charge.cycles = cycles;
  charge.macs = macs;
  charge.effectiveMacs = cycles * static_cast<double>(peak);
  charge.utilization =
      macs > 0 ? weightedUtilization / static_cast<double>(macs) : 1.0;
  charge.delta = delta;
  return charge;
}

ComputeCharge nbin::conv2dCharge(int64_t batch, int64_t outputChannels,
                                 int64_t inputChannels, int64_t outputHeight,
                                 int64_t outputWidth, int64_t kernelHeight,
                                 int64_t kernelWidth, int64_t group,
                                 int64_t peak) {
  ComputeCharge charge;
  if (group <= 0 || outputChannels % group != 0 || inputChannels % group != 0)
    return charge;

  // The streamed rows: one activation row per output position, across the whole
  // batch. Batch is derived from the shape here exactly as Section 7.3 requires
  // it to be everywhere else, and there is no N == 1 shortcut to take.
  const int64_t rows = batch * outputHeight * outputWidth;
  // The stationary weight matrix, per group.
  const int64_t reduction = (inputChannels / group) * kernelHeight * kernelWidth;
  const int64_t columns = outputChannels / group;

  const ComputeCharge perGroup = gemmCharge(rows, reduction, columns, peak);

  // Every group presents the same shape to the array, so the whole convolution
  // is the group charge multiplied by the group count. The utilization and the
  // delta are intensive quantities and do not multiply.
  charge.cycles = perGroup.cycles * static_cast<double>(group);
  charge.macs = perGroup.macs * group;
  charge.effectiveMacs = perGroup.effectiveMacs * static_cast<double>(group);
  charge.utilization = perGroup.utilization;
  charge.delta = perGroup.delta;
  return charge;
}
