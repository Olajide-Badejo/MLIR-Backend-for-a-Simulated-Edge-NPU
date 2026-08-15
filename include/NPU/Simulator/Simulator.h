//===- Simulator.h - Execute an encoded npuisa program ----------*- C++ -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_SIMULATOR_SIMULATOR_H
#define NPU_SIMULATOR_SIMULATOR_H

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/CostModel.h"

#include <cstdint>
#include <string>
#include <vector>

namespace npu {

// Simulated performance estimates. Never a measurement.
struct Stats {
  int64_t cycles = 0;
  int64_t dramBytesRead = 0;
  int64_t dramBytesWritten = 0;
  int64_t instructions = 0;

  std::string toJson() const;
};

struct SimResult {
  std::vector<std::vector<float>> outputs; // parallel to Program::outputs
  Stats stats;

  // Empty on a clean run. Set when a memory access was refused, naming the
  // instruction and what it tried to touch. A validated program cannot trigger
  // this, but the simulator is also reachable as a library and from hand built
  // Program values, so it reports rather than trusting the caller.
  std::string error;
};

class Simulator {
public:
  Simulator(const Program &program, CostModel cost = {})
      : program(program), cost(cost) {}

  // Run with inputs parallel to Program::inputs, each a flat row major fp32
  // buffer. Returns the outputs and the simulated statistics.
  SimResult run(const std::vector<std::vector<float>> &inputs);

private:
  const Program &program;
  CostModel cost;
};

} // namespace npu

#endif // NPU_SIMULATOR_SIMULATOR_H
