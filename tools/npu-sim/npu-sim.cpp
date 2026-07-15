//===- npu-sim.cpp - Run an encoded program in the simulator --------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Program.h"
#include "NPU/Simulator/Simulator.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> readBytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<float> readFloats(const std::string &path) {
  std::vector<uint8_t> bytes = readBytes(path);
  std::vector<float> out(bytes.size() / 4);
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::string nbin, inputPath, outputPath, statsPath;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--input" && i + 1 < argc)
      inputPath = argv[++i];
    else if (a == "--output" && i + 1 < argc)
      outputPath = argv[++i];
    else if (a == "--stats" && i + 1 < argc)
      statsPath = argv[++i];
    else
      nbin = a;
  }
  if (nbin.empty()) {
    std::cerr << "usage: npu-sim <model.nbin> [--input in.bin] "
                 "[--output out.bin] [--stats stats.json]\n";
    return 1;
  }

  auto program = npu::Program::decode(readBytes(nbin));
  if (!program) {
    std::cerr << "npu-sim: malformed .nbin\n";
    return 1;
  }

  std::vector<std::vector<float>> inputs;
  if (!inputPath.empty())
    inputs.push_back(readFloats(inputPath));

  npu::Simulator sim(*program);
  npu::SimResult result = sim.run(inputs);

  if (!outputPath.empty() && !result.outputs.empty()) {
    std::ofstream out(outputPath, std::ios::binary);
    const auto &o = result.outputs.front();
    out.write(reinterpret_cast<const char *>(o.data()),
              static_cast<std::streamsize>(o.size() * sizeof(float)));
  }

  std::string json = result.stats.toJson();
  if (!statsPath.empty()) {
    std::ofstream s(statsPath);
    s << json << "\n";
  }
  std::cout << json << "\n";
  return 0;
}
