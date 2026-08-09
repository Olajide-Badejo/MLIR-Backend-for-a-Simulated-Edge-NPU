//===- npu-sim.cpp - Run an encoded program in the simulator --------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
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

  npu::ValidationError error;
  auto program = npu::Program::decode(readBytes(nbin), &error);
  if (!program) {
    std::cerr << "npu-sim: " << nbin << " is not a runnable program\n"
              << "  " << error.toString() << "\n";
    return 1;
  }

  // Re-validate before execution. decode already did, but the program can be
  // constructed or mutated by other paths and the simulator does raw memory
  // arithmetic, so the check is cheap next to the cost of being wrong.
  if (auto bad = program->validate()) {
    std::cerr << "npu-sim: program failed validation\n"
              << "  " << bad->toString() << "\n";
    return 1;
  }

  std::vector<std::vector<float>> inputs;
  if (!inputPath.empty()) {
    std::vector<float> data = readFloats(inputPath);
    // The input file used to be memcpy'd into DRAM without ever being compared
    // against the region it was going into, so an oversized input.bin was a
    // heap overflow.
    if (program->inputs.empty()) {
      std::cerr << "npu-sim: " << inputPath
                << " was given but the program declares no inputs\n";
      return 1;
    }
    int64_t expected = 1;
    for (int64_t d : program->inputs[0].shape)
      expected *= d;
    if (static_cast<int64_t>(data.size()) != expected) {
      std::cerr << "npu-sim: " << inputPath << " holds " << data.size()
                << " float(s) but input 0 has shape [";
      for (size_t i = 0; i < program->inputs[0].shape.size(); ++i)
        std::cerr << (i ? "x" : "") << program->inputs[0].shape[i];
      std::cerr << "], which is " << expected << " float(s)\n";
      return 1;
    }
    inputs.push_back(std::move(data));
  }

  npu::Simulator sim(*program);
  npu::SimResult result = sim.run(inputs);
  if (!result.error.empty()) {
    std::cerr << "npu-sim: refused a memory access during execution\n"
              << "  " << result.error << "\n";
    return 1;
  }

  // Write every output, not just the first. A multi output model used to lose
  // all but outputs.front() with no warning. One output still goes to the given
  // path unchanged; more than one gets a numbered suffix beside it.
  if (!outputPath.empty() && !result.outputs.empty()) {
    auto writeOne = [](const std::string &path, const std::vector<float> &o) {
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char *>(o.data()),
                static_cast<std::streamsize>(o.size() * sizeof(float)));
    };
    if (result.outputs.size() == 1) {
      writeOne(outputPath, result.outputs.front());
    } else {
      std::string stem = outputPath;
      std::string suffix;
      size_t dot = outputPath.find_last_of('.');
      size_t slash = outputPath.find_last_of('/');
      if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        stem = outputPath.substr(0, dot);
        suffix = outputPath.substr(dot);
      }
      for (size_t i = 0; i < result.outputs.size(); ++i) {
        std::string path = stem + "." + std::to_string(i) + suffix;
        writeOne(path, result.outputs[i]);
        std::cerr << "npu-sim: wrote output " << i << " to " << path << "\n";
      }
    }
  }

  std::string json = result.stats.toJson();
  if (!statsPath.empty()) {
    std::ofstream s(statsPath);
    s << json << "\n";
  }
  std::cout << json << "\n";
  return 0;
}
