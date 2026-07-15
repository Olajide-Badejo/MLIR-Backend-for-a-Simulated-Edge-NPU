//===- npu-objdump.cpp - Disassemble an .nbin file ------------------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Encoding/Disassembler.h"
#include "NPU/Encoding/Program.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: npu-objdump <file.nbin>\n";
    return 1;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::cerr << "npu-objdump: cannot open " << argv[1] << "\n";
    return 1;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  auto program = npu::Program::decode(bytes);
  if (!program) {
    std::cerr << "npu-objdump: malformed or truncated .nbin\n";
    return 1;
  }
  std::cout << npu::disassemble(*program);
  return 0;
}
