//===- npu-objdump.cpp - Disassemble an .nbin file ------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
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
  // Deliberately the unvalidated decode. Inspecting a file you already suspect
  // is broken is what a disassembler is for, so refusing to dump the programs
  // validate rejects would remove the tool exactly when it is wanted. The
  // validation result is reported as a comment instead.
  auto program = npu::Program::decodeUnvalidated(bytes);
  if (!program) {
    std::cerr << "npu-objdump: " << argv[1]
              << " is truncated, or does not begin with the NPUB magic\n";
    return 1;
  }
  if (auto bad = program->validate())
    std::cout << "; WARNING: this program would be rejected by npu-sim\n"
              << ";   " << bad->toString() << "\n";
  std::cout << npu::disassemble(*program);
  return 0;
}
