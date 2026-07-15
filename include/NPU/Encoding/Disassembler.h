//===- Disassembler.h - Render an encoded program as text -------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_DISASSEMBLER_H
#define NPU_ENCODING_DISASSEMBLER_H

#include "NPU/Encoding/Program.h"

#include <string>

namespace npu {

// Render a decoded program as a human readable assembly listing, the way a small
// processor's objdump would.
std::string disassemble(const Program &program);

} // namespace npu

#endif // NPU_ENCODING_DISASSEMBLER_H
