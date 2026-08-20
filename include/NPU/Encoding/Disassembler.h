//===- Disassembler.h - printing a .nbin --------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// What `npu-objdump` prints.
//
// The per opcode format strings are **generated** from the ISA description, so
// a new opcode arrives with its disassembly already written and cannot arrive
// without one: `npu-isa-tblgen` refuses a description whose opcode has no
// format string.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_DISASSEMBLER_H
#define NPU_ENCODING_DISASSEMBLER_H

#include "NPU/Encoding/Program.h"

#include <optional>
#include <string>

namespace nbin {

/// Renders a program as text.
///
/// `failure` is the validation failure the caller already found, or nothing.
/// When it is set the output opens with a warning block naming it, because
/// Section 9.2 requires `npu-objdump`'s output on a suspect file to be
/// prefixed with a warning: a disassembly that looks the same whether or not
/// the file validated is a disassembly somebody will quote as evidence.
std::string disassemble(const Program &program,
                        const std::optional<ProgramError> &failure);

} // namespace nbin

#endif // NPU_ENCODING_DISASSEMBLER_H
