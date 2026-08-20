//===- InstructionEncoder.h - npuisa IR to .nbin ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The encoder of Section 9: allocated `npuisa` IR in, a `Program` out.
//
// **What it expects.** A module holding exactly one `func.func`, lowered by
// `-npu-lower-to-npuisa` and allocated by `-npu-allocate-scratchpad`. That
// means scratchpad buffers are `memref.view` operations over one flat arena
// with `memref.reinterpret_cast` above them where a layout or a broadcast
// needs one, and the function carries `npuisa.scratchpad_bytes`. Anything else
// is diagnosed rather than guessed at.
//
// **The function argument convention, decided here.** P4 left the in and out
// split as a positional rule and its handoff said that if P6 wanted it
// explicit, an argument attribute was the place. This is that decision: every
// argument of an encodable function carries `npuisa.arg`, a string attribute
// holding `"in"` or `"out"`, and the encoder refuses a function whose
// arguments do not all carry one. The trailing order is unchanged and is
// checked rather than assumed, so a pass that appends an argument in the wrong
// place is a diagnostic instead of a silently mislabelled input region.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_INSTRUCTIONENCODER_H
#define NPU_ENCODING_INSTRUCTIONENCODER_H

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Encoding/Program.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace nbin {

/// Encodes the single function of `module` into `program`.
///
/// Emits diagnostics on failure and leaves `program` unspecified. A module
/// holding more than one function is **diagnosed rather than truncated**: a
/// `.nbin` holds one program, and quietly encoding the first function of two
/// would produce a file that runs and computes the wrong model.
///
/// `stripDebug` produces a binary with an empty debug section, which is legal
/// and is what `npu-translate --strip-debug` writes.
mlir::LogicalResult encodeModule(mlir::ModuleOp module, Program &program,
                                 bool stripDebug = false);

} // namespace nbin

#endif // NPU_ENCODING_INSTRUCTIONENCODER_H
