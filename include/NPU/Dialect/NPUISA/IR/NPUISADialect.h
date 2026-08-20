//===- NPUISADialect.h - The npuisa dialect ---------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISADIALECT_H
#define NPU_DIALECT_NPUISA_IR_NPUISADIALECT_H

#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"

#include "NPU/Dialect/NPUISA/IR/NPUISAOpsDialect.h.inc"

namespace mlir {
namespace npuisa {

/// The argument attribute that says whether an argument of a lowered function
/// is an input region or an output region.
///
/// *Added at P6.* P4 fixed the shape of a lowered signature, the model's own
/// arguments first and its outputs appended after them, and left the split as
/// a positional convention. Its handoff said that if the encoder wanted the
/// split explicit then an argument attribute was the place, and this is that
/// attribute. `-npu-lower-to-npuisa` writes it because it is the pass that
/// performs the split; the encoder reads it and refuses a function without it,
/// so the convention is checked rather than assumed.
///
/// It lives on the dialect rather than in the encoder's header because the
/// `npuisa.` prefix is the dialect's namespace, and because a name defined
/// where its producer and its consumer can both see it is a name that cannot
/// be misspelled in one of the two.
inline constexpr llvm::StringLiteral kArgKindAttrName = "npuisa.arg";
/// The value on an argument the program reads.
inline constexpr llvm::StringLiteral kArgKindIn = "in";
/// The value on an argument the program writes.
inline constexpr llvm::StringLiteral kArgKindOut = "out";

} // namespace npuisa
} // namespace mlir

#endif // NPU_DIALECT_NPUISA_IR_NPUISADIALECT_H
