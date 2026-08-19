//===- NPUISATypes.h - npuisa dialect types ---------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISATYPES_H
#define NPU_DIALECT_NPUISA_IR_NPUISATYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOpsTypes.h.inc"

#endif // NPU_DIALECT_NPUISA_IR_NPUISATYPES_H
