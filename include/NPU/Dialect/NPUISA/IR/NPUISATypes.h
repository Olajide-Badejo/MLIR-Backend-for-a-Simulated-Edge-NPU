//===- NPUISATypes.h - NPUISA dialect types ---------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISATYPES_H
#define NPU_DIALECT_NPUISA_IR_NPUISATYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h.inc"

#endif // NPU_DIALECT_NPUISA_IR_NPUISATYPES_H
