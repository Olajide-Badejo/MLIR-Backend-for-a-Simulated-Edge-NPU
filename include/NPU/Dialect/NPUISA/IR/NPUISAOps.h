//===- NPUISAOps.h - NPUISA dialect operations ------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISAOPS_H
#define NPU_DIALECT_NPUISA_IR_NPUISAOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"

#define GET_OP_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h.inc"

#endif // NPU_DIALECT_NPUISA_IR_NPUISAOPS_H
