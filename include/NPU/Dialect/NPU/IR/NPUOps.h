//===- NPUOps.h - NPU dialect operations ------------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_IR_NPUOPS_H
#define NPU_DIALECT_NPU_IR_NPUOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"

#define GET_OP_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOps.h.inc"

#endif // NPU_DIALECT_NPU_IR_NPUOPS_H
