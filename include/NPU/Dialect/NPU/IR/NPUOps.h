//===- NPUOps.h - NPU dialect operations ------------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
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

// The fused activation enum and its specialized attribute are consumed by the
// generated op classes, so they must be visible first.
#include "NPU/Dialect/NPU/IR/NPUEnums.h.inc"

#define GET_OP_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOps.h.inc"

#endif // NPU_DIALECT_NPU_IR_NPUOPS_H
