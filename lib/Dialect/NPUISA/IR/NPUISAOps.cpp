//===- NPUISAOps.cpp - NPUISA dialect operation implementations -----------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"

#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npuisa;

#define GET_OP_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.cpp.inc"

LogicalResult DmaLoadOp::verify() {
  auto buffer = llvm::cast<BufferType>(getDest().getType());
  if (buffer.getTensorType() != getSource().getType())
    return emitOpError("scratchpad buffer shape ")
           << buffer.getTensorType() << " must match the DRAM source "
           << getSource().getType();
  return success();
}

LogicalResult DmaStoreOp::verify() {
  auto buffer = llvm::cast<BufferType>(getSource().getType());
  if (buffer.getTensorType() != getDest().getType())
    return emitOpError("scratchpad buffer shape ")
           << buffer.getTensorType() << " must match the DRAM destination "
           << getDest().getType();
  return success();
}
