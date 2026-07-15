//===- NPUOps.cpp - NPU dialect operation implementations -----------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::npu;

#define GET_OP_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto attrType = llvm::cast<ShapedType>(getValue().getType());
  auto resultType = llvm::cast<ShapedType>(getResult().getType());
  if (attrType != resultType)
    return emitOpError("value attribute type ")
           << attrType << " does not match result type " << resultType;
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValueAttr(); }
