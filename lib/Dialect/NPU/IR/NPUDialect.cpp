//===- NPUDialect.cpp - NPU dialect implementation ------------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"

#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/Dialect/NPU/IR/NPUOpsDialect.cpp.inc"

void NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"
      >();
}

Operation *NPUDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto elements = llvm::dyn_cast<ElementsAttr>(value))
    return ConstantOp::create(builder, loc, type, elements);
  return nullptr;
}
