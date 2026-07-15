//===- NPUISADialect.cpp - NPUISA dialect implementation ------------------===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::npuisa;

#include "NPU/Dialect/NPUISA/IR/NPUISAOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.cpp.inc"

void NPUISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.cpp.inc"
      >();
}
