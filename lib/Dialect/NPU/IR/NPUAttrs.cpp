//===- NPUAttrs.cpp - npu dialect attributes --------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPU/IR/NPUDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/Dialect/NPU/IR/NPUOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOpsAttrDefs.cpp.inc"

void NPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "NPU/Dialect/NPU/IR/NPUOpsAttrDefs.cpp.inc"
      >();
}
