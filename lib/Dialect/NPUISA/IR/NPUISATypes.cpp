//===- NPUISATypes.cpp - npuisa dialect types -------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::npuisa;

#define GET_TYPEDEF_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOpsTypes.cpp.inc"

void NPUISADialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "NPU/Dialect/NPUISA/IR/NPUISAOpsTypes.cpp.inc"
      >();
}
