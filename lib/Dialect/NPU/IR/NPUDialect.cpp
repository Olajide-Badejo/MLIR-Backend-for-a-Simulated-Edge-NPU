//===- NPUDialect.cpp - The npu dialect -------------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/Dialect/NPU/IR/NPUOpsDialect.cpp.inc"

void NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"
      >();
  registerAttributes();
}
