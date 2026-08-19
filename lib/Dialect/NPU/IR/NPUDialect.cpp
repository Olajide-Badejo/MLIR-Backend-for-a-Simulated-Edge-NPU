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
#include "mlir/Interfaces/TilingInterface.h"

using namespace mlir;
using namespace mlir::npu;

#include "NPU/Dialect/NPU/IR/NPUOpsDialect.cpp.inc"

void NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/Dialect/NPU/IR/NPUOps.cpp.inc"
      >();
  registerAttributes();

  // TilingInterface is implemented as an external model in its own translation
  // unit, so this library does not depend on the tiling stack. Promising it
  // here is what turns a forgotten call to
  // registerNPUTilingInterfaceExternalModels into a named error at the point of
  // use, saying that the interface was promised and not provided, rather than
  // into a silent "this operation does not implement TilingInterface" that
  // reads like a design decision.
  declarePromisedInterface<TilingInterface, Conv2DOp>();
  declarePromisedInterface<TilingInterface, MatMulOp>();
  declarePromisedInterface<TilingInterface, AddOp>();
  declarePromisedInterface<TilingInterface, MulOp>();
  declarePromisedInterface<TilingInterface, ReluOp>();
  declarePromisedInterface<TilingInterface, MaxPool2DOp>();
  declarePromisedInterface<TilingInterface, AvgPool2DOp>();
  declarePromisedInterface<TilingInterface, TransposeOp>();
  declarePromisedInterface<TilingInterface, ConcatOp>();
  declarePromisedInterface<TilingInterface, BatchNormOp>();
}
