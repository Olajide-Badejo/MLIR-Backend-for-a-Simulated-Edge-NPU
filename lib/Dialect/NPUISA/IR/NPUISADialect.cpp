//===- NPUISADialect.cpp - The npuisa dialect -------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::npuisa;

#include "NPU/Dialect/NPUISA/IR/NPUISAOpsDialect.cpp.inc"

void NPUISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.cpp.inc"
      >();
  registerTypes();

  // No declarePromisedInterface calls here, and the absence is the decision
  // Section 8 makes rather than an omission. Both interfaces this dialect
  // implements, DestinationStyleOpInterface and MemoryEffectOpInterface, are
  // declared in the ODS source and generated into the operation classes
  // themselves, so there is nothing to promise: they cannot be forgotten
  // because there is no separate registration call to forget.
  //
  // TilingInterface is promised on the npu operations and not here, because the
  // tiling pass runs before lowering and the interface has to exist where the
  // pass can see it. There is nothing on the memref side for it to be useful
  // on: getResultTilePosition and generateResultTileValue are what make tile
  // and fuse work and they are meaningful only on operations that have results,
  // and no compute operation in this dialect has one.
}
