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

/// The one operation of this dialect that can hold a value.
///
/// *Added at P9, as the fix for D-0033.* Without this hook `-sccp` computed the
/// right lattice over this dialect's IR and then had nowhere to put the answer,
/// so it reported no change on every input and its Section 16.2 ablation row
/// would have been a row of zeros for a reason that was four missing lines
/// rather than a property of the pass.
///
/// The guard is narrow on purpose. `npu.constant`'s verifier requires the
/// attribute's own type to equal the result type, so an attribute that is not
/// an `ElementsAttr`, or one whose type is not the requested type, has no
/// operation in this dialect that could hold it, and returning null is how a
/// materializer says so rather than building something that fails to verify.
Operation *NPUDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  auto elements = dyn_cast<ElementsAttr>(value);
  if (!elements || elements.getType() != type)
    return nullptr;
  return ConstantOp::create(builder, loc, type, elements);
}
