//===- NPUAttrs.h - npu dialect attributes ----------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_IR_NPUATTRS_H
#define NPU_DIALECT_NPU_IR_NPUATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "NPU/Dialect/NPU/IR/NPUOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "NPU/Dialect/NPU/IR/NPUOpsAttrDefs.h.inc"

#endif // NPU_DIALECT_NPU_IR_NPUATTRS_H
