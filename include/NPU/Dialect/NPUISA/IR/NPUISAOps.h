//===- NPUISAOps.h - npuisa dialect operations ------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_IR_NPUISAOPS_H
#define NPU_DIALECT_NPUISA_IR_NPUISAOPS_H

// The two memory space attributes this dialect's memref constraints are built
// on are defined by the npu dialect, and the generated type constraint
// functions below name them by their C++ class. Without this include the
// generated code does not compile, and the error names mlir::npu::ScratchpadAttr
// in a file nobody wrote, so the include is here rather than in the .cpp.
#include "NPU/Dialect/NPU/IR/NPUAttrs.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
// PatternMatch.h is needed by the generated header rather than by this one:
// the two asynchronous operations declare a static `canonicalize` taking a
// PatternRewriter, and without the declaration in scope the generated
// signature silently degrades to taking an `int &` and every definition of it
// fails to match. That failure names the definition rather than the missing
// include, so the include is here with the reason attached.
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h.inc"

#endif // NPU_DIALECT_NPUISA_IR_NPUISAOPS_H
