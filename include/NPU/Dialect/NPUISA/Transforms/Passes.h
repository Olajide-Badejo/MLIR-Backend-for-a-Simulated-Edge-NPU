//===- Passes.h - npuisa transformation passes ------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The declarations and the registration entry point for this dialect's passes.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H
#define NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::npuisa {

#define GEN_PASS_DECL
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

} // namespace mlir::npuisa

#endif // NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H
