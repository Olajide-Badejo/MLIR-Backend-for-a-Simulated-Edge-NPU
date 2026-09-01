//===- Passes.h - npu tensor level passes -----------------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The declarations and the registration entry point for the `npu` dialect's
// own passes, in the same shape the `npuisa` dialect's have had since P4.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_TRANSFORMS_PASSES_H
#define NPU_DIALECT_NPU_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::npu {

#define GEN_PASS_DECL
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

} // namespace mlir::npu

#endif // NPU_DIALECT_NPU_TRANSFORMS_PASSES_H
