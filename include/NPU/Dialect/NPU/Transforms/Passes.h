//===- Passes.h - NPU dialect transform passes ------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_TRANSFORMS_PASSES_H
#define NPU_DIALECT_NPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace npu {

#define GEN_PASS_DECL
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"

} // namespace npu
} // namespace mlir

#endif // NPU_DIALECT_NPU_TRANSFORMS_PASSES_H
