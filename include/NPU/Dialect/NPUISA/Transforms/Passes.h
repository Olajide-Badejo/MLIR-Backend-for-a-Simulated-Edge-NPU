//===- Passes.h - NPUISA transform passes -----------------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H
#define NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace npuisa {

#define GEN_PASS_DECL
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

} // namespace npuisa
} // namespace mlir

#endif // NPU_DIALECT_NPUISA_TRANSFORMS_PASSES_H
