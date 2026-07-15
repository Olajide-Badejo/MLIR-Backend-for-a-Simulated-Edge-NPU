//===- InstructionEncoder.h - npuisa MLIR to Program ------------*- C++ -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_ENCODING_INSTRUCTIONENCODER_H
#define NPU_ENCODING_INSTRUCTIONENCODER_H

#include "NPU/Encoding/Program.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace npu {

// Encode an allocated npuisa function (post scratchpad allocation) into the
// Program model. Returns failure with a diagnostic on the function if the IR is
// not in the expected fully lowered and allocated form.
mlir::FailureOr<Program> encodeFunction(mlir::func::FuncOp func);

} // namespace npu

#endif // NPU_ENCODING_INSTRUCTIONENCODER_H
