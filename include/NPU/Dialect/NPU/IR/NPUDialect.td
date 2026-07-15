//===- NPUDialect.td - NPU dialect definition --------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_TD
#define NPU_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NPU dialect
//===----------------------------------------------------------------------===//

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Tensor level dialect for a simulated edge NPU.";
  let description = [{
    The `npu` dialect models a trained convolutional neural network at the tensor
    level, before any decision about memory placement has been made. Values are
    builtin ranked tensors of fp32. Operators mirror the subset of ONNX that the
    frontend supports (convolution, matmul, pointwise arithmetic, activation,
    pooling, shape manipulation, and batch normalization).

    Fused operators are expressed as a single op that carries an optional bias
    operand and an activation attribute rather than as a combinatorial explosion
    of named fused ops. This gives the fusion patterns a single clean target and
    mirrors the fused operator style used by TFLite.

    Ops that compute a pure function of their operands carry the `Pure` trait so
    that the generic dead code elimination and constant folding infrastructure
    applies without any op specific support.
  }];
  let cppNamespace = "::mlir::npu";

  // The dialect materializes constants during folding, so it needs a hook.
  let hasConstantMaterializer = 1;

  // Ops operate on builtin tensors produced and consumed across dialect
  // boundaries; depend on the tensor dialect for interop.
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

#endif // NPU_DIALECT_TD
