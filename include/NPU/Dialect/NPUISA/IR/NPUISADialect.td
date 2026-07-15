//===- NPUISADialect.td - NPUISA dialect definition --------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_DIALECT_TD
#define NPUISA_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def NPUISA_Dialect : Dialect {
  let name = "npuisa";
  let summary = "Instruction level dialect for a simulated edge NPU.";
  let description = [{
    The `npuisa` dialect models the program after memory placement has been
    decided. It has an explicit two level memory model: tensors live in DRAM and
    are moved to and from a small fixed size scratchpad by DMA instructions.
    Compute instructions read and write scratchpad buffers only.

    A value of `!npuisa.buffer` type is a scratchpad resident tensor. The
    scratchpad allocator assigns each buffer a byte offset, recorded as the
    `address` attribute on the instruction that produces it. The instruction
    stream is straight line: inference graphs are static DAGs, so the ISA has no
    branches.
  }];
  let cppNamespace = "::mlir::npuisa";
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

class NPUISA_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPUISA_Dialect, mnemonic, traits>;

class NPUISA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NPUISA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // NPUISA_DIALECT_TD
