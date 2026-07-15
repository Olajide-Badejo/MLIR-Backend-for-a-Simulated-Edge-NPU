//===- NPUISATypes.td - NPUISA dialect types ---------------*- tablegen -*-===//
//
// Part of the npu-mlir project, under the MIT License.
// See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_TYPES_TD
#define NPUISA_TYPES_TD

include "NPU/Dialect/NPUISA/IR/NPUISADialect.td"

def NPUISA_BufferType : NPUISA_Type<"Buffer", "buffer"> {
  let summary = "A scratchpad resident buffer holding a tensor of a fixed shape.";
  let description = [{
    Wraps the logical ranked tensor type of the data it holds. The byte size is
    the number of elements times four (fp32). The scratchpad allocator assigns
    the actual offset separately, as an attribute on the producing instruction.
  }];
  let parameters = (ins "::mlir::RankedTensorType":$tensorType);
  let assemblyFormat = "`<` $tensorType `>`";
}

#endif // NPUISA_TYPES_TD
