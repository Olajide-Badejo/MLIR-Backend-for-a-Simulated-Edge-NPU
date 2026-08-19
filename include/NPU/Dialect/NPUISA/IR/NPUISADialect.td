//===- NPUISADialect.td - The npuisa dialect ---------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_DIALECT
#define NPUISA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def NPUISA_Dialect : Dialect {
  let name = "npuisa";
  let summary = "Instruction level operations for a simulated edge NPU.";
  let description = [{
    The `npuisa` dialect is one step above the encoded binary. Where the `npu`
    dialect works on tensors and has results, this one works on `memref` in two
    memory spaces and has none: every compute instruction reads its `ins`
    operands and writes its single `outs` destination, which is exactly what the
    machine does and exactly what the encoder has to emit.

    The two memory spaces are `#npu.scratchpad` and `#npu.dram`, and they are
    the attributes the `npu` dialect already defines rather than a second pair
    defined here. There is one machine, so there is one name per memory, and a
    dialect boundary is not a reason to give a memory a second spelling.

    A function in this dialect carries `npuisa.scratchpad_bytes` and
    `npuisa.scratchpad_budget`, which the encoder writes into the binary and the
    simulator reads back.
  }];
  let cppNamespace = "::mlir::npuisa";

  // Types, yes; attributes, no. This dialect defines exactly one type, the
  // token, and it defines no attributes of its own: the two memory spaces come
  // from the `npu` dialect and the two function level attributes are plain
  // integers under a dialect prefix rather than attribute definitions. Asking
  // for the default attribute printer and parser would generate declarations
  // that nothing defines.
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Registers the types this dialect defines. Defined in NPUISATypes.cpp so
    /// the generated storage lands in one translation unit.
    void registerTypes();

    /// The name of the function level attribute carrying the scratchpad bytes
    /// a function actually uses. Spelled once, here, because a string literal
    /// repeated across a verifier, an encoder, and a simulator is three places
    /// for a typo to hide and one of them will be the one nobody tests.
    static ::llvm::StringRef getScratchpadBytesAttrName() {
      return "npuisa.scratchpad_bytes";
    }

    /// The name of the function level attribute carrying the scratchpad budget
    /// the function was compiled against.
    static ::llvm::StringRef getScratchpadBudgetAttrName() {
      return "npuisa.scratchpad_budget";
    }
  }];
}

class NPUISA_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPUISA_Dialect, mnemonic, traits>;

class NPUISA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<NPUISA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // NPUISA_DIALECT
