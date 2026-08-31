//===- NPUDialect.td - The npu dialect ---------------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT
#define NPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Tensor level operations for a simulated edge NPU.";
  let description = [{
    The `npu` dialect is the entry level of this compiler. It holds the graph
    an ONNX model imports to: tensor typed operations with static shapes, in
    destination passing style, verified arithmetically so that an impossible
    convolution is refused at the boundary rather than producing a wrong number
    out of the simulator with nothing pointing at the cause.

    Everything below this level, from the `npuisa` dialect down, works on
    `memref` in two memory spaces. This dialect works on the builtin `tensor`
    type with an optional layout encoding, which keeps it on MLIR's own
    infrastructure rather than beside it.
  }];
  let cppNamespace = "::mlir::npu";

  // No `useDefaultTypePrinterParser` here, deliberately. This dialect defines
  // no types of its own: it works on the builtin `tensor`, which is the whole
  // point of the type choice. Asking for the default type printer and parser
  // would generate declarations for `parseType` and `printType` that nothing
  // defines, and the failure is a link error a long way from the cause.
  let useDefaultAttributePrinterParser = 1;

  // A dialect with no constant materializer cannot have a constant put back
  // into it. Every MLIR pass that arrives at a constant value and then needs an
  // operation to hold it goes through `materializeConstant`, and a dialect that
  // implements none makes those passes reach the right answer and then decline
  // to write it down. That is D-0033, found at P9 by measuring `-sccp` on this
  // dialect and getting no change at all on any input.
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    /// Registers the attributes this dialect defines. Defined in NPUAttrs.cpp
    /// so the generated storage lands in one translation unit.
    void registerAttributes();
  }];
}

class NPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<NPU_Dialect, mnemonic, traits>;

class NPU_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<NPU_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // NPU_DIALECT
