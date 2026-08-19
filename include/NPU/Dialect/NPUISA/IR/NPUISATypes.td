//===- NPUISATypes.td - npuisa types and constraints -------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// One real type, the DMA token, plus the memref constraints the instruction
// operations are written against.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_TYPES
#define NPUISA_TYPES

include "NPU/Dialect/NPUISA/IR/NPUISADialect.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// !npuisa.token
//===----------------------------------------------------------------------===//

def NPUISA_TokenType : NPUISA_Type<"Token", "token"> {
  let summary = "A handle on an outstanding asynchronous DMA transfer.";
  let description = [{
    ```mlir
    %t = npuisa.dma_load_async %src, %dst
       : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
    npuisa.await %t
    ```

    **The token is a scheduling handle, not data.** It carries no value, it has
    no width, and nothing computes with it. What it carries is an ordering
    obligation: the transfer it names has been started and has not been waited
    on, and the region of the program between the two is where the destination
    buffer is being written by hardware nobody in the instruction stream can
    see.

    That is why the type exists at all rather than the asynchronous form being
    a pair of operations matched by position. A value with a use is something
    the verifier can follow, and every rule in Section 8 is expressed as a
    property of that use: exactly one, an `await`, in the same block, after the
    producer. Matched by position, each of those rules would instead be a search
    over the block, and a pass that reordered two transfers would silently
    repair the search while breaking the program.

    The token deliberately has no parameters. A parameter naming the transfer,
    or the destination, or a channel number would be data that the verifier
    would then have to keep consistent with the operands it was derived from,
    and the operands are already there to be read.
  }];
}

//===----------------------------------------------------------------------===//
// The memref constraints.
//
// Every instruction in this dialect works on a statically shaped memref in one
// of two memory spaces. Static shape is a type level requirement rather than a
// verifier one, for the reason the `npu` dialect gives about its tensors: there
// is nothing below this level that can represent a dynamic extent, so accepting
// one here would be accepting IR that cannot be encoded, and the earliest layer
// that can name a problem is the one that should.
//===----------------------------------------------------------------------===//

// A memref in the scratchpad. The predicate reads the memory space attribute
// and asks whether it is `#npu.scratchpad`, which is the attribute the `npu`
// dialect defines. Reusing it rather than defining a second one is what keeps
// `memref<4x4xf32, #npu.scratchpad>` meaning the same thing on both sides of
// the lowering.
class NPUISA_MemRefInSpace<string space, list<Type> allowedTypes>
    : Type<And<[
        MemRefOf<allowedTypes>.predicate,
        HasStaticShapePred,
        CPred<"::llvm::isa<::mlir::npu::" # space # "Attr>("
              "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace())">
      ]>,
      "a statically shaped memref in the " # space # " memory space">;

def NPUISA_ScratchpadF32 : NPUISA_MemRefInSpace<"Scratchpad", [F32]>;
def NPUISA_ScratchpadAny : NPUISA_MemRefInSpace<"Scratchpad", [F32, I8]>;
def NPUISA_DramF32 : NPUISA_MemRefInSpace<"Dram", [F32]>;
def NPUISA_DramAny : NPUISA_MemRefInSpace<"Dram", [F32, I8]>;

// Either space. The DMA operations are written against the specific spaces,
// because the direction is the whole content of the operation; this one exists
// for the constant, whose result is DRAM, and for nothing else so far.
def NPUISA_AnyMemRef : Type<And<[
        MemRefOf<[F32, I8]>.predicate,
        HasStaticShapePred,
        CPred<"::llvm::isa<::mlir::npu::ScratchpadAttr, ::mlir::npu::DramAttr>("
              "::llvm::cast<::mlir::MemRefType>($_self).getMemorySpace())">
      ]>,
      "a statically shaped memref in the scratchpad or dram memory space">;

#endif // NPUISA_TYPES
