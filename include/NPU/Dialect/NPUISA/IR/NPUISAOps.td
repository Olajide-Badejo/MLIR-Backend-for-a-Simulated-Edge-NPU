//===- NPUISAOps.td - npuisa dialect operations ------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The instruction set of Section 5.4, as operations. Two of the sixteen
// opcodes, `QUANT` and `DEQUANT`, are deliberately absent here exactly as they
// are absent from the `npu` dialect: they arrive with their integer kernels and
// their converters together in the quantization phase, and an operation nothing
// can lower, encode, or simulate is the thing this project's second law
// forbids. `NOP` and `HALT` are absent for a different reason, and it is worth
// separating: they are not compiler concepts at all. Nothing in the IR ever
// wants to say "do nothing"; the encoder emits `HALT` at the end of a program
// and `NOP` if it ever needs padding, and both are properties of the encoding
// rather than of the instruction stream a pass manipulates.
//
// Every compute operation below is destination passing on memrefs and has NO
// results. That is the difference from the `npu` side and it is the point of
// the level: a machine instruction writes a buffer, it does not produce a
// value, and modelling the write as a result would mean every pass below here
// reasoning about a value that has no address.
//
//===----------------------------------------------------------------------===//

#ifndef NPUISA_OPS
#define NPUISA_OPS

include "NPU/Dialect/NPUISA/IR/NPUISADialect.td"
include "NPU/Dialect/NPUISA/IR/NPUISATypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// The shared class for the destination passing compute instructions.
//===----------------------------------------------------------------------===//

// Every compute instruction reads its `ins` and writes its single trailing
// `outs`, and says so through two interfaces at once.
//
// DestinationStyleOpInterface gives the operand partition: the destination is
// the last operand, always exactly one, which is the same trailing range of one
// the `npu` dialect uses and for the same reason. It is the simplest consecutive
// range that cannot be got wrong.
//
// The memory effects are declared rather than inferred, and they are the load
// bearing half at this level. `Pure` would be a lie here in a way it is not on
// the tensor side: these operations write memory, and an operation that reports
// itself free of effects is an operation the canonicalizer will happily delete
// because nothing reads its result, since it has no result. Section 8 asks for
// MemoryEffectOpInterface on these operations precisely so that the write is
// visible, and `unittests/Dialect/NPUISA/InterfaceTest.cpp` asserts that no
// compute operation reports an empty effect list.
//
// The effects are attached per operand through the ODS `Arg` mechanism in each
// operation's argument list rather than declared here as a blanket
// `MemoryEffects<[MemRead, MemWrite]>`, because a blanket declaration says the
// operation touches memory somewhere and the overlap rule of Section 8 needs to
// know *which* memref is written. `DeclareOpInterfaceMethods` is not used: the
// generated `getEffects` from the per operand `Arg` effects is exactly right
// and hand writing it would be reimplementing generated code.
class NPUISA_ComputeOp<string mnemonic, list<Trait> traits = []>
    : NPUISA_Op<mnemonic, traits # [DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    /// The destination is the last operand, always exactly one.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), getOperation()->getNumOperands() - 1, 1);
    }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npuisa.const
//===----------------------------------------------------------------------===//

def NPUISA_ConstOp : NPUISA_Op<"const", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "A constant buffer in DRAM.";
  let description = [{
    ```mlir
    %w = npuisa.const dense<1.000000e+00> : tensor<8x3x3x3xf32>
       -> memref<8x3x3x3xf32, #npu.dram>
    ```

    What `npu.constant` lowers to: the weights and the biases the model carried,
    resident in DRAM before the program starts. The result is **always** a DRAM
    memref, which is Section 8's rule that function arguments and `npuisa.const`
    results are the two things that live off chip. A constant in the scratchpad
    would be a constant the allocator had to find room for and the loader had to
    write, and the machine this compiles for loads its constants with a DMA like
    everything else.

    This is the one operation in this dialect with a result, and it is not a
    contradiction of the destination passing rule but the reason for it: a
    constant is where a buffer comes from, not a transformation of one. It has
    no `outs` because there is nothing to write into.

    The value attribute's shape and element type equal the result's, and the
    verifier says so rather than the two being allowed to drift.

    Both types are printed, and unlike `npu.constant` that is unavoidable here
    rather than a redundancy: a dense elements attribute carries a *tensor*
    type, the result is a *memref*, and neither can be written as the other. So
    the file says both and the verifier holds them equal, where the tensor level
    could print one and infer the other.

    It is **not** `ConstantLike` and it is **not** `Pure`. A `ConstantLike`
    operation is one the folder may materialise, duplicate, and sink freely, and
    duplicating a DRAM constant is duplicating a region of the address map that
    the encoder has already assigned. There is exactly one copy of each constant
    in a program, so it is declared with an allocation effect on its result and
    left where the lowering put it.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs Res<NPUISA_DramAny, "",
                          [MemAlloc<DefaultResource>]>:$result);

  // `->` rather than `:` between the attribute and the result type. The
  // attribute prints as `dense<...> : tensor<2x2xf32>`, so a `:` after it would
  // be the second colon on the line and the parser reads the first one as the
  // start of the result type and then fails on what follows.
  let assemblyFormat = "attr-dict $value `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The synchronous DMA operations.
//===----------------------------------------------------------------------===//

// The two directions share every rule except which side is which, so they share
// a class and differ only in their operand types. Section 8's boundary
// invariant is about where these appear in a program and is asserted by a lit
// test at the one point in the pipeline where it holds; nothing in this file
// can express it, and pretending otherwise with a verifier that fired at the
// wrong time would be worse than the honest split.
class NPUISA_DmaOp<string mnemonic, Type sourceType, Type destType>
    : NPUISA_Op<mnemonic> {
  let arguments = (ins
      Arg<sourceType, "the buffer read from", [MemRead<DefaultResource>]>:$source,
      Arg<destType, "the buffer written to", [MemWrite<DefaultResource>]>:$dest
  );

  let assemblyFormat = [{
      $source `,` $dest attr-dict `:` type($source) `to` type($dest)
  }];
  let hasVerifier = 1;
}

def NPUISA_DmaLoadOp : NPUISA_DmaOp<"dma_load", NPUISA_DramAny,
                                    NPUISA_ScratchpadAny> {
  let summary = "Copies a buffer from DRAM into the scratchpad.";
  let description = [{
    ```mlir
    npuisa.dma_load %src, %dst
      : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
    ```

    The direction is in the operation name and in the operand types both, and
    the redundancy is deliberate: the name is what a reader sees and the types
    are what the verifier can check, and a `dma_load` whose source was a
    scratchpad memref fails to parse rather than fails to verify.

    Shapes and element types agree exactly. The DMA of this machine moves bytes
    between two memories and does not change a layout on the way, which is why
    Section 12's relayouting DMA is written down as a future extension rather
    than assumed available: today, a layout change costs a `npuisa.transpose`
    and a pass over the data.

    Section 8 scopes the boundary invariant that governs where this operation is
    allowed to appear: immediately after lowering and before allocation, one
    load per DRAM value entering the scratchpad and one store per returned
    value, asserted by `test/Dialect/NPUISA/dma-boundaries.mlir` at exactly that
    point. Later passes add DMA, and exactly three are permitted to: the
    lowering itself, spilling, and tiling. A fourth producer is a defect rather
    than a style question, because it is DRAM traffic that would move published
    numbers.
  }];
}

def NPUISA_DmaStoreOp : NPUISA_DmaOp<"dma_store", NPUISA_ScratchpadAny,
                                     NPUISA_DramAny> {
  let summary = "Copies a buffer from the scratchpad back to DRAM.";
  let description = [{
    ```mlir
    npuisa.dma_store %src, %dst
      : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
    ```

    The mirror of `npuisa.dma_load`, with the same shape and element type rules
    and the same scoped boundary invariant.
  }];
}

//===----------------------------------------------------------------------===//
// The asynchronous DMA operations and the await.
//===----------------------------------------------------------------------===//

// The asynchronous pair carries the same operands as the synchronous one plus a
// token result, and the same verifier rules plus the four Section 8 adds. The
// four live in a shared C++ helper rather than being written twice, because two
// copies of a race condition check is one copy too many.
class NPUISA_DmaAsyncOp<string mnemonic, Type sourceType, Type destType>
    : NPUISA_Op<mnemonic> {
  let arguments = (ins
      Arg<sourceType, "the buffer read from", [MemRead<DefaultResource>]>:$source,
      Arg<destType, "the buffer written to", [MemWrite<DefaultResource>]>:$dest
  );
  let results = (outs NPUISA_TokenType:$token);

  let assemblyFormat = [{
      $source `,` $dest attr-dict `:` type($source) `to` type($dest)
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def NPUISA_DmaLoadAsyncOp : NPUISA_DmaAsyncOp<"dma_load_async", NPUISA_DramAny,
                                              NPUISA_ScratchpadAny> {
  let summary = "Starts a DRAM to scratchpad transfer and returns its token.";
  let description = [{
    ```mlir
    %t = npuisa.dma_load_async %src, %dst
       : memref<16x16xf32, #npu.dram> to memref<16x16xf32, #npu.scratchpad>
    npuisa.matmul ins(%a, %b : ...) outs(%c : ...)
    npuisa.await %t
    ```

    The transfer starts here and is complete at the `await`. In between, the
    instruction stream runs; the point of the form is that the two matmuls'
    worth of time a load costs can be spent computing instead of waiting.

    **The destination is being written by hardware for the whole of that window,
    and that is the race the token exists to prevent.** Section 8's rules, all
    of which this operation's verifier enforces:

    1. The token has exactly one use and that use is an `npuisa.await`.
    2. The `await` is in the same block as its producer and comes after it.
    3. Source and destination shapes and element types agree, as in the
       synchronous form.
    4. No operation between the async operation and its `await` may access
       memory overlapping the destination.

    Rule 4 is decided on `MemoryEffectOpInterface` effects **plus an explicit
    overlap test on view offsets and extents**, and never by scanning for the
    same SSA value. Once the allocator has materialised scratchpad offsets as
    `memref.view` operations over one flat buffer, two different SSA values are
    routinely two halves of the same memory, and a partially overlapping view is
    a real race that an identity check does not see. If the offsets are not
    static the async form is refused with a diagnostic, because "I cannot prove
    these overlap" and "these are disjoint" are different answers and only one
    of them is safe.

    An asynchronous operation whose `await` is the very next operation
    canonicalizes back to the synchronous form, so a scheduling pass that looked
    for something to overlap the transfer with and found nothing leaves no
    residue behind.

    It encodes to `DMA_LOAD`. There is no separate asynchronous opcode: the
    machine's DMA engine is asynchronous and the synchronous instruction is the
    engine start followed by the wait, so the dialect level distinction is about
    where the wait goes and disappears at encoding.
  }];
}

def NPUISA_DmaStoreAsyncOp : NPUISA_DmaAsyncOp<"dma_store_async",
                                               NPUISA_ScratchpadAny,
                                               NPUISA_DramAny> {
  let summary = "Starts a scratchpad to DRAM transfer and returns its token.";
  let description = [{
    ```mlir
    %t = npuisa.dma_store_async %src, %dst
       : memref<16x16xf32, #npu.scratchpad> to memref<16x16xf32, #npu.dram>
    npuisa.await %t
    ```

    The mirror of `npuisa.dma_load_async`, with the same four rules. The
    destination is the DRAM buffer here, so it is the DRAM buffer the overlap
    rule protects; the scratchpad source is read for the length of the window
    too, but a second read of a buffer being read is not a race and the rule
    does not pretend otherwise.

    It encodes to `DMA_STORE`.
  }];
}

def NPUISA_AwaitOp : NPUISA_Op<"await"> {
  let summary = "Waits for the transfer a token names to complete.";
  let description = [{
    ```mlir
    npuisa.await %t
    ```

    After this operation the destination buffer of the transfer holds the
    transferred data and may be read. Before it, it holds anything at all.

    The `await` carries no memory effect of its own, and that is a decision
    worth stating rather than an omission. What it does is establish an ordering
    on an effect that the asynchronous operation already declared, and giving it
    a duplicate write effect on a buffer it does not name as an operand would
    mean inventing an operand for the purpose. The verifier on the asynchronous
    side owns the whole rule, which keeps it in one place; the effect based
    scan runs from the producer to this operation and stops.

    A dangling `await`, whose token had no producer, cannot be written: the
    token is an SSA value and an operand with no definition is not parseable
    IR. That is one more thing the type buys over matching by position.
  }];

  let arguments = (ins NPUISA_TokenType:$token);
  let assemblyFormat = "$token attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npuisa.matmul
//===----------------------------------------------------------------------===//

def NPUISA_MatMulOp : NPUISA_ComputeOp<"matmul"> {
  let summary = "A rank 2 by rank 2 matrix multiplication with an optional bias.";
  let description = [{
    ```mlir
    npuisa.matmul ins(%a, %b : memref<4x16xf32, #npu.scratchpad>,
                               memref<16x10xf32, #npu.scratchpad>)
                  outs(%d : memref<4x10xf32, #npu.scratchpad>)
    ```

    `(M, K)` by `(K, N)` writes `(M, N)`, the same arithmetic `npu.matmul`
    verifies on tensors. The optional bias is length N.

    Every operand is in the scratchpad. That is the memory model rather than a
    restriction of this operation: the compute units of this machine address the
    scratchpad and nothing else, and an operand in DRAM is an operand that a
    `dma_load` has not been emitted for yet. Writing the rule into the operand
    types means that mistake is a parse failure at the point it is made rather
    than an addressing fault in the simulator three phases later.
  }];

  let arguments = (ins
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$lhs,
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$rhs,
      Arg<Optional<NPUISA_ScratchpadF32>, "", [MemRead<DefaultResource>]>:$bias,
      Arg<NPUISA_ScratchpadF32, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $lhs `,` $rhs (`,` $bias^)? `:`
          type($lhs) `,` type($rhs) (`,` type($bias)^)? `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// npuisa.conv2d
//===----------------------------------------------------------------------===//

def NPUISA_Conv2DOp : NPUISA_ComputeOp<"conv2d"> {
  let summary = "A two dimensional grouped convolution with an optional bias.";
  let description = [{
    ```mlir
    npuisa.conv2d ins(%x, %w : memref<1x3x8x8xf32, #npu.scratchpad>,
                               memref<8x3x3x3xf32, #npu.scratchpad>)
                  outs(%d : memref<1x8x8x8xf32, #npu.scratchpad>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
    ```

    The same convolution `npu.conv2d` describes, on buffers. `pads` is four
    entries in ONNX order, `padTop`, `padLeft`, `padBottom`, `padRight`;
    `strides` and `dilations` are two each, height then width; `group` divides
    both channel counts.

    The output extent arithmetic is verified against the destination here rather
    than inferred, because there is no result type to infer: the destination is
    an allocated buffer and the question is whether the convolution the
    attributes describe fits in it. That is the same windowed formula the `npu`
    dialect uses, called from the same helper, which is what stops the two
    levels from ever disagreeing about a shape across the lowering.
  }];

  let arguments = (ins
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$input,
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$filter,
      Arg<Optional<NPUISA_ScratchpadF32>, "", [MemRead<DefaultResource>]>:$bias,
      DenseI64ArrayAttr:$strides,
      DenseI64ArrayAttr:$pads,
      DenseI64ArrayAttr:$dilations,
      DefaultValuedAttr<I64Attr, "1">:$group,
      Arg<NPUISA_ScratchpadF32, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $input `,` $filter (`,` $bias^)? `:`
          type($input) `,` type($filter) (`,` type($bias)^)? `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// The elementwise instructions.
//===----------------------------------------------------------------------===//

class NPUISA_ElementwiseBinaryOp<string mnemonic>
    : NPUISA_ComputeOp<mnemonic> {
  let arguments = (ins
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$lhs,
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$rhs,
      Arg<NPUISA_ScratchpadF32, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

def NPUISA_AddOp : NPUISA_ElementwiseBinaryOp<"add"> {
  let summary = "Elementwise addition.";
  let description = [{
    ```mlir
    npuisa.add ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                            memref<1x8x4x4xf32, #npu.scratchpad>)
               outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    ```

    Both operands have the destination's shape exactly. No broadcasting, for the
    reason `npu.add` gives: the frontend materialises every broadcast it accepts
    and refuses the rest by name, so nothing below the frontend has to.
  }];
}

def NPUISA_MulOp : NPUISA_ElementwiseBinaryOp<"mul"> {
  let summary = "Elementwise multiplication.";
  let description = [{
    ```mlir
    npuisa.mul ins(%a, %b : memref<1x8x4x4xf32, #npu.scratchpad>,
                            memref<1x8x4x4xf32, #npu.scratchpad>)
               outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    ```

    Same shape rules as `npuisa.add`.
  }];
}

def NPUISA_ReluOp : NPUISA_ComputeOp<"relu"> {
  let summary = "The rectified linear unit, elementwise.";
  let description = [{
    ```mlir
    npuisa.relu ins(%x : memref<1x8x4x4xf32, #npu.scratchpad>)
                outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
    ```

    `max(x, 0)` elementwise, into a destination of the input's shape.

    Nothing here says the input and the destination may not be the same buffer.
    An in place relu is a legal program and a useful one, it is what the
    allocator produces when it reuses a dead interval, and the memory effects
    say read and write on the two operands so a pass that needs to know can ask.
  }];

  let arguments = (ins
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$input,
      Arg<NPUISA_ScratchpadF32, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// The pooling instructions.
//===----------------------------------------------------------------------===//

class NPUISA_PoolOp<string mnemonic> : NPUISA_ComputeOp<mnemonic> {
  let arguments = (ins
      Arg<NPUISA_ScratchpadF32, "", [MemRead<DefaultResource>]>:$input,
      DenseI64ArrayAttr:$kernel,
      DenseI64ArrayAttr:$strides,
      DenseI64ArrayAttr:$pads,
      DenseI64ArrayAttr:$dilations,
      DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
      Arg<NPUISA_ScratchpadF32, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

def NPUISA_PoolMaxOp : NPUISA_PoolOp<"pool_max"> {
  let summary = "Two dimensional max pooling.";
  let description = [{
    ```mlir
    npuisa.pool_max ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                    outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                    {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                     pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
    ```

    The maximum over each window, into a destination whose extent the same
    shared windowed arithmetic verifies. The opcode is `POOL_MAX`, which is why
    the mnemonic is `pool_max` here where the `npu` level calls the same
    operation `max_pool2d`: the tensor level takes its name from ONNX's
    `MaxPool` and the instruction level takes its name from the instruction set
    of Section 5.4, and the lowering is where the two names meet. Naming them
    identically would hide which level a diagnostic came from.
  }];
}

def NPUISA_PoolAvgOp : NPUISA_PoolOp<"pool_avg"> {
  let summary = "Two dimensional average pooling.";
  let description = [{
    ```mlir
    npuisa.pool_avg ins(%x : memref<1x8x8x8xf32, #npu.scratchpad>)
                    outs(%d : memref<1x8x4x4xf32, #npu.scratchpad>)
                    {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                     pads = array<i64: 0, 0, 0, 0>,
                     dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
    ```

    The mean over each window, dividing by the number of elements that actually
    contributed rather than by the window area, which is ONNX's
    `count_include_pad = 0` and the only behaviour this project implements.
  }];
}

//===----------------------------------------------------------------------===//
// The shape instructions.
//===----------------------------------------------------------------------===//

def NPUISA_ReshapeOp : NPUISA_ComputeOp<"reshape"> {
  let summary = "Copies a buffer under a different set of extents.";
  let description = [{
    ```mlir
    npuisa.reshape ins(%x : memref<4x8x2x2xf32, #npu.scratchpad>)
                   outs(%d : memref<4x32xf32, #npu.scratchpad>)
    ```

    Element counts and element types agree; the extents need not.

    **This one has a destination where `npu.reshape` has none, and the
    difference is real rather than an inconsistency.** At the tensor level a
    reshape is a retyping and moves nothing, which is why that operation has no
    destination and implements no destination passing interface. At the
    instruction level there is no retyping available: a buffer has an address
    and a shape, and producing the same bytes under different extents means
    either a copy or an aliasing view. This dialect models it as the copy,
    because the encoder has to emit a `RESHAPE` instruction that the simulator
    executes, and an instruction with no destination has nowhere to write.

    The allocator is what turns the copy back into free when it can, by giving
    the source and the destination the same offset, and that is a decision it
    can only make once offsets exist.
  }];

  let arguments = (ins
      Arg<NPUISA_ScratchpadAny, "", [MemRead<DefaultResource>]>:$input,
      Arg<NPUISA_ScratchpadAny, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

def NPUISA_TransposeOp : NPUISA_ComputeOp<"transpose"> {
  let summary = "Permutes a buffer's dimensions into the destination.";
  let description = [{
    ```mlir
    npuisa.transpose ins(%x : memref<1x3x8x8xf32, #npu.scratchpad>)
                     outs(%d : memref<1x8x8x3xf32, #npu.scratchpad>)
                     {permutation = array<i64: 0, 2, 3, 1>}
    ```

    `permutation` is a permutation of exactly the destination rank: every index
    from zero to rank minus one appears exactly once, and destination extent `i`
    equals input extent `permutation[i]`.

    Unlike the reshape this one genuinely moves data whatever the allocator
    does, and it is the only mechanism this machine has for changing a layout.
    That is the whole reason a relayouting DMA is a named future extension in
    Section 12 rather than something a pass may assume it can use.
  }];

  let arguments = (ins
      Arg<NPUISA_ScratchpadAny, "", [MemRead<DefaultResource>]>:$input,
      DenseI64ArrayAttr:$permutation,
      Arg<NPUISA_ScratchpadAny, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

def NPUISA_ConcatOp : NPUISA_ComputeOp<"concat"> {
  let summary = "Concatenates buffers along one axis into the destination.";
  let description = [{
    ```mlir
    npuisa.concat ins(%a, %b : memref<1x4x8x8xf32, #npu.scratchpad>,
                               memref<1x6x8x8xf32, #npu.scratchpad>)
                  outs(%d : memref<1x10x8x8xf32, #npu.scratchpad>)
                  {axis = 1 : i64}
    ```

    Every input has the destination's rank, the extents agree on every axis
    except the concatenation axis, and along that axis they sum to the
    destination extent. The axis is non negative and less than the rank; a
    negative axis is an ONNX convention the frontend normalises, and carrying it
    this far down would mean every consumer normalising it again.

    At least one input is required. A concatenation of nothing has no
    destination extent it could satisfy, so it is rejected rather than treated
    as a zero fill.
  }];

  let arguments = (ins
      Arg<Variadic<NPUISA_ScratchpadAny>, "", [MemRead<DefaultResource>]>:$inputs,
      I64Attr:$axis,
      Arg<NPUISA_ScratchpadAny, "", [MemWrite<DefaultResource>]>:$destination
  );

  let assemblyFormat = [{
      `ins` `(` $inputs `:` type($inputs) `)`
      `outs` `(` $destination `:` type($destination) `)` attr-dict
  }];
}

#endif // NPUISA_OPS
