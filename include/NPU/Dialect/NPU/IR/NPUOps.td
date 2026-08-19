//===- NPUOps.td - npu dialect operations ------------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The operator set of the `npu` dialect. Two operations named in the design,
// `quantize` and `dequantize`, are deliberately absent: they arrive with their
// converters, their integer kernels, and their models together in the
// quantization phase, and adding them here would be adding an operator that
// nothing can lower, encode, simulate, or import, which is exactly what law 2
// of this project forbids.
//
// Every operation below carries a classification line in its description,
// reading either
//
//     Reachability: imported computation.
//   or
//     Reachability: structural.
//
// `scripts/check-reachability.py` reads that line out of the generated dialect
// reference rather than out of a table it keeps itself. The point of putting it
// in the description is that reclassifying an operation to make the check pass
// cannot be done quietly: it changes the dialect source and it changes
// docs/DIALECT_REFERENCE.md, and both are reviewed.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS
#define NPU_OPS

include "NPU/Dialect/NPU/IR/NPUDialect.td"
include "NPU/Dialect/NPU/IR/NPUTypes.td"
include "NPU/Dialect/NPU/IR/NPUAttrs.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Shared class for the destination passing compute operations.
//===----------------------------------------------------------------------===//

// Every compute operation in this dialect takes a destination tensor as its
// last operand and implements DestinationStyleOpInterface over it. The
// destination is what makes the tiling interface meaningful: a tiled loop needs
// somewhere to insert the tile it computed, and without a destination operand
// there is no such value to thread through the loop as an iteration argument.
//
// The destination is always the single trailing operand, which is what
// `getDpsInitsMutable` needs: the interface requires the inits to be a
// consecutive range, and a trailing range of one is the simplest one that can
// never be got wrong.
class NPU_ComputeOp<string mnemonic, list<Trait> traits = []>
    : NPU_Op<mnemonic, traits # [Pure, DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    /// The destination is the last operand, always exactly one.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(
          getOperation(), getOperation()->getNumOperands() - 1, 1);
    }
  }];
}

//===----------------------------------------------------------------------===//
// npu.constant
//===----------------------------------------------------------------------===//

def NPU_ConstantOp : NPU_Op<"constant",
    [Pure, ConstantLike, DeclareOpInterfaceMethods<InferTypeOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "A constant tensor.";
  let description = [{
    Reachability: imported computation.

    Holds the weights, biases, and any other initializer the imported model
    carried, as a dense elements attribute whose type is the result type.

    ```mlir
    %w = npu.constant dense<1.000000e+00> : tensor<8x3x3x3xf32>
    ```

    The result type is not printed separately: a dense elements attribute
    carries its own type, and the verifier requires that type to equal the
    result type, so printing it twice would give one fact two spellings and a
    parse in which they disagreed would have to be resolved somehow.

    This operation has no destination operand and implements no destination
    passing interface, on purpose: it reads nothing and writes nowhere, it is
    the source of a value rather than a transformation of one, and giving it a
    destination would mean the importer materialising an empty tensor for every
    weight in the model for no purpose.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs NPU_AnyTensor:$result);

  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// npu.conv2d
//===----------------------------------------------------------------------===//

def NPU_Conv2DOp : NPU_ComputeOp<"conv2d",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "A two dimensional grouped convolution with an optional bias.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.conv2d ins(%x, %w, %b : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>,
                                     tensor<8xf32>)
                    outs(%d : tensor<1x8x8x8xf32>)
                    {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                     dilations = array<i64: 1, 1>, group = 1 : i64}
         -> tensor<1x8x8x8xf32>
    ```

    Input is rank 4 and laid out per its encoding, defaulting to NCHW. The
    filter is `(outputChannels, inputChannels / group, kernelHeight,
    kernelWidth)` and is not layout tagged: a filter has no batch dimension and
    no NHWC reading, so tagging it would be inventing a meaning.

    `pads` is four entries in ONNX order, `padTop`, `padLeft`, `padBottom`,
    `padRight`. `strides` and `dilations` are two entries each, height then
    width.

    The output spatial extent is the shared windowed arithmetic of this
    dialect, with `ceil_mode` fixed at 0. Convolution has no ceil mode in ONNX
    and this operation does not invent one; the same helper computes both pools
    with the mode they do have, which is what stops the four windowed
    operations from ever disagreeing on the same shape.

    The bias is optional, and when present its length is the output channel
    count. `group` divides both the input and the output channel counts.
  }];

  let arguments = (ins
      NPU_FloatTensor:$input,
      NPU_FloatTensor:$filter,
      Optional<NPU_FloatTensor>:$bias,
      DenseI64ArrayAttr:$strides,
      DenseI64ArrayAttr:$pads,
      DenseI64ArrayAttr:$dilations,
      DefaultValuedAttr<I64Attr, "1">:$group,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $input `,` $filter (`,` $bias^)? `:`
          type($input) `,` type($filter) (`,` type($bias)^)? `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.matmul
//===----------------------------------------------------------------------===//

def NPU_MatMulOp : NPU_ComputeOp<"matmul",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "A rank 2 by rank 2 matrix multiplication with an optional bias.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.matmul ins(%a, %b : tensor<4x16xf32>, tensor<16x10xf32>)
                    outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
    ```

    `(M, K)` by `(K, N)` produces `(M, N)`. M is the batch dimension of a fully
    connected layer and is never assumed to be 1: a flatten that produced
    `(1, N * features)` instead of `(N, features)` would compute one enormous
    row and still typecheck, which is why the rank is pinned at 2 by 2 here and
    asserted directly in the frontend rather than left implicit.

    The optional bias is length N, the output column count.
  }];

  let arguments = (ins
      NPU_FloatTensor:$lhs,
      NPU_FloatTensor:$rhs,
      Optional<NPU_FloatTensor>:$bias,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $lhs `,` $rhs (`,` $bias^)? `:`
          type($lhs) `,` type($rhs) (`,` type($bias)^)? `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The elementwise operations.
//===----------------------------------------------------------------------===//

class NPU_ElementwiseBinaryOp<string mnemonic>
    : NPU_ComputeOp<mnemonic> {
  let arguments = (ins
      NPU_FloatTensor:$lhs,
      NPU_FloatTensor:$rhs,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

def NPU_AddOp : NPU_ElementwiseBinaryOp<"add"> {
  let summary = "Elementwise addition.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.add ins(%a, %b : tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>)
                 outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
    ```

    Both operands have the result shape exactly. This operation does **not**
    broadcast, and the omission is a decision rather than a gap: the frontend
    materialises every broadcast it can at import time and rejects the rest by
    name, with one carve out for the rank 1 channel shaped addend that a
    separate bias add produces, and that carve out is expressed as a bias
    operand on the consuming convolution rather than as a broadcasting add.
    Broadcasting here as well would give the same fact two representations and
    let the bias fusion pass silently miss one of them.
  }];
}

def NPU_MulOp : NPU_ElementwiseBinaryOp<"mul"> {
  let summary = "Elementwise multiplication.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.mul ins(%a, %b : tensor<1x8x4x4xf32>, tensor<1x8x4x4xf32>)
                 outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
    ```

    Same shape rules as `npu.add`, and the same reason for them.
  }];
}

def NPU_ReluOp : NPU_ComputeOp<"relu"> {
  let summary = "The rectified linear unit, elementwise.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.relu ins(%x : tensor<1x8x4x4xf32>)
                  outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
    ```

    `max(x, 0)` elementwise. A `Clip` node with a lower bound of 0 and an absent
    or infinite upper bound imports to this; any other bounds are refused by
    name at import rather than approximated here.
  }];

  let arguments = (ins
      NPU_FloatTensor:$input,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The pooling operations.
//===----------------------------------------------------------------------===//

class NPU_PoolOp<string mnemonic>
    : NPU_ComputeOp<mnemonic, [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
      NPU_FloatTensor:$input,
      DenseI64ArrayAttr:$kernel,
      DenseI64ArrayAttr:$strides,
      DenseI64ArrayAttr:$pads,
      DenseI64ArrayAttr:$dilations,
      DefaultValuedAttr<I64Attr, "0">:$ceil_mode,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

def NPU_MaxPool2DOp : NPU_PoolOp<"max_pool2d"> {
  let summary = "Two dimensional max pooling.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                        outs(%d : tensor<1x8x4x4xf32>)
                        {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                         pads = array<i64: 0, 0, 0, 0>,
                         dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
         -> tensor<1x8x4x4xf32>
    ```

    The maximum over each window. The output extent is the shared windowed
    arithmetic of this dialect, resolved against the opset 19 specification
    rather than the opset 11 one, including the rule that a window whose first
    element would start inside the right padded region is dropped. That rule
    only ever fires with `ceil_mode = 1` and it changes the shape rather than
    the values, which is why it is implemented explicitly and tested with the
    arithmetic written out.
  }];
}

def NPU_AvgPool2DOp : NPU_PoolOp<"avg_pool2d"> {
  let summary = "Two dimensional average pooling.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.avg_pool2d ins(%x : tensor<1x8x8x8xf32>)
                        outs(%d : tensor<1x8x4x4xf32>)
                        {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                         pads = array<i64: 0, 0, 0, 0>,
                         dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
         -> tensor<1x8x4x4xf32>
    ```

    The mean over each window, dividing by the number of elements that actually
    contributed rather than by the window area. That is ONNX's
    `count_include_pad = 0` behaviour, it is the only behaviour this project
    implements, and the frontend rejects `count_include_pad = 1` by name rather
    than ignoring the attribute.

    `dilations` arrived on `AveragePool` at opset 19 and the arithmetic here
    carries the term, so the two pools and the convolution share one formula
    whatever the frontend decides to accept.
  }];
}

//===----------------------------------------------------------------------===//
// The shape operations.
//===----------------------------------------------------------------------===//

def NPU_ReshapeOp : NPU_Op<"reshape", [Pure]> {
  let summary = "Reinterprets a tensor's extents without moving data.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.reshape %x : tensor<4x8x2x2xf32> to tensor<4x32xf32>
    ```

    The element counts of operand and result are equal and the element types
    match. Nothing moves: below this level a reshape is a view, and it is here
    as a named operation rather than as an implicit type change so that the
    element count check happens once, in a verifier, rather than in every pass
    that would otherwise have to trust it.

    This operation takes **no destination operand** and implements no
    destination passing interface. It is not a compute operation: there is no
    tile of a reshape that a tiling pass could compute independently, because a
    reshape crossing a tile boundary is not expressible as a per tile reshape.
    Giving it a destination would be claiming an ability it does not have.
  }];

  let arguments = (ins NPU_AnyTensor:$input);
  let results = (outs NPU_AnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
}

def NPU_TransposeOp : NPU_ComputeOp<"transpose"> {
  let summary = "Permutes a tensor's dimensions.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                       outs(%d : tensor<1x8x8x3xf32>)
                       {permutation = array<i64: 0, 2, 3, 1>}
         -> tensor<1x8x8x3xf32>
    ```

    `permutation` is a permutation of exactly the result rank: every index from
    zero to rank minus one appears exactly once. Result extent `i` equals input
    extent `permutation[i]`.

    Unlike `npu.reshape` this one does move data, it costs a full pass over the
    tensor, and it is the only mechanism the layout assignment pass has for
    changing a layout. That is the whole reason a relayouting DMA is written
    down as a future extension elsewhere rather than assumed available here.
  }];

  let arguments = (ins
      NPU_AnyTensor:$input,
      DenseI64ArrayAttr:$permutation,
      NPU_AnyTensor:$destination
  );
  let results = (outs NPU_AnyTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $input `:` type($input) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

def NPU_ConcatOp : NPU_ComputeOp<"concat"> {
  let summary = "Concatenates tensors along one axis.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.concat ins(%a, %b : tensor<1x4x8x8xf32>, tensor<1x6x8x8xf32>)
                    outs(%d : tensor<1x10x8x8xf32>)
                    {axis = 1 : i64} -> tensor<1x10x8x8xf32>
    ```

    Every operand has the result's rank, the extents agree on every axis except
    the concatenation axis, and along that axis they sum to the result extent.
    The axis is non negative and less than the rank: a negative axis is an ONNX
    convention that the frontend normalises, and carrying it into the IR would
    mean every consumer normalising it again.
  }];

  let arguments = (ins
      Variadic<NPU_AnyTensor>:$inputs,
      I64Attr:$axis,
      NPU_AnyTensor:$destination
  );
  let results = (outs NPU_AnyTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $inputs `:` type($inputs) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.batch_norm
//===----------------------------------------------------------------------===//

def NPU_BatchNormOp : NPU_ComputeOp<"batch_norm"> {
  let summary = "Inference time batch normalization.";
  let description = [{
    Reachability: imported computation.

    ```mlir
    %r = npu.batch_norm ins(%x, %gamma, %beta, %mean, %var :
                            tensor<1x8x4x4xf32>, tensor<8xf32>, tensor<8xf32>,
                            tensor<8xf32>, tensor<8xf32>)
                        outs(%d : tensor<1x8x4x4xf32>)
                        {epsilon = 1.000000e-05 : f32} -> tensor<1x8x4x4xf32>
    ```

    `y = gamma * (x - mean) / sqrt(var + epsilon) + beta`, per channel, which is
    the inference form. The training form with five outputs is refused at
    import rather than here, because the earliest layer that can name the
    problem is the one that should.

    All four parameter tensors are rank 1 of length C, the channel extent of
    the input under its layout. A batch norm the folding pass did not fold is
    legal rather than a hard error: it decomposes at lowering into a multiply
    and an add, so an unfoldable one still compiles.
  }];

  let arguments = (ins
      NPU_FloatTensor:$input,
      NPU_FloatTensor:$gamma,
      NPU_FloatTensor:$beta,
      NPU_FloatTensor:$mean,
      NPU_FloatTensor:$variance,
      F32Attr:$epsilon,
      NPU_FloatTensor:$destination
  );
  let results = (outs NPU_FloatTensor:$result);

  let assemblyFormat = [{
      `ins` `(` $input `,` $gamma `,` $beta `,` $mean `,` $variance `:`
          type($input) `,` type($gamma) `,` type($beta) `,` type($mean) `,`
          type($variance) `)`
      `outs` `(` $destination `:` type($destination) `)`
      attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// npu.fused_op and npu.yield
//===----------------------------------------------------------------------===//

// SingleBlock rather than SingleBlockImplicitTerminator, and the difference is
// not cosmetic. The implicit terminator trait builds a terminator for an empty
// region by calling YieldOp::build with no operands, and npu.yield always
// carries the value it yields, so there is no such builder and there should not
// be: a fused region with an implicitly created yield would be a region
// yielding nothing while the operation has a result. The verifier below
// requires the yield explicitly instead.
def NPU_FusedOp : NPU_Op<"fused_op", [Pure, SingleBlock, IsolatedFromAbove]> {
  let summary = "A region holding a fused chain of npu operations.";
  let description = [{
    Reachability: structural.

    ```mlir
    %r = npu.fused_op ins(%x, %w, %d : tensor<1x3x8x8xf32>,
                                       tensor<8x3x3x3xf32>,
                                       tensor<1x8x8x8xf32>) {
    ^bb0(%a: tensor<1x3x8x8xf32>, %b: tensor<8x3x3x3xf32>,
         %e: tensor<1x8x8x8xf32>):
      %c = npu.conv2d ins(%a, %b : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                      outs(%e : tensor<1x8x8x8xf32>)
                      {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                       dilations = array<i64: 1, 1>, group = 1 : i64}
           -> tensor<1x8x8x8xf32>
      npu.yield %c : tensor<1x8x8x8xf32>
    } -> tensor<1x8x8x8xf32>
    ```

    This is a region rather than a two case enum or a family of fused operation
    names, and that is the design's most consequential representational choice
    at this level. Separate names produce a combinatorial explosion as the
    fusible set grows; an enum cannot express what fusion actually does, which
    is to keep an intermediate value in the scratchpad instead of writing it to
    DRAM. A region can express exactly that, and it can grow without a
    migration.

    The verifier holds four rules: the block argument count and types match the
    operands one for one, the yielded type matches the result type, the region
    contains only `npu` dialect operations, and the region is not empty.

    It is classified as structural because no ONNX node imports to it. A pass
    creates it, out of operations that were themselves imported, so requiring
    it to be importable would be requiring an ONNX node that does not exist.
  }];

  let arguments = (ins Variadic<NPU_AnyTensor>:$inputs);
  let results = (outs NPU_AnyTensor:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
      `ins` `(` $inputs `:` type($inputs) `)` $body attr-dict `->` type($result)
  }];
  let hasVerifier = 1;
}

def NPU_YieldOp : NPU_Op<"yield",
    [Pure, ReturnLike, Terminator, HasParent<"::mlir::npu::FusedOp">]> {
  let summary = "The terminator of an npu.fused_op region.";
  let description = [{
    Reachability: structural.

    ```mlir
    npu.yield %v : tensor<1x8x8x8xf32>
    ```

    Returns the region's value to the enclosing `npu.fused_op`. It is a region
    terminator and corresponds to no ONNX node, so importability is a
    requirement it can never meet and is not held to; the other requirements it
    meets by being eliminated at lowering, when the region is flattened.
  }];

  let arguments = (ins NPU_AnyTensor:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // NPU_OPS
