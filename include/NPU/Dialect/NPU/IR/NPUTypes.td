//===- NPUTypes.td - npu dialect type constraints ----------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This dialect defines no new types. It defines constraints over the builtin
// `tensor` type, which is a deliberate choice: a custom tensor type would cut
// this dialect off from every piece of MLIR infrastructure that already knows
// what a `RankedTensorType` is, and it would buy nothing that an encoding
// attribute does not already buy.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_TYPES
#define NPU_TYPES

include "NPU/Dialect/NPU/IR/NPUDialect.td"
include "mlir/IR/BuiltinTypes.td"

// The floating point activation and weight type of this machine. A **static
// shape** is required, not merely a static rank, and it is required at the type
// level rather than by a verifier because nothing below this dialect can
// represent a dynamic extent.
//
// `StaticShapeTensorOf` rather than `RankedTensorOf`, and the difference is not
// cosmetic. `RankedTensorOf` accepts `tensor<?x4xf32>`: the rank is static and
// the extent is not. Under that spelling the paragraph above described a rule
// nothing enforced, and `npu.reshape` then read the element count of such a
// tensor through `getNumElements()`, which asserts on a dynamic shape rather
// than returning anything. So the gap did not produce a wrong answer, it
// produced an abort with an LLVM stack trace and no diagnostic at all, which is
// the one failure mode every verifier in this dialect is written to avoid. See
// docs/DEFECT_LOG.md D-0015.
//
// This is also the spelling the `npuisa` side has carried since P2, where every
// memref constraint ANDs `HasStaticShapePred` for the same stated reason. The
// two levels now say the same thing in the same place.
def NPU_FloatTensor : StaticShapeTensorOf<[F32]>;

// The quantized tensor type.
//
// The element type is signless `i8`, and it is interpreted as two's complement
// **signed**, range -128 to 127. MLIR's `i8` carries no signedness of its own,
// so the interpretation has to be written down somewhere and enforced
// everywhere: this comment is the definition, the quantization operations of a
// later phase carry the same statement in their descriptions, and the zero
// point verifier bounds check is the mechanical half of it.
def NPU_QuantTensor : StaticShapeTensorOf<[I8]>;

// Either element type. Used by the shape only operations, which move data
// without arithmetic and so do not care which of the two they are moving.
def NPU_AnyTensor : StaticShapeTensorOf<[F32, I8]>;

// A signless `i32` attribute read as two's complement signed.
//
// It exists because the zero point is genuinely negative on most tensors this
// project quantizes, and neither builtin spelling is right. `I32Attr` prints
// the value the dialect wants, `-3 : i32`, and hands C++ a `uint32_t` through
// `getZExtValue`, so a negative zero point arrives as 4294967293 and every
// bound check on it passes for the wrong reason. `SI32Attr` reads the sign
// correctly and prints `-3 : si32`, which is a signedness this dialect does not
// otherwise use: its i8 tensors are signless and read as signed by convention,
// stated in `NPU_QuantTensor` above, and an attribute that announced its
// signedness beside them would be two conventions in one operation.
//
// So the storage is the signless one and the accessor is the signed one, which
// is the same choice the tensor type makes, made at the attribute.
def NPU_SignedI32Attr : SignlessIntegerAttrBase<
    I32, "32-bit signless integer attribute, read as two's complement signed"> {
  let returnType = "int32_t";
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

#endif // NPU_TYPES
