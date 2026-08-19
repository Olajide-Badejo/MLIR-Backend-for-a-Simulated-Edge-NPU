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

// The floating point activation and weight type of this machine. Static rank
// is required; a dynamic dimension is refused at the type level rather than by
// a verifier, because nothing below this dialect can represent one.
def NPU_FloatTensor : RankedTensorOf<[F32]>;

// The quantized tensor type.
//
// The element type is signless `i8`, and it is interpreted as two's complement
// **signed**, range -128 to 127. MLIR's `i8` carries no signedness of its own,
// so the interpretation has to be written down somewhere and enforced
// everywhere: this comment is the definition, the quantization operations of a
// later phase carry the same statement in their descriptions, and the zero
// point verifier bounds check is the mechanical half of it.
def NPU_QuantTensor : RankedTensorOf<[I8]>;

// Either element type. Used by the shape only operations, which move data
// without arithmetic and so do not care which of the two they are moving.
def NPU_AnyTensor : RankedTensorOf<[F32, I8]>;

#endif // NPU_TYPES
