// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// `npu.quantize`'s folder: a quantize of a dequantize with the same scale and
// zero point is the dequantize's input, because the pair returns every one of
// the 256 int8 values unchanged. `NPUSimulatorTests`'
// `Quantization.DequantThenQuantIsTheIdentity` is the machine agreeing with
// that over all 256, and `QuantizeOp::getRoundTripSource` is the argument.
//
// Section 12's negative test rule applies with more force than usual here,
// because a fold that fired on a pair that is not an identity would be a
// requantization deleted: every negative case below is one bit or one count
// away from the positive one.

// RUN: npu-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @identical
// CHECK-NOT: npu.quantize
// CHECK-NOT: npu.dequantize
// CHECK: return %arg0 : tensor<2x3xi8>
func.func @identical(%q: tensor<2x3xi8>) -> tensor<2x3xi8> {
  %d = npu.dequantize %q {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
       : tensor<2x3xi8> to tensor<2x3xf32>
  %r = npu.quantize %d {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
       : tensor<2x3xf32> to tensor<2x3xi8>
  return %r : tensor<2x3xi8>
}

// The dequantize has a second reader in f32, so it stays and only the
// quantize goes: the fold replaces one value and deletes nothing it does not
// own.
// CHECK-LABEL: func.func @shared
// CHECK: %[[D:.*]] = npu.dequantize %arg0
// CHECK-NOT: npu.quantize
// CHECK: return %arg0, %[[D]] : tensor<4xi8>, tensor<4xf32>
func.func @shared(%q: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xf32>) {
  %d = npu.dequantize %q {scale = 1.000000e-01 : f32, zero_point = 0 : i32}
       : tensor<4xi8> to tensor<4xf32>
  %r = npu.quantize %d {scale = 1.000000e-01 : f32, zero_point = 0 : i32}
       : tensor<4xf32> to tensor<4xi8>
  return %r, %d : tensor<4xi8>, tensor<4xf32>
}

// -----------------------------------------------------------------------------
// Left alone. Each of these is a requantization, not an identity.
// -----------------------------------------------------------------------------

// The scales differ in the last bit: 0x3CCCCCCD is the f32 nearest 0.025 and
// 0x3CCCCCCE is the next one up. They print alike at the default precision,
// which is why the comparison is on the bits.
// CHECK-LABEL: func.func @last_bit
// CHECK: npu.dequantize
// CHECK: npu.quantize
func.func @last_bit(%q: tensor<4xi8>) -> tensor<4xi8> {
  %d = npu.dequantize %q {scale = 0x3CCCCCCD : f32, zero_point = -3 : i32}
       : tensor<4xi8> to tensor<4xf32>
  %r = npu.quantize %d {scale = 0x3CCCCCCE : f32, zero_point = -3 : i32}
       : tensor<4xf32> to tensor<4xi8>
  return %r : tensor<4xi8>
}

// The zero points differ by one count, which moves every value by one.
// CHECK-LABEL: func.func @zero_point
// CHECK: npu.dequantize
// CHECK: npu.quantize
func.func @zero_point(%q: tensor<4xi8>) -> tensor<4xi8> {
  %d = npu.dequantize %q {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
       : tensor<4xi8> to tensor<4xf32>
  %r = npu.quantize %d {scale = 2.500000e-02 : f32, zero_point = -2 : i32}
       : tensor<4xf32> to tensor<4xi8>
  return %r : tensor<4xi8>
}

// 255 times 1e37 is past the largest f32, so the dequantize can overflow to an
// infinity and the quantize then saturates rather than returning the value.
// The folder declines rather than reasoning about infinities.
// CHECK-LABEL: func.func @overflowing_scale
// CHECK: npu.dequantize
// CHECK: npu.quantize
func.func @overflowing_scale(%q: tensor<4xi8>) -> tensor<4xi8> {
  %d = npu.dequantize %q {scale = 1.000000e+37 : f32, zero_point = 0 : i32}
       : tensor<4xi8> to tensor<4xf32>
  %r = npu.quantize %d {scale = 1.000000e+37 : f32, zero_point = 0 : i32}
       : tensor<4xf32> to tensor<4xi8>
  return %r : tensor<4xi8>
}

// A quantize of anything but a dequantize is a quantization, whatever its
// attributes say.
// CHECK-LABEL: func.func @not_a_pair
// CHECK: npu.quantize %arg0
func.func @not_a_pair(%x: tensor<4xf32>) -> tensor<4xi8> {
  %r = npu.quantize %x {scale = 2.500000e-02 : f32, zero_point = -3 : i32}
       : tensor<4xf32> to tensor<4xi8>
  return %r : tensor<4xi8>
}
