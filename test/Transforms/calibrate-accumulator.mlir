// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Section 14's static accumulator guard, which is this pass's job rather than
// the machine's.
//
// "No INT8 NPU accumulates in 64 bits", so the scheme holds the accumulator to
// int32 and proves it before the program exists: `K * 128 * 127 < 2^31`, which
// admits a reduction depth up to 132104. A guard here is both more honest than
// a wider accumulator and a better error than a trap at run time, and the
// simulator's own trap carries the same bound for a program that reaches it
// anyway.

// RUN: not npu-opt %s \
// RUN:   --npu-calibrate=profile=%S/Inputs/calibrate-deep.json 2>&1 \
// RUN:   | FileCheck %s

// CHECK: error: {{.*}}this operation reduces over 200000 elements
// CHECK-SAME: K * 128 * 127 must be below 2^31, which holds up to 132104

func.func @too_deep(%a: tensor<1x200000xf32>) -> tensor<1x2xf32> {
  %w = npu.constant dense<1.000000e+00> : tensor<200000x2xf32>
  %d = tensor.empty() : tensor<1x2xf32>
  %m = npu.matmul ins(%a, %w : tensor<1x200000xf32>, tensor<200000x2xf32>)
                  outs(%d : tensor<1x2xf32>) -> tensor<1x2xf32> loc("Gemm_0")
  return %m : tensor<1x2xf32>
}
