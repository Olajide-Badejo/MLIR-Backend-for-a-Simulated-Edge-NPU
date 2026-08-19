// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The P0 suite has nothing of this project's own to test, because the npu
// dialect does not exist until P1. What it can test, and what this file tests,
// is that the harness itself works end to end: that npu-opt was built, that lit
// substitutes it, that FileCheck is on the path, and that a module survives a
// parse and print round trip through the driver. If P1 opens with a broken
// dialect, this test having passed at P0 says the breakage is in the dialect
// and not in the build.
//
// A genuinely empty suite is not an option here even though the specification
// describes one: llvm-lit exits 2 on "did not discover any tests" and does so
// deliberately, with an upstream test asserting that --allow-empty-runs does
// not suppress it. See docs/ENGINEERING_LOG.md for the full note.

// RUN: npu-opt %s | FileCheck %s

// CHECK-LABEL: func.func @roundtrip
// CHECK: arith.addf
func.func @roundtrip(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
