// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Section 16.2's instrumentation, from the tool's side.
//
// `test/Python/test_pass_instrumentation.py` is where the file's contents are
// checked in detail, because the expectation it is checked against is the
// pipeline description and reading that is the driver's job. What is checked
// here is the half that belongs to `npu-opt`: that the flag writes the file at
// all, that the file describes the pipeline the level ran, that the ablation
// option removes exactly what it names, and that the tool without the flag is
// the tool it was before.

// The file is written, and it holds a before and an after count for every pass
// of the level in the order the level runs them.
// RUN: npu-opt %s --npu-O2 --npu-pass-stats-json=%t.stats.json -o /dev/null
// RUN: FileCheck --check-prefix=STATS %s < %t.stats.json

// STATS: "clock": "std::chrono::steady_clock"
// STATS: "generated_by": "lib/Pipeline/PassStats.cpp"
// STATS: "name": "npu-constant-fold"
// STATS: "ops_after_total"
// STATS: "ops_before_total"
// STATS: "pass_name": "NPUConstantFold"
// STATS: "pass_timing_source": "measured: PassInstrumentation runBeforePass and runAfterPass
// STATS: "wall_ms"
// STATS: "name": "canonicalize"
// STATS: "name": "npu-fuse-bias"
// STATS: "name": "npu-fold-batchnorm"
// STATS: "name": "npu-fuse-ops"
// STATS: "name": "canonicalize"
// STATS: "name": "cse"
// STATS: "name": "sccp"
// STATS: "name": "symbol-dce"
// STATS: "name": "npu-lower-to-npuisa"
// STATS: "name": "npu-allocate-scratchpad"
// STATS: "run_completed": true
// STATS: "still_running_at_exit": []
// STATS: "unmatched_after_pass": []

// The leave one out ablation of Section 16.2, through the level's own pipeline
// rather than through a pass list assembled beside it. `-cse` is gone from the
// recorded run, which is the property the harness reads back rather than
// trusting the flag to have worked.
// RUN: npu-opt %s '--npu-O2=ablate=cse' --npu-pass-stats-json=%t.ablated.json -o /dev/null
// RUN: FileCheck --check-prefix=ABLATED %s < %t.ablated.json
// RUN: not grep '"name": "cse"' %t.ablated.json

// ABLATED: "name": "npu-fuse-ops"
// ABLATED: "name": "canonicalize"
// ABLATED: "name": "sccp"

// `-canonicalize` has two positions at `-O2` and is one pass to ablate, so its
// row removes both. Section 12's table marks the duplication as deliberate and
// an ablation that removed one position would be measuring an ordering change.
// RUN: npu-opt %s '--npu-O2=ablate=canonicalize' --npu-pass-stats-json=%t.nocanon.json -o /dev/null
// RUN: not grep '"name": "canonicalize"' %t.nocanon.json

// A pass the table marks as not ablatable is not removed here. The refusal
// lives in the driver, which reads the ablatable set at run time; MLIR's
// pipeline registration gives this option no way to report an error, so the
// hole is closed by the statistics still naming the pass rather than by a
// diagnostic that could not be emitted.
// RUN: npu-opt %s '--npu-O2=ablate=npu-allocate-scratchpad' --npu-pass-stats-json=%t.notablatable.json -o /dev/null
// RUN: grep '"name": "npu-allocate-scratchpad"' %t.notablatable.json

// The tool without the flag is the tool it was. `npu-opt` grew a second entry
// path at P10 to reach the pass manager MlirOptMain builds, and this is the
// assertion that the path everything else takes did not move.
// RUN: npu-opt %s --npu-O2 -o %t.plain.mlir
// RUN: npu-opt %s --npu-O2 --npu-pass-stats-json=%t.unused.json -o %t.instrumented.mlir
// RUN: diff %t.plain.mlir %t.instrumented.mlir

// The same function `opt-levels.mlir` uses, and it is the same on purpose: one
// convolution, one bias add for the bias fusion, one batch norm to fold, one
// activation to fuse, and a subgraph nothing reads, so every `-O2` pass has
// something to do and a level that quietly ran none of them would show here.
func.func @small(%input: tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = npu.constant dense<5.000000e-01> : tensor<2x2x3x3xf32>
  %d0 = tensor.empty() : tensor<1x2x4x4xf32>
  %conv = npu.conv2d ins(%input, %w : tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>)
                     outs(%d0 : tensor<1x2x4x4xf32>)
                     {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                      dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x2x4x4xf32>

  %bias = npu.constant dense<[1.000000e-01, 2.000000e-01]> : tensor<2xf32>
  %d1 = tensor.empty() : tensor<1x2x4x4xf32>
  %biased = npu.add ins(%conv, %bias : tensor<1x2x4x4xf32>, tensor<2xf32>)
                    outs(%d1 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>

  %gamma = npu.constant dense<2.000000e+00> : tensor<2xf32>
  %beta = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %mean = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %var = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d2 = tensor.empty() : tensor<1x2x4x4xf32>
  %normed = npu.batch_norm ins(%biased, %gamma, %beta, %mean, %var :
                               tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>,
                               tensor<2xf32>, tensor<2xf32>)
                           outs(%d2 : tensor<1x2x4x4xf32>)
                           {epsilon = 1.000000e+00 : f32} -> tensor<1x2x4x4xf32>

  %d3 = tensor.empty() : tensor<1x2x4x4xf32>
  %out = npu.relu ins(%normed : tensor<1x2x4x4xf32>)
                  outs(%d3 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>

  %k = npu.constant dense<7.000000e+00> : tensor<1x2x4x4xf32>
  %d4 = tensor.empty() : tensor<1x2x4x4xf32>
  %dead = npu.mul ins(%out, %k : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
                  outs(%d4 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>

  return %out : tensor<1x2x4x4xf32>
}
