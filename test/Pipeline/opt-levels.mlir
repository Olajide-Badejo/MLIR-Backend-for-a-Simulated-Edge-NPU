// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The three `-O` levels of Section 12, and the description the driver reads.
//
// The load bearing checks are the three diffs: each level and the explicit list
// of the pass arguments it claims to run produce byte identical IR. Section 17.4
// says a test that runs a hardcoded pass list matching no optimization level
// enforces nothing, and the converse obligation is this one, that a level whose
// contents nobody compares against anything can drift from the passes it says
// it runs. Since P9 the level tables carry a `PassKind` beside each argument
// string and the builder switches on the kind, so these diffs are also what
// catches a kind and a name that disagree.

// RUN: npu-opt %s --npu-O0 -o %t.o0.mlir
// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad -o %t.o0e.mlir
// RUN: diff %t.o0.mlir %t.o0e.mlir
// RUN: FileCheck --check-prefix=LOWERED %s < %t.o0.mlir

// RUN: npu-opt %s --npu-O1 -o %t.o1.mlir
// RUN: npu-opt %s --npu-constant-fold --canonicalize --npu-lower-to-npuisa --npu-allocate-scratchpad -o %t.o1e.mlir
// RUN: diff %t.o1.mlir %t.o1e.mlir

// **The `-O2` list carries an option and that is the point of writing it out.**
// The pipeline hands `-npu-tile-to-scratchpad` the allocator's budget and tells
// it whether `-npu-double-buffer` is in this pipeline, because Section 13.2
// sizes the tiling search's working set for the prefetch. A flat list that
// spelled the pass without `double-buffer=true` would be a different
// compilation, so it is spelled with it, and the diff is then an assertion
// about the coupling rather than one that steps around it.
// RUN: npu-opt %s --npu-O2 -o %t.o2.mlir
// RUN: npu-opt %s --npu-constant-fold --canonicalize --npu-fuse-bias --npu-fold-batchnorm --npu-fuse-ops --canonicalize --cse --sccp --symbol-dce --npu-assign-layout --npu-tile-to-scratchpad=double-buffer=true --npu-lower-to-npuisa --npu-double-buffer --npu-allocate-scratchpad -o %t.o2e.mlir
// RUN: diff %t.o2.mlir %t.o2e.mlir

// The tensor level half of a level, which is what `npu-compile --emit npu`
// runs. It runs it through this pipeline and not through a pass list assembled
// in Python, which is the same rule seen from the driver's side.
// RUN: npu-opt %s '--npu-O0=stop-after=npu' | FileCheck --check-prefix=NPU0 %s
// RUN: npu-opt %s '--npu-O2=stop-after=npu' | FileCheck --check-prefix=NPU2 %s

// The budget reaches the allocator through the pipeline's own option rather
// than through the pass's, which is what makes `npu-compile --budget` a thing
// the driver can pass down without knowing which pass consumes it.
// RUN: npu-opt %s --npu-O0=budget=1024 | FileCheck --check-prefix=BUDGET %s

// A budget the program cannot be placed in is a diagnostic with both numbers
// in it, not a wrong program. The pipeline does not soften that.
// RUN: not npu-opt %s --npu-O0=budget=64 2>&1 | FileCheck --check-prefix=TOOSMALL %s

// The description of Section 16.2, which the driver reads at run time instead
// of keeping a second copy of the pass list in Python.
// RUN: npu-opt --npu-describe-pipeline | FileCheck --check-prefix=DESCRIBE %s

// A level that is not a level is still an unknown argument. -O3 is not named by
// this compiler at all, which is different from -O1 and -O2 being named and
// unbuilt, and that was the state those two were in until P9.
// RUN: not npu-opt %s --npu-O3 2>&1 | FileCheck --check-prefix=NOSUCH %s

// A convolution, a batch norm the folding pass can fold, an activation the
// fusion pass can fuse, a bias add the bias fusion can fold, and a subgraph
// nothing reads. One function that gives every pass in the -O2 table something
// to do, so a level that quietly ran none of them would show here.
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

  // Nothing reads this. `-O0` lowers it and `-O1` and `-O2` remove it, which is
  // the `eliminatesDeadCode` property doing its work.
  %k = npu.constant dense<7.000000e+00> : tensor<1x2x4x4xf32>
  %d4 = tensor.empty() : tensor<1x2x4x4xf32>
  %dead = npu.mul ins(%out, %k : tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>)
                  outs(%d4 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>

  return %out : tensor<1x2x4x4xf32>
}

// -O0 ran both of its passes and neither of them removes anything: the
// convolution, the bias add, the batch norm's decomposition and the dead
// multiply are all in the instruction stream.
// LOWERED: func.func @small
// LOWERED-SAME: memref<1x2x4x4xf32, #npu.dram>
// LOWERED-SAME: npuisa.scratchpad_bytes
// LOWERED: memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
// LOWERED: npuisa.dma_load
// LOWERED: npuisa.conv2d
// LOWERED: npuisa.add
// LOWERED: npuisa.relu
// LOWERED: npuisa.mul
// LOWERED: npuisa.dma_store

// -O0's tensor level half runs no pass at all, so it is the input verified and
// reprinted. That is what "-O0 is import and verify" means, and it is why
// `--emit import` and `--emit npu` agree in content at this level and at no
// other.
// NPU0: npu.conv2d
// NPU0: npu.add
// NPU0: npu.batch_norm
// NPU0: npu.relu
// NPU0: npu.mul

// -O2's tensor level half: the batch norm is gone into the convolution's
// weights, the bias add is gone into its bias operand, the convolution and its
// activation are inside one region, and the dead multiply is gone.
// NPU2: npu.fused_op
// NPU2: npu.conv2d
// NPU2: npu.relu
// NPU2: npu.yield
// NPU2-NOT: npu.batch_norm
// NPU2-NOT: npu.add
// NPU2-NOT: npu.mul

// BUDGET: npuisa.scratchpad_budget = 1024 : i64

// TOOSMALL: the scratchpad budget of 64 bytes is too small

// All three levels are implemented at P9 and none of them names an arriving
// phase any more.
// DESCRIBE:      "levels": [
// DESCRIBE:          "implemented": true,
// DESCRIBE-NEXT:     "level": 0,
// DESCRIBE-NEXT:     "name": "-O0",
// DESCRIBE:          "ablatable": false,
// DESCRIBE-NEXT:     "eliminates_dead_code": false,
// DESCRIBE:          "pass": "npu-lower-to-npuisa",
// DESCRIBE-NEXT:     "stage": "npuisa"
// DESCRIBE:          "pass": "npu-allocate-scratchpad",
// DESCRIBE:        "implemented": true,
// DESCRIBE-NEXT:   "level": 1,
// DESCRIBE-NEXT:   "name": "-O1",
// DESCRIBE:          "ablatable": true,
// DESCRIBE-NEXT:     "eliminates_dead_code": false,
// DESCRIBE:          "pass": "npu-constant-fold",
// DESCRIBE-NEXT:     "stage": "npu"
// DESCRIBE:          "ablatable": true,
// DESCRIBE-NEXT:     "eliminates_dead_code": true,
// DESCRIBE:          "pass": "canonicalize",
// DESCRIBE:        "implemented": true,
// DESCRIBE-NEXT:   "level": 2,
// DESCRIBE-NEXT:   "name": "-O2",
// DESCRIBE:          "pass": "npu-fuse-bias",
// DESCRIBE:          "pass": "npu-fold-batchnorm",
// DESCRIBE:          "pass": "npu-fuse-ops",
// DESCRIBE:          "pass": "cse",
// DESCRIBE:          "pass": "sccp",
// DESCRIBE:          "eliminates_dead_code": true,
// DESCRIBE:          "pass": "symbol-dce",
// The three P13 rows, in Section 12's own order. Layout assignment and tiling
// are at the tensor level and double buffering is not: it rewrites the
// asynchronous transfer tokens, which exist only below the conversion, which is
// why it sits after the lowering and before the allocator.
// DESCRIBE:          "ablatable": true,
// DESCRIBE:          "pass": "npu-assign-layout",
// DESCRIBE-NEXT:     "stage": "npu"
// DESCRIBE:          "ablatable": true,
// DESCRIBE:          "pass": "npu-tile-to-scratchpad",
// DESCRIBE-NEXT:     "stage": "npu"
// DESCRIBE:          "ablatable": false,
// DESCRIBE:          "pass": "npu-lower-to-npuisa",
// DESCRIBE-NEXT:     "stage": "npuisa"
// DESCRIBE:          "ablatable": true,
// DESCRIBE:          "pass": "npu-double-buffer",
// DESCRIBE-NEXT:     "stage": "npuisa"
// DESCRIBE:          "ablatable": false,
// DESCRIBE:          "pass": "npu-allocate-scratchpad",
// DESCRIBE-NEXT:     "stage": "npuisa"

// NOSUCH: Unknown command line argument '--npu-O3'
