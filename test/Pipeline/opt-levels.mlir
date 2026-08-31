// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// The -O0 pipeline of Section 12, and the description the driver reads.
//
// The load bearing check is the first one: `--npu-O0` and the explicit pass
// list produce byte identical IR. Section 17.4 says a test that runs a
// hardcoded pass list matching no optimization level enforces nothing, and the
// converse obligation is this one: a level whose contents nobody compares
// against anything is a level that can drift from the passes it claims to run.

// RUN: npu-opt %s --npu-O0 -o %t.level.mlir
// RUN: npu-opt %s --npu-lower-to-npuisa --npu-allocate-scratchpad -o %t.explicit.mlir
// RUN: diff %t.level.mlir %t.explicit.mlir
// RUN: FileCheck --check-prefix=LOWERED %s < %t.level.mlir

// The budget reaches the allocator through the pipeline's own option rather
// than through the pass's, which is what makes `npu-compile --budget` a thing
// the driver can pass down without knowing which pass consumes it.
// RUN: npu-opt %s --npu-O0=budget=512 | FileCheck --check-prefix=BUDGET %s

// A budget the program cannot be placed in is a diagnostic with both numbers
// in it, not a wrong program. The pipeline does not soften that.
// RUN: not npu-opt %s --npu-O0=budget=64 2>&1 | FileCheck --check-prefix=TOOSMALL %s

// The description of Section 16.2, which the driver reads at run time instead
// of keeping a second copy of the pass list in Python.
// RUN: npu-opt --npu-describe-pipeline | FileCheck --check-prefix=DESCRIBE %s

// -O1 and -O2 are named in the description and are not registered as pipelines,
// so asking for one is an unknown argument rather than a pipeline that runs and
// quietly produces -O0's answer.
// RUN: not npu-opt %s --npu-O2 2>&1 | FileCheck --check-prefix=NOTYET %s

func.func @small(%input: tensor<1x2x4x4xf32>, %scale: tensor<2xf32>) -> tensor<1x2x4x4xf32> {
  %empty = tensor.empty() : tensor<1x2x4x4xf32>
  %scaled = npu.mul ins(%input, %scale : tensor<1x2x4x4xf32>, tensor<2xf32>)
                    outs(%empty : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %empty2 = tensor.empty() : tensor<1x2x4x4xf32>
  %out = npu.relu ins(%scaled : tensor<1x2x4x4xf32>)
                  outs(%empty2 : tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  return %out : tensor<1x2x4x4xf32>
}

// The level ran both of its passes: lowering produced the memref boundary and
// the DMA pair, and allocation produced the arena and the offsets.
// LOWERED: func.func @small
// LOWERED-SAME: memref<1x2x4x4xf32, #npu.dram>
// LOWERED-SAME: npuisa.scratchpad_bytes
// LOWERED: memref.alloc() {alignment = 64 : i64, npuisa.scratchpad_arena}
// LOWERED: npuisa.dma_load
// LOWERED: npuisa.mul
// LOWERED: npuisa.relu
// LOWERED: npuisa.dma_never

// BUDGET: npuisa.scratchpad_budget = 512 : i64

// TOOSMALL: the scratchpad budget of 64 bytes is too small

// DESCRIBE:      "levels": [
// DESCRIBE:          "implemented": true,
// DESCRIBE-NEXT:     "level": 0,
// DESCRIBE-NEXT:     "name": "-O0",
// DESCRIBE:          "ablatable": false,
// DESCRIBE-NEXT:     "eliminates_dead_code": false,
// DESCRIBE:          "pass": "npu-lower-to-npuisa"
// DESCRIBE:          "ablatable": false,
// DESCRIBE-NEXT:     "eliminates_dead_code": false,
// DESCRIBE:          "pass": "npu-allocate-scratchpad"
// DESCRIBE:        "arrives_at": "P9",
// DESCRIBE-NEXT:   "implemented": false,
// DESCRIBE-NEXT:   "level": 1,
// DESCRIBE:        "arrives_at": "P9",
// DESCRIBE-NEXT:   "implemented": false,
// DESCRIBE-NEXT:   "level": 2,

// NOTYET: Unknown command line argument '--npu-O2'
