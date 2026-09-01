//===- Passes.td - npu tensor level passes -----------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The four `npu` level passes of Section 12 that Phase P9 lands. Layout
// assignment, tiling, double buffering and calibration are the four Section 12
// entries P9 leaves alone, and they are absent here rather than declared empty.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_PASSES
#define NPU_PASSES

include "mlir/Pass/PassBase.td"

def NPUConstantFold : Pass<"npu-constant-fold", "::mlir::func::FuncOp"> {
  let summary = "Evaluates npu operations whose operands are all constants.";
  let description = [{
    Section 12's `-npu-constant-fold`, at `-O1` and `-O2`. An `npu.add`,
    `npu.mul`, `npu.relu` or `npu.reshape` whose reads are all `npu.constant`
    becomes one `npu.constant` holding the answer.

    **The arithmetic is the kernel's arithmetic, element for element.** These
    are elementwise operations with no reduction in them, so folding one at
    compile time and executing it at run time produce the same `f32` bits.
    Section 17.6 puts P9 in the 1e-6 class because *some* of this phase's
    passes move numbers; this one does not, and saying which is which is the
    point of naming the mechanism.

    **A rank 1 channel broadcast addend is refused rather than folded**, and
    that refusal is load bearing. Folding `add(%x, %c)` where `%c` is the rank
    1 constant of Section 11's carve out would mean materialising the
    broadcast, which is exactly the `N x C x H x W` expansion the importer
    refuses to perform: it would inflate every DRAM byte count by the expansion
    factor and it would make `-npu-fuse-bias` unfireable on the result. The
    guard is that the reads have the result's shape exactly.

    The destination operand is not read, it is replaced: a folded operation
    leaves its `tensor.empty` with no user, and `-canonicalize` removes it.
    That is why Section 12 puts a canonicalization beside this pass rather than
    after the whole pipeline.
  }];

  let statistics = [
    Statistic<"foldedOps", "folded-ops", "Operations replaced by a constant">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def NPUFoldBatchNorm : Pass<"npu-fold-batchnorm", "::mlir::func::FuncOp"> {
  let summary = "Folds an inference batch norm into the convolution before it.";
  let description = [{
    Section 12's `-npu-fold-batchnorm`, at `-O2`, placed **before** fusion so
    that the convolution it rewrites still has no fused activation, with a
    canonicalization after it to remove the parameter constants it leaves dead.

    `batch_norm(conv2d(x, w, b), gamma, beta, mean, variance)` becomes one
    `conv2d(x, w', b')` over constants computed at rewrite time:

    ```
    invStd = 1 / sqrt(variance + epsilon)
    scale  = gamma * invStd
    w'[f]  = w[f] * scale[f]
    b'[f]  = (b[f] - mean[f]) * scale[f] + beta[f]
    ```

    **That evaluation order is written down because it is observable.** Floating
    point multiplication is not associative, so a reader comparing this against
    onnxruntime needs to know which of several algebraically equal forms
    produced the number. It is the same order
    `-npu-lower-to-npuisa` uses for the decomposition of a batch norm this pass
    did not fold, so the two spellings of the same identity agree with each
    other by construction.

    **This is the pass that moves numbers, and it is the only one at P9 that
    does.** Before the fold the machine computes a convolution and then scales
    its result; after it the machine convolves with pre scaled weights, so every
    product in the reduction is scaled instead of the sum being scaled once at
    the end. The two are equal in exact arithmetic and differ in the last bits
    of `f32`. `docs/BREAKING_CHANGES.md` carries the declaration and the
    measured magnitude.

    Five guards, each with a lit test proving the pass does not fire:

    - the producer is an `npu.conv2d`, not something else;
    - the convolution has exactly one use, which is this batch norm's input,
      because rewriting a convolution two consumers read would change what the
      other one sees;
    - `gamma`, `beta`, `mean` and `variance` are all `npu.constant`;
    - the convolution's filter, and its bias when it has one, are `npu.constant`;
    - `variance + epsilon` is strictly positive in every channel, because the
      fold takes a reciprocal square root of it.
  }];

  let statistics = [
    Statistic<"foldedBatchNorms", "folded-batch-norms",
              "Batch norms folded into a preceding convolution">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def NPUFuseBias : Pass<"npu-fuse-bias", "::mlir::func::FuncOp"> {
  let summary = "Folds a separate channel shaped add into a bias operand.";
  let description = [{
    Section 12's `-npu-fuse-bias`, at `-O2`. `add(conv2d(x, w), b)` becomes
    `conv2d(x, w, b)`, and `add(matmul(a, b), c)` becomes `matmul(a, b, c)`.

    **This is the pass Section 11's broadcast carve out exists for.** A rank 1
    initializer of length C broadcasting against a rank 4 activation over the
    channel axis is left unexpanded by the importer, as a rank 1 constant,
    precisely so that this guard can match. An importer that expanded it would
    make this pass structurally unfireable, its ablation row a row of zeros, and
    the phase look done while doing nothing.

    **It is exact.** The simulator's convolution kernel adds the bias to the
    same `f32` accumulator the unfused program would have stored and then added
    to, so the fused and the unfused answers agree bit for bit. That is measured
    rather than assumed, and it is why this phase's numerics movement is
    attributed to `-npu-fold-batchnorm` alone.

    The three negative cases Section 12 names, each with a lit test:

    - **already fused**: the producer already carries a bias operand, so there
      is nothing to move into;
    - **multiple uses**: the producer's result is read by something other than
      this add, and rewriting it would change what that reader sees;
    - **a non channel constant addend**: an addend that is not an
      `npu.constant`, or one whose shape is the result's rather than rank 1 of
      the channel extent.

    **The commuted form is explicitly not matched**, which is the other half of
    the choice Section 12 offers. `npu.add` refuses a rank 1 left hand operand
    in its verifier, so `add(b, conv(x, w))` is not representable in this
    dialect at all; the importer commutes at import and there is one spelling
    below it. Matching a form the verifier rejects would be dead code with a
    test that could never be written.
  }];

  let statistics = [
    Statistic<"fusedBiases", "fused-biases",
              "Adds folded into a producer's bias operand">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

def NPUFuseOps : Pass<"npu-fuse-ops", "::mlir::func::FuncOp"> {
  let summary = "Forms npu.fused_op regions over a producer and its activation.";
  let description = [{
    Section 12's `-npu-fuse-ops`, at `-O2`. A `npu.relu` whose input is produced
    by an `npu.conv2d` or an `npu.matmul` with exactly one use is moved, with
    its producer, into one `npu.fused_op` region.

    **The region is the representation, and Section 7.2 says why.** Separate
    fused operation names produce a combinatorial explosion as the fusible set
    grows, and an enum cannot express what fusion does, which is to keep an
    intermediate value in the scratchpad instead of writing it to DRAM. The
    region says exactly that and can grow without a migration.

    Every value the region reads arrives as an operand and a matching block
    argument, destinations included, because `npu.fused_op` is
    `IsolatedFromAbove` and because a `tensor.empty` is not an `npu` operation
    and the region's verifier admits only `npu` operations.

    **It is numerically inert, and that is a property of this memory model
    rather than a weakness of the pass.** `-npu-lower-to-npuisa` flattens the
    region into its parent, so the instruction stream is the one the unfused
    chain produced, and an unfused chain already keeps its intermediate in the
    scratchpad. What the region adds is that fusion is *stated* in the IR, which
    is what a later phase's tiling and double buffering read.

    The two guards Section 12 names, each with a lit test:

    - **the producer has exactly one use**, because a second reader of the
      intermediate would have to read it from inside a region it is not in;
    - **an already fused producer is not fused again**, so the pass reaches a
      fixed point in one run rather than nesting regions until it runs out.
  }];

  let statistics = [
    Statistic<"fusedRegions", "fused-regions", "npu.fused_op regions formed">,
  ];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::npu::NPUDialect",
    "::mlir::tensor::TensorDialect",
  ];
}

#endif // NPU_PASSES
