// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// Everything the lowering refuses, refused by name.
//
// The reason these live in their own file rather than beside the positive cases
// is the run line: a `-verify-diagnostics` run asserts that the errors it was
// told to expect are the errors that appeared, and a file mixing the two would
// have to be split anyway. Each case matches a substring the pass actually
// emits rather than a generic pattern, because a test matching only "error"
// would pass against a pass that refused everything for the wrong reason.
//
// All of them are emitted from the validation stage, before any operation has
// been rewritten. That is deliberate: a diagnostic emitted from inside a
// conversion pattern competes with the framework's own "failed to legalize
// operation" message, and the reader then has two messages and no idea which
// one is the answer.

// RUN: npu-opt %s --npu-lower-to-npuisa -split-input-file -verify-diagnostics

// =============================================================================
// Section 5.2: no branch instructions, so no control flow reaches here.
// =============================================================================

func.func @an_scf_loop(%x: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error @+1 {{an scf operation reaches the lowering}}
  %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %x)
      -> (tensor<1x4x4x4xf32>) {
    scf.yield %acc : tensor<1x4x4x4xf32>
  }
  return %r : tensor<1x4x4x4xf32>
}

// -----

// expected-error @+1 {{requires a single block function body, but @two_blocks has 2 blocks}}
func.func @two_blocks(%x: tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  cf.br ^tail
^tail:
  return %x : tensor<1x4x4x4xf32>
}

// -----

// =============================================================================
// A value that cannot be assigned a memory space is refused with a diagnostic
// naming it, which is the P4 gate's own wording. Three ways to be unassignable:
// a dynamic extent, an element type this machine has no memory for, and a
// layout encoding on a rank that has no layout reading.
// =============================================================================

// expected-error @+1 {{argument 0 of @a_dynamic_extent has type 'tensor<?x4xf32>', which cannot be assigned a memory space, because it has a dynamic extent}}
func.func @a_dynamic_extent(%x: tensor<?x4xf32>) -> tensor<?x4xf32> {
  return %x : tensor<?x4xf32>
}

// -----

// expected-error @+1 {{its element type i32 is not one this machine has a memory for; the two are f32 and i8}}
func.func @an_unsupported_element_type(%x: tensor<4x4xi32>) -> tensor<4x4xi32> {
  return %x : tensor<4x4xi32>
}

// -----

// A dynamic destination. The `npu` operations refuse a dynamic extent at the
// type level and so cannot carry one, but `tensor.empty` belongs to the tensor
// dialect and can, which leaves this pass as the layer that has to name it. It
// names the operation and the result rather than the function, because a
// function whose signature is fine is not where the reader should be looking.
func.func @a_dynamic_destination(%n: index) {
  // expected-error @+1 {{result 0 has type 'tensor<?x4xf32>', which cannot be assigned a memory space, because it has a dynamic extent}}
  %d = tensor.empty(%n) : tensor<?x4xf32>
  return
}

// -----

// =============================================================================
// A function declaration. This compiler has no calls, so a body is not
// optional.
// =============================================================================

// expected-error @+1 {{@declared_elsewhere is a declaration, and this compiler has no calls and no linking}}
func.func private @declared_elsewhere(%x: tensor<4x4xf32>) -> tensor<4x4xf32>

// -----

// =============================================================================
// The batch norm decomposition computes its multiplier and its addend at
// rewrite time, so a parameter that is not constant data is named rather than
// producing a generic legalization failure. That naming is the roadmap entry's
// own requirement.
// =============================================================================

func.func @a_computed_batch_norm_parameter(%x: tensor<1x2x2x2xf32>,
                                           %g: tensor<2xf32>)
    -> tensor<1x2x2x2xf32> {
  %b = npu.constant dense<1.000000e+00> : tensor<2xf32>
  %m = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<3.000000e+00> : tensor<2xf32>
  %d = tensor.empty() : tensor<1x2x2x2xf32>
  // expected-error @+1 {{the gamma operand of this batch norm is not an npu.constant}}
  %r = npu.batch_norm ins(%x, %g, %b, %m, %v : tensor<1x2x2x2xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d : tensor<1x2x2x2xf32>)
                      {epsilon = 1.000000e+00 : f32} -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}

// -----

// A variance and epsilon that sum to zero. The decomposition takes a square
// root and then a reciprocal of it, so this would produce an infinity rather
// than a diagnostic if it were not checked, and an infinity in a weight is a
// wrong answer that survives all the way to the output tensor.
func.func @a_zero_denominator(%x: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %g = npu.constant dense<1.000000e+00> : tensor<2xf32>
  %b = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %m = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %v = npu.constant dense<0.000000e+00> : tensor<2xf32>
  %d = tensor.empty() : tensor<1x2x2x2xf32>
  // expected-error @+1 {{cannot be decomposed on channel 0}}
  %r = npu.batch_norm ins(%x, %g, %b, %m, %v : tensor<1x2x2x2xf32>,
                          tensor<2xf32>, tensor<2xf32>, tensor<2xf32>,
                          tensor<2xf32>)
                      outs(%d : tensor<1x2x2x2xf32>)
                      {epsilon = 0.000000e+00 : f32} -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}

// -----

// =============================================================================
// The two layout cases this lowering does not represent.
//
// Neither is reachable from the frontend, which emits no layout encodings at
// all. Both become reachable the day `-npu-assign-layout` lands, and both are
// that pass's work rather than a hole here: it materialises its own permuted
// constants and it folds its own inverse transposes. They are diagnosed rather
// than left to produce a verifier failure from inside a pass, because a stated
// rule is something the next phase can implement against.
// =============================================================================

func.func @a_layout_encoded_constant() -> tensor<1x8x8x3xf32, #npu.layout<nhwc>> {
  // expected-error @+1 {{this constant carries the layout encoding #npu.layout<nhwc>, and the lowering does not permute constant data}}
  %c = npu.constant dense<1.000000e+00>
     : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
  return %c : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
}

// -----

func.func @a_relayouting_transpose(%x: tensor<1x3x8x8xf32>)
    -> tensor<1x8x8x3xf32, #npu.layout<nhwc>> {
  %d = tensor.empty() : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
  // expected-error @+1 {{this transpose changes the layout as well as the extents}}
  %r = npu.transpose ins(%x : tensor<1x3x8x8xf32>)
                     outs(%d : tensor<1x8x8x3xf32, #npu.layout<nhwc>>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x8x8x3xf32, #npu.layout<nhwc>>
  return %r : tensor<1x8x8x3xf32, #npu.layout<nhwc>>
}
