// Verifier failure tests: each section must produce the expected diagnostic.

// RUN: npu-opt %s -split-input-file -verify-diagnostics

func.func @bad_constant() -> tensor<2xf32> {
  // expected-error @+1 {{value attribute type 'tensor<3xf32>' does not match result type 'tensor<2xf32>'}}
  %0 = npu.constant {value = dense<1.000000e+00> : tensor<3xf32>} : tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

func.func @bad_conv_rank(%in: tensor<3x8x8xf32>,
                         %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  // expected-error @+1 {{expects a rank 4 NCHW input}}
  %0 = npu.conv2d %in, %w {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]}
    : (tensor<3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// -----

func.func @bad_conv_pads(%in: tensor<1x3x8x8xf32>,
                         %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  // expected-error @+1 {{expects a 4 element pads attribute}}
  %0 = npu.conv2d %in, %w {strides = [1, 1], pads = [1, 1], dilations = [1, 1]}
    : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// -----

func.func @bad_matmul(%a: tensor<8x16xf32>, %b: tensor<12x10xf32>) -> tensor<8x10xf32> {
  // expected-error @+1 {{contraction dimensions disagree: 16 vs 12}}
  %0 = npu.matmul %a, %b : (tensor<8x16xf32>, tensor<12x10xf32>) -> tensor<8x10xf32>
  return %0 : tensor<8x10xf32>
}

// -----

func.func @bad_reshape(%a: tensor<2x3xf32>) -> tensor<5xf32> {
  // expected-error @+1 {{element count changes across reshape: 6 vs 5}}
  %0 = npu.reshape %a : tensor<2x3xf32> to tensor<5xf32>
  return %0 : tensor<5xf32>
}

// -----

func.func @bad_transpose(%a: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // expected-error @+1 {{permutation index repeated: 0}}
  %0 = npu.transpose %a {permutation = [0, 0, 1]} : tensor<2x3x4xf32> to tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// -----

func.func @bad_concat(%a: tensor<2x3xf32>, %b: tensor<2x3xf32>) -> tensor<2x6xf32> {
  // expected-error @+1 {{axis 5 out of range for rank 2}}
  %0 = npu.concat %a, %b {axis = 5 : i64}
    : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
  return %0 : tensor<2x6xf32>
}

// -----

func.func @bad_batch_norm(%in: tensor<1x4x8x8xf32>, %s: tensor<3xf32>,
                          %o: tensor<4xf32>, %m: tensor<4xf32>,
                          %v: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  // expected-error @+1 {{scale length 3 does not match channel count 4}}
  %0 = npu.batch_norm %in, %s, %o, %m, %v
    : (tensor<1x4x8x8xf32>, tensor<3xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// -----

// Batch greater than 1 is refused until phase U6 adds real batch loops to the
// simulator. Before this guard existed these four ops verified cleanly and the
// simulator returned correct data for image 0 and garbage for the rest.

func.func @batched_conv(%in: tensor<2x3x8x8xf32>,
                        %w: tensor<4x3x3x3xf32>) -> tensor<2x4x8x8xf32> {
  // expected-error @+1 {{batch size 2 is not supported yet}}
  %0 = npu.conv2d %in, %w {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]}
    : (tensor<2x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<2x4x8x8xf32>
  return %0 : tensor<2x4x8x8xf32>
}

// -----

func.func @batched_max_pool(%in: tensor<4x3x8x8xf32>) -> tensor<4x3x4x4xf32> {
  // expected-error @+1 {{batch size 4 is not supported yet}}
  %0 = npu.max_pool2d %in {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    : (tensor<4x3x8x8xf32>) -> tensor<4x3x4x4xf32>
  return %0 : tensor<4x3x4x4xf32>
}

// -----

func.func @batched_avg_pool(%in: tensor<2x3x8x8xf32>) -> tensor<2x3x4x4xf32> {
  // expected-error @+1 {{batch size 2 is not supported yet}}
  %0 = npu.avg_pool2d %in {kernel_shape = [2, 2], strides = [2, 2], pads = [0, 0, 0, 0]}
    : (tensor<2x3x8x8xf32>) -> tensor<2x3x4x4xf32>
  return %0 : tensor<2x3x4x4xf32>
}

// -----

func.func @batched_batch_norm(%in: tensor<2x4x8x8xf32>, %s: tensor<4xf32>,
                              %o: tensor<4xf32>, %m: tensor<4xf32>,
                              %v: tensor<4xf32>) -> tensor<2x4x8x8xf32> {
  // expected-error @+1 {{batch size 2 is not supported yet}}
  %0 = npu.batch_norm %in, %s, %o, %m, %v
    : (tensor<2x4x8x8xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4x8x8xf32>
  return %0 : tensor<2x4x8x8xf32>
}
