// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT

// One case per verifier rule, each with an expected-error on the substring the
// verifier actually emits rather than a generic match. A test that matched only
// "error" would pass against a verifier that rejected everything for the wrong
// reason, which is the failure mode a negative test exists to rule out.

// RUN: npu-opt %s -split-input-file -verify-diagnostics

// =============================================================================
// The destination passing rules.
// =============================================================================

func.func @destination_shape_differs(%x: tensor<1x8x4x4xf32>,
                                     %d: tensor<1x8x4x5xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{destination type must equal the result type exactly, including any layout encoding, but the destination is 'tensor<1x8x4x5xf32>' and the result is 'tensor<1x8x4x4xf32>'}}
  %0 = npu.relu ins(%x : tensor<1x8x4x4xf32>) outs(%d : tensor<1x8x4x5xf32>)
       -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

// Same shape, same element type, different layout encoding. This is the case
// that makes "exactly" the operative word in the rule: without the encoding in
// the comparison, a tiling loop would insert an NCHW tile into an NHWC
// destination and nothing would say so.
func.func @destination_layout_differs(
    %x: tensor<1x8x4x4xf32, #npu.layout<nhwc>>,
    %d: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32, #npu.layout<nhwc>> {
  // expected-error @+1 {{operands must not mix layouts, but input is nhwc and destination is nchw (absent encoding)}}
  %0 = npu.relu ins(%x : tensor<1x8x4x4xf32, #npu.layout<nhwc>>)
                outs(%d : tensor<1x8x4x4xf32>)
       -> tensor<1x8x4x4xf32, #npu.layout<nhwc>>
  return %0 : tensor<1x8x4x4xf32, #npu.layout<nhwc>>
}

// =============================================================================
// The layout encoding rules.
// =============================================================================

// -----

// A layout encoding on a tensor that is not rank 4. There is no NCHW reading of
// a rank 2 matrix, so accepting one would mean inventing a meaning for it.
func.func @layout_on_rank_two(%x: tensor<4x4xf32, #npu.layout<nhwc>>,
                              %d: tensor<4x4xf32, #npu.layout<nhwc>>)
    -> tensor<4x4xf32, #npu.layout<nhwc>> {
  // expected-error @+1 {{a layout encoding is only meaningful on a rank 4 tensor, but operand input has rank 2 and carries nhwc}}
  %0 = npu.relu ins(%x : tensor<4x4xf32, #npu.layout<nhwc>>)
                outs(%d : tensor<4x4xf32, #npu.layout<nhwc>>)
       -> tensor<4x4xf32, #npu.layout<nhwc>>
  return %0 : tensor<4x4xf32, #npu.layout<nhwc>>
}

// -----

// Mixed layouts among the operands, refused with a message naming the operand
// and both layouts.
func.func @mixed_layouts(%a: tensor<1x4x4x8xf32, #npu.layout<nhwc>>,
                         %b: tensor<1x4x4x8xf32>,
                         %d: tensor<1x4x4x8xf32, #npu.layout<nhwc>>)
    -> tensor<1x4x4x8xf32, #npu.layout<nhwc>> {
  // expected-error @+1 {{operands must not mix layouts, but lhs is nhwc and rhs is nchw (absent encoding)}}
  %0 = npu.add ins(%a, %b : tensor<1x4x4x8xf32, #npu.layout<nhwc>>,
                            tensor<1x4x4x8xf32>)
               outs(%d : tensor<1x4x4x8xf32, #npu.layout<nhwc>>)
       -> tensor<1x4x4x8xf32, #npu.layout<nhwc>>
  return %0 : tensor<1x4x4x8xf32, #npu.layout<nhwc>>
}

// =============================================================================
// The windowed arithmetic.
// =============================================================================

// -----

// An arithmetically impossible convolution: a 3 by 3 kernel over a 2 by 2 input
// with no padding. The implied extent is (2 + 0 + 0 - 3) / 1 + 1 = 0, and the
// diagnostic quotes it.
func.func @impossible_convolution(%x: tensor<1x3x2x2xf32>,
                                  %w: tensor<8x3x3x3xf32>,
                                  %d: tensor<1x8x1x1xf32>)
    -> tensor<1x8x1x1xf32> {
  // expected-error @+1 {{on the height axis, input extent 2 with pads 0 and 0, kernel 3, dilation 1 and stride 1 implies an output extent of 0, which is not a representable extent}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x2x2xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x1x1xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x1x1xf32>
  return %0 : tensor<1x8x1x1xf32>
}

// -----

// The ceil_mode = 1 drop rule, from the failing side. These are the exact
// parameters of the round trip test's drop case:
//
//   inputExtent 6, kernel 2, stride 3, padBegin 0, padEnd 1, dilation 1
//   ceil((6 + 0 + 1 - 2) / 3) + 1 = ceil(5 / 3) + 1 = 3
//   lastWindowStart = (3 - 1) * 3 = 6, and 6 >= 6 + 0, so the window is dropped
//   extent = 2
//
// A verifier that implemented the ceiling and forgot the drop would compute 3
// and accept the shape below. It computes 2 and refuses it, quoting both.
func.func @ceil_mode_without_the_drop_rule(%x: tensor<1x8x6x6xf32>,
                                           %d: tensor<1x8x3x3xf32>)
    -> tensor<1x8x3x3xf32> {
  // expected-error @+1 {{result spatial extents must be 2 by 2, computed from the input 6 by 6 with kernel 2 by 2, strides 3 and 3, dilations 1 and 1, pads 0, 0, 1, 1 and ceil_mode 1, but got 3 by 3}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x6x6xf32>)
                      outs(%d : tensor<1x8x3x3xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 3, 3>,
                       pads = array<i64: 0, 0, 1, 1>,
                       dilations = array<i64: 1, 1>, ceil_mode = 1 : i64}
       -> tensor<1x8x3x3xf32>
  return %0 : tensor<1x8x3x3xf32>
}

// -----

func.func @non_positive_stride(%x: tensor<1x8x8x8xf32>,
                               %d: tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{on the height axis, stride must be strictly positive, got 0}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 0, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

func.func @non_positive_dilation(%x: tensor<1x8x8x8xf32>,
                                 %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{on the height axis, dilation must be strictly positive, got 0}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 0, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

func.func @negative_pad(%x: tensor<1x8x8x8xf32>, %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{on the height axis, pads must be non negative, got -1 and 0}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: -1, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

// A pad at or beyond the kernel extent, which is what makes an all padding
// window representable. An average pool over such a window would divide by a
// contributing count of zero, so the shape is refused rather than the kernel
// being made to guard against it.
func.func @pad_not_smaller_than_kernel(%x: tensor<1x8x8x8xf32>,
                                       %d: tensor<1x8x5x4xf32>)
    -> tensor<1x8x5x4xf32> {
  // expected-error @+1 {{on the height axis, each pad must be strictly smaller than the kernel extent 2, got pads 2 and 0}}
  %0 = npu.avg_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x5x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 2, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x5x4xf32>
  return %0 : tensor<1x8x5x4xf32>
}

// -----

func.func @ceil_mode_out_of_range(%x: tensor<1x8x8x8xf32>,
                                  %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{ceil_mode must be 0 or 1, but got 2}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x4x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 2 : i64}
       -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

func.func @wrong_pool_output_extent(%x: tensor<1x8x8x8xf32>,
                                    %d: tensor<1x8x3x4xf32>)
    -> tensor<1x8x3x4xf32> {
  // expected-error @+1 {{result spatial extents must be 4 by 4, computed from the input 8 by 8 with kernel 2 by 2, strides 2 and 2, dilations 1 and 1, pads 0, 0, 0, 0 and ceil_mode 0, but got 3 by 4}}
  %0 = npu.max_pool2d ins(%x : tensor<1x8x8x8xf32>)
                      outs(%d : tensor<1x8x3x4xf32>)
                      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>,
                       pads = array<i64: 0, 0, 0, 0>,
                       dilations = array<i64: 1, 1>, ceil_mode = 0 : i64}
       -> tensor<1x8x3x4xf32>
  return %0 : tensor<1x8x3x4xf32>
}

// -----

func.func @wrong_conv_output_extent(%x: tensor<1x3x8x8xf32>,
                                    %w: tensor<8x3x3x3xf32>,
                                    %d: tensor<1x8x7x8xf32>)
    -> tensor<1x8x7x8xf32> {
  // expected-error @+1 {{result spatial extents must be 8 by 8, but got 7 by 8}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x7x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x7x8xf32>
  return %0 : tensor<1x8x7x8xf32>
}

// =============================================================================
// npu.conv2d, the rest.
// =============================================================================

// -----

func.func @bias_length(%x: tensor<1x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                       %b: tensor<7xf32>, %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the bias length must equal the output channel count 8, but got 7}}
  %0 = npu.conv2d ins(%x, %w, %b : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>,
                                   tensor<7xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @group_does_not_divide_input_channels(%x: tensor<1x6x8x8xf32>,
                                                %w: tensor<8x2x3x3xf32>,
                                                %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{group 4 must divide the input channel count 6}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x6x8x8xf32>, tensor<8x2x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 4 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @group_does_not_divide_output_channels(%x: tensor<1x8x8x8xf32>,
                                                 %w: tensor<6x2x3x3xf32>,
                                                 %d: tensor<1x6x8x8xf32>)
    -> tensor<1x6x8x8xf32> {
  // expected-error @+1 {{group 4 must divide the output channel count 6}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<6x2x3x3xf32>)
                  outs(%d : tensor<1x6x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 4 : i64}
       -> tensor<1x6x8x8xf32>
  return %0 : tensor<1x6x8x8xf32>
}

// -----

func.func @non_positive_group(%x: tensor<1x8x8x8xf32>, %w: tensor<8x8x3x3xf32>,
                              %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{group must be strictly positive, but got 0}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x8x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 0 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @filter_second_dimension(%x: tensor<1x8x8x8xf32>,
                                   %w: tensor<8x4x3x3xf32>,
                                   %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the filter's second dimension must be the input channel count divided by group, which is 8 / 4 = 2, but got 4}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x8x8x8xf32>, tensor<8x4x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 4 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

// The filter carries no layout, because there is no NHWC reading of a filter
// and tagging one would mean inventing a meaning nothing below reads.
func.func @filter_carries_layout(%x: tensor<1x3x8x8xf32>,
                                 %w: tensor<8x3x3x3xf32, #npu.layout<nhwc>>,
                                 %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the filter carries no layout, but got nhwc}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>,
                               tensor<8x3x3x3xf32, #npu.layout<nhwc>>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

// Batch is a first class dimension, so an operation whose input and result
// disagree on N is refused by name rather than producing a result nobody can
// explain.
func.func @batch_disagreement(%x: tensor<2x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                              %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{input and result must agree on the batch extent, but the input has 2 and the result has 1}}
  %0 = npu.conv2d ins(%x, %w : tensor<2x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @conv_wrong_result_channels(%x: tensor<1x3x8x8xf32>,
                                      %w: tensor<8x3x3x3xf32>,
                                      %d: tensor<1x7x8x8xf32>)
    -> tensor<1x7x8x8xf32> {
  // expected-error @+1 {{result channel count must be the filter's output channel count 8, but got 7}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x7x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x7x8x8xf32>
  return %0 : tensor<1x7x8x8xf32>
}

// -----

func.func @conv_rank(%x: tensor<1x3x8xf32>, %w: tensor<8x3x3x3xf32>,
                     %d: tensor<1x3x8xf32>) -> tensor<1x3x8xf32> {
  // expected-error @+1 {{expects a rank 4 input, but got rank 3}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x3x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x3x8xf32>
  return %0 : tensor<1x3x8xf32>
}

// -----

func.func @wrong_stride_count(%x: tensor<1x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                              %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{expects 2 strides, height then width, but got 3}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1, 1>, pads = array<i64: 1, 1, 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @wrong_pad_count(%x: tensor<1x3x8x8xf32>, %w: tensor<8x3x3x3xf32>,
                           %d: tensor<1x8x8x8xf32>) -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{expects 4 pads in ONNX order, top, left, bottom, right, but got 2}}
  %0 = npu.conv2d ins(%x, %w : tensor<1x3x8x8xf32>, tensor<8x3x3x3xf32>)
                  outs(%d : tensor<1x8x8x8xf32>)
                  {strides = array<i64: 1, 1>, pads = array<i64: 1, 1>,
                   dilations = array<i64: 1, 1>, group = 1 : i64}
       -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// =============================================================================
// npu.matmul
// =============================================================================

// -----

func.func @matmul_contracted_dimensions(%a: tensor<4x16xf32>,
                                        %b: tensor<15x10xf32>,
                                        %d: tensor<4x10xf32>)
    -> tensor<4x10xf32> {
  // expected-error @+1 {{the contracted dimensions must agree, but the lhs is 4 by 16 and the rhs is 15 by 10}}
  %0 = npu.matmul ins(%a, %b : tensor<4x16xf32>, tensor<15x10xf32>)
                  outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// -----

func.func @matmul_rank(%a: tensor<2x4x16xf32>, %b: tensor<16x10xf32>,
                       %d: tensor<2x4x16xf32>) -> tensor<2x4x16xf32> {
  // expected-error @+1 {{is rank 2 by rank 2, but got ranks 3, 2 and 3}}
  %0 = npu.matmul ins(%a, %b : tensor<2x4x16xf32>, tensor<16x10xf32>)
                  outs(%d : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// -----

func.func @matmul_bias_length(%a: tensor<4x16xf32>, %b: tensor<16x10xf32>,
                              %c: tensor<9xf32>, %d: tensor<4x10xf32>)
    -> tensor<4x10xf32> {
  // expected-error @+1 {{the bias length must equal the output column count 10, but got 9}}
  %0 = npu.matmul ins(%a, %b, %c : tensor<4x16xf32>, tensor<16x10xf32>,
                                   tensor<9xf32>)
                  outs(%d : tensor<4x10xf32>) -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

// =============================================================================
// The elementwise operations.
// =============================================================================

// -----

// The only broadcast npu.add represents is the rank 1 channel one. A
// `1 x C x 1 x 1` rhs broadcasts perfectly well in numpy and is refused here:
// the frontend materialises every broadcast it can at import time, normalises
// the channel shaped ones to rank 1, and rejects the rest by name, so accepting
// a second spelling of the same fact would let -npu-fuse-bias silently miss it.
func.func @add_does_not_broadcast_a_rank_four_operand(%a: tensor<1x8x4x4xf32>,
                                                      %b: tensor<1x8x1x1xf32>,
                                                      %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{the rhs must either have the result shape exactly or be the rank 1 channel broadcast of a rank 4 result}}
  %0 = npu.add ins(%a, %b : tensor<1x8x4x4xf32>, tensor<1x8x1x1xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

// A rank 1 rhs whose length is not the channel extent. 4 is the spatial extent
// here, so this is the shape a rule that broadcast over the wrong axis would
// have produced, and it must not verify.
func.func @add_channel_broadcast_wrong_length(%a: tensor<1x8x4x4xf32>,
                                              %b: tensor<4xf32>,
                                              %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{a rank 1 rhs is the channel broadcast, so its length must equal the result channel extent 8 under layout nchw (absent encoding), but the rhs is 'tensor<4xf32>'}}
  %0 = npu.add ins(%a, %b : tensor<1x8x4x4xf32>, tensor<4xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

// Under NHWC the channel extent is dimension 3. A rhs of length 4 matches
// dimension 1 and is still wrong, which is what makes reading the extent
// through the layout load bearing rather than cosmetic.
func.func @add_channel_broadcast_wrong_axis_under_nhwc(
    %a: tensor<1x4x4x8xf32, #npu.layout<nhwc>>, %b: tensor<4xf32>,
    %d: tensor<1x4x4x8xf32, #npu.layout<nhwc>>)
    -> tensor<1x4x4x8xf32, #npu.layout<nhwc>> {
  // expected-error @+1 {{result channel extent 8 under layout nhwc}}
  %0 = npu.add ins(%a, %b : tensor<1x4x4x8xf32, #npu.layout<nhwc>>,
                            tensor<4xf32>)
               outs(%d : tensor<1x4x4x8xf32, #npu.layout<nhwc>>)
       -> tensor<1x4x4x8xf32, #npu.layout<nhwc>>
  return %0 : tensor<1x4x4x8xf32, #npu.layout<nhwc>>
}

// -----

// The carve out is rank 4 only. A rank 2 result has no channel axis to
// broadcast over, so a rank 1 rhs against one is refused rather than read as a
// row or a column vector.
func.func @add_channel_broadcast_needs_a_rank_four_result(%a: tensor<4x8xf32>,
                                                          %b: tensor<8xf32>,
                                                          %d: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  // expected-error @+1 {{the rhs must either have the result shape exactly or be the rank 1 channel broadcast of a rank 4 result}}
  %0 = npu.add ins(%a, %b : tensor<4x8xf32>, tensor<8xf32>)
               outs(%d : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// Only the rhs may be the broadcast operand. One spelling of a channel
// broadcast is the whole reason the rule is asymmetric, and the frontend
// commutes rather than emitting this.
func.func @mul_does_not_broadcast_the_lhs(%a: tensor<8xf32>,
                                          %b: tensor<1x8x4x4xf32>,
                                          %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{the lhs is the activation and is never the broadcast operand}}
  %0 = npu.mul ins(%a, %b : tensor<8xf32>, tensor<1x8x4x4xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

func.func @mul_does_not_broadcast(%a: tensor<1x8x1x1xf32>,
                                  %b: tensor<1x8x4x4xf32>,
                                  %d: tensor<1x8x4x4xf32>)
    -> tensor<1x8x4x4xf32> {
  // expected-error @+1 {{the lhs is the activation and is never the broadcast operand}}
  %0 = npu.mul ins(%a, %b : tensor<1x8x1x1xf32>, tensor<1x8x4x4xf32>)
               outs(%d : tensor<1x8x4x4xf32>) -> tensor<1x8x4x4xf32>
  return %0 : tensor<1x8x4x4xf32>
}

// -----

func.func @relu_shape(%x: tensor<1x8x4x4xf32>, %d: tensor<1x8x2x2xf32>)
    -> tensor<1x8x2x2xf32> {
  // expected-error @+1 {{is elementwise, so the input shape must equal the result shape}}
  %0 = npu.relu ins(%x : tensor<1x8x4x4xf32>) outs(%d : tensor<1x8x2x2xf32>)
       -> tensor<1x8x2x2xf32>
  return %0 : tensor<1x8x2x2xf32>
}

// =============================================================================
// npu.reshape
// =============================================================================

// -----

func.func @reshape_element_count(%x: tensor<4x8xf32>) -> tensor<4x9xf32> {
  // expected-error @+1 {{element counts must match, but the input 'tensor<4x8xf32>' has 32 elements and the result 'tensor<4x9xf32>' has 36}}
  %0 = npu.reshape %x : tensor<4x8xf32> to tensor<4x9xf32>
  return %0 : tensor<4x9xf32>
}

// -----

// A reshape rearranges extents, so a layout encoding on either side would claim
// a meaning the rearrangement does not preserve.
func.func @reshape_carries_layout(%x: tensor<1x2x3x4xf32, #npu.layout<nhwc>>)
    -> tensor<24xf32> {
  // expected-error @+1 {{carries no layout encoding on either side, because a reshape does not preserve one}}
  %0 = npu.reshape %x : tensor<1x2x3x4xf32, #npu.layout<nhwc>> to tensor<24xf32>
  return %0 : tensor<24xf32>
}

// =============================================================================
// npu.transpose
// =============================================================================

// -----

func.func @transpose_not_a_permutation(%x: tensor<1x2x3x4xf32>,
                                       %d: tensor<1x3x4x2xf32>)
    -> tensor<1x3x4x2xf32> {
  // expected-error @+1 {{the permutation must be a permutation, but 2 appears more than once}}
  %0 = npu.transpose ins(%x : tensor<1x2x3x4xf32>)
                     outs(%d : tensor<1x3x4x2xf32>)
                     {permutation = array<i64: 0, 2, 2, 1>}
       -> tensor<1x3x4x2xf32>
  return %0 : tensor<1x3x4x2xf32>
}

// -----

func.func @transpose_permutation_length(%x: tensor<1x2x3x4xf32>,
                                        %d: tensor<1x3x4x2xf32>)
    -> tensor<1x3x4x2xf32> {
  // expected-error @+1 {{the permutation must have exactly the result rank 4 entries, but got 3}}
  %0 = npu.transpose ins(%x : tensor<1x2x3x4xf32>)
                     outs(%d : tensor<1x3x4x2xf32>)
                     {permutation = array<i64: 0, 2, 3>}
       -> tensor<1x3x4x2xf32>
  return %0 : tensor<1x3x4x2xf32>
}

// -----

func.func @transpose_index_out_of_range(%x: tensor<1x2x3x4xf32>,
                                        %d: tensor<1x3x4x2xf32>)
    -> tensor<1x3x4x2xf32> {
  // expected-error @+1 {{permutation entry 1 is 7, which is outside the range 0 to 3}}
  %0 = npu.transpose ins(%x : tensor<1x2x3x4xf32>)
                     outs(%d : tensor<1x3x4x2xf32>)
                     {permutation = array<i64: 0, 7, 3, 1>}
       -> tensor<1x3x4x2xf32>
  return %0 : tensor<1x3x4x2xf32>
}

// -----

func.func @transpose_extent_mismatch(%x: tensor<1x2x3x4xf32>,
                                     %d: tensor<1x3x4x4xf32>)
    -> tensor<1x3x4x4xf32> {
  // expected-error @+1 {{result extent 3 must be input extent 1, which is 2, but got 4}}
  %0 = npu.transpose ins(%x : tensor<1x2x3x4xf32>)
                     outs(%d : tensor<1x3x4x4xf32>)
                     {permutation = array<i64: 0, 2, 3, 1>}
       -> tensor<1x3x4x4xf32>
  return %0 : tensor<1x3x4x4xf32>
}

// =============================================================================
// npu.concat
// =============================================================================

// -----

func.func @concat_extents_do_not_sum(%a: tensor<1x4x8x8xf32>,
                                     %b: tensor<1x6x8x8xf32>,
                                     %d: tensor<1x11x8x8xf32>)
    -> tensor<1x11x8x8xf32> {
  // expected-error @+1 {{the input extents along axis 1 sum to 10, but the result extent there is 11}}
  %0 = npu.concat ins(%a, %b : tensor<1x4x8x8xf32>, tensor<1x6x8x8xf32>)
                  outs(%d : tensor<1x11x8x8xf32>) {axis = 1 : i64}
       -> tensor<1x11x8x8xf32>
  return %0 : tensor<1x11x8x8xf32>
}

// -----

func.func @concat_off_axis_mismatch(%a: tensor<1x4x8x8xf32>,
                                    %b: tensor<1x6x8x9xf32>,
                                    %d: tensor<1x10x8x8xf32>)
    -> tensor<1x10x8x8xf32> {
  // expected-error @+1 {{input 1 has extent 9 on axis 3, but the result has 8, and extents must match on every axis except the concatenation axis 1}}
  %0 = npu.concat ins(%a, %b : tensor<1x4x8x8xf32>, tensor<1x6x8x9xf32>)
                  outs(%d : tensor<1x10x8x8xf32>) {axis = 1 : i64}
       -> tensor<1x10x8x8xf32>
  return %0 : tensor<1x10x8x8xf32>
}

// -----

func.func @concat_axis_out_of_range(%a: tensor<1x4x8x8xf32>,
                                    %d: tensor<1x4x8x8xf32>)
    -> tensor<1x4x8x8xf32> {
  // expected-error @+1 {{axis must be in the range 0 to 3 for a rank 4 result, but got 7}}
  %0 = npu.concat ins(%a : tensor<1x4x8x8xf32>)
                  outs(%d : tensor<1x4x8x8xf32>) {axis = 7 : i64}
       -> tensor<1x4x8x8xf32>
  return %0 : tensor<1x4x8x8xf32>
}

// -----

func.func @concat_rank_mismatch(%a: tensor<1x4x8x8xf32>, %b: tensor<1x6x8xf32>,
                                %d: tensor<1x10x8x8xf32>)
    -> tensor<1x10x8x8xf32> {
  // expected-error @+1 {{input 1 has rank 3, but every input must have the result rank 4}}
  %0 = npu.concat ins(%a, %b : tensor<1x4x8x8xf32>, tensor<1x6x8xf32>)
                  outs(%d : tensor<1x10x8x8xf32>) {axis = 1 : i64}
       -> tensor<1x10x8x8xf32>
  return %0 : tensor<1x10x8x8xf32>
}

// =============================================================================
// npu.batch_norm
// =============================================================================

// -----

func.func @batch_norm_parameter_length(%x: tensor<2x8x4x4xf32>,
                                       %g: tensor<7xf32>,
                                       %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // expected-error @+1 {{gamma must have length equal to the channel count 8, but got 7}}
  %0 = npu.batch_norm ins(%x, %g, %g, %g, %g : tensor<2x8x4x4xf32>,
                                               tensor<7xf32>, tensor<7xf32>,
                                               tensor<7xf32>, tensor<7xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {epsilon = 1.000000e-05 : f32} -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// -----

func.func @batch_norm_parameter_rank(%x: tensor<2x8x4x4xf32>,
                                     %g: tensor<1x8xf32>,
                                     %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // expected-error @+1 {{gamma must be rank 1, but got rank 2}}
  %0 = npu.batch_norm ins(%x, %g, %g, %g, %g : tensor<2x8x4x4xf32>,
                                               tensor<1x8xf32>,
                                               tensor<1x8xf32>,
                                               tensor<1x8xf32>,
                                               tensor<1x8xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {epsilon = 1.000000e-05 : f32} -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// -----

func.func @batch_norm_negative_epsilon(%x: tensor<2x8x4x4xf32>,
                                       %g: tensor<8xf32>,
                                       %d: tensor<2x8x4x4xf32>)
    -> tensor<2x8x4x4xf32> {
  // expected-error @+1 {{epsilon must be finite and non negative, but got -1}}
  %0 = npu.batch_norm ins(%x, %g, %g, %g, %g : tensor<2x8x4x4xf32>,
                                               tensor<8xf32>, tensor<8xf32>,
                                               tensor<8xf32>, tensor<8xf32>)
                      outs(%d : tensor<2x8x4x4xf32>)
                      {epsilon = -1.000000e+00 : f32} -> tensor<2x8x4x4xf32>
  return %0 : tensor<2x8x4x4xf32>
}

// =============================================================================
// npu.fused_op and npu.yield
// =============================================================================

// -----

func.func @fused_op_block_argument_count(%x: tensor<1x8x8x8xf32>,
                                         %d: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the region takes one block argument per operand, so it must have 2 block arguments, but it has 1}}
  %0 = npu.fused_op ins(%x, %d : tensor<1x8x8x8xf32>, tensor<1x8x8x8xf32>) {
  ^bb0(%a: tensor<1x8x8x8xf32>):
    npu.yield %a : tensor<1x8x8x8xf32>
  } -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @fused_op_block_argument_type(%x: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{block argument 0 has type 'tensor<1x8x4x4xf32>', which differs from operand 0 of type 'tensor<1x8x8x8xf32>'}}
  %0 = npu.fused_op ins(%x : tensor<1x8x8x8xf32>) {
  ^bb0(%a: tensor<1x8x4x4xf32>):
    %r = npu.reshape %a : tensor<1x8x4x4xf32> to tensor<1x8x8x8xf32>
    npu.yield %r : tensor<1x8x8x8xf32>
  } -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @fused_op_foreign_dialect(%x: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the region holds only npu dialect operations, but it contains arith.addf}}
  %0 = npu.fused_op ins(%x : tensor<1x8x8x8xf32>) {
  ^bb0(%a: tensor<1x8x8x8xf32>):
    %m = arith.addf %a, %a : tensor<1x8x8x8xf32>
    npu.yield %m : tensor<1x8x8x8xf32>
  } -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// -----

func.func @fused_op_yielded_type(%x: tensor<1x8x8x8xf32>)
    -> tensor<1x8x8x8xf32> {
  // expected-error @+1 {{the yielded type 'tensor<1x512xf32>' must equal the result type 'tensor<1x8x8x8xf32>'}}
  %0 = npu.fused_op ins(%x : tensor<1x8x8x8xf32>) {
  ^bb0(%a: tensor<1x8x8x8xf32>):
    %r = npu.reshape %a : tensor<1x8x8x8xf32> to tensor<1x512xf32>
    npu.yield %r : tensor<1x512xf32>
  } -> tensor<1x8x8x8xf32>
  return %0 : tensor<1x8x8x8xf32>
}

// =============================================================================
// npu.constant
// =============================================================================

// -----

func.func @constant_type_disagreement() -> tensor<2x2xf32> {
  // expected-error @+1 {{value attribute type 'tensor<3x3xf32>' must equal the result type 'tensor<2x2xf32>'}}
  %0 = "npu.constant"() {value = dense<1.0> : tensor<3x3xf32>}
       : () -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// =============================================================================
// A dynamic extent, refused at the type level.
//
// The regression cases for D-0015. NPUTypes.td always claimed that a dynamic
// dimension was refused at the type level, and until the constraints changed
// from RankedTensorOf to StaticShapeTensorOf that claim was a comment nothing
// enforced. The last two cases abort with an LLVM stack trace and no diagnostic
// at all against the old spelling, because npu.reshape reaches
// getNumElements(), which asserts rather than answering on a dynamic shape.
//
// There are three because they test different halves. The first proves the
// constraint fires on an ordinary compute operation; the other two prove it
// fires on the operation that used to crash, from each side of it, since a
// reshape has a dynamic extent to be refused on either the operand or the
// result.
// =============================================================================

// -----

func.func @relu_with_a_dynamic_extent(%x: tensor<?x8x4x4xf32>, %n: index)
    -> tensor<?x8x4x4xf32> {
  %d = tensor.empty(%n) : tensor<?x8x4x4xf32>
  // expected-error @+1 {{statically shaped tensor}}
  %0 = npu.relu ins(%x : tensor<?x8x4x4xf32>) outs(%d : tensor<?x8x4x4xf32>)
       -> tensor<?x8x4x4xf32>
  return %0 : tensor<?x8x4x4xf32>
}

// -----

func.func @reshape_from_a_dynamic_extent(%x: tensor<?x4xf32>)
    -> tensor<4x4xf32> {
  // expected-error @+1 {{statically shaped tensor}}
  %0 = npu.reshape %x : tensor<?x4xf32> to tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

func.func @reshape_to_a_dynamic_extent(%x: tensor<4x4xf32>)
    -> tensor<?x4xf32> {
  // expected-error @+1 {{statically shaped tensor}}
  %0 = npu.reshape %x : tensor<4x4xf32> to tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
