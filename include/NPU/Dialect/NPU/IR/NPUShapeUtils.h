//===- NPUShapeUtils.h - Shared shape arithmetic ----------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The windowed output extent arithmetic, in exactly one place.
//
// Four operations in this dialect compute an output spatial extent from an
// input extent, a kernel, a stride, a dilation, and a pair of pads: conv2d,
// max_pool2d, avg_pool2d, and, through the inference path, every one of them
// again. Each of those four has both a verifier and an `inferReturnTypes`, so
// without a shared helper there would be eight opportunities to write the
// formula and eight opportunities to write it differently. A shape the
// verifier accepts and the inference function would have computed differently
// is not a diagnosable error; it is a silently wrong tensor three passes later.
//
// So the formula lives here, once, and both paths call it.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_DIALECT_NPU_IR_NPUSHAPEUTILS_H
#define NPU_DIALECT_NPU_IR_NPUSHAPEUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::npu {

/// One spatial axis of a windowed operation. Everything the extent formula
/// needs and nothing it does not.
struct WindowParams {
  int64_t inputExtent = 0;
  int64_t kernelExtent = 0;
  int64_t stride = 1;
  int64_t dilation = 1;
  int64_t padBegin = 0;
  int64_t padEnd = 0;
  /// True only for the pooling operations with `ceil_mode = 1`. Convolution
  /// has no ceil mode in ONNX and this dialect does not invent one, so the
  /// convolution path always passes false.
  bool ceilMode = false;
};

/// Why a set of window parameters is not representable. Returned rather than
/// diagnosed here, because this header has no operation to name in a
/// diagnostic and a helper that emits its own errors cannot be reused by an
/// inference function that must stay silent.
enum class WindowError {
  None,
  NonPositiveStride,
  NonPositiveDilation,
  NonPositiveKernel,
  NegativePad,
  /// A pad at or beyond the kernel extent. Refused because it is what makes an
  /// all padding window representable, and an all padding average pool divides
  /// by a contributing count of zero.
  PadNotSmallerThanKernel,
  /// The arithmetic completed and produced an extent of zero or less, which
  /// means the window does not fit the padded input even once.
  ImpossibleExtent,
};

/// The result of the extent computation. `extent` is meaningful only when
/// `error` is `WindowError::None`; when the arithmetic produced a
/// non positive extent, `extent` still carries the number it produced, because
/// quoting the implied extent is what makes the diagnostic useful.
struct WindowResult {
  int64_t extent = 0;
  WindowError error = WindowError::None;

  bool ok() const { return error == WindowError::None; }
};

/// Computes the output extent of one spatial axis.
///
/// The rule, resolved against the opset 19 pooling specification rather than
/// the opset 11 one:
///
///   effectiveKernel = dilation * (kernel - 1) + 1
///   numerator       = inputExtent + padBegin + padEnd - effectiveKernel
///
///   ceil_mode = 0:  extent = floor(numerator / stride) + 1
///   ceil_mode = 1:  extent = ceil(numerator / stride) + 1, and then one
///                   further rule: a sliding window whose first element would
///                   start inside the right padded region is dropped, which
///                   can reduce the extent by one.
///
/// That last rule is the one routinely missed. It changes the shape rather
/// than the values, so a disagreement with onnxruntime surfaces as a broadcast
/// error several passes downstream with nothing pointing at the pooling
/// operation that caused it. It is implemented explicitly below rather than
/// left to emerge from the arithmetic.
WindowResult computeWindowedExtent(const WindowParams &params);

/// A human readable clause naming what is wrong, for the verifier to embed in
/// its own message. Returns an empty string for `WindowError::None`.
std::string describeWindowError(WindowError error, const WindowParams &params);

/// Computes every spatial output extent of a windowed operation. Returns the
/// first failure it meets, with `failedAxis` set to that axis, so the
/// diagnostic can name which axis is wrong rather than which operation.
struct WindowedShapeResult {
  SmallVector<int64_t> extents;
  WindowError error = WindowError::None;
  unsigned failedAxis = 0;
  /// The parameters of the failing axis, so a caller can quote the numbers.
  WindowParams failedParams;

  bool ok() const { return error == WindowError::None; }
};

WindowedShapeResult computeWindowedShape(ArrayRef<int64_t> inputSpatial,
                                         ArrayRef<int64_t> kernel,
                                         ArrayRef<int64_t> strides,
                                         ArrayRef<int64_t> dilations,
                                         ArrayRef<int64_t> pads, bool ceilMode);

//===----------------------------------------------------------------------===//
// Layout aware shape accessors.
//===----------------------------------------------------------------------===//

/// Which axis of a rank 4 tensor holds the channel count, under a layout.
/// An absent encoding means NCHW, which is why this takes the tensor and not
/// just an attribute: reading a shape without reading its encoding is always
/// wrong, and making the layout an unavoidable argument is how that is
/// enforced rather than remembered.
int64_t getBatchExtent(RankedTensorType type);
int64_t getChannelExtent(RankedTensorType type);

/// The two spatial extents, height then width, in that order whatever the
/// layout says about where they sit.
SmallVector<int64_t> getSpatialExtents(RankedTensorType type);

/// Builds a rank 4 shape from the four extents, placed per the layout of
/// `like`, and carrying the same encoding.
SmallVector<int64_t> buildNCHWLikeShape(RankedTensorType like, int64_t batch,
                                        int64_t channels,
                                        ArrayRef<int64_t> spatial);

/// True when the tensor is NHWC. An absent encoding is NCHW and so returns
/// false, which is the whole content of the "absent encoding means NCHW" rule
/// as far as any consumer is concerned.
bool isNHWC(RankedTensorType type);

/// The name of a tensor's layout, for a diagnostic: "nchw" or "nhwc". An
/// absent encoding prints as "nchw (absent encoding)", because a message that
/// said only "nchw" would leave the reader wondering which of the two ways of
/// writing it they had.
std::string describeLayout(RankedTensorType type);

/// True when two tensors carry the same layout, treating an absent encoding as
/// NCHW so that the two ways of writing NCHW compare equal.
bool sameLayout(RankedTensorType lhs, RankedTensorType rhs);

} // namespace mlir::npu

#endif // NPU_DIALECT_NPU_IR_NPUSHAPEUTILS_H
