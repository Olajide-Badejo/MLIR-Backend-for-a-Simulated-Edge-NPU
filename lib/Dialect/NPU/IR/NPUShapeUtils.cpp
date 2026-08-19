//===- NPUShapeUtils.cpp - Shared shape arithmetic --------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"
#include "NPU/Dialect/NPU/IR/NPUAttrs.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

/// Integer division rounding towards negative infinity. The C++ built in
/// rounds towards zero, which differs for a negative numerator, and a negative
/// numerator is exactly the case the impossible convolution test produces. Two
/// wrong shapes from one formula would be the worst possible outcome of this
/// file, so the rounding is spelled out rather than inherited.
int64_t floorDivide(int64_t numerator, int64_t denominator) {
  int64_t quotient = numerator / denominator;
  if ((numerator % denominator != 0) && ((numerator < 0) != (denominator < 0)))
    --quotient;
  return quotient;
}

/// Integer division rounding towards positive infinity.
int64_t ceilDivide(int64_t numerator, int64_t denominator) {
  int64_t quotient = numerator / denominator;
  if ((numerator % denominator != 0) && ((numerator < 0) == (denominator < 0)))
    ++quotient;
  return quotient;
}

} // namespace

WindowResult mlir::npu::computeWindowedExtent(const WindowParams &params) {
  WindowResult result;

  if (params.stride <= 0) {
    result.error = WindowError::NonPositiveStride;
    return result;
  }
  if (params.dilation <= 0) {
    result.error = WindowError::NonPositiveDilation;
    return result;
  }
  if (params.kernelExtent <= 0) {
    result.error = WindowError::NonPositiveKernel;
    return result;
  }
  if (params.padBegin < 0 || params.padEnd < 0) {
    result.error = WindowError::NegativePad;
    return result;
  }
  // Each pad strictly smaller than the corresponding kernel extent. This is
  // what makes an all padding window unrepresentable at the tensor level: an
  // average pool over a window with no real elements would divide by a
  // contributing count of zero, and refusing the shape removes that divide by
  // zero at its source rather than guarding it in the kernel.
  if (params.padBegin >= params.kernelExtent ||
      params.padEnd >= params.kernelExtent) {
    result.error = WindowError::PadNotSmallerThanKernel;
    return result;
  }

  const int64_t effectiveKernel =
      params.dilation * (params.kernelExtent - 1) + 1;
  const int64_t numerator =
      params.inputExtent + params.padBegin + params.padEnd - effectiveKernel;

  int64_t extent = params.ceilMode
                       ? ceilDivide(numerator, params.stride) + 1
                       : floorDivide(numerator, params.stride) + 1;

  // The opset 19 rule, and the part that is routinely missed.
  //
  // With ceil_mode = 1 the ceiling can produce one more window than the input
  // actually supports: the last window's first element lands at
  //
  //     (extent - 1) * stride - padBegin
  //
  // in input coordinates, and if that is at or past the end of the real input
  // then every element the window would read comes from the right padded
  // region. ONNX drops such a window. It reduces the extent by at most one,
  // because two consecutive windows starting in the right padding would need a
  // right pad of at least one stride beyond the input, which the pad bound
  // above already refuses.
  //
  // This changes the shape rather than the values, so getting it wrong shows up
  // as a broadcast error several passes downstream with nothing pointing back
  // here. It is therefore implemented as its own explicit step, and the test
  // suite carries a case whose arithmetic is written out in a comment.
  if (params.ceilMode && extent > 0) {
    const int64_t lastWindowStart = (extent - 1) * params.stride;
    if (lastWindowStart >= params.inputExtent + params.padBegin)
      --extent;
  }

  result.extent = extent;
  if (extent <= 0) {
    result.error = WindowError::ImpossibleExtent;
    return result;
  }
  return result;
}

std::string mlir::npu::describeWindowError(WindowError error,
                                           const WindowParams &params) {
  std::string message;
  llvm::raw_string_ostream os(message);
  switch (error) {
  case WindowError::None:
    break;
  case WindowError::NonPositiveStride:
    os << "stride must be strictly positive, got " << params.stride;
    break;
  case WindowError::NonPositiveDilation:
    os << "dilation must be strictly positive, got " << params.dilation;
    break;
  case WindowError::NonPositiveKernel:
    os << "kernel extent must be strictly positive, got " << params.kernelExtent;
    break;
  case WindowError::NegativePad:
    os << "pads must be non negative, got " << params.padBegin << " and "
       << params.padEnd;
    break;
  case WindowError::PadNotSmallerThanKernel:
    os << "each pad must be strictly smaller than the kernel extent "
       << params.kernelExtent << ", got pads " << params.padBegin << " and "
       << params.padEnd;
    break;
  case WindowError::ImpossibleExtent:
    os << "input extent " << params.inputExtent << " with pads "
       << params.padBegin << " and " << params.padEnd << ", kernel "
       << params.kernelExtent << ", dilation " << params.dilation
       << " and stride " << params.stride << " implies an output extent of "
       << (params.inputExtent + params.padBegin + params.padEnd -
           (params.dilation * (params.kernelExtent - 1) + 1)) /
                  params.stride +
              1
       << ", which is not a representable extent";
    break;
  }
  return message;
}

WindowedShapeResult mlir::npu::computeWindowedShape(
    ArrayRef<int64_t> inputSpatial, ArrayRef<int64_t> kernel,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> pads, bool ceilMode) {
  WindowedShapeResult result;
  const unsigned rank = inputSpatial.size();
  result.extents.reserve(rank);

  for (unsigned axis = 0; axis < rank; ++axis) {
    WindowParams params;
    params.inputExtent = inputSpatial[axis];
    params.kernelExtent = kernel[axis];
    params.stride = strides[axis];
    params.dilation = dilations[axis];
    // pads are in ONNX order: every begin, then every end.
    params.padBegin = pads[axis];
    params.padEnd = pads[rank + axis];
    params.ceilMode = ceilMode;

    WindowResult axisResult = computeWindowedExtent(params);
    if (!axisResult.ok()) {
      result.error = axisResult.error;
      result.failedAxis = axis;
      result.failedParams = params;
      return result;
    }
    result.extents.push_back(axisResult.extent);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Layout aware shape accessors.
//===----------------------------------------------------------------------===//

bool mlir::npu::isNHWC(RankedTensorType type) {
  auto layout = dyn_cast_or_null<LayoutAttr>(type.getEncoding());
  return layout && layout.isNHWC();
}

int64_t mlir::npu::getBatchExtent(RankedTensorType type) {
  // Batch is dimension zero under both layouts this dialect has, which is why
  // there is no branch here. It stays a named function anyway, so that a third
  // layout would be added in one place rather than found by grep.
  return type.getDimSize(0);
}

int64_t mlir::npu::getChannelExtent(RankedTensorType type) {
  return isNHWC(type) ? type.getDimSize(3) : type.getDimSize(1);
}

SmallVector<int64_t> mlir::npu::getSpatialExtents(RankedTensorType type) {
  if (isNHWC(type))
    return {type.getDimSize(1), type.getDimSize(2)};
  return {type.getDimSize(2), type.getDimSize(3)};
}

SmallVector<int64_t> mlir::npu::buildNCHWLikeShape(RankedTensorType like,
                                                   int64_t batch,
                                                   int64_t channels,
                                                   ArrayRef<int64_t> spatial) {
  if (isNHWC(like))
    return {batch, spatial[0], spatial[1], channels};
  return {batch, channels, spatial[0], spatial[1]};
}

std::string mlir::npu::describeLayout(RankedTensorType type) {
  auto layout = dyn_cast_or_null<LayoutAttr>(type.getEncoding());
  if (!layout)
    return "nchw (absent encoding)";
  return layout.isNHWC() ? "nhwc" : "nchw";
}

bool mlir::npu::sameLayout(RankedTensorType lhs, RankedTensorType rhs) {
  return isNHWC(lhs) == isNHWC(rhs);
}
