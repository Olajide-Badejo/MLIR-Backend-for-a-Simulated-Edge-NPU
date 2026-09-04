//===- Kernels.cpp - the arithmetic of the machine ------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// One kernel per opcode, and the dispatch table they hang from.
//
// **Every kernel indexes its operands through their strides.** That is the
// obligation `docs/ARCHITECTURE.md` placed on this phase at P4 and it is not a
// stylistic preference: the rank 1 channel broadcast of ADR 0005 arrives as a
// stride 0 operand and needs no special case at all, and an NHWC buffer arrives
// as NCHW extents with permuted strides. There is **no layout specific kernel
// variant** anywhere in this file, and there must never be one: a second set of
// kernels is a second set of kernels to keep in agreement with the first.
//
// **The result is contiguous.** The `Instruction` record carries operand
// strides and no result strides, so a result is written in row major order over
// its declared extents. That is not an omission in the format: an instruction
// writes the buffer the allocator gave it, and the allocator gives out flat
// spans.
//
// **Batch is first class.** Section 7.3: there is no `int64_t n = 0` shortcut
// in any kernel here, no path that is faster when N is 1, and every extent is
// derived from a shape rather than assumed.
//
// **The reductions are sequential and in their original order.** Section 10.3
// parallelises the convolution over the batch and output channel dimensions
// only, leaving the loops over input channel and kernel window strictly in
// order, because the accumulation order inside each output element determines
// its last bits and every golden file depends on it. An im2col plus GEMM
// restructuring is explicitly rejected: it would move every golden for a host
// speedup.
//
// **Whether that parallel region exists in a given build is a question about
// this file's compile line, and the answer is `kernelsUseOpenMP()` at the
// bottom.** It was written at P7 and first compiled at P12: the OpenMP usage
// requirement had been attached to the `NPUSimulator` target and not to the
// object library these sources compile in, so `-fopenmp` reached every consumer
// of the library and never reached the library. D-0047, and
// `lib/Simulator/CMakeLists.txt` carries the fix and the reason.
//
//===----------------------------------------------------------------------===//

#include "Kernels.h"

#include "NPU/Simulator/CostModel.h"
#include "NPU/Simulator/Simulator.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

using namespace nbin;
using namespace nbin::detail;

namespace {

//===----------------------------------------------------------------------===//
// Shared arithmetic.
//===----------------------------------------------------------------------===//

/// The product of a shape, or zero when it is empty. The extents have already
/// been through `checkedElementCount` on the validated path, and on the
/// unvalidated path an absurd product produces an address the accessors refuse.
int64_t elementsIn(llvm::ArrayRef<int64_t> shape) {
  // Empty is zero rather than one. A rank 0 buffer is not a shape this machine
  // has an address for, and the empty product convention would make every
  // kernel below think it had one element to move.
  if (shape.empty())
    return 0;
  int64_t product = 1;
  for (int64_t extent : shape) {
    if (extent <= 0)
      return 0;
    // The same overflow safe form the format uses: test before multiplying,
    // never after, because a guard that multiplies first is itself the
    // overflow.
    if (extent > Program::kShapeLimit / product)
      return 0;
    product *= extent;
  }
  return product;
}

/// The dot product of an index with a stride vector, which is how every operand
/// in this file is addressed.
int64_t offsetOf(llvm::ArrayRef<int64_t> index, llvm::ArrayRef<int64_t> strides) {
  int64_t offset = 0;
  for (size_t axis = 0; axis < strides.size(); ++axis)
    offset += index[axis] * strides[axis];
  return offset;
}

/// Walks an index space in row major order.
///
/// It exists so that the shape kernels are written once over any rank rather
/// than once per rank, which is the only way `TRANSPOSE` and `CONCAT` cover the
/// rank 4 NCHW to NHWC case and the rank 2 case with one body.
class Odometer {
public:
  explicit Odometer(llvm::ArrayRef<int64_t> shape)
      : shape(shape), index(shape.size(), 0) {}

  llvm::ArrayRef<int64_t> current() const { return index; }

  /// Advances, and says whether there is anything left.
  bool next() {
    for (size_t axis = shape.size(); axis-- > 0;) {
      if (++index[axis] < shape[axis])
        return true;
      index[axis] = 0;
    }
    return false;
  }

private:
  llvm::ArrayRef<int64_t> shape;
  llvm::SmallVector<int64_t> index;
};

/// Refuses an instruction whose element type this phase has no kernel for.
///
/// The integer kernels are Section 14's and land at Phase P14. Saying so by
/// name is the difference between a phase that has not arrived and a bug.
bool requireF32(Machine &machine, const Instruction &instruction,
                const char *name) {
  if (instruction.resultElementType == ElemType::F32)
    return true;
  machine.recordTrap(
      std::string(name) + " has no " +
      elemTypeName(instruction.resultElementType) +
      " kernel in this build. The integer kernels are Section 14's and land at "
      "Phase P14; Phase P7 implements the f32 set only");
  return false;
}

/// Refuses an operand whose rank does not match the space the kernel walks.
///
/// A validated program cannot get here for the opcodes whose shape relation
/// `Program::validate()` checks. The elementwise opcodes are not among them:
/// Section 9.2 has no check that an `ADD` operand has the result's rank,
/// because the rank alone says nothing about whether the operand covers the
/// result and the operand extent arithmetic already answers the question that
/// matters. So the kernel refuses rather than indexing off the end of a stride
/// vector.
bool requireRank(Machine &machine, const Operand &operand, size_t rank,
                 const char *name, int operandIndex) {
  if (operand.shape.size() == rank && operand.strides.size() == rank)
    return true;
  machine.recordTrap(std::string(name) + " operand " +
                     std::to_string(operandIndex) + " has rank " +
                     std::to_string(operand.shape.size()) +
                     " and the result has rank " + std::to_string(rank));
  return false;
}

/// Copies one element of `size` bytes between two addresses, through the
/// checked accessors at both ends.
bool copyElement(Machine &machine, MemSpace toSpace, int64_t toBase,
                 int64_t toElement, MemSpace fromSpace, int64_t fromBase,
                 int64_t fromElement, int64_t size, const char *what) {
  // The scaling is guarded rather than assumed. An element offset comes from a
  // stride vector, and on the unvalidated path a stride vector is whatever the
  // caller built.
  const int64_t limit = Program::kShapeLimit;
  if (toElement > limit || toElement < -limit || fromElement > limit ||
      fromElement < -limit) {
    machine.recordTrap(std::string("unaddressable ") + what +
                       ": an element offset is beyond the shape limit");
    return false;
  }
  const uint8_t *source =
      machine.readBytes(fromSpace, fromBase + fromElement * size, size, what);
  if (!source)
    return false;
  uint8_t *destination =
      machine.writeBytes(toSpace, toBase + toElement * size, size, what);
  if (!destination)
    return false;
  std::memcpy(destination, source, static_cast<size_t>(size));
  return true;
}

/// Applies the instruction's fused activation to a computed value.
float activate(const Instruction &instruction, float value) {
  return instruction.activation == Activation::Relu ? std::max(value, 0.0f)
                                                    : value;
}

//===----------------------------------------------------------------------===//
// The data movement kernels.
//
// These four are element size generic rather than f32 only, and that is not
// scope creep. A move performs no arithmetic, so there is nothing about it that
// waits for Phase P14's integer kernels, and writing them f32 only would mean a
// `DMA_LOAD` of an i8 buffer trapping for a reason that has nothing to do with
// what it does.
//===----------------------------------------------------------------------===//

/// `DMA_LOAD` and `DMA_STORE`, which are the same gather with the spaces
/// exchanged.
///
/// The operand carries strides and the result does not, so a transfer whose
/// source is strided gathers into a contiguous destination in the operand's own
/// shape order. That is what a descriptor driven DMA does, and it is why the
/// stride term of Section 5.5 exists: without it a strided NCHW gather and a
/// contiguous NHWC burst would cost exactly the same.
///
/// It does **not** permute dimensions. A DMA moves bytes and does not change a
/// layout on the way; `TRANSPOSE` is the only operation on this machine that
/// changes one.
KernelCost transfer(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  const Operand &source = instruction.operands.front();
  const int64_t elementSize = elementByteSize(instruction.resultElementType);
  const int64_t elements = elementsIn(instruction.resultShape);
  if (elements <= 0 || elementSize <= 0)
    return cost;

  const int64_t innermostStride =
      source.strides.empty() ? 1 : source.strides.back();
  cost.cycles = dmaCycles(elements * elementSize, elements, innermostStride);

  if (!requireRank(machine, source, instruction.resultShape.size(),
                   opcodeName(instruction.opcode), 0))
    return cost;
  if (!std::equal(source.shape.begin(), source.shape.end(),
                  instruction.resultShape.begin())) {
    machine.recordTrap(std::string(opcodeName(instruction.opcode)) +
                       " moves a buffer whose extents differ from its "
                       "result's, and a DMA does not reshape on the way");
    return cost;
  }

  // The contiguous case is a single checked span rather than one check per
  // element. It is the common case by a wide margin and it is the one the
  // sentence "a DMA moves bytes" describes literally.
  bool contiguous = true;
  int64_t expected = 1;
  for (size_t axis = source.shape.size(); axis-- > 0;) {
    if (source.strides[axis] != expected) {
      contiguous = false;
      break;
    }
    expected *= source.shape[axis];
  }

  if (contiguous) {
    machine.copy(instruction.resultSpace, instruction.resultAddress,
                 source.space, source.address, elements * elementSize);
    return cost;
  }

  Odometer walk(instruction.resultShape);
  int64_t destination = 0;
  do {
    if (!copyElement(machine, instruction.resultSpace,
                     instruction.resultAddress, destination, source.space,
                     source.address, offsetOf(walk.current(), source.strides),
                     elementSize, "DMA element"))
      return cost;
    ++destination;
  } while (walk.next());
  return cost;
}

/// `RESHAPE`: the same elements under different extents.
///
/// The operand is walked in its own row major order and the result is written
/// in its own, which is what makes the operation a copy rather than a view:
/// below this level the allocator has already decided the two buffers are
/// different spans, and an operand that arrives strided has to be gathered.
KernelCost kernelRESHAPE(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  const Operand &source = instruction.operands.front();
  const int64_t elementSize = elementByteSize(instruction.resultElementType);
  const int64_t elements = elementsIn(instruction.resultShape);
  cost.cycles = elementwiseCycles(elements);
  if (elements <= 0 || elementSize <= 0)
    return cost;
  if (source.shape.empty() ||
      source.shape.size() != source.strides.size()) {
    machine.recordTrap("RESHAPE reads an operand with no extents");
    return cost;
  }

  Odometer walk(source.shape);
  int64_t destination = 0;
  do {
    if (destination >= elements)
      break;
    if (!copyElement(machine, instruction.resultSpace,
                     instruction.resultAddress, destination, source.space,
                     source.address, offsetOf(walk.current(), source.strides),
                     elementSize, "RESHAPE element"))
      return cost;
    ++destination;
  } while (walk.next());
  return cost;
}

/// `TRANSPOSE`: result extent `i` is operand extent `axes[i]`.
///
/// The identity permutation is not a special case here and must not become one:
/// it falls out as the permutation that maps every axis to itself, and the P7
/// coverage list asks for it precisely because an implementation that special
/// cased it would have two paths where the machine has one.
KernelCost kernelTRANSPOSE(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  const Operand &source = instruction.operands.front();
  const int64_t elementSize = elementByteSize(instruction.resultElementType);
  const int64_t elements = elementsIn(instruction.resultShape);
  cost.cycles = elementwiseCycles(elements);
  if (elements <= 0 || elementSize <= 0)
    return cost;

  const size_t rank = instruction.resultShape.size();
  if (!requireRank(machine, source, rank, "TRANSPOSE", 0))
    return cost;
  if (instruction.axes.size() != rank) {
    machine.recordTrap("TRANSPOSE has " +
                       std::to_string(instruction.axes.size()) +
                       " permutation entries and a result of rank " +
                       std::to_string(rank));
    return cost;
  }
  for (int64_t axis : instruction.axes) {
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
      machine.recordTrap("TRANSPOSE names axis " + std::to_string(axis) +
                         " and the result has rank " + std::to_string(rank));
      return cost;
    }
  }

  // The permuted stride vector, built once: the stride the result's axis `i`
  // walks is the operand's stride for axis `axes[i]`. Doing it here rather than
  // inside the loop is what keeps this one dot product per element.
  llvm::SmallVector<int64_t> permuted(rank);
  for (size_t axis = 0; axis < rank; ++axis)
    permuted[axis] = source.strides[static_cast<size_t>(instruction.axes[axis])];

  Odometer walk(instruction.resultShape);
  int64_t destination = 0;
  do {
    if (!copyElement(machine, instruction.resultSpace,
                     instruction.resultAddress, destination, source.space,
                     source.address, offsetOf(walk.current(), permuted),
                     elementSize, "TRANSPOSE element"))
      return cost;
    ++destination;
  } while (walk.next());
  return cost;
}

/// `CONCAT`: the operands laid end to end along one axis.
///
/// Each operand is walked in its own index space and placed at an offset along
/// the axis, rather than the result being walked and the covering operand
/// looked up per element. The two are equivalent and this one is the shape that
/// stays one loop when there are seven operands.
KernelCost kernelCONCAT(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  const int64_t elementSize = elementByteSize(instruction.resultElementType);
  const int64_t elements = elementsIn(instruction.resultShape);
  cost.cycles = elementwiseCycles(elements);
  if (elements <= 0 || elementSize <= 0)
    return cost;

  const size_t rank = instruction.resultShape.size();
  if (instruction.axes.size() != 1) {
    machine.recordTrap("CONCAT carries " +
                       std::to_string(instruction.axes.size()) +
                       " axis entries and takes exactly one");
    return cost;
  }
  const int64_t axis = instruction.axes.front();
  if (axis < 0 || static_cast<size_t>(axis) >= rank) {
    machine.recordTrap("CONCAT names axis " + std::to_string(axis) +
                       " and the result has rank " + std::to_string(rank));
    return cost;
  }

  llvm::SmallVector<int64_t> resultStrides(rank, 1);
  for (size_t index = rank - 1; index-- > 0;)
    resultStrides[index] =
        resultStrides[index + 1] * instruction.resultShape[index + 1];

  int64_t along = 0;
  for (const auto &[index, operand] : llvm::enumerate(instruction.operands)) {
    if (!requireRank(machine, operand, rank, "CONCAT",
                     static_cast<int>(index)))
      return cost;

    Odometer walk(operand.shape);
    do {
      llvm::ArrayRef<int64_t> at = walk.current();
      int64_t destination = 0;
      for (size_t dimension = 0; dimension < rank; ++dimension) {
        const int64_t coordinate =
            dimension == static_cast<size_t>(axis)
                ? at[dimension] + along
                : at[dimension];
        destination += coordinate * resultStrides[dimension];
      }
      if (!copyElement(machine, instruction.resultSpace,
                       instruction.resultAddress, destination, operand.space,
                       operand.address, offsetOf(at, operand.strides),
                       elementSize, "CONCAT element"))
        return cost;
    } while (walk.next());

    along += operand.shape[static_cast<size_t>(axis)];
  }
  return cost;
}

//===----------------------------------------------------------------------===//
// The elementwise kernels.
//===----------------------------------------------------------------------===//

/// `ADD` and `MUL`, which differ by one operator.
///
/// Neither has a broadcast path and neither needs one. A rank 1 channel operand
/// arrives with the result's rank and a stride of zero on every axis but the
/// channel, so the dot product with the strides reads the same element for a
/// whole plane without a branch. That is the whole mechanism ADR 0005 asks for.
template <typename Op>
KernelCost binary(Machine &machine, const Instruction &instruction, Op op,
                  const char *name) {
  KernelCost cost;
  const int64_t elements = elementsIn(instruction.resultShape);
  cost.cycles = elementwiseCycles(elements);
  if (elements <= 0 || !requireF32(machine, instruction, name))
    return cost;

  const size_t rank = instruction.resultShape.size();
  const Operand &lhs = instruction.operands[0];
  const Operand &rhs = instruction.operands[1];
  if (!requireRank(machine, lhs, rank, name, 0) ||
      !requireRank(machine, rhs, rank, name, 1))
    return cost;

  Odometer walk(instruction.resultShape);
  int64_t destination = 0;
  do {
    llvm::ArrayRef<int64_t> at = walk.current();
    const float *left = machine.readF32(lhs.space, lhs.address,
                                        offsetOf(at, lhs.strides), name);
    const float *right = machine.readF32(rhs.space, rhs.address,
                                         offsetOf(at, rhs.strides), name);
    float *out = machine.writeF32(instruction.resultSpace,
                                  instruction.resultAddress, destination, name);
    if (!left || !right || !out)
      return cost;
    *out = activate(instruction, op(*left, *right));
    ++destination;
  } while (walk.next());
  return cost;
}

KernelCost kernelADD(Machine &machine, const Instruction &instruction) {
  return binary(
      machine, instruction, [](float a, float b) { return a + b; }, "ADD");
}

KernelCost kernelMUL(Machine &machine, const Instruction &instruction) {
  return binary(
      machine, instruction, [](float a, float b) { return a * b; }, "MUL");
}

/// `RELU`, which is `max(x, 0)` and nothing else.
///
/// The operand and the result may be the same buffer: an in place relu is what
/// the allocator produces when it reuses a dead interval, and the read happens
/// before the write for every element, so aliasing is not a case here.
KernelCost kernelRELU(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  const int64_t elements = elementsIn(instruction.resultShape);
  cost.cycles = elementwiseCycles(elements);
  if (elements <= 0 || !requireF32(machine, instruction, "RELU"))
    return cost;

  const Operand &source = instruction.operands.front();
  if (!requireRank(machine, source, instruction.resultShape.size(), "RELU", 0))
    return cost;

  Odometer walk(instruction.resultShape);
  int64_t destination = 0;
  do {
    const float *in = machine.readF32(source.space, source.address,
                                      offsetOf(walk.current(), source.strides),
                                      "RELU");
    float *out = machine.writeF32(instruction.resultSpace,
                                  instruction.resultAddress, destination,
                                  "RELU");
    if (!in || !out)
      return cost;
    *out = std::max(*in, 0.0f);
    ++destination;
  } while (walk.next());
  return cost;
}

//===----------------------------------------------------------------------===//
// The windowed kernels.
//===----------------------------------------------------------------------===//

/// The two pools, which differ only in how a window is reduced.
///
/// The output extents are the result's, taken as declared rather than
/// recomputed, because the encoder has already resolved `ceil_mode` and
/// recomputing here would be a second implementation of the windowed arithmetic
/// that could disagree with the first.
///
/// **The all padding window is the case this signature exists for.** With
/// enough padding a window can contain no input element at all. The average
/// then divides by zero unless somebody thought about it, and the maximum has
/// no elements to take a maximum of. `POOL_AVG` writes zero, which is what
/// dividing a sum of nothing by a count of nothing should mean; `POOL_MAX`
/// writes negative infinity, which is the identity of the maximum and is what
/// ONNX produces. Both are asserted by a test rather than left to whatever the
/// hardware happened to do.
template <bool IsMax>
KernelCost pool(Machine &machine, const Instruction &instruction,
                const char *name) {
  KernelCost cost;
  const int64_t elements = elementsIn(instruction.resultShape);
  if (instruction.resultShape.size() != 4 || instruction.kernel.size() != 2 ||
      instruction.strides.size() != 2 || instruction.dilations.size() != 2 ||
      instruction.pads.size() != 4) {
    machine.recordTrap(std::string(name) +
                       " needs a rank 4 result and its four window fields");
    return cost;
  }

  const int64_t kernelH = instruction.kernel[0];
  const int64_t kernelW = instruction.kernel[1];
  cost.cycles = elementwiseCycles(elements * kernelH * kernelW);
  if (elements <= 0 || !requireF32(machine, instruction, name))
    return cost;

  const Operand &source = instruction.operands.front();
  if (!requireRank(machine, source, 4, name, 0))
    return cost;

  const int64_t batch = instruction.resultShape[0];
  const int64_t channels = instruction.resultShape[1];
  const int64_t outputH = instruction.resultShape[2];
  const int64_t outputW = instruction.resultShape[3];
  const int64_t inputH = source.shape[2];
  const int64_t inputW = source.shape[3];
  const int64_t strideH = instruction.strides[0];
  const int64_t strideW = instruction.strides[1];
  const int64_t dilationH = instruction.dilations[0];
  const int64_t dilationW = instruction.dilations[1];
  const int64_t padTop = instruction.pads[0];
  const int64_t padLeft = instruction.pads[1];

  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t oh = 0; oh < outputH; ++oh) {
        for (int64_t ow = 0; ow < outputW; ++ow) {
          float accumulator =
              IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;
          int64_t contributing = 0;

          for (int64_t kh = 0; kh < kernelH; ++kh) {
            const int64_t ih = oh * strideH - padTop + kh * dilationH;
            if (ih < 0 || ih >= inputH)
              continue;
            for (int64_t kw = 0; kw < kernelW; ++kw) {
              const int64_t iw = ow * strideW - padLeft + kw * dilationW;
              if (iw < 0 || iw >= inputW)
                continue;
              const int64_t index[4] = {n, c, ih, iw};
              const float *value =
                  machine.readF32(source.space, source.address,
                                  offsetOf(index, source.strides), name);
              if (!value)
                return cost;
              if (IsMax)
                accumulator = std::max(accumulator, *value);
              else
                accumulator += *value;
              ++contributing;
            }
          }

          // count_include_pad = 0, which is the only behaviour this project
          // implements: the mean divides by the elements that actually
          // contributed rather than by the window area.
          float value = accumulator;
          if (!IsMax)
            value = contributing > 0 ? accumulator / static_cast<float>(
                                                         contributing)
                                     : 0.0f;

          const int64_t destination =
              ((n * channels + c) * outputH + oh) * outputW + ow;
          float *out = machine.writeF32(instruction.resultSpace,
                                        instruction.resultAddress, destination,
                                        name);
          if (!out)
            return cost;
          *out = value;
        }
      }
    }
  }
  return cost;
}

KernelCost kernelPOOL_MAX(Machine &machine, const Instruction &instruction) {
  return pool<true>(machine, instruction, "POOL_MAX");
}

KernelCost kernelPOOL_AVG(Machine &machine, const Instruction &instruction) {
  return pool<false>(machine, instruction, "POOL_AVG");
}

/// Reads a bias element for output channel `channel` at batch position `n` and
/// output position `(oh, ow)`.
///
/// A bias arrives either as a rank 1 buffer of length F, which is what the
/// encoder emits, or as a rank 4 stride 0 broadcast, which is what ADR 0005's
/// channel broadcast looks like when a pass has already given it the result's
/// rank. Both are addressed through the strides and neither is a special case
/// in the arithmetic; the branch is only about how many coordinates to dot.
const float *biasAt(Machine &machine, const Operand &bias, int64_t n,
                    int64_t channel, int64_t oh, int64_t ow,
                    const char *name) {
  if (bias.strides.size() == 1)
    return machine.readF32(bias.space, bias.address, channel * bias.strides[0],
                           name);
  const int64_t index[4] = {n, channel, oh, ow};
  return machine.readF32(bias.space, bias.address,
                         offsetOf(llvm::ArrayRef<int64_t>(index, 4),
                                  bias.strides),
                         name);
}

/// `CONV2D`: a two dimensional grouped, dilated, asymmetrically padded
/// convolution.
///
/// Every one of those words is a case the P7 coverage list asks for, and none
/// of them is a separate path: a dense convolution is `group == 1`, a depthwise
/// convolution is `group == C`, an undilated one has `dilations = [1, 1]`, and
/// symmetric padding is the case where the four pads happen to be equal. One
/// loop nest covers all of them, which is why there is one set of last bits to
/// keep stable rather than four.
KernelCost kernelCONV2D(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  if (instruction.resultShape.size() != 4 || instruction.strides.size() != 2 ||
      instruction.dilations.size() != 2 || instruction.pads.size() != 4) {
    machine.recordTrap("CONV2D needs a rank 4 result and its window fields");
    return cost;
  }

  const Operand &input = instruction.operands[0];
  const Operand &filter = instruction.operands[1];
  if (!requireRank(machine, input, 4, "CONV2D", 0) ||
      !requireRank(machine, filter, 4, "CONV2D", 1))
    return cost;

  const int64_t batch = instruction.resultShape[0];
  const int64_t outputChannels = instruction.resultShape[1];
  const int64_t outputH = instruction.resultShape[2];
  const int64_t outputW = instruction.resultShape[3];
  const int64_t inputChannels = input.shape[1];
  const int64_t inputH = input.shape[2];
  const int64_t inputW = input.shape[3];
  const int64_t kernelH = filter.shape[2];
  const int64_t kernelW = filter.shape[3];
  const int64_t group = instruction.group;

  // Section 7.3: an operation whose input and result disagree on N is refused
  // with a diagnostic naming the instruction and both extents. There is no
  // N == 1 shortcut anywhere below and no verifier that rejects N > 1.
  if (input.shape[0] != batch) {
    machine.recordTrap("CONV2D reads a batch of " +
                       std::to_string(input.shape[0]) +
                       " and writes a batch of " + std::to_string(batch));
    return cost;
  }
  if (group <= 0 || outputChannels % group != 0 || inputChannels % group != 0) {
    machine.recordTrap("CONV2D has group " + std::to_string(group) +
                       " and it divides neither " +
                       std::to_string(inputChannels) + " input channels nor " +
                       std::to_string(outputChannels) + " output channels");
    return cost;
  }
  if (filter.shape[0] != outputChannels ||
      filter.shape[1] != inputChannels / group) {
    machine.recordTrap("CONV2D has a filter of shape [" +
                       std::to_string(filter.shape[0]) + ", " +
                       std::to_string(filter.shape[1]) +
                       ", ...] and needs [" + std::to_string(outputChannels) +
                       ", " + std::to_string(inputChannels / group) + ", ...]");
    return cost;
  }

  const ComputeCharge charge =
      conv2dCharge(batch, outputChannels, inputChannels, outputH, outputW,
                   kernelH, kernelW, group, kPeakMacsPerCycleF32);
  cost.cycles = charge.cycles;
  cost.macs = charge.macs;
  cost.effectiveMacs = charge.effectiveMacs;
  cost.utilization = charge.utilization;
  cost.delta = charge.delta;

  if (!requireF32(machine, instruction, "CONV2D"))
    return cost;

  const int64_t channelsPerGroup = inputChannels / group;
  const int64_t filtersPerGroup = outputChannels / group;
  const int64_t strideH = instruction.strides[0];
  const int64_t strideW = instruction.strides[1];
  const int64_t dilationH = instruction.dilations[0];
  const int64_t dilationW = instruction.dilations[1];
  const int64_t padTop = instruction.pads[0];
  const int64_t padLeft = instruction.pads[1];
  const bool hasBias = instruction.operands.size() > 2;

  // Section 10.3: parallel over the batch and output channel dimensions only.
  // Each thread writes a disjoint output region, so there is no reduction race
  // and no atomics, and the reduction loops below stay strictly sequential and
  // in their original order because the accumulation order determines the last
  // bits and the golden files depend on it.
  //
  // **The team is capped at the number of output tiles, and that cap carries no
  // tuned constant.** The collapsed loop has exactly `batch * outputChannels`
  // iterations, so a team larger than that has threads with no iteration to
  // run. They are not free: every one of them still arrives at the region's
  // entry and its closing barrier, and on a host with many logical processors
  // that is most of what a small convolution's parallel region costs. Measured
  // at P12 on 28 logical processors, an uncapped team made five of the seven
  // suite models **slower than serial**, by as much as seven times on
  // `depthwise_separable`, whose two convolutions have eight and sixteen output
  // channels at batch 1 and were being handed twenty eight threads each.
  //
  // `if (teamSize > 1)` is the same argument at its limit: a convolution with
  // one output tile has no parallelism to find and forming a team to discover
  // that costs more than the tile.
  //
  // **Neither clause can move a bit.** They change how many threads run the
  // iterations and never which iterations exist, what any one of them computes,
  // or the order of the reductions inside it. That is the whole reason a cap is
  // an acceptable answer here and an im2col restructuring is not.
#ifdef _OPENMP
  const int64_t outputTiles = batch * outputChannels;
  const int teamSize = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(outputTiles, omp_get_max_threads())));
#pragma omp parallel for collapse(2) num_threads(teamSize) if (teamSize > 1)
#endif
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t f = 0; f < outputChannels; ++f) {
      const int64_t groupOf = f / filtersPerGroup;
      for (int64_t oh = 0; oh < outputH; ++oh) {
        for (int64_t ow = 0; ow < outputW; ++ow) {
          float accumulator = 0.0f;
          for (int64_t c = 0; c < channelsPerGroup; ++c) {
            const int64_t inputChannel = groupOf * channelsPerGroup + c;
            for (int64_t kh = 0; kh < kernelH; ++kh) {
              const int64_t ih = oh * strideH - padTop + kh * dilationH;
              if (ih < 0 || ih >= inputH)
                continue;
              for (int64_t kw = 0; kw < kernelW; ++kw) {
                const int64_t iw = ow * strideW - padLeft + kw * dilationW;
                if (iw < 0 || iw >= inputW)
                  continue;
                const int64_t at[4] = {n, inputChannel, ih, iw};
                const int64_t weightAt[4] = {f, c, kh, kw};
                const float *value =
                    machine.readF32(input.space, input.address,
                                    offsetOf(at, input.strides), "CONV2D");
                const float *weight =
                    machine.readF32(filter.space, filter.address,
                                    offsetOf(weightAt, filter.strides),
                                    "CONV2D");
                if (!value || !weight)
                  continue;
                accumulator += *value * *weight;
              }
            }
          }

          if (hasBias) {
            const float *bias = biasAt(machine, instruction.operands[2], n, f,
                                       oh, ow, "CONV2D");
            if (bias)
              accumulator += *bias;
          }

          const int64_t destination =
              ((n * outputChannels + f) * outputH + oh) * outputW + ow;
          float *out = machine.writeF32(instruction.resultSpace,
                                        instruction.resultAddress, destination,
                                        "CONV2D");
          if (out)
            *out = activate(instruction, accumulator);
        }
      }
    }
  }
  return cost;
}

/// `MATMUL`: `(M, K)` by `(K, N)` into `(M, N)`, with an optional bias of
/// length N.
///
/// M is the batch dimension of a fully connected layer and is never assumed to
/// be one.
KernelCost kernelMATMUL(Machine &machine, const Instruction &instruction) {
  KernelCost cost;
  if (instruction.resultShape.size() != 2) {
    machine.recordTrap("MATMUL writes a rank 2 result and this one has rank " +
                       std::to_string(instruction.resultShape.size()));
    return cost;
  }
  const Operand &lhs = instruction.operands[0];
  const Operand &rhs = instruction.operands[1];
  if (!requireRank(machine, lhs, 2, "MATMUL", 0) ||
      !requireRank(machine, rhs, 2, "MATMUL", 1))
    return cost;

  const int64_t rows = instruction.resultShape[0];
  const int64_t columns = instruction.resultShape[1];
  const int64_t reduction = lhs.shape[1];
  if (lhs.shape[0] != rows || rhs.shape[0] != reduction ||
      rhs.shape[1] != columns) {
    machine.recordTrap(
        "MATMUL multiplies [" + std::to_string(lhs.shape[0]) + ", " +
        std::to_string(lhs.shape[1]) + "] by [" + std::to_string(rhs.shape[0]) +
        ", " + std::to_string(rhs.shape[1]) + "] and writes [" +
        std::to_string(rows) + ", " + std::to_string(columns) + "]");
    return cost;
  }

  const ComputeCharge charge =
      gemmCharge(rows, reduction, columns, kPeakMacsPerCycleF32);
  cost.cycles = charge.cycles;
  cost.macs = charge.macs;
  cost.effectiveMacs = charge.effectiveMacs;
  cost.utilization = charge.utilization;
  cost.delta = charge.delta;

  if (!requireF32(machine, instruction, "MATMUL"))
    return cost;

  const bool hasBias = instruction.operands.size() > 2;
  for (int64_t m = 0; m < rows; ++m) {
    for (int64_t n = 0; n < columns; ++n) {
      float accumulator = 0.0f;
      for (int64_t k = 0; k < reduction; ++k) {
        const int64_t left[2] = {m, k};
        const int64_t right[2] = {k, n};
        const float *a = machine.readF32(lhs.space, lhs.address,
                                         offsetOf(left, lhs.strides), "MATMUL");
        const float *b = machine.readF32(rhs.space, rhs.address,
                                         offsetOf(right, rhs.strides),
                                         "MATMUL");
        if (!a || !b)
          return cost;
        accumulator += *a * *b;
      }
      if (hasBias) {
        const Operand &bias = instruction.operands[2];
        const int64_t biasAt2[2] = {m, n};
        const float *value =
            bias.strides.size() == 1
                ? machine.readF32(bias.space, bias.address,
                                  n * bias.strides[0], "MATMUL")
                : machine.readF32(bias.space, bias.address,
                                  offsetOf(biasAt2, bias.strides), "MATMUL");
        if (!value)
          return cost;
        accumulator += *value;
      }
      float *out =
          machine.writeF32(instruction.resultSpace, instruction.resultAddress,
                           m * columns + n, "MATMUL");
      if (!out)
        return cost;
      *out = activate(instruction, accumulator);
    }
  }
  return cost;
}

//===----------------------------------------------------------------------===//
// The quantization opcodes.
//
// They have kernels here because the ISA description marks them as computation
// and the dispatch table below requires one for every opcode that is. What the
// kernels do is refuse by name, because Section 10.1 is explicit that the
// integer semantics are Phase P14's and that P7's gate asks for the f32 list
// only. A refusal that names the phase is not a runtime surprise: it is the
// same answer the manual gives, given by the machine.
//===----------------------------------------------------------------------===//

KernelCost unimplementedInteger(Machine &machine, const Instruction &instruction,
                                const char *name) {
  KernelCost cost;
  cost.cycles = elementwiseCycles(elementsIn(instruction.resultShape));
  machine.recordTrap(
      std::string(name) +
      " has no kernel in this build. Section 10.1 puts the integer semantics at "
      "Phase P14, with the rest of Section 14's quantization path; at Phase P7 "
      "this opcode has structural coverage only and encodes, decodes, "
      "validates and round trips without computing anything");
  return cost;
}

KernelCost kernelQUANT(Machine &machine, const Instruction &instruction) {
  return unimplementedInteger(machine, instruction, "QUANT");
}

KernelCost kernelDEQUANT(Machine &machine, const Instruction &instruction) {
  return unimplementedInteger(machine, instruction, "DEQUANT");
}

KernelCost kernelDMA_LOAD(Machine &machine, const Instruction &instruction) {
  return transfer(machine, instruction);
}

KernelCost kernelDMA_STORE(Machine &machine, const Instruction &instruction) {
  return transfer(machine, instruction);
}

//===----------------------------------------------------------------------===//
// The dispatch table.
//
// This is the expansion Section 9.4 promised and Phase P6 could only generate.
// The `.def` names every opcode with a flag saying whether it is computation,
// and the two macros below turn a computation row into the address of a kernel
// named after it. **An opcode appended to the description with no kernel
// written does not compile**, because there is no such identifier to take the
// address of, and the failure names the opcode.
//===----------------------------------------------------------------------===//

#define NPU_KERNEL_ENTRY_0(NAME) nullptr,
#define NPU_KERNEL_ENTRY_1(NAME) &kernel##NAME,
#define NPU_KERNEL_PICK(NEEDS, NAME) NPU_KERNEL_ENTRY_##NEEDS(NAME)
#define NPU_ISA_OPCODE(NAME, VALUE, NEEDS_KERNEL)                              \
  NPU_KERNEL_PICK(NEEDS_KERNEL, NAME)

constexpr Kernel kKernels[] = {
#include "NPU/Encoding/NPUISADispatch.def"
};

#undef NPU_KERNEL_PICK
#undef NPU_KERNEL_ENTRY_1
#undef NPU_KERNEL_ENTRY_0

static_assert(sizeof(kKernels) / sizeof(kKernels[0]) == kNumOpcodes,
              "the kernel table and the ISA description disagree about how "
              "many opcodes there are, which means the generated dispatch "
              "skeleton was not the one this file expanded");

} // namespace

namespace nbin {
namespace detail {

Kernel kernelFor(Opcode opcode) {
  const uint32_t value = static_cast<uint32_t>(opcode);
  if (value >= kNumOpcodes)
    return nullptr;
  return kKernels[value];
}

Port portFor(Opcode opcode) {
  switch (opcode) {
  case Opcode::DMA_LOAD:
  case Opcode::DMA_STORE:
    return Port::Dma;

  // Everything else issues on the compute port, including the two control
  // opcodes, which cost their issue overhead and nothing more. Each is named
  // rather than swept into a default, so that an opcode appended to the
  // description has to be given a port before this file compiles.
  case Opcode::NOP:
  case Opcode::HALT:
  case Opcode::MATMUL:
  case Opcode::CONV2D:
  case Opcode::ADD:
  case Opcode::MUL:
  case Opcode::RELU:
  case Opcode::POOL_MAX:
  case Opcode::POOL_AVG:
  case Opcode::RESHAPE:
  case Opcode::TRANSPOSE:
  case Opcode::CONCAT:
  case Opcode::QUANT:
  case Opcode::DEQUANT:
    return Port::Compute;
  }
  return Port::Compute;
}

} // namespace detail

//===----------------------------------------------------------------------===//
// What this translation unit was compiled with.
//===----------------------------------------------------------------------===//
//
// Both answers are read here and nowhere else, which is the point. Every other
// file in this project that asks about OpenMP is asking its own preprocessor,
// and between P7 and P12 its own preprocessor and this one disagreed. D-0047.

bool kernelsUseOpenMP() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

int kernelThreadCount() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

} // namespace nbin
