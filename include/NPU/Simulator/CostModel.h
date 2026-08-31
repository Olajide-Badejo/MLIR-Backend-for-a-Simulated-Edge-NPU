//===- CostModel.h - the one home for the machine's constants -*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 5.5's cost model, with exactly one home.
//
// **Every number below is an assumption, not a measurement**, and the report
// says so wherever it quotes one. Each carries the reasoning that produced it
// and a stated uncertainty, because a constant with no derivation beside it is
// a number a reader has to take on trust and a number the author cannot defend
// six months later.
//
// **There is one copy of these numbers in C++ and one in Python**, and a test
// asserts the two agree. `python/npu_frontend/cost_model.py` is the mirror and
// `test/Python/test_cost_model_mirror.py` parses this header and compares.
// Two hand copied sets of numbers is how a report starts lying, and the mirror
// exists because the plotting and analysis code of the later phases is Python
// and would otherwise grow its own copy.
//
// This is also what the tiling pass of Phase P13 scores against, so the project
// has exactly one cost function rather than a modelled one and a heuristic one
// that can disagree.
//
// **The dataflow is weight stationary and it is pinned.** A 16 by 16 array is
// not a cost model until the dataflow is named. `docs/adr/0007-dataflow.md`
// carries the decision, the reason row stationary is rejected, and what would
// have to change for the answer to move.
//
//===----------------------------------------------------------------------===//

#ifndef NPU_SIMULATOR_COSTMODEL_H
#define NPU_SIMULATOR_COSTMODEL_H

#include <algorithm>
#include <cstdint>

namespace nbin {

//===----------------------------------------------------------------------===//
// The constants.
//
// The block between the two markers below is parsed by
// test/Python/test_cost_model_mirror.py, which asserts the Python mirror holds
// the same names and the same values. Every line inside it has the shape
//
//     inline constexpr <type> <name> = <literal>;
//
// on one line, because a parser that had to understand C++ to read a table of
// constants would be a worse mechanism than the drift it prevents. Keep new
// constants inside the markers and in that shape, and add them to the mirror in
// the same commit.
//===----------------------------------------------------------------------===//

// NPU_COST_CONSTANTS_BEGIN

/// The systolic array is `kArrayDim` by `kArrayDim` processing elements.
///
/// This is the machine Section 5.1 describes and the shape Phase P18's Gemmini
/// target is configured to. It appears in the cost model twice: as the divisor
/// of the spatial utilization term, and as the depth of the weight preload
/// pipeline that the temporal term charges for.
inline constexpr int64_t kArrayDim = 16;

/// Peak multiply accumulate throughput per cycle at f32.
///
/// One multiply accumulate per processing element per cycle, so this is
/// `kArrayDim * kArrayDim`. Stated as its own constant rather than computed,
/// because the int8 figure below is deliberately not `kArrayDim * kArrayDim * 4`
/// by definition and writing one as an expression over the other would hide
/// that they are two separate assumptions.
inline constexpr int64_t kPeakMacsPerCycleF32 = 256;

/// Peak multiply accumulate throughput per cycle at int8.
///
/// Four int8 multiplies packed per f32 lane. The packing factor is defensible
/// rather than embarrassing: at 45 nm an 8 bit integer multiply costs about
/// 0.2 pJ against about 3.7 pJ for a 32 bit float multiply [R40], so 4x is
/// conservative relative to what the multiplier cost ratio permits. It stays an
/// assumption, its contribution to any reported speedup is reported separately
/// from the contribution of reduced DRAM traffic, and it is cross checked
/// against Accelergy at Phase P11.
///
/// **Nothing charges against it at Phase P7**, because no integer kernel exists
/// until Phase P14. It is here so that the constant has one home from the
/// moment the cost model does, rather than arriving in the same commit as the
/// kernels that would make a mistake in it invisible.
inline constexpr int64_t kPeakMacsPerCycleI8 = 1024;

/// DRAM bandwidth, in bytes per cycle, for a contiguous burst.
///
/// A 128 bit interface at one transfer per cycle. Uncertainty: this is the
/// single most load bearing DMA assumption and the one most likely to be wrong
/// by a factor of two for a real part. Every DRAM bound conclusion in the
/// report is restated as a ratio against this number rather than as an absolute
/// time, so that a reader who disagrees with it can rescale.
inline constexpr double kDramBandwidthBytesPerCycle = 16.0;

/// The fixed cost of issuing one DMA descriptor, in cycles.
///
/// Address generation plus the round trip to the memory controller. It is why
/// many small transfers cost more than one large one, which is the effect the
/// double buffering pass of Phase P12 and the tiling pass of Phase P13 are
/// measured against. Uncertainty: order of magnitude.
inline constexpr double kDmaDescriptorCycles = 64.0;

/// The extra cost, in cycles per element moved, when a transfer's innermost
/// stride is not one.
///
/// **This term is not decoration.** Without it a strided NCHW gather and a
/// contiguous NHWC burst cost exactly the same, and the layout assignment pass
/// of Phase P13 becomes an optimization whose benefit the cost model is
/// structurally unable to express. A phase whose payoff the model cannot
/// represent must not be in the roadmap.
///
/// The value says that a strided transfer moves at a third of burst speed at
/// this bandwidth, which is the shape of a DRAM controller that cannot coalesce
/// a request into a full burst. The innermost stride is read out of the
/// instruction's operand strides, which is where Section 5.5 put the layout
/// decision below tensor level.
inline constexpr double kDmaStridedElementCycles = 0.5;

/// How many elements the elementwise unit consumes per cycle.
///
/// One vector lane per array column, which is why it matches `kArrayDim`. It is
/// its own constant because the elementwise unit is not the array and a machine
/// could size the two differently.
inline constexpr int64_t kElementwiseLaneWidth = 16;

/// The fixed issue overhead charged to every instruction, in cycles.
///
/// Decode, address generation and the handshake with the port. It is what makes
/// a program of a thousand tiny instructions cost something rather than
/// nothing, which is the effect the instruction count of Section 10.2 exists to
/// let a reader see.
inline constexpr double kIssueOverheadCycles = 4.0;

/// The depth of the weight preload pipeline, in cycles.
///
/// Under a weight stationary dataflow a weight tile is pushed into the array
/// before any activation row can be streamed against it, and the push takes as
/// many cycles as the array is deep. This is the whole content of the temporal
/// `delta` term: a tile streamed against many activation rows amortises the
/// fill, and a tile streamed against few does not.
inline constexpr double kWeightPreloadCycles = 16.0;

// NPU_COST_CONSTANTS_END

//===----------------------------------------------------------------------===//
// The charges.
//===----------------------------------------------------------------------===//

/// What one compute instruction costs, with everything a reader needs to
/// reconstruct the charge without reverse engineering it.
///
/// **`macs` is raw and stays raw.** Utilization describes how long the array
/// was busy, not how many multiplies happened, and Section 5.5 is explicit that
/// feeding a utilization scaled count into Accelergy would overstate the energy
/// of exactly the layers the evaluation cares most about. So `macs` is the
/// count, and `effectiveMacs`, `utilization` and `delta` sit beside it.
///
/// `effectiveMacs` is `cycles * peak`: the MAC count a fully occupied array
/// would have retired in the time this instruction actually took. It is always
/// at least `macs`, and the two are equal exactly when the tile fills the array
/// and the preload is fully amortised. Nothing in the energy path ever sees it.
struct ComputeCharge {
  /// The cycles the compute port is occupied, excluding the issue overhead.
  double cycles = 0.0;
  /// The multiply accumulate operations performed. Raw, always.
  int64_t macs = 0;
  /// `cycles * peak`. See above.
  double effectiveMacs = 0.0;
  /// The MAC weighted mean spatial occupancy of the array, in (0, 1].
  double utilization = 1.0;
  /// The temporal factor for the weight preload pipeline fill, in (0, 1].
  double delta = 1.0;
};

/// The cost of a DMA transfer.
///
/// `bytes / bandwidth`, plus a fixed per descriptor cost, plus the non unit
/// innermost stride penalty. The three terms are Section 5.5's, in its order.
///
/// `innermostStride` is the last entry of the operand's stride vector, in
/// elements, taken from the `Instruction` record rather than inferred from a
/// layout attribute that does not survive bufferization.
constexpr double dmaCycles(int64_t bytes, int64_t elements,
                           int64_t innermostStride) {
  if (bytes <= 0 || elements <= 0)
    return kDmaDescriptorCycles;
  double cycles = static_cast<double>(bytes) / kDramBandwidthBytesPerCycle;
  cycles += kDmaDescriptorCycles;
  if (innermostStride != 1)
    cycles += static_cast<double>(elements) * kDmaStridedElementCycles;
  return cycles;
}

/// The cost of an elementwise pass over `elements` elements.
constexpr double elementwiseCycles(int64_t elements) {
  if (elements <= 0)
    return 0.0;
  return static_cast<double>(elements) /
         static_cast<double>(kElementwiseLaneWidth);
}

/// The charge for a matrix multiplication of `m` streamed rows against a
/// `k` by `n` weight matrix, under the pinned weight stationary dataflow.
///
/// **What the three extents mean on this array.** The weights are stationary,
/// so the array holds a `kArrayDim` by `kArrayDim` tile of the `k` by `n`
/// weight matrix: reduction down the rows, output channels across the columns.
/// The `m` activation rows stream through it. A weight matrix larger than the
/// array is folded into `ceil(k / kArrayDim) * ceil(n / kArrayDim)` tiles and
/// each tile is charged separately, which is what makes the last, narrow tile
/// of a 256 by 120 fully connected layer cost what it actually costs.
///
/// **Per tile:**
///
///     utilization = (rows / kArrayDim) * (columns / kArrayDim)
///     delta       = m / (m + kWeightPreloadCycles)
///     cycles      = tileMacs / (utilization * delta * peak)
///
/// Written the other way round, as a factor at or above one multiplying MACs,
/// the formula would claim a small tile does *more* work, which is the opposite
/// of the effect being modelled and would flatter every narrow layer in the
/// suite.
///
/// **This is arithmetic over the shapes, not a claim about the kernel.** The
/// convolution kernel is direct and stays direct; Section 10.3 rejects an
/// im2col plus GEMM restructuring of it outright, because that would move every
/// golden file for a host speedup. Mapping a convolution onto this function is
/// how the *cost* of the array is computed, and the two are separate on purpose.
ComputeCharge gemmCharge(int64_t m, int64_t k, int64_t n, int64_t peak);

/// The convolution's charge, expressed through `gemmCharge`.
///
/// Per group, the weight matrix is `(inputChannelsPerGroup * kernelHeight *
/// kernelWidth)` by `outputChannelsPerGroup`, and the streamed row count is
/// `batch * outputHeight * outputWidth`. Grouping is what makes a depthwise
/// layer expensive in this model: with `group == C` each group presents a
/// single column to a sixteen column array, so the spatial term is 1/16 of what
/// a dense convolution of the same MAC count would get, which is the first
/// order correction Section 5.5 says the depthwise separable model in the suite
/// exists to expose.
///
/// **Padded positions are counted.** A weight stationary array is fed the
/// padding as zeros and the multiplies happen, so counting only the interior
/// would report a MAC count the hardware did not perform. Stated here because
/// it is the kind of convention two tools disagree about silently.
ComputeCharge conv2dCharge(int64_t batch, int64_t outputChannels,
                           int64_t inputChannels, int64_t outputHeight,
                           int64_t outputWidth, int64_t kernelHeight,
                           int64_t kernelWidth, int64_t group, int64_t peak);

/// `overlap_fraction` of Section 5.5.
///
///     (dmaTotal + computeTotal - total) / min(dmaTotal, computeTotal)
///
/// Read it as the fraction of the **shorter** timeline that was hidden
/// underneath the longer one. **0 means no overlap at all**, the two timelines
/// ran end to end, and **1 means perfect overlap**, the shorter timeline is
/// entirely hidden.
///
/// The naive form, one minus total over the sum, tops out near 0.5 even when
/// overlap is perfect, because the best achievable total is the longer timeline
/// rather than zero, and a metric whose maximum is 0.5 will be read by somebody
/// as half of something.
///
/// A program with nothing on one of the two ports has a zero denominator. It
/// reports 0, because there was no shorter timeline to hide and claiming
/// perfect overlap for a program that never overlapped anything would be the
/// same lie in the other direction.
constexpr double overlapFraction(double dmaTotal, double computeTotal,
                                 double total) {
  double shorter = std::min(dmaTotal, computeTotal);
  if (shorter <= 0.0)
    return 0.0;
  return (dmaTotal + computeTotal - total) / shorter;
}

} // namespace nbin

#endif // NPU_SIMULATOR_COSTMODEL_H
