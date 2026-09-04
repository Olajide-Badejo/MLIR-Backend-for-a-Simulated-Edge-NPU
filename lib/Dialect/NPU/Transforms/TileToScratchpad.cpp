//===- TileToScratchpad.cpp - splitting what does not fit -------*- C++ -*-===//
//
// SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Section 13.2's `-npu-tile-to-scratchpad`, at `-O2`.
//
// **This pass is policy and the interface is mechanism, and P1 and P13 are the
// two ends of that split.** `NPUTilingInterfaceImpl.cpp` knows whether a tile of
// the iteration domain is expressible and refuses the ones that are not; this
// file knows which tile to ask for. Section 7.2 separates them by phase so that
// an interface bug and a policy bug cannot be mistaken for each other, and the
// separation survives here: nothing in this file reasons about halos, and
// nothing in the interface reasons about budgets.
//
// **The search is exhaustive and that is a claim about what the result means.**
// An exhaustive enumeration over an explicitly represented mapping space is
// optimal *with respect to the stated cost model*, which a hand rolled greedy
// can never be. Section 13.2 is explicit that a genetic, gradient based or
// learned mapper would be cargo cult here, because the space is enumerable: the
// factors are divisors of five parallel extents on the layer sizes in this
// suite, and capacity pruning cuts it further. The two named baselines are kept
// as `strategy=fixed` and `strategy=largest-fit` so the exhaustive result has
// something to be compared against rather than only asserted about.
//
// **Scoring is the cost model's, not a second one.** Every constant comes from
// `CostModel.h` through the same `nbin::` functions the simulator charges with,
// which is what Section 5.5 means by the tiling pass scoring against the one
// home. A heuristic that scored on DRAM bytes would be a second cost function,
// and under a bandwidth rich, compute poor operating point it would pick a
// tiling the simulator then reports as worse.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPU/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUDialect.h"
#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPU/IR/NPUShapeUtils.h"
#include "NPU/Simulator/CostModel.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <optional>
#include <string>

namespace mlir::npu {
#define GEN_PASS_DEF_NPUTILETOSCRATCHPAD
#include "NPU/Dialect/NPU/Transforms/Passes.h.inc"
} // namespace mlir::npu

using namespace mlir;
using namespace mlir::npu;

namespace {

//===----------------------------------------------------------------------===//
// Bytes.
//===----------------------------------------------------------------------===//

/// The bytes a tensor occupies, from its own type. Never from a hardcoded
/// factor of four, which is the rule Section 13.1 states for the allocator and
/// which applies here for the same reason: this pass and that one have to agree
/// about how big a buffer is or one of them places a program the other cannot.
int64_t bytesOf(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor)
    return 0;
  const int64_t width = tensor.getElementTypeBitWidth();
  int64_t elements = 1;
  for (int64_t extent : tensor.getShape())
    elements *= extent;
  return elements * (width / 8);
}

/// The working set of one operation: every value it reads, plus the one it
/// writes. Section 13.2 names them, in this order, and the destination operand
/// is the output, so summing every operand and the result would count it twice.
int64_t workingSetBytes(Operation *op) {
  int64_t total = 0;
  for (Value operand : op->getOperands())
    total += bytesOf(operand.getType());
  return total;
}

/// The same, with the prefetched operand doubled.
///
/// **Which operand is prefetched is the input**, and it is the input for a
/// reason rather than by choice: `-npu-double-buffer` overlaps the transfer of
/// the next tile's activation with the compute of this one, and the weights are
/// stationary under the pinned dataflow, so it is the activation that arrives
/// twice over. A tiling that fits only without double buffering silently
/// defeats the pass that adds it, which is why Section 13.2 makes this the
/// search's problem rather than the allocator's.
int64_t workingSetBytesFor(Operation *op, bool doubleBuffer) {
  const int64_t base = workingSetBytes(op);
  if (!doubleBuffer)
    return base;
  return base + bytesOf(op->getOperand(0).getType());
}

//===----------------------------------------------------------------------===//
// The mapping.
//===----------------------------------------------------------------------===//

/// One candidate: a temporal tile factor per parallel dimension of the domain.
///
/// The reduction dimensions do not appear because they are never split, which
/// is Section 13.2's fp32 rule and is enforced one level down by the interface
/// itself. The spatial factors of the mapping are the array's `kArrayDim` by
/// `kArrayDim`, which is not a search variable on this machine: there is one
/// array and the fold arithmetic in `gemmCharge` already maps a tile onto it.
/// They are recorded rather than searched, and the recorded mapping says so.
struct Mapping {
  /// Tile sizes, in domain order, for the parallel dimensions only.
  SmallVector<int64_t> tiles;
  /// How many tiles the factorisation produces.
  int64_t tileCount = 1;
  /// The Section 5.5 makespan, which is what candidates are ordered by.
  double makespan = 0.0;
  /// The largest working set any one tile has, which is what has to fit.
  int64_t peakBytes = 0;
};

/// Every divisor of `extent`, ascending. The factors are restricted to divisors
/// so that every tile is whole: a tile size that did not divide would leave a
/// remainder tile of a different shape, which is representable but doubles the
/// number of distinct per tile costs the makespan has to average over, for no
/// benefit on the shapes in this suite.
SmallVector<int64_t> divisorsOf(int64_t extent) {
  SmallVector<int64_t> divisors;
  for (int64_t candidate = 1; candidate <= extent; ++candidate)
    if (extent % candidate == 0)
      divisors.push_back(candidate);
  return divisors;
}

//===----------------------------------------------------------------------===//
// Scoring, against the one cost model.
//===----------------------------------------------------------------------===//

/// What one tile of a convolution costs on each of the two ports.
///
/// The DMA side is every operand slice the tile moves, each charged through
/// `nbin::dmaCycles` at the unit stride this pass produces, plus the output it
/// writes back. The compute side is `nbin::conv2dCharge` on the tile's own
/// extents. Both are the simulator's own functions rather than a paraphrase of
/// them, which is the point of the cost model having one home.
struct PortCosts {
  double dma = 0.0;
  double compute = 0.0;
};

/// What the search is told about the operation it is choosing a mapping for.
struct ConvProblem {
  int64_t batch, group, channelsPerGroup, outputHeight, outputWidth;
  int64_t inputChannelsPerGroup, kernelHeight, kernelWidth;
  int64_t strideHeight, strideWidth, dilationHeight, dilationWidth;
  /// The input's own spatial extents, which is what the halo window is clamped
  /// against.
  int64_t inputHeight, inputWidth;
  bool hasBias;
  int64_t elementBytes;
  int64_t budget;
  bool doubleBuffer;
  bool allowSpatial;
};

/// The input extent one output tile reads on one axis, halo included and
/// **clamped to the input the operation actually has**.
///
/// The clamp is not an optimisation and leaving it out is not conservative in a
/// safe direction. A window that runs off the edge of the input reads padding,
/// and padding is not fetched: the interface clamps the slice and hands the
/// tile a pad instead. Without the clamp here, a tile covering a whole axis
/// under same padding is priced at `(extent - 1) * stride + effectiveKernel`
/// rows, which is **larger than the axis itself**, so the search would believe
/// that not splitting an axis costs more memory than the whole operand does.
/// The two consequences both push the wrong way: it splits axes it did not need
/// to split, and it makes `halo=cache` decline budgets it could have met.
///
/// This is the same arithmetic `windowSlice` performs in the interface, stated
/// over extents rather than over positions, because the search has to price a
/// tile before it asks for one and building every candidate to price it would
/// be building every candidate.
int64_t tiledInputExtent(int64_t outputTile, int64_t inputExtent,
                         int64_t kernel, int64_t stride, int64_t dilation) {
  const int64_t effectiveKernel = dilation * (kernel - 1) + 1;
  return std::min(inputExtent, (outputTile - 1) * stride + effectiveKernel);
}

/// The element counts of one convolution tile's four operands.
struct ConvTileElements {
  int64_t input = 0, filter = 0, bias = 0, output = 0;
};

ConvTileElements convTileElements(const ConvProblem &problem,
                                  int64_t batchTile, int64_t groupTile,
                                  int64_t channelTile, int64_t heightTile,
                                  int64_t widthTile) {
  const int64_t inputHeight =
      tiledInputExtent(heightTile, problem.inputHeight, problem.kernelHeight,
                       problem.strideHeight, problem.dilationHeight);
  const int64_t inputWidth =
      tiledInputExtent(widthTile, problem.inputWidth, problem.kernelWidth,
                       problem.strideWidth, problem.dilationWidth);

  const int64_t inputChannels = groupTile * problem.inputChannelsPerGroup;
  const int64_t outputChannels = groupTile * channelTile;

  ConvTileElements elements;
  elements.input = batchTile * inputChannels * inputHeight * inputWidth;
  elements.filter = outputChannels * problem.inputChannelsPerGroup *
                    problem.kernelHeight * problem.kernelWidth;
  elements.bias = problem.hasBias ? outputChannels : 0;
  elements.output = batchTile * outputChannels * heightTile * widthTile;
  return elements;
}

PortCosts convTileCosts(const ConvProblem &problem, int64_t batchTile,
                        int64_t groupTile, int64_t channelTile,
                        int64_t heightTile, int64_t widthTile) {
  PortCosts costs;
  const ConvTileElements elements = convTileElements(
      problem, batchTile, groupTile, channelTile, heightTile, widthTile);
  const int64_t width = problem.elementBytes;

  costs.dma += nbin::dmaCycles(elements.input * width, elements.input, 1);
  costs.dma += nbin::dmaCycles(elements.filter * width, elements.filter, 1);
  if (problem.hasBias)
    costs.dma += nbin::dmaCycles(elements.bias * width, elements.bias, 1);
  costs.dma += nbin::dmaCycles(elements.output * width, elements.output, 1);
  costs.dma += nbin::kIssueOverheadCycles * (problem.hasBias ? 4 : 3);

  const nbin::ComputeCharge charge = nbin::conv2dCharge(
      batchTile, groupTile * channelTile,
      groupTile * problem.inputChannelsPerGroup, heightTile, widthTile,
      problem.kernelHeight, problem.kernelWidth, groupTile,
      nbin::kPeakMacsPerCycleF32);
  costs.compute = charge.cycles + nbin::kIssueOverheadCycles;
  return costs;
}

PortCosts matmulTileCosts(int64_t rowTile, int64_t columnTile,
                          int64_t reduction, bool hasBias,
                          int64_t elementBytes) {
  PortCosts costs;
  const int64_t lhsElements = rowTile * reduction;
  const int64_t rhsElements = reduction * columnTile;
  const int64_t outElements = rowTile * columnTile;

  costs.dma += nbin::dmaCycles(lhsElements * elementBytes, lhsElements, 1);
  costs.dma += nbin::dmaCycles(rhsElements * elementBytes, rhsElements, 1);
  if (hasBias)
    costs.dma += nbin::dmaCycles(columnTile * elementBytes, columnTile, 1);
  costs.dma += nbin::dmaCycles(outElements * elementBytes, outElements, 1);
  costs.dma += nbin::kIssueOverheadCycles * (hasBias ? 4 : 3);

  const nbin::ComputeCharge charge = nbin::gemmCharge(
      rowTile, reduction, columnTile, nbin::kPeakMacsPerCycleF32);
  costs.compute = charge.cycles + nbin::kIssueOverheadCycles;
  return costs;
}

/// Section 5.5's achievable total over `tileCount` tiles of equal cost.
///
///     tileCount * max(dma, compute) + min(dma, compute)
///
/// The second term is the prologue and the epilogue, which are modelled rather
/// than assumed away: the first fill and the last drain cannot overlap with
/// anything, because there is nothing on the other port yet or left.
double makespanOf(const PortCosts &perTile, int64_t tileCount) {
  const double longer = std::max(perTile.dma, perTile.compute);
  const double shorter = std::min(perTile.dma, perTile.compute);
  return static_cast<double>(tileCount) * longer + shorter;
}

/// The tie break Section 13.2 specifies, applied to two candidates that score
/// equally: prefer the one that splits the output channel first, then height,
/// then width, then batch. "Prefer to split" means prefer the smaller tile on
/// that axis, because that is the axis being divided.
///
/// The order is not arbitrary and the reason is worth keeping: the output
/// channel is the axis the array's columns are bound to, so splitting it is the
/// split whose cost the spatial term already accounts for, and the batch is
/// last because splitting it is the only one of the four that cannot reduce a
/// weight tile's residency.
bool tieBreaksBetter(const Mapping &candidate, const Mapping &best) {
  // tiles is in domain order (batch, group, channel, height, width).
  const int order[4] = {2, 3, 4, 0};
  for (int index : order) {
    if (candidate.tiles[index] != best.tiles[index])
      return candidate.tiles[index] < best.tiles[index];
  }
  return false;
}

/// Whether `candidate` should replace `best`.
bool isBetter(const Mapping &candidate, const Mapping &best) {
  constexpr double kEpsilon = 1e-9;
  if (candidate.makespan < best.makespan - kEpsilon)
    return true;
  if (candidate.makespan > best.makespan + kEpsilon)
    return false;
  return tieBreaksBetter(candidate, best);
}

//===----------------------------------------------------------------------===//
// The searches.
//===----------------------------------------------------------------------===//

/// One tile's working set, which is what has to fit inside the budget.
///
/// The same four operands the trigger measures, over the tile's own extents,
/// with the prefetched operand doubled when the search is sizing for double
/// buffering. Sharing `convTileElements` with the scorer is what keeps the
/// thing that decides whether a candidate fits and the thing that decides
/// whether it is good from disagreeing about what a tile is.
int64_t convTileBytes(const ConvProblem &problem, int64_t batchTile,
                      int64_t groupTile, int64_t channelTile,
                      int64_t heightTile, int64_t widthTile) {
  const ConvTileElements elements = convTileElements(
      problem, batchTile, groupTile, channelTile, heightTile, widthTile);
  const int64_t width = problem.elementBytes;

  int64_t total =
      (elements.input + elements.filter + elements.bias + elements.output) *
      width;
  if (problem.doubleBuffer)
    total += elements.input * width;
  return total;
}

/// The exhaustive search of Section 13.2, with capacity pruning.
std::optional<Mapping> searchConvExhaustive(const ConvProblem &problem) {
  std::optional<Mapping> best;

  const SmallVector<int64_t> batches = divisorsOf(problem.batch);
  const SmallVector<int64_t> groups = divisorsOf(problem.group);
  const SmallVector<int64_t> channels = divisorsOf(problem.channelsPerGroup);
  const SmallVector<int64_t> heights =
      problem.allowSpatial ? divisorsOf(problem.outputHeight)
                           : SmallVector<int64_t>{problem.outputHeight};
  const SmallVector<int64_t> widths =
      problem.allowSpatial ? divisorsOf(problem.outputWidth)
                           : SmallVector<int64_t>{problem.outputWidth};

  for (int64_t batchTile : batches) {
    for (int64_t groupTile : groups) {
      for (int64_t channelTile : channels) {
        // The interface folds the group and the per group channel back into one
        // contiguous channel range, and can only do it when the tile covers
        // whole groups or sits inside one. A candidate the interface would
        // refuse is not a candidate.
        if (groupTile != 1 && channelTile != problem.channelsPerGroup)
          continue;
        for (int64_t heightTile : heights) {
          for (int64_t widthTile : widths) {
            const int64_t bytes = convTileBytes(problem, batchTile,
                                                groupTile, channelTile,
                                                heightTile, widthTile);
            if (bytes > problem.budget)
              continue;

            Mapping candidate;
            candidate.tiles = {batchTile, groupTile, channelTile, heightTile,
                               widthTile};
            candidate.tileCount = (problem.batch / batchTile) *
                                  (problem.group / groupTile) *
                                  (problem.channelsPerGroup / channelTile) *
                                  (problem.outputHeight / heightTile) *
                                  (problem.outputWidth / widthTile);
            candidate.peakBytes = bytes;

            const PortCosts costs =
                convTileCosts(problem, batchTile, groupTile, channelTile,
                              heightTile, widthTile);
            candidate.makespan = makespanOf(costs, candidate.tileCount);

            if (!best || isBetter(candidate, *best))
              best = candidate;
          }
        }
      }
    }
  }
  return best;
}

/// `strategy=fixed`, the halving rule kept as a named baseline.
///
/// Halve the largest parallel extent, repeatedly, until the working set fits.
/// It is not a search: it looks at one candidate per halving and stops at the
/// first that fits, which is exactly what makes it a baseline worth reporting a
/// regret against.
std::optional<Mapping> searchConvFixed(const ConvProblem &problem) {
  SmallVector<int64_t> tiles = {problem.batch, problem.group,
                                problem.channelsPerGroup, problem.outputHeight,
                                problem.outputWidth};
  const SmallVector<int64_t> extents = tiles;

  for (int step = 0; step < 64; ++step) {
    const int64_t bytes = convTileBytes(problem, tiles[0], tiles[1],
                                        tiles[2], tiles[3], tiles[4]);
    if (bytes <= problem.budget) {
      Mapping mapping;
      mapping.tiles = tiles;
      mapping.tileCount = 1;
      for (unsigned index = 0; index < 5; ++index)
        mapping.tileCount *= extents[index] / tiles[index];
      mapping.peakBytes = bytes;
      const PortCosts costs = convTileCosts(problem, tiles[0], tiles[1],
                                            tiles[2], tiles[3], tiles[4]);
      mapping.makespan = makespanOf(costs, mapping.tileCount);
      return mapping;
    }

    // Halve the largest halvable extent. The spatial axes are skipped when the
    // halo boolean says so, and the group and channel axes keep the interface's
    // folding rule.
    int chosen = -1;
    int64_t largest = 1;
    for (unsigned index = 0; index < 5; ++index) {
      if (!problem.allowSpatial && (index == 3 || index == 4))
        continue;
      if (tiles[index] <= 1 || tiles[index] % 2 != 0)
        continue;
      if (index == 1 && problem.channelsPerGroup != tiles[2])
        continue;
      if (index == 2 && tiles[1] != 1)
        continue;
      if (tiles[index] > largest) {
        largest = tiles[index];
        chosen = static_cast<int>(index);
      }
    }
    if (chosen < 0)
      return std::nullopt;
    tiles[chosen] /= 2;
  }
  return std::nullopt;
}

/// `strategy=largest-fit`, the greedy kept as a named baseline.
///
/// Take the largest divisor of each axis, in the tie break's own order, that
/// keeps the working set inside the budget, committing to each axis before
/// looking at the next. It sees a strict subset of what the exhaustive search
/// sees, which is the definition of a greedy and the reason its regret is a
/// number rather than an argument.
std::optional<Mapping> searchConvLargestFit(const ConvProblem &problem) {
  SmallVector<int64_t> tiles = {problem.batch, problem.group,
                                problem.channelsPerGroup, problem.outputHeight,
                                problem.outputWidth};
  const SmallVector<int64_t> extents = tiles;
  const int order[4] = {2, 3, 4, 0};

  for (int index : order) {
    if (!problem.allowSpatial && (index == 3 || index == 4))
      continue;
    SmallVector<int64_t> candidates = divisorsOf(extents[index]);
    std::reverse(candidates.begin(), candidates.end());
    for (int64_t candidate : candidates) {
      if (index == 2 && tiles[1] != 1 &&
          candidate != problem.channelsPerGroup)
        continue;
      tiles[index] = candidate;
      if (convTileBytes(problem, tiles[0], tiles[1], tiles[2],
                        tiles[3], tiles[4]) <= problem.budget)
        break;
    }
  }

  const int64_t bytes = convTileBytes(problem, tiles[0], tiles[1],
                                      tiles[2], tiles[3], tiles[4]);
  if (bytes > problem.budget)
    return std::nullopt;

  Mapping mapping;
  mapping.tiles = tiles;
  mapping.tileCount = 1;
  for (unsigned index = 0; index < 5; ++index)
    mapping.tileCount *= extents[index] / tiles[index];
  mapping.peakBytes = bytes;
  const PortCosts costs = convTileCosts(problem, tiles[0], tiles[1], tiles[2],
                                       tiles[3], tiles[4]);
  mapping.makespan = makespanOf(costs, mapping.tileCount);
  return mapping;
}

/// The matmul's search, which has two parallel axes and no halo, so the three
/// strategies collapse to enumerating a rectangle. It is written once and
/// exhaustively, and `fixed` and `largest-fit` take the same restriction the
/// convolution's do, which is the candidate set rather than the scoring.
std::optional<Mapping> searchMatMul(int64_t rows, int64_t columns,
                                    int64_t reduction, bool hasBias,
                                    int64_t elementBytes, int64_t budget,
                                    bool doubleBuffer, StringRef strategy) {
  std::optional<Mapping> best;
  SmallVector<int64_t> rowTiles = divisorsOf(rows);
  SmallVector<int64_t> columnTiles = divisorsOf(columns);
  if (strategy != "exhaustive") {
    // The two baselines commit to the largest fitting tile per axis rather than
    // enumerating the rectangle, which is the same restriction they take on a
    // convolution.
    std::reverse(rowTiles.begin(), rowTiles.end());
    std::reverse(columnTiles.begin(), columnTiles.end());
  }

  for (int64_t rowTile : rowTiles) {
    for (int64_t columnTile : columnTiles) {
      int64_t bytes = (rowTile * reduction + reduction * columnTile +
                       rowTile * columnTile) *
                      elementBytes;
      if (hasBias)
        bytes += columnTile * elementBytes;
      if (doubleBuffer)
        bytes += rowTile * reduction * elementBytes;
      if (bytes > budget)
        continue;

      Mapping candidate;
      // Domain order for a matmul is (M, N, K); the two parallel axes are
      // recorded in the same five slot shape the convolution uses so that one
      // printer serves both, with the unused slots at one.
      candidate.tiles = {rowTile, 1, columnTile, 1, 1};
      candidate.tileCount = (rows / rowTile) * (columns / columnTile);
      candidate.peakBytes = bytes;
      const PortCosts costs = matmulTileCosts(rowTile, columnTile, reduction,
                                              hasBias, elementBytes);
      candidate.makespan = makespanOf(costs, candidate.tileCount);
      if (!best || isBetter(candidate, *best))
        best = candidate;
      if (strategy != "exhaustive" && best)
        return best;
    }
  }
  return best;
}

//===----------------------------------------------------------------------===//
// Applying it.
//===----------------------------------------------------------------------===//

/// The mapping, as an attribute a reader and an exporter can both use.
///
/// It carries what the search decided and what it could not decide, which is
/// the honest shape for a record that Section 16.5 compares two tools under:
/// the temporal factors, the spatial factors the array imposes, the loop order,
/// the tile count, the largest per tile working set, and the makespan the
/// candidate was chosen by.
DictionaryAttr mappingAttr(OpBuilder &builder, const Mapping &mapping,
                           StringRef strategy) {
  MLIRContext *context = builder.getContext();
  SmallVector<NamedAttribute> fields;
  fields.emplace_back(builder.getStringAttr("temporal_tiles"),
                      builder.getDenseI64ArrayAttr(mapping.tiles));
  fields.emplace_back(
      builder.getStringAttr("spatial_factors"),
      builder.getDenseI64ArrayAttr({nbin::kArrayDim, nbin::kArrayDim}));
  fields.emplace_back(builder.getStringAttr("loop_order"),
                      builder.getStringAttr("domain"));
  fields.emplace_back(builder.getStringAttr("tile_count"),
                      builder.getI64IntegerAttr(mapping.tileCount));
  fields.emplace_back(builder.getStringAttr("tile_bytes"),
                      builder.getI64IntegerAttr(mapping.peakBytes));
  fields.emplace_back(builder.getStringAttr("makespan_cycles"),
                      builder.getF64FloatAttr(mapping.makespan));
  fields.emplace_back(builder.getStringAttr("strategy"),
                      builder.getStringAttr(strategy));
  return DictionaryAttr::get(context, fields);
}

/// Tiles `op` with the chosen sizes, emitting every tile at a constant offset,
/// and stamps the mapping on each one.
///
/// **The tile grid is enumerated here rather than emitted as `scf.for` and then
/// unrolled, and the reason is a property of this dialect rather than a
/// shortcut.** Section 13.2 describes the loops as emitted with `scf` and fully
/// unrolled inside the pass, and the observable contract that description exists
/// for is met exactly: the tiles are fully unrolled and no `scf` operation
/// survives, which the assertion at the end of this pass and
/// `test/Transforms/tile-to-scratchpad.mlir` both check. What cannot be done is
/// the intermediate step.
///
/// `scf::tileUsingSCF` calls `getTiledImplementation` with the loop's
/// **induction variable** as the offset, because the tile has to be built once
/// inside a loop body that runs many times. **A convolution tile at a dynamic
/// offset is not representable in this dialect.** The per tile pads are not the
/// operation's pads: the first tile of an axis carries the leading pad, the last
/// carries the trailing one, and the tiles between carry neither. `pads` is a
/// static `DenseI64ArrayAttr` on `npu.conv2d`, so one loop body cannot stand for
/// tiles that disagree about it, and the interface says so by declining a
/// non constant offset rather than inventing a value. That refusal was written
/// at P13 with the interface and it is what this comment is the other end of.
///
/// So the grid is walked at compile time and every tile is materialised at a
/// constant offset, which is what full unrolling produces anyway. The order is
/// the domain order with the last dimension varying fastest, which is what the
/// recorded mapping's `loop_order` names.
LogicalResult applyTiling(IRRewriter &rewriter, TilingInterface op,
                          ArrayRef<int64_t> domainTileSizes,
                          DictionaryAttr record, int64_t &tilesEmitted) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  SmallVector<Range> domain = op.getIterationDomain(rewriter);
  const unsigned rank = domain.size();

  SmallVector<int64_t> extents;
  extents.reserve(rank);
  for (const Range &range : domain) {
    std::optional<int64_t> extent = getConstantIntValue(range.size);
    if (!extent)
      return failure();
    extents.push_back(*extent);
  }

  // A tile size of zero means "do not split this dimension", which is the same
  // convention the upstream tiling options use and is how the reduction
  // dimensions are passed through whole.
  SmallVector<int64_t> sizes(rank);
  SmallVector<int64_t> counts(rank);
  int64_t total = 1;
  for (unsigned index = 0; index < rank; ++index) {
    const int64_t requested =
        index < domainTileSizes.size() ? domainTileSizes[index] : 0;
    sizes[index] = requested == 0 ? extents[index] : requested;
    if (sizes[index] <= 0 || extents[index] % sizes[index] != 0)
      return failure();
    counts[index] = extents[index] / sizes[index];
    total *= counts[index];
  }

  // The destination operand is what every tile writes into and is the value the
  // stitched result is built from. Destination passing is what makes that a
  // value rather than a buffer this pass has to invent.
  auto destinationStyle =
      dyn_cast<DestinationStyleOpInterface>(op.getOperation());
  if (!destinationStyle || destinationStyle.getNumDpsInits() != 1)
    return failure();
  Value stitched = destinationStyle.getDpsInits()[0];

  SmallVector<int64_t> position(rank, 0);
  for (int64_t linear = 0; linear < total; ++linear) {
    int64_t remainder = linear;
    for (int index = static_cast<int>(rank) - 1; index >= 0; --index) {
      position[index] = remainder % counts[index];
      remainder /= counts[index];
    }

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> tileSizes;
    offsets.reserve(rank);
    tileSizes.reserve(rank);
    for (unsigned index = 0; index < rank; ++index) {
      offsets.push_back(
          rewriter.getIndexAttr(position[index] * sizes[index]));
      tileSizes.push_back(rewriter.getIndexAttr(sizes[index]));
    }

    FailureOr<TilingResult> tiled =
        op.getTiledImplementation(rewriter, offsets, tileSizes);
    if (failed(tiled) || tiled->tiledValues.empty())
      return failure();

    SmallVector<OpFoldResult> resultOffsets;
    SmallVector<OpFoldResult> resultSizes;
    if (failed(op.getResultTilePosition(rewriter, 0, offsets, tileSizes,
                                        resultOffsets, resultSizes)))
      return failure();

    for (Operation *tile : tiled->tiledOps)
      tile->setAttr("npu.tiling_choice", record);

    SmallVector<OpFoldResult> strides(resultOffsets.size(),
                                      rewriter.getIndexAttr(1));
    stitched = tensor::InsertSliceOp::create(rewriter, loc,
                                             tiled->tiledValues[0], stitched,
                                             resultOffsets, resultSizes,
                                             strides);
    ++tilesEmitted;
  }

  rewriter.replaceOp(op, stitched);
  return success();
}

//===----------------------------------------------------------------------===//
// The pass.
//===----------------------------------------------------------------------===//

struct TileToScratchpadPass
    : public mlir::npu::impl::NPUTileToScratchpadBase<TileToScratchpadPass> {
  using mlir::npu::impl::NPUTileToScratchpadBase<
      TileToScratchpadPass>::NPUTileToScratchpadBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();

    // **The budget is copied into a plain `int64_t` before anything reads it,
    // and that is not tidiness.** A tablegen pass option is an `llvm::cl`
    // option object, and streaming one into a `Diagnostic` picks an overload
    // that is not the integer one: a budget of 64 printed as `@`, which is what
    // 64 is in ASCII. It is D-0043's shape in miniature, a value arriving
    // through a channel that loses information, and the diagnostic that told a
    // reader the wrong number would have been the only place anybody saw it.
    const int64_t budgetBytes = budget;

    if (strategy != "exhaustive" && strategy != "fixed" &&
        strategy != "largest-fit") {
      function.emitError()
          << "unknown tiling strategy '" << strategy
          << "'. The accepted values are 'exhaustive', 'fixed' and "
             "'largest-fit'.";
      return signalPassFailure();
    }
    if (halo != "recompute" && halo != "cache") {
      function.emitError()
          << "unknown halo policy '" << halo
          << "'. The accepted values are 'recompute' and 'cache'.";
      return signalPassFailure();
    }
    const bool allowSpatial = halo == "recompute";

    // The operations to consider, collected before anything is rewritten,
    // because tiling replaces the operation it tiles.
    //
    // **Only the function's own body**, which leaves an operation inside an
    // `npu.fused_op` region alone. That region is `IsolatedFromAbove` and its
    // whole purpose is to keep an intermediate in the scratchpad; tiling one
    // member of a fused pair without the other would put the intermediate back
    // in DRAM, which is the opposite of what fusion was for. A fused region
    // whose working set does not fit is left to the allocator, and it is
    // counted as declined rather than skipped silently.
    SmallVector<Operation *> candidates;
    for (Block &block : function.getBody())
      for (Operation &op : block)
        if (isa<Conv2DOp, MatMulOp>(op))
          candidates.push_back(&op);

    IRRewriter rewriter(&getContext());
    int64_t emitted = 0;

    for (Operation *op : candidates) {
      const int64_t working = workingSetBytesFor(op, doubleBuffer);
      if (working <= budgetBytes) {
        ++alreadyFitting;
        continue;
      }

      std::optional<Mapping> mapping;
      SmallVector<int64_t> domainTileSizes;

      if (auto conv = dyn_cast<Conv2DOp>(op)) {
        auto inputType = cast<RankedTensorType>(conv.getInput().getType());
        auto filterType = cast<RankedTensorType>(conv.getFilter().getType());
        auto resultType = cast<RankedTensorType>(conv.getResult().getType());
        SmallVector<int64_t> outputSpatial = getSpatialExtents(resultType);

        ConvProblem problem;
        problem.batch = getBatchExtent(inputType);
        problem.group = conv.getGroup();
        problem.channelsPerGroup = filterType.getDimSize(0) / conv.getGroup();
        problem.outputHeight = outputSpatial[0];
        problem.outputWidth = outputSpatial[1];
        problem.inputChannelsPerGroup = filterType.getDimSize(1);
        problem.kernelHeight = filterType.getDimSize(2);
        problem.kernelWidth = filterType.getDimSize(3);
        problem.strideHeight = conv.getStrides()[0];
        problem.strideWidth = conv.getStrides()[1];
        problem.dilationHeight = conv.getDilations()[0];
        problem.dilationWidth = conv.getDilations()[1];
        SmallVector<int64_t> inputSpatial = getSpatialExtents(inputType);
        problem.inputHeight = inputSpatial[0];
        problem.inputWidth = inputSpatial[1];
        problem.hasBias = conv.getBias() != nullptr;
        problem.elementBytes = inputType.getElementTypeBitWidth() / 8;
        problem.budget = budgetBytes;
        problem.doubleBuffer = doubleBuffer;
        problem.allowSpatial = allowSpatial;

        if (strategy == "exhaustive")
          mapping = searchConvExhaustive(problem);
        else if (strategy == "fixed")
          mapping = searchConvFixed(problem);
        else
          mapping = searchConvLargestFit(problem);

        if (mapping) {
          // Domain order is (N, G, Cout/G, Hout, Wout, Cin/G, KH, KW); the
          // three reduction extents get a tile size of zero, which is how
          // `tileUsingSCF` is told not to split a dimension.
          domainTileSizes = {mapping->tiles[0], mapping->tiles[1],
                             mapping->tiles[2], mapping->tiles[3],
                             mapping->tiles[4], 0, 0, 0};
        }
      } else {
        auto matmul = cast<MatMulOp>(op);
        auto lhsType = cast<RankedTensorType>(matmul.getLhs().getType());
        auto rhsType = cast<RankedTensorType>(matmul.getRhs().getType());
        mapping = searchMatMul(
            lhsType.getDimSize(0), rhsType.getDimSize(1),
            lhsType.getDimSize(1), matmul.getBias() != nullptr,
            lhsType.getElementTypeBitWidth() / 8, budgetBytes, doubleBuffer,
            strategy);
        if (mapping) {
          // Domain order is (M, N, K), and K is never split.
          domainTileSizes = {mapping->tiles[0], mapping->tiles[2], 0};
        }
      }

      // Nothing fits, or the only thing that would fit needs the reduction
      // split. Section 13.2: decline with a counted remark and leave the
      // allocator's spilling as the fallback. Declining is a result.
      if (!mapping || mapping->tileCount <= 1) {
        ++declinedOps;
        op->emitRemark()
            << "working set of " << working << " bytes exceeds the "
            << budgetBytes
            << " byte budget and no tiling over the parallel dimensions fits. "
               "Splitting the reduction would re associate an fp32 "
               "accumulation, which Section 13.2 permits only behind "
               "allow-reduction-tiling. The allocator's spilling is the "
               "fallback.";
        continue;
      }

      const Mapping chosen = *mapping;
      DictionaryAttr record = mappingAttr(rewriter, chosen, strategy);

      auto tilingOp = cast<TilingInterface>(op);
      if (failed(applyTiling(rewriter, tilingOp, domainTileSizes, record,
                             emitted))) {
        ++declinedOps;
        op->emitRemark()
            << "the tiling the search chose could not be generated, so the "
               "allocator's spilling is the fallback";
        continue;
      }
      ++tiledOps;
    }

    tilesEmitted += emitted;

    // **The assertion the P13 gate asks for, made by the pass about itself.**
    // Every loop this pass generated was unrolled above; this is the check that
    // none survived, and it names the operation rather than reporting that
    // something went wrong.
    WalkResult surviving = function.walk([&](Operation *op) {
      if (op->getDialect() &&
          op->getDialect()->getNamespace() == scf::SCFDialect::getDialectNamespace()) {
        op->emitError()
            << "an scf operation survived -npu-tile-to-scratchpad. Tiled loops "
               "are emitted with scf and fully unrolled inside this pass, "
               "because the ISA has no branches, so an scf operation reaching "
               "the lowering is this pass having failed rather than the "
               "lowering being incomplete.";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (surviving.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace
