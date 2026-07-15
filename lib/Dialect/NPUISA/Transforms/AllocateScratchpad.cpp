//===- AllocateScratchpad.cpp - Assign scratchpad addresses ---------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::npuisa {

#define GEN_PASS_DEF_NPUALLOCATESCRATCHPAD
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

namespace {

// Byte size of a scratchpad buffer value (fp32).
static int64_t bufferBytes(Value v) {
  auto buffer = llvm::cast<BufferType>(v.getType());
  return buffer.getTensorType().getNumElements() * 4;
}

static bool isBuffer(Value v) { return llvm::isa<BufferType>(v.getType()); }

// Record an assigned byte offset on the producing instruction.
static void setAddress(Operation *op, int64_t offset) {
  auto attr = IntegerAttr::get(IntegerType::get(op->getContext(), 64), offset);
  llvm::TypeSwitch<Operation *>(op)
      .Case<DmaLoadOp, Conv2DOp, MatMulOp, ReluOp, ReshapeOp, AddOp, MulOp,
            PoolMaxOp, PoolAvgOp>([&](auto typed) { typed.setAddressAttr(attr); });
}

struct Live {
  Value value;
  int defIdx;
  int lastUse;
  int64_t size;
  int64_t offset = -1;
};

// Liveness of every buffer over the block's straight line instruction order.
static SmallVector<Live> computeLiveness(Block &block) {
  DenseMap<Value, int> defIdx, lastUse;
  int idx = 0;
  SmallVector<Value> order;
  for (Operation &op : block) {
    for (Value operand : op.getOperands())
      if (isBuffer(operand))
        lastUse[operand] = idx;
    for (Value result : op.getResults())
      if (isBuffer(result)) {
        defIdx[result] = idx;
        lastUse[result] = idx;
        order.push_back(result);
      }
    ++idx;
  }
  SmallVector<Live> result;
  for (Value v : order)
    result.push_back({v, defIdx[v], lastUse[v], bufferBytes(v)});
  return result;
}

// Peak resident bytes and the instruction index where it occurs.
static std::pair<int64_t, int> peakPressure(ArrayRef<Live> lives, int numOps) {
  int64_t peak = 0;
  int peakIdx = 0;
  for (int i = 0; i < numOps; ++i) {
    int64_t sum = 0;
    for (const Live &l : lives)
      if (l.defIdx <= i && i <= l.lastUse)
        sum += l.size;
    if (sum > peak) {
      peak = sum;
      peakIdx = i;
    }
  }
  return {peak, peakIdx};
}

// Spill a buffer: store it to DRAM after it is produced and reload it before each
// later use, so it stops occupying the scratchpad across its live range.
static void spill(Value buffer, OpBuilder &builder) {
  Operation *def = buffer.getDefiningOp();
  auto tensorType = llvm::cast<BufferType>(buffer.getType()).getTensorType();

  builder.setInsertionPointAfter(def);
  auto store = DmaStoreOp::create(builder, def->getLoc(), tensorType, buffer);
  Value dram = store.getDest();

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : buffer.getUses())
    if (use.getOwner() != store)
      uses.push_back(&use);

  for (OpOperand *use : uses) {
    builder.setInsertionPoint(use->getOwner());
    auto reload = DmaLoadOp::create(builder, use->getOwner()->getLoc(),
                                    buffer.getType(), dram, /*address=*/nullptr);
    use->set(reload.getDest());
  }
}

// Pick the longest lived buffer resident at the peak point to spill.
static Value pickVictim(ArrayRef<Live> lives, int peakIdx) {
  Value victim;
  int bestSpan = 0;
  for (const Live &l : lives) {
    if (l.defIdx <= peakIdx && peakIdx <= l.lastUse && l.lastUse > l.defIdx) {
      int span = l.lastUse - l.defIdx;
      if (span > bestSpan) {
        bestSpan = span;
        victim = l.value;
      }
    }
  }
  return victim;
}

// First fit address assignment over a free list of (offset, size) intervals.
struct FreeList {
  SmallVector<std::pair<int64_t, int64_t>> intervals; // (offset, size)

  explicit FreeList(int64_t budget) { intervals.push_back({0, budget}); }

  std::optional<int64_t> allocate(int64_t size) {
    for (auto &iv : intervals) {
      if (iv.second >= size) {
        int64_t off = iv.first;
        iv.first += size;
        iv.second -= size;
        return off;
      }
    }
    return std::nullopt;
  }

  void free(int64_t offset, int64_t size) {
    intervals.push_back({offset, size});
    llvm::sort(intervals);
    SmallVector<std::pair<int64_t, int64_t>> merged;
    for (auto &iv : intervals) {
      if (iv.second == 0)
        continue;
      if (!merged.empty() && merged.back().first + merged.back().second == iv.first)
        merged.back().second += iv.second;
      else
        merged.push_back(iv);
    }
    intervals = std::move(merged);
  }
};

struct NPUAllocateScratchpadPass
    : public impl::NPUAllocateScratchpadBase<NPUAllocateScratchpadPass> {
  using NPUAllocateScratchpadBase::NPUAllocateScratchpadBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getBody().empty())
      return;
    Block &block = func.getBody().front();
    OpBuilder builder(&getContext());
    int64_t budget = budgetBytes;

    // Phase A: spill until the peak working set fits the budget.
    unsigned guard = 0;
    while (true) {
      SmallVector<Live> lives = computeLiveness(block);
      auto [peak, peakIdx] =
          peakPressure(lives, static_cast<int>(block.getOperations().size()));
      if (peak <= budget)
        break;
      Value victim = pickVictim(lives, peakIdx);
      if (!victim || ++guard > lives.size() + 1) {
        func.emitError("scratchpad budget ")
            << budget << " too small: a single buffer exceeds it";
        signalPassFailure();
        return;
      }
      spill(victim, builder);
    }

    // Phase B: assign byte offsets by linear scan first fit.
    SmallVector<Live> lives = computeLiveness(block);
    FreeList freeList(budget);
    SmallVector<Live> active;
    int64_t highWater = 0;
    for (const Live &b : lives) {
      // Expire buffers that died before this definition.
      for (auto it = active.begin(); it != active.end();) {
        if (it->lastUse < b.defIdx) {
          freeList.free(it->offset, it->size);
          it = active.erase(it);
        } else {
          ++it;
        }
      }
      std::optional<int64_t> off = freeList.allocate(b.size);
      if (!off) {
        func.emitError("scratchpad allocation failed after spilling");
        signalPassFailure();
        return;
      }
      setAddress(b.value.getDefiningOp(), *off);
      highWater = std::max(highWater, *off + b.size);
      Live placed = b;
      placed.offset = *off;
      active.push_back(placed);
    }

    func->setAttr("npuisa.scratchpad_bytes",
                  builder.getI64IntegerAttr(highWater));
    func->setAttr("npuisa.scratchpad_budget",
                  builder.getI64IntegerAttr(budget));
  }
};

} // namespace
} // namespace mlir::npuisa
