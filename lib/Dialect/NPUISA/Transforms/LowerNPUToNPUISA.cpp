//===- LowerNPUToNPUISA.cpp - Lower npu to npuisa -------------------------===//
//
// Part of the npu-mlir project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
//
//===----------------------------------------------------------------------===//

#include "NPU/Dialect/NPUISA/Transforms/Passes.h"

#include "NPU/Dialect/NPU/IR/NPUOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISADialect.h"
#include "NPU/Dialect/NPUISA/IR/NPUISAOps.h"
#include "NPU/Dialect/NPUISA/IR/NPUISATypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::npuisa {

#define GEN_PASS_DEF_NPULOWERTONPUISA
#include "NPU/Dialect/NPUISA/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type converter: DRAM tensor <-> scratchpad buffer
//===----------------------------------------------------------------------===//

class TensorToBufferConverter : public TypeConverter {
public:
  explicit TensorToBufferConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](RankedTensorType type) -> Type {
      return BufferType::get(ctx, type);
    });

    // A scratchpad buffer needed from a DRAM tensor is a dma_load.
    addTargetMaterialization([](OpBuilder &builder, BufferType type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return DmaLoadOp::create(builder, loc, type, inputs[0], /*address=*/nullptr)
          .getDest();
    });
    // A DRAM tensor needed from a scratchpad buffer is a dma_store.
    addSourceMaterialization([](OpBuilder &builder, RankedTensorType type,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      return DmaStoreOp::create(builder, loc, type, inputs[0]).getDest();
    });
  }
};

// Convert a (possibly already buffer) value to a scratchpad buffer.
static Value asBuffer(ConversionPatternRewriter &rewriter, Location loc,
                      const TypeConverter *tc, Value v) {
  if (llvm::isa<BufferType>(v.getType()))
    return v;
  Type bufferType = tc->convertType(v.getType());
  return tc->materializeTargetConversion(rewriter, loc, bufferType, v);
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

struct ConstantLowering : OpConversionPattern<npu::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(npu::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    Value dram = ConstOp::create(rewriter, loc, tensorType, op.getValueAttr());
    Type bufferType = getTypeConverter()->convertType(tensorType);
    Value loaded =
        DmaLoadOp::create(rewriter, loc, bufferType, dram, /*address=*/nullptr);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

struct ConvLowering : OpConversionPattern<npu::Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(npu::Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *tc = getTypeConverter();
    Value in = asBuffer(rewriter, loc, tc, adaptor.getInput());
    Value w = asBuffer(rewriter, loc, tc, adaptor.getWeight());
    Value bias = adaptor.getBias()
                     ? asBuffer(rewriter, loc, tc, adaptor.getBias())
                     : Value();
    Type resultType = tc->convertType(op.getType());
    auto activation =
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getActivation()));
    rewriter.replaceOpWithNewOp<Conv2DOp>(
        op, resultType, in, w, bias, op.getStridesAttr(), op.getPadsAttr(),
        op.getDilationsAttr(), op.getGroupAttr(), activation,
        /*address=*/IntegerAttr());
    return success();
  }
};

struct MatMulLowering : OpConversionPattern<npu::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(npu::MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *tc = getTypeConverter();
    Value lhs = asBuffer(rewriter, loc, tc, adaptor.getLhs());
    Value rhs = asBuffer(rewriter, loc, tc, adaptor.getRhs());
    Value bias = adaptor.getBias()
                     ? asBuffer(rewriter, loc, tc, adaptor.getBias())
                     : Value();
    Type resultType = tc->convertType(op.getType());
    auto activation =
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getActivation()));
    rewriter.replaceOpWithNewOp<MatMulOp>(op, resultType, lhs, rhs, bias,
                                          activation, /*address=*/IntegerAttr());
    return success();
  }
};

template <typename NpuOp, typename IsaOp>
struct UnaryLowering : OpConversionPattern<NpuOp> {
  using OpConversionPattern<NpuOp>::OpConversionPattern;
  using Adaptor = typename NpuOp::Adaptor;
  LogicalResult
  matchAndRewrite(NpuOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *tc = this->getTypeConverter();
    Value in = asBuffer(rewriter, op.getLoc(), tc, adaptor.getInput());
    Type resultType = tc->convertType(op.getType());
    rewriter.template replaceOpWithNewOp<IsaOp>(op, resultType, in,
                                                /*address=*/IntegerAttr());
    return success();
  }
};

template <typename NpuOp, typename IsaOp>
struct BinaryLowering : OpConversionPattern<NpuOp> {
  using OpConversionPattern<NpuOp>::OpConversionPattern;
  using Adaptor = typename NpuOp::Adaptor;
  LogicalResult
  matchAndRewrite(NpuOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *tc = this->getTypeConverter();
    Value lhs = asBuffer(rewriter, op.getLoc(), tc, adaptor.getLhs());
    Value rhs = asBuffer(rewriter, op.getLoc(), tc, adaptor.getRhs());
    Type resultType = tc->convertType(op.getType());
    rewriter.template replaceOpWithNewOp<IsaOp>(op, resultType, lhs, rhs,
                                                /*address=*/IntegerAttr());
    return success();
  }
};

template <typename NpuOp, typename IsaOp>
struct PoolLowering : OpConversionPattern<NpuOp> {
  using OpConversionPattern<NpuOp>::OpConversionPattern;
  using Adaptor = typename NpuOp::Adaptor;
  LogicalResult
  matchAndRewrite(NpuOp op, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *tc = this->getTypeConverter();
    Value in = asBuffer(rewriter, op.getLoc(), tc, adaptor.getInput());
    Type resultType = tc->convertType(op.getType());
    rewriter.template replaceOpWithNewOp<IsaOp>(
        op, resultType, in, op.getKernelShapeAttr(), op.getStridesAttr(),
        op.getPadsAttr(), /*address=*/IntegerAttr());
    return success();
  }
};

struct NPULowerToNPUISAPass
    : public impl::NPULowerToNPUISABase<NPULowerToNPUISAPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    TensorToBufferConverter converter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<NPUISADialect>();
    target.addLegalOp<func::FuncOp>();
    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      return llvm::all_of(op.getOperandTypes(), [](Type t) {
        return llvm::isa<RankedTensorType>(t);
      });
    });
    target.addIllegalDialect<npu::NPUDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConstantLowering, ConvLowering, MatMulLowering>(converter, ctx);
    patterns.add<UnaryLowering<npu::ReluOp, ReluOp>,
                 UnaryLowering<npu::ReshapeOp, ReshapeOp>>(converter, ctx);
    patterns.add<BinaryLowering<npu::AddOp, AddOp>,
                 BinaryLowering<npu::MulOp, MulOp>>(converter, ctx);
    patterns.add<PoolLowering<npu::MaxPool2DOp, PoolMaxOp>,
                 PoolLowering<npu::AvgPool2DOp, PoolAvgOp>>(converter, ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::npuisa
