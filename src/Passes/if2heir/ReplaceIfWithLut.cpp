// Author: Shen Ruiyu
#include "heir/Passes/if2heir/ReplaceIfWithLut.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"


using namespace mlir;
using namespace arith;
using namespace heir;
using namespace memref;

void ReplaceIfWithLutPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

class ReplaceIfWithLutPattern final : public OpConversionPattern<scf::IfOp>
{
public:
    using OpConversionPattern<scf::IfOp>::OpConversionPattern;


    LogicalResult matchAndRewrite(
        scf::IfOp ifOp, typename scf::IfOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        Value condition = ifOp.getCondition();

        // 检查条件是否由heir.materialize和heir.compare生成
        auto materializeOp = condition.getDefiningOp<heir::FHEMaterializeOp>();
        if (!materializeOp)
            return failure();

        auto compareOp = materializeOp.getOperand().getDefiningOp<heir::FHECmpOp>();
        if (!compareOp)
            return failure();

        // 从heir.encode中获取threshold值
        auto encodeOp = compareOp.getOperand(1).getDefiningOp<heir::FHEEncodeOp>();
        if (!encodeOp)
            return failure();
        // 获取encode操作的message属性作为threshold
        float threshold = encodeOp.message().convertToDouble();
        // 从scf.if的then和else分支中获取edge1和edge2的值
        // Then分支（真分支）
        Block &thenBlock = ifOp.getThenRegion().front();
        float edge1 = 0.0;
        if (auto thenMaterializeOp = dyn_cast<heir::FHEMaterializeOp>(thenBlock.front())) {
            // 获取常量操作
            if (auto constOp = thenMaterializeOp.getOperand().getDefiningOp<arith::ConstantOp>()) {
                auto constAttr = constOp.getValue().dyn_cast<FloatAttr>();
                if (constAttr)
                    edge1 = constAttr.getValueAsDouble();
            }
        }
        // Else分支（假分支）
        Block &elseBlock = ifOp.getElseRegion().front();
        float edge2 = 0.0;
        if (auto elseMaterializeOp = dyn_cast<heir::FHEMaterializeOp>(elseBlock.front())) {
            // 获取常量操作
            if (auto constOp = elseMaterializeOp.getOperand().getDefiningOp<arith::ConstantOp>()) {
                auto constAttr = constOp.getValue().dyn_cast<FloatAttr>();
                if (constAttr)
                    edge2 = constAttr.getValueAsDouble();
            }
        }
        // 获取heir.extract操作
        auto extractOp = compareOp.getOperand(0).getDefiningOp<heir::FHEExtractfinalOp>();
        if (!extractOp)
            return failure();

        // 创建heir.lut操作，使用edge1、edge2和threshold
        rewriter.setInsertionPointAfter(extractOp);
        auto lutOp = rewriter.create<heir::HEIRLutOp>(
            extractOp.getLoc(), extractOp.getType(), extractOp.getResult(),
            rewriter.getF32FloatAttr(edge2),  // edge2（下界）
            rewriter.getF32FloatAttr(edge1),  // edge1（上界）
            rewriter.getF32FloatAttr(threshold));

        // 收集scf.if的then和else分支中的heir.insert操作
        SmallVector<heir::FHEInsertfinalOp, 1> insertOps;
        // Then分支
        thenBlock.walk([&](heir::FHEInsertfinalOp insertOp) {
            insertOps.push_back(insertOp);
        });

        // 确保至少有一个insert操作
        if (insertOps.empty())
            return failure();

        // 使用lutOp的结果创建新的heir.insert操作，并替换原有的insert操作
        for (auto insertOp : insertOps) {
            rewriter.setInsertionPoint(ifOp);
            rewriter.create<heir::FHEInsertfinalOp>(
            insertOp.getLoc(),
            lutOp.getResult(),
            insertOp.memref(),
            insertOp.colAttr(),
            Attribute()
        );
        }
        // 移除原始的scf.if操作
        rewriter.eraseOp(ifOp);

        return success();
    }

 
};



void ReplaceIfWithLutPass::runOnOperation() {
    // 获取当前操作的上下文
    MLIRContext &context = getContext();

    // 定义转换目标
    mlir::ConversionTarget target(context);
    target.addLegalDialect<HEIRDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    // target.addLegalDialect<cf::ControlFlowDialect>();
    target.addIllegalOp<scf::IfOp>();


    // 定义重写模式
    mlir::RewritePatternSet patterns(&context);
    patterns.add<ReplaceIfWithLutPattern>(&context);

    // 应用部分转换
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}
