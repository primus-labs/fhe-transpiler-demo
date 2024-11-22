#include "heir/Passes/combine/CombineExtract.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Operation.h"
#include "mlir/Rewrite/PatternApplicator.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void CombineExtractPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<heir::HEIRDialect,
                    mlir::AffineDialect,
                    func::FuncDialect,
                    mlir::scf::SCFDialect>();
}

class RemoveUnusedComparePattern : public OpConversionPattern<heir::FHECmpOp> {
public:
    using OpConversionPattern<heir::FHECmpOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(heir::FHECmpOp compareOp, heir::FHECmpOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        if (compareOp->use_empty()) {
            auto encodeOp = compareOp.getOperand(1).getDefiningOp<heir::FHEEncodeOp>();
            rewriter.eraseOp(compareOp);
            if (encodeOp && encodeOp->use_empty()) {
                rewriter.eraseOp(encodeOp);
            }
            return success();
        }
        return failure();
    }
};




void CombineExtractPass::runOnOperation()
{
    MLIRContext &context = getContext();
    mlir::ConversionTarget target(context);
    target.addLegalDialect<heir::HEIRDialect,
                           mlir::AffineDialect,
                           func::FuncDialect,
                           mlir::scf::SCFDialect,
                           arith::ArithmeticDialect>();
    target.addIllegalOp<heir::FHECmpOp>();
    mlir::RewritePatternSet patterns(&context);
    patterns.add<RemoveUnusedComparePattern>(&context);
    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
    IRRewriter rewriter(&getContext());
    module.walk([&](func::FuncOp funcOp)
    {
        funcOp.walk([&](Operation *op)
        {
            if (auto opp = dyn_cast_or_null<FHEEncodeOp>(op))
            {

                if (opp->getUsers().empty()) {
                    rewriter.eraseOp(opp);
                }
            }
            return WalkResult::advance();
        });
    });
    module.walk([&](func::FuncOp funcOp)
    {
        funcOp.walk([&](Operation *op)
        {
            if (auto extractOp = dyn_cast<FHEExtractfinalOp>(op))
            {
                SmallVector<FHEExtractfinalOp, 64> extractOps;
                SmallVector<HEIRLutOp, 64> lutOps;
                SmallVector<FHEInsertfinalOp, 64> insertOps;
                SmallVector<int64_t, 64> indices;
                extractOps.push_back(extractOp);
                Value sourceVar = extractOp.getOperand();
                Value targetVar = nullptr;
                bool patternMatched = true;
                Operation *currentOp = extractOp->getNextNode();
                auto excolAttr = extractOp.colAttr().cast<IntegerAttr>();
                indices.push_back(excolAttr.getInt());
                while (currentOp)
                {
                    auto lutOp = dyn_cast<HEIRLutOp>(currentOp);
                    if (!lutOp || lutOp.getOperand() != extractOps.back().getResult())
                    {
                        patternMatched = false;
                        break;
                    }
                    lutOps.push_back(lutOp);

                    currentOp = currentOp->getNextNode();
                    auto insertOp = dyn_cast<FHEInsertfinalOp>(currentOp);
                    if (!insertOp)
                    {
                        patternMatched = false;
                        break;
                    }
                    auto incolAttr = insertOp.colAttr().cast<IntegerAttr>();
                    if (!incolAttr)
                    {
                        patternMatched = false;
                        break;
                    }
                    if (excolAttr.getInt() != incolAttr.getInt())
                    {
                        patternMatched = false;
                        break;
                    }
                    if (!targetVar)
                        targetVar = insertOp.memref();
                    else if (insertOp.memref() != targetVar)
                    {
                        patternMatched = false;
                        break;
                    }
                    insertOps.push_back(insertOp);
                    currentOp = insertOp->getNextNode();
                    if (auto nextExtractOp = dyn_cast<FHEExtractfinalOp>(currentOp))
                    {
                        if (nextExtractOp.getOperand() != sourceVar)
                        {
                            patternMatched = false;
                            break;
                        }
                        excolAttr = nextExtractOp.colAttr().cast<IntegerAttr>();
                        if (!excolAttr)
                        {
                            patternMatched = false;
                            break;
                        }

                        indices.push_back(excolAttr.getInt());
                        extractOps.push_back(nextExtractOp);
                        currentOp = nextExtractOp->getNextNode();
                    }
                    else
                    {
                        break;
                    }
                }
                if (patternMatched)
                {
                    auto vecType = sourceVar.getType().dyn_cast<LWECipherVectorType>();
                    if (!vecType)
                    {
                        return WalkResult::advance();
                    }
                    rewriter.setInsertionPoint(extractOps.front());
                    auto firstLutOp = lutOps.front();
                    auto edge1Attr = firstLutOp.edge1Attr();
                    auto edge2Attr = firstLutOp.edge2Attr();
                    auto thresholdAttr = firstLutOp.thresholdAttr();

                    auto newLutOp = rewriter.create<HEIRLutOp>(
                        extractOp.getLoc(),
                        sourceVar.getDefiningOp<FHEMaterializeOp>().getOperand().getType(),
                        sourceVar.getDefiningOp<FHEMaterializeOp>().getOperand(),
                        edge1Attr,
                        edge2Attr,
                        thresholdAttr);
                    rewriter.create<HEIRCopyOp>(extractOp.getLoc(), newLutOp.getResult(), targetVar.getDefiningOp<FHEMaterializeOp>().getOperand());
                    for (auto op : insertOps)
                    {
                        rewriter.eraseOp(op);
                    }
                    for (auto op : lutOps)
                    {
                        rewriter.eraseOp(op);
                    }
                    for (auto op : extractOps)
                    {
                        rewriter.eraseOp(op);
                    }
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
    });

    module.walk([&](func::FuncOp funcOp)
    {
        funcOp.walk([&](Operation *op)
        {
            if (auto extractOp = dyn_cast_or_null<FHEExtractfinalOp>(op))
            {
                SmallVector<FHEExtractfinalOp, 8> extractOps;
                SmallVector<FHEInsertfinalOp, 8> insertOps;
                extractOps.push_back(extractOp);
                Value sourceVar = extractOp.getOperand();
                Value targetVar = nullptr;
                bool patternMatched = true;
                Operation *currentOp = op->getNextNode();
                auto excolAttr = extractOp.colAttr().cast<IntegerAttr>();
                while (currentOp)
                {
                    if (auto insertOp = dyn_cast_or_null<FHEInsertfinalOp>(currentOp))
                    {
                        auto incolAttr = insertOp.colAttr().cast<IntegerAttr>();
                        if (!targetVar)
                            targetVar = insertOp.memref();
                        else if (insertOp.memref() != targetVar)
                        {
                            patternMatched = false;
                            break;
                        }
                        if (excolAttr != incolAttr)
                        {
                            patternMatched = false;
                            break;
                        }

                        insertOps.push_back(insertOp);
                        currentOp = insertOp->getNextNode();
                        if (auto nextExtractOp = dyn_cast_or_null<FHEExtractfinalOp>(currentOp))
                        {
                            if (nextExtractOp.getOperand() != sourceVar)
                            {
                                patternMatched = false;
                                break;
                            }
                            extractOps.push_back(nextExtractOp);
                            excolAttr = nextExtractOp.colAttr().cast<IntegerAttr>();
                            currentOp = nextExtractOp->getNextNode();
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                if (patternMatched && !insertOps.empty() && extractOps.size() == insertOps.size())
                {
                    rewriter.setInsertionPoint(extractOps.front());
                    rewriter.create<HEIRCopyOp>(extractOp.getLoc(), sourceVar, targetVar.getDefiningOp<FHEMaterializeOp>().getOperand());
                    for (auto it = insertOps.rbegin(); it != insertOps.rend(); ++it) {
                        rewriter.eraseOp(*it);
                    }
                    for (auto it = extractOps.rbegin(); it != extractOps.rend(); ++it) {
                        rewriter.eraseOp(*it);
                    }
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });
    });
}