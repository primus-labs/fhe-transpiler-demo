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

void CombineExtractPass::runOnOperation()
{
    auto module = getOperation();
    IRRewriter rewriter(&getContext());
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
