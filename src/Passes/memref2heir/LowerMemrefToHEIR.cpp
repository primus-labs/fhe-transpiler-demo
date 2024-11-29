// Author: Zian Zhao
#include "heir/Passes/memref2heir/LowerMemrefToHEIR.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"



using namespace mlir;
using namespace arith;
using namespace heir;
using namespace memref;

void LowerMemrefToHEIRPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

class MemRefLoadPattern final : public OpConversionPattern<LoadOp> {
public:
    using OpConversionPattern<LoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        LoadOp op, typename LoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();

        if (auto st = dstType.dyn_cast_or_null<Float32Type>()) {
            int size;
            Value batchedCipher;
            auto memRefType = op.getMemRef().getType().dyn_cast<MemRefType>();
            if (!memRefType || !memRefType.hasStaticShape())
                return failure();

            if (memRefType.getShape().size() == 1) {
                size = memRefType.getShape().front();

                batchedCipher = typeConverter->materializeTargetConversion(
                    rewriter,
                    op.getMemRef().getLoc(),
                    LWECipherVectorType::get(getContext(), st, size),
                    op.getMemRef());
            }
            else if (memRefType.getShape().size() == 2) {
                int row = memRefType.getShape().front();
                int column = memRefType.getShape().back();

                batchedCipher = typeConverter->materializeTargetConversion(
                    rewriter,
                    op.getMemRef().getLoc(),
                    LWECipherMatrixType::get(getContext(), st, row, column),
                    op.getMemRef());
            } else {
                return failure();
            }

            SmallVector<Value, 8> indices(adaptor.getIndices());

            rewriter.replaceOpWithNewOp<FHEExtractOp>(op, dstType, batchedCipher, indices);
            return success();
        }
        return failure();
    }
};



// Transform the AffineLoadOp into FHEExtractOp, convert memref type into LWECipherVec type
class AffineLoadPattern final : public OpConversionPattern<affine::AffineLoadOp>
{
public:
    using OpConversionPattern<affine::AffineLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        affine::AffineLoadOp op, typename affine::AffineLoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        
        if (auto st = dstType.dyn_cast_or_null<Float32Type>())
        {
            int size;
            Value batchedCipher;
            auto tt = op.getMemRef().getType().dyn_cast<MemRefType>();
            if (tt.hasStaticShape() && tt.getShape().size() == 1) {
                size = tt.getShape().front();
            
                batchedCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                            op.getMemRef().getLoc(),
                                                                            LWECipherVectorType::get(getContext(),
                                                                            st, size),
                                                                            op.getMemRef());
            }

            else if (tt.hasStaticShape() && tt.getShape().size() == 2) {
                int row = tt.getShape().front();
                int column = tt.getShape().back();

                batchedCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                            op.getMemRef().getLoc(),
                                                                            LWECipherMatrixType::get(getContext(),
                                                                            st, row, column), op.getMemRef());
            }

            SmallVector<Value, 8> indices(op.getMapOperands());
            auto resultOperands =
                affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
            if (!resultOperands)
                return failure();
        
            rewriter.replaceOpWithNewOp<FHEExtractOp>(op, dstType, batchedCipher, *resultOperands);
        }
        return success();
    }
};

// Transform the AffineLoadOp into FHEInsertOp, convert memref type into LWECipherVec type
class AffineStorePattern final : public OpConversionPattern<affine::AffineStoreOp>
{
public:
    using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        affine::AffineStoreOp op, typename affine::AffineStoreOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        Type elementType = op.getMemRef().getType();

        if (elementType.isa<MemRefType>())
            elementType = elementType.cast<MemRefType>().getElementType();
        if (auto st = elementType.dyn_cast_or_null<Float32Type>()) {
            Value batchedCipher;
            auto tt = op.getMemRef().getType().dyn_cast<MemRefType>();
            if (tt.hasStaticShape() && tt.getShape().size() == 1) {
                auto size = tt.getShape().front();
                batchedCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                            op.getMemRef().getLoc(),
                                                                            LWECipherVectorType::get(getContext(),
                                                                            st, size),
                                                                            op.getMemRef());
            // llvm::errs()<<"\ngo1\n"<<op.getMemRef()<<"\n"<<op.getMemRef().getType()<<"\n"<<batchedCipher<<"\n";
            }

            else if (tt.hasStaticShape() && tt.getShape().size() == 2) {
                int row = tt.getShape().front();
                int column = tt.getShape().back();
                batchedCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                            op.getMemRef().getLoc(),
                                                                            LWECipherMatrixType::get(getContext(),
                                                                            st, row, column), op.getMemRef());
            // llvm::errs()<<"\ngo2\n";
            }
            // llvm::errs()<<"\n"<<op<<"\n"<<op.getValue()<<"\n"<<op.getValueToStore()<<"\n"<<batchedCipher<<"\n";
            SmallVector<Value, 8> indices(op.getMapOperands());
            auto resultOperands =
                affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
            if (!resultOperands)
                return failure();

            rewriter.replaceOpWithNewOp<FHEInsertOp>(op, op.getValueToStore(), batchedCipher, *resultOperands);
            // llvm::errs()<<"\n"<<op<<"\n"<<op.getValue()<<"\n"<<op.getValueToStore()<<"\n"<<batchedCipher<<"\n";
        }
        
        return success();
    }
};

// For polygeist operator used to load a vector from a matrx, we replace this op
// into FHEVectorLoadOp in polygeist_eliminator.py script
// TODO: get rid of the scripts, define polygeist dialect in heir and replace these
// ops by MLIR C++ passes  
class FHEVectorLoadPattern final : public OpConversionPattern<FHEVectorLoadOp>
{
public:
    using OpConversionPattern<FHEVectorLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEVectorLoadOp op, typename FHEVectorLoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType) 
            return failure();
        
        auto memrefType = typeConverter->convertType(op.getMemref().getType());
        auto new_memref = typeConverter->materializeTargetConversion(rewriter, op.getMemref().getLoc(),
                                                                        memrefType, op.getMemref());
        
        auto cOp = op.getIndices().getDefiningOp<arith::ConstantOp>();
        if (!cOp)
        {
            emitError(op.getLoc(),
                    "cannot find the definition of index in heir.extract_init op!");
            return failure();
        }
        auto indexAttr = cOp.getValue().cast<IntegerAttr>();

        rewriter.replaceOpWithNewOp<FHEVectorLoadfinalOp>(op, dstType, new_memref, indexAttr);

        return success();

    }

};

// Transform the vector/matrix related operations into homomorphic operators
// and convert the data type of input/output from plaintext into LWECipher/LWECipherEVec
void LowerMemrefToHEIRPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<MemRefType>())
        {
            int size = -155;
            auto new_t = t.cast<MemRefType>();
            if (new_t.hasStaticShape() && new_t.getShape().size()==1) {
                size = new_t.getShape().front();
                return std::optional<Type>(LWECipherVectorType::get(&getContext(), new_t.getElementType(), size));
            }
            else if (new_t.hasStaticShape() && new_t.getShape().size()==2) {
                int row = new_t.getShape().front();
                int column = new_t.getShape().back();
                return std::optional<Type>(LWECipherMatrixType::get(&getContext(), new_t.getElementType(), row, column));
            }
            else 
                return std::optional<Type>(t);
        }
        else
            return std::optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        // // if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        // if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        // {
        //     assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
        //     auto old_type = vs.front().getType();
        //     // if (old_type.dyn_cast_or_null<MemRefType>())
        //     if (old_type.dyn_cast_or_null<Float32Type>())
        //     {
        //         return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
        //     }
        // }
        if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });
    type_converter.addArgumentMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        // // if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        // if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        // {
        //     assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
        //     auto old_type = vs.front().getType();
        //     // if (old_type.dyn_cast_or_null<MemRefType>())
        //     if (old_type.dyn_cast_or_null<Float32Type>())
        //     {
        //         return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
        //     }
        // }
        if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        // // if (auto bst = t.dyn_cast_or_null<MemRefType>())
        // if (auto bst = t.dyn_cast_or_null<Float32Type>())
        // {
        //     assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
        //     auto old_type = vs.front().getType();
        //     // if (auto ot = old_type.dyn_cast_or_null<LWECipherVectorType>())
        //     if (auto ot = old_type.dyn_cast_or_null<LWECipherType>())
        //         return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        // }
        if (auto bst = t.dyn_cast_or_null<MemRefType>())
        {
            // Only support float type in C Program
            if (bst.getElementType().dyn_cast_or_null<Float32Type>())
                return std::optional<Value>(std::nullopt);
            
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherVectorType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
            else if (auto ot = old_type.dyn_cast_or_null<LWECipherMatrixType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        return std::optional<Value>(std::nullopt);
    });
    
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithDialect, MemRefDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    // target.addIllegalOp<AffineForOp>();

    IRRewriter rewriter(&getContext());
    
    // MemRefToHEIR
    target.addIllegalOp<affine::AffineLoadOp>();
    target.addIllegalOp<affine::AffineStoreOp>();
    target.addIllegalOp<LoadOp>();
    // target.addIllegalOp<FHEVectorLoadOp>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
            AffineLoadPattern, AffineStorePattern, MemRefLoadPattern>(type_converter, patterns.getContext()); 
            // ArithmeticNegPattern>(type_converter, patterns.getContext());
            // ArithmeticNegPattern>(patterns.getContext());
    
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
}