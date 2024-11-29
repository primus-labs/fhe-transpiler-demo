// Author: Zian Zhao
#include "heir/Passes/func2heir/FuncToHEIR.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void FuncToHEIRPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

// Take data type conversion for FHE extract operations and
// Convert FHEExtractOp to FHEExtractfinalOp
class FHEExtractPattern final : public OpConversionPattern<FHEExtractOp>
{
public:
    using OpConversionPattern<FHEExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEExtractOp op, typename FHEExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        
        auto indices_size = op.getI().size();

        // LWECipherVector: one-dimensional data
        if (indices_size == 1) {
            auto cOp = op.getI().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(op.getLoc(),
                        "cannot find the definition of index in heir.extract_init op!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();

            rewriter.replaceOpWithNewOp<FHEExtractfinalOp>(op, dstType, op.getVector(), indexAttr, Attribute());
        }
        // LWECipherMatrix: two-dimensional data
        else if (indices_size == 2) {
            auto row_cOp = op.getI().front().getDefiningOp<arith::ConstantOp>();
            auto col_cOp = op.getI().back().getDefiningOp<arith::ConstantOp>();

            if(!row_cOp || !col_cOp) {
                emitError(op.getLoc(),
                        "cannot find the definition of indices in heir.extract_init op!");
                return failure();
            }
            auto row_indexAttr = row_cOp.getValue().cast<IntegerAttr>();
            auto col_indexAttr = col_cOp.getValue().cast<IntegerAttr>();

            llvm::SmallVector<Attribute> materialized_index;
            materialized_index.push_back(row_indexAttr);
            materialized_index.push_back(col_indexAttr);

            rewriter.replaceOpWithNewOp<FHEExtractfinalOp>(op, dstType, op.getVector(), col_indexAttr, row_indexAttr);
        }

        return success();
    }
};

// Take data type conversion for FHE insert operations and
// Convert FHEInsertOp to FHEInsertfinalOp
class FHEInsertPattern final : public OpConversionPattern<FHEInsertOp>
{
public:
    using OpConversionPattern<FHEInsertOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEInsertOp op, typename FHEInsertOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {   
        auto valueType = getTypeConverter()->convertType(op.getValue().getType());
        if (!valueType)
            return failure();
        auto new_value = typeConverter->materializeTargetConversion(rewriter, op.getValue().getLoc(), 
                                                                    valueType, op.getValue()); 
        // llvm::errs()<<"\n"<<"op.value: "<<op.value()<<"\n";
        auto indices_size = op.getIndex().size();
        
        // LWECipherVector: one-dimensional data
        if (indices_size == 1) {
            auto cOp = op.getIndex().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(op.getLoc(),
                        "cannot find the definition of index in heir.extract_init op!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();
            rewriter.replaceOpWithNewOp<FHEInsertfinalOp>(op, new_value, op.getMemref(), indexAttr, Attribute());
        }
        // LWECipherMatrix: two-dimensional data
        else if (indices_size == 2) {
            auto row_cOp = op.getIndex().front().getDefiningOp<arith::ConstantOp>();
            auto col_cOp = op.getIndex().back().getDefiningOp<arith::ConstantOp>();

            if(!row_cOp || !col_cOp) {
                emitError(op.getLoc(),
                        "cannot find the definition of indices in heir.extract_init op!");
                return failure();
            }
            auto row_indexAttr = row_cOp.getValue().cast<IntegerAttr>();
            auto col_indexAttr = col_cOp.getValue().cast<IntegerAttr>();
            llvm::SmallVector<Attribute> materialized_index;
            materialized_index.push_back(row_indexAttr);
            materialized_index.push_back(col_indexAttr);

            rewriter.replaceOpWithNewOp<FHEInsertfinalOp>(op, new_value, op.getMemref(), col_indexAttr, row_indexAttr);
        }

        return success();
    }
};

// Convert the input/output data type of FHEVectorLoadOp
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

// Convert types of arguments into ciphertext types and transform func::CallOp
// to heir::FHEFuncCallOp
class FuncCallPattern final : public OpConversionPattern<func::CallOp>
{
protected:
    using OpConversionPattern<func::CallOp>::typeConverter;
public:
    using OpConversionPattern<func::CallOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::CallOp op, typename func::CallOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        // Only support one result
        rewriter.setInsertionPoint(op);

        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType) {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                // assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
                materialized_operands.push_back(o);
        }
        auto func_name = op.getCallee();

        if (op.getNumResults() > 1)
            return failure();
        if (op.getNumResults() == 1) {
            auto resultType = getTypeConverter()->convertType(op.getResult(0).getType());
            if (!resultType)
                return failure();
            
            // for fun.call@sgn() in min_index testbench
            if (func_name == "sgn") {
                if (op.getNumOperands() == 1) {
                    rewriter.replaceOpWithNewOp<FHELUTForGTOp>(op, resultType, materialized_operands);
                    return success();
                }
                else 
                    return failure();
            }
        
            rewriter.replaceOpWithNewOp<FHEFuncCallOp>(op, 
                TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);
            
            // // To determine RNS modulus for TFHE Bootstrapping
            // std::string func_name_str = func_name.str();
            // size_t found = func_name_str.find("lut");
            // if (found == std::string::npos)
            //     rewriter.replaceOpWithNewOp<FHEFuncCallOp>(op, 
            //         TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);
            // else {
            //     auto result = op.getResult(0);
            //     while (!result.getUses().empty()) {
            //         // auto resUses = result.getUses();
            //         for (OpOperand &u: result.getUses()) {
            //             Operation *owner = u.getOwner();
            //         }
            //     }
            // }

        } else {
            return failure();
        }
        
        return success();
    }
};

// convert the type of return value in a function block
class ReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    if (op.getNumOperands() != 1)
    {
      emitError(op.getLoc(), "Currently only single value return operations are supported.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op.getOperandTypes().front());
    // llvm::errs()<<"\n"<<op.getOperandTypes().front()<<"\n";
    if (!dstType)
      return failure();


    rewriter.setInsertionPoint(op);
    Value returnCipher;
    if (auto dt = dstType.dyn_cast_or_null<LWECipherVectorType>()){
        returnCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                        op.getLoc(),
                                                                        dstType, op.getOperands());
        // llvm::errs()<<"\n"<<op.getLoc()<<"\n"<<marktype<<"\n"<<returnCipher1<<"\n";
        // returnCipher = typeConverter->materializeTargetConversion(rewriter,
        //                                                                 returnCipher1.getDefiningOp()->getLoc(),
        //                                                                 marktype, returnCipher1);
    }
    else{
        returnCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                        op.getLoc(),
                                                                        dstType, op.getOperands());
    }
    
    // llvm::errs()<<"\n"<<op<<"\n"<<returnCipher<<"\n";
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, returnCipher);
    // llvm::errs()<<"\n"<<op<<"\n";
    return success();
  }
};

// Convert the types of function arguments in a function block
class FunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Compute the new signature of the function.
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();
        if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            return failure();
        auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        rewriter.startOpModification(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = typeConverter->convertType(oldType);
            // llvm::errs()<<"\n"<<oldType<<"\n"<<newType<<"\n";
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        rewriter.finalizeOpModification(op);
        return success();
    }
};

// Transform Function Block and Function Call operations into FHE operations and
// ciphertext data types
void FuncToHEIRPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    // Add type converter to convert plaintext data type to LWECiphertext type
    type_converter.addConversion([&](Type t) {
        if (t.isa<Float32Type>())
            return std::optional<Type>(LWECipherType::get(&getContext(), t));
        else if (t.isa<MemRefType>())
        {
            int size = -155;
            auto new_t = t.cast<MemRefType>();
            if (new_t.hasStaticShape() && new_t.getShape().size()==1) {
                size = new_t.getShape().front();
                return std::optional<Type>(LWECipherVectorType::get(&getContext(), new_t.getElementType(), size));
            }
            else if (new_t.hasStaticShape() && new_t.getShape().size()==2) {
                auto row = new_t.getShape().front();
                auto col = new_t.getShape().back();
                return std::optional<Type>(LWECipherMatrixType::get(&getContext(), new_t.getElementType(), row, col));
            }
            else
                return std::optional<Type>(t);
        }
        else
            return std::optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
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
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
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
        if (auto bst = t.dyn_cast_or_null<Float32Type>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<MemRefType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherVectorType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
            else if (auto ot = old_type.dyn_cast_or_null<LWECipherMatrixType>())
                return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
            else 
                return std::optional<Value>(std::nullopt);
        }
        return std::optional<Value>(std::nullopt);
    });
    
    ConversionTarget target(getContext());
    target.addLegalDialect<affine::AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<FHEExtractOp>();
    target.addIllegalOp<FHEInsertOp>();
    target.addIllegalOp<FHEVectorLoadOp>();
    // We cannot 'remove' FuncOp
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addIllegalOp<func::CallOp>();
    
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });

    target.addDynamicallyLegalOp<FHEExtractfinalOp>(
        [&](Operation *op) {return type_converter.isLegal(op->getResultTypes()); });

    IRRewriter rewriter(&getContext());
    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
            FHEExtractPattern, FHEInsertPattern, FunctionConversionPattern, 
                ReturnPattern, FuncCallPattern, FHEVectorLoadPattern>(type_converter, patterns.getContext()); 
    
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
}
