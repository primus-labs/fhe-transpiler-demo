/*
Authors: HECO
Modified by Zian Zhao
Copyright:
Copyright (c) 2020 ETH Zurich.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "heir/Passes/heir2emitc/LowerHEIRToEmitC.h"
#include "heir/IR/FHE/HEIRDialect.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include <string>

using namespace mlir;
using namespace heir;



void LowerHEIRToEmitCPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<mlir::emitc::EmitCDialect>();
    registry.insert<func::FuncDialect>();
}

// Convert LWE/RLWE addition/substraction/multiplication operations to emic::CallOp
// for further transforming to C++ functions
template <typename OpType>
class EmitCBasicPattern final : public OpConversionPattern<OpType>
{
protected:
    using OpConversionPattern<OpType>::typeConverter;

public:
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // build a series of calls to our custom heal operator wrapper
        std::string op_str = "";
        if (std::is_same<OpType, LWEAddOp>())
            op_str = "Add";
        else if (std::is_same<OpType, LWESubOp>())
            op_str = "Sub";
        else if (std::is_same<OpType, LWEMulOp>())
            op_str = "Mul";
        else if (std::is_same<OpType, RLWEMulOp>())
            op_str = "Mul";
        else
            return failure();
        
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange(dstType), llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(),
            materialized_operands);
        
        return success();
    }
};

class EmitCRotatePattern final : public OpConversionPattern<FHERotateOp>
{
public:
    using OpConversionPattern<FHERotateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHERotateOp op, typename FHERotateOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands if necessary
        // llvm::SmallVector<Value> materialized_operands;
        Value o = op.getOperand();
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            // materialized_operands.push_back(new_operand);
            o = new_operand;
        }
        // else
        // {
        //     materialized_operands.push_back(o);
        // }
        auto aa = ArrayAttr::get(
            getContext(), { IntegerAttr::get(
                                IndexType::get(getContext()),
                                0), // means "first operand"
                            rewriter.getSI32IntegerAttr(op.getI()) });

        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, dstType, llvm::StringRef("RotateLeft"), aa, ArrayAttr(), o);
        return success();
    }
};


// Pattern for transform LWEMulOp to emitc:CallOpaqueOp
class EmitCMulPattern final : public OpConversionPattern<LWEMulOp>
{
protected:
    using OpConversionPattern<LWEMulOp>::typeConverter;

public:
    using OpConversionPattern<LWEMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        LWEMulOp op, typename LWEMulOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // build a series of calls to our custom halo operator wrapper
        std::string op_str = "lwe_multiply";
        
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange(dstType), llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(),
            materialized_operands);
        
        return success();
    }
};

// Transform to emitc::ConstantOp to initialize a new plaintext
class EmitCConstantPattern final : public OpConversionPattern<arith::ConstantOp>
{
protected:
    using OpConversionPattern<arith::ConstantOp>::typeConverter;

public:
    using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        arith::ConstantOp op, typename arith::ConstantOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();
        
        auto value = op.getValue();

        rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
            op, TypeRange(dstType), value);
        
        return success();
    }
};



// Transform FHEEncodeOp to emitc to encode a raw data (double) to FHE plaintext
class EmitCEncodePattern final : public OpConversionPattern<FHEEncodeOp>
{
protected:
    using OpConversionPattern<FHEEncodeOp>::typeConverter;

public:
    using OpConversionPattern<FHEEncodeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEEncodeOp op, typename FHEEncodeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        auto input = op.getMessage();
        double msg = input.convertToDouble();
        FloatAttr mat_input = FloatAttr::get(Float64Type::get(rewriter.getContext()), msg);

        auto constDstType = typeConverter->convertType(Float64Type::get(rewriter.getContext()));
        auto constOp = rewriter.create<emitc::ConstantOp>(op.getLoc(), constDstType, mat_input);


        llvm::SmallVector<Value> materialized_operands;
        materialized_operands.push_back(constOp.getResult());

        auto oldType = op.getType();
        
        emitc::OpaqueAttr const_value;
        if (auto t = oldType.dyn_cast_or_null<PlainType>())
            const_value = emitc::OpaqueAttr::get(OpConversionPattern::getContext(), 
                            llvm::StringRef("InterPlain(MemoryManager::GetPool())"));
        
        auto plainConstOp = rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, dstType, const_value);
        materialized_operands.push_back(plainConstOp.getResult());
        
        std::string op_str = "encode_sisd";

        rewriter.create<emitc::CallOpaqueOp>(plainConstOp.getLoc(), TypeRange(), 
                llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(), materialized_operands);

        return success();
    }
};

// Transform to generate C++ function call of a TFHE LUT
template <typename OpType>
class EmitCLUTPattern final : public OpConversionPattern<OpType>
{
protected:
    using OpConversionPattern<OpType>::typeConverter;

public:
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        auto input = op.getOperand();

        llvm::SmallVector<Value> materialized_operands;
        auto operandDstType = typeConverter->convertType(input.getType());
        if (!operandDstType)
            return failure();
        if (input.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, input);
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(input);
        }

        std::string op_str = "NAN";
        
        if (std::is_same<OpType, FHELUTForAddOp>()) {
            op_str = "lut_foradd";
        } else if (std::is_same<OpType, FHELUTForSubOp>()) {
            op_str = "lut_forsub";
        } else if (std::is_same<OpType, FHELUTForGTOp>()) {
            op_str = "lut_gtz";
        } else if (std::is_same<OpType, FHELUTForLTOp>()) {
            op_str = "lut_lsz";
        } else if (std::is_same<OpType, FHELUTOp>()) {
            op_str = "lut";
        }
        
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, TypeRange(dstType), 
            llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(), materialized_operands);

        return success();
    }
};

// Transform FHEMaterializeOp to emitc::CallOpaqueOp in case we have some problems
// in the previous passes
class EmitCMaterializePattern final : public OpConversionPattern<FHEMaterializeOp>
{
public:
    using OpConversionPattern<FHEMaterializeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEMaterializeOp op, typename FHEMaterializeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        auto input = op.getOperand();

        llvm::SmallVector<Value> materialized_operands;
        auto operandDstType = typeConverter->convertType(input.getType());
        if (!operandDstType)
            return failure();
        if (input.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, input);
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(input);
        }

        std::string op_str = "materialize";

        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, TypeRange(dstType), 
            llvm::StringRef(op_str), ArrayAttr(), ArrayAttr(), materialized_operands);

        return success();
    }
};

// Transform the ExtractOp into C++ function calls
class EmitCExtractPattern final : public OpConversionPattern<FHEExtractfinalOp>
{
public:
    using OpConversionPattern<FHEExtractfinalOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEExtractfinalOp op, typename FHEExtractfinalOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();
        
        auto input_vector = op.getOperand();
        auto operandDstType = typeConverter->convertType(input_vector.getType());
        if (!operandDstType)
            return failure();
        Value new_operand;
        if (input_vector.getType() != operandDstType)
        {
            new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, input_vector);
            if (auto specificOp = input_vector.getDefiningOp<FHEMaterializeOp>()) {
                if (auto specificOp2 = specificOp.getOperand().getDefiningOp<FHEMaterializeOp>()){
                    new_operand = specificOp2.getOperand();
                }
            }
            
        }
        else 
            new_operand = input_vector;
        // For Matrix, two-dimentional data
        if (auto t = input_vector.getType().dyn_cast_or_null<LWECipherMatrixType>()) {
            auto rowAttr = op.getRowAttr().cast<IntegerAttr>();
            auto colAttr = op.getColAttr().cast<IntegerAttr>();

            auto aa = ArrayAttr::get(
            getContext(), {
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  0), // means "first operand"
                              rewriter.getSI32IntegerAttr(colAttr.getInt()),
                              rewriter.getSI32IntegerAttr(rowAttr.getInt())
                          });

            rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, dstType, llvm::StringRef("load"),
                                            aa, ArrayAttr(), new_operand);
        }
        // For Vector, one-dimentional data
        else {
            auto colAttr = op.getColAttr().cast<IntegerAttr>();

            auto aa = ArrayAttr::get(
            getContext(), {
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  0), // means "first operand"
                              rewriter.getSI32IntegerAttr(colAttr.getInt())
                          });
            rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, dstType, llvm::StringRef("load"),
                                            aa, ArrayAttr(), new_operand);
        }
        
        return success();
    }
};

// Transform the VectorLoadOp into C++ function calls
class EmitCVectorLoadPattern final : public OpConversionPattern<FHEVectorLoadfinalOp>
{
public:
    using OpConversionPattern<FHEVectorLoadfinalOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEVectorLoadfinalOp op, typename FHEVectorLoadfinalOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();
        
        Value matrix = op.getMemref();
        auto operandDstType = typeConverter->convertType(matrix.getType());
        if (!operandDstType)
            return failure();
        Value new_matrix;
        if (matrix.getType() != operandDstType)
            new_matrix = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, matrix);
        else 
            new_matrix = matrix;
        auto indexAttr = op.getIndexAttr().cast<IntegerAttr>();

        auto aa = ArrayAttr::get(
            getContext(), {
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  0),
                              rewriter.getSI32IntegerAttr(indexAttr.getInt())
                          });
        
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, dstType, llvm::StringRef("vector_load"),
                                            aa, ArrayAttr(), new_matrix);

        return success();
    }
};

class EmitCCompareLutPattern final : public OpConversionPattern<HEIRLutOp>
{
public:
    using OpConversionPattern<HEIRLutOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        HEIRLutOp op, typename HEIRLutOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();
        Value input = op.getInput();
        auto inputDstType = typeConverter->convertType(input.getType());
        if (!inputDstType)
            return failure();

        Value new_input;
        if (input.getType() != inputDstType)
            new_input = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), inputDstType, input);
        else
            new_input = input;
        auto edge2Attr = op.getEdge2Attr();
        auto edge1Attr = op.getEdge1Attr();
        auto thresholdAttr = op.getThresholdAttr();
        auto args = ArrayAttr::get(
            getContext(),
            {
                IntegerAttr::get(IndexType::get(getContext()), 0),
                edge2Attr,
                edge1Attr,
                thresholdAttr
            });

        // 替换为 emitc::CallOpaqueOp
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op,
            TypeRange{dstType},
            llvm::StringRef("comparelut"),
            ArrayAttr(), 
            args,
            ArrayRef<Value>{new_input});
        return success();
    }
};


class EmitCCopyPattern final : public OpConversionPattern<HEIRCopyOp>
{
public:
    using OpConversionPattern<HEIRCopyOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(
        HEIRCopyOp op, typename HEIRCopyOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);
        Type sourceType = typeConverter->convertType(op.getSource().getType());
        Type targetType = typeConverter->convertType(op.getTarget().getType());
        auto new_source = typeConverter->materializeTargetConversion(rewriter, op.getSource().getLoc(),
                                                                            sourceType, op.getSource());
        auto new_target = typeConverter->materializeTargetConversion(rewriter, op.getTarget().getLoc(),
                                                                            targetType, op.getTarget());
        llvm::SmallVector<Value> materialized_operands;
        materialized_operands.push_back(new_source);
        materialized_operands.push_back(new_target);
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, TypeRange(), llvm::StringRef("copy"),
                                            ArrayAttr(), ArrayAttr(), materialized_operands);
        return success();
    }
};


// Transform the InsertOp into C++ function calls
class EmitCInsertPattern final : public OpConversionPattern<FHEInsertfinalOp>
{
public:
    using OpConversionPattern<FHEInsertfinalOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEInsertfinalOp op, typename FHEInsertfinalOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {        
        rewriter.setInsertionPoint(op);
        Type dstVecType = typeConverter->convertType(op.getMemref().getType());
        Type dstValType = typeConverter->convertType(op.getValue().getType());
        std::string vectorTypeString;
        llvm::StringRef addrTypeValue;
        if (auto t = dstVecType.dyn_cast_or_null<emitc::OpaqueType>()) {
            vectorTypeString = dstVecType.cast<emitc::OpaqueType>().getValue().str();
            addrTypeValue = llvm::StringRef(vectorTypeString.append("&"));
        }
        else
            return failure();
        auto addrType = emitc::OpaqueType::get(OpConversionPattern::getContext(), addrTypeValue);
        auto test_addrTypeValue = addrType.getValue().str();


        if (auto t = op.getMemref().getType().dyn_cast_or_null<LWECipherMatrixType>()) {
            auto rowAttr = op.getRowAttr().cast<IntegerAttr>();
            auto colAttr = op.getColAttr().cast<IntegerAttr>();
            auto new_vector = typeConverter->materializeTargetConversion(rewriter, op.getMemref().getLoc(),
                                                                            dstVecType, op.getMemref());

            auto new_valueToStore = typeConverter->materializeTargetConversion(rewriter, op.getValue().getLoc(),
                                                                                dstValType, op.getValue());
            llvm::SmallVector<Value> materialized_operands;
            materialized_operands.push_back(new_valueToStore);
            materialized_operands.push_back(new_vector);
        // For Matrix
            auto aa = ArrayAttr::get(
            getContext(), {
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  0), // means "first operand"
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  1), // means "second operand"
                              rewriter.getSI32IntegerAttr(colAttr.getInt()),
                              rewriter.getSI32IntegerAttr(rowAttr.getInt())
                          });

            rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, TypeRange(), llvm::StringRef("store"),
                                            aa, ArrayAttr(), materialized_operands);
        }
        // For Vector
        else {
            // llvm::SmallVector<Value> materialized_operands1;
            auto aaa = op.getMemref().getDefiningOp<FHEMaterializeOp>().getOperand();
            auto aaa1 = aaa.getDefiningOp<FHEMaterializeOp>().getOperand();
            auto new_valueToStore = typeConverter->materializeTargetConversion(rewriter, op.getValue().getLoc(),
                                                                                dstValType, op.getValue());
            llvm::SmallVector<Value> materialized_operands;
            materialized_operands.push_back(aaa1);
            materialized_operands.push_back(new_valueToStore);
            auto colAttr = op.getColAttr().cast<IntegerAttr>();

            auto aa = ArrayAttr::get(
            getContext(), {
                              IntegerAttr::get(
                                  IndexType::get(getContext()),
                                  0), // means "first operand"
                            //   IntegerAttr::get(
                            //       IndexType::get(getContext()),
                            //       1), // means "second operand"
                              rewriter.getSI32IntegerAttr(colAttr.getInt())
                          });
            rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(op, TypeRange(), llvm::StringRef("insert"),
                                            aa, ArrayAttr(), materialized_operands);
        }

        return success();

    }
};

// Transform FHEDefineOp to a C++ function call (Define a new ciphertext)
class EmitCDefinePattern final : public OpConversionPattern<FHEDefineOp>
{
public:
    using OpConversionPattern<FHEDefineOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEDefineOp op, typename FHEDefineOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Replace FHEDefineOp with emitc::ConstantOp
        rewriter.setInsertionPoint(op);

        auto oldType = op.getType();
        auto dstType = typeConverter->convertType(oldType);
        if (!dstType) 
            return failure();
        
        emitc::OpaqueAttr const_value;
        if (auto t = oldType.dyn_cast_or_null<LWECipherType>())
            const_value = emitc::OpaqueAttr::get(OpConversionPattern::getContext(), 
                            llvm::StringRef("LWECipher(MemoryManager::GetPool())"));
        else if(auto t = oldType.dyn_cast_or_null<RLWECipherType>())
            const_value = emitc::OpaqueAttr::get(OpConversionPattern::getContext(),
                            llvm::StringRef("RLWECipher(MemoryManager::GetPool())"));
        else if(auto t = oldType.dyn_cast_or_null<LWECipherVectorType>()) {
            int vectorSize = oldType.cast<LWECipherVectorType>().getSize();
            const_value = emitc::OpaqueAttr::get(OpConversionPattern::getContext(), 
                            llvm::StringRef("std::vector<LWECipher>(" + std::to_string(vectorSize) + ")"));
        }
        else if(auto t = oldType.dyn_cast_or_null<LWECipherMatrixType>()) {
            int rowSize = oldType.cast<LWECipherMatrixType>().getRow();
            int colSize = oldType.cast<LWECipherMatrixType>().getColumn();
            const_value = emitc::OpaqueAttr::get(OpConversionPattern::getContext(), 
                            llvm::StringRef("std::vector<std::vector<LWECipher>>(" + 
                            std::to_string(rowSize) + ", std::vector<LWECipher>(" + std::to_string(colSize) + "))"));
        }
        else 
            return failure();
        
        rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, dstType, const_value);

        return success(); 
    }    
};

// Transform the FHEFuncCallOp into C++ function calls
// For now, we transform all function calls into TFHE LUT operations
class EmitCFHECallPattern final : public OpConversionPattern<FHEFuncCallOp>
{
public:
    using OpConversionPattern<FHEFuncCallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEFuncCallOp op, typename FHEFuncCallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto resultType = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultType)
            return failure();
        
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : adaptor.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }
        auto func_name = op.getCallee();

        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);
    
        return success();
    }
};

// Transform the function call into C++ function format
class EmitCCallPattern final : public OpConversionPattern<func::CallOp>
{
public:
    using OpConversionPattern<func::CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::CallOp op, typename func::CallOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto resultType = getTypeConverter()->convertType(op.getResult(0).getType());
        if (!resultType)
            return failure();
        
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : adaptor.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                // assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }
        auto func_name = op.getCallee();

        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange(resultType), func_name, ArrayAttr(), ArrayAttr(), materialized_operands);

        return success();
    }   
};

/// This is basically just boiler-plate code,
/// nothing here actually depends on the current dialect thats being converted.
class EmitCFunctionPattern final : public OpConversionPattern<func::FuncOp>
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

/// More boiler-plate code that isn't dialect specific
class EmitCReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        if (op->getNumOperands() != 1)
        {
            emitError(op->getLoc(), "Only single value returns support for now.");
            return failure();
        }
        auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
        if (!dstType)
            return failure();
        if (auto bst = dstType.dyn_cast_or_null<emitc::OpaqueType>())
        {
            rewriter.setInsertionPoint(op);
            auto materialized =
                typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands());
            // build a new return op
            rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        } // else do nothing
        
        return success();
    }
};

// Transform the MLIR written in HEIR dialect into MLIR wiritten in emitc dialect
// Therefore, we can convert the output MLIR into C++ code
void LowerHEIRToEmitCPass::runOnOperation()
{
    auto type_converter = TypeConverter();
    // Type conversion, convert HEIR types into emitc C++ types
    type_converter.addConversion([&](Type t) {
        if (t.isa<LWECipherType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "LWECipher"));
        else if (t.isa<RLWECipherType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "Ctx"));
        else if (t.isa<LWECipherVectorType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "std::vector<LWECipher>"));
        else if (t.isa<LWECipherMatrixType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "std::vector<std::vector<LWECipher>>"));
        else if (t.isa<IndexType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "int"));
        else if (t.isa<PlainType>())
            return std::optional<Type>(emitc::OpaqueType::get(&getContext(), "InterPlain"));
        else
            return std::optional<Type>(t);
    });

    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<LWECipherType>())
            {
                if (ot.getValue().str() == "LWECipher")
                {
                    // llvm::errs() << "\n" << llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs)) << "\n";
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
                }
            }
            else if (old_type.dyn_cast_or_null<RLWECipherType>())
            {
                if (ot.getValue().str() == "Ctx")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<LWECipherVectorType>())
            {
                if (ot.getValue().str() == "std::vector<LWECipher>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<LWECipherMatrixType>())
            {
                if (ot.getValue().str() == "std::vector<std::vector<LWECipher>>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<IndexType>())
            {
                if (ot.getValue().str() == "int")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<PlainType>())
            {
                if (ot.getValue().str() == "InterPlain")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });

    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<emitc::OpaqueType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<LWECipherType>())
            {
                if (ot.getValue().str() == "LWECipher"){
                    // llvm::errs() << "\n1\n" << std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs)) << "\n";
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
                }
            }
            else if (old_type.dyn_cast_or_null<RLWECipherType>())
            {
                if (ot.getValue().str() == "Ctx")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<LWECipherVectorType>())
            {
                if (ot.getValue().str() == "std::vector<LWECipher>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<LWECipherMatrixType>())
            {
                if (ot.getValue().str() == "std::vector<std::vector<LWECipher>>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<IndexType>())
            {
                if (ot.getValue().str() == "int")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
            else if (old_type.dyn_cast_or_null<PlainType>())
            {
                if (ot.getValue().str() == "InterPlain")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt);
    });

    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "LWECipher"){
                    // llvm::errs() << "\n2\n" << std::optional<Value>(vs.front())<<"\n"<<bst<< "\n"<<loc << "\n";
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
                    // return std::optional<Value>(vs.front());
                }
        }
        else if (auto bst = t.dyn_cast_or_null<RLWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "Ctx")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "std::vector<LWECipher>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<LWECipherMatrixType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "std::vector<std::vector<LWECipher>>")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<IndexType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "int")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<PlainType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<emitc::OpaqueType>())
                if (ot.getValue().str() == "InterPlain")
                    return std::optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        return std::optional<Value>(std::nullopt);
    });
    ConversionTarget target(getContext());
    target.addIllegalDialect<HEIRDialect>();
    target.addIllegalOp<arith::ConstantOp>();
    target.addIllegalOp<func::CallOp>();
    target.addLegalOp<FHEMaterializeOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();
    // target.addLegalOp<FHERotateOp>(); 
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
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        EmitCMulPattern, EmitCBasicPattern<LWEAddOp>, EmitCBasicPattern<LWESubOp>, EmitCBasicPattern<RLWEMulOp>,EmitCConstantPattern, 
        EmitCRotatePattern, EmitCDefinePattern, EmitCFHECallPattern, EmitCCallPattern, EmitCFunctionPattern, EmitCConstantPattern,
        EmitCEncodePattern, EmitCLUTPattern<FHELUTForAddOp>, EmitCLUTPattern<FHELUTForSubOp>, EmitCLUTPattern<FHELUTForGTOp>,
        EmitCLUTPattern<FHELUTForLTOp>, EmitCLUTPattern<FHELUTOp>, EmitCBasicPattern<RLWEMulOp>, 
        EmitCReturnPattern, EmitCExtractPattern, EmitCVectorLoadPattern>(type_converter, patterns.getContext());
    patterns.add<EmitCInsertPattern, EmitCCompareLutPattern, EmitCCopyPattern>(type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

