//===------- heir-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "heir/IR/FHE/HEIRDialect.h"
#include "heir/Passes/arith2heir/LowerArithToHEIR.h"
#include "heir/Passes/memref2heir/LowerMemrefToHEIR.h"
#include "heir/Passes/heir2emitc/LowerHEIRToEmitC.h"
#include "heir/Passes/func2heir/FuncToHEIR.h"
#include "heir/Passes/batching/Batching.h"
#include "heir/Passes/nary/Nary.h"
#include "heir/Passes/slot2coeff/SlotToCoeff.h"
#include "heir/Passes/lwe2rlwe/LWEToRLWE.h"
#include "heir/Passes/unroll/UnrollLoop.h"
#include "heir/Passes/combine/CombineExtract.h
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace heir;

// pipeline for optimizing arithmetic circuits
void arithPipelineBuilder(OpPassManager &manager)
{
    manager.addPass(std::make_unique<LowerArithToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<LowerMemrefToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<FuncToHEIRPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<NaryPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
    manager.addPass(std::make_unique<BatchingPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
    manager.addPass(std::make_unique<LWEToRLWEPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(createCSEPass());
    manager.addPass(std::make_unique<SlotToCoeffPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<CombineExtractPass>());
    manager.addPass(createCanonicalizerPass());
    manager.addPass(std::make_unique<LowerHEIRToEmitCPass>());
    manager.addPass(createCanonicalizerPass());
}

int main(int argc, char **argv)
{
    mlir::MLIRContext context;
    context.enableMultithreading();

    mlir::DialectRegistry registry;
    registry.insert<HEIRDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<AffineDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<emitc::EmitCDialect>();
    context.loadDialect<HEIRDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<AffineDialect>();
    context.loadDialect<tensor::TensorDialect>();
    context.loadDialect<arith::ArithmeticDialect>();
    context.loadDialect<scf::SCFDialect>();
    context.loadDialect<memref::MemRefDialect>();
    context.loadDialect<emitc::EmitCDialect>();

    registerCanonicalizerPass();
    registerAffineLoopUnrollPass();
    registerCSEPass();
    PassRegistration<LowerArithToHEIRPass>();
    PassRegistration<LowerMemrefToHEIRPass>();
    PassRegistration<LowerHEIRToEmitCPass>();
    PassRegistration<UnrollLoopPass>();
    PassRegistration<FuncToHEIRPass>();
    PassRegistration<BatchingPass>();
    PassRegistration<CombineExtractPass>();
    PassRegistration<NaryPass>();
    PassRegistration<SlotToCoeffPass>();
    PassRegistration<LWEToRLWEPass>();

    PassPipelineRegistration<>("arith-emitc", "convert arithmetic circuit to emitc", arithPipelineBuilder);

    return asMainReturnCode(MlirOptMain(argc, argv, "HEIR optimizer driver\n", registry));
}
