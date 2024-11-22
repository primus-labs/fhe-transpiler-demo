get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(  ${dialect_libs} )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIREmitCDialect
        MLIROptLib
        HEIRHEIRDialect
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRParser
        MLIRPass
        MLIRInferTypeOpInterface
        MLIRControlFlowInterfaces
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms  
        HEIRArithToHEIR
        HEIRUnrollLoop
        HEIRMemrefToHEIR
        HEIRFuncToHEIR
        HEIRHEIRToEmitC
        HEIRCombineExtract
        HEIRBatching
        HEIRNary
        HEIRLWEToRLWE
        HEIRReplaceIfWithLut
        )
add_llvm_executable(heir-opt heir-opt.cpp)

llvm_update_compile_flags(heir-opt)
target_link_libraries(heir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(heir-opt)
