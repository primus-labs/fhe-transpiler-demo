#ifndef HEIR_PASSES_IF2HEIR_REPLACEIFWITHLUT_H_
#define HEIR_PASSES_IF2HEIR_REPLACEIFWITHLUT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct ReplaceIfWithLutPass : public mlir::PassWrapper<ReplaceIfWithLutPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "if2heir";
    }
};

#endif // HEIR_PASSES_IF2HEIR_REPLACEIFWITHLUT_H_
