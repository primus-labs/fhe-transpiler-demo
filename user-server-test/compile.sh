#!/bin/sh

export PYTHONPATH=~/mylibs/llvm19/python_packages/mlir_core:${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mylibs/lib

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR/.."

python user-server-test/compile.py
