#!/bin/sh

export PYTHONPATH=~/mylibs/llvm19/python_packages/mlir_core:${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mylibs/lib

python user-server-test/compile.py
