# fhe-transpiler-demo

## Installation
First, clone this repo.
```bash
git clone https://github.com/primus-labs/fhe-transpiler-demo.git
```

### Front-End
Starting from the ``fhe-transpiler-demo`` directory and clone this repo.
```bash
cd fhe-transpiler-demo/thirdparty
```

Then build ``LLVM19`` for the front-end.
```bash
git clone -b release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  \
-DLLVM_INSTALL_UTILS=ON -DPython3_EXECUTABLE=$(which python3)                          \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=~/mylibs/llvm19
cd build
ninja -j 64
cd ../../../
```

### Middle-End
Starting from the ``fhe-transpiler-demo`` directory.

Build fhe-transpiler-demo.
```bash
mkdir build &&  mkdir test
cd build
cmake .. -DMLIR_DIR=thirdparty/llvm-project/build/lib/cmake/mlir -DCMAKE_INSTALL_PREFIX=~/mylibs/fhetran
cmake --build . --target all -j
cd ../
```

### Back-End
Starting from the ``fhe-transpiler-demo`` directory.

Clone and make build directory for modified [OpenPEGASUS](https://github.com/ruiyushen/OpenPEGASUS) library.
```bash
cd thirdparty
git clone https://github.com/ruiyushen/OpenPEGASUS.git
mkdir build
cd ../../../
```

## Using Demo
Before using demo, go to the fhe-transpiler-demo folder and configure PYTHONPATH.
```bash
export PYTHONPATH=thirdparty/llvm-project/build/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
```

Run  ```fhecomplr_test.py``` to get the test results.
```bash
python fhecomplr_test.py
```





