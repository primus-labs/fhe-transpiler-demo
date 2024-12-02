# fhe-transpiler-demo

## Installation
First, clone this repo.
```bash
git clone https://github.com/primus-labs/fhe-transpiler-demo.git
```

### Requirements
- git 
- c++ compiler that supports at least C++17 standard
- cmake version >= 3.10
- Matplotlib
- NumPy
- GMP

### Front-End
Starting from the ``fhe-transpiler-demo`` directory and clone submodules.
```bash
cd fhe-transpiler-demo
git submodule update --init --recursive
```

Then build ``LLVM19`` for the front-end.
```bash
cd thirdparty/llvm-project
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

Make ``build`` directory for modified [OpenPEGASUS](https://github.com/ruiyushen/OpenPEGASUS) library.
```bash
cd thirdparty/OpenPEGASUS
mkdir build & cd build
cmake .. -DSEAL_USE_ZLIB=OFF -DSEAL_USE_MSGSL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/mylibs/pegasus
make -j
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





