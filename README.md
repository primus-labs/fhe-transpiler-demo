# fhe-transpiler-demo

## Installation
First, starting from any directory and make a new directory. Here we default to /home/
```bash
mkdir fhetran
cd fhetran
```

### Front-End
Starting from the ``fhetran`` and clone this repo.
```bash
git clone https://github.com/primus-labs/fhe-transpiler-demo
cd fhe-transpiler-demo
```

Then build the first LLVM for the front-end.
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 8e0daabe97cf5e73402bcb4c3e54b3583199ba8f
mkdir build
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="mlir"    \
-DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"          \
-DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=~/mylibs
cd build
ninja install
cd ../../..
```

### Middle-End
Starting from the ``fhetran`` directory.

Clone llvm-15 from Github and build it. Note that LLVM-15 used for fhe-transpiler-demo
Middle-End is not compatiable with LLVM for building Front-End.
```bash
git clone -b release/15.x https://github.com/llvm/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_INSTALL_UTILS=ON -DCMAKE_INSTALL_PREFIX=~/mylibs
ninja -j N
```

Build fhe-transpiler-demo.
```bash
cd ../../fhe-transpiler-demo
mkdir build && cd build
cmake .. -DMLIR_DIR=/home/fhetran/llvm-project/build/lib/cmake/mlir -DCMAKE_INSTALL_PREFIX=~/mylibs
cmake --build . --target all
cd ../../
```

### Back-End
Starting from the ``fhetran`` directory.

Clone and make build directory for [OpenPEGASUS](https://github.com/Alibaba-Gemini-Lab/OpenPEGASUS).
```bash
git clone https://github.com/Alibaba-Gemini-Lab/OpenPEGASUS
mkdir build-release
cd ../fhe-transpiler-demo
```

## Using Demo
First, Set the absolute addresses of each library in ``fhecomplr/config.ini``.

Next, we go to the fhe-transpiler-demo folder and configure PYTHONPATH.
```bash
export PYTHONPATH=llvm-project/build/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
```

Run  ```fhecomplr_test.py``` to get the test results.
```bash
python fhecomplr_test.py
```





