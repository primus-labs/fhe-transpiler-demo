# fhe-transpiler-demo

## Configure Front-End

Run the following command before running ```py2mlir.py``` and ```py2mlir-affine```.

### Manual installation

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 8e0daabe97cf5e73402bcb4c3e54b3583199ba8f
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="mlir"    \
-DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"          \
-DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) \
-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DLLVM_ENABLE_RTTI=ON
cd build
ninja install
cd ../..
```

### Automatic installation

```bash
./setup_dependencies.sh
```

## Run py2mlir.py or py2mlir-affine.py
In order to adapt to [HEIR](https://github.com/heir-compiler/HEIR),
```py2mlir-affine.py``` replaced scf with affine, which can be used
to generate mlir using affine.load and affine.for.

To run the Front-End, first, set ```PYTHONPATH``` environment variables.

```bash
export PYTHONPATH=llvm-project/build/tools/mlir/python_packages/mlir_core:${PYTHONPATH}
```

Next, run ```py2mlir.py``` or ```py2mlir-affine.py```.

```bash
python3 py2mlir.py
```
```bash
python3 py2mlir-affine.py
```

## Configure Middle-End
### Installation
Start with ``fhe-transpiler-demo`` directory.

Clone llvm-15 from Github. Note that LLVM-15 used for fhe-transpiler-demo
Middle-End is not compatiable with LLVM for building Front-End.
```bash
cd ..
git clone -b release/15.x https://github.com/llvm/llvm-project.git
cd llvm-project
```

Build LLVM/MLIR.
```bash
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON \
-DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_INSTALL_UTILS=ON
ninja -j N
```

Build fhe-transpiler-demo.
```sh
cd ../../fhe-transpiler-demo
mkdir build && cd build
cmake .. -DMLIR_DIR=/home/llvm-project/build/lib/cmake/mlir
cmake --build . --target all
```

### Using Demo
In Middle-End, HEIR uses `heir-opt` CLI to transform the input
MLIR program into programs with homomorphic operators 
reprsented in `emitc` dialect. There are parameters for 
`heir-opt`:

+ **--affine-loop-unroll="unroll-full unroll-num-reps=4"**: 
Add this parameter to unroll all the `for` loop in the 
input program.

+ **--arith-emitc**:
Can be replaced by **--arith2heir --canonicalize --memref2heir --canonicalize
--func2heir --canonicalize --nary --canonicalize --batching --canonicalize
  --lwe2rlwe --canonicalize --cse --slot2coeff --canonicalize --heir2emitc --canonicalize**

Next, can use `emitc-translate` to transform the MLIR file
into a C++ file:
```bash
tools/emitc-translate $fileName$.mlir --mlir-to-cpp >> $fileName$.cpp
```

Benchmarks:
```bash
tools/heir-opt ../benchmarks/boxblur/boxblur.mlir /
  --affine-loop-unroll="unroll-full unroll-num-reps=4" /
  --arith-emitc >> ../benchmarks/boxblur/boxblur_emitc.mlir

tools/emitc-translate ../benchmarks/boxblur/boxblur_emitc.mlir /
  --mlir-to-cpp >> ../benchmarks/boxblur/boxblur.cpp
```
```bash
tools/heir-opt ../benchmarks/robertscross/robertscross.mlir /
  --affine-loop-unroll="unroll-full unroll-num-reps=4" /
  --arith-emitc >> ../benchmarks/robertscross/robertscross_emitc.mlir

tools/emitc-translate ../benchmarks/robertscross/robertscross_emitc.mlir /
  --mlir-to-cpp >> ../benchmarks/robertscross/robertscross.cpp
```
