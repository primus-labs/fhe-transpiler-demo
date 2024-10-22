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
In order to adapt to [HEIR](https://github.com/heir-compiler/HEIR), ```py2mlir-affine.py``` replaced scf with affine, which can be used to generate mlir using affine.load and affine.for.

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
