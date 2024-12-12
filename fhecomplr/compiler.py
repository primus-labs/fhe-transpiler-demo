import os
import shutil
from fhecomplr.value import Imageplain
from fhecomplr.py2mlir import MLIRGenerator
from fhecomplr.cpp2cc import OpenPEGASUSGenerator
from fhecomplr.runner import Circuit
from mlir import ir
import subprocess
import inspect
import ast
import numpy as np
import configparser
from typing import Callable, Any

class Compiler:
    def __init__(self):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        self.llvm_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/llvm-project')
        self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')
        self.heir_opt_path = os.path.join(self.fhe_transpiler_path, 'build/bin/heir-opt')
        self.emitc_translate_path = os.path.join(self.fhe_transpiler_path, 'build/bin/emitc-translate')
    
    def pythontomlir(self, functionstr: str, mlir_output_path: str):
        parsed_ast = ast.parse(functionstr)
        os.makedirs(os.path.dirname(mlir_output_path), exist_ok=True)
        with ir.Context() as ctx:
            with ir.Location.unknown():
                module = ir.Module.create()
                generator = MLIRGenerator(module)
                generator.visit(parsed_ast)
                with open(mlir_output_path, 'w') as mlir_file:
                    mlir_file.write(str(module))

    def mlirtoemitc(self, mlir_path: str, emitc_output_path: str):
        heir_opt_command = [
            self.heir_opt_path,
            mlir_path,
            '--affine-loop-unroll=unroll-full unroll-num-reps=4',
            '--arith-emitc'
        ]
        with open(emitc_output_path, 'w') as emitc_mlir_file:
            subprocess.run(heir_opt_command, stdout=emitc_mlir_file, check=True)

    def emitctocpp(self, emitc_path: str, cpp_output_path: str):
        emitc_translate_command = [
            self.emitc_translate_path, 
            emitc_path,
            '--mlir-to-cpp'
        ]
        with open(cpp_output_path, 'w') as emitc_cpp_file:
            subprocess.run(emitc_translate_command, stdout=emitc_cpp_file, check=True)


    def compile_function(self, function: Callable):
        function_name = function.__name__
        source_code = inspect.getsource(function)
        benchmark_path = os.path.join(self.fhe_transpiler_path, f'test/{function_name}')
        os.makedirs(benchmark_path, exist_ok=True)

        mlir_output_path = os.path.join(benchmark_path, f'{function_name}.mlir')
        self.pythontomlir(source_code, mlir_output_path)
        print("Python to MLIR: Done.")

        emitc_mlir_output_path = os.path.join(benchmark_path, f'{function_name}_emitc.mlir')
        self.mlirtoemitc(mlir_output_path, emitc_mlir_output_path)
        print("MLIR to EmitC: Done.")

        emitc_cpp_output_path = os.path.join(benchmark_path, f'{function_name}.cpp')
        self.emitctocpp(emitc_mlir_output_path, emitc_cpp_output_path)
        print("EmitC to CPP: Done.")

        return emitc_cpp_output_path


    def compile(self, function: Callable):
        cpp_path = self.compile_function(function)

        return Circuit(cpp_path)
