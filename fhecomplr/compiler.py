import os
from fhecomplr.py2mlir import MLIRGenerator
from fhecomplr.runner import Circuit
from mlir import ir
import subprocess
import inspect
import ast
import re
from typing import Callable, Tuple, List

class Compiler:
    """
    Compiler class that provides methods to convert MLIR to EmitC, EmitC to C++,
    extract rotation steps from a file, and compile a function.
    """
    def __init__(self):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        self.llvm_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/llvm-project')
        self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')
        self.heir_opt_path = os.path.join(self.fhe_transpiler_path, 'build/bin/heir-opt')
        self.emitc_translate_path = os.path.join(self.fhe_transpiler_path, 'build/bin/emitc-translate')
    
    def pythontomlir(self, functionstr: str, mlir_output_path: str):
        """
        Converts Python function to MLIR format using the heir-opt tool.
        
        Args:
            functionstr (str): String representation of the Python function.
            mlir_output_path (str): Path to the output MLIR file.
        """
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
        """
        Converts MLIR file to EmitC format using the heir-opt tool.
        
        Args:
            mlir_path (str): Path to the input MLIR file.
            emitc_output_path (str): Path to the output EmitC file.
        """
        heir_opt_command = [
            self.heir_opt_path,
            mlir_path,
            '--affine-loop-unroll=unroll-full unroll-num-reps=4',
            '--arith-emitc'
        ]
        with open(emitc_output_path, 'w') as emitc_mlir_file:
            subprocess.run(heir_opt_command, stdout=emitc_mlir_file, check=True)

    def emitctocpp(self, emitc_path: str, cpp_output_path: str):
        """
        Converts EmitC file to C++ format using the emitc-translate tool.
        
        Args:
            emitc_path (str): Path to the input EmitC file.
            cpp_output_path (str): Path to the output C++ file.
        """
        emitc_translate_command = [
            self.emitc_translate_path, 
            emitc_path,
            '--mlir-to-cpp'
        ]
        with open(cpp_output_path, 'w') as emitc_cpp_file:
            subprocess.run(emitc_translate_command, stdout=emitc_cpp_file, check=True)

    def extract_rotate_left_steps(self, file_path: str) -> List[int]:
        """
        Extracts the steps for RotateLeft operations from a given file.
        
        Args:
            file_path (str): Path to the file to be processed.
        
        Returns:
            List[int]: A list of integers representing the steps for RotateLeft operations.
        """
        pattern = r'RotateLeft\(\s*[^,]+,\s*(\d+)\s*\)'
        with open(file_path, 'r') as file:
            content = file.read()
        steps = re.findall(pattern, content)
        return [int(step) for step in steps]


    def compile_function(self, function: Callable) -> Tuple[str, List[int]]:
        """
        Compiles a given function.
        
        Args:
            function (Callable): The function to be compiled.

        Returns:
            Tuple[str, List[int]]: A tuple containing the path to the compiled C++ file and a list of rotate steps.
        """
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

        rotate_steps = self.extract_rotate_left_steps(emitc_cpp_output_path)
        return emitc_cpp_output_path, rotate_steps


    def compile(self, function: Callable) -> Tuple[Circuit, List[int]]:
        """
        Compiles a given function and returns a Circuit object.

        Args:
            function (Callable): The function to be compiled.

        Returns:
            Circuit: The compiled circuit.
        """
        cpp_path, rotate_steps = self.compile_function(function)

        return Circuit(cpp_path), rotate_steps
