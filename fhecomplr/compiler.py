import os
import shutil
from fhecomplr.image import Imageplain
from fhecomplr.py2mlir import MLIRGenerator
from fhecomplr.cpp2cc import OpenPEGASUSGenerator
from mlir import ir
import subprocess
import inspect
import ast
import numpy as np
import configparser

class Compiler:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, 'config.ini')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        config = configparser.ConfigParser()
        config.read(config_file)
        paths = config['paths']
        self.fhe_transpiler_path = paths.get('FHE_TRANSPILER_PATH')
        self.llvm_path = paths.get('LLVM_PATH')
        self.openpegasus_path = paths.get('OPENPEGASUS_PATH')
        self.heir_opt_path = os.path.join(self.fhe_transpiler_path, 'build/bin/heir-opt')
        self.emitc_translate_path = os.path.join(self.fhe_transpiler_path, 'build/bin/emitc-translate')

    def back_to_fhe_transpiler(self):
        back_command = ['cd', self.fhe_transpiler_path]
        subprocess.run(back_command)

    def read(self, filepath):
        from PIL import Image as PILImage
        pil_img = PILImage.open(filepath).convert('L')
        width, height = pil_img.size
        data = list(map(float, list(pil_img.getdata())))
        return Imageplain(data, width, height)
    
    def pythontomlir(self, functionstr, mlir_output_path):
        parsed_ast = ast.parse(functionstr)
        os.makedirs(os.path.dirname(mlir_output_path), exist_ok=True)
        with ir.Context() as ctx:
            with ir.Location.unknown():
                module = ir.Module.create()
                generator = MLIRGenerator(module)
                generator.visit(parsed_ast)
                with open(mlir_output_path, 'w') as mlir_file:
                    mlir_file.write(str(module))

    def mlirtoemitc(self, mlir_path, emitc_output_path):
        heir_opt_command = [
            self.heir_opt_path,
            mlir_path,
            '--affine-loop-unroll=unroll-full unroll-num-reps=4',
            '--arith-emitc'
        ]
        with open(emitc_output_path, 'w') as emitc_mlir_file:
            subprocess.run(heir_opt_command, stdout=emitc_mlir_file, check=True)

    def emitctocpp(self, emitc_path, cpp_output_path):
        emitc_translate_command = [
            self.emitc_translate_path, 
            emitc_path,
            '--mlir-to-cpp'
        ]
        with open(cpp_output_path, 'w') as emitc_cpp_file:
            subprocess.run(emitc_translate_command, stdout=emitc_cpp_file, check=True)

    def movecctopegasus(self, cc_path):
        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"File not found: {cc_path}")
        if not os.path.isdir(self.openpegasus_path):
            raise FileNotFoundError(f"Pegasus path not found: {self.openpegasus_path}")
        filename = os.path.basename(cc_path)
        if not filename.endswith(".cc"):
            raise ValueError(f"Provided file is not a .cc file: {filename}")
        base_name = os.path.splitext(filename)[0]
        examples_path = os.path.join(self.openpegasus_path, "examples")
        os.makedirs(examples_path, exist_ok=True)
        target_path = os.path.join(examples_path, filename)
        shutil.copy(cc_path, target_path)
        cmake_file_path = os.path.join(examples_path, "CMakeLists.txt")
        cmake_content = f"""
add_executable({base_name}_exe {filename})
target_link_libraries({base_name}_exe pegasus)
        """
        with open(cmake_file_path, 'w') as cmake_file:
            cmake_file.write(cmake_content.strip() + "\n")
        return os.path.join(self.openpegasus_path, f'build-release/bin/{base_name}_exe')
    
    def is_directory_empty(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"The path {directory} is not a valid directory")
        with os.scandir(directory) as entries:
            for _ in entries:
                return False
        return True 

    def compile_and_run_pegasus(self, exe_path):
        build_dir = os.path.join(self.openpegasus_path, "build-release/")
        if os.path.exists(build_dir):
            if self.is_directory_empty(build_dir):
                pass
            else:
                delete_command = f'rm -rf {os.path.join(build_dir, "*")}'
                subprocess.run(delete_command, check=True, shell=True)
        else:
            print(f"Path does not exist: {build_dir}")
        pegasus_compile_command1 = [
            'cmake',
            '-B',
            build_dir,
            '-S',
            self.openpegasus_path,
            '-DSEAL_USE_ZLIB=OFF',
            '-DSEAL_USE_MSGSL=OFF',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DCMAKE_INSTALL_PREFIX=~/mylibs'
        ]
        pegasus_compile_command2 = [
            'make',
            '-C',
            build_dir,
            '-j'
        ]
        subprocess.run(pegasus_compile_command1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(pegasus_compile_command2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Build OpenPEGASUS: Done.")
        subprocess.run(exe_path, check=True)

    def compile_plain(self, function, image):
        function_name = function.__name__
        source_code = inspect.getsource(function)
        benchmark_path = os.path.join(self.fhe_transpiler_path, f'benchmarks/{function_name}')

        mlir_output_path = os.path.join(benchmark_path, f'{function_name}.mlir')
        self.pythontomlir(source_code, mlir_output_path)
        print("Python to MLIR: Done.")

        emitc_mlir_output_path = os.path.join(benchmark_path, f'{function_name}_emitc.mlir')
        self.mlirtoemitc(mlir_output_path, emitc_mlir_output_path)
        print("MLIR to EmitC: Done.")

        emitc_cpp_output_path = os.path.join(benchmark_path, f'{function_name}.cpp')
        self.emitctocpp(emitc_mlir_output_path, emitc_cpp_output_path)
        print("EmitC to CPP: Done.")

        cc_output_path = os.path.join(benchmark_path, f'{function_name}.cc')
        output_txt_path = os.path.join(benchmark_path, 'output_image.txt')
        cpptranspiler = OpenPEGASUSGenerator(emitc_cpp_output_path)
        cpptranspiler.cpptocc(cc_output_path, output_txt_path, image)
        print("CPP to CC: Done.")

        exe_path = self.movecctopegasus(cc_output_path)
        self.compile_and_run_pegasus(exe_path)
        print("Compile and run OpenPEGASUS: Done.")

        output_txt = np.loadtxt(output_txt_path)
        height, width = output_txt.shape
        output_image = Imageplain(output_txt.flatten(), height, width)
        delete_command = ['rm', output_txt_path]
        subprocess.run(delete_command)

        return output_image


    def compile_cipher(self, function, image):
        return 0

    def compile(self, function, image):
        if isinstance(image, Imageplain):
            output = self.compile_plain(function, image)
            return output