import os
import shutil
from fhecomplr.value import Imageplain, Cipher
from fhecomplr.py2mlir import MLIRGenerator
from fhecomplr.cpp2cc import OpenPEGASUSGenerator, OpenFHEGenerator
import subprocess
import inspect
import ast
import numpy as np
import configparser
from typing import Callable, Any

class Runner:
    def __init__(self):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        self.llvm_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/llvm-project')
        self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')
        self.heir_opt_path = os.path.join(self.fhe_transpiler_path, 'build/bin/heir-opt')
        self.emitc_translate_path = os.path.join(self.fhe_transpiler_path, 'build/bin/emitc-translate')
        self.output_txt_path = None
        self.exe_path = None
        self.output_cipher_path = None
        self.cipher_path = None
        self.slots = None


    def movecctopegasus(self, cc_path: str):
        if not os.path.isfile(cc_path):
            raise FileNotFoundError(f"File not found: {cc_path}")
        if not os.path.isdir(self.openpegasus_path):
            raise FileNotFoundError(f"Pegasus path not found: {self.openpegasus_path}")
        filename = os.path.basename(cc_path)
        if not filename.endswith(".cc"):
            raise ValueError(f"Provided file is not a .cc file: {filename}")
        base_name = os.path.splitext(filename)[0]
        examples_path = os.path.join(self.openpegasus_path, "fhetranexamples")
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
        return os.path.join(self.openpegasus_path, f'build/bin/{base_name}_exe')
    
    def compile_pegasus_backend(self):
        build_dir = os.path.join(self.openpegasus_path, "build/")
        pegasus_compile_command2 = [
            'make',
            '-C',
            build_dir,
            '-j'
        ]
        subprocess.run(pegasus_compile_command2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def compile_openfhe_backend(self):
        cmake_file_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend/CMakeLists.txt')
        cmake_content = 'add_executable(eval eval.cpp)'
        with open(cmake_file_path, 'r') as cmake_file:
            lines = cmake_file.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('add_executable'):
                del lines[i]
                break
        lines.append(cmake_content)
        with open(cmake_file_path, 'w') as cmake_file:
            cmake_file.writelines(lines)

        build_dir = os.path.join(self.fhe_transpiler_path, "openfhebackend/build/")
        openfhe_compile_command = [
            'make',
            '-C',
            build_dir,
            '-j'
        ]
        subprocess.run(openfhe_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        export_command = [
            'bash',
            '-c',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mylibs/lib'
        ]
        subprocess.run(export_command)
    
    def is_directory_empty(self, directory: str):
        if not os.path.isdir(directory):
            raise ValueError(f"The path {directory} is not a valid directory")
        with os.scandir(directory) as entries:
            for _ in entries:
                return False
        return True 

    def compile_pegasus(self, cpp_path: str):
        cc_output_path = os.path.join(self.fhe_transpiler_path, 'test/runner_temp.cc')
        output_cipher_path = os.path.join(self.fhe_transpiler_path, 'test/evaluated_cipher.bin')
        self.output_cipher_path = output_cipher_path
        cpptranspiler = OpenPEGASUSGenerator(cpp_path)
        cpptranspiler.enc_cpptocc(self.cipher_path, cc_output_path, output_cipher_path, self.slots)
        print("CPP to CC: Done.")

        exe_path = self.movecctopegasus(cc_output_path)
        self.exe_path = exe_path
        self.compile_pegasus_backend()
        print("Compile OpenPEGASUS: Done.")

    def compile_openfhe(self, cpp_path: str):
        cpp_output_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend/eval.cpp')
        output_cipher_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend/build/ciphertexteval.bin')
        build_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend/build')
        self.output_cipher_path = output_cipher_path
        cpptranspiler = OpenFHEGenerator(cpp_path)
        cpptranspiler.cpptoof(self.cipher_path, cpp_output_path, output_cipher_path, build_path)
        print("CPP to OpenFHE: Done.")

        exe_path = os.path.join(build_path, 'eval')
        self.exe_path = exe_path
        self.compile_openfhe_backend()

    def exec(self, cpp_path: str, scheme: str):
        if scheme == 'pegasus':
            self.compile_pegasus(cpp_path)
        elif scheme == 'openfhe':
            self.compile_openfhe(cpp_path)
        else:
            raise ValueError(f"Invalid scheme: {scheme}")
        subprocess.run(self.exe_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print("Cipher evaluated and saved to: ", self.output_cipher_path)
        return Cipher(self.output_cipher_path, self.slots)
    
    def readcipher(self, cipher: Cipher):
        self.cipher_path = cipher.path()
        self.slots = cipher.slot()


class Circuit(Runner):
    def __init__(self, cpp_path: str):
        super().__init__()
        self.cpp_path = cpp_path
        self.scheme = 'openfhe' 

    def run(self, cipher: Cipher):
        self.readcipher(cipher)
        self.exec(self.cpp_path, self.scheme)
        return Cipher(self.output_cipher_path, self.slots)
    
    def set_scheme(self, scheme: str):
        self.scheme = scheme