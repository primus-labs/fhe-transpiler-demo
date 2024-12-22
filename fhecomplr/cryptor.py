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

class Encryptor:
    def __init__(self, rotate_steps = None):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        if rotate_steps is None:
            self.scheme = 'pegasus'
            self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')
        else:
            self.scheme = 'openfhe'
            self.rotate_steps = rotate_steps
            self.openfhe_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend')

    def read(self, filepath: str):
        from PIL import Image as PILImage
        pil_img = PILImage.open(filepath).convert('L')
        width, height = pil_img.size
        data = list(map(float, list(pil_img.getdata())))
        self.width = width
        self.height = height
        return Imageplain(data, width, height)
    
    def compilerunpegasus(self, exe_path: str):
        build_path = os.path.join(self.openpegasus_path, 'build')
        pegasus_compile_command = [
            'make',
            '-C',
            build_path,
            '-j'
        ]
        subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_command = [exe_path]
        subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def compilerunopenfhe(self, exe_path: str):
        build_path = os.path.join(self.openfhe_path, 'build')
        pegasus_compile_command = [
            'make',
            '-C',
            build_path,
            '-j'
        ]
        export_command = [
            'bash',
            '-c',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mylibs/lib'
        ]
        subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_command = [exe_path]
        subprocess.run(export_command)
        subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin'):
        if self.scheme == 'pegasus':
            generator = OpenPEGASUSGenerator()
            cc_path = os.path.join(self.openpegasus_path, 'fhetranexamples/cc_encrypt.cc')
            cmake_file_path = os.path.join(self.openpegasus_path, 'fhetranexamples/CMakeLists.txt')
            cmake_content = ''
            cmake_content += 'add_executable(cc_encrypt_exe cc_encrypt.cc)'
            cmake_content += 'target_link_libraries(cc_encrypt_exe pegasus)'
            with open(cmake_file_path, 'w') as cmake_file:
                cmake_file.write(cmake_content.strip() + "\n")
            generator.cc_encrypt(cc_path, image, output_path)
            self.compilerunpegasus(os.path.join(self.openpegasus_path, f'build/bin/cc_encrypt_exe'))
            return Cipher(output_path, image.width * image.height)
        
        elif self.scheme == 'openfhe':
            generator = OpenFHEGenerator()
            cpp_path = os.path.join(self.openfhe_path, 'encrypt.cpp')
            cmake_file_path = os.path.join(self.openfhe_path, 'CMakeLists.txt')
            cmake_content = 'add_executable(encrypt encrypt.cpp)'
            with open(cmake_file_path, 'r') as cmake_file:
                lines = cmake_file.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith('add_executable'):
                    del lines[i]
                    break
            lines.append(cmake_content)
            with open(cmake_file_path, 'w') as cmake_file:
                cmake_file.writelines(lines)
            generator.encrypt(cpp_path, image, self.rotate_steps, os.path.join(self.openfhe_path, 'build'))
            self.compilerunopenfhe(os.path.join(self.openfhe_path, 'build/encrypt'))
            return Cipher(os.path.join(self.openfhe_path, 'build/ciphertextenc.bin'), image.width * image.height)
            
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")
    

class Decryptor:
    def __init__(self, encryptor: Encryptor):
        self.width = encryptor.width
        self.height = encryptor.height
        if encryptor.scheme == 'pegasus':
            self.scheme = 'pegasus'
            self.openpegasus_path = encryptor.openpegasus_path
        elif encryptor.scheme == 'openfhe':
            self.scheme = 'openfhe'
            self.openfhe_path = encryptor.openfhe_path

    def decrypt(self, cipher_path: str):
        if self.scheme == 'pegasus':
            generator = OpenPEGASUSGenerator()
            cc_path = os.path.join(self.openpegasus_path, 'fhetranexamples/cc_decrypt.cc')
            cmake_file_path = os.path.join(self.openpegasus_path, 'fhetranexamples/CMakeLists.txt')
            cmake_content = ''
            cmake_content += 'add_executable(cc_decrypt_exe cc_decrypt.cc)'
            cmake_content += 'target_link_libraries(cc_decrypt_exe pegasus)'
            with open(cmake_file_path, 'w') as cmake_file:
                cmake_file.write(cmake_content.strip() + "\n")
            generator.cc_decrypt(cc_path, cipher_path, os.path.join(self.openpegasus_path, 'decrypted.txt'), self.width, self.height)
            self.compilerunpegasus(os.path.join(self.openpegasus_path, f'build/bin/cc_decrypt_exe'))
            output_txt = np.loadtxt(os.path.join(self.openpegasus_path, 'decrypted.txt'))
            height, width = output_txt.shape
            output_image = Imageplain(output_txt.flatten(), height, width)
            print("Cipher decrypted")
            return output_image
        
        elif self.scheme == 'openfhe':
            generator = OpenFHEGenerator()
            cpp_path = os.path.join(self.openfhe_path, 'decrypt.cpp')
            cmake_file_path = os.path.join(self.openfhe_path, 'CMakeLists.txt')
            cmake_content = 'add_executable(decrypt decrypt.cpp)'
            with open(cmake_file_path, 'r') as cmake_file:
                lines = cmake_file.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith('add_executable'):
                    del lines[i]
                    break
            lines.append(cmake_content)
            with open(cmake_file_path, 'w') as cmake_file:
                cmake_file.writelines(lines)
            generator.decrypt(cpp_path, os.path.join(self.openfhe_path, 'decrypted.txt'), cipher_path, self.width, self.height, os.path.join(self.openfhe_path, 'build'))
            self.compilerunopenfhe(os.path.join(self.openfhe_path, 'build/decrypt'))
            output_txt = np.loadtxt(os.path.join(self.openfhe_path, 'decrypted.txt'))
            height, width = output_txt.shape
            output_image = Imageplain(output_txt.flatten(), height, width)
            print("Cipher decrypted")
            return output_image
    
    def compilerunpegasus(self, exe_path: str):
        build_path = os.path.join(self.openpegasus_path, 'build')
        pegasus_compile_command = [
            'make',
            '-C',
            build_path,
            '-j'
        ]
        subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_command = [exe_path]
        subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def compilerunopenfhe(self, exe_path: str):
        build_path = os.path.join(self.openfhe_path, 'build')
        pegasus_compile_command = [
            'make',
            '-C',
            build_path,
            '-j'
        ]
        export_command = [
            'bash',
            '-c',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mylibs/lib'
        ]
        subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        run_command = [exe_path]
        subprocess.run(export_command)
        subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Cryptor:
    def __init__(self, scheme: str = 'openfhe', rotate_steps: list = []):
        self.scheme = scheme
        if scheme == 'openfhe':
            if rotate_steps == []:
                rotate_steps = [1]
            self.encryptor = Encryptor(rotate_steps)
        elif scheme == 'pegasus':
            self.encryptor = Encryptor()
        else:
            raise ValueError(f"Invalid scheme: {scheme}")

    def read(self, filepath: str):
        image = self.encryptor.read(filepath)
        self.decryptor = Decryptor(self.encryptor)
        return image
    
    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin'):
        cipher = self.encryptor.encrypt(image, output_path)
        return cipher
    
    def decrypt(self, cipher: Cipher):
        return self.decryptor.decrypt(cipher.path())
    
    def set_scheme(self, scheme: str):
        self.scheme = scheme