import os
import shutil
from fhecomplr.value import Imageplain, Cipher
from fhecomplr.py2mlir import MLIRGenerator
from fhecomplr.cpp2cc import OpenPEGASUSGenerator
import subprocess
import inspect
import ast
import numpy as np
import configparser
from typing import Callable, Any

class Encryptor:
    def __init__(self):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')

    def read(self, filepath: str):
        from PIL import Image as PILImage
        pil_img = PILImage.open(filepath).convert('L')
        width, height = pil_img.size
        data = list(map(float, list(pil_img.getdata())))
        self.width = width
        self.height = height
        return Imageplain(data, width, height)
    
    def compileandrun(self, exe_path: str):
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

    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin'):
        generator = OpenPEGASUSGenerator()
        cc_path = os.path.join(self.openpegasus_path, 'fhetranexamples/cc_encrypt.cc')
        cmake_file_path = os.path.join(self.openpegasus_path, 'fhetranexamples/CMakeLists.txt')
        cmake_content = f"""
add_executable(cc_encrypt_exe cc_encrypt.cc)
target_link_libraries(cc_encrypt_exe pegasus)
        """
        with open(cmake_file_path, 'w') as cmake_file:
            cmake_file.write(cmake_content.strip() + "\n")
        generator.cc_encrypt(cc_path, image, output_path)
        self.compileandrun(os.path.join(self.openpegasus_path, f'build/bin/cc_encrypt_exe'))
        print("Encrypted image saved to: ", output_path)
        return Cipher(output_path, image.width * image.height)
    

class Decryptor:
    def __init__(self, encryptor: Encryptor):
        self.width = encryptor.width
        self.height = encryptor.height
        self.openpegasus_path = encryptor.openpegasus_path

    def decrypt(self, cipher_path: str):
        generator = OpenPEGASUSGenerator()
        cc_path = os.path.join(self.openpegasus_path, 'fhetranexamples/cc_decrypt.cc')
        cmake_file_path = os.path.join(self.openpegasus_path, 'fhetranexamples/CMakeLists.txt')
        cmake_content = f"""
add_executable(cc_decrypt_exe cc_decrypt.cc)
target_link_libraries(cc_decrypt_exe pegasus)
        """
        with open(cmake_file_path, 'w') as cmake_file:
            cmake_file.write(cmake_content.strip() + "\n")
        generator.cc_decrypt(cc_path, cipher_path, os.path.join(self.openpegasus_path, 'decrypted.txt'), self.width, self.height)
        self.compileandrun(os.path.join(self.openpegasus_path, f'build/bin/cc_decrypt_exe'))
        output_txt = np.loadtxt(os.path.join(self.openpegasus_path, 'decrypted.txt'))
        height, width = output_txt.shape
        output_image = Imageplain(output_txt.flatten(), height, width)
        print("Cipher decrypted")
        return output_image
    
    def compileandrun(self, exe_path: str):
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


class Cryptor:
    def __init__(self):
        self.encryptor = Encryptor()
        # self.decryptor = Decryptor(self.encryptor)

    def read(self, filepath: str):
        image = self.encryptor.read(filepath)
        self.decryptor = Decryptor(self.encryptor)
        return image
    
    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin'):
        cipher = self.encryptor.encrypt(image, output_path)
        return cipher
    
    def decrypt(self, cipher: Cipher):
        return self.decryptor.decrypt(cipher.path())