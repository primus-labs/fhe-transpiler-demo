import os
from fhecomplr.value import Imageplain, Cipher
from fhecomplr.cpp2cc import OpenPEGASUSGenerator, OpenFHEGenerator
import subprocess
import numpy as np
from typing import List

class Encryptor:
    """
    Encryptor class that provides methods to read an image, encrypt it
    using OpenPEGASUS or OpenFHE, and generate an image cipher.
    """
    def __init__(self, rotate_steps: List[int] = None):
        absolute_path = os.path.abspath(__file__)
        library_dir = os.path.dirname(absolute_path)
        self.fhe_transpiler_path = os.path.dirname(library_dir)
        self.exeexist = 0
        if rotate_steps is None:
            self.scheme = 'pegasus'
            self.openpegasus_path = os.path.join(self.fhe_transpiler_path, 'thirdparty/OpenPEGASUS')
        else:
            self.scheme = 'openfhe'
            self.rotate_steps = rotate_steps
            self.openfhe_path = os.path.join(self.fhe_transpiler_path, 'openfhebackend')

    def read(self, filepath: str) -> Imageplain:
        """
        Reads an image from the given file path and returns an Imageplain object.

        Args:
            filepath (str): Path to the image file.

        Returns:
            Imageplain: Imageplain object containing the image data.
        """
        from PIL import Image as PILImage
        pil_img = PILImage.open(filepath).convert('L')
        width, height = pil_img.size
        data = list(map(float, list(pil_img.getdata())))
        self.width = width
        self.height = height
        return Imageplain(data, width, height)
    
    def compilerunpegasus(self, exe_path: str):
        """
        Compiles and runs the OpenPEGASUS encryption code.

        Args:
            exe_path (str): Path to the executable file.
        """
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
        """
        Compiles and runs the OpenFHE encryption code.

        Args:
            exe_path (str): Path to the executable file.
        """
        if self.exeexist == 0:
            build_path = os.path.join(self.openfhe_path, 'build')
            pegasus_compile_command = [
                'make',
                '-C',
                build_path,
                '-j'
            ]
            subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run_command = [exe_path]
            subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            run_command = [exe_path]
            subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin') -> Cipher:
        """
        Encrypts the given image using OpenPEGASUS or OpenFHE.

        Args:
            image (Imageplain): Imageplain object containing the image data.
            output_path (str): Path to save the encrypted cipher.

        Returns:
            Cipher: Cipher object containing the encrypted data.
        """
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
            build_path = os.path.join(self.openfhe_path, 'build')
            if not os.path.exists(build_path):
                os.makedirs(build_path)
            image_data_path = os.path.join(build_path, 'image_data.txt')
            with open(image_data_path, 'w') as img_file:
                for value in image.data:
                    img_file.write(f"{value/255}\n")
            if self.rotate_steps != None:
                rotate_steps_str = '_'.join(map(str, self.rotate_steps))
            else:
                rotate_steps_str = ''
            size_str = f"{image.width}x{image.height}"
            exe_name = f"encrypt_rs{rotate_steps_str}_sz{size_str}"
            exe_path = os.path.join(build_path, exe_name)

            if os.path.exists(exe_path):
                print(f"Executable {exe_name} already exists. Skipping compilation.")
                self.exeexist = 1
                self.compilerunopenfhe(exe_path)

            else:
                print(f"Generating and compiling executable {exe_name}.")
                cpp_code_path = os.path.join(self.openfhe_path, f"{exe_name}.cpp")
                generator.encrypt(
                    output_file_path=cpp_code_path,
                    image=image,
                    rotate_steps=self.rotate_steps,
                    build_path=build_path,
                    image_data_path=image_data_path
                )
                
                cmake_file_path = os.path.join(self.openfhe_path, 'CMakeLists.txt')
                cmake_content = f"add_executable({exe_name} {exe_name}.cpp)\n"
                with open(cmake_file_path, 'r') as cmake_file:
                    lines = cmake_file.readlines()
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith('add_executable'):
                        del lines[i]
                        break
                lines.append(cmake_content)
                with open(cmake_file_path, 'w') as cmake_file:
                    cmake_file.writelines(lines)
                self.compilerunopenfhe(exe_path)
            
            cipher_path = os.path.join(build_path, 'ciphertextenc.bin')
            return Cipher(cipher_path, image.width * image.height)
            
        else:
            raise ValueError(f"Invalid scheme: {self.scheme}")
    

class Decryptor:
    """
    Decryptor class that provides methods to decrypt an encrypted cipher
    using OpenPEGASUS or OpenFHE.
    """
    def __init__(self, encryptor: Encryptor):
        self.width = encryptor.width
        self.height = encryptor.height
        self.exeexist = 0
        if encryptor.scheme == 'pegasus':
            self.scheme = 'pegasus'
            self.openpegasus_path = encryptor.openpegasus_path
        elif encryptor.scheme == 'openfhe':
            self.scheme = 'openfhe'
            self.openfhe_path = encryptor.openfhe_path

    def decrypt(self, cipher_path: str) -> Imageplain:
        """
        Decrypts the given cipher using OpenPEGASUS or OpenFHE.

        Args:
            cipher_path (str): Path to the encrypted cipher file.

        Returns:
            Imageplain: Imageplain object containing the decrypted image data.
        """
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
            build_path = os.path.join(self.openfhe_path, 'build')
            cp_command = [
                'cp',
                '-f',
                cipher_path,
                os.path.join(build_path, 'cipher.bin')
            ]
            subprocess.run(cp_command)
            decrypted_txt_path = os.path.join(build_path, 'decrypted.txt')
            if not os.path.exists(build_path):
                os.makedirs(build_path)
            
            exe_name = f"decrypt_sz{self.width}x{self.height}"
            exe_path = os.path.join(build_path, exe_name)

            if os.path.exists(exe_path):
                print(f"Executable {exe_name} already exists. Skipping compilation.")
                self.exeexist = 1
                self.compilerunopenfhe(exe_path)

            else:
                print(f"Generating and compiling executable {exe_name}.")
                cpp_code_path = os.path.join(self.openfhe_path, f"{exe_name}.cpp")
                generator.decrypt(
                    output_file_path=cpp_code_path,
                    output_txt_path=decrypted_txt_path,
                    cipher_path=os.path.join(build_path, 'cipher.bin'),
                    width=self.width,
                    height=self.height,
                    build_path=build_path
                )
                
                cmake_file_path = os.path.join(self.openfhe_path, 'CMakeLists.txt')
                cmake_content = f"add_executable({exe_name} {exe_name}.cpp)\n"
                with open(cmake_file_path, 'r') as cmake_file:
                    lines = cmake_file.readlines()
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith('add_executable'):
                        del lines[i]
                        break
                lines.append(cmake_content)
                with open(cmake_file_path, 'w') as cmake_file:
                    cmake_file.writelines(lines)
                self.compilerunopenfhe(exe_path)

            output_txt = np.loadtxt(decrypted_txt_path)
            height, width = output_txt.shape
            output_image = Imageplain(output_txt.flatten(), height, width)
            print("Cipher decrypted")
            return output_image
    
    def compilerunpegasus(self, exe_path: str):
        """
        Compiles and runs the OpenPEGASUS decryption code.

        Args:
            exe_path (str): Path to the executable file.
        """
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
        """
        Compiles and runs the OpenFHE decryption code.

        Args:
            exe_path (str): Path to the executable file.
        """
        if self.exeexist == 0:
            build_path = os.path.join(self.openfhe_path, 'build')
            pegasus_compile_command = [
                'make',
                '-C',
                build_path,
                '-j'
            ]
            subprocess.run(pegasus_compile_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run_command = [exe_path]
            subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            run_command = [exe_path]
            subprocess.run(run_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Cryptor:
    """
    Cryptor class that provides methods to read an image, encrypt it
    using OpenPEGASUS or OpenFHE, and decrypt the encrypted cipher.
    """
    def __init__(self, rotate_steps: list = [], scheme: str = 'openfhe',):
        self.scheme = scheme
        if scheme == 'openfhe':
            if rotate_steps == []:
                rotate_steps = [1]
            self.encryptor = Encryptor(rotate_steps)
        elif scheme == 'pegasus':
            self.encryptor = Encryptor()
        else:
            raise ValueError(f"Invalid scheme: {scheme}")

    def read(self, filepath: str) -> Imageplain:
        """
        Reads an image from the given file path.

        Args:
            filepath (str): Path to the image file.

        Returns:
            Imageplain: Imageplain object containing the image data.
        """
        image = self.encryptor.read(filepath)
        self.decryptor = Decryptor(self.encryptor)
        return image
    
    def encrypt(self, image: Imageplain, output_path: str = './test/encrypted_cipher.bin') -> Cipher:
        """
        Encrypts the given image using OpenPEGASUS or OpenFHE.

        Args:
            image (Imageplain): Imageplain object containing the image data.
            output_path (str): Path to save the encrypted cipher.

        Returns:
            Cipher: Cipher object containing the encrypted data.
        """
        cipher = self.encryptor.encrypt(image, output_path)
        return cipher
    
    def decrypt(self, cipher: Cipher) -> Imageplain:
        """
        Decrypts the given cipher.

        Args:
            cipher (Cipher): Cipher object containing the encrypted data.

        Returns:
            Imageplain: Imageplain object containing the decrypted image data.
        """
        return self.decryptor.decrypt(cipher.path())
    
    def set_scheme(self, scheme: str):
        """
        Sets the encryption scheme.

        Args:
            scheme (str): The encryption scheme to be used.
        """
        self.scheme = scheme
