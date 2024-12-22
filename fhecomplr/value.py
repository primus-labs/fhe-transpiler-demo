import matplotlib.pyplot as plt
import numpy as np
import os

class Cipher:
    """
    Cipher class that provides methods to read and save a cipher. 
    Has a path and slots attribute.
    """
    def __init__(self, cipher_path: str, slots: int):
        self.cipher_path = cipher_path
        self.slots = slots
    
    def path(self):
        """
        Returns the path of the cipher.
        """
        return self.cipher_path
    
    def slot(self):
        """
        Returns the number of slots of the cipher.
        """
        return self.slots

    def _get_file_info(self):
        """
        Returns a dictionary with file information (if the file exists).
        """
        if os.path.exists(self.cipher_path):
            file_info = {
                "size": os.path.getsize(self.cipher_path),
                "file_name": os.path.basename(self.cipher_path)
            }
            return file_info
        else:
            return None
    
    def show(self):
        """
        Shows the cipher.
        """
        file_info = self._get_file_info()
        if file_info:
            print(f"Cipher(", f"file: {file_info['file_name']}, size: {file_info['size']} bytes)")
        else:
            print(f"Cipher(path='{self.cipher_path}', file: <not found>)")

    
    def __str__(self):
        """
        Custom string representation for printing.
        """
        return f"Cipher(path='{self.cipher_path}')\nCipher can't be shown."

    def __repr__(self):
        """
        Official string representation for debugging.
        """
        return f"Cipher(path='{self.cipher_path}', slots={self.slots})"
    
class Imageplain:
    """
    Imageplain class that provides methods to show and save an image.
    Has a flattened data, width, and height attribute.
    """
    def __init__(self, data: list[float], width: int, height: int):
        self.data = data
        self.width = width
        self.height = height

    def show(self):
        """
        Shows the image.
        """
        image_array = np.array(self.data)
        image = image_array.reshape((self.height, self.width))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    
    def __str__(self):
        """
        Custom string representation for printing.
        """
        return f"Imageplain(width={self.width}, height={self.height})"
    
    def __repr__(self):
        """
        Official string representation for debugging.
        """
        return f"Imageplain(data={self.data}, width={self.width}, height={self.height})"
    
    def save(self, output_path: str, file_name: str = 'output_image'):
        """
        Saves the image.

        Args:
            output_path (str): The path to save the image.
            file_name (str): The name of the file to be saved.
        """
        image_array = np.array(self.data)
        image = image_array.reshape((self.height, self.width))
        file_extension = 'png'
        if '.' not in os.path.basename(file_name):
            file_name += '.png'
        else:
            file_extension = os.path.splitext(file_name)[1]
            if file_extension == '':
                file_name += 'png'
        if not file_name.endswith(".png"):
            file_name += ".png"
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, file_name)
        plt.imsave(output_path, image, cmap='gray', format='png')
        print(f"Image saved as {output_path}")
