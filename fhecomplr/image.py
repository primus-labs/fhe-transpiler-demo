import matplotlib.pyplot as plt
import numpy as np
import os

class Imageplain:
    def __init__(self, data: list[float], width, height):
        self.data = data
        self.width = width
        self.height = height

    def show(self):
        image_array = np.array(self.data)
        image = image_array.reshape((self.height, self.width))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()
    
    def save(self, output_path, file_name = 'output_image'):
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