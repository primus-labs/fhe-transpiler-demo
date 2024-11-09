// This file should be moved to the OpenPEGASUS-master/build-release.

import numpy as np
import matplotlib.pyplot as plt

def read_txt_to_image(file_path):
    data = np.loadtxt(file_path)
    height, width = data.shape
    image = data.reshape((height, width))
    return image

def save_image(image, output_path):
    plt.imsave(output_path, image, cmap='gray', format='png')

def main():
    input_file = 'output_image.txt'
    output_file = 'output_image.png'
    image = read_txt_to_image(input_file)
    save_image(image, output_file)
    print(f'Image saved as {output_file}')

if __name__ == '__main__':
    main()
