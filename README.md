import imageio
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk mengaplikasikan kernel Robert
def robert_operator(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    
    output_x = np.zeros(image.shape)
    output_y = np.zeros(image.shape)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            patch = image[i-1:i+1, j-1:j+1]
            output_x[i, j] = np.sum(patch * kernel_x)
            output_y[i, j] = np.sum(patch * kernel_y)
    
    output = np.sqrt(output_x**2 + output_y**2)
    return output

# Fungsi untuk mengaplikasikan kernel Sobel
def sobel_operator(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    output_x = np.zeros(image.shape)
    output_y = np.zeros(image.shape)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            patch = image[i-1:i+2, j-1:j+2]
            output_x[i, j] = np.sum(patch * kernel_x)
            output_y[i, j] = np.sum(patch * kernel_y)
    
    output = np.sqrt(output_x**2 + output_y**2)
    return output

# Muat gambar
image = imageio.imread('gambar.jpg', as_gray=True)

# Aplikasikan kernel Robert
robert_output = robert_operator(image)

# Aplikasikan kernel Sobel
sobel_output = sobel_operator(image)

# Tampilkan hasil
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')

plt.subplot(1, 3, 2)
plt.imshow(robert_output, cmap='gray')
plt.title('Hasil Robert')

plt.subplot(1, 3, 3)
plt.imshow(sobel_output, cmap='gray')
plt.title('Hasil Sobel')

plt.show()
