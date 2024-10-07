from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.ndimage as ndi

img = Image.open('../images/arara.webp').convert('L')
kernel = ImageFilter.Kernel((3, 3), (-0.1, -0.1, -0.1, -0.1,  0.8, -0.1, -0.1, -0.1, -0.1), 1, 0)
img_mean = img.filter(kernel)
plt.imshow(img_mean, cmap='gray')
plt.show()

# Usando OpenCV e SciPy
# a = cv.imread('images/arara.webp')
# # Converting the image to grayscale.
# a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
#
# # Definir o filtro de borda Laplaciano (3x3) correspondente ao que você usou no Pillow
# matrix = np.array([[-0.01, -0.01, -0.01],
#                    [-0.01,  0.08, -0.01],
#                    [-0.01, -0.01, -0.01]])
#
# # Aplicar a convolução com o modo de borda "constant"
# b = ndi.convolve(a, matrix, mode='constant', cval=0.0)
#
# # Converter para uint8
# b = np.uint8(b)
#
# # Exibir a imagem filtrada
# plt.imshow(b, cmap='gray')
# plt.show()
