import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage as ndi

# Carregar e converter a imagem para tons de cinza
a = cv.imread('../images/death.jpg')
a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

# Definir manualmente o kernel Sobel (x)
sobel_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Aplicar o filtro Sobel usando scipy.ndimage.convolve
sobel_filtered = ndi.convolve(a, sobel_kernel)

# Exibir a imagem resultante
plt.imshow(sobel_filtered, cmap='gray')
plt.show()
