import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import cv2 as cv

# Carregar e processar a imagem
image = cv.imread('gab.png')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Aplicar filtros de suavização
gaussian = cv.GaussianBlur(image_gray, (5, 5), 0)
median = cv.medianBlur(gaussian, 13)

# Aplicar threshold Otsu para binarizar
thresh = threshold_otsu(median)
binary_image = np.uint8((median < thresh) * 255)

# Calcular a distância euclidiana para a imagem binária
distance = ndi.distance_transform_edt(binary_image)

# Encontrar picos locais para definir marcadores
# A máscara é a imagem binária
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_image)

# Gerar marcadores baseados nos picos locais
markers = np.zeros_like(binary_image, dtype=np.int32)
markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)

# Aplicar o algoritmo Watershed
labels = watershed(-distance, markers, mask=binary_image)

# Visualizar os resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(binary_image, cmap='gray')
ax[0].set_title('Imagem binarizada')

ax[1].imshow(-distance, cmap='gray')
ax[1].set_title('Distância transformada')

ax[2].imshow(labels, cmap='nipy_spectral')
ax[2].set_title('Segmentação Watershed')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()
