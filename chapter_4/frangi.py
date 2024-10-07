import cv2
import numpy as np
from skimage.filters import frangi
from skimage.color import rgb2gray

# Carregar a imagem em escala de cinza
img = cv2.imread('../images/angiogram1.png', cv2.IMREAD_GRAYSCALE)

# Converter a imagem para um array numpy (se já não estiver)
img1 = np.asarray(img)

# Aplicar o filtro de Frangi
img2 = frangi(img1, black_ridges=True)

# Normalizar a imagem para o intervalo [0, 255]
img3 = 255 * (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

# Converter para tipo uint8 (necessário para salvar com cv2)
img3 = img3.astype(np.uint8)

# Salvar a imagem resultante
cv2.imwrite('../images/enderson.png', img3)
