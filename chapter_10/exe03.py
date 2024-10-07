import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

# Carregar imagem
image_color = cv2.imread('images/coins.jpg')  # imagem colorida
gray_image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Aplicar filtro gaussiano e de mediana para suavizar a imagem
gaussian = cv2.GaussianBlur(gray_image, (5, 5), 0)
median = cv2.medianBlur(gaussian, 13)

# Aplicar binarização usando threshold Otsu
thresh = threshold_otsu(median)
binary_image = median > thresh

# Converter para formato binário adequado (0 e 255)
binary_image = np.uint8(binary_image * 255)

# Aplicar dilatação e erosão para suavizar os contornos
dilate = cv2.dilate(binary_image, np.ones((5, 5), np.uint8), iterations=1)
erode = cv2.erode(dilate, np.ones((5, 5), np.uint8), iterations=1)

cleared = ~erode

# Etiquetar as moedas (regiões conectadas)
label_image = label(cleared)

# Obter propriedades das regiões (moedas)
props = regionprops(label_image)

# Definir intervalos de tamanho (área) para as moedas
small_size = 11000  # Ajustar valor de acordo com a imagem
medium_size = 13000
large_size = 15000

# Inicializar contadores
small_count = 0
medium_count = 0
large_count = 0

# Analisar propriedades de cada moeda
for prop in props:
    area = prop.area
    if area < small_size:
        small_count += 1
    elif small_size <= area < medium_size:
        medium_count += 1
    else:
        large_count += 1

contours, _ = cv2.findContours(cleared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoured_image = image_color.copy()  # Copiar a imagem original
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)  # Desenhar os contornos em verde (BGR: 0, 255, 0)


# Mostrar os resultados na imagem original
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))  # Mostrar a imagem colorida original

# Adicionar texto com contagem de cada tipo de moeda
ax.text(10, 0, f'Pequenas: {small_count}', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
ax.text(10, 30, f'Médias: {medium_count}', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))
ax.text(10, 60, f'Grandes: {large_count}', color='white', fontsize=9, bbox=dict(facecolor='black', alpha=0.5))

# Mostrar imagem com contagens
plt.show()
