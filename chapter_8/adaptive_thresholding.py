import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Carrega a imagem em modo colorido (BGR) e a versão em escala de cinza
image_color = cv2.imread('images/moeda.png')  # imagem colorida
a = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Aplicando filtros (uso de medianBlur para reduzir ruídos)
gaussian = cv2.GaussianBlur(a, (5, 5), 0)
median = cv2.medianBlur(gaussian, 13)

# Fechamento (dilatação seguida de erosão)
dilate = cv2.dilate(median, np.ones((5, 5)), iterations=3)
erode = cv2.erode(dilate, np.ones((5, 5)), iterations=3)

# Aplicando limiar adaptativo
b = cv2.adaptiveThreshold(erode, erode.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 17, 10)

# Encontrar os contornos
contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos por área mínima (evitando contornos muito pequenos que podem ser ruído)
min_area = 100 # Ajuste este valor conforme necessário
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Desenhar os contornos na imagem colorida
image_with_contours = image_color.copy()
cv2.drawContours(image_with_contours, filtered_contours, -1, (0, 255, 0), 3)  # cor verde

# Contar o número de contornos (objetos detectados)
num_contours = len(filtered_contours)
print(f'Número de objetos detectados: {num_contours}')

# Exibir a imagem com contornos
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Imagem binária (limiar)')
plt.imshow(b, 'gray')

plt.subplot(1, 2, 2)
plt.title(f'Imagem com contornos - {num_contours} objetos')
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))  # Convertendo para RGB para exibir corretamente no Matplotlib

plt.show()
