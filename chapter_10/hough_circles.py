import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import cv2

# Carregar e converter a imagem para escala de cinza
a = Image.open('images/gab.png')
gray = a.convert('L')

# Aplicar filtro gaussiano
img = ndi.gaussian_filter(gray, 1)

# Detectar círculos usando a Transformada de Hough
circles = cv2.HoughCircles(
    img,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=10,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=30
)

# Verificar se círculos foram encontrados
if circles is not None:
    circles = np.uint16(np.around(circles))

    # Para cada círculo detectado
    for i in circles[0, :]:
        center = (i[0], i[1])  # Coordenadas do centro
        radius = i[2]          # Raio do círculo

        # Extrair a região correspondente ao círculo
        mask = np.zeros_like(img)
        cv2.circle(mask, center, radius, 255, thickness=-1)  # Criar máscara circular

        # Calcular a média dos valores de pixel na região do círculo
        mean_val = cv2.mean(img, mask=mask)[0]  # [0] para obter o valor médio

        # Determinar se o círculo está preenchido ou não
        if mean_val < 128:  # Ajuste este valor de limiar conforme necessário
            filled = True
        else:
            filled = False

        # Imprimir as propriedades do círculo
        print(f"Círculo encontrado: Centro = {center}, Raio = {radius}, Preenchido = {filled}")

        # Desenhar o círculo na imagem original
        color = (0, 255, 0) if filled else (0, 0, 255)  # Verde para preenchido, vermelho para não preenchido
        cv2.circle(img, center, radius, color, 2)  # Círculo externo
        cv2.circle(img, center, 2, (255, 0, 0), 3)  # Centro do círculo

# Mostrar a imagem resultante
plt.imshow(img, cmap='gray')
plt.axis('off')  # Remover eixos
plt.show()
