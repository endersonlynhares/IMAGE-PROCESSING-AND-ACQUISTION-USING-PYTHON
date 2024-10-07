import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para aplicar a transformação de Fourier
def apply_fourier_transform(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum, fshift

# Função para aplicar a transformada inversa
def inverse_fourier_transform(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

# Carrega a imagem
img = cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)

# Aplica o desfoque gaussiano
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# Aplica a transformação de Fourier
magnitude_spectrum, fshift = apply_fourier_transform(blurred_img)

# Aplica um filtro passa-alta (opcional, para destacar bordas)
rows, cols = blurred_img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 30
cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)
fshift_filtered = fshift * (1 - mask)

# Inverte a transformada
img_filtered_back = inverse_fourier_transform(fshift_filtered)

# Aplica o detector de bordas de Canny
edges = cv2.Canny(np.uint8(img_filtered_back), threshold1=100, threshold2=200)

# Exibe os resultados
plt.figure(figsize=(15, 10))
plt.subplot(1, 4, 1)
plt.title('Imagem Original')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Magnitude da Transformada de Fourier')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Imagem Filtrada (Bordas Realçadas)')
plt.imshow(img_filtered_back, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Bordas Detectadas (Canny)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
