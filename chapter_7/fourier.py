import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fp
import cv2

f = cv2.imread('images/fft1.png')
f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.title('Imagem Original')
plt.imshow(f, cmap='gray')
plt.colorbar()
plt.show()

F = fp.fft2(f)

Fm = np.absolute(F)
Fm /= np.max(Fm)
Fm = fp.fftshift(Fm)
Fm = np.log(Fm)

plt.figure()
plt.title('Imagem em escala de log')
plt.imshow(Fm, cmap='gray')
plt.colorbar()
plt.show()
