import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/angiogram1.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gamma = 8

pixels  = img.astype(float)
max_pixel = np.max(pixels)

# imagem normalizada
img_out = pixels / max_pixel

img_out = np.clip(img_out, 1e-8, 1.0)

gamma_correction = np.log(img_out) * gamma
gamma_correction_p = np.exp(gamma_correction) * 255

img_out = gamma_correction_p.astype(int)

plt.imshow(img_out, cmap='gray')
plt.show()

