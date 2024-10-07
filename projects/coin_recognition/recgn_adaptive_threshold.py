import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('images/moeda.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaussian = cv.GaussianBlur(gray, (7, 7), 0)
adapt_gaussian = cv.adaptiveThreshold(gaussian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 15, 3)
kernel = np.ones((4,4), np.uint8)
erode = cv.erode(adapt_gaussian, kernel)
dilate = cv.dilate(erode, kernel, iterations=3)

elementos = dilate.copy()

contornos, hierarquia = cv.findContours(elementos, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img_processada = img.copy()
img_processada_rgb = cv.cvtColor(img_processada, cv.COLOR_BGR2RGB)
for cnt in contornos:
  area= cv.contourArea(cnt)
  if len(cnt) >= 100:
    elipse = cv.fitEllipse(cnt)
    cv.ellipse(img_processada_rgb, elipse, (0, 255, 0), 3)

plt.imshow(img_processada_rgb, 'gray')
plt.show()