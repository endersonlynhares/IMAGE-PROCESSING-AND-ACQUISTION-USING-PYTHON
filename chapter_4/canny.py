import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
a = cv2.imread('images/moeda.png')
a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
b = ndi.gaussian_filter(a, sigma=2, mode='constant')
adapt_gaussian = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

t, th = cv2.threshold(b, 150,200, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# b = cv2.Canny(th, 100, 150)

result = np.hstack((a,b))

plt.imshow(adapt_gaussian, cmap='gray')
plt.show()