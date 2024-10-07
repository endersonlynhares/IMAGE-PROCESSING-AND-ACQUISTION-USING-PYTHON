import numpy as np
import scipy.ndimage as ndi
import cv2 as cv
import matplotlib.pyplot as plt

a = cv.imread('../images/pepper_noise.webp')
a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

b = ndi.maximum_filter(a, size=4)
b = np.uint8(b)

plt.subplot(1, 2, 1)
plt.imshow(a, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(b, cmap='gray')
plt.show()