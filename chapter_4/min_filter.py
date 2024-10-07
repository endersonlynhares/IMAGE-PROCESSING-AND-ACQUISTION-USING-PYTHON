import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2 as cv

a = cv.imread('../images/salt_noise.webp')
a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

b = ndi.minimum_filter(a, size=4)
b = np.uint8(b)

plt.imshow(b, cmap='gray')
plt.show()
