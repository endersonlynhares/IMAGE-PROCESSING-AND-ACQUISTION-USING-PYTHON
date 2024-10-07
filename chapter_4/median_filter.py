import numpy as np
import cv2 as cv
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

a = cv.imread('../images/ct_saltandpepper.png')
a = cv.cvtColor(a, cv.COLOR_BGR2GRAY)

b = ndi.median_filter(a, size=8)
b = np.uint8(b)

plt.subplot(1,2,1)
plt.imshow(a, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(b, cmap='gray')
plt.show()
