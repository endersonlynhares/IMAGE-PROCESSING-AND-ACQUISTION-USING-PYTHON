import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

img = cv.imread('../../images/ct_saltandpepper.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

window = np.ones((6, 6)) / 36

img_out = ndi.convolve(img, window)

img_out = np.uint8(img_out)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img_out, cmap='gray')
plt.show()

