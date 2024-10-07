import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist

img = cv2.imread('img.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = equalize_adapthist(img, clip_limit=0.05)

img3 = img2 * 255

img3 = np.uint8(img3)

plt.imshow(img3, cmap='gray')
plt.show()
