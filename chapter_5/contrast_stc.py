import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('images/bse.png')
# Finding the maximum and minimum pixel values
b = im.max()
a = im.min()
print(a,b)
# Converting im1 to float.
c = im.astype(float)
# Contrast stretching transformation.
im1 = 255.0*(c-a)/(b-a+0.0000001)
im1 = np.uint8(im1)
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(im1, cmap='gray')
plt.show()
