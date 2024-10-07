import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/imageinverse_input.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img_out = 255 - img

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img_out, cmap='gray')
plt.show()
