import cv2
import matplotlib.pyplot as plt
from skimage.exposure import adjust_sigmoid

img1 = cv2.imread('images/hequalization_input.png')
img2 = adjust_sigmoid(img1, gain=15)

plt.imshow(img2, cmap='gray')
plt.show()