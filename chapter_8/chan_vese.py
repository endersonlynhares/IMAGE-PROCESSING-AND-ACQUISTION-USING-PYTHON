import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import chan_vese
import numpy as np

# Opening the image and converting it into grayscale
img = Image.open('img.png').convert('L')
img = np.array(img)

gaussian = cv2.GaussianBlur(img, (5, 5), 0)
# median = cv2.medianBlur(gaussian, 13)

# Fechamento (dilatação seguida de erosão)
# dilate = cv2.dilate(median, np.ones((5, 5)), iterations=3)
# erode = cv2.erode(dilate, np.ones((5, 5)), iterations=3)

cv1 = chan_vese(gaussian, mu=0.2)
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()
ax[0].imshow(gaussian, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv1, cmap="gray")
ax[1].set_axis_off()
ax[1].set_title("mu=0.1", fontsize=12)

plt.show()