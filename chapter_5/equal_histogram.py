import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('img.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

f1 = img1.flatten()

hist,bins = np.histogram(img1, 256, (0, 255))

cdf = hist.cumsum()

cdf_mask = np.ma.masked_equal(cdf, 0)

num_cdf_m = (cdf_mask - cdf_mask.min()) * 255

den_cdf_m  = (cdf_mask.max() - cdf_mask.min())

cdf_mask = num_cdf_m / den_cdf_m
cdf = np.ma.filled(cdf_mask, 0).astype('uint8')
im2 = cdf[f1]
img3 = np.reshape(im2, img1.shape)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(img3, cmap='gray')
plt.title('Transformed Image')

plt.subplot(2, 2, 3)
plt.hist(f1, bins=256, range=(0, 255), color='blue', alpha=0.7)
plt.title('Histogram of Original Image')

plt.subplot(2, 2, 4)
plt.hist(im2, bins=256, range=(0, 255), color='green', alpha=0.7)
plt.title('Histogram of Transformed Image')

plt.tight_layout()
plt.show()