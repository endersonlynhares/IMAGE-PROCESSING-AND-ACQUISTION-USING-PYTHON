import cv2
import numpy as np
import matplotlib.pyplot as plt

a = cv2.imread('images/bse.png')
b1 = a.astype(float)
b2 = np.max(b1)
c = (255 * np.log(1 + b1) / np.log(1 + b2))

c1 = c.astype(int)

plt.subplot(1, 2, 1)
plt.imshow(a, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(c1, cmap='gray')
plt.show()
