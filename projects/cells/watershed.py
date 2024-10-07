import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, median

a = cv2.imread('images/teste.jpg')
a1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
thresh,b1 = cv2.threshold(a1, 0, 255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
b2 = cv2.erode(b1, np.ones((3, 3)),iterations = 2)

dist_trans = cv2.distanceTransform(b2, 3, 0)
thresh, dt = cv2.threshold(dist_trans, 1,
           255, cv2.THRESH_BINARY)
labelled, ncc = label(dt)
cv2.watershed(a, labelled)
contours, _ = cv2.findContours(labelled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

plt.subplot(1, 2, 1)
plt.imshow(a1, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(labelled, cmap="BrBG_r")
plt.show()
