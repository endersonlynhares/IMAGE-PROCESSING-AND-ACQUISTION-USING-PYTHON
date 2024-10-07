import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('images/moeda.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gaussian = cv.GaussianBlur(gray, (5, 5), 3)

thresh = cv.threshold(gaussian, 0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

sure_bg = cv.dilate(opening, kernel, iterations=3)
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

sure_fg = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)[1]
sure_fg = np.uint8(sure_fg)

desconhecido = cv.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers += 1
markers[desconhecido == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [0, 255, 0]

plt.imshow(img)
plt.show()