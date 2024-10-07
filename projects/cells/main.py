import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.thresholding import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

image = cv.imread('images/teste.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gaussian = cv.GaussianBlur(gray, (9, 9), 5)
histogram = cv.calcHist([gaussian], [0], None, [256], [0, 256])

ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# thresh_teste = threshold_otsu(gaussian)
#
#
# image_otsu = 255 * (gaussian < thresh)
# image_otsu = np.uint8(image_otsu)

kernel = np.ones((4,4), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv.erode(opening, kernel, iterations=4)
sure_bg = cv.dilate(sure_bg, kernel, iterations=4)
dist_trans = cv.distanceTransform(opening, cv.DIST_L2, 5)
ret, sure_fg = cv.threshold(dist_trans, 0.095 * dist_trans.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# dilate = cv.dilate(thresh, np.ones((3, 3)), iterations=2)
# erode = cv.erode(dilate, np.ones((3, 3)), iterations=2)

# distance = ndi.distance_transform_edt(erode)
# coords = peak_local_max(distance, footprint=np.ones((3,3)), labels=erode)
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndi.label(mask)
# labels = watershed(-distance, markers, mask=erode)

histogram_otsu = cv.calcHist([thresh], [0], None, [256], [0, 256])

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
markers = cv.watershed(image, markers)
image[markers == -1] = [0,255,0]

markers = markers+1

markers[unknown==255] = 0

# plt.subplot(1, 3, 1)
# plt.imshow(sure_fg, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow( sure_bg, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(unknown, cmap='gray')
# plt.show()

plt.imshow(image)
plt.show()

# plt.figure(figsize=(10, 8))
#
# plt.subplot(2, 2, 1)
# plt.imshow(gaussian, cmap='gray')
# plt.title("Imagem em Tons de Cinza")
# plt.axis('off')  # Remove os eixos
#
# plt.subplot(2, 2, 2)
# plt.plot(histogram, color='black')
# plt.title("Histograma da Imagem em Tons de Cinza")
# plt.xlabel("Intensidade de Pixels")
# plt.ylabel("Número de Pixels")
# plt.xlim([0, 256])
#
# plt.subplot(2, 2, 3)
# plt.imshow(erode, cmap='gray')
# plt.title("Imagem em Tons de Cinza")
# plt.axis('off')  # Remove os eixos
#
# plt.subplot(2, 2, 4)
# plt.plot(histogram_otsu, color='black')
# plt.title("Histograma da Imagem em Tons de Cinza")
# plt.xlabel("Intensidade de Pixels")
# plt.ylabel("Número de Pixels")
# plt.xlim([0, 256])
#
# plt.tight_layout()
# plt.show()
