import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, median
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter

a = cv2.imread('images/teste.jpg')
a1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
a1 = cv2.GaussianBlur(a1, (7, 7), 5)
thresh,b1 = cv2.threshold(a1, 0, 255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
b2 = cv2.erode(b1, np.ones((3, 3)),iterations = 2)

dist_trans = cv2.distanceTransform(b2, 3, 0)
thresh, dt = cv2.threshold(dist_trans, 1,
           255, cv2.THRESH_BINARY)
labelled, ncc = label(dt)
cv2.watershed(a, labelled)
contours, _ = cv2.findContours(labelled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoured_image = a.copy()  # Copiar a imagem original
cv2.drawContours(contoured_image, contours, -1, (0, 0, 255), 2)  # Desenhar os contornos em verde (BGR: 0, 255, 0)

plt.subplot(1, 2, 1)
plt.imshow(contoured_image, cmap='gray')
plt.title('Contoured Image')

plt.subplot(1, 2, 2)
plt.imshow(labelled, cmap="BrBG")
plt.title(f'Number of Objects: {ncc}')

plt.show()


# cv2.imwrite('images/output1.png', labelled)