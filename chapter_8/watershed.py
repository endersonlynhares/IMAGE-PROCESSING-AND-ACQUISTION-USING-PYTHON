import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Opening the image.
a = cv2.imread('images/coins.jpg')
# Converting to grayscale.
a1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
# Thresholding the image to obtain cell pixels.
thresh,b1 = cv2.threshold(a1, 0, 255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Since Otsu’s method has over segmented the image
# erosion operation is performed.
b2 = cv2.erode(b1, None,iterations = 2)
# Distance transform is performed
dist_trans = cv2.distanceTransform(b2, 2, 3)
# Thresholding the distance transform image to obtain
# pixels that are foreground.
thresh, dt = cv2.threshold(dist_trans, 1,
           255, cv2.THRESH_BINARY)
# Performing labeling.
labelled, ncc = label(dt)
# Performing watershed.
cv2.watershed(a, labelled)
# Saving the image as watershed_output.png

plt.imshow(labelled)
plt.show()