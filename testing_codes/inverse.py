import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('pepples.webp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)  # 255 - gray
# gray = cv2.equalizeHist(gray)

edged = cv2.Canny(gray, 50, 150)
thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
kernel = np.ones((4, 4), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # Close
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Open

# ret, thres = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# find contours (i.e., outlines) of the foreground objects in the thresholded image
cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()

count = 0
# loop over the contours
for c in cnts:
    # draw each contour on the output image with a 2px random color
    # outline, then display the output contours one at a time
    # print(cv2.contourArea(c))
    if cv2.contourArea(c) > 5:  # ignore small objects
        cv2.drawContours(output, [c], -1,
                         (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
        count += 1

text = "Found {} objects".format(count)
plt.figure(figsize=(20, 15))
plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original image',
                                                                                                 size=20)
plt.subplot(222), plt.imshow(edged, cmap='gray'), plt.axis('off'), plt.title('Edges', size=20)
plt.subplot(223), plt.imshow(thresh, cmap='gray'), plt.axis('off'), plt.title('Binary image', size=20)
plt.subplot(224), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title(
    'Counting objects: ' + text, size=20)
plt.show()