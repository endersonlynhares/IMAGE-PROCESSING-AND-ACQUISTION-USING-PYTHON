import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops
from skimage.filters import gaussian, median
from skimage.filters.thresholding import threshold_otsu

image = Image.open('images/gab.png')
img = cv2.imread('images/gab.png')
image_gray = image.convert('L')

image_gray = np.asarray(image_gray)

gaussian = gaussian(image_gray, 0)

thresh = threshold_otsu(gaussian)
image_binary = thresh > gaussian

image_binary = binary_dilation(image_binary, iterations=2)

image_labeling = label(image_binary)
props = regionprops(image_labeling, intensity_image=image_binary)

# contours, _ = cv2.findContours(image_binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if 1000 < area < 2000:  # Ajuste esse valor conforme necessário
#         cv2.drawContours(img, [contour], -1, (0, 255, 0), 1)

circle_questions = list()

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(image_binary, 'gray')

i = 0
for prop in props:
    if 850 < prop.area < 2200:
        i += 1
        painted = False
        if prop.area > 1000:
            painted = True
        print((prop.area, painted))
        circle_questions.append((prop, painted))

print(i)

correct_questions = 0
vertical_line_questions = list()
for i, (circle, painted) in enumerate(circle_questions):
    # print(f'Área: {prop.area}, Intensidade média: {prop.mean_intensity}')
    if painted:
        vertical_line_questions.append(circle.centroid[0])
        plt.plot(circle.centroid[1], circle.centroid[0], 'ro')
    else:
        plt.plot(circle.centroid[1], circle.centroid[0], 'bo')

qtd_questions = 1

for i in range(1, len(vertical_line_questions)):
    if vertical_line_questions[i] - vertical_line_questions[i-1] > 10:
        qtd_questions += 1
    else:
        qtd_questions -= 1

qtd_options = len(circle_questions) // qtd_questions
matrix = list()
question = list()
for i, (_, painted) in enumerate(circle_questions):
    if (i+1) % qtd_options != 0:
        question.append((np.uint8(painted), np.uint8(_.centroid[1])))
    else:
        question.append((np.uint8(painted), np.uint8(_.centroid[1])))
        matrix.append(question.copy())
        question = []

# print(matrix)
sorted_matrix = [sorted(row, key=lambda x: x[1], reverse=True) for row in matrix]
matrix = [[x[0] for x in sorted(row, key=lambda x: x[1])] for row in matrix]

# print(qtd_options)
# print(qtd_questions)
# print(sorted_matrix)

# print(f'Number of questions considered: {correct_questions}')
# print(f'Number of questions considered: {len(circle_questions) - correct_questions}')

plt.show()