import cv2 as cv
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np

QUESTOES = 4
OPCOES = 3

matrix_correction = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])

def sort_and_group_contours(contours, num_questions, num_options):
    centroids = []

    for contour in contours:
        M = cv.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY, contour))

    centroids_sorted = sorted(centroids, key=lambda c: c[1])

    row_height_threshold = 40
    rows = [[] for _ in range(num_questions)]
    current_row_index = 0
    current_row_y = centroids_sorted[0][1]

    for cX, cY, contour in centroids_sorted:
        if abs(cY - current_row_y) > row_height_threshold and len(rows[current_row_index]) == num_options:
            current_row_index += 1
            current_row_y = cY

        if current_row_index < num_questions:
            rows[current_row_index].append((cX, cY, contour))
    print(rows)
    for row in rows:
        row.sort(key=lambda c: c[0])

    return rows

image = cv.imread('gab2.png')
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

gaussian = cv.GaussianBlur(image_gray, (5, 5), 0)
thresh = threshold_otsu(gaussian)
binary_image = np.uint8((gaussian < thresh) * 255)

contours, _ = cv.findContours(binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

filtered_contours = [contour for contour in contours if 1200 < cv.contourArea(contour) < 2500]

sorted_rows = sort_and_group_contours(filtered_contours, QUESTOES, OPCOES)

matrix = np.zeros((QUESTOES, OPCOES), dtype=int)
matrix_result = np.zeros((QUESTOES, OPCOES))

for row_index, row in enumerate(sorted_rows):
    for col_index, (cX, cY, contour) in enumerate(row):
        mask = np.zeros_like(image_gray)
        cv.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv.mean(image_gray, mask=mask)[0]
        if mean_val <= thresh:
            matrix[row_index, col_index] = 1
        else:
            matrix[row_index, col_index] = 0

matrix_result = matrix * matrix_correction
count_question_corrected = 0
for row_index, row in enumerate(sorted_rows):
    for col_index, (cX, cY, contour) in enumerate(row):
        mask = np.zeros_like(image_gray)
        cv.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv.mean(image_gray, mask=mask)[0]
        painted = mean_val <= thresh

        if painted and matrix_result[row_index, col_index] == 1:
            cv.drawContours(image, [contour], -1, (0, 255, 0), 2)
            count_question_corrected += 1

        if painted and matrix_result[row_index, col_index] == 0:
            cv.drawContours(image, [contour], -1, (0, 0, 255), 2)

        if not painted and matrix_correction[row_index, col_index] == 1:
            cv.drawContours(image, [contour], -1, (0, 255, 255), 2)

print(f'Quantidade de questões ao todo: {QUESTOES}')
print(f'Quantidade de questões acertadas: {count_question_corrected}')
print(f'Quantidade de questões erradas: {QUESTOES - count_question_corrected}')
print(f'Porcetagem de acertos: {(count_question_corrected * 100) / QUESTOES }%')
print(f'Nota: {(count_question_corrected * 10) / QUESTOES }')


plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()
