import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
import cv2

a = Image.open('images/gab2.png')
gray = a.convert('L')

img_gray = ndi.gaussian_filter(gray, 1)
img_gray = np.array(img_gray)

circles = cv2.HoughCircles(
    img_gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=10,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=30
)

img_color = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2BGR)

matrix = list()

if circles is not None:
    circles = np.uint16(np.around(circles))

    tolerance = 10
    grouped_circles = []

    sorted_circles = sorted(circles[0], key=lambda x: x[1])

    current_group = [sorted_circles[0]]

    for i in range(1, len(sorted_circles)):
        if abs(sorted_circles[i][1] - sorted_circles[i - 1][1]) <= tolerance:
            current_group.append(sorted_circles[i])
        else:
            grouped_circles.append(current_group)
            current_group = [sorted_circles[i]]

    grouped_circles.append(current_group)

    for group in grouped_circles:
        lista = []
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        for circle in group:
            center = (circle[0], circle[1])
            radius = circle[2]

            mask = np.zeros_like(img_gray, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, thickness=-1)

            mean_val = cv2.mean(img_gray, mask=mask)[0]

            filled = mean_val < 128

            if filled:
                cv2.circle(img_color, center, radius, (0, 0, 255), 2)  # Círculo preenchido (vermelho)
            else:
                cv2.circle(img_color, center, radius, (0, 255, 0), 2)  # Círculo não preenchido (verde)

            lista.append((center[0], np.uint8(filled)))

            cv2.circle(img_color, center, 2, (255, 0, 0), 3)  # Centro do círculo

            print(f"Círculo no centro {center}, Raio: {radius}, Preenchido: {filled}")
        matrix.append(lista.copy())

sorted_matrix = []
for group in matrix:
    group_sorted = sorted(group, key=lambda x: x[0])
    second_values = [x[1] for x in group_sorted]
    sorted_matrix.append(second_values)

print(sorted_matrix)

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
