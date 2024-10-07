import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
from PIL import Image

def apply_lowpass(img):
    img1 = ft.fft2(img)
    img2 = ft.fftshift(img1)

    img_fre = abs(img1)
    img_fre /= img_fre.max()
    img_fre = ft.fftshift(img_fre)
    img_fre = np.log(img_fre + 1e-8)

    M = img2.shape[0]
    N = img2.shape[1]

    H = np.ones((M, N))
    center1 = M/2
    center2 = N/2
    d_0 = 30.0

    for i in range(1, M):
        for j in range(1, N):
            r1 = (i - center1) ** 2 + (j - center2) ** 2
            r = math.sqrt(r1)
            if r > d_0:
                H[i, j] = 0.0

    H = Image.fromarray(H)
    con = img2 * H
    e = abs(ft.ifft2(con))

    return e, img_fre

if __name__ == '__main__':
    # img = Image.open('images/endothelium.png').convert('L')
    img = cv2.imread('images/WIN_20240924_11_59_22_Pro.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e, img_fre = apply_lowpass(img)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img_fre, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(e, cmap='gray')
    plt.show()

