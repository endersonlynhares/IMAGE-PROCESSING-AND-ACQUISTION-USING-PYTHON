import numpy as np
from matplotlib.image import imread
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image, ImageFilter

a = Image.open('../images/sobel_vcir.png').convert('L')
b = a.filter(ImageFilter.Kernel(
    (3,3),
    [
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ],
    1,
    0
))

plt.imshow(b, cmap='gray')
plt.show()
