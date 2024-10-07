import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, warp

img = Image.open('images/angiogram1.png').convert('L')

img1 = np.array(img)

transformation = AffineTransform(rotation=0.8)
img2 = warp(img1, transformation)
img3 = Image.fromarray((255 * img2).astype(np.uint8))

img3.show()