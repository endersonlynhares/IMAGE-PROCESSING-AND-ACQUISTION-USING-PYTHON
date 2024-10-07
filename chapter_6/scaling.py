import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, warp

img = Image.open('images/angiogram1.png').convert('L')
img1 = np.array(img)

transformation  = AffineTransform(scale=(.2, .2))
img2 = warp(img1, transformation, order=5)
img4 = Image.fromarray((255 * img2).astype(np.uint8))

img4.show()