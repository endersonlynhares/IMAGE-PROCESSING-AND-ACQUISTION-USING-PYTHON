import numpy as np
from PIL import Image
from skimage.transform import AffineTransform, warp

img = Image.open('images/angiogram1.png').convert('L')
img1 = np.array(img)

transformation = AffineTransform(translation=(10, 4))
img2 = warp(img1, transformation)
img4 = Image.fromarray((img2 * 255).astype(np.uint8))

img4.show()