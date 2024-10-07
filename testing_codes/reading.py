from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
img = Image.open('image.jpg').convert('L')

img = np.array(img)

img_out = Image.fromarray(img)

img_out.save('OUTPUT.jpg')

plt.imshow(img, 'gray')
plt.show()