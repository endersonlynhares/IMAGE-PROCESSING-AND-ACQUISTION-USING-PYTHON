import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as fftim
from PIL import Image

# Opening the image and converting it to grayscale.
b = Image.open('images/fft1.png').convert('L')
# Performing FFT.
c = abs(fftim.fft2(b))
c /= np.max(c)
# Shifting the Fourier frequency image.
d = fftim.fftshift(c)
d = np.log(d)
# Converting the d to floating type and saving it
# as fft1_output.raw in Figures folder.
plt.figure()
plt.title('Imagem em escala de log')
plt.imshow(d, cmap='gray')
plt.colorbar()
plt.show()