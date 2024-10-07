from PIL import Image

im = Image.open('lena.png')

box = (100, 100, 400, 400)
region = im.crop(box)

region = region.transpose(Image.Transpose.ROTATE_90)

im.paste(region, box)

#
# for infile in sys.argv[1:]:
#     f, e = os.path.splitext(infile)
#     outfile = f + '.jpg'
#     if infile != outfile:
#         try:
#             with Image.open(infile) as im:
#                 im.save(outfile)
#         except OSError:
#             print('cannot convert', infile)

#
#
# print(im.format, im.size, im.mode)
#
im.show()
#

# import imageio.v2 as imageio  # Usar a versão 2 explicitamente
# from scipy import ndimage
# import matplotlib.pyplot as plt
#
# # Carregar a imagem usando imageio.v2
# image = imageio.imread('lena.png')
#
# # Aplicar um filtro de suavização com SciPy
# smoothed_image = ndimage.gaussian_filter(image, sigma=2)
#
# # Mostrar a imagem processada
# plt.imshow(smoothed_image, cmap='gray')
# plt.show()
