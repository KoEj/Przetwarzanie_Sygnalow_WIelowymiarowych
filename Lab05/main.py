import skimage
import numpy as np
import matplotlib.pyplot as plt
import re

fig, ax = plt.subplots(3,4, figsize=(10,10))

# Zadanie 1
n = 100
x = np.linspace(0, 11*np.pi, n)
sin = np.sin(x)
img = sin[:, np.newaxis]*sin[np.newaxis, :]

img = (img - np.min(img))
img = img / np.max(img)
l = 2**8-1
img = np.rint(img*l)
ax[0, 0].imshow(img, cmap='binary_r')

ft = np.fft.fft2(img)
ft1 = np.fft.fftshift(ft)
img2 = np.abs(ft1)
ax[0, 1].imshow(img2, cmap='binary_r')

img3 = np.log(img2)
ax[0, 2].imshow(img3, cmap='binary_r')

# Zadanie 2
lin = np.linspace(0, 11*np.pi, 100)
x, y = np.meshgrid(lin, lin)

Amplitudes = [0, 2, 3, 5, 10]
Angles = [0, 3, 5, 9, 14]
Wavelength = [1, 3, 5, 6, 10]

matrix = np.zeros((100, 100))

zipped = zip(Amplitudes, Angles, Wavelength)

for i in zipped:
    amp = i[0]
    ang = i[1]
    wvl = i[2]
    matrix += amp * np.sin((2*np.pi * ((x * np.cos(ang)) + (y * np.sin(ang)))) / wvl)


ax[1, 0].imshow(matrix, cmap='binary_r')

ft = np.fft.fft2(matrix)
ft2 = np.fft.fftshift(ft)
img4 = np.abs(ft2)
ax[1, 1].imshow(img4, cmap='binary_r')

img5 = np.log(img4)
ax[1, 2].imshow(img5, cmap='binary_r')

camera = skimage.data.camera()
ax[2, 0].imshow(camera, cmap='binary_r')

ft = np.fft.fft2(camera)
ft3 = np.fft.fftshift(ft)
img6 = np.abs(ft3)
ax[2, 1].imshow(img6, cmap='binary_r')

img7 = np.log(img6)
ax[2, 2].imshow(img7, cmap='binary_r')

# Zadanie 3


def makeColour(ft_photo):
    x, y = ft_photo.shape
    matrix = np.zeros((x, y, 3))

    # czerwony
    ift1 = np.fft.ifftshift(ft_photo.real)
    ift1 = np.fft.ifft2(ift1)
    ift1.real = (ift1.real - np.min(ift1.real))
    ift1.real = ift1.real / np.max(ift1.real)
    matrix[:, :, 0] = ift1.real

    # niebieski
    ift2 = np.fft.ifftshift(ft_photo)
    ift2 = np.fft.ifft2(ift2)
    ift2.real = (ift2.real - np.min(ift2.real))
    ift2.real = ift2.real / np.max(ift2.real)
    matrix[:, :, 2] = ift2.real

    # zielone
    ft_photo.real = 0
    ift3 = np.fft.ifftshift(ft_photo)
    ift3 = np.fft.ifft2(ift3)
    ift3.real = (ift3.real - np.min(ift3.real))
    ift3.real = ift3.real / np.max(ift3.real)
    matrix[:, :, 1] = ift3.real

    # getImg = ft_photo.imag
    # for i in range(y):
    #     for j in range(x):
    #         # print(getImg[j, i])
    #         getImg[j, i] = complex(0, abs(j))
    #         break
    #     break
    # ift3 = np.fft.ifftshift(getImg)
    # ift3 = np.fft.ifft2(ift3)
    # ift3.real = (ift3.real - np.min(ift3.real))
    # ift3.real = ift3.real / np.max(ift3.real)
    # matrix[:, :, 1] = ift3.real

    return matrix


ax[0, 3].imshow(makeColour(ft1))
ax[1, 3].imshow(makeColour(ft2))
ax[2, 3].imshow(makeColour(ft3))


plt.savefig('zad05.png')