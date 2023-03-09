import skimage
import numpy as np
import matplotlib.pyplot as plt
from numpy import shape
from skimage.draw import disk

np.seterr(divide='ignore')

fig, ax = plt.subplots(3,2, figsize=(10,10))

image = plt.imread('image1.jpeg')
image = np.mean(image[::, ::], 2)
ax[0, 0].imshow(image, cmap='binary_r')

ft = np.fft.fft2(image)
ft1 = np.fft.fftshift(ft)
image2 = np.log(np.abs(ft1))
ax[0, 1].imshow(image2, cmap='binary_r')

img = (image2 - np.min(image2))
img = img / np.max(img)

image3 = np.where(img < 0.5, 0, 1)
image4 = np.where(img < 0.8, 0, 1)

ax[1, 0].imshow(image3, cmap='binary_r')
ax[1, 1].imshow(image4, cmap='binary_r')

points = np.argwhere(image4)
finalPoints = []
for point in points:
    if point[0]<100 and point[1]<100 or point[0]>125 and point[1] > 125:
        finalPoints.append(point)

#print(finalPoints)
# array[75, 93],
# array[93, 87],
# array[131, 137],
# array[149, 131]

#Zadanie 2
ftCopied = np.array(ft1)
ftCopied[75] = np.zeros(ftCopied.shape[1])
ftCopied[93] = np.zeros(ftCopied.shape[1])
ftCopied[131] = np.zeros(ftCopied.shape[1])
ftCopied[149] = np.zeros(ftCopied.shape[1])

image5 = np.log(np.abs(ftCopied))
ax[2, 0].imshow(image5, cmap='binary_r')

ift2 = np.fft.ifftshift(ftCopied)
ift2 = np.fft.ifft2(ift2)
ax[2, 1].imshow(ift2.real, cmap='binary_r')

plt.savefig('zad06.png')

#Zadanie 3
fig, ax = plt.subplots(2,2, figsize=(10,10))
secondImage = plt.imread('image2.jpg')

ax[0, 0].imshow(secondImage, cmap='binary_r')

SecondFt = np.fft.fft2(secondImage)
SecondFt1 = np.fft.fftshift(SecondFt)
SecondImage2 = np.log(np.abs(SecondFt1))
ax[0, 1].imshow(SecondImage2, cmap='binary_r')

# SecondImage3 = (SecondImage2 - np.min(SecondImage2))
# SecondImage3 = SecondImage3 / np.max(SecondImage3)
# SecondImage3 = np.where(SecondImage3 < 0.7, 0, 1)
# ax[1, 0].imshow(SecondImage3, cmap='binary_r')

# print(np.argwhere(SecondImage3))
# [111, 321]
# [233, 395]
# [235, 485]
# [295, 402]
# [358, 320]
# [295, 237]
# [233, 155]
# [173, 237]

secondPoints = np.array([[111, 321],[170, 399],[235, 485],[295, 402],[358, 320],[295, 237],[233, 155], [173, 237]])

SecondFt2 = SecondFt1

for point in secondPoints:
    rr, cc = disk((point[0], point[1]), 10, shape=SecondFt2.shape)
    SecondFt2[rr, cc] = 0

# print(secondPoints)

SecondImage3 = np.log(np.abs(SecondFt2))
ax[1, 0].imshow(SecondImage3, cmap='binary_r')

ift3 = np.fft.ifftshift(SecondFt2)
ift3 = np.fft.ifft2(ift3)
ax[1, 1].imshow(ift3.real, cmap='binary_r')

plt.savefig('zad06-2.png')