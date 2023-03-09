import skimage
import numpy as np
import matplotlib.pyplot as plt

## Zadanie 1 i 2
fig, ax = plt.subplots(2, 2, figsize=(10,10))
originalImage = skimage.data.shepp_logan_phantom()
ax[0, 0].imshow(originalImage, cmap='binary_r')

theta = np.linspace(0, 180, 100)
sinImage = skimage.transform.radon(originalImage, theta)
ax[0, 1].imshow(sinImage, cmap='binary_r', aspect='auto', interpolation='nearest')

reconImage = skimage.transform.iradon(sinImage, theta)
ax[1, 0].imshow(reconImage, cmap='binary_r')

error = reconImage - originalImage
ax[1, 1].imshow(error, cmap='binary_r')
plt.savefig('zad07-1.png')

## Zadanie 3


def makeSin(image, index, thetaVal):
    ax[index, 0].imshow(image, cmap='binary_r')
    thetaSec = np.linspace(0, 180, thetaVal)
    sinImage = skimage.transform.radon(image, thetaSec)
    ax[index, 1].imshow(sinImage, cmap='binary_r', aspect='auto', interpolation='nearest')

    reconImage = skimage.transform.iradon(sinImage, thetaSec)
    ax[index, 2].imshow(reconImage, cmap='binary_r')

    error = reconImage - image
    ax[index, 3].imshow(error, cmap='binary_r')

# Czesc 1
fig, ax = plt.subplots(4, 4, figsize=(10,10))
originalImage = skimage.data.shepp_logan_phantom()
n = [10, 30, 100, 1000]
for n in enumerate(n):
    makeSin(originalImage, n[0], n[1])

plt.savefig('zad07-2.png')

# Czesc 2
fig, ax = plt.subplots(4, 4, figsize=(10,10))
zoomInImage = originalImage[50:350, 50:350]

n = [10, 30, 100, 1000]
for n in enumerate(n):
    makeSin(zoomInImage, n[0], n[1])

plt.savefig('zad07-3.png')



