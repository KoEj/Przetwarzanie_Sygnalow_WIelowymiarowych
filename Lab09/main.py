import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, binary_erosion, binary_dilation, binary_opening, binary_closing


# ZADANIE 1
def pictureNormalization(photo):
    photo = (photo - np.min(photo))
    photo = photo / np.max(photo)
    return photo


fig, ax = plt.subplots(1, 4, figsize=(10, 10))
image = plt.imread('fingerprint.jpg')

picture = np.mean(image, 2)
picture = pictureNormalization(picture)
picture = np.where(picture > 0.6, 0, 1)
footprint = disk(1)

ax[0].imshow(binary_erosion(picture, footprint=footprint), cmap='binary')
ax[1].imshow(binary_dilation(picture, footprint=footprint), cmap='binary')
ax[2].imshow(binary_opening(picture, footprint=footprint), cmap='binary')
ax[3].imshow(binary_closing(picture, footprint=footprint), cmap='binary')
plt.savefig('zad09-1.png')

# ZADANIE 2
fig, ax = plt.subplots(4, 5, figsize=(10, 10))
def showPictures(i, footprint):
    ax[i, 0].imshow(footprint, cmap='binary')
    ax[i, 1].imshow(binary_erosion(picture, footprint=footprint), cmap='binary')
    ax[i, 2].imshow(binary_dilation(picture, footprint=footprint), cmap='binary')
    ax[i, 3].imshow(binary_opening(picture, footprint=footprint), cmap='binary')
    ax[i, 4].imshow(binary_closing(picture, footprint=footprint), cmap='binary')

kernels = [disk(1), disk(5), rectangle(10, 1), rectangle(1, 10)]
for i, kernel in enumerate(kernels):
    showPictures(i, kernel)

ax[2, 0].imshow(rectangle(10, 1), cmap='binary_r')
ax[3, 0].imshow(rectangle(1, 10), cmap='binary_r')

plt.savefig('zad09-2.png')

# ZADANIE 3
fig, ax = plt.subplots(4, 6, figsize=(10, 10))
def showPictures(i, footprint):
    tempPicture = np.zeros(image.shape)
    ax[i, 0].imshow(footprint, cmap='binary')
    ax[i, 1].imshow(binary_erosion(picture, footprint=footprint), cmap='binary')
    ax[i, 2].imshow(binary_dilation(picture, footprint=footprint), cmap='binary')
    ax[i, 3].imshow(binary_opening(picture, footprint=footprint), cmap='binary')
    ax[i, 4].imshow(binary_closing(picture, footprint=footprint), cmap='binary')
    tempPicture[:, :, 0] = binary_dilation(picture, footprint=footprint)
    tempPicture[:, :, 1] = binary_opening(picture, footprint=footprint)
    tempPicture[:, :, 2] = binary_closing(picture, footprint=footprint)
    ax[i, 5].imshow(tempPicture)

kernels = [disk(1), disk(5), rectangle(10, 1), rectangle(1, 10)]
for i, kernel in enumerate(kernels):
    showPictures(i, kernel)

ax[2, 0].imshow(rectangle(10, 1), cmap='binary_r')
ax[3, 0].imshow(rectangle(1, 10), cmap='binary_r')

plt.savefig('zad09-3.png')