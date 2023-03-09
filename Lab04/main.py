import skimage
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,4, figsize=(10,10))

importedImage = plt.imread('vessel.jpeg')

importedImageMean = np.mean(importedImage[::, ::], 2)

#Sobel S1-S4
SOBEL_S1 = np.zeros((3,3))

SOBEL_S1[0,0] = SOBEL_S1[2,0] = -1
SOBEL_S1[1,0] = -2
SOBEL_S1[1,2] = 2
SOBEL_S1[0,2] = SOBEL_S1[2,2] = 1

SOBEL_S2 = np.zeros((3,3))
SOBEL_S2[1,0] = SOBEL_S2[2,1] = -1
SOBEL_S2[2,0] = -2
SOBEL_S2[0,2] = 2
SOBEL_S2[1,2] = SOBEL_S2[0,1] = 1

SOBEL_S3 = np.zeros((3,3))
SOBEL_S3[0,0] = SOBEL_S3[0,2] = 1
SOBEL_S3[0,1] = 2
SOBEL_S3[2,0] = SOBEL_S3[2,2] = -1
SOBEL_S3[2,1] = -2

SOBEL_S4 = np.zeros((3,3))
SOBEL_S4[0,0] = 2
SOBEL_S4[0,1] = SOBEL_S4[1,0] = 1
SOBEL_S4[2,2] = -2
SOBEL_S4[2,1] = SOBEL_S4[1,2] = -1


img1 = ndimage.convolve(importedImageMean, SOBEL_S1)
img2 = ndimage.convolve(importedImageMean, SOBEL_S2)
img3 = ndimage.convolve(importedImageMean, SOBEL_S3)
img4 = ndimage.convolve(importedImageMean, SOBEL_S4)

ax[0, 0].imshow(img1, cmap='binary_r')
ax[0, 1].imshow(img2, cmap='binary_r')
ax[0, 2].imshow(img3, cmap='binary_r')
ax[0, 3].imshow(img4, cmap='binary_r')

#Zad2

def cor(image, J):
    i, j = image.shape
    result = np.zeros((i-2, j-2))

    for y in range(i-2):
        for x in range(j-2):
            result[y,x] = mnozJad(image, J, y, x)

    return result

def mnozJad(image, J, x, y):
    returnedValue = 0
    returnedValue = returnedValue + image[x, y] * J[0, 0]
    returnedValue = returnedValue + image[x, y+1] * J[0, 1]
    returnedValue = returnedValue + image[x, y+2] * J[0, 2]
    returnedValue = returnedValue + image[x+1, y] * J[1, 0]
    returnedValue = returnedValue + image[x+1, y+1] * J[1, 1]
    returnedValue = returnedValue + image[x+1, y+2] * J[1, 2]
    returnedValue = returnedValue + image[x+2, y] * J[2, 0]
    returnedValue = returnedValue + image[x+2, y+1] * J[2, 1]
    returnedValue = returnedValue + image[x+2, y+2] * J[2, 2]
    return returnedValue


ax[1, 0].imshow(cor(importedImageMean, SOBEL_S1), cmap='binary_r')
ax[1, 1].imshow(cor(importedImageMean, SOBEL_S2), cmap='binary_r')
ax[1, 2].imshow(cor(importedImageMean, SOBEL_S3), cmap='binary_r')
ax[1, 3].imshow(cor(importedImageMean, SOBEL_S4), cmap='binary_r')

#ZAD 3


def changeFilter(kern):
    vert = np.flipud(kern)
    return np.fliplr(vert)


def cor_3(image, kern):
    J = changeFilter(kern)
    i, j = image.shape
    jy, jx = J.shape
    result = np.zeros((i-2, j-2))

    if jx >= j or jy >= i:
        temp = J
        J = image
        image = temp

    for y in range(i-(jy-1)):
        for x in range(j-(jx-1)):
            result[y,x] = mnozJad_3(image, J, y, x, jy, jx)

    return result


def mnozJad_3(image, J, y, x, jy, jx):
    vec = image[y:y+jy, x:x+jx] * J
    return np.sum(vec)


# SOBEL_S5 = np.zeros((800, 700))
# ax[2, 0].imshow(cor_3(importedImageMean, SOBEL_S5), cmap='binary_r')

ax[2, 0].imshow(cor_3(importedImageMean, SOBEL_S1), cmap='binary_r')
ax[2, 1].imshow(cor_3(importedImageMean, SOBEL_S2), cmap='binary_r')
ax[2, 2].imshow(cor_3(importedImageMean, SOBEL_S3), cmap='binary_r')
ax[2, 3].imshow(cor_3(importedImageMean, SOBEL_S4), cmap='binary_r')

plt.savefig('zad04.png')


