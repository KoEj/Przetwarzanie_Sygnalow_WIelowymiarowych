import numpy as np
import skimage
from skimage.transform import resize
import matplotlib.pyplot as plt


# ZADANIE 1
img = skimage.data.camera()
img = resize(img, (128, 128))

gx = np.zeros((128, 128))
gy = np.zeros((128, 128))
mag = np.zeros((128, 128))
angle = np.zeros((128, 128))

fig, ax = plt.subplots(2, 3, figsize=(10,10))

for y in range(np.shape(img)[0]):
    for x in range(np.shape(img)[1]):
        if 0 < x < 127:
            gx[x][y] = img[x + 1][y] - img[x - 1][y]
        if 0 < y < 127:
            gy[x][y] = img[x][y + 1] - img[x][y - 1]

        mag[x][y] = np.sqrt((gx[x][y]) ** 2 + (gy[x][y]) ** 2)

angle = np.arctan(gy / gx) + (np.pi / 2)

ax[0, 0].imshow(img, cmap='binary_r')
ax[0, 1].imshow(gx, cmap='binary_r')
ax[0, 2].imshow(gy, cmap='binary_r')
ax[1, 0].imshow(mag, cmap='binary_r')
ax[1, 1].imshow(angle, cmap='binary_r')
plt.savefig('zad12-1.png')

# ZADANIE 2
s = 8
bins = 9
mask = np.zeros((128, 128))
superPixelValue = 0

for i in range(int(img.shape[0]/s)):
    for j in range(int(img.shape[1]/s)):
        mask[(i * s): (i * s + s), (j * s): (j * s + s)] = superPixelValue
        superPixelValue += 1

hog = np.zeros(shape=(int((img.shape[0]/s) * (img.shape[1]/s)), bins))
step = np.pi / bins

row = 0
for i, x in enumerate(mask):
    if i % 16:
        row += 1

    ang_v = angle[i][row].flatten()
    mag_v = mag[i][row].flatten()

angle = np.arctan(gy / gx) + (np.pi / 2)
fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0, 0].imshow(mask)
ax[0, 1].imshow(hog)
plt.savefig('zad12-2.png')
