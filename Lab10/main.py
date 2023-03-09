import numpy as np
import skimage
import matplotlib.pyplot as plt
from skimage import segmentation
from skimage import color
from skimage import feature
from skimage.color import rgb2gray

chelseaPhoto = skimage.data.chelsea()
fig, ax = plt.subplots(3, 3, figsize=(10,10))

# ZADANIE 1
slicChelseaPhoto = segmentation.slic(chelseaPhoto)
watershedChelseaPhoto = segmentation.watershed(np.mean(chelseaPhoto, 2))
quickshiftChelseaPhoto = segmentation.quickshift(chelseaPhoto)

ax[0, 0].imshow(slicChelseaPhoto, cmap='twilight')
ax[0, 0].set_title(str(np.unique(slicChelseaPhoto).size))
ax[1, 0].imshow(watershedChelseaPhoto, cmap='twilight')
ax[1, 0].set_title(str(np.unique(watershedChelseaPhoto).size))
ax[2, 0].imshow(quickshiftChelseaPhoto, cmap='twilight')
ax[2, 0].set_title(str(np.unique(quickshiftChelseaPhoto).size))

# ZADANIE 2
out0_1 = color.label2rgb(slicChelseaPhoto, chelseaPhoto, kind='overlay')
ax[0, 1].imshow(out0_1)
out0_2 = color.label2rgb(slicChelseaPhoto, chelseaPhoto, kind='avg')
ax[0, 2].imshow(out0_2)

out1_1 = color.label2rgb(watershedChelseaPhoto, chelseaPhoto, kind='overlay')
ax[1, 1].imshow(out1_1)
out1_2 = color.label2rgb(watershedChelseaPhoto, chelseaPhoto, kind='avg')
ax[1, 2].imshow(out1_2)

out2_1 = color.label2rgb(quickshiftChelseaPhoto, chelseaPhoto, kind='overlay')
ax[2, 1].imshow(out2_1)
out2_2 = color.label2rgb(quickshiftChelseaPhoto, chelseaPhoto, kind='avg')
ax[2, 2].imshow(out2_2)

plt.savefig('zad10-1.png')

# ZADANIE 3
edges = feature.canny(np.mean(chelseaPhoto, 2), sigma=3.0)

out0_2[edges] = 1
out1_2[edges] = 1
out2_2[edges] = 1

ax[0, 2].imshow(out0_2)
ax[1, 2].imshow(out1_2)
ax[2, 2].imshow(out2_2)

plt.savefig('zad10-3.png')

