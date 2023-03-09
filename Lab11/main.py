import numpy as np
from skimage.draw import disk
from skimage.measure import label
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN
from sklearn.metrics import adjusted_rand_score


# Zadanie 1
fig, ax = plt.subplots(2, 3, figsize=(10,10))
image = np.zeros((100, 100, 3)).astype(int)
groundTruth = np.zeros((100, 100)).astype(int)

for i in range(3):
    randomDiskSize = random.randint(10, 40)
    centerDiskX = random.randint(randomDiskSize, 80 - randomDiskSize)
    centerDiskY = random.randint(randomDiskSize, 80 - randomDiskSize)
    rr, cc = disk((centerDiskX, centerDiskY), randomDiskSize, shape=groundTruth.shape)

    randomCanal = random.randint(0, 2)
    randomAddValue = random.randint(100, 255)
    image[rr, cc, randomCanal] += randomAddValue
    groundTruth[rr, cc] += randomAddValue


mu, sigma = 0, 16 # mean and standard deviation
s = np.random.normal(mu, sigma, size=image.shape).astype(int)

imageWithNormalNoise = image + s
np.clip(imageWithNormalNoise, 0, 255)

label(groundTruth)

ax[0, 0].imshow(imageWithNormalNoise)
ax[0, 1].imshow(groundTruth)

plt.savefig('zad11-1.png')

# Zadanie 2
def pictureNormalization(photo):
    photo = (photo - np.mean(photo))
    photo = photo / np.std(photo)
    return photo


X = np.reshape(imageWithNormalNoise, (100*100, 3))
xx, yy = np.meshgrid(np.arange(100), np.arange(100))

xx = xx.flatten()
yy = yy.flatten()
x = np.concatenate((X, xx[:, np.newaxis], yy[:, np.newaxis]), axis=1).astype(float)
x = pictureNormalization(x)
y = np.reshape(groundTruth, (100*100))

print(x.shape, y.shape)
print(x[0], y[0])

# Zadanie 3
Clusters = [KMeans(), MiniBatchKMeans(), Birch(), DBSCAN()]
for i, cls in enumerate(Clusters):
    clsFit = cls.fit_predict(x)
    image = np.reshape(clsFit, (100, 100))
    score = adjusted_rand_score(clsFit, y)
    # print(i)
    # print(score)
    if i == 0:
        ax[0, 2].imshow(image)
        ax[0, 2].set_title(str(cls) + ' ' + str(score))
    else:
        ax[1, i-1].imshow(image)
        ax[1, i-1].set_title(str(cls) + ' ' + str(score))

plt.savefig('zad11-2.png')

