import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=np.inf)

fig, ax = plt.subplots(2, 3, figsize=(10,10))
picture = scipy.io.loadmat('SalinasA_corrected.mat')['salinasA_corrected']

ax[0, 0].imshow(picture[:, :, 10], cmap='binary_r')
ax[0, 1].imshow(picture[:, :, 100], cmap='binary_r')
ax[0, 2].imshow(picture[:, :, 200], cmap='binary_r')

ax[1, 0].plot(picture[10, 10, :])
ax[1, 1].plot(picture[40, 40, :])
ax[1, 2].plot(picture[80, 80, :])

plt.savefig('zad08-1.png')


# ZADANIE 2
def pictureNormalization(photo):
    photo = (photo - np.min(photo))
    photo = photo / np.max(photo)
    return photo


X_PHOTO, Y_PHOTO, Z_PHOTO = picture.shape
picture2 = np.zeros((X_PHOTO, Y_PHOTO, 3))

fig, ax = plt.subplots(1, 2, figsize=(10,10))

colorRed = picture[:, :, 4]
colorGreen = picture[:, :, 12]
colorBlue = picture[:, :, 26]

colorRed = pictureNormalization(colorRed)
colorGreen = pictureNormalization(colorGreen)
colorBlue = pictureNormalization(colorBlue)

picture2[:, :, 0] = colorRed
picture2[:, :, 1] = colorGreen
picture2[:, :, 2] = colorBlue
ax[0].imshow(picture2, cmap='binary_r')

picturePcaReshaped = np.reshape(picture, (X_PHOTO*Y_PHOTO, Z_PHOTO))
pca = PCA(n_components=3)
pcaPicture = pca.fit_transform(picturePcaReshaped)
picturePcaReshaped = np.reshape(pcaPicture, (X_PHOTO, Y_PHOTO, 3))

picture3 = np.zeros((X_PHOTO, Y_PHOTO, 3))
picture3[:, :, 0] = pictureNormalization(picturePcaReshaped[:, :, 0])
picture3[:, :, 1] = pictureNormalization(picturePcaReshaped[:, :, 1])
picture3[:, :, 2] = pictureNormalization(picturePcaReshaped[:, :, 2])

ax[1].imshow(picture3, cmap='binary_r')
plt.savefig('zad08-2.png')


# ZADANIE 3
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1234)
clf = GaussianNB()


def classDifference(pic, Y):
    scores = []
    for train_index, test_index in rkf.split(pic):
        X_train, X_test = pic[train_index], pic[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))
    mean_score_clf = np.mean(scores)
    print(mean_score_clf)


Y = scipy.io.loadmat('SalinasA_gt.mat')['salinasA_gt']

tempY = Y.flatten()
Y = Y[Y != 0]

pictureReshaped = np.reshape(picture, (X_PHOTO*Y_PHOTO, Z_PHOTO))
pictureReshaped = pictureReshaped[tempY != 0]

# picture3 = np.reshape(picture3, (X_PHOTO*Y_PHOTO, Z_PHOTO))
# classDifference(picture3, tempY)
classDifference(pictureReshaped, Y)









