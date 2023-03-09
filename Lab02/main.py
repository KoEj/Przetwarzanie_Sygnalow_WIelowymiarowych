#Zad 1
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.interpolate import interpolate, interp2d
from skimage.transform import warp, AffineTransform

fig, ax = plt.subplots(4,2, figsize=(10,10))

cat = skimage.data.chelsea()
ax[0,0].imshow(cat)




cat_2 = np.mean(cat[::8, ::8], 2)
cat2 = (cat_2 - np.min(cat_2))
cat2 = cat2 / np.max(cat2)
ax[0,1].imshow(cat2, cmap='binary_r')

cat3arr = np.array(cat2)

fi = -(np.pi/12)
color = np.zeros((3,3)).astype(float)
color[0][0] = np.cos(fi)
color[1][1] = np.cos(fi)
color[0][1] = -np.sin(fi)
color[1][0] = np.sin(fi)
color[2][2] = 1

transform = AffineTransform(matrix = color)
cat3 = warp(cat2, transform)
ax[1,0].imshow(cat3, cmap='binary_r')

cy = 0
cx = -0.5
color = np.zeros((3,3)).astype(float)
color[0][0] = 1
color[1][1] = 1
color[2][2] = 1
color[0][1] = cx
color[1][0] = cy

transform = AffineTransform(matrix = color)
cat4 = warp(cat2, transform)
ax[1,1].imshow(cat4, cmap='binary_r')

#Zad 2
x = np.shape(cat3)[0]
y = np.shape(cat3)[1]

newX = np.linspace(0, x*8, x)
newY = np.linspace(0, y*8, y)
secondnewX = np.linspace(0, x*8, x*8)
secondnewY = np.linspace(0, y*8, y*8)

cat5 = interp2d(newY, newX, cat3, kind='cubic')
cat5 = cat5(secondnewY, secondnewX)
ax[2,0].imshow(cat5, cmap='binary_r')

print(np.round((cat5[0:15,0:15]),1))

x = np.shape(cat4)[0]
y = np.shape(cat4)[1]

newX = np.linspace(0, x*8, x)
newY = np.linspace(0, y*8, y)
secondnewX = np.linspace(0, x*8, x*8)
secondnewY = np.linspace(0, y*8, y*8)

cat6 = interp2d(newY, newX, cat4, kind='cubic')
cat6 = cat6(secondnewY, secondnewX)
ax[2,1].imshow(cat6, cmap='binary_r')
print(np.round((cat6[0:15,0:15]),1))

#Zad 3
# x = np.shape(cat3)[0]
# y = np.shape(cat3)[1]
#
# newX = np.linspace(0, x*8, x)
# newY = np.linspace(0, y*8, y)
# secondnewX = np.linspace(0, x*8, x*8)
# secondnewY = np.linspace(0, y*8, y*8)
#
# xp = [.1,.2,.3,.4,.5,.6,.7,.8]
# x = newX
# y = np.interp(y, secondnewY, secondnewY)
#
#
# print(x)
# print(y)
# cat5 = interp2d(newY, newX, cat3, kind='cubic')
# cat5 = cat5(secondnewY, secondnewX)
# ax[2,0].imshow(cat5, cmap='binary_r')



plt.savefig('zad02.png')