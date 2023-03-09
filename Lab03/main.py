import skimage
import numpy as np
import matplotlib.pyplot as plt

rgb = ['r', 'g', 'b']

def hist(image, x, y):
    for i in range(3):
        vhist = np.zeros(L)
        hist = np.unique(image[:, :, i], return_counts=True)
        vhist[hist[0]] = hist[1]
        vhist /= np.sum(vhist)
        ax[x, y].plot(vhist, c=rgb[i])

fig, ax = plt.subplots(6,3, figsize=(10,10))
chelsea = skimage.data.chelsea()
L = np.power(2, 8).astype(int)

#Zad 1 + 2
L_PTOZ = np.linspace(0, L-1, L).astype(int)
ax[0, 0].plot(L_PTOZ)
ax[0, 1].imshow(L_PTOZ[chelsea], cmap='binary_r')
hist(L_PTOZ[chelsea], 0, 2)

L_NEG = np.linspace(L-1, 0, L).astype(int)
ax[1, 0].plot(L_NEG)
ax[1, 1].imshow(L_NEG[chelsea], cmap='binary_r')
hist(L_NEG[chelsea], 1, 2)

L_PROG = np.zeros(L)
L_PROG[50:200] = L-1
L_PROG = L_PROG.astype(int)
ax[2, 0].plot(L_PROG)
ax[2, 1].imshow(L_PROG[chelsea], cmap='binary_r')
hist(L_PROG[chelsea], 2, 2)

L_SIN = np.linspace(0, 2 * np.pi, L)
lsinnorm = (np.sin(L_SIN) + 1) / 2
L_SIN = ((lsinnorm * L) - 1).astype(int)
ax[3,0].plot(L_SIN)
ax[3,1].imshow(L_SIN[chelsea], cmap='binary_r')
hist(L_SIN[chelsea], 3, 2)

base = np.arange(0, L)
firstGamma = 0.3
gamma = (np.power((base/(L-1)),(1/firstGamma)) * (L-1)).astype(int)
ax[4,0].plot(gamma)
ax[4,1].imshow(gamma[chelsea], cmap='binary_r')
hist(gamma[chelsea], 4, 2)

secondGamma = 3
secgamma = (np.power((base/(L-1)),(1/secondGamma)) * (L-1)).astype(int)
ax[5,0].plot(secgamma)
ax[5,1].imshow(secgamma[chelsea], cmap='binary_r')
hist(secgamma[chelsea], 5, 2)

plt.savefig('zad03.png')

# zadanie 3
fig, ax = plt.subplots(2,3, figsize=(10,10))
moon = skimage.data.moon()
L2 = np.power(2, 8).astype(int)

ax[0, 0].imshow(moon, cmap='binary_r')

vhist = np.zeros(L2)
hist = np.unique(moon, return_counts=True)
vhist[hist[0]] = hist[1]
vhist /= np.sum(vhist)
ax[0, 1].plot(vhist)

vdystryb = np.cumsum(vhist)
ax[0, 2].plot(vdystryb)

lut = (vdystryb * (L2-1)).astype(int)
ax[1, 0].plot(lut)

moon2 = lut[moon]
ax[1, 1].imshow(moon2, cmap='binary_r')

vhist = np.zeros(L2)
hist = np.unique(moon2, return_counts=True)
vhist[hist[0]] = hist[1]
vhist /= np.sum(vhist)
ax[1, 2].plot(vhist)


plt.savefig('zad03-3.png')
