#Zad 1
import matplotlib.pyplot as plt
import numpy

mono = numpy.zeros((30,30)).astype(int)

mono[10:20,10:20] = 1
mono[15:25,15:25] = 2

print(mono)


#Zad 2
fig, ax = plt.subplots(2,2, figsize=(7,7))

ax[0,0].imshow(mono)
ax[0,1].imshow(mono, cmap='binary')
ax[0,0].set_title('obraz monochromatyczny')
ax[0,1].set_title('obraz monochromatyczny')


#Zad 3

color = numpy.zeros((30,30,3)).astype(float)
color[15:25,5:15,0] = 1
color[10:20,10:20,1] = 1
color[5:15,15:25,2] = 1

ax[1,0].imshow(color)
negative = 1 - color
ax[1,1].imshow(negative)

ax[1,0].set_title('obraz barwny')
ax[1,1].set_title('obraz negatyw')

plt.tight_layout()
plt.savefig('test.png')