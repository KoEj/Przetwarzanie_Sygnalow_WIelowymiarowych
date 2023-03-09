#Zad 1
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(3,3, figsize=(10,10))
x = np.linspace(0, 4*np.pi, num=40)
y = np.sin(x)

ax[0,0].plot(x,y)

newArray = y[:, np.newaxis]*y[np.newaxis, :]
ax[0,1].imshow(newArray, cmap='binary')
ax[0,1].set_title('min: ' + str(round(np.min(newArray), 3)) + ' max: ' + str(round(np.max(newArray), 3)))

newArrays = (newArray - np.min(newArray))
newArrays = newArrays / np.max(newArrays)
ax[0,2].imshow(newArrays, cmap='binary')
ax[0,2].set_title('min: ' + str(round(np.min(newArrays), 3)) + ' max: ' + str(round(np.max(newArrays), 3)))

# Zad 2
l1 = 2**2-1
l2 = 2**4-1
l3 = 2**8-1

w1 = np.rint(newArrays*l1)
w2 = np.rint(newArrays*l2)
w3 = np.rint(newArrays*l3)

ax[1,0].imshow(w1, cmap='binary')
ax[1,0].set_title('min: ' + str(int(np.min(w1))) + ' max: ' + str(int(np.max(w1))))
ax[1,1].imshow(w2, cmap='binary')
ax[1,1].set_title('min: ' + str(int(np.min(w2))) + ' max: ' + str(int(np.max(w2))))
ax[1,2].imshow(w3, cmap='binary')
ax[1,2].set_title('min: ' + str(int(np.min(w3))) + ' max: ' + str(int(np.max(w3))))


Param1 = 0
Param2 = 15
# Zad 3
newDs = newArrays + np.random.normal(Param1,Param2,size=np.shape(newArrays))
ax[2,0].imshow(newDs, cmap='binary')
ax[2,0].set_title('noised')

sum = newArrays
for x in range(50):
    newDs = newArrays + np.random.normal(Param1,Param2,size=np.shape(newArrays))
    sum = np.add(sum, newDs)
n50 = (sum - np.min(sum))
n50 = n50 / np.max(n50)
ax[2,1].imshow(n50, cmap='binary')
ax[2,1].set_title('n=50')

sum = newArrays
for x in range(100000):
    newDs = newArrays + np.random.normal(Param1,Param2,size=np.shape(newArrays))
    sum = np.add(sum, newDs)
n1000 = (sum - np.min(sum))
n1000 = n1000 / np.max(n1000)
ax[2,2].imshow(n1000, cmap='binary')
ax[2,2].set_title('n=1000')

plt.savefig('test.png')