import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def conv_fast_iterated(img, kernel, x, y, result):
    offset = len(kernel)//2
    img1 = img[x-offset:x+offset+1, y-offset:y+offset+1]
    result[x][y]=np.sum(np.multiply(img1, kernel))

def conv_fast(img, kernel, result):
    offset = len(kernel)//2
    for i in range(offset, len(img)-offset):
        for j in range(offset, len(img[i])-offset):
            conv_fast_iterated(img, kernel, i, j, result)

image = cv2.imread('./ДЗ2/images/Puppy.png')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = image.astype('uint8')

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

kernel = np.array(
[
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])

kernel = np.flip(kernel)
for i in kernel:
    i = np.flip(i)

result = np.zeros_like(image)
conv_fast(image, kernel, result)


fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy())
ax1.set_title('Начальное изображение', fontsize=15)

ax2.imshow(result.copy())
ax2.set_title('Конечное изображение', fontsize=15)

plt.show()