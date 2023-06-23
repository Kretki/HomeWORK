import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def conv_fast_iterated(img, kernel, x, y, result):
    offset = len(kernel)//2
    img1 = img[x-offset:x+offset+1, y-offset:y+offset+1]
    result[x][y]+=np.sum(np.multiply(img1, kernel))
    return result

def conv_fast(img, kernel, result): #нужно сделать доп пустые пиксели по краям, потом обрезать их и подумать как ускорить
    kernel = np.flip(kernel)
    for i in kernel:
        i = np.flip(i)
    offset = len(kernel)//2
    for i in range(offset, len(img)-offset):
        for j in range(offset, len(img[i])-offset):
            result = conv_fast_iterated(img, kernel, i, j, result)
    return result

image1 = cv2.imread('./ДЗ2/images/shelf.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread('./ДЗ2/images/template.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

kernel = image2

result = np.zeros(image1.shape)

t1 = time()
result = conv_fast(image1, kernel, result)
t2 = time()
print(f'секунд - {t2 - t1}')

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

image1 = image1.astype('uint8')
#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(image1.copy())
ax1.set_title('Начальное изображение', fontsize=15)

result = result.astype('uint8')
#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
ax2.imshow(result.copy())
ax2.set_title('Конечное изображение', fontsize=15)

plt.show()