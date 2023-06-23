import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def conv_nested_iterated(img, kernel, x, y, result):
    for i in range(-(len(kernel)//2), len(kernel)//2+1):
        for j in range(-(len(kernel[i])//2), len(kernel[i])//2+1):
            result[x][y]+=img[x+i][y+j]*kernel[len(kernel)//2+i][len(kernel[i])//2+j]
    #result[x][y]=result[x][y]//(len(kernel)*len(kernel))
    return result

def conv_fast_iterated(img, kernel, x, y, result):
    offset = len(kernel)//2
    img1 = img[x-offset:x+offset+1, y-offset:y+offset+1]
    result[x][y]+=np.sum(np.multiply(img1, kernel))#//(len(kernel)*len(kernel))
    return result

def conv_nested(img, kernel, result):
    kernel = np.flip(kernel)
    for i in kernel:
        i = np.flip(i)
    offset = len(kernel)//2
    for i in range(offset, len(img)-offset):
        for j in range(offset, len(img[i])-offset):
            result = conv_nested_iterated(img, kernel, i, j, result)
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
], dtype=np.float32)
#kernel /= np.sum(kernel)
#np.insert(image, len(image)-1, np.zeros(len(image)), axis=1)
#image = np.zeros((9, 9))
#image[3:6, 3:6] = 50
result = np.zeros_like(image)

t1 = time()
#result = cv2.filter2D(image.copy(), -1, kernel)
result = conv_fast(image, kernel, result)
t2 = time()
print(f'секунд - {t2 - t1}')
print(result)

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(image.copy())
ax1.set_title('Начальное изображение', fontsize=15)

result = result.astype('uint8')
#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
ax2.imshow(result.copy())
ax2.set_title('Конечное изображение', fontsize=15)


#result1 = image-result
#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
#ax3.imshow(result1.copy())
#ax3.set_title('Конечное изображение', fontsize=15)

plt.show()