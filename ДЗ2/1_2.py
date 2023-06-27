import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def conv_fast(image, kernel):
    rows, cols = image.shape
    ker_rows, ker_cols = kernel.shape
    kernel = np.flip(np.flip(kernel, 0), 1)
    result = np.zeros((rows, cols))
    image1 = np.zeros((rows+2*(ker_rows//2), cols+2*(ker_cols//2)))
    image1[ker_rows//2:rows+ker_rows//2, ker_cols//2:cols+ker_cols//2] = image
    for x in range(rows):
        for y in range(cols):
            result[x, y] = np.sum(image1[x:x+ker_rows,y:y+ker_cols]*kernel)
    return result

def correlation(image, kernel):
    kernel = np.flip(np.flip(kernel,0),1)
    result = conv_fast(image, kernel)
    return result

def zero_correlation(image, kernel):
    rows, cols = image.shape
    ker_rows, ker_cols = kernel.shape
    kernel = kernel-np.mean(kernel)
    result = np.zeros((rows, cols))
    image1 = np.zeros((rows+2*(ker_rows//2), cols+2*(ker_cols//2)))
    image1[ker_rows//2:rows+ker_rows//2, ker_cols//2:cols+ker_cols//2] = image
    for x in range(rows):
        for y in range(cols):
            result[x, y] = np.sum(image1[x:x+ker_rows,y:y+ker_cols]*kernel)
    return result


def zero_mean_correlation(image, kernel):
    rows, cols = image.shape
    ker_rows, ker_cols = kernel.shape
    kernel = (kernel-np.mean(kernel))/np.std(kernel)
    result = np.zeros((rows, cols))
    image1 = np.zeros((rows+2*(ker_rows//2), cols+2*(ker_cols//2)))
    image1[ker_rows//2:rows+ker_rows//2, ker_cols//2:cols+ker_cols//2] = image
    for x in range(rows):
        for y in range(cols):
            result[x, y] = np.sum((image1[x:x+ker_rows,y:y+ker_cols]-np.mean(image1[x:x+ker_rows,y:y+ker_cols]))/np.std(image1[x:x+ker_rows,y:y+ker_cols])*kernel)
    return result

image1 = cv2.imread('./ДЗ2/images/shelf_dark.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = cv2.imread('./ДЗ2/images/template.jpg')
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

kernel = image2
result = zero_mean_correlation(image1, kernel)

fig, axs = plt.subplots(1, 2, figsize = (15, 6))
ax1, ax2 = axs

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(image1.copy())
ax1.set_title('Начальное изображение', fontsize=15)

#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
ax2.imshow(result.copy())
ax2.set_title('Конечное изображение', fontsize=15)

plt.show()