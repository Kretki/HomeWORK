import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

def conv_fast(image, kernel):
    rows, cols = image.shape
    ker_rows, ker_cols = kernel.shape
    result = np.zeros((rows, cols))
    image1 = np.zeros((rows+2*(ker_rows//2), cols+2*(ker_cols//2)))
    image1[ker_rows//2:rows+ker_rows//2, ker_cols//2:cols+ker_cols//2] = image
    for x in range(rows):
        for y in range(cols):
            result[x, y] = np.sum(image1[x:x+ker_cols,y:y+ker_rows]*np.flip(kernel))
    return result

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


img = cv2.imread('./ДЗ2/images/dog.jpg', 0)

kernel = np.array(
[
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
], dtype=np.float32)

test_output = conv_fast(img, kernel)

fig, axs = plt.subplots(1, 1, figsize = (10, 4))
ax1= axs

ax1.imshow(test_output.copy())

plt.show()