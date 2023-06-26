import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

def conv_nested(image, kernel):
    result = np.zeros_like(image)
    offset = len(kernel)//2
    for x in range(len(result)):
        for y in range(len(result[x])): 
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if 0<=x-i+offset<len(kernel) and 0<=y-j+offset<len(kernel):
                        result[x][y] += image[i][j]*kernel[x-i+offset][y-j+offset]
        if x%10 == 0:
            print(x, len(result))
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
])

# Create a test image: a white square in the middle
#test_img = np.zeros((9, 9))
#test_img[3:6, 3:6] = 1

# Run your conv_nested function on the test image
test_output = conv_nested(img, kernel)
test_output = test_output.astype('uint8')

fig, axs = plt.subplots(1, 1, figsize = (10, 4))
ax1= axs

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(test_output.copy())

plt.show()