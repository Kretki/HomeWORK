import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def find_cells(image, size, quantity):
    for i in range(len(image)-1, -1, -1):
        if np.sum(image[i]) == 0:
            image = np.delete(image, i, 0)
    rows, cols = image.shape
    pos = [(0,0)]*quantity
    for k in range(quantity):
        max_intens = 0
        for i in range(rows-size):
            for j in range(cols-size):
                img_cut = image[i:i+size, j:j+size]
                if np.sum(img_cut) == 0: break
                hist1 = cv2.calcHist([img_cut.ravel()], [0], None, [256], [0,256])
                hist1 /= hist1.max()
                intens = np.sum(hist1)
                if intens>max_intens:
                    max_intens = intens
                    pos[k] = (i,j)
                    print(intens, pos[k], image.shape)
        image = np.delete(image, range(i, i+size), 0)
    return pos

def cut_images(image, pos, size):
    images = np.zeros((size*(len(pos)//2+1), size*(len(pos)//2+1), 3))
    for i in range(len(pos)):
        images[(i//(size//2))*size:(i//(size//2))*size+size, (i%(size//2))*size:(i%(size//2))*size+size, :] = image[pos[i][0]:pos[i][0]+size, pos[i][1]:pos[i][1]+size, :]
    return images

image = cv2.imread(f'./HW3/images/cells/train0_1.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#result = compete_images(images)
kernel = np.ones((30, 30), np.uint8)
erode = cv2.erode(image, kernel, iterations=1)
dilate = cv2.dilate(image, kernel, iterations=1)
gradient = dilate - erode

size = 400; quantity = 1

pos = find_cells(gradient, size, quantity)
print('ok')
result = cut_images(image, pos, size)


fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy())
ax1.set_title('res', fontsize=15)

ax2.imshow(image.copy())
ax2.set_title('res', fontsize=15)

plt.show()