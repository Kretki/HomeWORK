from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


# for i in range(rows-size):
#             for j in range(cols-size):
#                 img_cut = image[i:i+size, j:j+size]
#                 if np.sum(img_cut) == 0: break
#                 hist1 = cv2.calcHist([img_cut.ravel()], [0], None, [256], [0,256])
#                 hist1 /= hist1.max()
#                 intens = np.sum(hist1)
def cut_corners(image):
    for i in range(len(image)-1, -1, -1):
        if np.sum(image[i]) == 0:
            image = np.delete(image, i, 0)
    image = image.T
    for i in range(len(image)-1, -1, -1):
        if np.sum(image[i]) == 0:
            image = np.delete(image, i, 0)
    image = image.T
    return image

def find_max_zone(image, k):
    rows, cols = image.shape
    if k == 0:
        res1 = (0, rows//2, 0, cols)
        res2 = (rows//2, rows, 0, cols)
    else:
        res1 = (0, rows, 0, cols//2)
        res2 = (0, rows, cols//2, cols)
    if np.sum(image[res1[0]:res1[1], res1[2]:res1[3]])>np.sum(image[res2[0]:res2[1], res2[2]:res2[3]]): return res1
    else: return res2

def find_max_precise(image, size):
    rows, cols = image.shape
    max_intens = 0
    cords = (0,0,0,0)
    for i in range(rows-size):
        for j in range(cols-size):
            img_cut = image[i:i+size, j:j+size]
            intens = np.sum(img_cut)
            if intens > max_intens:
                max_intens = intens
                cords = (i, i+size,j, j+size)
    return cords

def cut_one_image(image, size):
    rows, cols = image.shape
    cut_cords = [0, rows, 0, cols]
    while cut_cords[1]-cut_cords[0]>size*2:
        cut_cords_1 = find_max_zone(image[cut_cords[0]:cut_cords[1], cut_cords[2]:cut_cords[3]], 0)
        cut_cords[1] = cut_cords[0]+cut_cords_1[1]
        cut_cords[0] = cut_cords[0]+cut_cords_1[0]
    while cut_cords[3]-cut_cords[2]>size*2:
        cut_cords_1 = find_max_zone(image[cut_cords[0]:cut_cords[1], cut_cords[2]:cut_cords[3]], 1)
        cut_cords[3] = cut_cords[2]+cut_cords_1[3]
        cut_cords[2] = cut_cords[2]+cut_cords_1[2]
    cut_cords_1 = find_max_precise(image[cut_cords[0]:cut_cords[1], cut_cords[2]:cut_cords[3]], size)
    cut_cords[1] = cut_cords[0]+cut_cords_1[1]
    cut_cords[0] = cut_cords[0]+cut_cords_1[0]
    cut_cords[3] = cut_cords[2]+cut_cords_1[3]
    cut_cords[2] = cut_cords[2]+cut_cords_1[2]
    result = cut_cords
    return result
    
def cut_image(image, size, number):

    imageGR = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    max_values = np.iinfo(imageGR.dtype).max
    imageGR ^= max_values
    rows = int(np.ceil(np.sqrt(number)))
    result = np.zeros((rows*size, rows*size, 3), dtype=np.uint8)
    for i in range(number):
        cords = cut_one_image(imageGR, size)
        result[(i//rows)*size:(i//rows+1)*size, (i%rows)*size:(i%rows+1)*size, :] = image[cords[0]:cords[1], cords[2]:cords[3], :]
        imageGR[cords[0]:cords[1], cords[2]:cords[3]] = np.zeros((cords[1]-cords[0], cords[3]-cords[2]), dtype=np.uint8)
    return result

image = cv2.imread(f'./HW3/images/cells/train0_1.jpeg')

res = cut_image(image, 400, 4)

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy())
ax1.set_title('image', fontsize=15)

ax2.imshow(res.copy())
ax2.set_title('res', fontsize=15)

plt.show()
