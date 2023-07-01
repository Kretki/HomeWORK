import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2
from skimage import morphology

def fill_the_gaps(image):
    result = image
    rows, cols = result.shape
    for i in range(rows):
        ind = np.argwhere(result[i])
        if ind.shape[0] > 1:
            result[i, ind[0][0]:ind[-1][0]] = 255
    result = result.T
    for i in range(cols):
        ind = np.argwhere(result[i])
        if ind.shape[0] > 1:
            result[i, ind[0][0]:ind[-1][0]] = 255
    result = result.T
    return result

image  = cv2.imread('./HW3/images/segmentation/noise.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([image.ravel()], [0], None, [256], [0, 256])
#plt.plot(hist)
ind = np.argpartition(hist.ravel(), -3)[-3:]#3 max values
#print(ind)

median = cv2.medianBlur(image, 7)

kernel = np.ones((3,3), np.uint8)
# blur = cv2.GaussianBlur(image, (5,5), 0)
# th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4)
ret3, th3 = cv2.threshold(median, ind[0], 255, cv2.THRESH_BINARY)# + cv2.THRESH_OTSU)

imglab = morphology.label(th3)
cleaned = morphology.remove_small_objects(imglab, min_size=24, connectivity=2)
cleaned = cleaned.astype('uint8')
ret4, th4 = cv2.threshold(cleaned, 1, 255, cv2.THRESH_BINARY)
res1 = fill_the_gaps(th4)

kernel = np.ones((5,5), np.uint8)
dilation1 = cv2.dilate(res1, kernel, iterations=2)
dilation1b = dilation1.astype('bool')
# blur = cv2.blur(dilation, (7,7), 0)

# kernel = np.ones((11,11), np.uint8)
# ret4, th4 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

# for i in range(8):
#     closing = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)
#     blur = cv2.blur(closing, (7,7), 0)
#     ret4, th4 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

ret5, th5 = cv2.threshold(median, ind[2], 255, cv2.THRESH_BINARY)

kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th5, kernel, iterations=1)

th3b = th3.astype('bool')
th5b = th5.astype('bool')
th5b = np.logical_and(erosion, np.logical_not(th3b))
th5b = np.logical_and(np.logical_not(dilation1b), th5b)

th5 = th5b.astype('uint8')

kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th5, kernel, iterations=1)

imglab2 = morphology.label(erosion)
cleaned2 = morphology.remove_small_objects(imglab2, min_size=24, connectivity=2)
cleaned2 = cleaned2.astype('uint8')
res2 = fill_the_gaps(cleaned2)
closing2 = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)


# dilation = cv2.dilate(erosion, kernel, iterations=1)

# kernel = np.ones((3,3), np.uint8)
# closing2 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(image.copy(), 'gray')
ax1.set_title('res', fontsize=15)

ax2.imshow(dilation1b.copy(), 'gray')
ax2.set_title('res', fontsize=15)

ax3.imshow(closing2.copy(), 'gray')
ax3.set_title('res', fontsize=15)
plt.show()