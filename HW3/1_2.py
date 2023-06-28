import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2
import skimage

image  = cv2.imread('./HW3/images/segmentation/noise.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([image.ravel()], [0], None, [256], [0, 256])
#plt.plot(hist)
ind = np.argpartition(hist.ravel(), -3)[-3:]#3 max values
print(ind)

median = cv2.medianBlur(image, 7)

kernel = np.ones((3,3), np.uint8)
# blur = cv2.GaussianBlur(image, (5,5), 0)
# th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4)
ret3, th3 = cv2.threshold(median, ind[0], 255, cv2.THRESH_BINARY)# + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(th3, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)

blur = cv2.blur(dilation, (7,7), 0)

kernel = np.ones((11,11), np.uint8)
ret4, th4 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

for i in range(8):
    closing = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)
    blur = cv2.blur(closing, (7,7), 0)
    ret4, th4 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

ret3, th3 = cv2.threshold(median, ind[2], 255, cv2.THRESH_BINARY)
th3 = th3 - closing
closing2 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(closing2, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=1)

kernel = np.ones((3,3), np.uint8)
closing2 = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(image.copy(), 'gray')
ax1.set_title('res', fontsize=15)

ax2.imshow(closing.copy(), 'gray')
ax2.set_title('res', fontsize=15)

ax3.imshow(closing2.copy(), 'gray')
ax3.set_title('res', fontsize=15)
plt.show()