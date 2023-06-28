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


# kernel = np.array([
#     [0, 0, 1, 0, 0],
#     [0, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1],
#     [0, 1, 1, 1, 0],
#     [0, 0, 1, 0, 0],
# ], np.uint8)
# dilation = cv2.dilate(closing, kernel, iterations=1)


#res = cv2.bitwise_and(image, image, mask = dilation)

#edges = cv2.Canny(erosion, 100, 200)

p_l = np.percentile(image, 2)
p_h = np.percentile(image, 95)
image_rescale = skimage.exposure.rescale_intensity(image, in_range=(p_l, p_h))
hist = cv2.calcHist([image_rescale.ravel()], [0], None, [256], [0, 256])
#plt.plot(hist)

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy(), 'gray')
ax1.set_title('res', fontsize=15)

ax2.imshow(closing.copy(), 'gray')
ax2.set_title('res', fontsize=15)

plt.show()