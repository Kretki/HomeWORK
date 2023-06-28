import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

image  = cv2.imread('./HW3/images/segmentation/noise.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5, 5), np.uint8)
blur = cv2.GaussianBlur(image, (5,5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
], np.uint8)
dilation = cv2.dilate(closing, kernel, iterations=1)


res = cv2.bitwise_and(image, image, mask = dilation)

edges = cv2.Canny(res, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy(), 'gray')
ax1.set_title('res', fontsize=15)

ax2.imshow(res.copy(), 'gray')
ax2.set_title('res', fontsize=15)

plt.show()