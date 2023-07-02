import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage

def connect_start_and_end(contours):
    min_range = 999999
    cords = []
    for i in contours[0]:
        for j in contours[1]:
            if min_range>np.sqrt(np.square(i[0][0]-j[0][0])+np.square(i[0][1]-j[0][1])):
                min_range = np.sqrt(np.square(i[0][0]-j[0][0])+np.square(i[0][1]-j[0][1]))
                cords = [i[0], j[0]]
    return cords

image = cv2.imread(f'./HW4/images/whale_tail/test_image_00.jpg')

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, ksize=(7, 7), sigmaX=2, sigmaY=2)
cv2.line(gray_img, (995, 107), (1007, 107), (0,0,0), thickness=3)

edges = skimage.feature.canny(gray_img, sigma=2)
# ## выделяем границы
# laplac = cv2.Laplacian(gray_img, cv2.THRESH_BINARY, scale=0.55, ksize=5)
edges = edges.astype('uint8')
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont = []
for cont in contours:
    if cont.shape[0] > 360:
        new_cont.append(cont)
cv2.drawContours(edges, new_cont, -1, (255,0,0), 3)#, cv2.LINE_AA, hierarchy, 1)
contours1, hierarchy1 = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont = []
for cont in contours1:
    if cont.shape[0] > 360:
        new_cont.append(cont)
res = np.zeros_like(edges)
M = cv2.moments(new_cont[0])
print(M)#found good

res = res.astype('uint8')
cv2.drawContours(res, new_cont, -1, (255,0,0), 3)#, cv2.LINE_AA, hierarchy, 1)
fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

ax1.imshow(image.copy())
ax1.set_title('image', fontsize=15)

ax2.imshow(edges.copy(), 'gray')
ax2.set_title('res', fontsize=15)

plt.show()