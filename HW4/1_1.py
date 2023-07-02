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

image1 = cv2.imread(f'./HW4/images/whale_tail/test_image_00.jpg')

gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, ksize=(7, 7), sigmaX=2, sigmaY=2)
cv2.line(gray_img, (995, 107), (1007, 107), (0,0,0), thickness=3)

edges = skimage.feature.canny(gray_img, sigma=2)
edges = edges.astype('uint8')
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont = []
for cont in contours:
    if cont.shape[0] > 360:
        new_cont.append(cont)
res = np.zeros((500, 1100))#found good
res = res.astype('uint8')
cv2.drawContours(res, new_cont, -1, (255,0,0), 3)

h,w = res.shape
pts1 = np.float32([[24, 102], [474, 412], [1007, 50]])
pts2 = np.float32([[5, 50], [500, 495], [1085, 5]])#good transformation for _00
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(res, M, (w, h))
rows, cols = gray_img.shape
gray_img_1 = np.zeros((500, 1100), dtype=np.uint8)
gray_img_1[:rows, :cols] = gray_img
dst1 = cv2.warpAffine(gray_img_1, M, (1100, 500))


contours1, hierarchy1 = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont = []
for cont in contours1:
    if cont.shape[0] > 360:
        new_cont.append(cont)
#So, now we have good contour from 00 pic

image2 = cv2.imread(f'./HW4/images/whale_tail/test_image_06.jpg')
dst2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(dst2, ksize=(7, 7), sigmaX=2, sigmaY=2)
p_l = np.percentile(gray2, 2)
p_h = np.percentile(gray2, 95)
gray2_contrast = skimage.exposure.rescale_intensity(gray2, in_range=(p_l, p_h))

edges3 = skimage.feature.canny(gray2_contrast, sigma=0.55)
edges3 = edges3.astype('uint8')
contours3, hierarchy3 = cv2.findContours(edges3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont3 = []
for cont in contours3:
    if cont.shape[0] > 50:
        new_cont3.append(cont)
res3 = np.zeros((500, 1100))#found good
res3 = res3.astype('uint8')
cv2.drawContours(res3, new_cont3, -1, (255,0,0), 3)

d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
print(d1, d2, d3)

fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(image1.copy())
ax1.set_title('image', fontsize=15)

ax2.imshow(dst1.copy(), 'gray')
ax2.set_title('res', fontsize=15)

ax3.imshow(res3.copy(), 'gray')
ax3.set_title('res2', fontsize=15)

plt.show()