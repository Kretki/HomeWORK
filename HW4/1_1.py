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

#cv2.drawContours(dst1, new_cont, -1, (255,0,0),3)
#So, now we have good contour from 00 pic
#hard

def affine_rotate(image):
    M = cv2.moments(image)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])#whale center
    M1 = cv2.getRotationMatrix2D((Cx, Cy), 1, scale=1.0)
    dst2 = image.copy()
    d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
    d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
    d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
    j = True
    while j:
        dst2 = cv2.warpAffine(dst2.copy(), M1, (cols, rows))
        d1_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
        d2_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
        d3_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
        if d1_new>d1 and d2_new>d2 and d3_new>d3:
            j = False
    M1 = cv2.getRotationMatrix2D((Cx, Cy), -1, scale=1.0)
    d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
    d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
    d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
    j = True
    while j:
        dst2 = cv2.warpAffine(dst2.copy(), M1, (cols, rows))
        d1_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
        d2_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
        d3_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
        if d1_new>d1 and d2_new>d2 and d3_new>d3:
            j = False
    return dst2

def affine_perspective_rotate(image):
    M = cv2.moments(image)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])#whale center
    pts1 = np.float32([[Cx-50, Cy-50], [Cx-50, Cy+50],[Cx+50, Cy-50], [Cx+50, Cy+50]])
    pts2 = np.float32([[Cx-50, Cy-50], [Cx-50, Cy+50],[Cx+49, Cy-50], [Cx+49, Cy+50]])
    M1 = cv2.getPerspectiveTransform(pts1, pts2)
    dst2 = image.copy()
    d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
    d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
    d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
    j = True
    while j:
        dst2 = cv2.warpPerspective(dst2.copy(), M1, (cols, rows))
        d1_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
        d2_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
        d3_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
        if d1_new>d1 and d2_new>d2 and d3_new>d3:
            j = False
    pts1 = np.float32([[Cx-50, Cy-50], [Cx-50, Cy+50],[Cx+50, Cy-50], [Cx+50, Cy+50]])
    pts2 = np.float32([[Cx-49, Cy-50], [Cx-49, Cy+50],[Cx+50, Cy-50], [Cx+50, Cy+50]])
    M1 = cv2.getPerspectiveTransform(pts1, pts2)
    d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
    d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
    d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
    j = True
    while j:
        dst2 = cv2.warpPerspective(dst2.copy(), M1, (cols, rows))
        d1_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
        d2_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
        d3_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
        if d1_new>d1 and d2_new>d2 and d3_new>d3:
            j = False
    return dst2

def affine_perspective_resize(image):
    M = cv2.moments(image)
    Cx = int(M['m10'] / M['m00'])
    Cy = int(M['m01'] / M['m00'])#whale center
    pts1 = np.float32([[Cx-100, Cy-100], [Cx-100, Cy+100],[Cx+100, Cy-100], [Cx+100, Cy+100]])
    pts2 = np.float32([[Cx-101, Cy-100], [Cx-100, Cy+100],[Cx+100, Cy-100], [Cx+100, Cy+100]])
    M1 = cv2.getPerspectiveTransform(pts1, pts2)
    dst2 = image.copy()
    d1 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
    d2 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
    d3 = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
    i = 0
    j = True
    while j:
        dst2 = cv2.warpPerspective(dst2.copy(), M1, (cols, rows))
        d1_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I1,0)
        d2_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I2,0)
        d3_new = cv2.matchShapes(dst1, dst2, cv2.CONTOURS_MATCH_I3,0)
        if d1_new>d1 and d2_new>d2 and d3_new>d3:
            j = False
        if i>50:
            break
        i+=1
    return dst2


image2 = cv2.imread(f'./HW4/images/whale_tail/test_image_06.jpg')
dst2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.GaussianBlur(dst2, ksize=(7, 7), sigmaX=2, sigmaY=2)
p_l = np.percentile(gray2, 30)
p_h = np.percentile(gray2, 70)
gray2_contrast = skimage.exposure.rescale_intensity(gray2, in_range=(p_l, p_h))

edges3 = skimage.feature.canny(gray2_contrast, sigma=2)
edges3 = edges3.astype('uint8')
contours3, hierarchy3 = cv2.findContours(edges3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
new_cont3 = []
for cont in contours3:
    if cont.shape[0] > 150:
        new_cont3.append(cont)
res3 = np.zeros((500, 1100))
res3 = res3.astype('uint8')
i = 10
cv2.drawContours(res3, [new_cont3[i]], -1, (255,0,0), 3)

res4 = np.zeros((500, 1100))
res4 = res4.astype('uint8')
rows, cols = gray2_contrast.shape
pts1 = np.float32([[0, 0], [0, rows], [cols, 0]])
pts2 = np.float32([[0, 0], [0, 500], [1100, 0]])#good transformation for _00
M = cv2.getAffineTransform(pts1, pts2)
dst2 = cv2.warpAffine(gray2_contrast, M, (w, h))
res4 = dst2[:500, :1100]

res5 = affine_rotate(res4)
res6 = affine_perspective_rotate(res5)
res7 = affine_perspective_resize(res6)


d1 = cv2.matchShapes(dst1, res7, cv2.CONTOURS_MATCH_I1,0)
d2 = cv2.matchShapes(dst1, res7, cv2.CONTOURS_MATCH_I2,0)
d3 = cv2.matchShapes(dst1, res7, cv2.CONTOURS_MATCH_I3,0)
print(d1, d2, d3)

fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(image1.copy())
ax1.set_title('image', fontsize=15)

ax2.imshow(res6.copy(), 'gray')
ax2.set_title('res', fontsize=15)

ax3.imshow(res7.copy(), 'gray')
ax3.set_title('res2', fontsize=15)

plt.show()