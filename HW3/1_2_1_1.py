import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def find_piece(img1, img2):
    lead = 0
    if len(img1)<len(img2):
        temp = img1
        img1 = img2
        img2 = temp
        lead = 1
    #теперь img1 - самое большое по строкам
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)

    max_cor = [-1]*4
    cor_place = [(0,0,0,0)]*4
    for i in range(rows1-rows):
        for j in range(5, cols//2):
            img1_cut = img1[i:rows+i, (cols//2)+j:cols1]#Справа отрезаем полосу
            img2_cut = img2[:,:j]#Слева отрезаем полосу
            hist1 = cv2.calcHist([img1_cut.ravel()], [0], None, [256], [0,256])
            hist2 = cv2.calcHist([img2_cut.ravel()], [0], None, [256], [0,256])
            hist1 /= hist1.max()
            hist2 /= hist2.max()
            cor = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[0]:
                max_cor[0] = cor
                cor_place[0] = (i, cols//2+j, lead, 0)
            img1_cut = img1[i:rows+i, :j]#Слева отрезаем полосу
            img2_cut = img2[:,(cols//2)+j:cols2]#Справа отрезаем полосу
            hist1 = cv2.calcHist([img1_cut.ravel()], [0], None, [256], [0,256])
            hist2 = cv2.calcHist([img2_cut.ravel()], [0], None, [256], [0,256])
            hist1 /= hist1.max()
            hist2 /= hist2.max()
            cor = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[1]:
                max_cor[1] = cor
                cor_place[1] = (i, j, lead, 1)
    res_cor = (0,0,0,0)
    print(max_cor)
    for k in range(4):
        if max_cor[k] == max(max_cor): res_cor = cor_place[k]
    return res_cor


def connect_images(img1, img2, place):
    if place[2] == 1:
        temp = img1
        img1 = img2
        img2 = temp
    #img1 - самое большое по строкам

    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape


    if place[3] == 0:#к 1 добавляем слева
        result = np.zeros((rows1, place[1]+cols1, 3), dtype=np.int64)
        result[:rows1, place[1]:place[1]+cols1, :] = img1
        result[place[0]:place[0]+rows2, :cols2, :] = img2
    else:
        result = np.zeros((rows1, cols2-place[1]+cols1, 3), dtype=np.int64)
        result[place[0]:place[0]+rows2, :cols2, :] = img2
        result[:rows1, cols2-place[1]:cols2-place[1]+cols1, :] = img1
    return result


img1 = cv2.imread('./HW3/images/china_street_cut/image_part_04.jpg')
img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('./HW3/images/china_street_cut/image_part_05.jpg')
img2_bw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


place = find_piece(img1_bw, img2_bw)
print(place)
result = connect_images(img1, img2, place)


fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(img1.copy())
ax1.set_title('img1', fontsize=15)

ax2.imshow(img2.copy())
ax2.set_title('img2', fontsize=15)

ax3.imshow(result.copy())
ax3.set_title('res', fontsize=15)
plt.show()