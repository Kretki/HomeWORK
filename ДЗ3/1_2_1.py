import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def find_piece(img1, img2):
    rows1, cols1 = img1.shape
    rows2, cols2 = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)
    max_cor = -1
    cor_place = (0,0)
    if rows == rows1:#left-right check
        for i in range(rows2-rows):
            for j in range(5,cols//2):
                img1_cut = img1[:, (cols//2)+j:cols1]
                img2_cut = img2[i:rows+i,:j]
                hist1 = cv2.calcHist([img2_cut.ravel()], [0], None, [256], [0,256])
                hist2 = cv2.calcHist([img1_cut.ravel()], [0], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if cor>max_cor:
                    max_cor = cor
                    cor_place = (i, cols//2+j)
                #    print(i, j, cor, cor_place)
    return cor_place

def connect_images(img1, img2, place):
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    rows_max = max(rows1, rows2)
    cols_max = max(cols1, cols2)
    
    rows_min = min(rows1, rows2)
    cols_min = min(cols1, cols2)

    result = np.zeros((rows_max, place[1]+cols2, 3), dtype=np.int64)
    result[place[0]:place[0]+rows1, :cols1, :] = img1
    #print(result, 'so\n', img1)
    result[:rows2, place[1]:place[1]+cols2, :] = img2

    return result


img1 = cv2.imread('./ДЗ3/images/china_street_cut/image_part_04.jpg')
img1_bw = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('./ДЗ3/images/china_street_cut/image_part_05.jpg')
img2_bw = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


place = find_piece(img1_bw, img2_bw)
result = connect_images(img1, img2, place)


fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(img1.copy())
ax1.set_title('img1', fontsize=15)

#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
ax2.imshow(img2.copy())
ax2.set_title('img2', fontsize=15)

ax3.imshow(result.copy())
ax3.set_title('res', fontsize=15)
plt.show()