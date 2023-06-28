import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2
def find_piece(img1, img2):
    lead = 0
    if img1.shape[0]<img2.shape[0]:
        temp = img1
        img1 = img2
        img2 = temp
        lead = 1
    #img1 - самая большая по строкам
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)
    max_cor = -1
    place = (0,0,0,0)

    CONST_CORR = 20

    for i in range(rows1-rows+1):
        for j in range(cols-CONST_CORR):
            cor = 0
            for k in range(3):
                img1_cut = img1[i:rows+i, cols1-CONST_CORR:cols1,:]#справа отрезаем
                img2_cut = img2[:,j:j+CONST_CORR,:]#слева отрезаем
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if (max_cor<cor):
                    max_cor = cor
                    place = (i, j, 0, lead)
            cor = 0
            for k in range(3):
                img1_cut = img1[i:rows+i, :CONST_CORR,:]#слева отрезаем
                img2_cut = img2[:,cols2-j-CONST_CORR:cols2-j,:]#справа отрезаем
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if max_cor<cor:
                    max_cor = cor
                    place = (i, j, 1, lead)
    
    lead = 0
    if img1.shape[1]<img2.shape[1]:
        temp = img1
        img1 = img2
        img2 = temp
        lead = 1
    #img1 - самая большая по стобцам
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)

    for j in range(cols1-cols+1):
        for i in range(rows-CONST_CORR):
            cor = 0
            for k in range(3):
                img1_cut = img1[rows1-CONST_CORR:rows1, j:cols+j,:]#снизу отрезаем
                img2_cut = img2[i:i+CONST_CORR,:,:]#сверху отрезаем
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if (max_cor<cor):
                    max_cor = cor
                    place = (i, j, 2, lead)
            cor = 0
            for k in range(3):
                img1_cut = img1[:CONST_CORR, j:cols+j,:]#сверху отрезаем
                img2_cut = img2[rows2-i-CONST_CORR:rows2-i,:,:]#снизу отрезаем
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if max_cor<cor:
                    max_cor = cor
                    place = (i, j, 3, lead)
    print(max_cor)
    return place

def connect_images(img1, img2, place):
    if place[3] == 1:
        temp = img1
        img1 = img2
        img2 = temp
    #img1 - самая большая по строкам
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    CONST_CORR = 20

    rows_max = max(rows1, rows2)
    cols_max = max(cols1, cols2)
    
    rows_min = min(rows1, rows2)
    cols_min = min(cols1, cols2)

    if place[2] == 0:
        result = np.zeros((rows1, cols1-CONST_CORR+cols2-place[1], 3), dtype=np.int64)
        result[:rows1, :cols1, :] = img1
        result[place[0]:rows2+place[0], place[1]:place[1]+cols2, :] = img2
    elif place[2] == 1:
        result = np.zeros((rows1, cols1-CONST_CORR+cols2-place[1], 3), dtype=np.int64)
        result[place[0]:place[0]+rows2, :cols2, :] = img2
        result[:rows1, cols2-place[1]-CONST_CORR:cols2-place[1]+cols1-CONST_CORR, :] = img1
    elif place[2] == 2:
        result = np.zeros((rows1-CONST_CORR-place[0]+rows2, cols1, 3), dtype=np.int64)
        result[:rows1, :cols1, :] = img1
        result[rows1-CONST_CORR:rows1-CONST_CORR+rows2, place[1]:place[1]+cols2, :] = img2
    else:
        result = np.zeros((rows1-CONST_CORR-place[0]+rows2, cols1, 3), dtype=np.int64)
        result[:rows2, place[1]:place[1]+cols2, :] = img2
        result[place[0]:place[0]+rows1, :cols1, :] = img1
    return result


img1 = cv2.imread('./HW3/images/china_street_cut/image_part_04.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('./HW3/images/china_street_cut/image_part_05.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3 = cv2.imread('./HW3/images/china_street_cut/image_part_06.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

place = find_piece(img3, img2)
print(place)
result = connect_images(img3, img2, place)

#place = find_piece(result, img3)
#print(place)
#result = connect_images(result, img3, place)

fig, axs = plt.subplots(1, 3, figsize = (10, 4))
ax1, ax2, ax3 = axs

ax1.imshow(img3.copy())
ax1.set_title('img1', fontsize=15)

ax2.imshow(img2.copy())
ax2.set_title('img2', fontsize=15)

ax3.imshow(result.copy())
ax3.set_title('res', fontsize=15)
plt.show()