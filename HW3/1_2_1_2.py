import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def find_piece(image1, image2):
    lead = 0
    core_row1, core_col1, _ = image1.shape
    core_row2, core_col2, _ = image2.shape
    if core_row1<core_row2:
        temp = image1
        img1 = image2
        img2 = temp
        lead = 1
    else:
        img1 = image1
        img2 = image2
    #теперь img1 - самое большое по строкам
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)
    
    CORR_COST=20

    max_cor = [-1]*4
    cor_place = [(0,0,0,0)]*4
    for i in range(rows1-rows+1):
        for j in range(CORR_COST,3*(cols//4)):
            img1_cut = img1[i:rows+i, cols1-CORR_COST:cols1, :]#Справа отрезаем полосу
            img2_cut = img2[:,j-CORR_COST:j, :]#Слева отрезаем полосу
            cor = 0
            for k in range(0, 3):
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[0]:
                max_cor[0] = cor
                cor_place[0] = (i, cols+j-CORR_COST, lead, 0)
            cor = 0
            img1_cut = img1[i:rows+i, :CORR_COST, :]#Слева отрезаем полосу
            img2_cut = img2[:,cols2-CORR_COST-j:cols2-j, :]#Справа отрезаем полосу
            for k in range(0, 3):
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[1]:
                max_cor[1] = cor
                cor_place[1] = (i, j+CORR_COST, lead, 1)#CORR_COST??

    lead = 0
    if core_col1<core_col2:
        temp = image1
        img1 = image2
        img2 = temp
        lead = 1
    else:
        img1 = image1
        img2 = image2
    #теперь img1 - самое большое по столбцам
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape
    
    cols = min(cols1, cols2)
    rows = min(rows1, rows2)
    for i in range(cols1-cols+1):
        for j in range(CORR_COST, 3*(rows//4)):
            img1_cut = img1[rows1-CORR_COST:rows1, i:cols+i, :]#снизу отрезаем
            img2_cut = img2[j-CORR_COST:j, :, :]#Сверху отрезаем
            cor = 0
            for k in range(0, 3):
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[2]:
                max_cor[2] = cor
                cor_place[2] = (rows+j-CORR_COST, i, lead, 2)
            img1_cut = img1[:CORR_COST, i:cols+i, :]#Сверху отрезаем
            img2_cut = img2[rows-j-CORR_COST:rows-j, :, :]#Снизу отрезаем
            cor = 0
            for k in range(0, 3):
                hist1 = cv2.calcHist([img1_cut.copy()], [k], None, [256], [0,256])
                hist2 = cv2.calcHist([img2_cut.copy()], [k], None, [256], [0,256])
                hist1 /= hist1.max()
                hist2 /= hist2.max()
                cor += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            if cor>max_cor[3]:
                max_cor[3] = cor
                cor_place[3] = (j+CORR_COST, i, lead, 3)
    print(max_cor)
    for k in range(4):
        if max_cor[k] == max(max_cor): return cor_place[k], max_cor[k]

def connect_images(img1, img2, place):
    if place[2] == 1:
        temp = img1
        img1 = img2
        img2 = temp

    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    if place[3] == 0:#к 1 добавляем cправа
        result = np.zeros((rows1, place[1]+cols2, 3), dtype=np.uint8)
        result[:rows1, :cols1, :] = img1
        result[place[0]:place[0]+rows2, place[1]:place[1]+cols2, :] = img2
    elif place[3] == 1:
        result = np.zeros((rows1, cols2-place[1]+cols1, 3), dtype=np.uint8)
        result[place[0]:place[0]+rows2, :cols2, :] = img2
        result[:rows1, cols2-place[1]:cols2-place[1]+cols1, :] = img1
    elif place[3] == 2:
        result = np.zeros((place[0]+rows2,cols1, 3), dtype=np.uint8)
        result[:rows1, :cols1, :] = img1
        result[place[0]:place[0]+rows2, place[1]:place[1]+cols2] = img2
    else:
        result = np.zeros((rows1-place[0]+rows2, cols1, 3), dtype=np.uint8)
        result[:rows2, place[1]:place[1]+cols2, :] = img2
        result[rows2-place[0]:rows2-place[0]+rows1, :cols1, :] = img1
    return result

def compete_images(images):
    images_bw = [None]*len(images)
    for i in range(len(images)):
        images_bw[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    result = images[0]
    for i in range(1, len(images)):
        #res_bw = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        place, cor = find_piece(result, images[i])
        if cor>2.8:
            result = connect_images(result, images[i], place)
    return result

images = []
for i in [4, 5, 3]:
    images.append(cv2.imread(f'./HW3/images/china_street_cut/image_part_0{i}.jpg'))

result = compete_images(images)


fig, axs = plt.subplots(1, 1, figsize = (10, 4))
ax1 = axs

ax1.imshow(result.copy())
ax1.set_title('res', fontsize=15)
plt.show()