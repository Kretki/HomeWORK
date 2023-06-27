import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2

def find_piece(img1, img2):
    core_rows, core_cols = img1.shape
    rows, cols = img2.shape
    #max_inters = -1
    #inters_place = (0,0)
    max_cor = -1
    cor_place = (0,0)
    for i in range(core_cols - cols):
        for j in range(core_rows-rows):
            img1_cut = img1[j:rows+j, i:cols+i]
            hist1 = cv2.calcHist([img1_cut.ravel()], [0], None, [256], [0,256])
            hist2 = cv2.calcHist([img2.ravel()], [0], None, [256], [0,256])
            hist1 /= hist1.max()
            hist2 /= hist2.max()
            cor = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            #inters = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) / np.sum(hist1)
            #cor3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
            if cor>max_cor:
                max_cor = cor
                cor_place = (i,j)
            #if inters>max_inters:
            #    max_inters = inters
            #    inters_place = (i,j)
    #print(f'Correlation: {max_cor}\n'
    #    f'{cor_place} - place\n'
    #    f'Intersection: {max_inters}\n'
    #    f'{inters_place} is position')
    return cor_place#, inters_place

core_image = cv2.imread('./ДЗ3/images/china_street_cut/image.jpg')
core_image = cv2.cvtColor(core_image, cv2.COLOR_BGR2GRAY)

img1 = cv2.imread('./ДЗ3/images/china_street_cut/image_part_02.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

place = find_piece(core_image, img1)

rows, cols = img1.shape
result = np.zeros(core_image.shape)
result[place[0]:rows+place[0], place[1]:cols+place[1]] = img1
#t1 = time()
#result1 = zero_correlation(core_image[place1[1]:rows+place1[1], place1[0]:cols+place1[0]], img1)
#print(f'buf {time()-t1}') #итог - больше подходит корреляция

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

#image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(core_image.copy())
ax1.set_title('Начальное изображение', fontsize=15)

#result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
ax2.imshow(result.copy())
ax2.set_title('Correlation', fontsize=15)

plt.show()