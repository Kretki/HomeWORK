import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./ДЗ1/image/image2_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

height, width, _ = image.shape


hsv_low_border = (0.04 * 360, 50, 100)
hsv_high_border = (0.08 * 360, 255, 255)

borders = cv2.inRange(image_hsv, hsv_low_border, hsv_high_border)
borders1 = borders[height//2:height//2+1, :]

prev_white = -999
count_borders = 0
borders_width = []
for i in range(len(borders1[0])):
    if borders1[0][i] != 0:
        if(prev_white+25<i): 
            if(count_borders!=0): 
                if(prev_white!=-999): borders_width.append([prev_white])
                else: borders_width.append([0])
                borders_width[count_borders-1].append(i)
            count_borders+=1
        prev_white=i
open_lines = [1 for i in range(count_borders-1)]
#print(borders_width, open_lines)


hsv_low_obstacles = (0.0 * 360, 50, 100)
hsv_high_obstacles = (0.03 * 360, 255, 255)

obstacles = cv2.inRange(image_hsv, hsv_low_obstacles, hsv_high_obstacles)

for j in range(len(obstacles)):
    for i in range(len(obstacles[j])):
        if obstacles[j][i] != 0:
            for k in range(count_borders-1):
                if borders_width[k][0] < i < borders_width[k][1]:
                    open_lines[k] = 0
#print(open_lines)
for i in range(count_borders-1):
    if open_lines[i] == 1:
        print(f'Нужно перестроиться на дорогу номер {i+1}')