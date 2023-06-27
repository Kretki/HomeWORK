import cv2
import numpy as np
import matplotlib.pyplot as plt

def inc_nearby_cells(matrix, j, i, path):
    if(matrix[i-1][j]==0): matrix[i-1][j]==matrix[i][j]+1; path.append([j, i-1])
    if(matrix[i][j-1]==0): matrix[i][j-1]==matrix[i][j]+1; path.append([j-1, i])
    if(matrix[i][j+1]==0): matrix[i][j+1]==matrix[i][j]+1; path.append([j+1, i])
    if(matrix[i+1][j]==0): matrix[i+1][j]==matrix[i][j]+1; path.append([j, i+1])
    return matrix, path

image = cv2.imread('./ДЗ1/image/image1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

matrix = image.astype('int64')
matrix[matrix < 1] = -1
matrix[matrix > 254] = 0

path = []
for i in range(len(matrix[0])):
    if matrix[0][i] == 0:
       path = [[i, 0]]
       break

st1 = -1
fin1 = 0
for i in range(len(matrix[len(matrix)-1])):
    if matrix[len(matrix)-1][i] == 0:
        if(st1 == -1): st1 = i
        else: fin1 = i

while all(i == 0 for i in matrix[len(matrix)-1][st1:fin1]):
    for i in range(len(path)):
        matrix, path = inc_nearby_cells(matrix, path[i][0], path[i][1], path)
    if(len(path)%100 == 0):
        print(len(path))