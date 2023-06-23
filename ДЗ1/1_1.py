import cv2
import numpy as np
import matplotlib.pyplot as plt

def wave(mat, path, fin):
    for point in path:
        if point == fin: break
        ret = 0
        if point[0]-1 >= 0:
            if(mat[point[0]-1][point[1]]==0): mat[point[0]-1][point[1]]=mat[point[0]][point[1]]+1; path.append([point[0]-1, point[1]]); ret = 1
        if point[0]+1 < len(mat[0]):
            if(mat[point[0]+1][point[1]]==0): mat[point[0]+1][point[1]]=mat[point[0]][point[1]]+1; path.append([point[0]+1, point[1]]); ret = 1
        if point[1]-1 >= 0:
            if(mat[point[0]][point[1]-1]==0): mat[point[0]][point[1]-1]=mat[point[0]][point[1]]+1; path.append([point[0], point[1]-1]); ret = 1
        if point[1]+1 < len(mat[0]):
            if(mat[point[0]][point[1]+1]==0): mat[point[0]][point[1]+1]=mat[point[0]][point[1]]+1; path.append([point[0], point[1]+1]); ret = 1
        if ret == 0: path.remove(point)
    return mat, path

def back_wave(mat, point):
    if mat[point[0]][point[1]] == 1: 
        mat[point[0]][point[1]] = -10
        return mat, point
    if point[0]-1 >= 0:
        if mat[point[0]-1][point[1]]==mat[point[0]][point[1]]-1: 
            mat[point[0]][point[1]] = -10
            return mat, [point[0]-1, point[1]]
    if point[0]+1 < len(mat[0]):
        if mat[point[0]+1][point[1]]==mat[point[0]][point[1]]-1:
            mat[point[0]][point[1]] = -10
            return mat, [point[0]+1, point[1]]
    if point[1]-1 >= 0:
        if mat[point[0]][point[1]-1]==mat[point[0]][point[1]]-1:
            mat[point[0]][point[1]] = -10
            return mat, [point[0], point[1]-1]
    if point[1]+1 < len(mat[0]):
        if mat[point[0]][point[1]+1]==mat[point[0]][point[1]]-1:
            mat[point[0]][point[1]] = -10
            return mat, [point[0], point[1]+1]
    return mat, point
    
        

image = cv2.imread('./ДЗ1/image/image1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

matrix = image.astype('int64')
matrix[matrix < 1] = -1
matrix[matrix > 254] = 0

rows, cols = matrix.shape

st = -99
fin = 0
sqarelen = cols+1
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if matrix[i][j] == -1:
            if st == -99: st = j
            else: 
                if matrix[i][j-1]!=-1: fin = j; break
    sqarelen = min(sqarelen, fin-st)
    st = -99
    fin = 0
matrix_size = cols//sqarelen
matrix1 = np.ndarray((2*matrix_size+1,2*matrix_size+1), dtype='int64')
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if i%(sqarelen/2)==0 and j%(sqarelen/2)==0:
            matrix1[int(i//(sqarelen/2))][int(j//(sqarelen/2))] = matrix[i][j]


st1 = -1
for i in range(len(matrix1[0])):
    if matrix1[0][i] == 0: st1 = i; break

fin1 = -1
for i in range(len(matrix1[len(matrix1)-1])):
    if matrix1[len(matrix1)-1][i] == 0: fin1 = i; break

matrix1[0][st1] = 1
paths = [[0, st1]]


while matrix1[len(matrix1)-1][fin1] == 0:
    matrix1, paths = wave(matrix1, paths, [len(matrix1)-1, fin1])
point = [len(matrix1)-1, fin1]
while matrix1[0][st1] != -10:
    matrix1, point = back_wave(matrix1, point)

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs

image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
ax1.imshow(image.copy())
ax1.set_title('Начальное изображение', fontsize=15)

matrix1[matrix1 >= 0] = 255
matrix1[matrix1 == -10] = 50
matrix1[matrix1 == -1] = 0
matrix1 = matrix1.astype('uint8')
matrix1 = cv2.cvtColor(matrix1, cv2.COLOR_GRAY2RGB)

ax2.imshow(matrix1)
ax2.set_title('Матрица', fontsize=15)

plt.show()
