import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./ДЗ1/image/RGB_cube.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rows, cols, _ = image.shape

fig, axs = plt.subplots(1, 2, figsize = (10, 4))
ax1, ax2 = axs


pts1 = np.float32([[50, 50], [400, 50], [50, 200]])
pts2 = np.float32([[100, 100], [200, 20], [100, 250]])

M1 = cv2.getAffineTransform(pts1, pts2) 

start_points = np.array([[0., 0.], [cols, 0.], [cols, rows], [0., rows]])
ones = np.ones(shape=(len(start_points), 1))
points_ones = np.hstack([start_points, ones])

transformed_points = M1.dot(points_ones.T).T

height = max(abs(transformed_points[0][1]-transformed_points[2][1]), abs(transformed_points[1][1]-transformed_points[3][1]))
ymin = min(transformed_points[0][1], transformed_points[2][1], transformed_points[1][1], transformed_points[3][1])
width = max(abs(transformed_points[0][0]-transformed_points[2][0]), abs(transformed_points[1][0]-transformed_points[3][0]))
xmin = min(transformed_points[0][0], transformed_points[2][0], transformed_points[1][0], transformed_points[3][0])

scale = min(rows/height, cols/width)

M1[0][2]-=xmin
M1[1][2]-=ymin

ax1.imshow(image.copy())
ax1.grid()
ax1.set_title('Начальное изображение', fontsize=15)

dst2 = cv2.warpAffine(image.copy(), M1, (int(width)+1, int(height)+1))
img = cv2.resize(dst2, (cols, rows))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax2.imshow(img)
ax2.grid()
#ax2.set_title(f'Повернутое изображение на {angle} градусов относительно {xc}, {yc}', fontsize=8)

plt.show()