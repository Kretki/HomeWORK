import numpy as np
import matplotlib.pyplot as plt
import cv2

for i in range(0, 17):
    image = cv2.imread(f'./ДЗ3/images/test_image_{i}.jpg')
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    #0 - blue, 1 - green, 2 - red
    result1 = cv2.calcHist([image], [0, 1, 2], None, [180, 256, 256], [25, 65, 120, 255, 100, 255])
    result2 = cv2.calcHist([image], [0, 1], None, [180, 256], [80, 140, 50, 255])
    if(np.sum(result1)<np.sum(result2)):
        print(f'{i} - лес')
    else:
        print(f'{i} - пустыня')