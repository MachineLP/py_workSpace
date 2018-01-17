
import cv2  
import numpy as np  
from matplotlib import pyplot as plt  


# 直方图均衡化  
file = "lp.jpg"
img = cv2.imread(file, 0)  
equ = cv2.equalizeHist(img)  
res = np.hstack((img, equ))   # 将图像拼在一起  
res = cv2.resize(res,(800,200))
cv2.namedWindow("equ")  
cv2.imshow("equ", res)  
cv2.waitKey(0)  
cv2.destroyAllWindows()  