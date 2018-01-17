# -*- coding:utf-8 -*-  
__author__ = 'Microcosm'  
  
import cv2  
import numpy as np  
  
img = cv2.imread("lp2.jpg")

#转换hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

print (hsv[100,100])
#获取mask
lower_blue=np.array([0,0,0])
upper_blue=np.array([200,255,60])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask = 255-mask
cv2.imshow('Mask', mask)

cv2.waitKey(0)