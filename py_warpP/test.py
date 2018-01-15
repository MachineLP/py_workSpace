
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def gen_point(event,x,y,flags,param):
    global i, p1, p2, p3, p4
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print ('i:', i)
        print ([x,y])
        if i % 4 == 0:
            p1 = [x, y]
        if i % 4 == 1:
            p2 = [x, y]
        if i % 4 == 2:
            p3 = [x, y]
        if i % 4 == 3:
            p4 = [x, y]
        i = i + 1
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        pts1 = np.float32([p1,p2,p3,p4])  
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])  
        M = cv2.getPerspectiveTransform(pts1,pts2)  
        dst = cv2.warpPerspective(img,M,(300,300)) 

        cv2.imwrite('tmp.jpg', dst)
        i = 0
        print ("success!")

# img = np.zeros((512,512,3),np.uint8)
img = cv2.imread("lp.jpg")
cv2.namedWindow('image', 2)
cv2.setMouseCallback('image',gen_point)

i = 0
while(1):
    cv2.imshow('image',img)

    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

