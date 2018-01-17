# coding=utf-8

from threading import Lock
import os
import cv2
import numpy as np
import time



img=cv2.imread('lp.jpg')  
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
gray=np.float32(gray)  
  
dst=cv2.cornerHarris(gray,2,3,0.04)  
dst=cv2.dilate(dst,None)  
  
img[dst>0.01*dst.max()]=[0,0,255]  
cv2.imshow('dst',img)  
if cv2.waitKey(0) & 0xff==27:  
    cv2.destroyAllWindows()  


print('begin...')
filename = 'train_crop/11-06 NF (discolor)/2017-11-06 15-41-36-604.jpg'

# Read image
img = cv2.imread(filename)

w, h, c = img.shape
t1 = time.time()
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t1 = time.time()
im = cv2.equalizeHist(im)
t2 = time.time()
print('cost time is: ' + str((t2-t1)*1000) + ' ms')

####################################################
#configuration
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 50
# Filter by Area.
params.filterByArea = True
# params.minArea = 1500
params.minArea = 30
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
####################################################

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (105, 0, 205) , cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(len(keypoints), keypoints)

# Show keypoints
cv2.namedWindow("Keypoints", 2)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

cv2.destroyAllWindows()
#cv2.imwrite('result.jpg', im_with_keypoints)
print('done!')