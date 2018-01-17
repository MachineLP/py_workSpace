import cv2
import numpy as np


# 图像的平移。
img = cv2.imread('lp.jpg')
print (img[20][20])
# 平移参数。
H = np.float32([[1,0,500],[0,1,50]])
# RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=0, scale=0.5)
# h,w = img.shape[:2]
w,h = 1920, 1080
RotImg = cv2.warpAffine(img, H, (w,h), borderValue=(129,137,130)) #需要图像、变换矩阵、变换后的大小
cv2.imwrite("out.jpg", RotImg)

# w,h = 1920, 1080
w,h = 1920, 1080
w,h = img.shape[1], img.shape[0]
# 旋转参数。 angle图像的旋转角度，scale图像的缩放比例。 scale = 0.5
RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=45, scale=0.7)
# RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0]*2, img.shape[1]*2))
RotImg = cv2.warpAffine(img, RotateMatrix, (w,h), borderValue=(129,137,130))
cv2.imwrite("out_rot.jpg", RotImg)