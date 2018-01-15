
# coding=utf-8
import cv2  
import numpy as np  
img=cv2.imread('chess_board.png')  
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
gray=np.float32(gray)  
  
dst=cv2.cornerHarris(gray,2,3,0.04)  
dst=cv2.dilate(dst,None)  

tmp = dst>0.01*dst.max()
n = 0
for y in range (tmp.shape[0]):
    for x in range (tmp.shape[1]):
        if tmp[y][x] == True:
            print (x, y)
            n = n+1
print (dst.shape)
print (n)
  
img[dst>0.01*dst.max()]=[0,0,255]  

###--------------------------------------------------------------------------------###
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

#找到重心
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

#定义迭代次数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
#返回角点
#绘制
print (len(centroids))
print (len(corners))
for i in range(len(corners)):
    print (i, "===>", corners[i][0], corners[i][1])

res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imwrite('subpixel5.png',img)
###--------------------------------------------------------------------------------###

cv2.imshow('dst',img)  
if cv2.waitKey(0) & 0xff==27:  
    cv2.destroyAllWindows()  