
# coding=utf-8

import cv2
import numpy as np

############################################################################  
# 函数：im_read  
# 描述：读取图像  
#  
# 输入：图像路径  
# 返回：图像矩阵  
############################################################################ 
def im_read(img_path):
	img=cv2.imread(img_path)
	return img

############################################################################  
# 函数：to_gray  
# 描述：BGR转灰度  
#  
# 输入：BGR图像  
# 返回：灰度图像  
############################################################################
def to_gray(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	return gray

# ############################################################################  
# 函数：to_corner  
# 描述：Harris角点检测  
#  
# 输入：灰度图像  
# 返回：图像角点信息  
############################################################################
def to_corner(img):
	dst=cv2.cornerHarris(img,2,3,0.04) 
	return dst

# ############################################################################  
# 函数：to_dilate  
# 描述：膨胀  
#  
# 输入：图像  
# 返回：膨胀后的图像  
############################################################################
def to_dilate(img):
	dst=cv2.dilate(img,None)
	return dst

# ############################################################################  
# 函数：to_dilate  
# 描述：图像二值化，大于阈值的为白色，小于阈值的为黑色。  
#  
# 输入：图像,灰度图
# 返回：二值化后的图像 
############################################################################
def to_binarization(img):
	ret, dst=cv2.threshold(img,0.01*img.max(),255,0)
	return ret, dst

# ############################################################################  
# 函数：to_centroids  
# 描述：找到重心  
#  
# 输入：图像，二维矩阵
# 返回：二值化后的图像 
############################################################################
def to_centroids(img_, gray):
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img_)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	return centroids, corners

# ############################################################################  
# 函数：to_coordinate
# 描述：获取角点坐标
#  
# 输入：输入为corners
# 返回：所有角点的坐标。 
############################################################################
def to_coordinate(corners):
	# print (len(corners))
	tmp = []
	for i in range(len(corners)):
		if i !=0:
			#print (i, "===>", corners[i][0], corners[i][1])
			tmp.append([corners[i][0], corners[i][1]])
	return tmp

# ############################################################################  
# 函数：to_showimg
# 描述：显示图像
#  
# 输入：名字和图像
# 返回：无 
############################################################################
def to_showimg(name, img):
	cv2.imshow(name,img)

# ############################################################################  
# 函数：to_imwrite
# 描述：将图像写入文件
#  
# 输入：名字和图像
# 返回：无
############################################################################
def to_imwrite(name, img):
	cv2.imwrite(name, img)

# ############################################################################  
# 函数：to_stop()
############################################################################
def to_stop():
	if cv2.waitKey(0) & 0xff==27:  
		cv2.destroyAllWindows()

# ############################################################################  
# 函数：to_sort
# 描述：每一行的9个进行排序
#  
# 输入：tmp
# 返回：tmp
############################################################################
def to_sort(tmp):
    liu0 = [];liu1 = [];liu2 = [];liu3 = [];liu4 = [];liu5 = [];liu6 = [];liu7 = [];liu8 = []
    for i in range(len(tmp)):
        if i<9:
            # print (i, "=====>", tmp[i])
            liu0.append([tmp[i][0], tmp[i][1]])
        elif i<18:
            # print (i, "=====>", tmp[i])
            liu1.append([tmp[i][0], tmp[i][1]])
        elif i<27:
            # print (i, "=====>", tmp[i])
            liu2.append([tmp[i][0], tmp[i][1]])
        elif i<36:
            # print (i, "=====>", tmp[i])
            liu3.append([tmp[i][0], tmp[i][1]])
        elif i<45:
            # print (i, "=====>", tmp[i])
            liu4.append([tmp[i][0], tmp[i][1]])
        elif i<54:
            # print (i, "=====>", tmp[i])
            liu5.append([tmp[i][0], tmp[i][1]])
        elif i<63:
            # print (i, "=====>", tmp[i])
            liu6.append([tmp[i][0], tmp[i][1]])
        elif i<72:
            # print (i, "=====>", tmp[i])
            liu7.append([tmp[i][0], tmp[i][1]])
        elif i<81:
            # print (i, "=====>", tmp[i])
            liu8.append([tmp[i][0], tmp[i][1]])
    liu0.sort(key=lambda x:x[0])
    # print ('(liu0)======>>>>>', liu0)
    liu1.sort(key=lambda x:x[0])
    # print ('(liu1)======>>>>>', liu1)
    liu2.sort(key=lambda x:x[0])
    # print ('(liu2)======>>>>>', liu2)
    liu3.sort(key=lambda x:x[0])
    # print ('(liu3)======>>>>>', liu3)
    liu4.sort(key=lambda x:x[0])
    # print ('(liu4)======>>>>>', liu4)
    liu5.sort(key=lambda x:x[0])
    # print ('(liu5)======>>>>>', liu5)
    liu6.sort(key=lambda x:x[0])
    # print ('(liu6)======>>>>>', liu6)
    liu7.sort(key=lambda x:x[0])
    # print ('(liu7)======>>>>>', liu7)
    liu8.sort(key=lambda x:x[0])
    # print ('(liu8)======>>>>>', liu8)
    for i in range(len(tmp)):
        if i<9:
            tmp[i] = liu0[i]
        elif i<18:
            tmp[i] = liu1[i-9]
        elif i<27:
            tmp[i] = liu2[i-18]
        elif i<36:
            tmp[i] = liu3[i-27]
        elif i<45:
            tmp[i] = liu4[i-36]
        elif i<54:
            tmp[i] = liu5[i-45]
        elif i<63:
            tmp[i] = liu6[i-54]
        elif i<72:
            tmp[i] = liu7[i-63]
        elif i<81:
            tmp[i] = liu8[i-72]
    return tmp


# ############################################################################  
# 函数：to_chess_coord
# 描述：计算格子的中心坐标
#  
# 输入：tmp
# 返回：peng (x坐标，y坐标)
############################################################################
def to_chess_coord(tmp):
    liu0 = [];liu1 = [];liu2 = [];liu3 = [];liu4 = [];liu5 = [];liu6 = [];liu7 = []
    peng = []
    for i in range(len(tmp)-9):
        if i<8:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu0.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 8<i<17:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu1.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 17<i<26:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu2.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 26<i<35:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu3.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 35<i<44:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu4.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 44<i<53:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu5.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 53<i<62:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu6.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        elif 62<i<71:
            # print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
            liu7.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
            peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
        #elif i<80:
        #    print (i, "=====>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
        #    liu1.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
    return peng

