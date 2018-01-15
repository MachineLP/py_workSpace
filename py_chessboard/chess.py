
# coding=utf-8
from src.core import im_read
from src.core import to_gray
from src.core import to_corner
from src.core import to_dilate
from src.core import to_binarization
from src.core import to_centroids
from src.core import to_coordinate
from src.core import to_showimg
from src.core import to_imwrite
from src.core import to_stop
from src.core import to_sort
from src.core import to_chess_coord
import numpy as np

filename = 'chess_board.png'
img=im_read(filename)
gray=to_gray(img)
gray=np.float32(gray)
dst=to_corner(gray)
dst=to_dilate(dst)
#img[dst>0.01*dst.max()]=[0,0,255]

###--------------------------------------------------------------------------------###
ret, dst = to_binarization(dst)
print ('==============>',ret)
dst = np.uint8(dst)

centroids, corners = to_centroids(dst, gray)
# 获取得到棋盘的拐点
tmp = to_coordinate(corners)
for i in range(len(tmp)):
    print (i, "===>", tmp[i][0], tmp[i][1])
print ('tmp=======>', len(tmp))

tmp1 = to_sort(tmp)
print ('tmp1=======>', len(tmp1))

'''
peng = []
for i in range(len(tmp)-9):
    if (i!=8 and i!=17 and i!=26 and i!=35 and i!=44 and i!=53 and i!=62 and i!=71 and i!=80):
        print (i, "===>", (tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2)
        peng.append([(tmp[i][0]+tmp[i+1][0])/2, (tmp[i][1]+tmp[i+9][1])/2])
print (len(peng))'''

tmp2 = np.int0(to_chess_coord(tmp1))
print ('tmp2=======>', len(tmp2))
print (tmp2[63])
# 画出棋盘各格子的中心。
for i in range(len(tmp2)):
    print("第{}个方格: x={}, y={}".format(i+1,tmp2[i][0],tmp2[i][1]))
    img[tmp2[i][1],tmp2[i][0]]=[0,0,255]

# 下面是画的拐点。
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

to_imwrite('subpixel5.png',img)
###--------------------------------------------------------------------------------###
to_showimg('dst',img)
to_stop()
