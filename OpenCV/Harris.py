# coding=utf-8


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
# from scipy.ndimage import *
def Harris(img,op='H',w=4,k=0.1):
    # below is wrong for trying to create a
    # ndarray with demision number greater than 32
    # img = np.ndarray(img, dtype=np.float32)
    # if use np.mat, be ware of the difference in matrix multiply
    img = np.array(img,dtype=np.float32)
    # row,col = img.shape
    # out = np.zeros(img.shape)
    difx,dify=np.gradient(img)

    # detect edge with cv2.Sobel
    # EdgeX = cv2.Sobel(img, cv2.CV_32FC1, 1, 0)
    # EdgeY = cv2.Sobel(img, cv2.CV_32FC1, 0, 1)
    # Mag = np.sqrt(EdgeX ** 2 + EdgeY ** 2)

    # or use canny edge detector from skimage.feature
    Mag = canny(img, 1, 0.4, 0.2)

    # compute autocorralation
    difx2=difx**2
    dify2=dify**2
    difxy=difx*dify

    # or use cv2.multiply
    # difx2 = cv2.multiply(difx,difx)
    # dify2 = cv2.multiply(dify,dify)
    # difxy = cv2.multiply(difx, dify)

    # mean filter in scipy.ndimage
    # A = uniform_filter(difx2,size=w)
    # B = uniform_filter(dify2,size=w)
    # C = uniform_filter(difxy,size=w)

    # or use mean filter in cv2
    A = cv2.blur(difx2,(w,w))
    B = cv2.blur(dify2,(w,w))
    C = cv2.blur(difxy,(w,w))

    if op =='H':
        out = A*B - C**2 -k*(A+B)**2
        out[Mag == 0] = 0
    else:
        out = (A*dify2-2*C*difxy+B*difx2)/(difx2+dify2+1)
        out[difx2+dify2==0]=0

    # next section for debug
    # plt.subplot(221);plt.imshow(img)
    # plt.subplot(222);plt.imshow(difx)
    # plt.subplot(223);plt.imshow(difx2);plt.show()
    # # plt.subplot(224);plt.imshow(C);plt.show()
    return out

if __name__ == '__main__':
    image = cv2.imread('lp.jpg', flags=0)
    image = cv2.resize(image, (100, 100))
    corimg = Harris(image, op='TI')
    plt.subplot(121);plt.imshow(image);plt.title('Original')
    plt.subplot(122);plt.imshow(corimg);plt.title('Harris Corner')
    plt.show()