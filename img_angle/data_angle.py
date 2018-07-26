# -*- coding: utf-8 -*-
"""
    Created on 2017 10.17
    @author: liupeng
    """

import sys
import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import exposure
from data_load import load_image
import threading
# from data_aug import DataAugmenters

def gen_rotation(img, angle):
    w,h = img.shape[1], img.shape[0]
    rotate_matrix = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle=angle, scale=0.7)
    img = cv2.warpAffine(img, rotate_matrix, (w,h), borderMode=cv2.BORDER_REPLICATE)
    return img

# 图片在二级目录
file_path = 'shoes'
img_path, _, _, _, _, _, _ = load_image(file_path, 1.0).gen_train_valid()
print (img_path)

def add_img_padding(img):
    h, w, _ = img.shape
    width = np.max([h, w])
    # 按照长宽中大的初始化一个正方形
    img_padding = np.zeros([width, width,3])
    # 找出padding的中间位置
    h1 = int(width/2-h/2)
    h2 = int(width/2+h/2)
    w1 = int(width/2-w/2)
    w2 = int(width/2+w/2)
    # 进行padding， img为什么也要进行扣取？ 原因在于除以2取整会造成尺寸不一致的情况。
    img_padding[h1:h2, w1:w2, :] = img[0:(h2-h1),0:(w2-w1),:]
    return img_padding

def save_img(img, angle, path):
    img_padding = gen_rotation(img, angle)
    # 从现有的图片路径创建新的图片路径
    img_dir = 'train'
    img_sub_dir = path.split('/')
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(img_dir+'/'+img_sub_dir[-2]+str(angle)):
        os.makedirs(img_dir+'/'+img_sub_dir[-2]+str(angle))
    # 新的图片路径
    img_new_file = img_dir+'/'+img_sub_dir[-2] +str(angle)+ '/' + img_sub_dir[-1]
    cv2.imwrite(img_new_file, img_padding)

# 遍历所有的图片
for path in img_path:
    print (path)
    # 对于每一张图片
    img = cv2.imread(path)
    if img is None:
        continue
    angles = [15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]

    threads = [threading.Thread(target=save_img, args=(img, angle, path, )) for angle in angles]

    for t in threads:
        t.start()  #启动一个线程
    for t in threads:
        t.join()  #等待每个线程执行结束
