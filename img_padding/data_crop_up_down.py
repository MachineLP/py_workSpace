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

# 图片在二级目录
file_path = '/Volumes/machinelp/database/img_body_pose_train_test/img_body_pose_train'
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

def crop_img_up(img):
    h, w, _ = img.shape
    h1 = int(h / 2.0)
    img_crop_up = np.zeros([h1, w, 3])
    img_crop_up = img[0:h1, 0:w, :]
    return img_crop_up

def crop_img_down(img):
    h, w, _ = img.shape
    h1 = int(h / 2.0)
    img_crop_down = np.zeros([h1, w, 3])
    img_crop_down = img[h1:h, 0:w, :]
    return img_crop_down


# 遍历所有的图片
for path in img_path:
    print (path)
    # 对于每一张图片
    img = cv2.imread(path)
    if img is None:
        continue
    img_padding = crop_img_down(img)
    # 从现有的图片路径创建新的图片路径
    img_sub_dir = path.split('/')
    _sub = '_crop_down'
    if not os.path.isdir(img_sub_dir[-3]+_sub):
        os.makedirs(img_sub_dir[-3]+_sub)
    if not os.path.isdir(img_sub_dir[-3]+_sub+'/'+img_sub_dir[-2]):
        os.makedirs(img_sub_dir[-3]+_sub+'/'+img_sub_dir[-2])
    # 新的图片路径
    img_new_file = img_sub_dir[-3]+_sub+'/'+img_sub_dir[-2] + '/' + img_sub_dir[-1]
    cv2.imwrite(img_new_file, img_padding)



