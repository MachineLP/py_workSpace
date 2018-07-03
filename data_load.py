# -*- coding: utf-8 -*-
"""
Created on 2017 10.17
@author: liupeng
"""

import sys
import numpy as np
import os
import cv2
#from skimage import exposure
#from lib.utils.utils import shuffle_train_data

class load_image(object):

    def __init__(self, img_dir, train_rate):
        self.img_dir = img_dir
        self.train_rate = train_rate
        self.train_imgs = []
        self.train_labels = []
        self.note_label = []
    
    def _load_img_path(self, img_sub_dir, img_label):
        data = []
        label = []
        # 遍历该文件下的所有图片。
        for root, dirs, files in os.walk( os.path.join(self.img_dir, img_sub_dir) ):
            for sub_file in files:
                if sub_file.endswith('jpg') or sub_file.endswith('jpeg') or sub_file.endswith('gif') or sub_file.endswith('png') or sub_file.endswith('bmp'):
                    img_path = os.path.join(root, sub_file)
                    if cv2.imread(img_path) is not None:
                        data.append(img_path)
                        label.append(int(img_label))
        return data, label

    def _load_database_path(self):
        file_path = os.listdir(self.img_dir)
        for i, path in enumerate(file_path):
            if os.path.isfile(os.path.join(self.img_dir, path)):
                continue
            data, label = self._load_img_path(path, i)
            self.train_imgs.extend(data)
            self.train_labels.extend(label)
            self.note_label.append([path, i])
            # print (path, i)
        #self.train_imgs, self.train_labels = shuffle_train_data(self.train_imgs, self.train_labels)
    
    def gen_train_valid(self):
        self._load_database_path()
        image_n = len(self.train_imgs)
        train_n = int(image_n*self.train_rate)
        valid_n = int(image_n*(1-self.train_rate))
        train_data, train_label = self.train_imgs[0:train_n], self.train_labels[0:train_n]
        valid_data, valid_label = self.train_imgs[train_n:image_n], self.train_labels[train_n:image_n]
        return train_data, train_label, valid_data, valid_label, train_n, valid_n, self.note_label

if __name__ == '__main__':
    imgs_data,_,_,_,_,_,_ = load_image('gender', 0.9).gen_train_valid()
    print (imgs_data[0])

